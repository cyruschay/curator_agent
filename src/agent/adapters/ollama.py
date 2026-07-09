from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

_NS_PER_S = 1_000_000_000.0  # nanoseconds -> seconds conversion

# Config-only meta-keys that control project logic but are NOT valid ChatOllama
# constructor parameters.  Stripped before the kwargs dict is passed to ChatOllama.
_META_KEYS: frozenset[str] = frozenset({"structured_output"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmptyOllamaResponseError(ValueError):
    """Raised when a chat invocation returns no usable response content."""


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OllamaThinking:
    """Reasoning payload from a thinking-capable Ollama model."""

    reasoning_content: str | None
    raw_additional_kwargs: dict[str, Any]


@dataclass(slots=True)
class OllamaInvocationMetadata:
    """Normalized invocation metadata from a ChatOllama response.

    Duration breakdown (all in seconds):
      total = load + prompt_eval + thinking + eval  (roughly)
      thinking_duration = total - (load + prompt_eval + eval)
        -> the phase where reasoning tokens are generated; not exposed by
          Ollama as a named field, so computed as the residual.

    Token breakdown:
      Ollama's ``eval_count`` (output_tokens) covers only the final visible
      response, not the reasoning tokens generated during thinking.
      thinking_tokens is read from ``output_token_details.reasoning`` when
      the LangChain/Ollama layer populates it; otherwise it is estimated
      from the length of reasoning_content (~4 chars per token) and
      thinking_tokens_estimated is set to True.
    """

    model: str | None
    created_at: str | None
    done_reason: str | None
    done: bool | None
    # Durations
    total_duration_seconds: float | None
    load_duration_seconds: float | None
    prompt_eval_duration_seconds: float | None
    thinking_duration_seconds: float | None   # residual: total - (load + prefill + gen)
    eval_duration_seconds: float | None
    # Tokens
    input_tokens: int | None
    thinking_tokens: int | None               # reasoning tokens (exact or estimated)
    thinking_tokens_estimated: bool           # True when derived from reasoning_content length
    output_tokens: int | None
    total_tokens: int | None
    raw_response_metadata: dict[str, Any]
    raw_usage_metadata: dict[str, Any]


@dataclass(slots=True)
class OllamaJSONResponse:
    """Parsed JSON response with optional reasoning and invocation metadata."""

    payload: dict[str, Any]
    thinking: OllamaThinking | None
    invocation_metadata: OllamaInvocationMetadata
    text: str
    raw_message: Any


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


_STRIP_FORMAT_LOGGED: set[str] = set()  # one-time log guard per (model_key, model)


def _build_chat_kwargs(
    config_data: dict[str, Any],
    model_key: str,
    overrides: Mapping[str, Any] | None,
    format: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge config layers into a flat dict of ChatOllama constructor kwargs.

    Reads from curator agent's config.yaml structure:
      ollama_models:
        <model_key>: "<model_name>"
      ollama_params:
        <model_key>:
          temperature: ...
          num_predict: ...
          ...
    """
    kwargs: dict[str, Any] = {}

    model_name = config_data.get("ollama_models", {}).get(model_key)
    if not model_name:
        raise ValueError(f"No Ollama model configured for '{model_key}' in ollama_models.")
    kwargs["model"] = model_name

    params = (config_data.get("ollama_params", {}) or {}).get(model_key, {}) or {}
    kwargs.update(params)

    if format is not None:
        kwargs["format"] = format
    if overrides:
        kwargs.update(overrides)

    # MLX policy: strip format entirely when the model opts out of structured output.
    supports_format = kwargs.get("structured_output", True)
    if not supports_format and "format" in kwargs:
        log_key = f"{model_key}:{model_name}"
        if log_key not in _STRIP_FORMAT_LOGGED:
            _STRIP_FORMAT_LOGGED.add(log_key)
            print(
                f"[ollama] structured_output=false for '{model_key}' ({model_name}); "
                f"stripping format={kwargs['format']!r} -- MLX runner does not support "
                f"the format parameter. Relying on prompt instructions + _extract_json."
            )
        kwargs.pop("format", None)

    for _mk in _META_KEYS:
        kwargs.pop(_mk, None)

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs


def build_chat_ollama(
    *,
    config_data: dict[str, Any],
    model_key: str,
    overrides: Mapping[str, Any] | None = None,
    format: str | dict[str, Any] | None = "json",
) -> Any:
    """Build a ChatOllama instance from project config.

    For tool calling or structured output, pass ``format=None`` and call
    ``.bind_tools()`` or ``.with_structured_output()`` on the returned model.
    """
    try:
        from langchain_ollama import ChatOllama
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'langchain-ollama' package is required. Install it in the project env."
        ) from exc

    return ChatOllama(**_build_chat_kwargs(config_data, model_key, overrides, format))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise EmptyOllamaResponseError("ChatOllama returned blank content.")

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.DOTALL).strip()
    if not payload:
        raise EmptyOllamaResponseError("ChatOllama returned blank content after stripping code fence.")

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))

    if not isinstance(obj, dict):
        raise ValueError("Expected a top-level JSON object from ChatOllama.")
    return obj


def _parse_response(message: Any) -> OllamaJSONResponse:
    """Parse a raw ChatOllama message into an OllamaJSONResponse."""
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise EmptyOllamaResponseError("ChatOllama returned an empty message content.")
    text = content.strip()

    # Thinking / reasoning
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    reasoning_content = additional_kwargs.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        reasoning_content = str(reasoning_content)
    else:
        reasoning_content = "<empty>"
    thinking = (
        OllamaThinking(
            reasoning_content=reasoning_content,
            raw_additional_kwargs=additional_kwargs,
        )
        if (reasoning_content is not None or additional_kwargs)
        else None
    )

    # Invocation metadata
    response_metadata = dict(getattr(message, "response_metadata", {}) or {})
    usage_metadata = dict(getattr(message, "usage_metadata", {}) or {})

    def _ns_to_s(value: Any) -> float | None:
        return float(value) / _NS_PER_S if isinstance(value, (int, float)) else None

    # Durations
    total_s = _ns_to_s(response_metadata.get("total_duration"))
    load_s = _ns_to_s(response_metadata.get("load_duration"))
    prompt_eval_s = _ns_to_s(response_metadata.get("prompt_eval_duration"))
    eval_s = _ns_to_s(response_metadata.get("eval_duration"))

    thinking_duration_s: float | None = None
    if all(v is not None for v in (total_s, load_s, prompt_eval_s, eval_s)):
        thinking_duration_s = max(0.0, total_s - load_s - prompt_eval_s - eval_s)  # type: ignore[operator]

    # Tokens
    input_tokens = usage_metadata.get("input_tokens")
    output_tokens = usage_metadata.get("output_tokens")
    total_tokens = usage_metadata.get("total_tokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens

    thinking_tokens: int | None = None
    thinking_tokens_estimated = False
    output_token_details = usage_metadata.get("output_token_details") or {}
    exact_thinking = output_token_details.get("reasoning")
    if isinstance(exact_thinking, int):
        thinking_tokens = exact_thinking
    elif isinstance(reasoning_content, str) and reasoning_content:
        thinking_tokens = max(1, len(reasoning_content) // 4)
        thinking_tokens_estimated = True

    invocation_metadata = OllamaInvocationMetadata(
        model=response_metadata.get("model"),
        created_at=response_metadata.get("created_at"),
        done_reason=response_metadata.get("done_reason"),
        done=response_metadata.get("done"),
        total_duration_seconds=total_s,
        load_duration_seconds=load_s,
        prompt_eval_duration_seconds=prompt_eval_s,
        thinking_duration_seconds=thinking_duration_s,
        eval_duration_seconds=eval_s,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        thinking_tokens=thinking_tokens,
        thinking_tokens_estimated=thinking_tokens_estimated,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        raw_response_metadata=response_metadata,
        raw_usage_metadata=usage_metadata,
    )

    return OllamaJSONResponse(
        payload=_extract_json(text),
        thinking=thinking,
        invocation_metadata=invocation_metadata,
        text=text,
        raw_message=message,
    )


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def invoke_json(
    *,
    chat_model: Any,
    system_prompt: str | None,
    user_prompt: str,
) -> OllamaJSONResponse:
    """Invoke a pre-built ChatOllama model and return a parsed JSON response."""
    messages: list[tuple[str, str]] = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("human", user_prompt))
    return _parse_response(chat_model.invoke(messages))


def chat_json(
    *,
    config_data: dict[str, Any],
    model_key: str,
    system_prompt: str | None,
    user_prompt: str,
    overrides: Mapping[str, Any] | None = None,
) -> OllamaJSONResponse:
    """Build a ChatOllama model from config, invoke it, and return a parsed JSON response."""
    chat_model = build_chat_ollama(config_data=config_data, model_key=model_key, overrides=overrides, format="json")
    return invoke_json(chat_model=chat_model, system_prompt=system_prompt, user_prompt=user_prompt)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_invocation_metadata(metadata: OllamaInvocationMetadata) -> None:
    """Print a three-line Ollama invocation summary.

    Line 1 -- status:   model, done flag, done reason, timestamp
    Line 2 -- duration: total | think | load / prefill / gen (all in seconds)
    Line 3 -- tokens:   input | think (~estimated) | output | total
    """
    def _fmt_s(v: float | None, width: int = 1) -> str:
        return f"{v:{width}.1f}s" if v is not None else f"{'?':>{width+1}}"

    def _fmt_n(v: int | None, prefix: str = "", estimated: bool = False) -> str:
        if v is None:
            return f"{prefix}?"
        suffix = "~" if estimated else ""
        return f"{prefix}{v}{suffix}"

    level = "[WARN]" if metadata.done_reason == "length" else "[INFO]"
    timestamp = datetime.now().strftime("%H:%M:%S")
    done_str = str(metadata.done) if metadata.done is not None else "?"
    done_reason = metadata.done_reason or "?"

    # Duration line
    total   = _fmt_s(metadata.total_duration_seconds)
    think_d = _fmt_s(metadata.thinking_duration_seconds)
    load    = _fmt_s(metadata.load_duration_seconds)
    prefill = _fmt_s(metadata.prompt_eval_duration_seconds)
    gen     = _fmt_s(metadata.eval_duration_seconds)

    # Token line
    tok_in    = _fmt_n(metadata.input_tokens,    prefix="in=")
    tok_think = _fmt_n(metadata.thinking_tokens, prefix="think=",
                       estimated=metadata.thinking_tokens_estimated)
    tok_out   = _fmt_n(metadata.output_tokens,   prefix="out=")
    tok_total = _fmt_n(metadata.total_tokens,    prefix="total=")

    print(
        f"{level} {timestamp} | {metadata.model or '?'}  done={done_str} ({done_reason})\n"
        f"                |- duration  total={total} (load={load} fill={prefill} think={think_d} gen={gen})\n"
        f"                +- tokens    {tok_total} ({tok_in} {tok_think} {tok_out})"
    )
