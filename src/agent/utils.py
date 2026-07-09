from __future__ import annotations
import json

from Bio import Entrez
from lxml import etree
import os
import re
import hashlib
import jsonschema

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage

from states import (  # type: ignore
    AgentState,
)

# ======================================================================================
# Preprocessing
# ======================================================================================

def line_hash_check(line: str, listf: Path | str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Register the current article in the curated list and decide whether to process.

    Parameters
    ----------
    line : str
        The raw input line (e.g., from a JSONL queue).
    listf : Path | str
        Path to the curated list JSONL file.

    Returns
    -------
    Dict[str, Any]
        Metadata describing the action to take, including:
        - action: "skip" or "process"
        - record_index: zero-based index within curated_list.jsonl
        - ingest_hash: SHA256 hash of the input line
        - record: the curated list entry (latest version)
    """

    path = Path(listf)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()

    # hash the line to see if already done
    line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
    if inputs is None:
        inputs = json.loads(line)
    processing_id = inputs["processing_id"]
    pmid = inputs.get("pmid", "[ERROR_NO_PMID]")

    # load curated list records
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    match_index: Optional[int] = None
    for idx, rec in enumerate(records):
        if (
            rec.get("processing_id") == processing_id
            and rec.get("pmid") == pmid
        ):
            match_index = idx
            break

    now = datetime.utcnow().isoformat()

    if match_index is not None:
        record = records[match_index]
        if record.get("ingest_hash") == line_hash and record.get("status") in ["completed", "error"]:
            record["article_index"] = match_index
            return {
                "action": "skip",
                "record_index": match_index,
                "ingest_hash": line_hash,
                "record": record,
            }

        # Resume or refresh existing record
        record.setdefault("started_at", record.get("updated_at", now))
        record["status"] = "processing"
        record["ingest_hash"] = line_hash
        record["updated_at"] = now
        records[match_index] = record
    else:
        record = {
            "processing_id": processing_id,
            "pmid": pmid,
            "status": "processing",
            "ingest_hash": line_hash,
            "started_at": now,
            "updated_at": now,
        }
        records.append(record)
        match_index = len(records) - 1

    # refresh article_index for all records to maintain positional ordering
    for idx, rec in enumerate(records):
        rec["article_index"] = idx

    with open(path, "w", encoding="utf-8") as outfile:
        for rec in records:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    record = records[match_index]
    return {
        "action": "process",
        "record_index": match_index,
        "ingest_hash": line_hash,
        "record": record,
    }


def update_curated_status(processing_id: str, pmid: Optional[str],
    listf: Path | str, status: str, ingest_hash: Optional[str] = None,) -> bool:
    """Update the curated list entry for a given article."""

    path = Path(listf)
    if not path.exists():
        return False

    updated = False
    records: List[Dict[str, Any]] = []
    now = datetime.utcnow().isoformat()

    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if (
                rec.get("processing_id") == processing_id
                and (pmid is None or rec.get("pmid") == pmid)
            ):
                rec["status"] = status
                if ingest_hash is not None:
                    rec["ingest_hash"] = ingest_hash
                if status == "completed":
                    rec["completed_at"] = now
                rec["updated_at"] = now
                updated = True

            records.append(rec)

    if not updated:
        return False

    for idx, rec in enumerate(records):
        rec["article_index"] = idx

    with open(path, "w", encoding="utf-8") as outfile:
        for rec in records:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return True


def schema_validator(json_obj: dict, schema: dict) -> List[str]:
    """
    Validate a JSON-like object against a schema.

    Returns a list of error messages; empty list means valid.

    Parameters
    ----------
    json_obj : dict
    schema : dict

    Returns
    -------
    List[str]
        Validation errors (empty if valid).
    """
    # TODO: implement with jsonschema or a lightweight check.
    return []


# Provided signature by user (keep exact signature & return)
def validate_schema(json_record: dict, schema: dict) -> str: # Status: okay
    """
    Wrapper around `schema_validator` that returns 'valid' or 'invalid'.

    Parameters
    ----------
    json_record : dict
    schema : dict

    Returns
    -------
    str
        'valid' if ok, else 'invalid'
    """
    try:
        jsonschema.validate(instance=json_record, schema=schema)
        return "valid"
    except jsonschema.ValidationError as e:
        print(f"[Schema Error] {e.message}")
        return "invalid"


# ======================================================================================
# LLM + logging helpers (shared by graph nodes)
# ======================================================================================

from adapters.ollama import (
    OllamaInvocationMetadata,
    _extract_json,
    log_invocation_metadata,
)


def _print_msg(msg_type: str = "INFO", content: str = "") -> None:
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{current_time} {msg_type} - {content}")

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _relations_found(relations: Optional[Sequence[Dict[str, Any]]]) -> bool:
    return bool(relations) and len(relations) > 0

def _dedup_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate relations by key fields with evidence merging.

    Dedup key: (glycan_id, disease_id, disease_annotation, treatment_annotation, direction, specimen.mapped_id, species_id)
    - Merges evidence_locators, metrics, method_names from duplicates
    - Preserves first occurrence of other fields

    Input shape expected similar to RelationCandidate / MappedRelation dicts.
    """
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for r in relations or []:
        specimen_info = r.get("specimen") or {}
        if isinstance(specimen_info, dict):
            specimen_key = _norm(
                str(
                    specimen_info.get("mapped_id")
                    or specimen_info.get("mapped_name")
                    or specimen_info.get("original")
                    or ""
                )
            )
        else:
            specimen_key = _norm(str(specimen_info))

        key = (
            _norm(str(r.get("glycan_id") or r.get("glycan") or "")),
            _norm(str(r.get("disease_id") or r.get("disease") or "")),
            _norm(str(r.get("disease_annotation") or "")),
            _norm(str(r.get("treatment_annotation") or "")),
            _norm(str(r.get("direction") or "")),
            specimen_key,
            _norm(str(r.get("species_id") or "")),
        )
        if key not in seen:
            seen[key] = deepcopy(r)
            seen[key]["evidence_locators"] = list(r.get("evidence_locators", []))
            seen[key]["metrics"] = list(r.get("metrics", []))
            seen[key]["method_names"] = list(r.get("method_names", []))
        else:
            # Merge evidence locators
            for loc in r.get("evidence_locators", []):
                if loc not in seen[key]["evidence_locators"]:
                    seen[key]["evidence_locators"].append(loc)

            # Merge metrics (by name to avoid duplicates)
            existing_metric_names = {m.get("name") for m in seen[key].get("metrics", [])}
            for metric in r.get("metrics", []):
                if metric.get("name") not in existing_metric_names:
                    seen[key]["metrics"].append(metric)

            # Merge method names
            for method in r.get("method_names", []):
                if method not in seen[key]["method_names"]:
                    seen[key]["method_names"].append(method)

    return list(seen.values())


def _build_invocation_metadata(raw_resp: Any) -> OllamaInvocationMetadata:
    """Extract OllamaInvocationMetadata from a raw ChatOllama response."""
    _NS = 1_000_000_000.0
    response_metadata = dict(getattr(raw_resp, "response_metadata", {}) or {})
    usage_metadata = dict(getattr(raw_resp, "usage_metadata", {}) or {})

    def _ns(v: Any) -> float | None:
        return float(v) / _NS if isinstance(v, (int, float)) else None

    total_s = _ns(response_metadata.get("total_duration"))
    load_s = _ns(response_metadata.get("load_duration"))
    prompt_eval_s = _ns(response_metadata.get("prompt_eval_duration"))
    eval_s = _ns(response_metadata.get("eval_duration"))

    thinking_s: float | None = None
    if all(v is not None for v in (total_s, load_s, prompt_eval_s, eval_s)):
        thinking_s = max(0.0, total_s - load_s - prompt_eval_s - eval_s)

    input_tokens = usage_metadata.get("input_tokens")
    output_tokens = usage_metadata.get("output_tokens")
    total_tokens = usage_metadata.get("total_tokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens

    additional_kwargs = dict(getattr(raw_resp, "additional_kwargs", {}) or {})
    reasoning_content = additional_kwargs.get("reasoning_content")

    thinking_tokens: int | None = None
    thinking_tokens_estimated = False
    output_token_details = usage_metadata.get("output_token_details") or {}
    exact_thinking = output_token_details.get("reasoning")
    if isinstance(exact_thinking, int):
        thinking_tokens = exact_thinking
    elif isinstance(reasoning_content, str) and reasoning_content:
        thinking_tokens = max(1, len(reasoning_content) // 4)
        thinking_tokens_estimated = True

    return OllamaInvocationMetadata(
        model=response_metadata.get("model"),
        created_at=response_metadata.get("created_at"),
        done_reason=response_metadata.get("done_reason"),
        done=response_metadata.get("done"),
        total_duration_seconds=total_s,
        load_duration_seconds=load_s,
        prompt_eval_duration_seconds=prompt_eval_s,
        thinking_duration_seconds=thinking_s,
        eval_duration_seconds=eval_s,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        thinking_tokens=thinking_tokens,
        thinking_tokens_estimated=thinking_tokens_estimated,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        raw_response_metadata=response_metadata,
        raw_usage_metadata=usage_metadata,
    )


def _invoke_once(
    payload: List[BaseMessage],
    model: Any,
) -> Dict[str, Any]:
    """Single LLM invocation returning a normalized resp dict."""
    resp: Dict[str, Any] = {
        "reasoning": "", "output": None, "error": None,
        "tool_calls": None, "metadata": None, "truncated": False,
    }
    try:
        raw_resp = model.invoke(payload)
        try:
            resp_content = raw_resp.content
            reasoning = None
            if hasattr(raw_resp, "additional_kwargs") and isinstance(raw_resp.additional_kwargs, dict):
                reasoning = raw_resp.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                resp["reasoning"] = reasoning.strip()

            resp["output"] = resp_content.strip() if isinstance(resp_content, str) else ""

            if hasattr(raw_resp, "tool_calls") and raw_resp.tool_calls is not None:
                resp["tool_calls"] = raw_resp.tool_calls

            resp["metadata"] = raw_resp.response_metadata

            inv_meta = _build_invocation_metadata(raw_resp)
            log_invocation_metadata(inv_meta)

            if inv_meta.done_reason == "length":
                resp["truncated"] = True

        except Exception as e:
            resp["error"] = f"Response has no content. Error message: {e}"
    except Exception as e:
        resp["error"] = f"Model cannot be invoked. Error message: {e}"
    return resp


def invoke_and_parse(
    sm: Optional[BaseMessage],
    hm: Optional[BaseMessage],
    model: Any,
    *,
    messages: Optional[Sequence[BaseMessage]] = None,
    node_id: str = "?",
) -> Tuple[Dict[str, Any], List[BaseMessage], List[BaseMessage], Any, Dict[str, Any]]:
    """Invoke LLM, log metadata, parse JSON — replaces the 4-line boilerplate.

    If the response is truncated (done_reason == "length"), retries once with
    temperature=0.7 and uses whichever attempt yielded valid JSON.

    Returns
    -------
    (resp, new_messages, new_reasonings, output_json, llm_log)
    """
    if messages is not None:
        payload = list(messages)
    else:
        payload = [m for m in (sm, hm) if m is not None]

    resp = _invoke_once(payload, model)

    # Truncation retry: re-invoke with temperature=0.7
    if resp["truncated"] and not resp.get("error"):
        _print_msg("WARN", f"[{node_id}] Response truncated (hit num_predict). Retrying with temperature=0.7")
        try:
            retry_model = model.bind(temperature=0.7) if hasattr(model, "bind") else model
            retry_resp = _invoke_once(payload, retry_model)

            # Use retry if it produced more output and wasn't also truncated
            first_len = len(resp.get("output") or "")
            retry_len = len(retry_resp.get("output") or "")
            if not retry_resp.get("error") and retry_len > first_len:
                _print_msg("INFO", f"[{node_id}] Retry produced longer output ({retry_len} vs {first_len} chars). Using retry.")
                resp = retry_resp
            else:
                _print_msg("INFO", f"[{node_id}] Retry did not improve. Using original truncated response.")
        except Exception as e:
            _print_msg("WARN", f"[{node_id}] Truncation retry failed: {e}. Using original response.")

    # Parse into LangGraph messages + JSON
    new_messages: List[BaseMessage] = []
    new_reasonings: List[BaseMessage] = []
    output_json: Any = {}

    if resp.get("reasoning"):
        new_reasonings.append(AIMessage(content=f"Node_{node_id}. {resp['reasoning']}"))
    if resp.get("error"):
        new_messages.append(AIMessage(content=f"Node_{node_id}. ERROR: Model invocation. {resp['error']}"))
    if resp["truncated"]:
        new_messages.append(AIMessage(content=f"Node_{node_id}. WARN: Response may be truncated (done_reason=length)."))

    if resp.get("output"):
        raw_output = resp["output"].strip()
        try:
            output_json = _extract_json(raw_output)
        except Exception as exc:
            new_messages.append(
                AIMessage(content=f"Node_{node_id}. ERROR: LLM output is not valid JSON. {exc}")
            )
            output_json = {}
    else:
        if not resp.get("error"):
            new_messages.append(AIMessage(content=f"Node_{node_id}. ERROR: Empty LLM content output."))

    llm_log: Dict[str, Any] = {
        "node": node_id,
        "tool_calls": resp.get("tool_calls"),
        "metadata": resp.get("metadata"),
        "truncated": resp["truncated"],
    }

    return resp, new_messages, new_reasonings, output_json, llm_log


def build_index_lookup(corpus_indexed: Dict[str, Any]) -> Dict[int, Tuple[str, Dict[str, Any]]]:
    """Map global sentence index -> (locator_key, locator_dict)."""
    return {
        loc["global_index"]: (key, loc)
        for key, loc in corpus_indexed.items()
        if isinstance(loc.get("global_index"), int)
    }


def build_sections_dict(corpus_indexed: Dict[str, Any]) -> Dict[str, List[Tuple[int, str]]]:
    """Group sentences by section name, sorted by sentence number."""
    sections: Dict[str, List[Tuple[int, str]]] = {}
    for key, loc in corpus_indexed.items():
        sec = loc.get("section", "UNKNOWN")
        sent = loc.get("sentence", "")
        match = re.match(r"<S:(\d+)>", key)
        s_num = int(match.group(1)) if match else 0
        sections.setdefault(sec, []).append((s_num, sent))
    for sec in sections:
        sections[sec].sort(key=lambda x: x[0])
    return sections


def resolve_evidence_locators(
    sentence_indexes: Sequence[Any],
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    """Resolve sentence indexes to evidence locator dicts and texts."""
    evidence_locators: List[Dict[str, Any]] = []
    evidence_texts: List[str] = []
    normalized_indexes: List[int] = []

    for s_idx in sentence_indexes:
        try:
            s_idx_int = int(s_idx)
        except (TypeError, ValueError):
            continue
        lookup = index_lookup.get(s_idx_int)
        if not lookup:
            continue
        loc_key, loc = lookup
        evidence_locators.append({
            "locator_key": loc.get("locator_key", loc_key),
            "section": loc.get("section", ""),
            "global_index": s_idx_int,
        })
        evidence_texts.append(loc.get("sentence", ""))
        normalized_indexes.append(s_idx_int)

    return evidence_locators, evidence_texts, normalized_indexes


def build_agent_state_from_input_obj(obj: dict, ingest_hash: str, article_index: Optional[int] = None,) -> AgentState:
    """
    Construct initial AgentState from a minimal input object.

    Parameters
    ----------
    obj : dict
        Must contain: processing_id, pmid, pmcid, title, abstract, metadata.
    ingest_hash : str

    Returns
    -------
    AgentState
        A minimally seeded AgentState. (Delegates to state.build_agent_state_from_input)
    """
    from states import build_agent_state_from_input  # lazy import

    return build_agent_state_from_input(
        processing_id=obj.get("processing_id", ""),
        pmid=obj.get("pmid"),
        pmcid=obj.get("pmcid"),
        title=obj.get("title", ""),
        abstract=obj.get("abstract", ""),
        keywords=obj.get("keywords"),
        chemical_names=obj.get("chemical_names"),
        chemical_rn=obj.get("chemical_rn"),
        chemical_uid=obj.get("chemical_uid"),
        mesh_names=obj.get("mesh_names"),
        mesh_uid=obj.get("mesh_uid"),
        journal_abbr=obj.get("journal_abbr"),
        article_date=obj.get("article_date"),
        ingest_hash=ingest_hash,
        article_index=article_index,
    )


# ======================================================================================
# N01 — Abstract Screener
# ======================================================================================

def feature_scorer(title: str, abstract: str, metadata: dict) -> Dict[str, any]:
    
    JOURNALS_EXCL = ["MAbs", "Bioeng Transl Med", "J Biol Eng",
                           "J Biosci Bioeng", "Cell Mol Bioeng", 
                           "ACS Biomater Sci Eng", "Biotechnol Bioeng"]
    
    LEXICON_EXCL = ["MAb", "MAbs", "monoclonal antibody", "monocloncal antibodies",
                        "bioengineered", "bioengineering", "engineered",
                        "mutagenesis", "vaccine development", "workflow"]
    
    LEXICON_INCL = ["glycome", "N-glycome", "O-glycome", "mass spectrometry", "MS/MS",
                    "lectin array", "lectin arrays", "liquid chromatography",
                    "biomarker", "biomarkers"]
    
    features = {"journal_excl": [], "lexicon_excl": [], "lexicon_incl": [], "doc_len": 1, "score": 0} ###
    
    # Check journal abbreviation
    journal_abbr = metadata.get("journal_abbr", "")
    if journal_abbr in JOURNALS_EXCL:
        features["journal_excl"].append(journal_abbr)
    
    # Concat all text fields, then search for rel/irrel words
    concat_str = " ".join([title, abstract] + 
                         metadata.get("keywords", []) + 
                         metadata.get("mesh_names", []) + 
                         metadata.get("chemical_names", []) + [" "])

    for word in LEXICON_EXCL:
        # Escape special characters; create word boundary pattern
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, concat_str, re.IGNORECASE):
            features["lexicon_excl"].append(word)
    
    for word in LEXICON_INCL:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, concat_str, re.IGNORECASE):
            features["lexicon_incl"].append(word)
    
    features["doc_len"] = len(concat_str)
    
    # Calculate confidence score (0 to 1) for article relevance
    num_incl = len(features["lexicon_incl"])
    num_excl = len(features["lexicon_excl"])
    doc_len = features["doc_len"]
    
    if features["journal_excl"]:
        features["score"] = -1.0
    else:
        # Normalized log transformed score
        from math import log
        Sc = max(0, (num_incl * 0.75 - num_excl) / doc_len)
        Kn = 150000
        score = log(1 + Sc * Kn) / log(1 + Kn)
        features["score"] = score
    
    return features


# ======================================================================================
# N02 — Retrieval & Corpus
# ======================================================================================

Entrez.email = "you@example.com" ### FIX LATER -> loadenv()


if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")

def retrieve_pmc_fulltext(pmcid: str) -> dict:
    """
    Fetch and parse a PMC article in JATS XML format,
    extracting key sections into nested dictionary:
    title, abstract, body sections, license URL
    
    Tables, figures, formulas, and supplementary materials are not extracted.
    Sections and subsections are dictionaries with 'title' and 'content' keys, where 'content' is a list of strings (paragraphs) or further subsection dicts.

    Args: pmcid (str): The PMC identifier (e.g., "PMC11461603")
    """
    
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    # Fetch JATS XML from PMC
    h = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    journal_xml_bytes = h.read()
    h.close()
    
    # Utils for parsing XML
    def _norm(s: str) -> str:
        return " ".join(s.split())

    def _xp(node, path, ns=None):
        return node.xpath(path, namespaces=ns or {})

    def _text_of(node):
        return _norm("".join(node.itertext()))

    def _title_text(node):
        t = _xp(node, "./*[local-name()='title']")
        return _norm("".join(t[0].itertext())) if t else ""

    def _get_attr_by_localname(el, localname: str):
        for k, v in el.attrib.items():
            if k.split('}')[-1] == localname:
                return v
        return None

    def _dedup_leading_title(title: str, content: str) -> str:
        if not title or not content:
            return content
        pattern = r"^\s*" + re.escape(title) + r"([.:;,-])?\s+"
        return re.sub(pattern, "", content, count=1, flags=re.IGNORECASE)

    def _strip_banned(node):
        _BANNED = {
            "table-wrap","table","thead","tbody","tr","td","th",
            "caption","fig","figure","graphic","media",
            "inline-graphic","disp-formula","inline-formula",
            "supplementary-material"
        }
        node_copy = etree.fromstring(etree.tostring(node))
        for bad in _xp(node_copy, ".//*"):
            if isinstance(bad.tag, str):
                ln = etree.QName(bad.tag).localname
                if ln in _BANNED:
                    parent = bad.getparent()
                    if parent is not None:
                        parent.remove(bad)
        return node_copy

    def is_subtitle(s: str) -> bool:
        s = s.strip()
        return ('.' not in s and '!' not in s and '?' not in s and len(s) > 5)

    def nest_content(content_list: list) -> list:
        new_content = []
        current_sub = None
        for item in content_list:
            if isinstance(item, dict):
                if current_sub:
                    new_content.append(current_sub)
                new_content.append(item)
                current_sub = None
                continue
            # str
            if is_subtitle(item):
                if current_sub:
                    new_content.append(current_sub)
                current_sub = {"title": item, "content": []}
            else:
                if current_sub:
                    current_sub["content"].append(item)
                else:
                    new_content.append(item)
        if current_sub:
            new_content.append(current_sub)
        return new_content

    def parse_sec(node) -> dict:
        title = _title_text(node)
        content = []
        for child in node:
            if not isinstance(child.tag, str):
                continue
            ln = etree.QName(child.tag).localname
            if ln == "title":
                continue
            if ln == "sec":
                sub = parse_sec(child)
                content.append(sub)
            else:
                stripped = _strip_banned(child)
                txt = _text_of(stripped).strip()
                if txt:
                    content.append(txt)
        # Dedup leading title from first paragraph if applicable
        if content and isinstance(content[0], str) and title:
            content[0] = _dedup_leading_title(title, content[0])
        # Nest detected subsections
        content = nest_content(content)
        return {"title": title, "content": content}

    # Parse JATS XML to dict with title, abstract, sections, license
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(journal_xml_bytes, parser=parser)

    output_json = {}

    # Title
    title_nodes = _xp(root, ".//*[local-name()='article-title']")
    if title_nodes:
        output_json["Title"] = _text_of(title_nodes[0])

    # Abstract
    abs_nodes = _xp(root, ".//*[local-name()='abstract']")
    if abs_nodes:
        output_json["Abstract"] = parse_sec(abs_nodes[0])

    # Body sections
    first_level_secs = _xp(root, ".//*[local-name()='body']/*[local-name()='sec']")
    for sec in first_level_secs:
        sec_dict = parse_sec(sec)
        key = sec_dict["title"] or "UNTITLED"
        if key != "Supplementary Information":
            output_json[key] = sec_dict

    # License URL
    ns = {
        "xlink": "http://www.w3.org/1999/xlink",
        "ali": "http://www.niso.org/schemas/ali/1.0/",
    }
    lic_urls = []

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license_ref'] | .//*[local-name()='license_ref']", ns):
        txt = _text_of(n)
        if txt.startswith(("http://", "https://")):
            lic_urls.append(txt)

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license'] | .//*[local-name()='license']", ns):
        href = _get_attr_by_localname(n, "href")
        if href and href.startswith(("http://", "https://")):
            lic_urls.append(href)

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license']//*[local-name()='ext-link']", ns):
        href = _get_attr_by_localname(n, "href")
        if href and href.startswith(("http://", "https://")):
            lic_urls.append(href)

    # Pick canonical license URL
    if lic_urls:
        seen, ordered = set(), []
        for u in lic_urls:
            if not u:
                continue
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)
        if ordered:
            for u in ordered:
                if "creativecommons.org/licenses" in u:
                    output_json["License"] = u
                    break
            else:
                output_json["License"] = ordered[0]
    return output_json


def index_corpus(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a locator index mapping "<S:n>@SECTION" -> {
        section, subheading, paragraph_idx, sentence_idx, char_start, char_end
    }

    Notes to self:
      - `subheading` is the lowest-level title: the value of the sibling "title" that
        sits at the same level as the string paragraph found under "content".
      - Traversal preserves the original top-level key order from the input JSON.
      - Recursively descend through {title, content[list]} nodes until a string paragraph
        is reached; the current node's 'title' becomes the subheading for those strings.
      - `paragraph_idx` is the index of the string inside its immediate parent's 'content' list.
      - Global sentence IDs start at 0 and increment across the whole document.
      - char_* offsets are within a *virtual section-raw* assembled during traversal:
        each (lowest-level) subheading is emitted once as "Heading\\n", each sentence
        adds its length + one trailing space, and each paragraph ends with a newline.
    """

    # sentence splitter (conservative)

    URL_FULL_RE = re.compile(r'^(?:https?|ftp)://\S+$', re.IGNORECASE)

    ABBREV_TOKENS = {
        "e.g.", "i.e.", "vs.", "Fig.", "Figs.", "Dr.", "Mr.", "Ms.", "Mrs.", "Prof.",
        "al.", "et al.", "No.", "Eq.", "Eqs.", "Ref.", "Refs.", "Inc.", "Co.", "Jr.", "Sr.",
        "St.", "Ch.",
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.",
        "Oct.", "Nov.", "Dec.",
        "(Fig.", "(e.g.", "(i.e."
    }

    def sent_tokenize(text: str) -> List[str]:

        if URL_FULL_RE.fullmatch(text.strip()):
            return [text.strip()]

        sents: List[str] = []
        start = 0
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if ch in ".?!":
                # Look left to check for common abbreviations
                left_start = max(0, i - 12)
                left = text[left_start:i+1]
                last_space = left.rfind(" ")
                token = left[(last_space + 1):] if last_space != -1 else left
                token = token.strip()
                is_abbrev = token in ABBREV_TOKENS

                # Find the next non-space character
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1

                # ---- Guard 1: decimal/version number like 3.14 or v2.0 ----
                prev_is_digit = (i > 0 and text[i-1].isdigit())
                next_is_digit = (j < n and text[j].isdigit())
                is_decimal = prev_is_digit and next_is_digit
                if is_decimal:
                    i += 1
                    continue

                # Determine if what follows *looks* like a sentence start
                next_is_start = (j >= n) or text[j].isupper() or text[j].isdigit() or text[j] in "([\"'"

                if not is_abbrev and next_is_start:
                    s = text[start:i+1].strip()
                    if s:
                        sents.append(s)
                    start = j
                    i = j
                    continue
            i += 1

        tail = text[start:].strip()
        if tail:
            sents.append(tail)
        return sents

    # ---------------- utilities ----------------
    def section_label(name: str) -> str:
        # Use the original top-level key, upper-cased, to serve as SECTION
        return name.strip().upper()

    # ---------------- core traversal ----------------
    locs: Dict[str, Dict[str, Any]] = {}
    s_counter = 0

    # For each SECTION we maintain its own "raw" cursor and the last emitted lowest title.
    def build_section_locators(sec_name: str, sec_value: Any):
        nonlocal s_counter
        sec = section_label(sec_name)

        section_char_pos = 0
        last_emitted_lowest_title: Optional[str] = None

        def maybe_emit_heading(heading: Optional[str]):
            nonlocal section_char_pos, last_emitted_lowest_title
            if heading is None:
                return
            if heading != last_emitted_lowest_title:
                section_char_pos += len(f"{heading}\n")
                last_emitted_lowest_title = heading

        def emit_paragraph(subheading: Optional[str], p_idx: int, para_text: str):
            """Tokenize paragraph into sentences and emit locator records."""
            nonlocal section_char_pos, s_counter
            # Ensure heading for this lowest-level block is present in raw
            maybe_emit_heading(subheading)

            sentences = sent_tokenize(str(para_text))
            for s_idx, s_text in enumerate(sentences):
                key = f"<S:{s_counter}>@{sec}"
                char_start = section_char_pos
                char_end = char_start + len(s_text)

                locs[key] = {
                    "section": sec,
                    "subheading": subheading, # lowest-level title
                    "paragraph_idx": p_idx, # index within the immediate parent's 'content' list
                    "sentence_idx": s_idx,
                    "sentence": s_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "global_index": s_counter,
                    "locator_key": key,
                }

                # advance for "virtual raw": sentence + trailing space
                section_char_pos = char_end + 1
                s_counter += 1

            # paragraph break newline
            section_char_pos += 1

        def walk(node: Any, current_lowest_title: Optional[str], *, p_index_context: Optional[int] = None):
            """
            Recursively descend through {title, content[list]} nodes until we reach string paragraphs.
            When we see a string in a 'content' list, we emit it with:
              - subheading = current node's 'title' (lowest-level title at that point)
              - paragraph_idx = index of that string within that list
            """
            # Case 1: leaf paragraph at the top level (SECTION value is a string)
            if isinstance(node, str):
                emit_paragraph(current_lowest_title, p_index_context if p_index_context is not None else 0, node)
                return

            # Case 2: dictionary node expected to have "title" and "content"
            if isinstance(node, dict):
                title_here = node.get("title")
                content = node.get("content")

                # If this dict has no 'content', treat any string fields as one paragraph
                if not isinstance(content, list):
                    # If there's a 'title' and any other stringy value, emit that
                    # Otherwise, nothing to do
                    other_strs = []
                    for k, v in node.items():
                        if k != "title" and isinstance(v, str) and v.strip():
                            other_strs.append(v)
                    if other_strs:
                        emit_paragraph(title_here or current_lowest_title, 0, "\n".join(other_strs))
                    return

                # Iterate the content list preserving order
                for idx, item in enumerate(content):
                    if isinstance(item, str):
                        # We have arrived at the lowest level: use *this* node's title
                        emit_paragraph(title_here or current_lowest_title, idx, item)
                    elif isinstance(item, dict):
                        # Recurse deeper; pass down the child's title, if present
                        child_title = item.get("title", title_here or current_lowest_title)
                        walk(item, child_title, p_index_context=idx)
                    else:
                        # Unknown payload type; coerce to string
                        emit_paragraph(title_here or current_lowest_title, idx, str(item))
                return

            # Case 3: list at top level (unusual) — treat each element
            if isinstance(node, list):
                for idx, item in enumerate(node):
                    if isinstance(item, str):
                        emit_paragraph(current_lowest_title, idx, item)
                    else:
                        walk(item, current_lowest_title, p_index_context=idx)
                return

            # Fallback: nothing to emit
            return

        # Start walking the section
        walk(sec_value, current_lowest_title=None)

    # Preserve the original top-level order
    for top_key, top_val in doc.items():
        build_section_locators(top_key, top_val)

    return locs


def count_chars_per_section(locators_map: dict) -> dict:
    data = locators_map
    keys = list(data.keys())
    processing_section = None
    output = {}

    for key in keys:
        section = data[key].get("section")
        start_char = data[key].get("char_start")
        char_end = data[key].get("char_end")
        
        # First key
        if processing_section is None:
            processing_section = section
        
        if section == processing_section:
            processing_section = section
            processing_end_char = char_end
        else:
            output[processing_section] = {"section": processing_section, "char": processing_end_char}
            processing_section = section
            processing_end_char = char_end

    # Last key
    output[processing_section] = {"section": processing_section, "char": processing_end_char}

    return output




# ======================================================================================
# N03 / N03a — NER helpers (deterministic)
# ======================================================================================

def assemble_summarized_fulltext_for_ner(title, corpus_indexed, section_titles, summarized_sections) -> str:
    fulltext_for_ner = ""

    if title:
        fulltext_for_ner += f"<TITLE>{title}</TITLE>\n\n"
    
    # Group corpus_indexed by section
    sections_dict: Dict[str, List[Tuple[int, str]]] = {}
    for key, loc in corpus_indexed.items():
        sec = loc.get("section", "UNKNOWN")
        s_idx = loc.get("sentence_idx", 0)
        sent = loc.get("sentence", "")
        if sec not in sections_dict:
            sections_dict[sec] = []
        # Extract sentence number from key: "<S:n>@SECTION"
        match = re.match(r"<S:(\d+)>", key)
        s_num = int(match.group(1)) if match else s_idx
        sections_dict[sec].append((s_num, sent))
    
    # Sort each section's sentences by sentence number
    for sec in sections_dict:
        sections_dict[sec].sort(key=lambda x: x[0])
    
    # Build fulltext: summarized sections vs sentence-tagged sections
    for sec_title in section_titles:
        fulltext_for_ner += f"\n<SECTION:{sec_title}>\n"
        
        if sec_title in summarized_sections:
            # Use summary
            fulltext_for_ner += f"{summarized_sections[sec_title]}\n"
        else:
            # Emit sentences with IDs
            for s_num, sent in sections_dict.get(sec_title, []):
                fulltext_for_ner += f"<S:{s_num}>{sent}</S:{s_num}> "
            fulltext_for_ner += "\n"
        
        fulltext_for_ner += f"</SECTION:{sec_title}>\n"
        
    return fulltext_for_ner


def negation_hedge_detector(text: str) -> Dict[str, bool]:
    """
    Detect negation/hedging tokens in text.

    Returns
    -------
    Dict[str, bool]
        {"negated": bool, "hedged": bool}
    """
    # TODO: implement token-based detection.
    return {"negated": False, "hedged": False}


def normalize_method_names(names: Sequence[str]) -> List[str]:
    """
    Canonicalize assay/method variants (e.g., LC-MS/MS synonyms).

    Returns
    -------
    List[str]
        Normalized method names.
    """
    # TODO: implement normalization table.
    return list(names)


"""
# Example usage of retrieve_pmc_fulltext
if __name__ == "__main__":
    pmcid = "PMC11461603"
    sections = retrieve_pmc_fulltext(pmcid)
    for k, v in sections.items():
        print(f"\n=== {k} ===\n{v}")
"""
