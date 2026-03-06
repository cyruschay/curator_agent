# graph.py — LangGraph pipeline

from typing import Any, Dict, List, Optional, Tuple, Sequence
from copy import deepcopy
from pathlib import Path
import yaml
import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage

load_dotenv()

# Load configuration from YAML
config_path = Path(__file__).parents[2] / "config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# STATE
from states import AgentState

# PROMPTS
from prompts import (
    SCREENING_SYS_PROMPT, NER_ASSIST_SYS_PROMPT, RELATION_EXTRACT_SYS_PROMPT,
    ONTOLOGY_MAPPING_SYS_PROMPT, VALIDATION_SYS_PROMPT, SUMMARIZER_SYS_PROMPT,
    SECTION_FILTER_SYS_PROMPT
)

# SCHEMAS
from schemas import INPUT_SCHEMA

# TOOLS (LLM-bound tools)
from tools import (
    onto_gsd_tool, onto_doid_tool, onto_uberon_tool,
    onto_cellline_tool, onto_taxonomy_tool, onto_protein_tool,
)

ontology_mapping_tools = [
    onto_gsd_tool, onto_doid_tool, onto_uberon_tool,
    onto_cellline_tool, onto_taxonomy_tool, onto_protein_tool,
]

# UTILS (deterministic, non-LLM helpers)
from utils import (
    # Preprocessing
    line_hash_check,                 # (line:str, listf:Path|str) -> "skip"|"process"
    validate_schema,                 # (json_record:dict, schema:dict) -> "valid"|"invalid"
    build_agent_state_from_input_obj,# (obj:dict, ingest_hash:str) -> AgentState
    update_curated_status,

    # Shared helpers
    OllamaModelParams,
    _print_msg,
    _relations_found,
    _dedup_relations,
    _invoke_llm,
    _parse_llm_resp,
    _print_llm_metadata,
    _llm_log_entry,

    # N01 — Screener
    feature_scorer,                  # (features:Dict[str,int]) -> Dict

    # N02 — Retrieval & Corpus
    retrieve_pmc_fulltext,           # (pmcid:str) -> Dict[str,Any]
    index_corpus,                   # (sections_json:dict) -> {key->Locator}
    count_chars_per_section,            # (sections_json:dict, token_threshold:int) -> dict

    # N03 — NER helpers
    assemble_summarized_fulltext_for_ner,  # (title_doc, corpus_indexed, section_titles, summarized_sections) -> str

    # N04 — Relation helpers
    negation_hedge_detector,         # (text:str) -> {"negated":bool,"hedged":bool}
    normalize_method_names,          # (names:[str]) -> [str]
)


# -----------------------------------------------------------------------------
# Configuration (paths & models)
# -----------------------------------------------------------------------------

# Project paths
INFILE_NAME = config_data["paths"]["input_abstracts"]
LIST_NAME = "curated_list.jsonl"

input_path = Path(__file__).parents[2] / "data" / "raw" / "abstracts" / INFILE_NAME
curation_dir = Path(__file__).parents[2] / "data" / "processed" / "curation" / config_data["paths"]["curation_output_dir"]
curation_dir.mkdir(parents=True, exist_ok=True)

list_path = curation_dir / LIST_NAME
if not list_path.exists():
    list_path.touch()


# Models
REASONING_MODEL = ChatOllama(
    model=config_data["ollama_models"]["reasoning_model"],
    **OllamaModelParams.from_config(config_data, "reasoning_model").to_kwargs(),
)
MINI_MODEL = ChatOllama(
    model=config_data["ollama_models"]["mini_model"],
    **OllamaModelParams.from_config(config_data, "mini_model").to_kwargs(),
)

NANO_MODEL = ChatOllama(
    model=config_data["ollama_models"]["nano_model"],
    **OllamaModelParams.from_config(config_data, "nano_model").to_kwargs(),
)


# Safety limits (avoid runaway generations / context truncation)
NER_MAX_INPUT_CHARS = 1500000


screening_model = REASONING_MODEL
ner_model = REASONING_MODEL
re_model = REASONING_MODEL
mapper_model = REASONING_MODEL.bind_tools(ontology_mapping_tools)
validator_model = REASONING_MODEL
mini_model = MINI_MODEL
nano_model = NANO_MODEL


# -----------------------------------------------------------------------------
# Shared Helpers (dict extraction, evidence resolution, post-processing)
# -----------------------------------------------------------------------------

# Field name lists used by _pick / _pick_first to reduce verbose .get() chains
_RELATION_KEYS = [
    "glycan_name", "glycan_mapped_name", "glycan_id", "glycan_metadata",
    "disease_name", "disease_mapped_name", "disease_id",
    "disease_annotation", "treatment_annotation",
    "specimen",
    "species_name", "species_mapped_name", "species_id",
    "protein_name", "protein_mapped_name", "protein_id",
    "biomarker_type", "direction",
]

_ENTITY_META_KEYS = [
    "glycan_structure_term", "non_structural_descriptor", "alignment", "aglycon",
]

_ENTITY_EXPORT_KEYS = [
    "glycan_structure_term", "non_structural_descriptor",
    "alignment", "aglycon", "chemical_structure", "evidence_sentence",
]


def _pick(src: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Extract a subset of keys from *src*."""
    return {k: src.get(k) for k in keys}


def _pick_first(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    keys: Sequence[str],
) -> Dict[str, Any]:
    """For each key, use *primary* value if truthy, otherwise *fallback*."""
    return {k: primary.get(k) or fallback.get(k) for k in keys}


def _build_locator_index(
    corpus_indexed: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Map ``global_index`` → corpus entry for fast sentence lookup."""
    return {
        entry["global_index"]: entry
        for entry in corpus_indexed.values()
        if isinstance(entry.get("global_index"), int)
    }


def _resolve_evidence(
    evidence_locators: list,
    corpus_indexed: Dict[str, Any],
    locator_by_index: Dict[int, Dict[str, Any]],
    *,
    relation_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Resolve evidence locators → ``[{sentence, section, locator}, ...]``.

    Skips non-dict entries (corrupt data from LLM) with a warning.
    Mutates each *loc* in-place to back-fill ``section_id`` / ``locator_key``.
    """
    results: List[Dict[str, Any]] = []
    for loc in evidence_locators:
        if not isinstance(loc, dict):
            if relation_idx is not None:
                _print_msg("WARN", f"Relation {relation_idx}: skipping non-dict evidence locator")
            continue

        sentence_text = ""
        section_id = loc.get("section_id", "")
        locator_key = loc.get("locator_key")

        if locator_key and locator_key in corpus_indexed:
            entry = corpus_indexed[locator_key]
            sentence_text = entry.get("sentence", "")
            section_id = entry.get("section", section_id)
            loc.setdefault("section_id", section_id)
        else:
            global_idx = loc.get("global_index", loc.get("sentence_index"))
            if isinstance(global_idx, int):
                entry = locator_by_index.get(global_idx)
                if entry:
                    sentence_text = entry.get("sentence", "")
                    section_id = entry.get("section", section_id)
                    loc.setdefault("section_id", section_id)
                    loc.setdefault("locator_key", entry.get("locator_key"))

        if sentence_text:
            results.append({"sentence": sentence_text, "section": section_id, "locator": loc})
    return results


def _build_evidence_locators(
    sentence_indexes: list,
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    """Convert raw sentence indexes → ``(locators, evidence_texts, normalized_indexes)``."""
    locators: List[Dict[str, Any]] = []
    texts: List[str] = []
    normalized: List[int] = []
    for s_idx in sentence_indexes:
        try:
            s_idx_int = int(s_idx)
        except (TypeError, ValueError):
            continue
        lookup = index_lookup.get(s_idx_int)
        if not lookup:
            continue
        locator_key, loc = lookup
        locators.append({
            "section_id": loc.get("section", ""),
            "sentence_index": s_idx_int,
            "global_index": s_idx_int,
            "char_start": loc.get("char_start", 0),
            "char_end": loc.get("char_end", 0),
            "locator_key": locator_key,
        })
        texts.append(loc.get("sentence", ""))
        normalized.append(s_idx_int)
    return locators, texts, normalized


def _postprocess_relations(
    extracted: List[Dict[str, Any]],
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]],
) -> None:
    """In-place: attach evidence locators, normalize methods, detect hedging."""
    for rel in extracted:
        locators, texts, normalized = _build_evidence_locators(
            rel.get("evidence_sentence_indexes", []), index_lookup,
        )
        rel["evidence_locators"] = locators
        if normalized:
            rel["evidence_sentence_indexes"] = normalized

        rel["method_names"] = normalize_method_names(rel.get("method_names") or [])
        rel["metrics"] = [
            {"name": m.get("name", ""), "value": m.get("value"), "raw": m.get("raw")}
            for m in rel.get("metrics", [])
        ]

        if "negated_or_hedged" not in rel:
            ev_txt = " ".join(t for t in texts if t)
            if ev_txt:
                hedges = negation_hedge_detector(ev_txt)
                rel["negated_or_hedged"] = bool(hedges.get("negated") or hedges.get("hedged"))
            else:
                rel["negated_or_hedged"] = False


def _build_node_output(
    base: Dict[str, Any],
    messages: Optional[List] = None,
    reasonings: Optional[List] = None,
    llm_logs: Optional[List] = None,
) -> Dict[str, Any]:
    """Attach optional message / reasoning / log lists to a node output dict."""
    if messages:
        base["messages"] = messages
    if reasonings:
        base["reasonings"] = reasonings
    if llm_logs:
        base["llm_logs"] = llm_logs
    return base


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------
# N01 — Abstract Screener
def node_n01_screener(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N01: Abstract Screener")
    
    # Read from state
    title = state["doc"]["title"]
    abstract = state["doc"]["abstract"]
    metadata = state["doc"].get("metadata", {})
    
    # Handle missing abstract
    if abstract is None or abstract.strip() == "":
        node_n01_output = {
            "screening": {
                "decision": "IRRELEVANT",
                "reason": "No abstract text provided.",
            },
            "messages": [AIMessage(content="Node_n01. Missing abstract text for screening.")]
        }
        return node_n01_output
    
    # Deterministic feature scoring
    features = feature_scorer(title, abstract, metadata)
    score = features.get("score", 0)

    # Invoke LLM
    sm = SystemMessage(content=SCREENING_SYS_PROMPT)
    hm = HumanMessage(content=f"Title:\n{title}\n\nAbstract:\n{abstract}")
    resp = _invoke_llm(sm, hm, screening_model)
    
    _print_llm_metadata(resp)
    llm_logs = [_llm_log_entry(resp, "n01")]
    new_messages, new_reasonings, output_json = _parse_llm_resp(resp=resp, node_id="n01")

    # Extract output_json fields; override LLM's RELEVANT decision if feature score is -1
    if "decision" in output_json:
        decision = output_json["decision"] if score != -1 else "IRRELEVANT (overridden)"
    else:
        new_messages.append(AIMessage(content=f"Node_n01. ERROR: Missing field 'decision' in LLM output"))
        decision = "IRRELEVANT"
        
    reason = output_json.get("reasoning", "[ERROR]")
    study_type = output_json.get("study_type", "[ERROR]")
    needs_fulltext = output_json.get("needs_fulltext", False)

    
    # build output
    node_n01_output = {
        "screening": {
            "decision": decision,
            "reason": reason,
            "study_type": study_type,
            "flags": {"needs_fulltext": bool(needs_fulltext)},
            "confidence": float(score),
        },
        "flags": {"needs_fulltext": bool(needs_fulltext)},
        "tags": {"lexicon_hits": features},
        "messages": new_messages, "reasonings": new_reasonings,
        "llm_logs": llm_logs,
    }

    return node_n01_output

# N02 — Full Text Retrieval   # A little messy
def node_n02_retrieval(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N02: Full Text Retrieval")

    
    flags = state.get("flags", {})
    flags["text_mode"] = "abstract"
    flags["can_use_fulltext"] = False
    flags["has_pmcid"] = False
    
    # First check if PMCID exists; if not, fall back to abstract-only
    pmcid = state["ids"].get("pmcid", None)
    if pmcid and pmcid.isdigit():
        flags["has_pmcid"] = True
    else:
        return {"flags": flags}

    # PMCID exists. Initialize these agent state top-level keys
    corpus_raw = {}
    corpus_indexed = {}
    chunk_index = {}

    try:
        corpus_raw = retrieve_pmc_fulltext(str(pmcid))
        flags["can_use_fulltext"] = True
        flags["fulltext_license"] = corpus_raw.get("License")

        # Index sentences from corpus_raw
        corpus_indexed = index_corpus(corpus_raw)

        ### Note TO DEVELOPER: Chunking is currently not supported but intended in future versions
        section_lengths = count_chars_per_section(corpus_indexed)
        total_chars = sum(v["char"] for v in section_lengths.values())
        if total_chars > 300000: # threshold approximating the model input tokens limit
            flags["text_mode"] = "chunked"
        else:
            flags["text_mode"] = "fulltext"

    except Exception as e:
        # leave corpus as abstract-only
        new_messages = [AIMessage(content=f"Node_n02. ERROR: Full text retrieval (retriever/indexer) PMCID:{pmcid}. {e}")]
        return {"flags": flags, "messages": new_messages}
    
    node_n02_output = {
        "corpus_raw": corpus_raw,
        "corpus_indexed": corpus_indexed,
        "chunk_index": {"total_chars": total_chars},
        "flags": flags,
    }

    return node_n02_output

# N03 — Full Text NER
def node_n03_fulltext_ner(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N03: Full Text NER")
    
    # Retrieve indexed corpus from state, get section titles
    corpus_indexed = state.get("corpus_indexed", {})
    section_titles = list(dict.fromkeys(v["section"] for v in corpus_indexed.values()))
    
    # LLM: decides which sections to summarize, access summarizer model, returns summarized_sections
    summarized_sections = {}

    # Step 1 — Filter section titles to summarize -> selected_titles
    selected_titles = []
    filt_sm = SystemMessage(content=(SECTION_FILTER_SYS_PROMPT))
    filt_hm = HumanMessage(content=json.dumps({"section_titles": section_titles}))
    filt_resp = _invoke_llm(filt_sm, filt_hm, nano_model)
    
    _print_llm_metadata(filt_resp)
    llm_logs: List[Dict[str, Any]] = [_llm_log_entry(filt_resp, "n03_section_filter")]
    filt_messages, filt_reasonings, filt_json = _parse_llm_resp(filt_resp, node_id="n03_section_filter")

    if isinstance(filt_json, list):
        selected_titles = [str(t) for t in filt_json]
    elif isinstance(filt_json, dict) and isinstance(filt_json.get("titles"), list):
        selected_titles = [str(t) for t in filt_json.get("titles", [])]
    else:
        filt_messages.append(AIMessage(content=f"ERROR: Unexpected nano filter output format."))
        # Fallback: simple keyword filter
        kw = ("INTRO", "METHOD", "MATERIAL", "EXPERIMENT", "PROCEDURE")
        selected_titles = [t for t in section_titles if any(k in t.upper() for k in kw)]

    # Initialize message lists
    sum_messages: List[BaseMessage] = []
    sum_reasonings: List[BaseMessage] = []

    # Step 2 — Summarize selected sections with the mini model by pulling sentences from corpus_indexed
    if selected_titles:
        # Build 'per-section text' from sentences in corpus_indexed
        per_section_text = {}

        for title in selected_titles:
            sents = [v.get("sentence", "") for v in corpus_indexed.values() if v.get("section") == title]
            # Light length control
            joined = " ".join(s for s in sents if s).strip()
            per_section_text[title] = joined[:100000]  # cap to keep prompt manageable

        # If we have content, ask mini model to summarize per section
        if any(per_section_text.values()):
            # Prefer existing prompt if suitable; otherwise instruct JSON output
            sum_sm = SystemMessage(content=(SUMMARIZER_SYS_PROMPT))
            sum_payload = {"sections": [{"title": t, "text": per_section_text[t]} for t in selected_titles]}
            sum_hm = HumanMessage(content=json.dumps(sum_payload)[:100000])
            
            mini_resp = _invoke_llm(sum_sm, sum_hm, mini_model)
            #print(mini_resp)
            _print_llm_metadata(mini_resp)
            llm_logs.append(_llm_log_entry(mini_resp, "n03_section_summarizer"))
            sum_messages, sum_reasonings, sum_json = _parse_llm_resp(mini_resp, node_id="n03_section_summarizer")

            if isinstance(sum_json, dict) and isinstance(sum_json.get("summarized_sections"), dict):
                summarized_sections = {str(k): str(v) for k, v in sum_json["summarized_sections"].items()}
            else:
                sum_messages.append(AIMessage(content=f"ERROR: Unexpected mini summarizer output format."))
                # Fallback: naive first-3-sentences summary
                for t, txt in per_section_text.items():
                    parts = re.split(r"(?<=[.!?])\s+", txt)
                    summarized_sections[t] = " ".join(parts[:3]).strip()
                    
    # Step 3 - Assemble full text for LLM with summarized sections + sentence IDs
    #   For sections in summarized_sections: replace with summary text (no sentence IDs)
    #   For all other sections: emit sentences with XML tags <S:n>sentence</S:n>    
    title = state["doc"]["title"]
    fulltext_for_ner = assemble_summarized_fulltext_for_ner(title, corpus_indexed, section_titles, summarized_sections)

    # Truncate if extremely long
    fulltext_for_ner = fulltext_for_ner[:NER_MAX_INPUT_CHARS]
    
    # Invoke NER model to extract glycan entities
    ner_sm = SystemMessage(content=NER_ASSIST_SYS_PROMPT)
    ner_hm = HumanMessage(content=fulltext_for_ner)

    entities: List[Dict[str, Any]] = []
    ner_resp = _invoke_llm(ner_sm, ner_hm, ner_model)
    _print_llm_metadata(ner_resp)
    llm_logs.append(_llm_log_entry(ner_resp, "n03_ner"))
    ner_messages, ner_reasonings, ner_json = _parse_llm_resp(ner_resp, node_id="n03_ner")
    
    # Patch: handle missing 'glycans' field and list only
    if isinstance(ner_json, list):
        ner_json = {"glycans": ner_json}
        ner_messages.append(AIMessage(content="WARNING: NER output was a list; wrapped into {'glycans': [...] }"))
    
    glycans_list = ner_json.get("glycans", "ERROR")
    if glycans_list == "ERROR":
        ner_messages.append(AIMessage(content="ERROR: Missing 'glycans' field in NER output"))
        glycans_list = []

    # Build index lookup to map sentence_index back to sentence text
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for key, loc in corpus_indexed.items():
        sentence_idx = loc.get("global_index")
        if sentence_idx is not None:
            index_lookup[sentence_idx] = (key, loc)

    for g in glycans_list:
        evidence_idx = g.get("evidence_sentence_index")
        evidence_sentence = None
        
        # Map evidence_sentence_index back to actual sentence text
        if evidence_idx is not None and evidence_idx in index_lookup:
            _, loc_entry = index_lookup[evidence_idx]
            evidence_sentence = loc_entry.get("sentence")
        
        ent = {
            "glycan_structure_term": g.get("glycan_structure_term", None),
            "non_structural_descriptor": g.get("non_structural_descriptor", None),
            "alignment": g.get("alignment", None),
            "aglycon": g.get("aglycon", None),
            "chemical_structure": g.get("chemical_structure", None),
            "evidence_sentence": evidence_sentence,
        }
        if ent["glycan_structure_term"]:
            entities.append(ent)

    # Deduplicate by glycan_structure_term (case-insensitive)
    uniq, seen = [], set()
    for e in entities:
        key = e["glycan_structure_term"]
        if key not in seen:
            uniq.append(e)
            seen.add(key)
    
    cands = state.get("candidates", {}) or {}
    cands["entities"] = uniq

    new_messages: List[BaseMessage] = []
    for chunk in (filt_messages, sum_messages, ner_messages):
        new_messages.extend(chunk)

    new_reasonings: List[BaseMessage] = []
    for chunk in (filt_reasonings, sum_reasonings, ner_reasonings):
        new_reasonings.extend(chunk)
    node_n03_output = {
        "candidates": cands,
        "summarized_sections": summarized_sections,
        "messages": new_messages,
        "reasonings": new_reasonings,
        "llm_logs": llm_logs,
    }
    
    return node_n03_output

# N03a — Abstract Only NER
def node_n03a_abstract_ner(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N03a: Abstract Only NER")
    
    doc = state.get("doc", {})
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
   
    # Create a minimal doc structure for index_corpus
    abstract_doc = {
        "Title": title,
        "Abstract": abstract  # flat string will be tokenized
    }

    # Index abstract, recycle assemble_summarized_fulltext_for_ner
    abstract_indexed = index_corpus(abstract_doc)

    section_titles = list(dict.fromkeys(v["section"] for v in abstract_indexed.values())) # i.e. ["TITLE","ABSTRACT"]
    summarized_sections = {}
    fulltext_for_ner = assemble_summarized_fulltext_for_ner(title, abstract_indexed, section_titles, summarized_sections)
    

    # Invoke NER model
    sm = SystemMessage(content=NER_ASSIST_SYS_PROMPT)
    hm = HumanMessage(content=fulltext_for_ner[:12000])
    
    entities = []
    resp = _invoke_llm(sm, hm, ner_model)
    _print_llm_metadata(resp)
    llm_logs = [_llm_log_entry(resp, "n03a_ner")]
    ner_messages, ner_reasonings, ner_json = _parse_llm_resp(resp, node_id="n03a_ner")

    glycans_list = ner_json.get("glycans", "ERROR")
    if glycans_list == "ERROR":
        ner_messages.append(AIMessage(content="ERROR: Missing 'glycans' field in NER output"))
        glycans_list = []
    
    # Build index lookup to map sentence_index back to sentence text
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for key, loc in abstract_indexed.items():
        sentence_idx = loc.get("global_index")
        if sentence_idx is not None:
            index_lookup[sentence_idx] = (key, loc)
    
    for g in glycans_list:
        evidence_idx = g.get("evidence_sentence_index")
        evidence_sentence = None
        
        # Map evidence_sentence_index back to actual sentence text
        if evidence_idx is not None and evidence_idx in index_lookup:
            _, loc_entry = index_lookup[evidence_idx]
            evidence_sentence = loc_entry.get("sentence")
        
        ent = {
            "glycan_structure_term": g.get("glycan_structure_term", ""),
            "non_structural_descriptor": g.get("non_structural_descriptor"),
            "alignment": g.get("alignment", "whole_structure"),
            "aglycon": g.get("aglycon", "other"),
            "chemical_structure": g.get("chemical_structure", None),
            "spans": [], # DEV
            "evidence_sentence": evidence_sentence,
        }
        if ent["glycan_structure_term"]:
            entities.append(ent)

    # Deduplicate
    uniq, seen = [], set()
    for e in entities:
        key = e["glycan_structure_term"].lower()
        if key not in seen:
            uniq.append(e)
            seen.add(key)

    cands = state.get("candidates", {}) or {}
    cands["entities"] = uniq
    
    node_n03a_output = {
        "candidates": cands,
        "corpus_indexed": abstract_indexed,
        "messages": ner_messages,
        "reasonings": ner_reasonings,
        "llm_logs": llm_logs,
    }
    
    return node_n03a_output

# N04 — Full Text RE
def node_n04_fulltext_re(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N04: Relation Extraction (full text)")
    
    # Retrieve corpus and entities from state
    corpus_indexed = state.get("corpus_indexed", {})
    section_titles = list(dict.fromkeys(v["section"] for v in corpus_indexed.values()))
    summarized_sections = state.get("summarized_sections", {})
    entities = state.get("candidates", {}).get("entities", [])
    
    # Build full text for LLM (same strategy as N03)
    fulltext_for_re = ""
    
    # Add title
    title_doc = state.get("doc", {}).get("title", "")
    if title_doc:
        fulltext_for_re += f"<TITLE>{title_doc}</TITLE>\n\n"
    
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
        fulltext_for_re += f"\n<SECTION:{sec_title}>\n"
        
        if sec_title in summarized_sections:
            # Use summary (no sentence IDs for summarized content)
            fulltext_for_re += f"{summarized_sections[sec_title]}\n"
        else:
            # Emit sentences with IDs
            for s_num, sent in sections_dict.get(sec_title, []):
                fulltext_for_re += f"<S:{s_num}>{sent}</S:{s_num}> "
            fulltext_for_re += "\n"
        
        fulltext_for_re += f"</SECTION:{sec_title}>\n"
    
    # Append entity list for reference
    entity_list_str = json.dumps([{
        "glycan_structure_term": e.get("glycan_structure_term"),
        "alignment": e.get("alignment"),
        "aglycon": e.get("aglycon"),
        "chemical_structure": e.get("chemical_structure"),
    } for e in entities], indent=2)
    
    fulltext_for_re += f"\n\n<ENTITIES>\n{entity_list_str}\n</ENTITIES>\n"
    
    # Truncate if extremely long
    fulltext_for_re = fulltext_for_re[:150000]
    
    # Invoke RE model
    sm = SystemMessage(content=RELATION_EXTRACT_SYS_PROMPT)
    hm = HumanMessage(content=fulltext_for_re)
    
    resp = _invoke_llm(sm, hm, re_model)
    _print_llm_metadata(resp)
    llm_logs = [_llm_log_entry(resp, "n04")]
    re_messages, re_reasonings, output_json = _parse_llm_resp(resp=resp, node_id="n04")

    if "relations" in output_json:
        extracted = output_json.get("relations")
    else:
        extracted = []
        re_messages.append(AIMessage(content=f"[n04_parse_error] Missing 'relations' field in LLM output"))
    
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for key, loc in corpus_indexed.items():
        global_idx = loc.get("global_index")
        if isinstance(global_idx, int):
            index_lookup[global_idx] = (key, loc)

    # Post-process: build evidence locators, normalize methods, detect hedging
    _postprocess_relations(extracted, index_lookup)
    
    cands = state.get("candidates", {}) or {}
    cands["relations"] = extracted
    return {"candidates": cands, "messages": re_messages, "reasonings": re_reasonings, "llm_logs": llm_logs}

# N04a — Abstract-only RE
def node_n04a_abstract_re(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N04a: Relation Extraction (abstract)")
    
    doc = state.get("doc", {})
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    entities = state.get("candidates", {}).get("entities", [])
    
    # Retrieve abstract_indexed from N03a (or rebuild if missing)
    abstract_indexed = state.get("corpus_indexed", {})
    
    if not abstract_indexed:
        # Rebuild if N03a didn't persist it
        abstract_doc = {
            "Title": title,
            "Abstract": abstract
        }
        abstract_indexed = index_corpus(abstract_doc)
    
    # Build fulltext for RE with sentence IDs (same strategy as N04)
    fulltext_for_re = ""
    
    if title:
        fulltext_for_re += f"<TITLE>{title}</TITLE>\n\n"
    
    # Group by section
    sections_dict: Dict[str, List[Tuple[int, str]]] = {}
    for key, loc in abstract_indexed.items():
        sec = loc.get("section", "UNKNOWN")
        sent = loc.get("sentence", "")
        if sec not in sections_dict:
            sections_dict[sec] = []
        match = re.match(r"<S:(\d+)>", key)
        s_num = int(match.group(1)) if match else 0
        sections_dict[sec].append((s_num, sent))
    
    # Sort sentences
    for sec in sections_dict:
        sections_dict[sec].sort(key=lambda x: x[0])
    
    # Build structured text
    for sec_title in ["TITLE", "ABSTRACT"]:
        if sec_title not in sections_dict:
            continue
        fulltext_for_re += f"\n<SECTION:{sec_title}>\n"
        for s_num, sent in sections_dict.get(sec_title, []):
            fulltext_for_re += f"<S:{s_num}>{sent}</S:{s_num}> "
        fulltext_for_re += "\n"
        fulltext_for_re += f"</SECTION:{sec_title}>\n"
    
    
    # Append entity list
    entity_list_str = json.dumps([{
        "glycan_structure_term": e.get("glycan_structure_term"),
        "alignment": e.get("alignment"),
        "aglycon": e.get("aglycon"),
        "chemical_structure": e.get("chemical_structure"),
    } for e in entities], indent=2)
    
    fulltext_for_re += f"\n\n<ENTITIES>\n{entity_list_str}\n</ENTITIES>\n"
    
    # (log suppressed) Fulltext length
    
    # Invoke RE model
    sm = SystemMessage(content=RELATION_EXTRACT_SYS_PROMPT)
    hm = HumanMessage(content=fulltext_for_re[:12000])
    
    extracted: List[Dict[str, Any]] = []
    resp = _invoke_llm(sm, hm, re_model)
    _print_llm_metadata(resp)
    llm_logs = [_llm_log_entry(resp, "n04a")]
    re_messages, re_reasonings, output_json = _parse_llm_resp(resp=resp, node_id="n04a")

    
    if isinstance(output_json, dict) and "relations" in output_json:
        extracted = output_json.get("relations", [])
    else:
        re_messages.append(AIMessage(content="Node_n04a. ERROR: Missing 'relations' in LLM output"))
        extracted = []
    
    _print_msg("INFO", f"Extracted {len(extracted)} relations")
    
    index_lookup: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for key, loc in abstract_indexed.items():
        global_idx = loc.get("global_index")
        if isinstance(global_idx, int):
            index_lookup[global_idx] = (key, loc)

    # Post-process: build evidence locators, normalize methods, detect hedging
    _postprocess_relations(extracted, index_lookup)
        
    cands = state.get("candidates", {}) or {}
    cands["relations"] = extracted
    return {"candidates": cands, "messages": re_messages, "reasonings": re_reasonings, "llm_logs": llm_logs}

# N05 — Ontology Mapper
def node_n05_mapper(state: AgentState) -> AgentState:
    """
    Map extracted relation entities to standardized ontology identifiers using LLM + tools.
    
    1. Read the relations and identify which ontology tools to call
    2. Call multiple ontology tools (GSD, DOID, Uberon, Cellosaurus, Taxonomy, UniProt)
    3. Review the results and select the best matching IDs
    """
    _print_msg("INFO", "Running N05: Ontology Mapping")
    
    candidates_state = state.get("candidates", {}) or {}
    cands = candidates_state.get("relations", []) or []
    entities = candidates_state.get("entities", []) or []
    
    new_messages: List[BaseMessage] = []
    new_reasonings: List[BaseMessage] = []
    llm_logs: List[Dict[str, Any]] = []

    if not cands:
        return {"mapped": {"relations": []}}

    # Build quick lookup so we can carry entity metadata forward into relations
    entity_lookup: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        surface = (ent.get("glycan_structure_term") or "").strip().lower()
        if surface and surface not in entity_lookup:
            entity_lookup[surface] = ent

    def _resolve_glycan_entity_metadata(*candidates: Optional[str]) -> Optional[Dict[str, Any]]:
        for candidate in candidates:
            key = (candidate or "").strip().lower()
            if key and key in entity_lookup:
                ent = entity_lookup[key]
                return {
                    "glycan_structure_term": ent.get("glycan_structure_term"),
                    "non_structural_descriptor": ent.get("non_structural_descriptor"),
                    "alignment": ent.get("alignment"),
                    "aglycon": ent.get("aglycon"),
                }
        return None
    
    # Prepare input for LLM: relations with context
    relations_input = []
    for idx, r in enumerate(cands):
        relations_input.append({
            "index": idx,
            "glycan": r.get("glycan"),
            "glycan_metadata": r.get("glycan_metadata"),
            "biomarker_type": r.get("biomarker_type"),
            "change": r.get("change"),
            "disease": r.get("disease"),
            "disease_annotation": r.get("disease_annotation"),
            "treatment_annotation": r.get("treatment_annotation"),
            "specimen": r.get("specimen"),
            "species": r.get("species"),
            "protein_name": r.get("protein_name"),
            "cazy_enzyme": r.get("cazy_enzyme"),
            "direction": r.get("change") or r.get("direction"),
            "evidence_locators": r.get("evidence_locators", []),
            "metrics": r.get("metrics", []),
            "method_names": r.get("method_names", []),
            "negated_or_hedged": r.get("negated_or_hedged", False),
        })
    

    mapping_context = {"relations": relations_input}
    
    # Create a tools dictionary for manual execution
    tools_dict = {
        onto_gsd_tool.name: onto_gsd_tool,
        onto_doid_tool.name: onto_doid_tool,
        onto_uberon_tool.name: onto_uberon_tool,
        onto_cellline_tool.name: onto_cellline_tool,
        onto_taxonomy_tool.name: onto_taxonomy_tool,
        onto_protein_tool.name: onto_protein_tool,
    }
    
    sm = SystemMessage(content=ONTOLOGY_MAPPING_SYS_PROMPT)
    hm = HumanMessage(content=json.dumps(mapping_context, indent=2))
    
    def _fallback_relations(note: str) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for r in cands:
            glycan_surface = r.get("glycan")
            glycan_mapped = r.get("glycan")
            meta = _resolve_glycan_entity_metadata(glycan_surface, glycan_mapped)
            fallback_rel = {
                "glycan_name": r.get("glycan"),
                "glycan_mapped_name": None,
                "glycan_id": None,
                "glycan_metadata": r.get("glycan_metadata"),
                "disease_name": r.get("disease"),
                "disease_mapped_name": None,
                "disease_id": None,
                "disease_annotation": r.get("disease_annotation"),
                "treatment_annotation": r.get("treatment_annotation"),
                "specimen": {
                    "original": r.get("specimen"),
                    "mapped_name": None,
                    "mapped_id": None,
                    "category": None,
                    "ontology": None,
                },
                "species_name": r.get("species"),
                "species_mapped_name": None,
                "species_id": None,
                "protein_name": r.get("protein_name"),
                "protein_mapped_name": None,
                "protein_id": None,
                "biomarker_type": r.get("biomarker_type"),
                "direction": r.get("change") or r.get("direction"),
                "evidence_locators": r.get("evidence_locators", []),
                "metrics": r.get("metrics", []),
                "method_names": r.get("method_names", []),
                "notes": note,
            }
            if meta:
                fallback_rel["glycan_entity_metadata"] = meta
            mapped.append(fallback_rel)
        return mapped

    try:
        conversation: List[BaseMessage] = [sm, hm]
        resp = _invoke_llm(sm, hm, mapper_model)
        _print_llm_metadata(resp)
        llm_logs.append(_llm_log_entry(resp, "n05"))

        if resp.get("reasoning"):
            new_reasonings.append(AIMessage(content=f"Node_n05. {resp['reasoning']}"))
        if resp.get("error"):
            new_messages.append(AIMessage(content=f"Node_n05. ERROR: {resp['error']}"))
            raise RuntimeError(resp["error"])

        # Extract output and build initial AI message
        parse_messages, parse_reasonings, mapping_json = _parse_llm_resp(resp, node_id="n05")
        new_messages.extend(parse_messages)
        new_reasonings.extend(parse_reasonings)

        # Build AIMessage for conversation history (content may be empty if tool_calls-only)
        ai_message = AIMessage(
            content=resp.get("output") or "",
            tool_calls=resp.get("tool_calls") or [],
        )
        conversation.append(ai_message)

        # Keep invoking until no more tool calls
        max_iterations = 10  # Safety limit
        iteration = 0
        tool_calls = resp.get("tool_calls")
        
        while tool_calls and iteration < max_iterations:
            iteration += 1
            # (log suppressed) Tool call iteration

            tool_messages: List[ToolMessage] = []
            for tool_call in tool_calls:
                tool_name = tool_call.get('name') or tool_call.get('type')
                tool_args = tool_call.get('args') or {}
                tool_call_id = tool_call.get('id')
                
                _print_msg("INFO", f"[N05] Calling tool: {tool_name} with args: {tool_args}")
                
                # Check if tool exists
                if tool_name not in tools_dict:
                    result = f"Error: Tool {tool_name} does not exist."
                    print(f"[N05] {result}")
                else:
                    try:
                        # Invoke the tool
                        result = tools_dict[tool_name].invoke(tool_args)
                        #_print_msg("INFO", f"[N05] Tool result length: {len(str(result))}")
                    except Exception as e:
                        result = f"Error calling tool {tool_name}: {str(e)}"
                        print(f"[N05] {result}")
                
                # Create ToolMessage with the result
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        content=str(result)
                    )
                )
            
            # Add all tool messages to the conversation
            conversation.extend(tool_messages)

            resp = _invoke_llm(None, None, mapper_model, messages=conversation)
            _print_llm_metadata(resp)
            llm_logs.append(_llm_log_entry(resp, f"n05_iter{iteration}"))
            if resp.get("reasoning"):
                new_reasonings.append(AIMessage(content=f"Node_n05_iter{iteration}. {resp['reasoning']}"))
            if resp.get("error"):
                new_messages.append(AIMessage(content=f"Node_n05_iter{iteration}. ERROR: {resp['error']}"))
                raise RuntimeError(resp["error"])

            # Parse the new response
            iter_messages, iter_reasonings, iter_json = _parse_llm_resp(resp, node_id=f"n05_iter{iteration}")
            new_messages.extend(iter_messages)
            new_reasonings.extend(iter_reasonings)

            # Update mapping_json with latest iteration results
            if isinstance(iter_json, dict):
                mapping_json = iter_json

            # Build AIMessage for next iteration (content may be empty if tool_calls-only)
            ai_message = AIMessage(
                content=resp.get("output") or "",
                tool_calls=resp.get("tool_calls") or [],
            )
            conversation.append(ai_message)
            
            # Get tool calls for next iteration
            tool_calls = resp.get("tool_calls")
        
        #print(f"[N05] Tool execution complete after {iteration} iterations")

        if not isinstance(mapping_json, dict):
            new_messages.append(AIMessage(content="Node_n05. ERROR: Mapper returned non-dict payload"))
            mapping_json = {}

        mapped_relations = mapping_json.get("mapped_relations", [])
        
        # Convert to MappedRelation format
        mapped: List[Dict[str, Any]] = []
        for mr in mapped_relations:
            original_idx = mr.get("original_relation_index", 0)
            original_rel = cands[original_idx] if original_idx < len(cands) else {}
            
            specimen_payload = mr.get("specimen") or {}
            if not isinstance(specimen_payload, dict):
                specimen_payload = {}
            specimen_payload.setdefault("original", mr.get("specimen_original") or original_rel.get("specimen"))
            specimen_payload.setdefault("mapped_name", mr.get("specimen_mapped_name"))
            specimen_payload.setdefault("mapped_id", mr.get("specimen_mapped_id"))
            specimen_payload.setdefault("category", mr.get("specimen_category") or (specimen_payload.get("category") if isinstance(specimen_payload, dict) else None))
            specimen_payload.setdefault("ontology", mr.get("specimen_ontology") or (specimen_payload.get("ontology") if isinstance(specimen_payload, dict) else None))
            mapped_rel = {
                "glycan_name": mr.get("glycan_name") or original_rel.get("glycan"),
                "glycan_mapped_name": mr.get("glycan_mapped_name"),
                "glycan_id": mr.get("glycan_id"),
                "glycan_metadata": mr.get("glycan_metadata") or original_rel.get("glycan_metadata"),
                "disease_name": mr.get("disease_name") or original_rel.get("disease"),
                "disease_mapped_name": mr.get("disease_mapped_name"),
                "disease_id": mr.get("disease_id"),
                "disease_annotation": mr.get("disease_annotation") or original_rel.get("disease_annotation"),
                "treatment_annotation": mr.get("treatment_annotation") or original_rel.get("treatment_annotation"),
                "specimen": specimen_payload,
                "species_name": mr.get("species_name") or original_rel.get("species"),
                "species_mapped_name": mr.get("species_mapped_name"),
                "species_id": mr.get("species_id"),
                "protein_name": mr.get("protein_name") or original_rel.get("protein_name"),
                "protein_mapped_name": mr.get("protein_mapped_name"),
                "protein_id": mr.get("protein_id"),
                "biomarker_type": mr.get("biomarker_type") or original_rel.get("biomarker_type"),
                "direction": mr.get("direction") or original_rel.get("change") or original_rel.get("direction"),
                "evidence_locators": mr.get("evidence_locators") or original_rel.get("evidence_locators", []),
                "metrics": mr.get("metrics") or original_rel.get("metrics", []),
                "method_names": mr.get("method_names") or original_rel.get("method_names", []),
                "notes": mr.get("notes"),
            }

            glycan_surface = mapped_rel.get("glycan_name") or original_rel.get("glycan")
            glycan_mapped = mapped_rel.get("glycan_mapped_name")
            metadata = _resolve_glycan_entity_metadata(glycan_surface, glycan_mapped)
            if metadata:
                mapped_rel["glycan_entity_metadata"] = metadata
            mapped.append(mapped_rel)
        
        # Return only the mapped relations, not the message history
        # This prevents token bloat in subsequent nodes
        return _build_node_output(
            {"mapped": {"relations": mapped}}, new_messages, new_reasonings, llm_logs,
        )
    
    except json.JSONDecodeError as e:
        print(f"[N05] JSON parsing error: {e}")
        new_messages.append(AIMessage(content=f"Node_n05. ERROR: JSON parse failure {e}"))
        mapped = _fallback_relations(f"Mapping failed: JSON parse error - {str(e)}")
        return _build_node_output(
            {"mapped": {"relations": mapped}}, new_messages, new_reasonings, llm_logs,
        )
    
    except Exception as e:
        print(f"[N05] Mapper error: {e}")
        new_messages.append(AIMessage(content=f"Node_n05. ERROR: {e}"))
        mapped = _fallback_relations(f"Mapping failed: {str(e)}")
        return _build_node_output(
            {"mapped": {"relations": mapped}}, new_messages, new_reasonings, llm_logs,
        )

# N06 — Validate & Refine
def node_n06_validate_refine(state: AgentState) -> AgentState:
    """
    Validate mapped relations against evidence and refine glycan names.
    
    1. LLM validates evidence alignment
    2. Normalizes glycan names
    3. Splits compound relations into separate biomarkers when appropriate
    4. Deduplicates final relation list
    """
    _print_msg("INFO", "Running N06: Evidence Quality Check And Refinement")
    
    doc = state.get("doc", {})
    corpus_indexed = state.get("corpus_indexed", {})
    mapped = state.get("mapped", {}).get("relations", []) or []
    locator_by_index = _build_locator_index(corpus_indexed)

    new_messages: List[BaseMessage] = []
    new_reasonings: List[BaseMessage] = []
    llm_logs: List[Dict[str, Any]] = []

    if not mapped:
        return {
            "validated": {"relations": []},
            "cleaned": {"relations": []},
            "violations": [],
            "loop_step": 0,
            "schema_status": "valid",
        }

    refined_relations: List[Dict[str, Any]] = []
    new_violations: List[Dict[str, Any]] = []

    for idx, rel in enumerate(mapped):
        # Resolve evidence locators → sentence strings
        evidence_data = _resolve_evidence(
            rel.get("evidence_locators", []),
            corpus_indexed, locator_by_index, relation_idx=idx,
        )
        evidence_sentences = [ev["sentence"] for ev in evidence_data]

        # Prepare context for LLM validation
        validation_context = {
            "relation_index": idx,
            **_pick(rel, _RELATION_KEYS),
            "glycan_id": rel.get("glycan_id") or "unmapped",
            "disease_id": rel.get("disease_id") or "unmapped",
            "specimen": rel.get("specimen") or {},
            "metrics": rel.get("metrics", []),
            "method_names": rel.get("method_names", []),
            "evidence_sentences": evidence_sentences,
            "title": doc.get("title", ""),
            "abstract": (doc.get("abstract") or "")[:4000],
        }
        
        sm = SystemMessage(content=VALIDATION_SYS_PROMPT)
        hm = HumanMessage(content=json.dumps({"relations": [validation_context]}, indent=2))
        
        resp = _invoke_llm(sm, hm, validator_model)
        _print_llm_metadata(resp)
        llm_logs.append(_llm_log_entry(resp, f"n06_rel_{idx}"))
        parse_messages, parse_reasonings, validation_json = _parse_llm_resp(resp, node_id=f"n06_rel_{idx}")
        new_messages.extend(parse_messages)
        new_reasonings.extend(parse_reasonings)

        try:
            if not isinstance(validation_json, dict):
                raise ValueError("Validator returned non-dict payload")

            validated_rels = validation_json.get("relations", [])
            
            if not validated_rels:
                # Fallback: keep original
                refined_relations.append(rel)
                continue
            
            val_rel = validated_rels[0]  # Should only be one since we sent one
            action = val_rel.get("action", "keep")
            
            if action == "reject":
                # Skip this relation, log as violation
                new_violations.append({
                    "relation_id": idx,
                    "code": "REJECTED_BY_VALIDATOR",
                    "reason": val_rel.get("reason", "Evidence does not support relation"),
                    "details": rel
                })
                continue
            
            elif action == "split":
                split_biomarkers = val_rel.get("split_biomarkers", [])
                if split_biomarkers:
                    for split_rel in split_biomarkers:
                        refined_rel = deepcopy(rel)
                        refined_rel.update(_pick_first(split_rel, rel, _RELATION_KEYS))
                        if split_rel.get("specimen") is not None:
                            refined_rel["specimen"] = split_rel["specimen"]
                        refined_relations.append(refined_rel)
                else:
                    refined_relations.append(rel)
            
            elif action == "fix":
                # Apply fixes (mainly glycan normalization)
                refined_rel = deepcopy(rel)
                glycan_normalized = val_rel.get("glycan_normalized")
                if glycan_normalized:
                    # Update glycan name - may need remapping if significantly changed
                    # For now, preserve glycan_id but note that it might need remapping
                    refined_rel["glycan_normalized"] = glycan_normalized
                    refined_rel["normalization_note"] = val_rel.get("reason", "")
                refined_relations.append(refined_rel)
            
            else:  # "keep"
                # Keep as is
                refined_relations.append(rel)
        
        except json.JSONDecodeError as e:
            print(f"[N06] JSON parse error for relation {idx}: {e}")
            new_messages.append(AIMessage(content=f"Node_n06_rel_{idx}. ERROR: JSON parse failure {e}"))
            refined_relations.append(rel)
        except Exception as e:
            print(f"[N06] Validation error for relation {idx}: {e}")
            new_messages.append(AIMessage(content=f"Node_n06_rel_{idx}. ERROR: {e}"))
            refined_relations.append(rel)
    
    # Deduplicate relations
    deduped = _dedup_relations(refined_relations)
    
    # Final constraint check
    for i, r in enumerate(deduped):
        if not r.get("evidence_locators"):
            new_violations.append({"relation_id": i, "code": "MISSING_EVIDENCE", "details": r})
    
    loop_step = int(state.get("loop_step", 0))
    schema_status = "valid"
    
    _print_msg("INFO", f"Validated {len(mapped)} / Refined {len(refined_relations)} / Final {len(deduped)}")
    
    return _build_node_output(
        {
            "validated": {"relations": refined_relations},
            "cleaned": {"relations": deduped},
            "violations": new_violations,
            "loop_step": loop_step + 1,
            "schema_status": schema_status,
        },
        new_messages, new_reasonings, llm_logs,
    )

# N07 — Evidence Scorer
def node_n07_evidence_scorer(state: AgentState) -> AgentState:
    """
    Score evidence quality for each relation using rule-based criteria.
    
    Scoring rules (start at 0, clip to [0,1]):
    - +0.30 if evidence in Results/Discussion (else +0.10 if only Abstract)
    - +0.20 if strong claim tokens (significantly, elevated, decreased, associated, OR/HR/AUC)
    - +0.15 if ≥2 independent sentences/sections support relation
    - +0.10 if quantitative metrics present (AUC/OR/HR/fold-change/p-value/CI)
    - +0.10 if sample_size_n ≥ 50
    - +0.05 if validated assay method (LC-MS/MS, lectin microarray, NMR)
    - -0.20 if negation/hedging detected
    
    Labels: strong ≥0.7, moderate 0.4-0.69, weak <0.4
    """
    _print_msg("INFO", "Running N07: Evidence Scorer")
    
    corpus_indexed = state.get("corpus_indexed", {})
    cleaned = state.get("cleaned", {}).get("relations", []) or []
    locator_by_index = _build_locator_index(corpus_indexed)

    if not cleaned:
        return {"evidence": [], "scored": {"relations": []}}
    
    # Strong claim tokens
    STRONG_CLAIM_TOKENS = [
        "significantly", "elevated", "increased", "decreased", "reduced",
        "associated with", "correlation", "predictive", "diagnostic",
        "odds ratio", "hazard ratio", " OR ", " HR ", " AUC", "area under"
    ]
    
    # Validated assay methods
    VALIDATED_METHODS = [
        "LC-MS", "LC/MS", "mass spectrometry", "MS/MS",
        "lectin microarray", "lectin array",
        "NMR", "nuclear magnetic resonance",
        "HPLC", "capillary electrophoresis"
    ]
    
    # Negation/hedging patterns
    NEGATION_HEDGING = [
        "no significant", "not significant", "no difference", "no association",
        "trend towards", "marginally", "borderline", "may be", "might be",
        "possibly", "potentially", "unclear", "uncertain"
    ]
    
    # Quantitative metric patterns
    METRIC_PATTERNS = [
        "AUC", "odds ratio", "hazard ratio", " OR ", " HR ",
        "fold change", "fold-change", "p-value", "p <", "p=",
        "confidence interval", " CI", "95% CI"
    ]
    
    scored_relations: List[Dict[str, Any]] = []
    all_evidence_records: List[Dict[str, Any]] = []
    
    for rel_idx, rel in enumerate(cleaned):
        evidence_data = _resolve_evidence(
            rel.get("evidence_locators", []), corpus_indexed, locator_by_index,
        )
        
        if not evidence_data:
            # No evidence found - assign minimum score
            scored_relations.append({
                "relation": rel,
                "evidence_sentences": [],
                "aggregate_score": 0.0,
                "label": "weak",
                "score_breakdown": {"reason": "No evidence sentences found"}
            })
            continue
        
        # Initialize score
        score = 0.0
        score_breakdown = {}
        
        # Extract all sentences and sections
        sentences = [ev["sentence"] for ev in evidence_data]
        sections = [ev["section"] for ev in evidence_data]
        all_text = " ".join(sentences).lower()
        
        # Rule 1: Section quality (+0.30 for Results/Discussion, +0.10 for Abstract)
        has_results_discussion = any(
            section.upper() in ["RESULTS", "DISCUSSION", "RESULTS AND DISCUSSION", 
                               "FINDINGS", "CONCLUSION", "CONCLUSIONS"]
            for section in sections
        )
        if has_results_discussion:
            score += 0.30
            score_breakdown["section_quality"] = "+0.30 (Results/Discussion)"
        else:
            score += 0.10
            score_breakdown["section_quality"] = "+0.10 (Abstract only)"
        
        # Rule 2: Strong claim tokens (+0.20)
        has_strong_claim = any(token.lower() in all_text for token in STRONG_CLAIM_TOKENS)
        if has_strong_claim:
            score += 0.20
            score_breakdown["strong_claims"] = "+0.20"
        
        # Rule 3: Multiple independent sentences/sections (+0.15)
        unique_sections = len(set(sections))
        if len(sentences) >= 2 or unique_sections >= 2:
            score += 0.15
            score_breakdown["multiple_evidence"] = f"+0.15 ({len(sentences)} sentences, {unique_sections} sections)"
        
        # Rule 4: Quantitative metrics (+0.10)
        has_metrics = any(pattern.lower() in all_text for pattern in METRIC_PATTERNS)
        metrics = rel.get("metrics", [])
        if has_metrics or metrics:
            score += 0.10
            score_breakdown["quantitative_metrics"] = "+0.10"
        
        # Rule 5: Sample size ≥50 (+0.10)
        # Check for sample size patterns
        sample_size = 0
        sample_patterns = [
            r'n\s*=\s*(\d+)',
            r'n=(\d+)',
            r'(\d+)\s+patients',
            r'(\d+)\s+subjects',
            r'(\d+)\s+samples',
            r'cohort\s+of\s+(\d+)'
        ]
        for pattern in sample_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                try:
                    sample_size = max(int(m) for m in matches)
                    break
                except:
                    pass
        
        if sample_size >= 50:
            score += 0.10
            score_breakdown["sample_size"] = f"+0.10 (n={sample_size})"
        
        # Rule 6: Validated assay method (+0.05)
        methods = rel.get("method_names", [])
        method_text = " ".join(methods).lower() if methods else ""
        has_validated_method = any(
            method.lower() in all_text or method.lower() in method_text
            for method in VALIDATED_METHODS
        )
        if has_validated_method:
            score += 0.05
            score_breakdown["validated_method"] = "+0.05"
        
        # Rule 7: Negation/hedging penalty (-0.20)
        has_negation = any(pattern.lower() in all_text for pattern in NEGATION_HEDGING)
        if has_negation:
            score -= 0.20
            score_breakdown["negation_hedging"] = "-0.20"
        
        # Clip score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Assign label
        if score >= 0.7:
            label = "strong"
        elif score >= 0.4:
            label = "moderate"
        else:
            label = "weak"
        
        # Create evidence records with full sentences
        evidence_records = []
        for ev in evidence_data:
            locator = ev.get("locator", {}) or {}
            sentence_index = locator.get("sentence_index", 0)
            evidence_records.append({
                "sentence": ev.get("sentence"),
                "section": ev.get("section"),
                "sentence_index": sentence_index,
                "locator": locator,
            })
            all_evidence_records.append({
                "sentence": ev.get("sentence"),
                "section": ev.get("section"),
                "sentence_index": sentence_index,
            })
        
        scored_relations.append({
            "relation": rel,
            "evidence_sentences": evidence_records,
            "aggregate_score": round(score, 3),
            "label": label,
            "score_breakdown": score_breakdown
        })
    
    _print_msg("INFO", f"[N07] Scored {len(scored_relations)} relations")
    
    return {
        "evidence": all_evidence_records,
        "scored": {"relations": scored_relations}
    }

# N08 — Exporter
def node_n08_exporter(state: AgentState) -> AgentState:
    """
    Export curation results to JSONL files with batching (100 curations per file).
    Files are organized in batches: *_batch_001.jsonl, *_batch_002.jsonl, etc.
    """
    _print_msg("INFO", "Running N08: Exporting Curations")

    # Determine batch number by inspecting existing files
    batch_files = sorted(curation_dir.glob("curations_batch_*.jsonl"))
    if batch_files:
        last_batch_file = batch_files[-1]
        try:
            current_batch = int(last_batch_file.stem.split("_")[-1])
        except ValueError:
            current_batch = 1
    else:
        current_batch = 1

    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    curation_batch_path = curation_dir / f"curations_batch_{current_batch:03d}.jsonl"
    articles_in_batch = _count_lines(curation_batch_path)

    if articles_in_batch >= 100:
        current_batch += 1
        curation_batch_path = curation_dir / f"curations_batch_{current_batch:03d}.jsonl"
        articles_in_batch = 0
    
    runlog_batch_path = curation_dir / f"runlog_batch_{current_batch:03d}.jsonl"
    violations_batch_path = curation_dir / f"violations_batch_{current_batch:03d}.jsonl"
    errors_batch_path = curation_dir / f"errors_batch_{current_batch:03d}.jsonl"
    reasonings_batch_path = curation_dir / f"reasonings_batch_{current_batch:03d}.jsonl"
    tool_calls_batch_path = curation_dir / f"tool_calls_batch_{current_batch:03d}.jsonl"
    metadata_batch_path = curation_dir / f"metadata_batch_{current_batch:03d}.jsonl"
    
    # Extract data from state
    doc = state.get("doc", {})
    ids = state.get("ids", {})
    screening = state.get("screening", {})
    flags = state.get("flags", {})
    entities = state.get("candidates", {}).get("entities", [])
    scored_relations = state.get("scored", {}).get("relations", [])
    violations = state.get("violations", [])
    provenance = state.get("provenance", {}) or {}
    state_messages = state.get("messages", []) or []
    state_reasonings = state.get("reasonings", []) or []
    llm_logs = state.get("llm_logs", []) or []

    batch_article_index = articles_in_batch

    entity_lookup: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        surface = (ent.get("glycan_structure_term") or "").strip().lower()
        if surface and surface not in entity_lookup:
            entity_lookup[surface] = ent

    def _export_entity_metadata(rel_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidate_meta = rel_obj.get("glycan_entity_metadata") or {}
        if candidate_meta:
            return _pick(candidate_meta, _ENTITY_META_KEYS)

        for candidate in [rel_obj.get("glycan_name"), rel_obj.get("glycan_mapped_name")]:
            key = (candidate or "").strip().lower()
            if key and key in entity_lookup:
                return _pick(entity_lookup[key], _ENTITY_META_KEYS)
        return None
    
    # Build comprehensive curation record
    curation_record = {
        "pid": ids.get("processing_id"),
        "pmid": ids.get("pmid"),
        "pmcid": ids.get("pmcid"),
        "title": doc.get("title"),
        ###"abstract": doc.get("abstract"),
        "screening": {
            "decision": screening.get("decision"),
            "reason": screening.get("reason"),
            "study_type": screening.get("study_type"),
            "confidence": screening.get("confidence")
        },
        "flags": flags,
        "provenance": {
            "ingest_hash": provenance.get("ingest_hash"),
            "article_index": provenance.get("article_index"),
            "batch_number": current_batch,
            "batch_offset": batch_article_index,
        },
        "entities": [_pick(e, _ENTITY_EXPORT_KEYS) for e in entities],
        "relations": []
    }
    
    # Add relations with full evidence sentences
    for scored_rel in scored_relations:
        rel = scored_rel.get("relation", {})
        evidence_sentences = scored_rel.get("evidence_sentences", [])
        glycan_entity_metadata = _export_entity_metadata(rel)
        
        relation_record = {
            **_pick(rel, _RELATION_KEYS),
            "glycan_entity_metadata": glycan_entity_metadata,
            "metrics": rel.get("metrics", []),
            "method_names": rel.get("method_names", []),
            "evidence_sentences": [
                _pick(ev, ["sentence", "section", "sentence_index"])
                for ev in evidence_sentences
            ],
            "evidence_score": scored_rel.get("aggregate_score"),
            "evidence_label": scored_rel.get("label"),
            "score_breakdown": scored_rel.get("score_breakdown", {}),
        }
        
        # Include normalization info if present
        if rel.get("glycan_normalized"):
            relation_record["glycan_normalized"] = rel.get("glycan_normalized")
            relation_record["normalization_note"] = rel.get("normalization_note")
        
        curation_record["relations"].append(relation_record)
    
    # Build runlog record
    runlog_record = {
        "processing_id": ids.get("processing_id"),
        "pmid": ids.get("pmid"),
        "pmcid": ids.get("pmcid"),
        "screening_decision": screening.get("decision"),
        "study_type": screening.get("study_type"),
        "text_mode": flags.get("text_mode", "abstract"),
        "counts": {
            "entities": len(entities),
            "relations_extracted": len(state.get("candidates", {}).get("relations", [])),
            "relations_mapped": len(state.get("mapped", {}).get("relations", [])),
            "relations_validated": len(state.get("validated", {}).get("relations", [])),
            "relations_final": len(scored_relations),
            "violations": len(violations)
        },
        "evidence_distribution": {
            "strong": sum(1 for sr in scored_relations if sr.get("label") == "strong"),
            "moderate": sum(1 for sr in scored_relations if sr.get("label") == "moderate"),
            "weak": sum(1 for sr in scored_relations if sr.get("label") == "weak")
        },
        "provenance": {
            "ingest_hash": provenance.get("ingest_hash"),
            "article_index": provenance.get("article_index"),
            "batch_number": current_batch,
            "batch_offset": batch_article_index,
        },
    }
    
    # Write curation record
    with open(curation_batch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(curation_record, ensure_ascii=False) + "\n")
    
    # Write runlog record
    with open(runlog_batch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(runlog_record, ensure_ascii=False) + "\n")
    
    # Write violations if any
    if violations:
        violation_record = {
            "pmid": ids.get("pmid"),
            "violations": violations
        }
        with open(violations_batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(violation_record, ensure_ascii=False) + "\n")

    def _serialize_message(msg: Any) -> Dict[str, Any]:
        try:
            role = getattr(msg, "type", getattr(msg, "role", msg.__class__.__name__))
        except Exception:
            role = msg.__class__.__name__
        
        content = None
        try:
            content = getattr(msg, "content", None)
        except Exception:
            content = None
            
        # Fallback to str if not directly accessible
        if content is None:
            try:
                content = str(msg)
            except Exception:
                content = "[unserializable message]"
        return {"role": role, "content": content}

    # Write state messages (errors) to errors batch file
    if state_messages:
        errors_record = {
            "processing_id": ids.get("processing_id"),
            "pmid": ids.get("pmid"),
            "error_messages": [_serialize_message(m) for m in state_messages].pop(0),  # removes the first system message
        }
        with open(errors_batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(errors_record, ensure_ascii=False) + "\n")

    # Write reasonings to reasonings batch file
    if state_reasonings:
        reasonings_record = {
            "processing_id": ids.get("processing_id"),
            "pmid": ids.get("pmid"),
            "reasonings": [_serialize_message(m) for m in state_reasonings],
        }
        with open(reasonings_batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(reasonings_record, ensure_ascii=False) + "\n")

    # Write tool calls to tool_calls batch file
    if llm_logs:
        tool_calls_record = {
            "processing_id": ids.get("processing_id"),
            "pmid": ids.get("pmid"),
            "tool_calls": [
                {
                    "node": log.get("node"),
                    "tool_calls": log.get("tool_calls"),
                }
                for log in llm_logs
                if log.get("tool_calls") is not None and log.get("tool_calls") != [] # little sloppy here
            ],
        }
        if tool_calls_record["tool_calls"]:
            with open(tool_calls_batch_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tool_calls_record, ensure_ascii=False) + "\n")

    # Write metadata to metadata batch file
    if llm_logs:
        metadata_record = {
            "processing_id": ids.get("processing_id"),
            "pmid": ids.get("pmid"),
            "metadata": [
                {
                    "node": log.get("node"),
                    "metadata": log.get("metadata"),
                }
                for log in llm_logs
                if log.get("metadata") is not None
            ],
        }
        if metadata_record["metadata"]:
            with open(metadata_batch_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")
    
    articles_in_batch += 1
    _print_msg("INFO", f"[N08] Exported to batch {current_batch:03d} (article {articles_in_batch}/100)")
    
    return {
        "outputs": {
            "curation_batch_path": str(curation_batch_path),
            "runlog_batch_path": str(runlog_batch_path),
            "violations_batch_path": str(violations_batch_path),
            "errors_batch_path": str(errors_batch_path),
            "reasonings_batch_path": str(reasonings_batch_path),
            "tool_calls_batch_path": str(tool_calls_batch_path),
            "metadata_batch_path": str(metadata_batch_path),
            "batch_number": current_batch,
            "articles_in_batch": articles_in_batch,
            "batch_article_index": batch_article_index,
        }
    }

# -----------------------------------------------------------------------------
# Conditional Edges
# -----------------------------------------------------------------------------

def cond_after_screener(state: AgentState) -> str:
    scr = state.get("screening", {}) or {}
    decision = scr.get("decision", "IRRELEVANT")
    needs_fulltext = bool(scr.get("flags", {}).get("needs_fulltext"))
    if decision.startswith("IRRELEVANT"): # There is also "IRRELEVANT (overridden)"
        return "DISCARD"
    return "FULLTEXT" if needs_fulltext else "ABSTRACT_ONLY"

def cond_after_retrieval(state: AgentState) -> str:
    mode = state.get("flags", {}).get("text_mode", "abstract")
    return "SUCCESS" if mode in ("fulltext", "chunked") else "FALLBACK"

def cond_after_re(state: AgentState) -> str:
    rels = state.get("candidates", {}).get("relations", [])
    return "HAVE" if _relations_found(rels) else "NONE"

def cond_after_validate(state: AgentState) -> str:
    status = state.get("schema_status", "valid")
    loop_step = int(state.get("loop_step", 0))
    # one bounded retry if invalid
    if status != "valid" and loop_step <= 1:
        return "RETRY"
    return "CONTINUE"

# -----------------------------------------------------------------------------
# Build Graph
# -----------------------------------------------------------------------------

graph = StateGraph(AgentState)

# Nodes
graph.add_node("N01_AbstractScreener", node_n01_screener)
graph.add_node("N02_FullTextRetrieval", node_n02_retrieval)
graph.add_node("N03_FullTextNER", node_n03_fulltext_ner)
graph.add_node("N03a_AbstractOnlyNER", node_n03a_abstract_ner)
graph.add_node("N04_FullTextRE", node_n04_fulltext_re)
graph.add_node("N04a_AbstractOnlyRE", node_n04a_abstract_re)
graph.add_node("N05_OntologyMapper", node_n05_mapper)
graph.add_node("N06_ValidateRefine", node_n06_validate_refine)
graph.add_node("N07_EvidenceScorer", node_n07_evidence_scorer)
graph.add_node("N08_Exporter", node_n08_exporter)

# Edges per plan
graph.add_edge(START, "N01_AbstractScreener")

graph.add_conditional_edges(
    "N01_AbstractScreener",
    cond_after_screener,
    {
        "DISCARD": "N08_Exporter",         # log then end
        "FULLTEXT": "N02_FullTextRetrieval",
        "ABSTRACT_ONLY": "N03a_AbstractOnlyNER",
    },
)

graph.add_conditional_edges(
    "N02_FullTextRetrieval",
    cond_after_retrieval,
    {
        "SUCCESS": "N03_FullTextNER",
        "FALLBACK": "N03a_AbstractOnlyNER",
    },
)

graph.add_edge("N03_FullTextNER", "N04_FullTextRE")
graph.add_edge("N03a_AbstractOnlyNER", "N04a_AbstractOnlyRE")

graph.add_conditional_edges(
    "N04_FullTextRE",
    cond_after_re,
    {
        "HAVE": "N05_OntologyMapper",
        "NONE": "N08_Exporter",  # end if no relations
    },
)

graph.add_conditional_edges(
    "N04a_AbstractOnlyRE",
    cond_after_re,
    {
        "HAVE": "N05_OntologyMapper",
        "NONE": "N08_Exporter",
    },
)

graph.add_edge("N05_OntologyMapper", "N06_ValidateRefine")

graph.add_conditional_edges(
    "N06_ValidateRefine",
    cond_after_validate,
    {
        "RETRY": "N06_ValidateRefine",
        "CONTINUE": "N07_EvidenceScorer",
    },
)

graph.add_edge("N07_EvidenceScorer", "N08_Exporter")
graph.add_edge("N08_Exporter", END)

# Compile
app = graph.compile()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

diagram_path = Path(__file__).parents[2] / "static" / "graph_diagram.png"
def save_graph_png(diagram_path):
    with open(diagram_path, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())

# -----------------------------------------------------------------------------
# MAIN (production harness with resume capability)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    #save_graph_png(diagram_path)

    def print_stream(stream):
        """Print LangGraph messages for monitoring."""
        for s in stream:
            try:
                message = s.get("messages", [])
                if message and len(message) > 0:
                    last_msg = message[-1]
                    if hasattr(last_msg, 'pretty_print'):
                        last_msg.pretty_print()
            except Exception:
                pass

    processed_count = 0
    error_count = 0

    # Count total articles in input file
    total_articles = 0
    with open(input_path, "r", encoding="utf-8") as inputf:
        total_articles = sum(1 for _ in inputf)

    print("Biomarker Curation Agent (pre-release version)")
    print(f"{'='*80}")
    print(f"Input path:  {input_path}")
    print(f"Output dir: {curation_dir}")
    print(f"Records in queue: {total_articles}")
    print("Note: To rerun from a specific record: delete lines in <output dir>/curated_list.jsonl")
    print(f"{'='*80}")

    try:
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.touch(exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as inputf:
            for current_line, line in enumerate(inputf, start=1):
                print(f"\n{'─'*80}")
                print(f"- Processing record idx={current_line}")

                try:
                    input_json = json.loads(line)
                except json.JSONDecodeError as e:
                    _print_msg("ERROR", f"[ERROR] Line {current_line}: Invalid JSON - {e}")
                    error_count += 1
                    continue

                try:
                    registration = line_hash_check(line, list_path, inputs=input_json)
                except Exception as e:
                    _print_msg("ERROR", f"[ERROR] Line {current_line}: Unable to register curated entry - {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue

                action = registration.get("action", "process")
                record = registration.get("record", {})
                ingest_hash = registration.get("ingest_hash", "")
                article_index = registration.get("record_index", current_line - 1)
                article_number = article_index + 1
                processing_id = record.get("processing_id", input_json.get("processing_id", f"line-{current_line}"))
                pmid = record.get("pmid", input_json.get("pmid", "unknown"))

                if action == "skip":
                    _print_msg("INFO", f"[SKIP] PMID {pmid} already processed with matching content (article #{article_number}).")
                    processed_count += 1
                    continue

                # Validate schema before running the agent
                check2 = validate_schema(input_json, INPUT_SCHEMA)
                if check2 != "valid":
                    _print_msg("ERROR", f"[ERROR] Line {current_line}: Invalid schema - {check2}")
                    update_curated_status(processing_id, pmid, list_path, "error", ingest_hash=ingest_hash)
                    error_count += 1
                    continue

                _print_msg("INFO", f"Processing {processing_id} PMID:{pmid} ({article_number}/{total_articles})")

                agent_state = build_agent_state_from_input_obj(
                    input_json,
                    ingest_hash=ingest_hash,
                    article_index=article_index,
                )

                try:
                    final_state = None
                    for s in app.stream(agent_state, stream_mode="updates"): # eval: "values", "updates", "debug"
                        #print(s)
                        pass
                        
                    _print_msg("INFO", f"Completed processing {processing_id} PMID:{pmid}")
                    processed_count += 1
                    update_curated_status(processing_id, pmid, list_path, "completed", ingest_hash=ingest_hash)
                    
                except Exception as e:
                    print(f"[ERROR] Pipeline failed for PMID {pmid}: {e}")
                    import traceback
                    traceback.print_exc()
                    update_curated_status(processing_id, pmid, list_path, "error", ingest_hash=ingest_hash)
                    error_count += 1

                if processed_count % 10 == 0:
                    print(f"\n{'='*80}")
                    print(f"PROGRESS: {processed_count} processed, {error_count} errors")
                    print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Manual stop requested.")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total processed: {processed_count}")
        print(f"Total errors: {error_count}")
        print(f"Output directory: {curation_dir}")
        print(f"Curated list path: {list_path}")
        print(f"{'='*80}\n")
