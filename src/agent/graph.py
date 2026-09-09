# graph.py — LangGraph pipeline

from typing import Any, Dict, List, Optional, Tuple, Sequence
from copy import deepcopy
from pathlib import Path
import yaml
import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from adapters.ollama import build_chat_ollama

load_dotenv()

# Load configuration from YAML
config_path = Path(__file__).parents[2] / "config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# STATE
from states import AgentState

# PROMPTS
from prompts import (
    SCREENING_SYS_PROMPT, NER_ASSIST_SYS_PROMPT,
    ROLE_ID_SYS_PROMPT, ONTOLOGY_ADJUDICATE_SYS_PROMPT,
    VALIDATION_SYS_PROMPT, SUMMARIZER_SYS_PROMPT, SECTION_FILTER_SYS_PROMPT,
    build_describe_prompt, build_structure_prompt,
)

# ROLES
from roles import canonical_role, ROLE_SPECS, ROLE_LIST


def _role_required_evidence(roles: List[str]) -> str:
    """Natural language summary of what each assigned role's evidence must prove + its required fields."""
    parts = []
    for role in roles or []:
        spec = ROLE_SPECS.get(canonical_role(role) or "")
        if not spec:
            continue
        req = ", ".join(k for k, _ in spec["required_fields"])
        parts.append(f"{spec['label']}: must prove — {spec['must_prove']} (required: {req})")
    return " | ".join(parts) if parts else "(no role assigned)"

# SCHEMAS
from schemas import (
    INPUT_SCHEMA, build_structure_schema, build_role_id_schema,
    build_ner_schema, build_screening_schema, build_validation_schema,
    build_section_filter_schema, build_adjudication_schema,
)

# ONTOLOGY (deterministic resolvers; no LLM tool-calling)
import ontology as onto

# UTILS (deterministic, non-LLM helpers)
from utils import (
    # Preprocessing
    line_hash_check,
    validate_schema,
    build_agent_state_from_input_obj,
    update_curated_status,

    # Shared helpers
    _print_msg,
    _relations_found,
    _dedup_relations,
    invoke_and_parse,
    build_index_lookup,
    build_sections_dict,
    resolve_evidence_locators,

    # N01 — Screener
    feature_scorer,

    # N02 — Retrieval & Corpus
    retrieve_pmc_fulltext,
    index_corpus,
    count_chars_per_section,

    # N03 — NER helpers
    assemble_summarized_fulltext_for_ner,

    # N04 — Relation helpers
    negation_hedge_detector,
    normalize_method_names,
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


# Models (built via adapter from config.yaml)
REASONING_MODEL = build_chat_ollama(config_data=config_data, model_key="reasoning_model", format=None)
MINI_MODEL = build_chat_ollama(config_data=config_data, model_key="mini_model", format=None)
NANO_MODEL = build_chat_ollama(config_data=config_data, model_key="nano_model", format=None)

# Safety limits (avoid runaway generations / context truncation)
NER_MAX_INPUT_CHARS = 1500000


# Grammar-constrain every fixed-structure call: gpt-oss:20b under repeat_penalty=1.4 mangles
# free-text JSON keys (dropping data), so we force the schema wherever the shape is fixed.
# re_model stays unbound because role-ID/structure bind their own per-role schemas per call, and
# the describe step is intentionally free-text prose.
screening_model = REASONING_MODEL.bind(format=build_screening_schema())
ner_model = REASONING_MODEL.bind(format=build_ner_schema())
re_model = REASONING_MODEL           # role-driven RE: role-ID, per-role describe, per-role structure
adjudicator_model = REASONING_MODEL.bind(format=build_adjudication_schema())  # N05, bounded single call
validator_model = REASONING_MODEL.bind(format=build_validation_schema())
mini_model = MINI_MODEL.bind(format="json")                        # summarizer: dynamic section keys
nano_model = NANO_MODEL.bind(format=build_section_filter_schema())

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
        new_messages = [AIMessage(content="Node_n01. Missing abstract text for screening.")]
        node_n01_output = {
            "screening": {
                "decision": "IRRELEVANT",
                "reason": "No abstract text provided.",
            },
            "messages": new_messages
        }
        return node_n01_output
    
    # Deterministic feature scoring
    features = feature_scorer(title, abstract, metadata)
    score = features.get("score", 0)

    # Invoke LLM
    sm = SystemMessage(content=SCREENING_SYS_PROMPT)
    hm = HumanMessage(content=f"Title:\n{title}\n\nAbstract:\n{abstract}")
    resp, new_messages, new_reasonings, output_json, log = invoke_and_parse(sm, hm, screening_model, node_id="n01")
    llm_logs = [log]

    # Extract output_json fields; override LLM's RELEVANT decision if feature score is -1
    if "decision" in output_json:
        decision = output_json["decision"] if score != -1 else "IRRELEVANT (overridden)"
    else:
        new_messages.append(AIMessage(content=f"Node_n01. ERROR: Missing field 'decision' in LLM output"))
        decision = "IRRELEVANT"
        
    reason = output_json.get("reasoning", "[ERROR]")
    study_type = output_json.get("study_type", "[ERROR]")
    needs_fulltext = output_json.get("needs_fulltext", False)
    
    # build node output: screening
    screening = {
        "decision": decision,
        "reason": reason,
        "study_type": study_type,
        "flags": {"needs_fulltext": bool(needs_fulltext)},
        "confidence": float(score),
    }
    
    # build output
    node_n01_output = {
        "screening": screening,
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
        "corpus_indexed": corpus_indexed,
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
    
    filt_resp, filt_messages, filt_reasonings, filt_json, filt_log = invoke_and_parse(filt_sm, filt_hm, nano_model, node_id="n03_section_filter")
    llm_logs: List[Dict[str, Any]] = [filt_log]

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
            
            mini_resp, sum_messages, sum_reasonings, sum_json, sum_log = invoke_and_parse(sum_sm, sum_hm, mini_model, node_id="n03_section_summarizer")
            llm_logs.append(sum_log)

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
    ner_resp, ner_messages, ner_reasonings, ner_json, ner_log = invoke_and_parse(ner_sm, ner_hm, ner_model, node_id="n03_ner")
    llm_logs.append(ner_log)
    
    # Patch: handle missing 'glycans' field and list only
    if isinstance(ner_json, list):
        ner_json = {"glycans": ner_json}
        ner_messages.append(AIMessage(content="WARNING: NER output was a list; wrapped into {'glycans': [...] }"))
    
    glycans_list = ner_json.get("glycans", "ERROR")
    if glycans_list == "ERROR":
        ner_messages.append(AIMessage(content="ERROR: Missing 'glycans' field in NER output"))
        glycans_list = []

    # Build index lookup to map sentence_index back to sentence text
    index_lookup = build_index_lookup(corpus_indexed)

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

    # Deduplicate by glycan_structure_term (case-insensitive; matches the N03a abstract path)
    uniq, seen = [], set()
    for e in entities:
        key = (e["glycan_structure_term"] or "").strip().lower()
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
    resp, ner_messages, ner_reasonings, ner_json, log = invoke_and_parse(sm, hm, ner_model, node_id="n03a_ner")
    llm_logs = [log]

    glycans_list = ner_json.get("glycans", "ERROR")
    if glycans_list == "ERROR":
        ner_messages.append(AIMessage(content="ERROR: Missing 'glycans' field in NER output"))
        glycans_list = []
    
    # Build index lookup to map sentence_index back to sentence text
    index_lookup = build_index_lookup(abstract_indexed)
    
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

# N04 — Relation Extraction (role-driven; shared by full-text and abstract paths)
#
# Three steps, each a SIMPLE single-purpose call (the operating constraint of gpt-oss:20b):
#   (1) role-ID   — which of the 7 BEST roles the article supports + the glycans for each
#   (2) describe  — per present role, a role-specific guided-question prose description
#   (3) structure — per present role, grammar-constrained JSON with that role's schema
# The extraction path (prompt + output schema) therefore changes with the inferred role.

def _assemble_re_text(title, corpus_indexed, section_titles, summarized_sections, entities, char_cap):
    """Build the sentence-tagged article text plus the <ENTITIES> block."""
    text = assemble_summarized_fulltext_for_ner(title, corpus_indexed, section_titles, summarized_sections)
    entity_list_str = json.dumps([
        {
            "glycan_structure_term": e.get("glycan_structure_term"),
            "alignment": e.get("alignment"),
            "aglycon": e.get("aglycon"),
            "chemical_structure": e.get("chemical_structure"),
        }
        for e in entities
    ], indent=2)
    text += f"\n\n<ENTITIES>\n{entity_list_str}\n</ENTITIES>\n"
    return text[:char_cap]


def _merge_role_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge relations that share glycan+disease+direction across role passes, so a glycan that
    plays several roles becomes one relation with a role list and role-keyed annotations."""
    merged: Dict[Tuple, Dict[str, Any]] = {}
    for rel in relations:
        key = (
            str(rel.get("glycan", "")).strip().lower(),
            str(rel.get("disease", "")).strip().lower(),
            str(rel.get("change", "")).strip().lower(),
            str(rel.get("specimen", "")).strip().lower(),
        )
        if key not in merged:
            merged[key] = rel
            continue
        base = merged[key]
        for role in rel.get("biomarker_role", []):
            if role not in base.setdefault("biomarker_role", []):
                base["biomarker_role"].append(role)
        base.setdefault("role_annotations", {}).update(rel.get("role_annotations", {}) or {})
        base["is_multicomponent"] = bool(base.get("is_multicomponent")) or bool(rel.get("is_multicomponent"))
        # union evidence sentence indexes
        idxs = list(base.get("evidence_sentence_indexes", []) or [])
        for i in rel.get("evidence_sentence_indexes", []) or []:
            if i not in idxs:
                idxs.append(i)
        base["evidence_sentence_indexes"] = idxs
    return list(merged.values())


def _postprocess_relations(extracted: List[Dict[str, Any]], corpus_indexed: Dict[str, Any]) -> None:
    """In place: resolve evidence locators, normalize methods/metrics, detect negation."""
    index_lookup = build_index_lookup(corpus_indexed)
    for rel in extracted:
        sentence_indexes = rel.get("evidence_sentence_indexes", []) or []
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
            locator_key, loc = lookup
            evidence_locators.append({
                "section_id": loc.get("section", ""),
                "sentence_index": s_idx_int,
                "global_index": s_idx_int,
                "char_start": loc.get("char_start", 0),
                "char_end": loc.get("char_end", 0),
                "locator_key": locator_key,
            })
            evidence_texts.append(loc.get("sentence", ""))
            normalized_indexes.append(s_idx_int)
        rel["evidence_locators"] = evidence_locators
        if normalized_indexes:
            rel["evidence_sentence_indexes"] = normalized_indexes
        rel["method_names"] = normalize_method_names(rel.get("method_names") or [])
        rel["metrics"] = [
            {"name": m.get("name", ""), "value": m.get("value"), "raw": m.get("raw")}
            for m in (rel.get("metrics", []) or [])
        ]
        if "negated_or_hedged" not in rel:
            ev_txt = " ".join(t for t in evidence_texts if t)
            hedges = negation_hedge_detector(ev_txt) if ev_txt else {}
            rel["negated_or_hedged"] = bool(hedges.get("negated") or hedges.get("hedged"))


def run_role_driven_re(state, re_text, corpus_indexed, entities, node_prefix):
    """Role-ID -> per-role describe -> per-role structure -> merge. Returns a node output dict."""
    messages: List[BaseMessage] = []
    reasonings: List[BaseMessage] = []
    llm_logs: List[Dict[str, Any]] = []

    entity_terms = [e.get("glycan_structure_term") for e in entities if e.get("glycan_structure_term")]
    cands = state.get("candidates", {}) or {}
    if not entity_terms:
        cands["relations"] = []
        return {"candidates": cands}

    # Step 1 — role identification (grammar-constrained)
    rid_model = re_model.bind(format=build_role_id_schema())
    rid_resp, m, r, rid_json, log = invoke_and_parse(
        SystemMessage(content=ROLE_ID_SYS_PROMPT), HumanMessage(content=re_text),
        rid_model, node_id=f"{node_prefix}_roleid",
    )
    messages += m; reasonings += r; llm_logs.append(log)

    # Normalized lookup so minor reformatting (case/whitespace) by the role-ID or structure model
    # does not drop a glycan that the NER step did extract; canonicalize back to the NER surface.
    def _ekey(s: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    entity_norm = {_ekey(t): t for t in entity_terms}

    roles_present: List[Tuple[str, List[str]]] = []
    for item in (rid_json.get("roles", []) if isinstance(rid_json, dict) else []):
        role = canonical_role(item.get("role"))
        glys = [entity_norm[_ekey(g)] for g in (item.get("glycans") or []) if _ekey(g) in entity_norm]
        if role and glys:
            roles_present.append((role, glys))
    multicomponent = bool(rid_json.get("multicomponent")) if isinstance(rid_json, dict) else False

    # Safe default: if triage found no role, treat entities as current-state (diagnostic) candidates.
    if not roles_present:
        roles_present = [("diagnostic", entity_terms)]
        messages.append(AIMessage(content=f"Node_{node_prefix}. NOTE: role-ID empty; defaulted to diagnostic."))

    all_relations: List[Dict[str, Any]] = []
    for role, glys in roles_present:
        # Step 2 — describe (free-text prose grounded in the article text; not JSON)
        d_resp, dm, dr, _dj, dlog = invoke_and_parse(
            SystemMessage(content=build_describe_prompt(role, glys)), HumanMessage(content=re_text),
            re_model, node_id=f"{node_prefix}_{role}_describe", expect_json=False,
        )
        messages += dm; reasonings += dr; llm_logs.append(dlog)
        # gpt-oss sometimes emits the whole description into the reasoning (analysis) channel and
        # leaves the final channel empty; the reasoning IS the verbal description, so fall back to it.
        describe_text = (d_resp.get("output") or "").strip() or (d_resp.get("reasoning") or "").strip()
        if not describe_text:
            continue

        # Step 3 — structure (grammar-constrained to this role's schema; description-only input)
        s_model = re_model.bind(format=build_structure_schema(role))
        struct_input = (
            f"<ENTITIES>\n{json.dumps(glys)}\n</ENTITIES>\n\n"
            f"<DESCRIPTION>\n{describe_text}\n</DESCRIPTION>"
        )
        s_resp, sm2, sr2, s_json, slog = invoke_and_parse(
            SystemMessage(content=build_structure_prompt(role)), HumanMessage(content=struct_input),
            s_model, node_id=f"{node_prefix}_{role}_structure",
        )
        messages += sm2; reasonings += sr2; llm_logs.append(slog)

        for rel in (s_json.get("relations", []) if isinstance(s_json, dict) else []) or []:
            canon = entity_norm.get(_ekey(rel.get("glycan")))
            if canon is None:
                continue  # do not admit glycans outside the NER entity list
            rel["glycan"] = canon  # canonicalize to the NER surface form
            rel["biomarker_role"] = [role]
            rel["role_annotations"] = {role: rel.get("role_annotations", {}) or {}}
            rel["is_multicomponent"] = bool(rel.get("is_multicomponent")) or multicomponent
            all_relations.append(rel)

    merged = _merge_role_relations(all_relations)
    _postprocess_relations(merged, corpus_indexed)

    cands["relations"] = merged
    _print_msg("INFO", f"[{node_prefix}] roles={[r for r,_ in roles_present]} relations={len(merged)}")
    out = {"candidates": cands}
    if messages: out["messages"] = messages
    if reasonings: out["reasonings"] = reasonings
    if llm_logs: out["llm_logs"] = llm_logs
    return out


# N04 — Full Text RE
def node_n04_fulltext_re(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N04: Relation Extraction (full text, role-driven)")
    corpus_indexed = state.get("corpus_indexed", {})
    section_titles = list(dict.fromkeys(v["section"] for v in corpus_indexed.values()))
    summarized_sections = state.get("summarized_sections", {})
    entities = state.get("candidates", {}).get("entities", [])
    title = state.get("doc", {}).get("title", "")
    re_text = _assemble_re_text(title, corpus_indexed, section_titles, summarized_sections, entities, 150000)
    return run_role_driven_re(state, re_text, corpus_indexed, entities, "n04")


# N04a — Abstract-only RE
def node_n04a_abstract_re(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N04a: Relation Extraction (abstract, role-driven)")
    doc = state.get("doc", {})
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    entities = state.get("candidates", {}).get("entities", [])
    abstract_indexed = state.get("corpus_indexed", {})
    if not abstract_indexed:
        abstract_indexed = index_corpus({"Title": title, "Abstract": abstract})
    section_titles = list(dict.fromkeys(v["section"] for v in abstract_indexed.values()))
    re_text = _assemble_re_text(title, abstract_indexed, section_titles, {}, entities, 12000)
    return run_role_driven_re(state, re_text, abstract_indexed, entities, "n04a")


def _tally_match_type(entity_type: str, res: Dict[str, Any], stats: Dict[str, Dict[str, int]]) -> None:
    """Record which lookup tier answered one resolve, keyed by entity type.

    Matched results carry an explicit match_type (alias/exact/api); unmatched ones
    only carry a status, so "candidates" is recorded as semantic and everything
    else as none. Makes the exact-match hit rate readable straight off the runlog.
    """
    if not isinstance(res, dict):
        return
    tier = res.get("match_type")
    if not tier:
        tier = "semantic" if res.get("status") == "candidates" else "none"
    bucket = stats.setdefault(entity_type, {})
    bucket[tier] = bucket.get(tier, 0) + 1


# N05 — Ontology Mapper (deterministic; no LLM tool-calling loop)
#
# Exact/alias lookups are accepted directly. Semantic-only candidates from all relations are
# collected into ONE bounded adjudication call, so the model never issues its own tool calls
# (the failure mode of the previous version). No fuzzy matching; a match is never forced.
def node_n05_mapper(state: AgentState) -> AgentState:
    _print_msg("INFO", "Running N05: Ontology Mapping (deterministic)")

    cands = (state.get("candidates", {}) or {}).get("relations", []) or []
    if not cands:
        return {"mapped": {"relations": []}}

    new_messages: List[BaseMessage] = []
    new_reasonings: List[BaseMessage] = []
    llm_logs: List[Dict[str, Any]] = []

    # ---- 1) Resolve every entity deterministically; queue semantic-only hits for adjudication.
    adjudicate_queries: List[Dict[str, Any]] = []
    query_registry: Dict[Tuple[str, str], int] = {}

    def _register(entity_type: str, res: Dict[str, Any]) -> int:
        key = (entity_type, res.get("query") or "")
        if key in query_registry:
            return query_registry[key]
        qid = len(adjudicate_queries)
        adjudicate_queries.append({
            "id": qid, "type": entity_type,
            "query": res.get("query"), "candidates": res.get("candidates", [])[:onto.SEMANTIC_TOP_K],
        })
        query_registry[key] = qid
        return qid

    per_rel: List[Dict[str, Any]] = []
    match_stats: Dict[str, Dict[str, int]] = {}
    for r in cands:
        species_res = onto.resolve_species(r.get("species"))  # defaults to human when unnamed
        rr = {
            "glycan": onto.resolve_glycan(r.get("glycan")),
            "disease": onto.resolve_disease(r.get("disease")),
            "specimen": onto.resolve_specimen(r.get("specimen")),
            "species": species_res,
            "protein": onto.resolve_protein(r.get("protein_name"), species_res.get("mapped_name")),
        }
        for etype, res in rr.items():
            _tally_match_type(etype, res, match_stats)
        for etype in ("glycan", "disease", "specimen"):
            if rr[etype].get("status") == "candidates":
                rr[etype]["_adj_id"] = _register(etype, rr[etype])
        per_rel.append(rr)

    # ---- 2) Single bounded adjudication call for all semantic candidates.
    decisions: Dict[int, Optional[str]] = {}
    if adjudicate_queries:
        resp, m, r2, adj_json, log = invoke_and_parse(
            SystemMessage(content=ONTOLOGY_ADJUDICATE_SYS_PROMPT),
            HumanMessage(content=json.dumps({"queries": adjudicate_queries}, ensure_ascii=False)),
            adjudicator_model, node_id="n05_adjudicate",
        )
        new_messages += m; new_reasonings += r2; llm_logs.append(log)
        if isinstance(adj_json, dict):
            for d in adj_json.get("decisions", []) or []:
                try:
                    decisions[int(d.get("id"))] = d.get("accepted_entry")
                except (TypeError, ValueError):
                    continue

    # ---- 3) Build mapped relations.
    def _finalize(res: Dict[str, Any], parse_kind: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        st = res.get("status")
        if st == "matched":
            return res.get("mapped_name"), res.get("mapped_id"), res.get("ontology")
        if st == "candidates":
            acc = decisions.get(res.get("_adj_id"))
            if acc:
                parsed = onto.parse_accepted_entry(parse_kind, acc)
                return parsed.get("mapped_name"), parsed.get("mapped_id"), parsed.get("ontology")
        return None, None, None

    mapped: List[Dict[str, Any]] = []
    for r, rr in zip(cands, per_rel):
        g_name, g_id, g_onto = _finalize(rr["glycan"], "glycan")
        d_name, d_id, d_onto = _finalize(rr["disease"], "disease")
        sp_name, sp_id, sp_onto = _finalize(rr["specimen"], "specimen")
        species_res, protein_res = rr["species"], rr["protein"]

        mapped.append({
            "glycan_name": onto.normalize_glycan_term(r.get("glycan")),
            "glycan_mapped_name": g_name,
            "glycan_id": g_id,
            "glycan_metadata": r.get("glycan_metadata"),
            "disease_name": r.get("disease"),
            "disease_mapped_name": d_name,
            "disease_id": d_id,
            "disease_annotation": r.get("disease_annotation"),
            "specimen": {
                "original": r.get("specimen"),
                "mapped_name": sp_name,
                "mapped_id": sp_id,
                "category": rr["specimen"].get("category"),
                "ontology": sp_onto,
            },
            "species_name": species_res.get("query"),
            "species_mapped_name": species_res.get("mapped_name"),
            "species_id": species_res.get("mapped_id"),
            "protein_name": protein_res.get("query"),
            "protein_mapped_name": protein_res.get("mapped_name"),
            "protein_id": protein_res.get("mapped_id"),
            "biomarker_role": r.get("biomarker_role", []),
            "role_annotations": r.get("role_annotations", {}),
            "is_multicomponent": bool(r.get("is_multicomponent")),
            "direction": r.get("change") or r.get("direction"),
            "evidence_locators": r.get("evidence_locators", []),
            "metrics": r.get("metrics", []),
            "method_names": r.get("method_names", []),
            "negated_or_hedged": r.get("negated_or_hedged", False),
            "notes": None,
        })

    node_output: Dict[str, Any] = {"mapped": {"relations": mapped}}
    if new_messages: node_output["messages"] = new_messages
    if new_reasonings: node_output["reasonings"] = new_reasonings
    if llm_logs: node_output["llm_logs"] = llm_logs
    if match_stats: node_output["ontology_match_stats"] = match_stats
    return node_output


# N06 — Validate & Refine
def _specimen_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build the mapped-specimen shape from a validator-supplied specimen string.

    n05 has already run, so there is no adjudicator left to arbitrate semantic
    candidates here; only deterministic (alias/exact) hits are taken and anything
    else is left unmapped for the downstream ontology remap rather than guessed.
    """
    text = (text or "").strip()
    if not text:
        return None
    res = onto.resolve_specimen(text)
    matched = res.get("status") == "matched"
    return {
        "original": text,
        "mapped_name": res.get("mapped_name") if matched else None,
        "mapped_id": res.get("mapped_id") if matched else None,
        "category": res.get("category"),
        "ontology": res.get("ontology") if matched else None,
    }


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

    locator_by_index: Dict[int, Dict[str, Any]] = {}
    for key, entry in corpus_indexed.items():
        global_idx = entry.get("global_index")
        if isinstance(global_idx, int):
            locator_by_index[int(global_idx)] = entry
    
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
    
    # Process each relation
    refined_relations: List[Dict[str, Any]] = []
    new_violations: List[Dict[str, Any]] = []
    
    for idx, rel in enumerate(mapped):
        # Fetch evidence sentences
        evidence_locators = rel.get("evidence_locators", [])
        evidence_sentences = []
        
        for loc in evidence_locators:
            sentence_text = ""
            locator_key = loc.get("locator_key")

            if locator_key and locator_key in corpus_indexed:
                entry = corpus_indexed[locator_key]
                sentence_text = entry.get("sentence", "")
                loc.setdefault("section_id", entry.get("section", ""))
            else:
                global_idx = loc.get("global_index", loc.get("sentence_index"))
                if isinstance(global_idx, int):
                    entry = locator_by_index.get(int(global_idx))
                    if entry:
                        sentence_text = entry.get("sentence", "")
                        loc.setdefault("section_id", entry.get("section", ""))
                        loc.setdefault("locator_key", entry.get("locator_key"))

            if sentence_text:
                evidence_sentences.append(sentence_text)
        
        # Prepare context for LLM
        specimen_info = rel.get("specimen") or {}
        validation_context = {
            "relation_index": idx,
            "glycan_name": rel.get("glycan_name"),
            "glycan_mapped_name": rel.get("glycan_mapped_name"),
            "glycan_id": rel.get("glycan_id") or "unmapped",
            "glycan_metadata": rel.get("glycan_metadata"),
            "disease_name": rel.get("disease_name"),
            "disease_mapped_name": rel.get("disease_mapped_name"),
            "disease_id": rel.get("disease_id") or "unmapped",
            "disease_annotation": rel.get("disease_annotation"),
            "specimen": specimen_info,
            "species_name": rel.get("species_name"),
            "species_mapped_name": rel.get("species_mapped_name"),
            "species_id": rel.get("species_id"),
            "protein_name": rel.get("protein_name"),
            "protein_mapped_name": rel.get("protein_mapped_name"),
            "protein_id": rel.get("protein_id"),
            "biomarker_role": rel.get("biomarker_role", []),
            "role_annotations": rel.get("role_annotations", {}),
            "role_required_evidence": _role_required_evidence(rel.get("biomarker_role", [])),
            "direction": rel.get("direction"),
            "metrics": rel.get("metrics", []),
            "method_names": rel.get("method_names", []),
            "evidence_sentences": evidence_sentences,
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", "")[:4000]  # First 4000 chars for context
        }
        
        sm = SystemMessage(content=VALIDATION_SYS_PROMPT)
        hm = HumanMessage(content=json.dumps({"relations": [validation_context]}, indent=2))
        
        resp, parse_messages, parse_reasonings, validation_json, val_log = invoke_and_parse(sm, hm, validator_model, node_id=f"n06_rel_{idx}")
        llm_logs.append(val_log)
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
                # Split into multiple biomarkers
                split_biomarkers = val_rel.get("split_biomarkers", [])
                if split_biomarkers:
                    for split_rel in split_biomarkers:
                        # Preserve original fields, update glycan
                        refined_rel = deepcopy(rel)
                        refined_rel["glycan_name"] = split_rel.get("glycan_name") or rel.get("glycan_name")
                        refined_rel["glycan_mapped_name"] = split_rel.get("glycan_mapped_name") or rel.get("glycan_mapped_name")
                        refined_rel["glycan_id"] = split_rel.get("glycan_id") or rel.get("glycan_id")
                        refined_rel["disease_name"] = split_rel.get("disease_name") or rel.get("disease_name")
                        refined_rel["disease_mapped_name"] = split_rel.get("disease_mapped_name") or rel.get("disease_mapped_name")
                        refined_rel["disease_id"] = split_rel.get("disease_id") or rel.get("disease_id")
                        refined_rel["disease_annotation"] = split_rel.get("disease_annotation") or rel.get("disease_annotation")
                        refined_rel["treatment_annotation"] = split_rel.get("treatment_annotation") or rel.get("treatment_annotation")
                        split_specimen = _specimen_from_text(split_rel.get("specimen"))
                        if split_specimen is not None:
                            refined_rel["specimen"] = split_specimen
                        refined_rel["species_name"] = split_rel.get("species_name") or rel.get("species_name")
                        refined_rel["species_mapped_name"] = split_rel.get("species_mapped_name") or rel.get("species_mapped_name")
                        refined_rel["species_id"] = split_rel.get("species_id") or rel.get("species_id")
                        refined_rel["protein_name"] = split_rel.get("protein_name") or rel.get("protein_name")
                        refined_rel["protein_mapped_name"] = split_rel.get("protein_mapped_name") or rel.get("protein_mapped_name")
                        refined_rel["protein_id"] = split_rel.get("protein_id") or rel.get("protein_id")
                        # Note: If glycan name changed, glycan_id might be null - will need remapping
                        refined_relations.append(refined_rel)
                else:
                    # Fallback: keep original
                    refined_relations.append(rel)
            
            elif action == "fix":
                # Apply fixes (glycan normalization, and a specimen the extractor missed)
                refined_rel = deepcopy(rel)
                fixed_specimen = _specimen_from_text(val_rel.get("specimen_fix"))
                if fixed_specimen is not None and not (rel.get("specimen") or {}).get("original"):
                    refined_rel["specimen"] = fixed_specimen
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
    
    node_output = {
        "validated": {"relations": refined_relations},  # Before dedup, with validation details
        "cleaned": {"relations": deduped},  # After dedup, ready for scoring
        "violations": new_violations,
        "loop_step": loop_step + 1,
        "schema_status": schema_status,
    }
    if new_messages:
        node_output["messages"] = new_messages
    if new_reasonings:
        node_output["reasonings"] = new_reasonings
    if llm_logs:
        node_output["llm_logs"] = llm_logs
    return node_output

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
    cleaned_relations = state.get("cleaned", {}).get("relations", [])
    corpus_indexed = state.get("corpus_indexed", {})
    violations = state.get("violations", [])

    # Reconstruct evidence sentence text from the relation's locators (was previously done in N07).
    _locator_by_index: Dict[int, Dict[str, Any]] = {}
    for _k, _entry in corpus_indexed.items():
        _gi = _entry.get("global_index")
        if isinstance(_gi, int):
            _locator_by_index[_gi] = _entry

    def _evidence_sentences_for(rel_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for loc in rel_obj.get("evidence_locators", []) or []:
            entry = None
            lk = loc.get("locator_key")
            if lk and lk in corpus_indexed:
                entry = corpus_indexed[lk]
            else:
                gi = loc.get("global_index", loc.get("sentence_index"))
                if isinstance(gi, int):
                    entry = _locator_by_index.get(gi)
            if entry:
                out.append({
                    "sentence": entry.get("sentence"),
                    "section": entry.get("section"),
                    "sentence_index": loc.get("sentence_index", entry.get("global_index")),
                })
        return out
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
            return {
                "glycan_structure_term": candidate_meta.get("glycan_structure_term"),
                "non_structural_descriptor": candidate_meta.get("non_structural_descriptor"),
                "alignment": candidate_meta.get("alignment"),
                "aglycon": candidate_meta.get("aglycon"),
            }

        for candidate in [rel_obj.get("glycan_name"), rel_obj.get("glycan_mapped_name")]:
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
        "entities": [
            {
                "glycan_structure_term": e.get("glycan_structure_term"),
                "non_structural_descriptor": e.get("non_structural_descriptor"),
                "alignment": e.get("alignment"),
                "aglycon": e.get("aglycon"),
                "chemical_structure": e.get("chemical_structure"),
                "evidence_sentence": e.get("evidence_sentence")
            }
            for e in entities
        ],
        "relations": []
    }
    
    # Add relations with full evidence sentences and role-specific annotations
    for rel in cleaned_relations:
        evidence_sentences = _evidence_sentences_for(rel)
        glycan_entity_metadata = _export_entity_metadata(rel)

        relation_record = {
            "glycan_name": rel.get("glycan_name"),
            "glycan_mapped_name": rel.get("glycan_mapped_name"),
            "glycan_id": rel.get("glycan_id"),
            "glycan_metadata": rel.get("glycan_metadata"),
            "glycan_entity_metadata": glycan_entity_metadata,
            "disease_name": rel.get("disease_name"),
            "disease_mapped_name": rel.get("disease_mapped_name"),
            "disease_id": rel.get("disease_id"),
            "disease_annotation": rel.get("disease_annotation"),
            "specimen": rel.get("specimen"),
            "species_name": rel.get("species_name"),
            "species_mapped_name": rel.get("species_mapped_name"),
            "species_id": rel.get("species_id"),
            "protein_name": rel.get("protein_name"),
            "protein_mapped_name": rel.get("protein_mapped_name"),
            "protein_id": rel.get("protein_id"),
            "biomarker_role": rel.get("biomarker_role", []),
            "role_annotations": rel.get("role_annotations", {}),
            "is_multicomponent": bool(rel.get("is_multicomponent")),
            "direction": rel.get("direction"),
            "metrics": rel.get("metrics", []),
            "method_names": rel.get("method_names", []),
            "negated_or_hedged": rel.get("negated_or_hedged", False),
            "evidence_sentences": evidence_sentences,
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
            "relations_final": len(cleaned_relations),
            "violations": len(violations)
        },
        "role_distribution": {
            role: sum(1 for rel in cleaned_relations if role in (rel.get("biomarker_role") or []))
            for role in ROLE_LIST
        },
        # Which ontology lookup tier answered each resolve, per entity type
        "ontology_match_types": state.get("ontology_match_stats", {}) or {},
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
            "error_messages": [_serialize_message(m) for m in state_messages][1:], # drop the leading system message, keep all real errors
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
        "CONTINUE": "N08_Exporter",
    },
)

graph.add_edge("N08_Exporter", END)

# Compile
app = graph.compile()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

DOC_DIR = Path(__file__).parents[2] / "doc"

def save_graph_png():
    # Diagram rendering needs network/graphviz; never let it crash the pipeline import.
    try:
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        diagram_path = DOC_DIR / "graph_diagram.png"
        diagram_path.write_bytes(app.get_graph().draw_mermaid_png())
        print(f"Graph diagram saved to {diagram_path}")
    except Exception as e:
        _print_msg("WARN", f"Could not render graph diagram: {e}")

save_graph_png()

# -----------------------------------------------------------------------------
# MAIN (production harness with resume capability)
# -----------------------------------------------------------------------------

if __name__ == "__main__":

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
