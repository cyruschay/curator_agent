# Single-state definition for the LangGraph pipeline

import operator
from dataclasses import field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AnyMessage, SystemMessage
from langgraph.graph import add_messages


# Enums
Decision = Literal["RELEVANT", "IRRELEVANT"]
StudyType = Literal["case-control", "cohort", "cross-sectional", "intervention", "review", "meta-analysis", "editorial", "other"]
Direction = Literal["increased", "decreased", "absent", "novel_presence", "associated"]
TextMode = Literal["abstract", "fulltext", "chunked"]
StrengthLabel = Literal["weak", "moderate", "strong"]

# Records
class IDs(TypedDict, total=False):
    processing_id: str
    pmid: Optional[str]
    pmcid: Optional[str]

class Doc(TypedDict, total=False):
    title: str
    abstract: str
    full_text: Optional[Dict[str, Any]]  # sections_json (structured JATS → JSON)
    metadata: Dict[str, Any]

class Flags(TypedDict, total=False):
    has_pmcid: bool
    can_use_fulltext: bool
    needs_fulltext: bool
    discard: bool
    text_mode: TextMode  # "abstract" | "fulltext" | "chunked"
    fulltext_license: Optional[str]

class Provenance(TypedDict, total=False):
    ingest_hash: str
    source_line_no: Optional[int]
    article_index: Optional[int]

class ScreeningFlags(TypedDict, total=False):
    needs_fulltext: bool

class Screening(TypedDict, total=False):
    decision: Decision
    reason: str
    study_type: StudyType
    flags: ScreeningFlags
    confidence: float  # deterministic feature score in [0,1]

class Section(TypedDict, total=False):
    id: str                 # e.g., "RESULTS.2"
    title: Optional[str]
    text: str               # concatenated sentences
    path: List[str]         # hierarchy, e.g., ["RESULTS", "Subsection"]

class Corpus(TypedDict, total=False):
    Title: Optional[str]
    Abstract: Optional[Dict[str, Any]]
    Introduction: Optional[Dict[str, Any]]
    Methods: Optional[Dict[str, Any]]
    Results: Optional[Dict[str, Any]]
    Discussion: Optional[Dict[str, Any]]
    SupplementaryInformation: Optional[Dict[str, Any]]
    License: Optional[str]

class SentenceIndexed(TypedDict, total=False):
    section_id: str         # matches Section.id
    sentence_index: int     # global <S:n> index across entire document
    char_start: int
    char_end: int
    global_index: int       # alias for sentence_index (kept for backwards compatibility)
    locator_key: str        # canonical locator key "<S:n>@SECTION"

# New ver. locator record for corpus_indexed map includes the sentence text itself
class Locator(TypedDict, total=False):
    section: str                 # SECTION label (e.g., "RESULTS")
    subheading: Optional[str]    # lowest-level title if present
    paragraph_idx: int
    sentence_idx: int
    char_start: int
    char_end: int
    sentence: str                # the actual sentence text
    global_index: int
    locator_key: str

class Span(TypedDict, total=False):
    section_id: str
    sentence_index: int
    start: int
    end: int

class Entity(TypedDict, total=False):
    glycan_structure_term: str                  # the matched glycan structure text
    non_structural_descriptor: Optional[str]    # e.g., 'large', 'WA-binding', 'Fc-specific'
    alignment: Literal["whole_structure", "substructure", "core"]
    aglycon: Literal["R", "Ser/Thr", "Asn", "Cer", "free_glycan", "other"]
    chemical_structure: Optional[str]           # IUPAC-like sequence if explicitly provided
    spans: List[Span]                           # locator references
    evidence_sentence: Optional[str]            # the actual sentence text where this glycan was first mentioned


class GlycanEntityMetadata(TypedDict, total=False):
    glycan_structure_term: Optional[str]
    non_structural_descriptor: Optional[str]
    alignment: Optional[Literal["whole_structure", "substructure", "core"]]
    aglycon: Optional[Literal["R", "Ser/Thr", "Asn", "Cer", "free_glycan", "other"]]

class Metric(TypedDict, total=False):
    name: str               # e.g., "AUC", "OR", "HR", "p"
    value: Optional[float]
    raw: Optional[str]      # preserve original token if parsing is partial

class RelationCandidate(TypedDict, total=False):
    glycan: str                              # surface or provisional norm_id
    glycan_metadata: Optional[str]           # ratio info, lectin binding, descriptors (big, highly, etc.)
    biomarker_type: Optional[str]            # diagnostic, prognostic, predictive, monitoring, etc.
    change: Direction                        # "increased"/"decreased"/...
    disease: str                             # surface or provisional norm_id
    disease_annotation: Optional[str]        # disease stage, grade, subtype/severity context
    treatment_annotation: Optional[str]      # treatment/drug/surgical context
    specimen: Optional[str]
    species: Optional[str]
    protein_name: Optional[str]              # HGNC notation preferred, else common name
    cazy_enzyme: Optional[str]               # CAZy enzyme (HGNC > common name)
    evidence_sentence_indexes: List[int]     # sentence numbers from corpus_indexed keys
    evidence_locators: List[SentenceIndexed] # populated post-processing
    metrics: List[Metric]
    negated_or_hedged: bool
    method_names: List[str]

class Candidates(TypedDict, total=False):
    entities: List[Entity]
    relations: List[RelationCandidate]
    red_flags: List[str]  # any heuristic concerns

class SpecimenMapping(TypedDict, total=False):
    category: Optional[Literal["tissue", "cell_line", "fluid", "other"]]
    original: Optional[str]
    mapped_name: Optional[str]
    mapped_id: Optional[str]
    ontology: Optional[str]


class MappedRelation(TypedDict, total=False):
    glycan_name: Optional[str]                   # original surface form
    glycan_mapped_name: Optional[str]            # preferred name from ontology
    glycan_id: Optional[str]                     # GSD Term UUID or placeholder
    glycan_metadata: Optional[str]               # supplementary descriptors from extraction stage
    glycan_entity_metadata: Optional[GlycanEntityMetadata]  # carry entity metadata (descriptor/alignment/aglycon)
    disease_name: Optional[str]                  # original disease text
    disease_mapped_name: Optional[str]           # preferred disease label
    disease_id: Optional[str]                    # DOID
    disease_annotation: Optional[str]            # disease stage, grade, subtype/severity context
    treatment_annotation: Optional[str]          # treatment/drug/surgical context
    specimen: Optional[SpecimenMapping]          # consolidated specimen mapping info
    species_name: Optional[str]                  # original species text
    species_mapped_name: Optional[str]           # canonical taxonomy label
    species_id: Optional[str]                    # NCBI TaxID if available
    protein_name: Optional[str]                  # original protein text
    protein_mapped_name: Optional[str]           # canonical protein label
    protein_id: Optional[str]                    # UniProt accession if protein specified
    biomarker_type: Optional[str]                # diagnostic, prognostic, predictive, monitoring, etc.
    direction: Direction
    evidence_locators: List[SentenceIndexed]
    metrics: List[Metric]
    method_names: List[str]
    notes: Optional[str]                         # mapping conflict notes

class ValidatedRelation(TypedDict, total=False):
    relation: MappedRelation
    valid: bool
    issues: List[str]

class CleanedRelation(TypedDict, total=False):
    relation: MappedRelation    # de-duplicated & normalized
    cluster_id: str             # intra-article dedup group

class EvidenceRecord(TypedDict, total=False):
    locator: SentenceIndexed
    sentence_index: int
    sentence_preview: Optional[str]  # optional (OA or short quote policy)
    score: float                     # 0–1 numeric
    label: StrengthLabel             # weak/moderate/strong

class LLMLog(TypedDict, total=False):
    node: str
    tool_calls: Optional[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]]

class ScoredRelation(TypedDict, total=False):
    relation: MappedRelation
    evidence: List[EvidenceRecord]
    aggregate_score: float
    label: StrengthLabel

class Outputs(TypedDict, total=False):
    curations_path: Optional[str]   # JSONL output path
    approvals_path: Optional[str]
    runlog_path: Optional[str]
    violations_path: Optional[str]


# State collections
class Tags(TypedDict, total=False):
    lexicon_hits: Optional[Dict[str, int]]  # feature counts, optional

class ChunkIndex(TypedDict, total=False):
    # opaque chunking index only if text is extremely long
    spec: Dict[str, Any]

class Approvals(TypedDict, total=False):
    queue: List[Dict[str, Any]]  # new glycan terms, etc.


# AgentState
# - For lists: use operator.add (append)
# - For mappings: use operator.or_ (shallow merge)
# - For messages: use add_messages
# - For scalars: last-writer-wins (no reducer annotation)
class AgentState(TypedDict, total=False):
    # IDs & doc
    ids: Annotated[IDs, operator.or_]
    doc: Annotated[Doc, operator.or_]

    # Flags & provenance
    flags: Annotated[Flags, operator.or_]
    provenance: Annotated[Provenance, operator.or_]

    # Lightweight tags (e.g., lexicon feature counts)
    tags: Annotated[Tags, operator.or_]

    # Screening decision (abstract-based)
    screening: Annotated[Screening, operator.or_]

    # Corpus, locators, indices
    corpus_raw: Annotated[Corpus, operator.or_]
    corpus_indexed: Annotated[Dict[str, Locator], operator.or_]  # key: "<S:n>@SECTION"
    summarized_sections: Annotated[Dict[str, Any], operator.or_]
    chunk_index: Annotated[ChunkIndex, operator.or_]

    # Extraction candidates
    candidates: Annotated[Candidates, operator.or_]

    # Ontology-mapped / validated / cleaned
    mapped: Annotated[Dict[str, List[MappedRelation]], operator.or_]      # {"relations": [...]}
    validated: Annotated[Dict[str, List[ValidatedRelation]], operator.or_]# {"relations": [...]}
    cleaned: Annotated[Dict[str, List[CleanedRelation]], operator.or_]    # {"relations": [...]}

    # Evidence, scoring, violations, approvals
    evidence: Annotated[List[EvidenceRecord], operator.add]
    scored: Annotated[Dict[str, List[ScoredRelation]], operator.or_]      # {"relations": [...]}
    violations: Annotated[List[Dict[str, Any]], operator.add]
    approvals: Annotated[Approvals, operator.or_]

    # Outputs
    outputs: Annotated[Outputs, operator.or_]

    # LangGraph messages
    messages: Annotated[Sequence[AnyMessage], add_messages]
    reasonings: Annotated[Sequence[AnyMessage], add_messages]

    # LLM tool calls + metadata logs
    llm_logs: Annotated[List[LLMLog], operator.add]

    # Internal loop/retry counter for N06 bounded loop
    loop_step: Annotated[int, operator.add]


# Helpers: minimal factory defaults
# Use when instantiating a fresh state object
def empty_state() -> AgentState:
    """Construct a minimally valid, merge-safe empty AgentState."""
    return AgentState(
        ids=IDs(processing_id="", pmid=None, pmcid=None),
        doc=Doc(title="", abstract="", full_text=None, metadata={}),
        flags=Flags(
            has_pmcid=False,
            can_use_fulltext=False,
            needs_fulltext=False,
            discard=False,
            text_mode="abstract",
            fulltext_license=None,
        ),
        provenance=Provenance(ingest_hash="", source_line_no=None, article_index=None),
        tags=Tags(lexicon_hits={}),
        screening=Screening(
            decision="IRRELEVANT",
            reason="",
            study_type="other",
            flags=ScreeningFlags(needs_fulltext=False),
            confidence=0.0,
        ),
        corpus_raw=Corpus(),
        corpus_indexed={},                         # id -> Locator
        summarized_sections={},
        chunk_index=ChunkIndex(spec={}),
        candidates=Candidates(entities=[], relations=[], red_flags=[]),
        mapped={"relations": []},
        validated={"relations": []},
        cleaned={"relations": []},
        evidence=[],
        scored={"relations": []},
        violations=[],
        approvals=Approvals(queue=[]),
        outputs=Outputs(
            curations_path=None,
            approvals_path=None,
            runlog_path=None,
            violations_path=None,
        ),
        messages=[SystemMessage(content="You are a biomedical text mining agent.")],
        reasonings=[],
        llm_logs=[],
        loop_step=0,
    )


# State builder (for use in preprocessing step)
def build_agent_state_from_input(
    processing_id: str,
    pmid: Optional[str],
    pmcid: Optional[str],
    title: str,
    abstract: str,
    keywords: Optional[List[str]] = None,
    chemical_names: Optional[List[str]] = None,
    chemical_rn: Optional[List[str]] = None,
    chemical_uid: Optional[List[str]] = None,
    mesh_names: Optional[List[str]] = None,
    mesh_uid: Optional[List[str]] = None,
    journal_abbr: Optional[str] = None,
    article_date: Optional[List[Dict[str, Any]]] = None,
    ingest_hash: str = "",
    article_index: Optional[int] = None,
) -> AgentState:
    """Preprocessing helper to seed the state."""
    
    s = empty_state()
    s["ids"]["processing_id"] = processing_id
    s["ids"]["pmid"] = pmid
    s["ids"]["pmcid"] = pmcid

    s["doc"]["title"] = title
    s["doc"]["abstract"] = abstract
    
    s["doc"]["metadata"]["keywords"] = keywords
    s["doc"]["metadata"]["chemical_names"] = chemical_names
    s["doc"]["metadata"]["chemical_rn"] = chemical_rn
    s["doc"]["metadata"]["chemical_uid"] = chemical_uid
    s["doc"]["metadata"]["mesh_names"] = mesh_names
    s["doc"]["metadata"]["mesh_uid"] = mesh_uid
    s["doc"]["metadata"]["journal_abbr"] = journal_abbr
    s["doc"]["metadata"]["article_date"] = article_date

    s["provenance"]["ingest_hash"] = ingest_hash
    s["provenance"]["article_index"] = article_index
    s["flags"]["has_pmcid"] = bool(pmcid)
    return s
