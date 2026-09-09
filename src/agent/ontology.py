# ontology.py — deterministic ontology resolution.
#
# Replaces the previous LLM tool-calling loop (which failed on long, multi-call sessions)
# with a deterministic resolver: exact/alias lookup first, then semantic (embedding) search
# with a wider top-k. A single bounded LLM adjudication call (in graph N05) decides equivalence
# for semantic-only hits — the model never issues its own tool calls.
#
# Rules honored:
#   - No fuzzy string matching on glycans or cell lines (exact + semantic only); string
#     similarity does not imply biochemical equivalence.
#   - Never force a match: an entity that has no equivalent stays unmapped (id = None).
#   - Type-aware routing: glycan->GSD, disease->DOID, species->NCBI taxonomy,
#     tissue/fluid/circulating-cell->UBERON/CL, cell line->Cellosaurus,
#     single-gene protein->UniProt, protein complex->Complex Portal.

from __future__ import annotations

import os
import re
import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

from adapters.chroma import build_ollama_embeddings
from aliases import DISEASE_ALIASES, SPECIES_ALIASES, SPECIMEN_ALIASES

load_dotenv()


def _entrez():
    """Lazily import and configure Bio.Entrez (only the taxonomy path needs it)."""
    from Bio import Entrez
    Entrez.email = os.getenv("NCBI_EMAIL", "glycan.curator@example.com")
    Entrez.api_key = os.getenv("NCBI_API_KEY", None)
    return Entrez

# Wider than the old k=3; short ontology entries mean recall matters more than precision here.
SEMANTIC_TOP_K = 8

# Embedding batch size for building vectorstores (the Ollama runner EOFs on very large batches).
_EMBED_BATCH = 256

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

WORKSPACE_DIR = Path(__file__).parents[2] / "data" / "workspace"

_RESOURCES = {
    "gsd": {
        "path": WORKSPACE_DIR / "gsd_terms.txt",
        "persist": WORKSPACE_DIR / "gsd_vectorstore",
        "collection": "glycan_structure_dictionary",
    },
    "doid": {
        "path": WORKSPACE_DIR / "disease_ontology_v2.txt",
        "persist": WORKSPACE_DIR / "doid_vectorstore",
        "collection": "disease_ontology",
    },
    "uberon": {
        "path": WORKSPACE_DIR / "uberon_terms.txt",
        "persist": WORKSPACE_DIR / "uberon_vectorstore",
        "collection": "uberon_ontology",
    },
}

_embeddings = None
_vectorstores: Dict[str, Chroma] = {}
_exact_dicts: Dict[str, Dict[str, str]] = {}


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = build_ollama_embeddings()
    return _embeddings


def _split_name(line: str) -> str:
    """Surface name of an ontology entry: text before the name boundary.

    Split at the EARLIEST of ' | ' (GSD field separator) or ' (' (DOID/UBERON id parenthesis).
    GSD definitions contain later '(' that must not be mistaken for the name boundary, so the
    earliest separator — not a fixed precedence — is what delimits the name.
    """
    idxs = [i for i in (line.find(" | "), line.find(" (")) if i != -1]
    return (line[:min(idxs)] if idxs else line).strip()


def _build_exact_dict(kind: str, lines: List[str]) -> Dict[str, str]:
    """name/synonym (lowercased) -> full entry line. Handles GSD, DOID, UBERON+CL."""
    d: Dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        name = _split_name(line).lower()
        if name:
            d.setdefault(name, line)
        # exact synonyms
        if kind == "gsd" and "Exact synonyms:" in line and "Exact synonyms: None" not in line:
            syns = line.split("Exact synonyms:")[1].split(" | ")[0]
        elif kind == "doid" and "exact synonyms:" in line:
            syns = line.split("exact synonyms:")[1].split("and related")[0]
        elif kind == "uberon" and "Synonyms:" in line:
            syns = line.split("Synonyms:")[1].split(".")[0]
        else:
            syns = ""
        for syn in syns.split(";"):
            s = syn.strip().lower()
            if s and s.lower() != "none":
                d.setdefault(s, line)
    return d


def _persisted_store_is_stale(kind: str, persist: Path, res: Dict[str, Any]) -> bool:
    """True if a persisted store predates the GSD: -> bGSL: glycan id switch.

    Its documents still embed retired ids, so the semantic tier would hand back
    entries that parse_gsd_entry can no longer extract an id from. Drop the store
    so the caller rebuilds it from the current data file.
    """
    if kind != "gsd":
        return False
    try:
        probe = Chroma(
            persist_directory=str(persist),
            collection_name=res["collection"],
            embedding_function=_get_embeddings(),
        )
        docs = (probe.get(limit=25, include=["documents"]) or {}).get("documents") or []
    except Exception:
        # Can't inspect it — leave it alone rather than discarding a usable store
        return False
    if not any(_RETIRED_GLYCAN_ID_PREFIX in (d or "") for d in docs):
        return False
    import shutil
    del probe
    shutil.rmtree(persist, ignore_errors=True)
    persist.mkdir(parents=True, exist_ok=True)
    return True


def _init(kind: str) -> None:
    if kind in _vectorstores:
        return
    res = _RESOURCES[kind]
    path = res["path"]
    if not path.exists():
        raise FileNotFoundError(f"Ontology data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    _exact_dicts[kind] = _build_exact_dict(kind, lines)

    persist = Path(res["persist"])
    persist.mkdir(parents=True, exist_ok=True)
    if any(persist.iterdir()) and not _persisted_store_is_stale(kind, persist, res):
        _vectorstores[kind] = Chroma(
            persist_directory=str(persist),
            collection_name=res["collection"],
            embedding_function=_get_embeddings(),
        )
    else:
        # Build in batches: the Ollama embedding runner EOFs if handed all ~16k texts at once.
        store = Chroma(
            persist_directory=str(persist),
            collection_name=res["collection"],
            embedding_function=_get_embeddings(),
        )
        docs = [Document(page_content=ln) for ln in lines]
        try:
            for i in range(0, len(docs), _EMBED_BATCH):
                store.add_documents(docs[i:i + _EMBED_BATCH])
        except Exception:
            # Remove the partial store so the next run rebuilds cleanly (an empty/partial
            # persist dir would otherwise be loaded as if complete).
            import shutil
            del store
            shutil.rmtree(persist, ignore_errors=True)
            raise
        _vectorstores[kind] = store


def _semantic(kind: str, query: str, top_k: int = SEMANTIC_TOP_K) -> List[Dict[str, Any]]:
    _init(kind)
    retriever = _vectorstores[kind].as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(query)
    return [{"rank": i + 1, "entry": d.page_content} for i, d in enumerate(docs)]


def _local_lookup(kind: str, term: str, aliases: Dict[str, str], top_k: int = SEMANTIC_TOP_K) -> Dict[str, Any]:
    """Exact/alias -> semantic for a file-backed ontology (GSD/DOID/UBERON)."""
    _init(kind)
    q = (term or "").strip()
    key = q.lower()
    if key in aliases:
        return {"query": q, "match_type": "alias", "entry": aliases[key], "candidates": []}
    if key in _exact_dicts.get(kind, {}):
        return {"query": q, "match_type": "exact", "entry": _exact_dicts[kind][key], "candidates": []}
    try:
        cands = _semantic(kind, q, top_k)
    except Exception as e:  # embedding/vectorstore failure -> unmapped, not fatal
        return {"query": q, "match_type": "error", "entry": None, "candidates": [], "error": str(e)}
    return {"query": q, "match_type": "semantic" if cands else "none", "entry": None, "candidates": cands}


# ---------------------------------------------------------------------------
# Entry parsers  (entry text -> (preferred_name, id))
# ---------------------------------------------------------------------------

_RE_DOID = re.compile(r"\(DOID:([0-9]+)\)")
_RE_OBO = re.compile(r"\((UBERON|CL):([0-9]+)\)")
# Glycan term UUIDs use the bGSL: prefix; "GSD:" is retired.
_RETIRED_GLYCAN_ID_PREFIX = "GSD:"
_RE_GSD = re.compile(r"Term UUID:\s*(bGSL:[0-9a-fA-F-]+)")


def parse_gsd_entry(entry: str) -> Dict[str, Optional[str]]:
    m = _RE_GSD.search(entry or "")
    return {"mapped_name": _split_name(entry), "mapped_id": m.group(1) if m else None, "ontology": "GSD"}


def parse_doid_entry(entry: str) -> Dict[str, Optional[str]]:
    m = _RE_DOID.search(entry or "")
    return {"mapped_name": _split_name(entry), "mapped_id": f"DOID:{m.group(1)}" if m else None, "ontology": "DOID"}


def parse_obo_entry(entry: str) -> Dict[str, Optional[str]]:
    m = _RE_OBO.search(entry or "")
    onto = m.group(1) if m else "UBERON"
    return {
        "mapped_name": _split_name(entry),
        "mapped_id": f"{m.group(1)}:{m.group(2)}" if m else None,
        "ontology": onto,
    }


# ---------------------------------------------------------------------------
# Glycan term hygiene
# ---------------------------------------------------------------------------

# Bare descriptor adjectives that must become "<adj> glycan" (galactosylation-style process
# nouns are already acceptable and are left untouched).
_BARE_GLYCAN_ADJECTIVES = {
    "sialylated", "asialylated", "desialylated", "fucosylated", "afucosylated",
    "galactosylated", "agalactosylated", "mannosylated", "glucosylated",
    "branched", "bisected", "truncated", "sulfated", "phosphorylated",
    "sialyl", "fucosyl", "high-mannose", "high mannose",
}


def normalize_glycan_term(term: Optional[str]) -> Optional[str]:
    """Deterministic guard mirroring the NER rule: a bare descriptor adjective becomes
    '<adjective> glycan' so it is a noun phrase, not a dangling adjective."""
    if not term:
        return term
    t = term.strip()
    if t.lower() in _BARE_GLYCAN_ADJECTIVES:
        return f"{t} glycan"
    return t


_CELL_LINE_HINT = re.compile(r"cell\s*line", re.IGNORECASE)
# e.g. HEK293, MCF-7, A549, SW480, HCT116, PC-3 (letters + digits, uppercase-initial)
_CELL_LINE_TOKEN = re.compile(r"^[A-Z][A-Za-z]{0,6}-?\d{1,4}[A-Za-z0-9-]*$")


def looks_like_cell_line(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.strip()
    if _CELL_LINE_HINT.search(n):
        return True
    return bool(_CELL_LINE_TOKEN.match(n)) and not n[0].isdigit()


# ---------------------------------------------------------------------------
# API-backed searches (Cellosaurus / UniProt / Complex Portal / NCBI taxonomy)
# ---------------------------------------------------------------------------

def search_cellosaurus(name: str) -> Dict[str, Any]:
    q = (name or "").strip()
    try:
        url = (
            f"https://api.cellosaurus.org/search/cell-line?q=id%3A{q}"
            "&start=0&rows=5&format=json&fld=id&fld=sy&fld=ac"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        cell_lines = r.json().get("Cellosaurus", {}).get("cell-line-list", [])
        for cl in cell_lines:
            acc = next((a.get("value") for a in cl.get("accession-list", []) if a.get("type") == "primary"), None)
            ident = next((n.get("value") for n in cl.get("name-list", []) if n.get("type") == "identifier"), None)
            if acc:
                return {"query": q, "match_type": "api", "mapped_name": ident or q,
                        "mapped_id": acc, "ontology": "Cellosaurus"}
        return {"query": q, "match_type": "none", "mapped_name": None, "mapped_id": None, "ontology": "Cellosaurus"}
    except Exception as e:
        return {"query": q, "match_type": "error", "mapped_name": None, "mapped_id": None,
                "ontology": "Cellosaurus", "error": str(e)}


def search_uniprot(name: str, species_name: Optional[str] = None) -> Dict[str, Any]:
    q = (name or "").strip()
    query = f"protein_name:{q} AND organism_name:{species_name}" if species_name else f"protein_name:{q}"
    try:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": query, "fields": "accession,protein_name,gene_primary", "format": "tsv"},
            timeout=10,
        )
        r.raise_for_status()
        rows = r.text.splitlines()
        if len(rows) >= 2:
            acc, pname = (rows[1].split("\t") + ["", ""])[:2]
            return {"query": q, "match_type": "api", "mapped_name": pname or q,
                    "mapped_id": acc or None, "ontology": "UniProt"}
        return {"query": q, "match_type": "none", "mapped_name": None, "mapped_id": None, "ontology": "UniProt"}
    except Exception as e:
        return {"query": q, "match_type": "error", "mapped_name": None, "mapped_id": None,
                "ontology": "UniProt", "error": str(e)}


def search_complex(name: str, species_name: Optional[str] = None) -> Dict[str, Any]:
    """EBI Complex Portal — for named complexes/receptors (IgG, CD3, ...)."""
    q = (name or "").strip()
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/intact/complex-ws/search/{q}",
            params={"format": "json"}, timeout=10,
        )
        r.raise_for_status()
        elements = r.json().get("elements", []) or []
        if species_name:
            sl = species_name.lower()
            # Do NOT fall back to other organisms when the requested species has no entry —
            # returning a different organism's complex would be a silent wrong-species match.
            elements = [e for e in elements if sl in str(e.get("organismName", "")).lower()]
        if elements:
            best = elements[0]
            return {"query": q, "match_type": "api", "mapped_name": best.get("complexName") or q,
                    "mapped_id": best.get("complexAC"), "ontology": "ComplexPortal"}
        return {"query": q, "match_type": "none", "mapped_name": None, "mapped_id": None, "ontology": "ComplexPortal"}
    except Exception as e:
        return {"query": q, "match_type": "error", "mapped_name": None, "mapped_id": None,
                "ontology": "ComplexPortal", "error": str(e)}


def search_taxonomy(name: str) -> Dict[str, Any]:
    q = (name or "").strip()
    key = q.lower()
    if key in SPECIES_ALIASES:
        row = SPECIES_ALIASES[key].split("\t")
        return {"query": q, "match_type": "alias", "mapped_name": row[1] if len(row) > 1 else q,
                "mapped_id": row[0], "ontology": "NCBITaxon"}
    try:
        Entrez = _entrez()
        h = Entrez.esearch(db="taxonomy", term=q, retmode="xml")
        res = Entrez.read(h); h.close()
        ids = res.get("IdList", [])
        if not ids:
            return {"query": q, "match_type": "none", "mapped_name": None, "mapped_id": None, "ontology": "NCBITaxon"}
        h = Entrez.efetch(db="taxonomy", id=ids[0], retmode="xml")
        rec = Entrez.read(h); h.close()
        if rec:
            return {"query": q, "match_type": "api", "mapped_name": rec[0].get("ScientificName"),
                    "mapped_id": str(rec[0].get("TaxId")), "ontology": "NCBITaxon"}
        return {"query": q, "match_type": "none", "mapped_name": None, "mapped_id": None, "ontology": "NCBITaxon"}
    except Exception as e:
        return {"query": q, "match_type": "error", "mapped_name": None, "mapped_id": None,
                "ontology": "NCBITaxon", "error": str(e)}


# ---------------------------------------------------------------------------
# High-level resolvers  (one per entity type)
# ---------------------------------------------------------------------------
# Each returns:
#   status: "matched" (accept id deterministically), "candidates" (needs adjudication),
#           "none" (leave unmapped)
#   plus mapped_name/mapped_id/ontology/candidates as applicable.

def _from_entry(parser, entry: str, match_type: str) -> Dict[str, Any]:
    parsed = parser(entry)
    return {"status": "matched", "match_type": match_type, **parsed}


def resolve_glycan(term: Optional[str]) -> Dict[str, Any]:
    term = normalize_glycan_term(term)
    if not term:
        return {"status": "none", "query": term}
    r = _local_lookup("gsd", term, aliases={})
    if r["match_type"] in ("alias", "exact"):
        return {"query": term, **_from_entry(parse_gsd_entry, r["entry"], r["match_type"])}
    if r["candidates"]:
        return {"status": "candidates", "type": "glycan", "query": term, "candidates": r["candidates"]}
    return {"status": "none", "query": term, "ontology": "GSD"}


def resolve_disease(term: Optional[str]) -> Dict[str, Any]:
    if not term or not term.strip():
        return {"status": "none", "query": term}
    r = _local_lookup("doid", term, aliases=DISEASE_ALIASES)
    if r["match_type"] in ("alias", "exact"):
        return {"query": term, **_from_entry(parse_doid_entry, r["entry"], r["match_type"])}
    if r["candidates"]:
        return {"status": "candidates", "type": "disease", "query": term, "candidates": r["candidates"]}
    return {"status": "none", "query": term, "ontology": "DOID"}


def resolve_specimen(term: Optional[str]) -> Dict[str, Any]:
    if not term or not term.strip():
        return {"status": "none", "query": term}
    # Cell line -> Cellosaurus (no fuzzy; API id-search is exact).
    if looks_like_cell_line(term):
        hit = search_cellosaurus(term)
        if hit.get("mapped_id"):
            return {"status": "matched", "query": term, "category": "cell_line", **hit}
        # fall through to UBERON/CL in case it is actually a tissue token
    # Tissue / body fluid / circulating cell -> UBERON/CL.
    r = _local_lookup("uberon", term, aliases=SPECIMEN_ALIASES)
    if r["match_type"] in ("alias", "exact"):
        return {"query": term, "category": "tissue", **_from_entry(parse_obo_entry, r["entry"], r["match_type"])}
    if r["candidates"]:
        return {"status": "candidates", "type": "specimen", "query": term,
                "category": "tissue", "candidates": r["candidates"]}
    return {"status": "none", "query": term, "ontology": "UBERON"}


_HUMAN = {"status": "matched", "mapped_name": "Homo sapiens", "mapped_id": "9606", "ontology": "NCBITaxon"}


def resolve_species(term: Optional[str]) -> Dict[str, Any]:
    # No organism named -> default to human (the clinical biomarker default).
    if not term or not term.strip():
        return {"query": term, **_HUMAN}
    hit = search_taxonomy(term)
    if hit.get("mapped_id"):
        return {"status": "matched", "query": term, **hit}
    return {"status": "none", "query": term, "ontology": "NCBITaxon"}


def resolve_protein(name: Optional[str], species_name: Optional[str] = None) -> Dict[str, Any]:
    if not name or not name.strip():
        return {"status": "none", "query": name}
    # Single-gene protein -> UniProt; if UniProt has nothing, try Complex Portal.
    hit = search_uniprot(name, species_name)
    if hit.get("mapped_id"):
        return {"status": "matched", "query": name, **hit}
    comp = search_complex(name, species_name)
    if comp.get("mapped_id"):
        return {"status": "matched", "query": name, **comp}
    return {"status": "none", "query": name, "ontology": "UniProt"}


# Map an accepted adjudication entry (verbatim ontology text) back to id/name by type.
_ENTRY_PARSER = {"glycan": parse_gsd_entry, "disease": parse_doid_entry, "specimen": parse_obo_entry}


def parse_accepted_entry(entity_type: str, entry: str) -> Dict[str, Optional[str]]:
    parser = _ENTRY_PARSER.get(entity_type)
    return parser(entry) if parser else {"mapped_name": _split_name(entry), "mapped_id": None, "ontology": None}
