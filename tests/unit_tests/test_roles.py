"""Unit tests for the per-role curation layer and deterministic helpers.

These exercise pure logic (no LLM, no network): role schema selection, glycan-term hygiene,
the cell-line heuristic, and post-parse normalization — including the smart-quote case that
previously corrupted JSON when normalization ran before parsing.
"""

import json
import sys
from pathlib import Path

import jsonschema
import pytest

# Flat module layout: src/agent modules import each other by bare name.
AGENT_DIR = Path(__file__).resolve().parents[2] / "src" / "agent"
sys.path.insert(0, str(AGENT_DIR))

import roles  # noqa: E402
import schemas  # noqa: E402
import ontology  # noqa: E402
import utils  # noqa: E402


# --------------------------------------------------------------------------- roles / schemas

def test_all_roles_have_schema_with_required_keys():
    for role in roles.ROLE_LIST:
        sch = schemas.build_structure_schema(role)
        jsonschema.Draft7Validator.check_schema(sch)
        req = sch["properties"]["relations"]["items"]["properties"]["role_annotations"]["required"]
        assert req == roles.role_required_keys(role)
        assert req, f"role {role} has no required fields"


def test_role_schema_rejects_missing_required_annotation():
    sch = schemas.build_structure_schema("predictive")
    good = {"relations": [{
        "glycan": "sLeA", "disease": "colorectal cancer", "change": "increased",
        "evidence_sentence_indexes": [3],
        "role_annotations": {
            "medical_product_or_exposure_agent": "anti-EGFR", "effect_type": "favorable",
            "effect_endpoint": "PFS", "biomarker_defined_group": "sLeA-high",
            "comparator_biomarker_group": "sLeA-low", "treated_or_exposed_group": "anti-EGFR arm",
            "interaction_claim": "significant biomarker×treatment interaction",
        },
    }]}
    jsonschema.validate(good, sch)  # passes
    bad = json.loads(json.dumps(good))
    del bad["relations"][0]["role_annotations"]["effect_type"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, sch)


def test_role_aliases_map_to_canonical():
    assert roles.canonical_role("risk") == "susceptibility"
    assert roles.canonical_role("pharmacodynamic") == "response"
    assert roles.canonical_role("PROGNOSTIC") == "prognostic"
    assert roles.canonical_role("nonsense") is None


# --------------------------------------------------------------------------- glycan hygiene

def test_bare_adjective_becomes_glycan_noun_phrase():
    assert ontology.normalize_glycan_term("sialylated") == "sialylated glycan"
    assert ontology.normalize_glycan_term("branched") == "branched glycan"
    # process noun and real motif names are left untouched
    assert ontology.normalize_glycan_term("galactosylation") == "galactosylation"
    assert ontology.normalize_glycan_term("sialyl Lewis A") == "sialyl Lewis A"


def test_cell_line_heuristic():
    for name in ["HEK293", "A549", "MCF-7", "PC-3", "HeLa cell line", "SW480"]:
        assert ontology.looks_like_cell_line(name), name
    for name in ["serum", "breast", "cerebrospinal fluid", "liver"]:
        assert not ontology.looks_like_cell_line(name), name


def test_entry_parsers():
    assert ontology.parse_gsd_entry("core fucosylation | Term UUID: GSD:abc-123 | x")["mapped_id"] == "GSD:abc-123"
    assert ontology.parse_doid_entry("Breast cancer (DOID:1612) is ...")["mapped_id"] == "DOID:1612"
    assert ontology.parse_obo_entry("Blood serum (UBERON:0001977) - ...")["mapped_id"] == "UBERON:0001977"
    assert ontology.parse_obo_entry("Lymphocyte (CL:0000542) - ...")["mapped_id"] == "CL:0000542"


def test_gsd_name_not_polluted_by_definition_parenthesis():
    # GSD definitions contain later "(...)"; the preferred name must be the text before the first
    # " | ", never truncated at a parenthesis buried in the definition.
    entry = ("Tn antigen | Term UUID: GSD:25e2c658 | GSD ID: GSD000167 | "
             "Definition: an N-acetyl-D-galactosamine (GalNAc) with alpha-configuration")
    assert ontology._split_name(entry) == "Tn antigen"
    assert ontology.parse_gsd_entry(entry)["mapped_name"] == "Tn antigen"


# --------------------------------------------------------------------------- normalization

def test_normalize_parsed_strings_fixes_smart_quotes_in_values():
    obj = {"reason": "the glycan “increased” in patients’ serum", "n": 5, "ok": True}
    out = utils.normalize_parsed_strings(obj)
    assert out["reason"] == 'the glycan "increased" in patients\' serum'
    assert out["n"] == 5 and out["ok"] is True  # non-strings untouched


def test_normalization_must_run_after_parse_not_before():
    # A realistic messy value containing smart quotes. Normalizing the RAW envelope first would
    # turn the curly quotes delimiting nothing... but crucially, ASCII-izing quotes inside the
    # envelope is what broke json.loads previously. Post-parse normalization is always safe.
    raw = json.dumps({"note": "shows “association”"}, ensure_ascii=False)
    parsed = json.loads(raw)  # parses fine because the smart quotes are inside a JSON string
    out = utils.normalize_parsed_strings(parsed)
    assert out["note"] == 'shows "association"'


def test_normalize_collapses_whitespace_and_zero_width():
    out = utils.normalize_parsed_strings({"x": "a​ b\n\n c"})
    assert out["x"] == "a b c"
