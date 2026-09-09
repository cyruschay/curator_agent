INPUT_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "processing_id": {
      "type": "string"
    },
    "pmid": {
      "type": "string"
    },
    "pmcid": {
      "type": ["string", "null"]
    },
    "title": {
      "type": "string"
    },
    "abstract": {
      "type": ["string", "null"]
    },
    "keywords": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_names": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_rn": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_uid": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "mesh_names": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "mesh_uid": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "journal_abbr": {
      "type": "string"
    },
    "article_date": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Year": {
            "type": "string"
          },
          "Month": {
            "type": "string"
          },
          "Day": {
            "type": "string"
          }
        },
        "required": ["Year", "Month", "Day"]
      }
    }
  },
  "required": ["processing_id", "pmid", "title", "abstract"]
}


# ---------------------------------------------------------------------------
# Per-role relation-extraction schemas (grammar-constrained "structure" step)
# ---------------------------------------------------------------------------
#
# Built from roles.ROLE_SPECS so the output schema changes with the biomarker role.
# The role's REQUIRED annotation keys are always emitted (present in output) but are
# nullable strings, so the model is never forced to fabricate a value when the evidence
# is silent — the required-field *enforcement* happens at role-ID gating and at the
# role-aware validator (N06), not by making the grammar reject nulls.

from roles import ROLE_SPECS, canonical_role  # noqa: E402


def _nullable_string(description: str) -> dict:
    return {"type": ["string", "null"], "description": description}


def build_structure_schema(role: str) -> dict:
    """JSON schema for the structure step of one biomarker role.

    Shape: {"relations": [ {base identity fields + role_annotations{...}} ]}
    """
    spec = ROLE_SPECS.get(canonical_role(role) or "")
    if spec is None:
        raise ValueError(f"Unknown biomarker role: {role!r}")

    role_props: dict = {}
    for key, desc in spec["required_fields"]:
        role_props[key] = _nullable_string(desc)
    for key, desc in spec["optional_fields"]:
        role_props[key] = _nullable_string(desc)
    required_role_keys = [k for k, _ in spec["required_fields"]]

    relation_schema = {
        "type": "object",
        "properties": {
            "glycan": {"type": "string", "description": "exact glycan term from the entity list"},
            "glycan_metadata": _nullable_string("ratio/lectin-binding/other local descriptors"),
            "disease": {"type": "string", "description": "disease/condition named in the text"},
            "disease_annotation": _nullable_string("stage/grade/subtype/severity/status"),
            "change": {
                "type": "string",
                "enum": ["increased", "decreased", "absent", "novel_presence", "associated"],
                "description": "direction of the glycan change",
            },
            "specimen": _nullable_string("tissue/organ/body-fluid/cell/cell-line; full form preferred"),
            "species": _nullable_string("human/mouse/etc."),
            "protein_name": _nullable_string("carrier protein/enzyme (HGNC symbol or common name)"),
            "method_names": {"type": "array", "items": {"type": "string"}},
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["number", "null"]},
                        "raw": {"type": ["string", "null"]},
                    },
                    "required": ["name"],
                },
            },
            "evidence_sentence_indexes": {"type": "array", "items": {"type": "integer"}},
            "negated_or_hedged": {"type": "boolean"},
            "is_multicomponent": {"type": "boolean", "description": "part of a multi-biomarker panel/score"},
            "role_annotations": {
                "type": "object",
                "properties": role_props,
                "required": required_role_keys,
            },
        },
        "required": [
            "glycan",
            "disease",
            "change",
            "specimen",
            "evidence_sentence_indexes",
            "role_annotations",
        ],
    }

    return {
        "type": "object",
        "properties": {"relations": {"type": "array", "items": relation_schema}},
        "required": ["relations"],
    }


# Grammar schemas for the remaining fixed-structure LLM calls. Grammar constraint guarantees valid
# JSON AND correct field names — free-text output under repeat_penalty=1.4 mangles keys
# (e.g. "alignement", "non_structural_modifier"), silently dropping data.

def build_ner_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "glycans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "glycan_structure_term": {"type": "string"},
                        "non_structural_descriptor": {"type": ["string", "null"]},
                        "alignment": {"type": "string", "enum": ["whole_structure", "substructure", "core"]},
                        "aglycon": {"type": "string", "enum": ["R", "Ser/Thr", "Asn", "Cer", "free_glycan", "other"]},
                        "chemical_structure": {"type": ["string", "null"]},
                        "evidence_sentence_index": {"type": ["integer", "null"]},
                    },
                    "required": ["glycan_structure_term", "alignment", "aglycon"],
                },
            }
        },
        "required": ["glycans"],
    }


def build_screening_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["RELEVANT", "IRRELEVANT"]},
            "reasoning": {"type": "string"},
            "study_type": {
                "type": "string",
                "enum": ["case-control", "cohort", "cross-sectional", "review",
                         "meta-analysis", "editorial", "other"],
            },
            "needs_fulltext": {"type": "boolean"},
        },
        "required": ["decision", "reasoning", "study_type", "needs_fulltext"],
    }


def build_validation_schema() -> dict:
    split_item = {
        "type": "object",
        "properties": {
            k: {"type": ["string", "null"]} for k in (
                "glycan_id", "glycan_name", "glycan_mapped_name",
                "disease_id", "disease_name", "disease_mapped_name", "disease_annotation",
                "specimen",
                "species_id", "species_name", "species_mapped_name",
                "protein_id", "protein_name", "protein_mapped_name",
            )
        },
    }
    return {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original_index": {"type": "integer"},
                        "action": {"type": "string", "enum": ["keep", "fix", "split", "reject"]},
                        "reason": {"type": "string"},
                        "glycan_normalized": {"type": ["string", "null"]},
                        "specimen_fix": {"type": ["string", "null"]},
                        "split_biomarkers": {"type": "array", "items": split_item},
                    },
                    "required": ["original_index", "action", "reason"],
                },
            }
        },
        "required": ["relations"],
    }


def build_section_filter_schema() -> dict:
    return {
        "type": "object",
        "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
        "required": ["titles"],
    }


def build_adjudication_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "accepted_entry": {"type": ["string", "null"]},
                        "reason": {"type": ["string", "null"]},
                    },
                    "required": ["id", "accepted_entry"],
                },
            }
        },
        "required": ["decisions"],
    }


# Compact schema for the role-identification step (which roles are present + their glycans).
def build_role_id_schema() -> dict:
    from roles import ROLE_LIST
    return {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ROLE_LIST},
                        "glycans": {"type": "array", "items": {"type": "string"}},
                        "rationale": {"type": ["string", "null"]},
                    },
                    "required": ["role", "glycans"],
                },
            },
            "multicomponent": {"type": "boolean"},
        },
        "required": ["roles"],
    }