# Workflow Design

Goal: supply **role-specific prompts & output schemas** according to the
biomarker role(s) inferred from the article.

## Architecture

The extraction path changes with the identified biomarker role:

    diagnostic · susceptibility · prognostic · predictive · response · safety · monitoring

`multicomponent` is an _architecture_ flag, not a route.  
Role is a **non-exclusive list**: one article can yield relations under
several roles. We therefore do **multi-pass** extraction gated by a role-ID step, rather than
single-best-role routing (the pre-session plan) or a shallow multi-tag on one shared prompt
(the collapsed version). Each LLM call stays simple and single-role - key consideration given
the small `gpt-oss:20b` model.

## Node flow

    N01 Screener        reject irrelevant, tag study_type + needs_fulltext (T/F)
    N02 Retrieval       PMC full text when needed
    N03/N03a NER        glycan entities, exclusion rules apply
    N04/N04a RE         ROLE-DRIVEN, three steps in one node:
        (1) role-ID     which of 7 roles are present + which glycans each + multicomponent
        (2) describe    per present role: role-specific guided questions + required/optional field list
        (3) structure   per present role: grammar-constrained JSON with that role's schema
    N05 Ontology        DETERMINISTIC (previously cancelled the LLM tool-calling loop): exact/alias -> semantic(top-k>5)
                        -> single bounded adjudication call for semantic-only hits. Role/type-aware
                        routing (uniprot vs complex; uberon vs CL vs cellosaurus; disease-tissue ->
                        nearest non-disease specimen + annotation). No fuzzy match on glycans/cell lines.
    N06 Validate/Refine per-relation, role-aware
    N07 Export          role_annotations subset per row

## Key rules

- Reject: animal/cell-line-model & biochemical (tissue/cell) studies; bioengineering of
  non-natural glycans/proteins/antibodies (unless the protein/antibody is a therapeutic/exposure
  agent); assay/pipeline-development papers; reviews (semantic, not string match).
- Glycans: exact -> semantic only (NO fuzzy). Adjectives become "<adj> glycan"
  (sialylated -> sialylated glycan). Glycan term carries no protein/tissue
  (asialotransferrin -> asialylated glycan + transferrin; Tn-MUC1 -> Tn antigen + MUC1).
- Undecipherable peak labels (GP20, IGP33) excluded at NER.
- Specimen must be tissue/organ/body-fluid/cell/cell-line; full form preferred
  (CSF -> cerebrospinal fluid); disease tissue -> nearest non-disease term + annotation
  (breast tumor -> breast + "tumor"); tissue microarray -> the specific tissue.
- Ontology matching never forces a match (null is allowed).
- Normalization of LLM output applied to parsed string values (post-parse), never to raw JSON.
- Bug-fix hygiene: replace a failed fix, do not stack fail-safes.

## Changed files

- `roles.py` NEW — ROLE_SPECS (definition, guided questions, required/optional fields) + prompt/schema builders
- `prompts.py` screening/NER hardening; ontology adjudication sub-prompts; role-shared RE preamble
- `schemas.py` per-role structure schema builder
- `states.py` biomarker_role: List[str]; role_annotations; is_multicomponent
- `ontology.py` NEW — deterministic resolvers (rewritten from tools.py); top-k>5; routing
- `graph.py` role-driven N04/N04a; deterministic N05; remove N07; rewire; export subset
- `utils.py` normalize_parsed_strings (post-parse)
- `tests/unit_tests/test_roles.py` NEW — role schema selection, glycan adjective normalization, post-parse normalization
