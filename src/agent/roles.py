# roles.py — Per-biomarker-role curation specifications.
#
# One place that encodes, for each BEST biomarker role:
#   - a short curation-oriented definition,
#   - the guided questions the relation extractor must answer for that role,
#   - the role-specific REQUIRED and OPTIONAL annotation fields (from biomarker.md §3),
#   - the hard gate (what the evidence must prove for the role to apply).
#
# The relation-extraction node consumes these to build role-specific *describe* prompts
# and role-specific *structure* schemas, so the extraction path itself changes with the
# inferred role instead of a single role-agnostic prompt.
#
# Design note: each entry is plain data so it is unit-testable without the LLM, and each
# generated prompt stays small and single-role — the operating constraint of gpt-oss:20b.

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Role registry
# ---------------------------------------------------------------------------

# The 7 routable biological roles. `multicomponent` is an architecture flag, not a route.
ROLE_LIST: List[str] = [
    "diagnostic",
    "susceptibility",
    "prognostic",
    "predictive",
    "response",
    "safety",
    "monitoring",
]

# Surface synonyms the model may emit -> canonical role key.
ROLE_ALIASES: Dict[str, str] = {
    "diagnostic": "diagnostic",
    "diagnosis": "diagnostic",
    "susceptibility": "susceptibility",
    "susceptibility/risk": "susceptibility",
    "risk": "susceptibility",
    "prognostic": "prognostic",
    "prognosis": "prognostic",
    "predictive": "predictive",
    "response": "response",
    "pharmacodynamic": "response",
    "pd": "response",
    "surrogate": "response",
    "safety": "safety",
    "toxicity": "safety",
    "monitoring": "monitoring",
    "monitor": "monitoring",
}


def canonical_role(raw: str) -> str | None:
    """Map a model-emitted role string to a canonical role key, or None if unknown."""
    if not raw:
        return None
    return ROLE_ALIASES.get(str(raw).strip().lower())


# Each field entry is (key, human description used in the prompt).
Field = Tuple[str, str]


ROLE_SPECS: Dict[str, Dict[str, Any]] = {
    "diagnostic": {
        "label": "diagnostic",
        "definition": (
            "A diagnostic biomarker detects or confirms the presence of a disease/condition, "
            "or identifies a disease subtype, at the time of measurement — a current-state "
            "classification (cases vs controls, or subtype vs subtype)."
        ),
        "must_prove": (
            "The glycan distinguishes disease from non-disease (or one subtype from another) "
            "in samples measured at a single, current timepoint."
        ),
        "guided_questions": [
            "Which disease/condition is being detected, and does the glycan mark its presence/absence or a subtype?",
            "What is the comparison group (healthy controls, benign disease, another subtype)?",
            "Is there a reference/gold-standard diagnosis the biomarker is checked against?",
            "Is any diagnostic performance reported (sensitivity, specificity, AUC, PPV/NPV)?",
        ],
        "required_fields": [
            ("target_condition", "the disease/condition being diagnosed"),
            ("diagnostic_task", "'presence_absence' or 'subtype_classification'"),
            ("comparison_group", "what the cases are compared against (controls / other subtype)"),
        ],
        "optional_fields": [
            ("case_definition_or_reference", "reference/gold-standard diagnosis used"),
            ("disease_subtype", "subtype identified, if any"),
            ("disease_stage_or_severity", "stage/grade/severity context"),
            ("control_type", "healthy / benign / other-disease control"),
            ("diagnostic_performance", "sensitivity/specificity/AUC/PPV/NPV as stated"),
        ],
    },
    "susceptibility": {
        "label": "susceptibility/risk",
        "definition": (
            "A susceptibility/risk biomarker indicates the potential to DEVELOP a disease in a "
            "subject who does NOT currently have clinically apparent disease — future risk in the "
            "disease-free. It is distinct from prognostic, which applies after disease is present."
        ),
        "must_prove": (
            "Subjects are disease-free at baseline and the glycan is linked to future/incident "
            "onset of the target condition."
        ),
        "guided_questions": [
            "Are the subjects free of the target disease at baseline?",
            "Which future/incident disease does the glycan predict the risk of?",
            "In which direction does the glycan move risk (higher vs lower), and in which at-risk population?",
            "Is a risk time horizon or a reference/low-risk comparison group given?",
        ],
        "required_fields": [
            ("target_future_condition", "the incident disease being risk-stratified"),
            ("baseline_disease_absence", "statement that subjects are disease-free at baseline"),
            ("risk_direction", "increased vs decreased future risk"),
            ("at_risk_population", "the disease-free population studied"),
        ],
        "optional_fields": [
            ("risk_event", "the incident-disease event definition"),
            ("comparison_group", "low-risk / reference group"),
            ("risk_time_horizon", "follow-up window for incident risk"),
            ("exposure_agent", "exposure the risk marker reflects, if any (e.g. tobacco metabolite)"),
            ("risk_estimate", "absolute/relative risk as stated"),
        ],
    },
    "prognostic": {
        "label": "prognostic",
        "definition": (
            "A prognostic biomarker indicates the likelihood of a future clinical event "
            "(recurrence, progression, death, new event) in subjects who ALREADY have the "
            "disease/condition, measured at a defined baseline."
        ),
        "must_prove": (
            "Subjects already have the disease and the glycan is linked to a future clinical "
            "outcome over follow-up."
        ),
        "guided_questions": [
            "Do the subjects already have the disease at the baseline measurement?",
            "Which future event/outcome is predicted (survival, recurrence, progression, time-to-event)?",
            "Over what follow-up period, and in which direction (better vs worse outcome)?",
            "Is a hazard/odds ratio, survival metric, or risk-group cutoff reported?",
        ],
        "required_fields": [
            ("existing_condition", "the disease the subjects already have"),
            ("baseline_disease_state", "baseline disease stage/state at the measurement (disease PRESENT at baseline)"),
            ("baseline_timepoint", "the defined baseline timepoint of measurement"),
            ("future_event_or_transition", "recurrence / progression / death / new event"),
            ("endpoint_of_interest", "the clinical endpoint measured"),
            ("event_direction", "glycan associated with better vs worse outcome"),
        ],
        "optional_fields": [
            ("follow_up_period", "length of follow-up"),
            ("disease_stage", "stage/grade"),
            ("disease_subtype", "subtype"),
            ("background_treatment", "standard-of-care/background therapy"),
            ("outcome_metric", "HR/OR/survival/recurrence/progression metric"),
            ("risk_group_cutoff", "threshold defining risk strata"),
        ],
    },
    "predictive": {
        "label": "predictive",
        "definition": (
            "A predictive biomarker identifies subjects more or less likely to experience a "
            "favorable or unfavorable EFFECT from a specific medical product or exposure, relative "
            "to biomarker-negative subjects. Evidence generally needs a comparison across "
            "biomarker groups (and, ideally, treatment vs control)."
        ),
        "must_prove": (
            "An exposure/product is present AND the glycan marks a DIFFERENTIAL benefit or harm "
            "between biomarker-positive and biomarker-negative subjects."
        ),
        "guided_questions": [
            "Which specific medical product / therapy / environmental agent is involved?",
            "Does the glycan mark a favorable or unfavorable effect of that product?",
            "What are the biomarker-defined groups being compared (biomarker+ vs biomarker-)?",
            "Is there a treated-vs-control (or alternative-intervention) contrast, i.e. an interaction effect?",
        ],
        "required_fields": [
            ("medical_product_or_exposure_agent", "the product/therapy/agent"),
            ("effect_type", "'favorable' or 'unfavorable' differential effect"),
            ("effect_endpoint", "the outcome the effect is measured on"),
            ("biomarker_defined_group", "the biomarker-positive group"),
            ("comparator_biomarker_group", "the biomarker-negative/other group"),
            ("treated_or_exposed_group", "the treated/exposed subject group"),
            ("interaction_claim", "explicit biomarker×treatment interaction (vs a mere prognostic association under treatment)"),
        ],
        "optional_fields": [
            ("control_or_alternative_intervention", "treatment vs control / alternative arm (when available)"),
            ("target_condition", "disease/prevention context"),
            ("dose", "dose/exposure level"),
            ("companion_diagnostic_flag", "used as a companion diagnostic"),
            ("biomarker_cutoff", "threshold defining biomarker positivity"),
        ],
    },
    "response": {
        "label": "response/pharmacodynamic",
        "definition": (
            "A response/pharmacodynamic biomarker shows a biological CHANGE after exposure to a "
            "medical product or environmental agent. A pharmacodynamic biomarker indicates "
            "biological activity but does not by itself establish efficacy or clinical benefit; "
            "when used as a trial endpoint it is a surrogate endpoint biomarker."
        ),
        "must_prove": (
            "An exposure/product is present AND the glycan changes after that exposure."
        ),
        "guided_questions": [
            "Which product/therapy/agent was administered, and was the glycan measured after exposure?",
            "In which direction and by how much did the glycan change after exposure?",
            "At what timing after exposure was it measured (and relative to a baseline)?",
            "Is the change presented as a surrogate endpoint for clinical benefit (candidate / reasonably-likely / validated)?",
        ],
        "required_fields": [
            ("medical_product_or_exposure_agent", "the product/therapy/agent"),
            ("exposure_status", "that the glycan is measured after exposure"),
            ("response_measurement", "what biological response is measured"),
            ("response_direction_or_magnitude", "direction/magnitude of change after exposure"),
        ],
        "optional_fields": [
            ("timing_after_exposure", "timepoint relative to exposure"),
            ("baseline_measurement", "pre-exposure baseline"),
            ("dose", "dose/exposure level"),
            ("target_engagement", "target-engagement/mechanistic pathway"),
            ("surrogate_endpoint_flag", "used as a surrogate endpoint"),
            ("surrogate_validation_level", "candidate / reasonably_likely / validated"),
            ("clinical_outcome_link", "link to a clinical outcome, if claimed"),
        ],
    },
    "safety": {
        "label": "safety",
        "definition": (
            "A safety biomarker is measured before or after exposure to a medical product or "
            "environmental agent to indicate the likelihood, presence, or extent of TOXICITY as "
            "an adverse effect."
        ),
        "must_prove": (
            "An exposure/product is present AND the glycan indicates toxicity/adverse effect "
            "(likelihood, presence, or extent)."
        ),
        "guided_questions": [
            "Which product/therapy/agent, and what toxicity or adverse effect is indicated?",
            "Does the glycan indicate the likelihood, the presence, or the extent of toxicity?",
            "Is it measured before exposure (to avoid harm) or after (developing toxicity)?",
            "Which organ system is affected, and in which direction does harm move?",
        ],
        "required_fields": [
            ("medical_product_or_exposure_agent", "the product/therapy/agent"),
            ("toxicity_concept", "the toxicity/adverse effect"),
            ("toxicity_status", "'likelihood' | 'presence' | 'extent'"),
            ("timing_relative_to_exposure", "before vs after exposure"),
        ],
        "optional_fields": [
            ("harm_direction", "direction of harm"),
            ("organ_system", "affected organ system"),
            ("toxicity_severity", "severity/grade"),
            ("dose", "dose/exposure level"),
            ("action_threshold", "threshold triggering action"),
            ("clinical_action", "dose adjustment / interruption / contraindication"),
        ],
    },
    "monitoring": {
        "label": "monitoring",
        "definition": (
            "A monitoring biomarker is measured REPEATEDLY over time to assess the status or "
            "trajectory of a disease, or the effect of an exposure/product. It is a temporal "
            "wrapper over diagnostic/prognostic/response/safety use — it requires serial measurement."
        ),
        "must_prove": (
            "The glycan is measured at multiple timepoints to track a disease/exposure trajectory."
        ),
        "guided_questions": [
            "Is the glycan measured repeatedly/serially (multiple timepoints), not just once?",
            "What is being monitored (disease burden, progression/recurrence, treatment response, toxicity)?",
            "What is the baseline value and the direction/rate/magnitude of change over time?",
            "At what interval is it monitored, and does a change trigger a defined action?",
        ],
        "required_fields": [
            ("repeated_measurement", "statement of serial/repeated measurement"),
            ("monitored_entity", "disease status / progression / response / toxicity being tracked"),
            ("change_direction", "direction of change over time"),
        ],
        "optional_fields": [
            ("timepoints", "the timepoints/schedule"),
            ("baseline_value", "baseline/prior value"),
            ("change_magnitude_or_rate", "magnitude or rate of change"),
            ("monitoring_interval", "interval between measurements"),
            ("medical_product_or_exposure_agent", "product/exposure being monitored, if any"),
            ("action_threshold", "threshold triggering a clinical action"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_role_spec(role: str) -> Dict[str, Any] | None:
    key = canonical_role(role)
    return ROLE_SPECS.get(key) if key else None


def role_required_keys(role: str) -> List[str]:
    spec = get_role_spec(role)
    return [k for k, _ in spec["required_fields"]] if spec else []


def role_optional_keys(role: str) -> List[str]:
    spec = get_role_spec(role)
    return [k for k, _ in spec["optional_fields"]] if spec else []


def role_annotation_keys(role: str) -> List[str]:
    """All role-specific annotation keys (required + optional)."""
    return role_required_keys(role) + role_optional_keys(role)
