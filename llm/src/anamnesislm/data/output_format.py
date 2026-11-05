"""What the model is trained to write, versus what Extractor returns.

The full fact (FACT_FIELDS) includes evidence offsets and medication detail fields. The model
does not need to write those: offsets are computed in code from the evidence text (the model
counts characters badly), and dose/unit/route/frequency are always null for now. Writing only
MODEL_FIELDS roughly halves each fact's tokens, leaving fewer chances to drift or run out of
max_new_tokens. Extractor.complete_facts adds the rest back, so callers see the same contract.
"""

import json
from typing import Any

MODEL_FIELDS = ("kind", "mention", "assertion", "experiencer", "temporality", "evidence")
DERIVED_FIELDS = ("evidence_start", "evidence_end")
UNFILLED_FIELDS = ("dose", "unit", "route", "frequency")
FACT_FIELDS = MODEL_FIELDS + DERIVED_FIELDS + UNFILLED_FIELDS


def compact_response(expected_json: str) -> str:
    """expected_json with each fact cut down to MODEL_FIELDS."""
    facts = json.loads(expected_json)["facts"]
    return json.dumps({"facts": [{key: fact[key] for key in MODEL_FIELDS} for fact in facts]})


def complete_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add any FACT_FIELDS a fact lacks: unlocated offsets as -1, detail fields as None."""
    defaults: dict[str, Any] = {key: -1 for key in DERIVED_FIELDS}
    defaults.update(dict.fromkeys(UNFILLED_FIELDS))
    return [{**defaults, **fact} for fact in facts]
