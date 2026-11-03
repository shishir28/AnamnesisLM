from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


@dataclass(frozen=True)
class ModelOutput:
    raw_output: str
    facts: list[dict[str, Any]] | None
    model_version: str
    error: str | None = None
    warnings: tuple[str, ...] = ()
    generation_token_limit: int = 0


class Fact(BaseModel):
    """Validation only: never coerce or repair generated values."""

    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["condition", "symptom", "medication", "allergy"]
    mention: str = Field(min_length=1)
    assertion: Literal["affirmed", "negated", "uncertain"]
    experiencer: Literal["patient", "family", "unspecified"]
    temporality: Literal["current", "historical", "unspecified"]
    evidence: str = Field(min_length=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(ge=0)
    dose: str | float | None
    unit: str | None
    route: str | None
    frequency: str | None


def validate_facts(note: str, output: ModelOutput) -> list[str]:
    """Check schema and copied evidence, not medical or semantic correctness."""
    if output.error is not None or output.facts is None:
        return [output.error or "Model output is not a facts list."]
    errors: list[str] = []
    for index, data in enumerate(output.facts):
        try:
            fact = Fact.model_validate(data)
        except ValidationError as exc:
            for issue in exc.errors(include_input=False, include_url=False):
                location = ".".join(str(part) for part in issue["loc"])
                errors.append(f"facts[{index}].{location}: {issue['msg']}")
            continue
        if not 0 <= fact.evidence_start < fact.evidence_end <= len(note):
            errors.append(f"facts[{index}]: evidence offsets are outside the note.")
        elif note[fact.evidence_start : fact.evidence_end] != fact.evidence:
            errors.append(f"facts[{index}]: evidence does not match the note at its offsets.")
        if fact.mention not in fact.evidence:
            errors.append(f"facts[{index}]: mention is not contained in its evidence.")
        if fact.kind != "medication" and any(
            value is not None for value in (fact.dose, fact.unit, fact.route, fact.frequency)
        ):
            errors.append(f"facts[{index}]: medication fields must be null for this fact kind.")
    return errors
