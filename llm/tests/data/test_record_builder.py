import json
from typing import Any

import pytest

from anamnesislm.data.record_builder import (
    MAX_FACTS,
    _indefinite_article,
    _select_allergies,
    _select_conditions,
    _select_medications,
    build_record,
)
from anamnesislm.data.synthea.models import (
    AllergyIntolerance,
    CodeableConcept,
    Coding,
    Condition,
    MedicationRequest,
    Patient,
    PatientBundle,
)
from anamnesislm.data.templates import HOLDOUT_FAMILY, TEMPLATE_FAMILIES

FACT_KEYS = {
    "kind",
    "mention",
    "assertion",
    "experiencer",
    "temporality",
    "evidence",
    "evidence_start",
    "evidence_end",
    "dose",
    "unit",
    "route",
    "frequency",
}


def _concept(label: str, code: str | None = None) -> CodeableConcept:
    return CodeableConcept(coding=(Coding(code=code, display=label),))


def _condition(label: str, status: str = "active") -> Condition:
    return Condition(id=label, code=_concept(label), clinical_status=_concept(status, status))


def _allergy(label: str, status: str = "active") -> AllergyIntolerance:
    return AllergyIntolerance(
        id=label, code=_concept(label), clinical_status=_concept(status, status)
    )


def _request(drug: str, status: str = "active") -> MedicationRequest:
    return MedicationRequest(
        id=drug, status=status, medication_codeable_concept=CodeableConcept(text=drug)
    )


def _bundle(
    patient_id: str = "p1",
    birth_date: str | None = "1970-01-01",
    gender: str | None = "female",
    deceased_date_time: str | None = None,
) -> PatientBundle:
    return PatientBundle(
        source_path="<test>",
        patient=Patient(
            id=patient_id,
            gender=gender,
            birth_date=birth_date,
            deceased_date_time=deceased_date_time,
        ),
        conditions=(
            _condition("Diabetes mellitus type 2 (disorder)"),
            _condition("Acute bronchitis (disorder)", "resolved"),
        ),
        medication_requests=(_request("Metformin 500 MG Oral Tablet"),),
        allergies=(_allergy("Allergic disposition (finding)"), _allergy("Peanut (substance)")),
    )


def _facts(patient_bundle: PatientBundle) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = json.loads(build_record(patient_bundle).expected_json)["facts"]
    return facts


MANY_PATIENTS = [_bundle(f"patient-{i}") for i in range(200)]


# build_record


@pytest.mark.parametrize("patient_bundle", MANY_PATIENTS[:50], ids=lambda b: b.patient_id)
def test_evidence_offsets_point_at_the_evidence(patient_bundle: PatientBundle) -> None:
    record = build_record(patient_bundle)

    for fact in json.loads(record.expected_json)["facts"]:
        assert record.note[fact["evidence_start"] : fact["evidence_end"]] == fact["evidence"]
        assert fact["mention"] in fact["evidence"]


def test_every_fact_has_the_contract_keys() -> None:
    for patient_bundle in MANY_PATIENTS[:50]:
        for fact in _facts(patient_bundle):
            assert fact.keys() == FACT_KEYS


def test_same_patient_gives_the_same_record() -> None:
    first, second = build_record(_bundle()), build_record(_bundle())

    assert (first.note, first.expected_json) == (second.note, second.expected_json)
    assert first.record_id == "p1"


def test_never_more_than_max_facts() -> None:
    assert all(len(_facts(b)) <= MAX_FACTS for b in MANY_PATIENTS)


def test_header_states_age_and_gender() -> None:
    note = build_record(_bundle(birth_date="1970-01-01", gender="male")).note

    assert note.startswith("Patient is a 55-year-old male.")


def test_header_uses_an_before_eight() -> None:
    note = build_record(_bundle(birth_date="2016-06-01")).note

    assert note.startswith("Patient is an 8-year-old female.")


def test_deceased_patient_age_stops_at_death() -> None:
    note = build_record(
        _bundle(birth_date="1970-01-01", deceased_date_time="2000-06-01T10:00:00Z")
    ).note

    assert note.startswith("Patient is a 30-year-old female.")


@pytest.mark.parametrize(
    "patient_bundle", [_bundle(birth_date=None), _bundle(gender=None)], ids=["birth_date", "gender"]
)
def test_missing_birth_date_or_gender_raises(patient_bundle: PatientBundle) -> None:
    with pytest.raises(ValueError, match="missing birth_date or gender"):
        build_record(patient_bundle)


def test_real_facts_are_affirmed_patient_facts() -> None:
    facts = _facts(_bundle())
    real = [f for f in facts if f["kind"] in ("medication", "allergy")]

    assert {(f["kind"], f["mention"]) for f in real} == {
        ("medication", "Metformin 500 MG Oral Tablet"),
        ("allergy", "Peanut"),
    }
    assert all((f["assertion"], f["experiencer"]) == ("affirmed", "patient") for f in real)


def test_augmentation_produces_every_label_across_patients() -> None:
    facts = [f for b in MANY_PATIENTS for f in _facts(b)]

    assert {f["assertion"] for f in facts} == {"affirmed", "negated", "uncertain"}
    assert {f["experiencer"] for f in facts} == {"patient", "family"}


def test_augmented_facts_are_not_always_last() -> None:
    positions = []
    for patient_bundle in MANY_PATIENTS:
        facts = _facts(patient_bundle)
        for i, fact in enumerate(facts):
            if (fact["assertion"], fact["experiencer"]) != ("affirmed", "patient"):
                positions.append(i == len(facts) - 1)

    assert not all(positions)


def test_holdout_family_is_never_rendered() -> None:
    [holdout] = [f for f in TEMPLATE_FAMILIES if f["name"] == HOLDOUT_FAMILY]
    holdout_sentence = holdout["medication"].format(mention="Metformin 500 MG Oral Tablet")

    assert all(holdout_sentence not in build_record(b).note for b in MANY_PATIENTS)


# selection helpers


def test_conditions_keep_disorders_dedupe_and_prefer_current() -> None:
    conditions = (
        _condition("Sinusitis (disorder)", "resolved"),
        _condition("Stress (finding)"),
        _condition("Sinusitis (disorder)", "active"),
        _condition("Fracture of tooth (traumatic)"),
    )

    assert _select_conditions(conditions) == [("Sinusitis", "current")]


def test_allergies_skip_disposition_and_inactive() -> None:
    allergies = (
        _allergy("Allergic disposition (finding)"),
        _allergy("Shellfish (substance)", "inactive"),
        _allergy("Peanut (substance)"),
    )

    assert _select_allergies(allergies) == ["Peanut"]


def test_medications_keep_active_distinct_names() -> None:
    patient_bundle = PatientBundle(
        source_path="<test>",
        patient=Patient(id="p1"),
        medication_requests=(
            _request("Aspirin 81 MG", "completed"),
            _request("Lisinopril 10 MG"),
            _request("Lisinopril 10 MG"),
        ),
    )

    assert _select_medications(patient_bundle) == ["Lisinopril 10 MG"]


@pytest.mark.parametrize(
    ("age", "article"),
    [(1, "a"), (8, "an"), (11, "an"), (18, "an"), (28, "a"), (80, "an"), (89, "an"), (108, "a")],
)
def test_indefinite_article(age: int, article: str) -> None:
    assert _indefinite_article(age) == article
