import json
from pathlib import Path
from typing import Any

import msgspec
import pytest

from anamnesislm.data.synthea import iter_bundle_files, parse_patient_bundle, read_patient_bundle


def _bundle(*resources: dict[str, Any], full_urls: dict[str, str] | None = None) -> bytes:
    full_urls = full_urls or {}
    entries = [
        {"fullUrl": full_urls.get(r["id"], f"urn:uuid:{r['id']}"), "resource": r} for r in resources
    ]
    return json.dumps({"resourceType": "Bundle", "type": "transaction", "entry": entries}).encode()


PATIENT = {
    "resourceType": "Patient",
    "id": "p1",
    "gender": "female",
    "birthDate": "1970-01-01",
    "name": [{"family": "Ignored"}],
}
CONDITION = {
    "resourceType": "Condition",
    "id": "c1",
    "clinicalStatus": {"coding": [{"code": "active"}]},
    "code": {"coding": [{"code": "44054006", "display": "Diabetes mellitus type 2 (disorder)"}]},
    "onsetDateTime": "2000-01-01T00:00:00+00:00",
}
OBSERVATION = {"resourceType": "Observation", "id": "o1", "status": "final"}


def test_parses_patient_and_conditions_ignoring_unused_resources() -> None:
    patient_bundle = parse_patient_bundle(_bundle(PATIENT, CONDITION, OBSERVATION))

    assert patient_bundle is not None
    assert patient_bundle.patient_id == "p1"
    assert patient_bundle.patient.gender == "female"
    assert patient_bundle.patient.birth_date == "1970-01-01"
    [condition] = patient_bundle.conditions
    assert condition.code.label == "Diabetes mellitus type 2 (disorder)"
    assert condition.clinical_status is not None
    assert condition.clinical_status.first_code == "active"


def test_bundle_without_patient_returns_none() -> None:
    practitioner = {"resourceType": "Practitioner", "id": "dr1"}

    assert parse_patient_bundle(_bundle(practitioner)) is None


def test_more_than_one_patient_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected one Patient"):
        parse_patient_bundle(_bundle(PATIENT, {**PATIENT, "id": "p2"}))


def test_missing_required_field_raises_validation_error() -> None:
    condition_without_code = {"resourceType": "Condition", "id": "c2"}

    with pytest.raises(msgspec.ValidationError, match="code"):
        parse_patient_bundle(_bundle(PATIENT, condition_without_code))


def test_medication_name_inline_and_by_reference() -> None:
    medication = {
        "resourceType": "Medication",
        "id": "m1",
        "code": {"text": "levonorgestrel Intrauterine System"},
    }
    inline_request = {
        "resourceType": "MedicationRequest",
        "id": "r1",
        "status": "active",
        "medicationCodeableConcept": {"text": "Clopidogrel 75 MG Oral Tablet"},
        "dosageInstruction": [
            {
                "timing": {"repeat": {"frequency": 2, "period": 1.0, "periodUnit": "d"}},
                "doseAndRate": [{"doseQuantity": {"value": 1.0}}],
            }
        ],
    }
    referenced_request = {
        "resourceType": "MedicationRequest",
        "id": "r2",
        "status": "stopped",
        "medicationReference": {"reference": "urn:uuid:m1"},
    }

    patient_bundle = parse_patient_bundle(
        _bundle(PATIENT, medication, inline_request, referenced_request)
    )

    assert patient_bundle is not None
    inline, referenced = patient_bundle.medication_requests
    assert patient_bundle.medication_name(inline) == "Clopidogrel 75 MG Oral Tablet"
    assert patient_bundle.medication_name(referenced) == "levonorgestrel Intrauterine System"
    [dosage] = inline.dosage_instruction
    assert dosage.timing is not None and dosage.timing.repeat is not None
    assert dosage.timing.repeat.frequency == 2
    assert dosage.timing.repeat.period_unit == "d"
    assert dosage.dose_and_rate[0].dose_quantity is not None
    assert dosage.dose_and_rate[0].dose_quantity.value == 1.0


def test_iter_bundle_files_is_sorted_and_reads_from_disk(tmp_path: Path) -> None:
    for state in ("Texas", "California"):
        fhir_dir = tmp_path / state / "fhir"
        fhir_dir.mkdir(parents=True)
        (fhir_dir / "b.json").write_bytes(_bundle(PATIENT))
        (fhir_dir / "a.json").write_bytes(_bundle(PATIENT))
    (tmp_path / "Texas" / "metadata").mkdir()
    (tmp_path / "Texas" / "metadata" / "run.json").write_text("{}")

    files = list(iter_bundle_files(tmp_path))

    assert [f.relative_to(tmp_path).as_posix() for f in files] == [
        "California/fhir/a.json",
        "California/fhir/b.json",
        "Texas/fhir/a.json",
        "Texas/fhir/b.json",
    ]
    patient_bundle = read_patient_bundle(files[0])
    assert patient_bundle is not None
    assert patient_bundle.source_path == str(files[0])
