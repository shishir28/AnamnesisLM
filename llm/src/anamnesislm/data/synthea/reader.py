"""Discover and decode Synthea FHIR bundle files.

A Synthea cohort looks like ``<cohort>/<State>/fhir/*.json``. Most files are patient
bundles; each state also has ``hospitalInformation*.json`` and
``practitionerInformation*.json``, which contain no Patient and are skipped.
"""

from collections.abc import Iterator
from pathlib import Path

import msgspec

from anamnesislm.data.synthea.models import (
    AllergyIntolerance,
    Condition,
    Medication,
    MedicationRequest,
    Patient,
    PatientBundle,
)


class _ResourceHeader(msgspec.Struct, rename="camel"):
    resource_type: str


class _Entry(msgspec.Struct, rename="camel"):
    # Left undecoded: most entries (Observation, Claim, ...) are never needed.
    resource: msgspec.Raw
    full_url: str | None = None


class _Bundle(msgspec.Struct, rename="camel"):
    resource_type: str
    entry: tuple[_Entry, ...] = ()


_bundle_decoder = msgspec.json.Decoder(_Bundle)
_header_decoder = msgspec.json.Decoder(_ResourceHeader)
_patient_decoder = msgspec.json.Decoder(Patient)
_condition_decoder = msgspec.json.Decoder(Condition)
_medication_decoder = msgspec.json.Decoder(Medication)
_medication_request_decoder = msgspec.json.Decoder(MedicationRequest)
_allergy_decoder = msgspec.json.Decoder(AllergyIntolerance)


def iter_bundle_files(cohort_dir: Path) -> Iterator[Path]:
    """Yield every ``<State>/fhir/*.json`` file in a stable, sorted order."""
    yield from sorted(cohort_dir.glob("*/fhir/*.json"))


def read_patient_bundle(path: Path) -> PatientBundle | None:
    """Read one file; return None if it is not a patient bundle.

    Raises ``OSError`` if the file cannot be read and ``msgspec.DecodeError`` (or its
    subclass ``msgspec.ValidationError``) if it is malformed or a needed field is missing.
    """
    return parse_patient_bundle(path.read_bytes(), source_path=str(path))


def parse_patient_bundle(data: bytes, source_path: str = "<memory>") -> PatientBundle | None:
    """Decode bundle JSON bytes; return None if the bundle has no Patient resource."""
    bundle = _bundle_decoder.decode(data)
    if bundle.resource_type != "Bundle":
        return None

    patients: list[Patient] = []
    conditions: list[Condition] = []
    medication_requests: list[MedicationRequest] = []
    allergies: list[AllergyIntolerance] = []
    medications_by_url: dict[str, Medication] = {}

    for entry in bundle.entry:
        resource_type = _header_decoder.decode(entry.resource).resource_type
        if resource_type == "Patient":
            patients.append(_patient_decoder.decode(entry.resource))
        elif resource_type == "Condition":
            conditions.append(_condition_decoder.decode(entry.resource))
        elif resource_type == "MedicationRequest":
            medication_requests.append(_medication_request_decoder.decode(entry.resource))
        elif resource_type == "AllergyIntolerance":
            allergies.append(_allergy_decoder.decode(entry.resource))
        elif resource_type == "Medication":
            medication = _medication_decoder.decode(entry.resource)
            medications_by_url[entry.full_url or f"urn:uuid:{medication.id}"] = medication

    if not patients:
        return None
    if len(patients) > 1:
        raise ValueError(f"{source_path}: expected one Patient, found {len(patients)}")

    return PatientBundle(
        source_path=source_path,
        patient=patients[0],
        conditions=tuple(conditions),
        medication_requests=tuple(medication_requests),
        allergies=tuple(allergies),
        medications_by_url=medications_by_url,
    )
