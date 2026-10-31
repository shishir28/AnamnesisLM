"""Typed views of the Synthea FHIR R4 resources used for note rendering.

Only the fields we read are declared; msgspec ignores everything else in the JSON.
Field names are snake_case and map to FHIR camelCase through ``rename="camel"``.
"""

from dataclasses import dataclass, field

import msgspec


class Coding(msgspec.Struct, frozen=True):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class CodeableConcept(msgspec.Struct, frozen=True):
    text: str | None = None
    coding: tuple[Coding, ...] = ()

    @property
    def label(self) -> str | None:
        """Human-readable name: ``text`` if present, else the first coding's display."""
        if self.text:
            return self.text
        for coding in self.coding:
            if coding.display:
                return coding.display
        return None

    @property
    def first_code(self) -> str | None:
        """The first coding's code, e.g. ``"active"`` for a clinical status."""
        for coding in self.coding:
            if coding.code:
                return coding.code
        return None


class Reference(msgspec.Struct, frozen=True):
    reference: str | None = None


class Quantity(msgspec.Struct, frozen=True):
    value: float | None = None
    unit: str | None = None


class TimingRepeat(msgspec.Struct, frozen=True, rename="camel"):
    frequency: int | None = None
    period: float | None = None
    period_unit: str | None = None


class Timing(msgspec.Struct, frozen=True):
    repeat: TimingRepeat | None = None


class DoseAndRate(msgspec.Struct, frozen=True, rename="camel"):
    dose_quantity: Quantity | None = None


class Dosage(msgspec.Struct, frozen=True, rename="camel"):
    text: str | None = None
    as_needed_boolean: bool | None = None
    timing: Timing | None = None
    dose_and_rate: tuple[DoseAndRate, ...] = ()


class Patient(msgspec.Struct, frozen=True, rename="camel"):
    id: str
    gender: str | None = None
    birth_date: str | None = None
    deceased_date_time: str | None = None


class Condition(msgspec.Struct, frozen=True, rename="camel"):
    id: str
    code: CodeableConcept
    clinical_status: CodeableConcept | None = None
    onset_date_time: str | None = None
    abatement_date_time: str | None = None


class Medication(msgspec.Struct, frozen=True):
    id: str
    code: CodeableConcept


class MedicationRequest(msgspec.Struct, frozen=True, rename="camel"):
    """A prescription; the drug is either inline or a reference to a Medication entry."""

    id: str
    status: str
    authored_on: str | None = None
    medication_codeable_concept: CodeableConcept | None = None
    medication_reference: Reference | None = None
    dosage_instruction: tuple[Dosage, ...] = ()


class AllergyIntolerance(msgspec.Struct, frozen=True, rename="camel"):
    id: str
    code: CodeableConcept
    clinical_status: CodeableConcept | None = None
    type: str | None = None
    category: tuple[str, ...] = ()
    criticality: str | None = None


@dataclass(frozen=True)
class PatientBundle:
    """One patient's resources, grouped by type. Notes are rendered from this, not raw FHIR."""

    source_path: str
    patient: Patient
    conditions: tuple[Condition, ...] = ()
    medication_requests: tuple[MedicationRequest, ...] = ()
    allergies: tuple[AllergyIntolerance, ...] = ()
    # Medication resources keyed by the bundle entry's fullUrl ("urn:uuid:...").
    medications_by_url: dict[str, Medication] = field(default_factory=dict)

    @property
    def patient_id(self) -> str:
        return self.patient.id

    def medication_name(self, request: MedicationRequest) -> str | None:
        """Drug name for a request, resolving ``medicationReference`` within this bundle."""
        if request.medication_codeable_concept is not None:
            return request.medication_codeable_concept.label
        if request.medication_reference is not None and request.medication_reference.reference:
            medication = self.medications_by_url.get(request.medication_reference.reference)
            if medication is not None:
                return medication.code.label
        return None
