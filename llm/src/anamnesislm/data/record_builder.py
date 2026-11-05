"""Render a patient's structured record into a clinical note and its expected facts."""

import json
import random
from datetime import date
from hashlib import sha256

from anamnesislm.data.mentions import NameSet, made_up_name, short_drug_name
from anamnesislm.data.records import ExtractionRecord
from anamnesislm.data.synthea.models import AllergyIntolerance, Condition, PatientBundle
from anamnesislm.data.synthea.snomed import split_semantic_tag
from anamnesislm.data.templates import (
    FAMILY_CONDITIONS,
    FAMILY_MEMBERS,
    SYMPTOMS,
    TRAIN_TEMPLATES,
    Templates,
)

# Synthea simulation end date (cohort metadata "endTime"); ages are measured to it.
REFERENCE_DATE = date(2025, 1, 1)
MAX_CONDITIONS = 2
MAX_MEDICATIONS = 1
MAX_ALLERGIES = 1
AUGMENT_PROBABILITY = 0.35
MAX_FACTS = 4
# Share of real names replaced by made-up ones, so the model must copy names, not recall them.
MENTION_SWAP_PROBABILITY = 0.3
# Share of remaining drug names shortened ("Metformin hydrochloride", "Metformin").
SHORT_DRUG_PROBABILITY = 0.3

Fact = dict[str, str | int | None]


def _select_conditions(conditions: tuple[Condition, ...]) -> list[tuple[str, str]]:
    """Return up to MAX_CONDITIONS (mention, temporality) pairs for distinct disorders.

    Synthea repeats a condition per episode; if any episode is active, it is "current".
    """
    selected: dict[str, str] = {}
    for c in conditions:
        if c.code.label is None:
            continue
        mention, tag = split_semantic_tag(c.code.label)
        if tag != "disorder":
            continue
        status = c.clinical_status.first_code if c.clinical_status else None
        temporality = "current" if status == "active" else "historical"
        if selected.get(mention) != "current":
            selected[mention] = temporality
    return list(selected.items())[:MAX_CONDITIONS]


def _select_medications(patient_bundle: PatientBundle) -> list[str]:
    """Return up to MAX_MEDICATIONS distinct drug names from active requests."""
    selected: dict[str, None] = {}
    for request in patient_bundle.medication_requests:
        if request.status != "active":
            continue
        drug = patient_bundle.medication_name(request)
        if drug is None:
            continue
        selected[drug] = None
    return list(selected)[:MAX_MEDICATIONS]


def _select_allergies(allergies: tuple[AllergyIntolerance, ...]) -> list[str]:
    """Return up to MAX_ALLERGIES distinct allergens that are still active.

    Every allergy template reads as current, so inactive allergies are left out.
    Synthea lists "Allergic disposition (finding)" first; it marks the patient, not an allergen.
    """
    selected: dict[str, None] = {}
    for a in allergies:
        if a.code.label is None:
            continue
        status = a.clinical_status.first_code if a.clinical_status else None
        if status != "active":
            continue
        mention, _ = split_semantic_tag(a.code.label)
        if mention == "Allergic disposition":
            continue
        selected[mention] = None
    return list(selected)[:MAX_ALLERGIES]


def _indefinite_article(age: int) -> str:
    """Return "an" for ages read with a leading vowel sound: 8, 11, 18, 80-89."""
    return "an" if str(age).startswith("8") or age in (11, 18) else "a"


def _fact(
    sentence: str,
    kind: str,
    mention: str,
    temporality: str,
    assertion: str = "affirmed",
    experiencer: str = "patient",
) -> Fact:
    """Return the fact a sentence states; offsets are set later by _render."""
    return {
        "kind": kind,
        "mention": mention,
        "assertion": assertion,
        "experiencer": experiencer,
        "temporality": temporality,
        "evidence": sentence,
        "evidence_start": -1,
        "evidence_end": -1,
        "dose": None,
        "unit": None,
        "route": None,
        "frequency": None,
    }


def _phrase(templates: Templates, key: str, wording_rng: random.Random, **fields: str) -> str:
    """One randomly chosen phrasing of templates[key], filled in."""
    return wording_rng.choice(templates[key]).format(**fields)


def _data_augmentation(
    facts: list[Fact], templates: Templates, rng: random.Random, wording_rng: random.Random
) -> None:
    """Maybe insert a symptom and a family-history fact at random positions in facts.

    Teaches the labels the real record never produces (negated, uncertain, family).
    Mentions come from fixed lists, so each label is true by construction.
    Random positions stop the model learning "last sentence = augmented".
    Stops at MAX_FACTS to keep note + JSON within the sequence length.
    """
    if len(facts) < MAX_FACTS and rng.random() < AUGMENT_PROBABILITY:
        symptom = rng.choice(SYMPTOMS)
        assertion = rng.choice(["affirmed", "negated", "uncertain"])
        sentence = _phrase(templates, f"symptom_{assertion}", wording_rng, mention=symptom)
        fact = _fact(sentence, "condition", symptom, "current", assertion=assertion)
        facts.insert(rng.randint(0, len(facts)), fact)

    if len(facts) < MAX_FACTS and rng.random() < AUGMENT_PROBABILITY:
        family_member = rng.choice(FAMILY_MEMBERS)
        condition = rng.choice(FAMILY_CONDITIONS)
        sentence = _phrase(
            templates, "family", wording_rng, mention=condition, family_member=family_member
        )
        fact = _fact(sentence, "condition", condition, "current", experiencer="family")
        facts.insert(rng.randint(0, len(facts)), fact)


def _mention(real: str, kind: str, name_rng: random.Random, name_set: NameSet) -> str:
    """The name to write for a real record entry.

    "unseen": always a made-up name from UNSEEN_SYLLABLES (test_unseen_names split only).
    "train": usually the real name; sometimes a made-up one, or a shortened drug name.
    """
    if name_set == "unseen":
        return made_up_name(kind, name_rng, "unseen")
    if name_rng.random() < MENTION_SWAP_PROBABILITY:
        return made_up_name(kind, name_rng, "train")
    if kind == "medication" and name_rng.random() < SHORT_DRUG_PROBABILITY:
        short = short_drug_name(real)
        return short.split()[0] if name_rng.random() < 0.5 else short
    return real


def _render(header: str, facts: list[Fact]) -> str:
    """Join header and fact sentences into the note, setting each fact's evidence offsets."""
    note = header
    for fact in facts:
        sentence = str(fact["evidence"])
        if str(fact["mention"]) not in sentence:
            raise ValueError(f"Mention is not inside its evidence: {fact}")
        start = len(note) + 1
        note += " " + sentence
        fact["evidence_start"], fact["evidence_end"] = start, len(note)
    return note


def build_record(
    patient_bundle: PatientBundle,
    templates: Templates = TRAIN_TEMPLATES,
    name_set: NameSet = "train",
) -> ExtractionRecord:
    """Render one patient bundle into a note and the facts that note states.

    Labels come from the rendered text, never from the raw FHIR fields alone.
    Each sentence uses a random phrasing from templates; pass HOLDOUT_TEMPLATES to build the
    unseen-wording test set, or name_set="unseen" to build the unseen-names test set.
    """
    patient = patient_bundle.patient
    record_id = patient_bundle.patient_id
    # Deterministic choices, seeded from the patient id. Three generators: rng picks the content
    # (which facts, augmentation), wording_rng picks phrasings, name_rng picks substitute names.
    # Changing templates or names then never changes the facts, so a held-out record differs
    # from its test twin only in wording (or only in names).
    rng = random.Random(sha256(record_id.encode()).digest())
    wording_rng = random.Random(sha256(f"{record_id}:wording".encode()).digest())
    name_rng = random.Random(sha256(f"{record_id}:names".encode()).digest())

    if patient.birth_date is None or patient.gender is None:
        raise ValueError(f"Patient {record_id} is missing birth_date or gender.")
    born = date.fromisoformat(patient.birth_date)
    # Deceased patients: the note is a snapshot at death, so age stops there.
    as_of = (
        date.fromisoformat(patient.deceased_date_time[:10])
        if patient.deceased_date_time
        else REFERENCE_DATE
    )
    age = as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))

    header = f"Patient is {_indefinite_article(age)} {age}-year-old {patient.gender}."
    facts: list[Fact] = []

    # Vary the count so some notes leave room for augmentation under MAX_FACTS.
    n_conditions = rng.randint(1, MAX_CONDITIONS)
    for real, temporality in _select_conditions(patient_bundle.conditions)[:n_conditions]:
        mention = _mention(real, "condition", name_rng, name_set)
        key = "active_condition" if temporality == "current" else "resolved_condition"
        sentence = _phrase(templates, key, wording_rng, mention=mention)
        facts.append(_fact(sentence, "condition", mention, temporality))

    for real in _select_medications(patient_bundle):
        drug = _mention(real, "medication", name_rng, name_set)
        sentence = _phrase(templates, "medication", wording_rng, mention=drug)
        facts.append(_fact(sentence, "medication", drug, "current"))

    for real in _select_allergies(patient_bundle.allergies):
        allergen = _mention(real, "allergy", name_rng, name_set)
        sentence = _phrase(templates, "allergy", wording_rng, mention=allergen)
        facts.append(_fact(sentence, "allergy", allergen, "current"))

    _data_augmentation(facts, templates, rng, wording_rng)
    note = _render(header, facts)
    return ExtractionRecord(
        note=note, expected_json=json.dumps({"facts": facts}), record_id=record_id
    )
