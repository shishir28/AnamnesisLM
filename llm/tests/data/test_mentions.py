import random

import pytest

from anamnesislm.data.mentions import (
    TRAIN_SYLLABLES,
    UNSEEN_SYLLABLES,
    made_up_name,
    short_drug_name,
)

KINDS = ["condition", "medication", "allergy"]


def _names(name_set: str, count: int = 3000) -> set[str]:
    rng = random.Random(0)
    return {made_up_name(rng.choice(KINDS), rng, name_set) for _ in range(count)}  # type: ignore[arg-type]


def test_syllable_sets_are_disjoint() -> None:
    assert set(TRAIN_SYLLABLES).isdisjoint(UNSEEN_SYLLABLES)


def test_unseen_names_never_match_training_names() -> None:
    assert _names("train").isdisjoint(_names("unseen"))


def test_names_are_varied() -> None:
    assert len(_names("train")) > 2000


def test_same_seed_gives_the_same_name() -> None:
    assert made_up_name("condition", random.Random(7), "train") == made_up_name(
        "condition", random.Random(7), "train"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_names_start_with_a_capital_and_have_no_placeholder(kind: str) -> None:
    rng = random.Random(1)
    for _ in range(200):
        name = made_up_name(kind, rng, "unseen")
        assert name[0].isupper()
        assert "{" not in name


def test_medication_names_look_like_drugs() -> None:
    rng = random.Random(2)
    names = [made_up_name("medication", rng, "train") for _ in range(200)]

    assert any(n.endswith("MG Oral Tablet") for n in names)
    assert any(" " not in n for n in names)


@pytest.mark.parametrize(
    ("drug", "short"),
    [
        ("Metformin hydrochloride 500 MG Oral Tablet", "Metformin hydrochloride"),
        ("Clopidogrel 75 MG Oral Tablet", "Clopidogrel"),
        ("Acetaminophen 300 MG / Hydrocodone Bitartrate 5 MG Oral Tablet", "Acetaminophen"),
        ("Insulin Lispro", "Insulin Lispro"),
    ],
)
def test_short_drug_name_drops_strength_and_form(drug: str, short: str) -> None:
    assert short_drug_name(drug) == short
