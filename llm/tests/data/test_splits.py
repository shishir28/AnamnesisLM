from collections import Counter

import pytest

from anamnesislm.data.splits import SPLITS, assign_split


def test_same_patient_always_gets_same_split() -> None:
    assert assign_split("9579ac1e-c14c-affb-64ec-4be44843bc2a") == assign_split(
        "9579ac1e-c14c-affb-64ec-4be44843bc2a"
    )


def test_proportions_are_close_to_80_10_10() -> None:
    counts = Counter(assign_split(f"patient-{i}") for i in range(20_000))

    assert set(counts) == set(SPLITS)
    assert 0.78 < counts["train"] / 20_000 < 0.82
    assert 0.09 < counts["validation"] / 20_000 < 0.11
    assert 0.09 < counts["test"] / 20_000 < 0.11


def test_custom_percentages() -> None:
    assert {assign_split(f"p{i}", train_percent=100, validation_percent=0) for i in range(100)} == {
        "train"
    }


def test_invalid_percentages_rejected() -> None:
    with pytest.raises(ValueError):
        assign_split("p1", train_percent=90, validation_percent=20)
