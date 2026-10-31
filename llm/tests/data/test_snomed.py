import pytest

from anamnesislm.data.synthea.snomed import split_semantic_tag


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Diabetes mellitus type 2 (disorder)", ("Diabetes mellitus type 2", "disorder")),
        ("Stress (finding)", ("Stress", "finding")),
        ("Medication review due (situation)", ("Medication review due", "situation")),
        ("Refugee (person)", ("Refugee", "person")),
        ("Laceration (morphologic abnormality)", ("Laceration", "morphologic abnormality")),
        ("Body mass index 30+ - obesity (finding)", ("Body mass index 30+ - obesity", "finding")),
    ],
)
def test_splits_known_tag(label: str, expected: tuple[str, str]) -> None:
    assert split_semantic_tag(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Eggs (edible) (substance)", ("Eggs (edible)", "substance")),
        ("Severe anxiety (panic) (finding)", ("Severe anxiety (panic)", "finding")),
    ],
)
def test_only_last_parenthesis_is_the_tag(label: str, expected: tuple[str, str]) -> None:
    assert split_semantic_tag(label) == expected


@pytest.mark.parametrize(
    "label",
    [
        "Dental caries",
        "Fracture of tooth (traumatic)",
        "Malignant neoplasm of tonsillar pillar (anterior) (posterior)",
        "Malignant neoplasm of bone (C40-C41)",
        "Fracture of tooth (traumatic), initial encounter for closed fracture",
    ],
)
def test_unknown_or_missing_tag_leaves_label_unchanged(label: str) -> None:
    assert split_semantic_tag(label) == (label, None)


def test_surrounding_whitespace_is_trimmed() -> None:
    assert split_semantic_tag("  Asthma (disorder) ") == ("Asthma", "disorder")
    assert split_semantic_tag(" Dental caries ") == ("Dental caries", None)
