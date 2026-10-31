from string import Formatter

import pytest

from anamnesislm.data.templates import TEMPLATE_FAMILIES

TEMPLATE_KEYS = [key for key in TEMPLATE_FAMILIES[0] if key != "name"]
EXTRA_PLACEHOLDERS = {"medication_with_frequency": ["frequency"], "family": ["family_member"]}
ALL_TEMPLATES = [
    (family["name"], key, family[key]) for family in TEMPLATE_FAMILIES for key in TEMPLATE_KEYS
]


def _placeholders(template: str) -> list[str]:
    return [field for _, field, _, _ in Formatter().parse(template) if field is not None]


def test_enough_families_to_hold_one_out() -> None:
    assert len(TEMPLATE_FAMILIES) >= 4


def test_family_names_are_unique() -> None:
    names = [family["name"] for family in TEMPLATE_FAMILIES]

    assert len(names) == len(set(names))


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES, ids=lambda f: f["name"])
def test_every_family_has_the_same_keys(family: dict[str, str]) -> None:
    assert family.keys() == TEMPLATE_FAMILIES[0].keys()


@pytest.mark.parametrize(("name", "key", "template"), ALL_TEMPLATES)
def test_template_has_expected_placeholders(name: str, key: str, template: str) -> None:
    expected = ["mention", *EXTRA_PLACEHOLDERS.get(key, [])]

    assert sorted(_placeholders(template)) == sorted(expected)


@pytest.mark.parametrize(("name", "key", "template"), ALL_TEMPLATES)
def test_mention_appears_verbatim_once(name: str, key: str, template: str) -> None:
    mention = "Diabetes mellitus type 2"

    sentence = template.format(mention=mention, frequency="twice daily", family_member="Mother")

    assert sentence.count(mention) == 1
    assert sentence.endswith(".")
