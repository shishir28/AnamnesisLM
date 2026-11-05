import re
from string import Formatter

import pytest

from anamnesislm.data.templates import (
    FAMILY_CONDITIONS,
    FAMILY_MEMBERS,
    HOLDOUT_TEMPLATES,
    SYMPTOMS,
    TRAIN_TEMPLATES,
)

TEMPLATE_KEYS = list(TRAIN_TEMPLATES)
EXTRA_PLACEHOLDERS = {"medication_with_frequency": ["frequency"], "family": ["family_member"]}
ALL_TEMPLATES = [
    (source, key, template)
    for source, templates in [("train", TRAIN_TEMPLATES), ("holdout", HOLDOUT_TEMPLATES)]
    for key, phrasings in templates.items()
    for template in phrasings
]
# Keys record_builder renders today (medication_with_frequency is not used yet).
RENDERED_KEYS = [key for key in TEMPLATE_KEYS if key != "medication_with_frequency"]
MIN_TRAIN_PHRASINGS = 10
MIN_HOLDOUT_PHRASINGS = 4  # one phrasing per key made holdout F1 swing 35-71% between seeds
# A cue word must mean one label everywhere, or the model learns it as a coin flip
# (e.g. "not" in an uncertain phrasing taught "do not include" -> uncertain).
CUE_WORDS = {
    "symptom_negated": ["no", "not", "denies", "denied", "without", "absent", "free", "negative"],
    "symptom_uncertain": ["possible", "possibly", "suspected", "probable", "query", "uncertain"],
    "symptom_affirmed": ["present"],
}


def _placeholders(template: str) -> list[str]:
    return [field for _, field, _, _ in Formatter().parse(template) if field is not None]


def _words(template: str) -> set[str]:
    """Lower-case words outside placeholders: "Rx: {mention}." -> {"rx"}."""
    return set(re.findall(r"[a-z/']+", re.sub(r"\{[a-z_]+\}", " ", template.lower())))


def _fixed_text(template: str) -> list[str]:
    """The template's own words, split at placeholders: "Known allergy to {mention}." ->
    ["Known allergy to"]. Fragments shorter than two words are too common to compare."""
    pieces = (p.strip(" .,()") for p in re.split(r"\{[a-z_]+\}", template))
    return [p for p in pieces if len(p.split()) >= 2 or "'" in p]


def test_holdout_and_train_have_the_same_keys() -> None:
    assert HOLDOUT_TEMPLATES.keys() == TRAIN_TEMPLATES.keys()


@pytest.mark.parametrize("key", RENDERED_KEYS)
def test_each_rendered_key_has_many_training_phrasings(key: str) -> None:
    assert len(TRAIN_TEMPLATES[key]) >= MIN_TRAIN_PHRASINGS


@pytest.mark.parametrize("key", RENDERED_KEYS)
def test_each_rendered_key_has_several_holdout_phrasings(key: str) -> None:
    assert len(HOLDOUT_TEMPLATES[key]) >= MIN_HOLDOUT_PHRASINGS


@pytest.mark.parametrize("templates", [TRAIN_TEMPLATES, HOLDOUT_TEMPLATES])
@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_no_duplicate_phrasings(templates: dict[str, list[str]], key: str) -> None:
    assert len(templates[key]) == len(set(templates[key]))


@pytest.mark.parametrize(("source", "key", "template"), ALL_TEMPLATES)
def test_template_has_expected_placeholders(source: str, key: str, template: str) -> None:
    expected = ["mention", *EXTRA_PLACEHOLDERS.get(key, [])]

    assert sorted(_placeholders(template)) == sorted(expected)


@pytest.mark.parametrize(("source", "key", "template"), ALL_TEMPLATES)
def test_mention_appears_verbatim_once(source: str, key: str, template: str) -> None:
    mention = "Diabetes mellitus type 2"

    sentence = template.format(mention=mention, frequency="twice daily", family_member="Mother")

    assert sentence.count(mention) == 1
    assert sentence.endswith(".")


def test_holdout_templates_never_appear_in_training() -> None:
    trained = {t for phrasings in TRAIN_TEMPLATES.values() for t in phrasings}

    assert [t for p in HOLDOUT_TEMPLATES.values() for t in p if t in trained] == []


@pytest.mark.parametrize(
    "fragment",
    sorted({f for p in HOLDOUT_TEMPLATES.values() for t in p for f in _fixed_text(t)}),
)
def test_holdout_wording_never_appears_inside_training_templates(fragment: str) -> None:
    """Otherwise the held-out set would partly measure wording the model already saw."""
    trained = [t for phrasings in TRAIN_TEMPLATES.values() for t in phrasings]

    assert [t for t in trained if fragment.lower() in t.lower()] == []


@pytest.mark.parametrize(
    ("key", "template"), [(k, t) for k, phrasings in HOLDOUT_TEMPLATES.items() for t in phrasings]
)
def test_holdout_uses_only_words_from_training(key: str, template: str) -> None:
    """A from-scratch model has untrained embeddings for words it never saw, so a phrasing with
    a new word measures luck, not generalisation."""
    trained = set().union(*(_words(t) for p in TRAIN_TEMPLATES.values() for t in p))

    assert _words(template) - trained == set()


def test_words_ignores_placeholders() -> None:
    assert _words("{family_member}'s FHx w/ {mention}.") == {"'s", "fhx", "w/"}


def test_fixed_text_splits_at_placeholders() -> None:
    assert _fixed_text("{family_member}'s history includes {mention}.") == ["'s history includes"]
    assert _fixed_text("No {mention}.") == []


@pytest.mark.parametrize(
    ("owner", "word"), [(owner, word) for owner, words in CUE_WORDS.items() for word in words]
)
@pytest.mark.parametrize("templates", [TRAIN_TEMPLATES, HOLDOUT_TEMPLATES])
def test_cue_word_is_used_only_by_its_own_label(
    templates: dict[str, list[str]], owner: str, word: str
) -> None:
    users = {
        key
        for key, phrasings in templates.items()
        for template in phrasings
        if re.search(rf"\b{word}\b", template.lower())
    }

    assert users <= {owner}


@pytest.mark.parametrize("symptom", SYMPTOMS)
def test_symptoms_contain_no_cue_words(symptom: str) -> None:
    cue_words = {word for words in CUE_WORDS.values() for word in words}

    assert cue_words.isdisjoint(symptom.lower().split())


@pytest.mark.parametrize(
    ("vocabulary", "minimum"), [(SYMPTOMS, 20), (FAMILY_CONDITIONS, 20), (FAMILY_MEMBERS, 8)]
)
def test_augmentation_vocabularies_are_large_and_distinct(
    vocabulary: list[str], minimum: int
) -> None:
    assert len(vocabulary) >= minimum
    assert len(vocabulary) == len(set(vocabulary))
