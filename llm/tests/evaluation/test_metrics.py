from typing import Any

import pytest

from anamnesislm.evaluation.metrics import (
    Counts,
    compare_facts,
    compare_facts_by_kind,
    diff_facts,
    fact_key,
    is_supported,
)


def _fact(mention: str = "Asthma", kind: str = "condition", **overrides: Any) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "kind": kind,
        "mention": mention,
        "assertion": "affirmed",
        "experiencer": "patient",
        "temporality": "current",
        "evidence": f"History of {mention}.",
        "evidence_start": 0,
        "evidence_end": 10,
        "dose": None,
    }
    fact.update(overrides)
    return fact


# Counts


def test_scores_from_counts() -> None:
    counts = Counts(true_positives=3, false_positives=1, false_negatives=2)

    assert counts.precision == pytest.approx(3 / 4)
    assert counts.recall == pytest.approx(3 / 5)
    assert counts.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_empty_counts_score_zero_instead_of_dividing_by_zero() -> None:
    counts = Counts()

    assert (counts.precision, counts.recall, counts.f1) == (0.0, 0.0, 0.0)


def test_no_correct_facts_scores_zero_f1() -> None:
    counts = Counts(false_positives=2, false_negatives=3)

    assert (counts.precision, counts.recall, counts.f1) == (0.0, 0.0, 0.0)


def test_add_sums_each_tally() -> None:
    total = Counts(true_positives=1, false_positives=2, false_negatives=3)

    total.add(Counts(true_positives=10, false_positives=20, false_negatives=30))

    assert total == Counts(true_positives=11, false_positives=22, false_negatives=33)


def test_micro_average_differs_from_averaging_scores() -> None:
    small = Counts(true_positives=1)  # precision 1.0
    large = Counts(true_positives=1, false_positives=9)  # precision 0.1
    total = Counts()

    total.add(small)
    total.add(large)

    assert total.precision == pytest.approx(2 / 11)  # not (1.0 + 0.1) / 2


# fact_key


def test_key_ignores_fields_outside_the_match_rule() -> None:
    a = _fact(evidence="Asthma.", evidence_start=5, dose="10 mg")
    b = _fact(evidence="History of asthma.", evidence_start=0, dose=None)

    assert fact_key(a) == fact_key(b)


@pytest.mark.parametrize("field", ["kind", "mention", "assertion", "experiencer", "temporality"])
def test_key_changes_with_each_match_field(field: str) -> None:
    assert fact_key(_fact()) != fact_key(_fact(**{field: "different"}))


def test_missing_fields_become_none() -> None:
    assert fact_key({"kind": "allergy"}) == ("allergy", None, None, None, None)


# compare_facts


def test_identical_lists_are_all_true_positives() -> None:
    facts = [_fact("Asthma"), _fact("Aspirin", kind="medication")]

    assert compare_facts(facts, facts) == Counts(true_positives=2)


def test_order_does_not_matter() -> None:
    a, b = _fact("Asthma"), _fact("Gingivitis")

    assert compare_facts([a, b], [b, a]) == Counts(true_positives=2)


def test_wrong_assertion_is_one_false_positive_and_one_false_negative() -> None:
    predicted = [_fact(assertion="negated")]
    expected = [_fact(assertion="affirmed")]

    assert compare_facts(predicted, expected) == Counts(false_positives=1, false_negatives=1)


def test_invented_and_omitted_facts() -> None:
    predicted = [_fact("Asthma"), _fact("Invented")]
    expected = [_fact("Asthma"), _fact("Omitted")]

    assert compare_facts(predicted, expected) == Counts(
        true_positives=1, false_positives=1, false_negatives=1
    )


def test_duplicate_prediction_matches_only_once() -> None:
    assert compare_facts([_fact(), _fact()], [_fact()]) == Counts(
        true_positives=1, false_positives=1
    )


def test_duplicate_expected_facts_each_need_a_prediction() -> None:
    assert compare_facts([_fact()], [_fact(), _fact()]) == Counts(
        true_positives=1, false_negatives=1
    )


def test_no_predictions_omits_everything() -> None:
    assert compare_facts([], [_fact("A"), _fact("B")]) == Counts(false_negatives=2)


def test_both_empty_is_all_zero() -> None:
    assert compare_facts([], []) == Counts()


def test_unhashable_values_in_model_output_do_not_crash() -> None:
    predicted = [_fact(mention=["not", "text"])]  # type: ignore[arg-type]

    assert compare_facts(predicted, [_fact()]) == Counts(false_positives=1, false_negatives=1)


def test_inputs_are_not_modified() -> None:
    predicted, expected = [_fact()], [_fact()]

    compare_facts(predicted, expected)

    assert (predicted, expected) == ([_fact()], [_fact()])


# compare_facts_by_kind


def test_counts_are_split_by_kind() -> None:
    predicted = [_fact("Asthma"), _fact("Aspirin", kind="medication")]
    expected = [_fact("Asthma"), _fact("Ibuprofen", kind="medication")]

    assert compare_facts_by_kind(predicted, expected) == {
        "condition": Counts(true_positives=1),
        "medication": Counts(false_positives=1, false_negatives=1),
    }


def test_kind_seen_only_in_expected_or_only_in_predicted_is_reported() -> None:
    predicted = [_fact("Aspirin", kind="medication")]
    expected = [_fact("Peanut", kind="allergy")]

    assert compare_facts_by_kind(predicted, expected) == {
        "allergy": Counts(false_negatives=1),
        "medication": Counts(false_positives=1),
    }


def test_kinds_come_back_sorted() -> None:
    facts = [_fact(kind=k) for k in ["medication", "allergy", "condition"]]

    assert list(compare_facts_by_kind(facts, facts)) == ["allergy", "condition", "medication"]


@pytest.mark.parametrize("bad_kind", [None, 5, ["condition"]])
def test_missing_or_non_text_kind_counts_as_invalid(bad_kind: object) -> None:
    predicted = [{k: v for k, v in _fact(kind="x").items() if k != "kind"}]
    if bad_kind is not None:
        predicted[0]["kind"] = bad_kind

    assert compare_facts_by_kind(predicted, []) == {"invalid": Counts(false_positives=1)}


def test_per_kind_counts_add_up_to_overall() -> None:
    predicted = [_fact("A"), _fact("B", kind="medication"), _fact("C", kind="allergy")]
    expected = [_fact("A"), _fact("X", kind="medication")]
    total = Counts()

    for counts in compare_facts_by_kind(predicted, expected).values():
        total.add(counts)

    assert total == compare_facts(predicted, expected)


def test_no_facts_gives_no_kinds() -> None:
    assert compare_facts_by_kind([], []) == {}


# is_supported


def test_mention_in_note_is_supported() -> None:
    assert is_supported(_fact("Gingivitis"), "Past problem: Gingivitis.")


def test_mention_not_in_note_is_invented() -> None:
    assert not is_supported(_fact("Asthma"), "Past problem: Gingivitis.")


def test_match_is_case_sensitive() -> None:
    assert not is_supported(_fact("gingivitis"), "Past problem: Gingivitis.")


@pytest.mark.parametrize("mention", [None, "", 5, ["Gingivitis"]])
def test_missing_empty_or_non_text_mention_is_not_supported(mention: object) -> None:
    assert not is_supported({"kind": "condition", "mention": mention}, "Gingivitis")


def test_fact_without_mention_is_not_supported() -> None:
    assert not is_supported({"kind": "condition"}, "Gingivitis")


# diff_facts


def test_diff_returns_extra_and_missing_facts() -> None:
    asthma, invented, omitted = _fact("Asthma"), _fact("Invented"), _fact("Omitted")

    extra, missing = diff_facts([asthma, invented], [asthma, omitted])

    assert (extra, missing) == ([invented], [omitted])


def test_diff_keeps_the_second_of_two_identical_predictions_as_extra() -> None:
    extra, missing = diff_facts([_fact(), _fact()], [_fact()])

    assert (len(extra), missing) == (1, [])


def test_diff_of_matching_lists_is_empty() -> None:
    facts = [_fact("A"), _fact("B")]

    assert diff_facts(facts, list(reversed(facts))) == ([], [])
