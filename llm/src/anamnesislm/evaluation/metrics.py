"""Strict fact matching and precision / recall / F1.

A predicted fact counts as correct only if every field in MATCH_FIELDS equals an expected
fact's field. Freeze these rules before the main test run; changing them later makes scores
incomparable.
"""

from dataclasses import dataclass
from typing import Any

Fact = dict[str, Any]
FactKey = tuple[Any, ...]

# Fields compared for a strict match. Evidence offsets and dose/unit/route/frequency are left
# out for now (the last four are always null in the data).
MATCH_FIELDS = ("kind", "mention", "assertion", "experiencer", "temporality")


@dataclass
class Counts:
    """Matching tallies for one group of facts (all facts, or one kind)."""

    true_positives: int = 0  # predicted and expected
    false_positives: int = 0  # predicted, not expected
    false_negatives: int = 0  # expected, not predicted (omitted)

    def add(self, other: "Counts") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives

    @property
    def precision(self) -> float:
        num_predicted = self.true_positives + self.false_positives
        return 0.0 if num_predicted == 0 else self.true_positives / num_predicted

    @property
    def recall(self) -> float:
        num_expected = self.true_positives + self.false_negatives
        return 0.0 if num_expected == 0 else self.true_positives / num_expected

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def fact_key(fact: Fact) -> FactKey:
    """The MATCH_FIELDS values of a fact, as a tuple (missing fields become None)."""
    return tuple(fact.get(field) for field in MATCH_FIELDS)


def diff_facts(predicted: list[Fact], expected: list[Fact]) -> tuple[list[Fact], list[Fact]]:
    """(extra, missing): predicted facts with no expected match, expected facts never matched.

    Each expected fact can match only one prediction, so duplicates are handled: two identical
    predictions against one expected fact leave one of them in extra.
    """
    missing = list(expected)
    extra: list[Fact] = []
    for fact in predicted:
        key = fact_key(fact)
        match = next((i for i, e in enumerate(missing) if fact_key(e) == key), None)
        if match is None:
            extra.append(fact)
        else:
            missing.pop(match)
    return extra, missing


def compare_facts(predicted: list[Fact], expected: list[Fact]) -> Counts:
    """Match predicted against expected by fact_key and count tp / fp / fn."""
    extra, missing = diff_facts(predicted, expected)
    return Counts(
        true_positives=len(predicted) - len(extra),
        false_positives=len(extra),
        false_negatives=len(missing),
    )


def compare_facts_by_kind(predicted: list[Fact], expected: list[Fact]) -> dict[str, Counts]:
    """compare_facts separately for each kind (condition / medication / allergy).

    Kinds come back sorted, so reports list them in the same order every run.
    """
    counts_by_kind: dict[str, Counts] = {}
    for kind in sorted({_kind(fact) for fact in predicted + expected}):
        predicted_of_kind = [fact for fact in predicted if _kind(fact) == kind]
        expected_of_kind = [fact for fact in expected if _kind(fact) == kind]
        counts_by_kind[kind] = compare_facts(predicted_of_kind, expected_of_kind)
    return counts_by_kind


def _kind(fact: Fact) -> str:
    """The fact's kind, or "invalid" when model output has no kind or a non-text one."""
    kind = fact.get("kind")
    return kind if isinstance(kind, str) else "invalid"


def is_supported(fact: Fact, note: str) -> bool:
    """True if the fact's mention appears in the note; False means the model invented it."""
    mention = fact.get("mention")
    return isinstance(mention, str) and mention != "" and mention in note
