import json
from typing import Any, cast

import pytest

from anamnesislm.data.records import ExtractionRecord
from anamnesislm.evaluation.evaluate import (
    EvaluationReport,
    evaluate,
    format_failures,
    format_report,
    remove_overlap,
    report_to_dict,
)
from anamnesislm.evaluation.metrics import Counts
from anamnesislm.inference.extractor import ExtractionResult, Extractor, parse_facts


def _fact(mention: str, kind: str = "condition", **overrides: Any) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "kind": kind,
        "mention": mention,
        "assertion": "affirmed",
        "experiencer": "patient",
        "temporality": "current",
    }
    fact.update(overrides)
    return fact


def _json(*facts: dict[str, Any]) -> str:
    return json.dumps({"facts": list(facts)})


def _record(note: str, *facts: dict[str, Any], record_id: str = "r") -> ExtractionRecord:
    return ExtractionRecord(note=note, expected_json=_json(*facts), record_id=record_id)


class ScriptedExtractor:
    """Stand-in for Extractor: returns pre-written raw outputs, one per extract call."""

    model_version = "fake@step1"

    def __init__(self, *raw_outputs: str) -> None:
        self.raw_outputs = list(raw_outputs)
        self.notes: list[str] = []

    def extract(self, note: str) -> ExtractionResult:
        self.notes.append(note)
        return parse_facts(self.raw_outputs.pop(0), self.model_version)


def _run(records: list[ExtractionRecord], *raw_outputs: str) -> EvaluationReport:
    return evaluate(cast(Extractor, ScriptedExtractor(*raw_outputs)), records)


# remove_overlap


def test_notes_seen_in_train_are_dropped() -> None:
    train = [_record("Asthma."), _record("Fever.")]
    test = [_record("Asthma.", record_id="t1"), _record("Gingivitis.", record_id="t2")]

    kept = remove_overlap(test, train)

    assert [r.record_id for r in kept] == ["t2"]


def test_overlap_compares_note_text_not_record_id() -> None:
    train = [_record("Asthma.", record_id="same")]
    test = [_record("Fever.", record_id="same")]

    assert remove_overlap(test, train) == test


def test_nothing_seen_keeps_everything_in_order() -> None:
    test = [_record("B"), _record("A"), _record("C")]

    assert remove_overlap(test, []) == test


# evaluate


def test_each_note_is_passed_to_the_extractor() -> None:
    extractor = ScriptedExtractor(_json(), _json())
    records = [_record("First note."), _record("Second note.")]

    evaluate(cast(Extractor, extractor), records)

    assert extractor.notes == ["First note.", "Second note."]


def test_perfect_output_scores_full_marks() -> None:
    facts = [_fact("Asthma"), _fact("Aspirin", kind="medication")]

    report = _run([_record("Asthma. Aspirin.", *facts)], _json(*facts))

    assert report.overall == Counts(true_positives=2)
    assert report.valid_json_rate == 1.0
    assert report.num_unsupported == 0


def test_invalid_output_counts_expected_facts_as_omitted() -> None:
    report = _run([_record("Asthma.", _fact("Asthma"))], "not json")

    assert report.num_valid_json == 0
    assert report.overall == Counts(false_negatives=1)
    assert report.per_kind == {"condition": Counts(false_negatives=1)}


def test_valid_empty_output_counts_as_valid_json() -> None:
    report = _run([_record("Asthma.", _fact("Asthma"))], _json())

    assert report.num_valid_json == 1
    assert report.overall == Counts(false_negatives=1)


def test_counts_add_up_across_records() -> None:
    records = [
        _record("Asthma.", _fact("Asthma")),
        _record("Fever.", _fact("Fever", assertion="negated")),
        _record("Aspirin.", _fact("Aspirin", kind="medication")),
    ]

    report = _run(
        records,
        _json(_fact("Asthma")),  # correct
        _json(_fact("Fever")),  # wrong assertion: fp + fn
        "{broken",  # invalid: fn
    )

    assert report.num_records == 3
    assert report.num_valid_json == 2
    assert report.overall == Counts(true_positives=1, false_positives=1, false_negatives=2)
    assert report.per_kind == {
        "condition": Counts(true_positives=1, false_positives=1, false_negatives=1),
        "medication": Counts(false_negatives=1),
    }


def test_invented_mentions_are_counted_as_unsupported() -> None:
    record = _record("Asthma.", _fact("Asthma"))

    report = _run([record], _json(_fact("Asthma"), _fact("Diabetes"), _fact("Cancer")))

    assert report.num_unsupported == 2
    assert report.overall == Counts(true_positives=1, false_positives=2)


def test_report_carries_the_model_version() -> None:
    assert _run([]).model_version == "fake@step1"


def test_no_records_gives_an_empty_report() -> None:
    report = _run([])

    assert report == EvaluationReport(model_version="fake@step1")
    assert (report.valid_json_rate, report.seconds_per_record) == (0.0, 0.0)


def test_extraction_time_is_recorded() -> None:
    report = _run([_record("A."), _record("B.")], _json(), _json())

    assert report.total_seconds > 0
    assert report.seconds_per_record == pytest.approx(report.total_seconds / 2)


# format_report


def _report() -> EvaluationReport:
    return EvaluationReport(
        model_version="best.pt@step8000",
        num_records=4,
        num_valid_json=3,
        overall=Counts(true_positives=3, false_positives=1, false_negatives=1),
        per_kind={
            "medication": Counts(false_positives=1),
            "allergy": Counts(true_positives=1),
            "condition": Counts(true_positives=2, false_negatives=1),
        },
        num_unsupported=1,
        total_seconds=2.0,
    )


def test_report_text_shows_the_headline_numbers() -> None:
    text = format_report(_report())

    assert "Model version: best.pt@step8000" in text
    assert "Records scored: 4" in text
    assert "Valid JSON rate: 75.0%" in text
    assert "Overall: precision 75.0%, recall 75.0%, F1 75.0% (tp 3, fp 1, fn 1)" in text
    assert "Unsupported facts: 1" in text
    assert "Omitted facts: 1" in text
    assert "Seconds per record: 0.500" in text


def test_report_lists_kinds_in_sorted_order() -> None:
    lines = format_report(_report()).splitlines()

    kind_lines = [line.split(":")[0].strip() for line in lines if line.startswith("  ")]
    assert kind_lines == ["allergy", "condition", "medication"]


def test_empty_report_formats_without_errors() -> None:
    text = format_report(EvaluationReport(model_version="v"))

    assert "Records scored: 0" in text
    assert "Valid JSON rate: 0.0%" in text


# report_to_dict


def test_report_dict_keeps_raw_counts_and_scores() -> None:
    data = report_to_dict(_report())

    assert data["overall"] == {
        "true_positives": 3,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.75,
        "recall": 0.75,
        "f1": 0.75,
    }
    assert data["valid_json_rate"] == 0.75
    assert data["num_omitted"] == 1
    assert data["seconds_per_record"] == 0.5


def test_report_dict_lists_kinds_in_sorted_order() -> None:
    assert list(report_to_dict(_report())["per_kind"]) == ["allergy", "condition", "medication"]


def test_report_dict_is_json_serialisable() -> None:
    data = report_to_dict(_report())

    assert json.loads(json.dumps(data)) == data


# failures


def test_correct_records_are_not_failures() -> None:
    report = _run([_record("Asthma.", _fact("Asthma"))], _json(_fact("Asthma")))

    assert report.failures == []


def test_failure_keeps_extra_and_missing_facts() -> None:
    record = _record("Fever.", _fact("Fever", assertion="negated"), record_id="r1")

    [failure] = _run([record], _json(_fact("Fever"))).failures

    assert failure.record_id == "r1"
    assert failure.note == "Fever."
    assert failure.parse_error is None
    assert failure.extra == [_fact("Fever")]
    assert failure.missing == [_fact("Fever", assertion="negated")]


def test_invalid_output_is_a_failure_with_its_raw_text() -> None:
    [failure] = _run([_record("Asthma.", _fact("Asthma"))], '{"facts": [').failures

    assert failure.raw_output == '{"facts": ['
    assert failure.parse_error is not None
    assert failure.missing == [_fact("Asthma")]


def test_failures_are_not_in_the_saved_dict() -> None:
    report = _run([_record("Secret note.", _fact("Asthma"))], "not json")

    data = report_to_dict(report)

    assert data["num_failed_records"] == 1
    assert "Secret note." not in json.dumps(data)


def test_failure_text_shows_note_and_facts() -> None:
    record = _record("Fever.", _fact("Fever", assertion="negated"), record_id="r1")
    report = _run([record], _json(_fact("Fever"), _fact("Cancer")))

    text = format_failures(report)

    assert "[1/1] record r1" in text
    assert "Note: Fever." in text
    assert "Missing: condition | Fever | negated | patient | current" in text
    assert "Extra:   condition | Fever | affirmed | patient | current\n" in text
    assert "Extra:   condition | Cancer | affirmed | patient | current  (not in note)" in text


def test_failure_text_shows_invalid_output() -> None:
    report = _run([_record("Asthma.", _fact("Asthma"))], "{broken")

    assert "Invalid output (JSONDecodeError" in format_failures(report)
    assert "{broken" in format_failures(report)


def test_no_failures_text() -> None:
    assert format_failures(EvaluationReport(model_version="v")) == "No failures."
