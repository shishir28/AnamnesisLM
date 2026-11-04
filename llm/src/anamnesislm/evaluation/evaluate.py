"""Run an Extractor over records and collect a report.

Lives in the package (not the script) so it can be tested with a fake extractor.
Nothing here prints; the script decides what to show.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from anamnesislm.data.records import ExtractionRecord
from anamnesislm.evaluation.metrics import (
    Counts,
    Fact,
    compare_facts,
    compare_facts_by_kind,
    diff_facts,
    fact_key,
    is_supported,
)
from anamnesislm.inference.extractor import Extractor


@dataclass
class RecordFailure:
    """One record the model got wrong. Holds note text, so it is printed only, never saved."""

    record_id: str
    note: str
    raw_output: str
    parse_error: str | None  # set when the output was not valid JSON
    extra: list[Fact]  # predicted but not expected
    missing: list[Fact]  # expected but not predicted


@dataclass
class EvaluationReport:
    """Aggregate scores for one evaluation run, plus the records that went wrong."""

    model_version: str
    num_records: int = 0
    num_valid_json: int = 0
    overall: Counts = field(default_factory=Counts)
    per_kind: dict[str, Counts] = field(default_factory=dict)
    num_unsupported: int = 0  # predicted facts whose mention is not in the note
    total_seconds: float = 0.0  # extraction time summed over records
    failures: list[RecordFailure] = field(default_factory=list)  # left out of report_to_dict

    @property
    def valid_json_rate(self) -> float:
        """num_valid_json / num_records; 0.0 when there are no records."""
        return 0.0 if self.num_records == 0 else self.num_valid_json / self.num_records

    @property
    def seconds_per_record(self) -> float:
        """total_seconds / num_records; 0.0 when there are no records."""
        return 0.0 if self.num_records == 0 else self.total_seconds / self.num_records


def remove_overlap(
    records: list[ExtractionRecord], seen: list[ExtractionRecord]
) -> list[ExtractionRecord]:
    """Drop records whose note text also appears in seen (e.g. test notes found in train)."""
    seen_notes = {record.note for record in seen}
    return [record for record in records if record.note not in seen_notes]


def evaluate(extractor: Extractor, records: list[ExtractionRecord]) -> EvaluationReport:
    """For each record: time extractor.extract(note), parse expected_json, and update the report.

    Invalid model output is scored as an empty prediction, so every expected fact counts as
    omitted (false negative).
    """
    report = EvaluationReport(model_version=extractor.model_version)
    for record in records:
        start = time.perf_counter()
        result = extractor.extract(record.note)
        report.total_seconds += time.perf_counter() - start

        expected = json.loads(record.expected_json)["facts"]
        if result.facts is not None:
            report.num_valid_json += 1
        predicted = result.facts or []
        extra, missing = diff_facts(predicted, expected)

        report.num_records += 1
        report.overall.add(compare_facts(predicted, expected))
        for kind, counts in compare_facts_by_kind(predicted, expected).items():
            report.per_kind.setdefault(kind, Counts()).add(counts)
        report.num_unsupported += sum(not is_supported(fact, record.note) for fact in predicted)

        if result.error is not None or extra or missing:
            report.failures.append(
                RecordFailure(
                    record_id=record.record_id,
                    note=record.note,
                    raw_output=result.raw_output,
                    parse_error=result.error,
                    extra=extra,
                    missing=missing,
                )
            )
    return report


def format_report(report: EvaluationReport) -> str:
    """Readable text: valid-JSON rate, overall and per-kind P/R/F1, unsupported, timing."""
    lines = [
        f"Model version: {report.model_version}",
        f"Records scored: {report.num_records}",
        f"Valid JSON rate: {report.valid_json_rate:.1%}",
        f"Overall: {_scores(report.overall)}",
        "Per kind:",
    ]
    lines += [f"  {kind}: {_scores(counts)}" for kind, counts in sorted(report.per_kind.items())]
    lines += [
        f"Unsupported facts: {report.num_unsupported}",
        f"Omitted facts: {report.overall.false_negatives}",
        f"Total extraction time: {report.total_seconds:.3f} seconds",
        f"Seconds per record: {report.seconds_per_record:.3f}",
    ]
    return "\n".join(lines)


def format_failures(report: EvaluationReport) -> str:
    """Each failed record: note, raw output if invalid, then missing and extra facts.

    Facts are shown as their match fields: kind | mention | assertion | experiencer | temporality.
    """
    total = len(report.failures)
    blocks: list[str] = []
    for number, failure in enumerate(report.failures, start=1):
        lines = [f"[{number}/{total}] record {failure.record_id}", f"  Note: {failure.note}"]
        if failure.parse_error is not None:
            lines.append(f"  Invalid output ({failure.parse_error}): {failure.raw_output}")
        lines += [f"  Missing: {_describe(fact)}" for fact in failure.missing]
        lines += [
            f"  Extra:   {_describe(fact)}"
            + ("" if is_supported(fact, failure.note) else "  (not in note)")
            for fact in failure.extra
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No failures."


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """JSON-ready report: raw counts (to recombine later) plus the scores derived from them.

    failures are left out: they contain note text.
    """
    return {
        "model_version": report.model_version,
        "num_records": report.num_records,
        "num_valid_json": report.num_valid_json,
        "valid_json_rate": report.valid_json_rate,
        "overall": _counts_to_dict(report.overall),
        "per_kind": {kind: _counts_to_dict(c) for kind, c in sorted(report.per_kind.items())},
        "num_unsupported": report.num_unsupported,
        "num_omitted": report.overall.false_negatives,
        "num_failed_records": len(report.failures),
        "total_seconds": report.total_seconds,
        "seconds_per_record": report.seconds_per_record,
    }


def _counts_to_dict(counts: Counts) -> dict[str, Any]:
    return {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
    }


def _describe(fact: Fact) -> str:
    return " | ".join(str(value) for value in fact_key(fact))


def _scores(counts: Counts) -> str:
    """One line of P/R/F1 plus the raw tallies behind them."""
    return (
        f"precision {counts.precision:.1%}, recall {counts.recall:.1%}, F1 {counts.f1:.1%} "
        f"(tp {counts.true_positives}, fp {counts.false_positives}, "
        f"fn {counts.false_negatives})"
    )
