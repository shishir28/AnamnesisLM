import json
from pathlib import Path

import pytest

from anamnesislm.data.records import load_records


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _line(record_id: str, note: str = "Asthma.", **extra: str) -> str:
    return json.dumps(
        {"record_id": record_id, "note": note, "expected_json": '{"facts": []}', **extra}
    )


def test_each_line_becomes_a_record(tmp_path: Path) -> None:
    path = _write(tmp_path / "records.jsonl", [_line("a", "Fever."), _line("b", "Asthma.")])

    records = load_records(path)

    assert [(r.record_id, r.note, r.expected_json) for r in records] == [
        ("a", "Fever.", '{"facts": []}'),
        ("b", "Asthma.", '{"facts": []}'),
    ]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path / "records.jsonl", ["", _line("a"), "   ", _line("b"), ""])

    assert [r.record_id for r in load_records(path)] == ["a", "b"]


def test_extra_fields_are_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path / "records.jsonl", [_line("a", source="synthea")])

    assert load_records(path)[0].record_id == "a"


def test_non_ascii_notes_round_trip(tmp_path: Path) -> None:
    note = "Patient reports naïve café exposure — 37 °C."
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"record_id": "a", "note": note, "expected_json": "{}"}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_records(path)[0].note == note


def test_empty_file_gives_no_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_records(path) == []


def test_missing_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "records.jsonl", [json.dumps({"record_id": "a", "note": "x"})])

    with pytest.raises(KeyError, match="expected_json"):
        load_records(path)


def test_malformed_line_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "records.jsonl", [_line("a"), "{not json"])

    with pytest.raises(json.JSONDecodeError):
        load_records(path)


def test_real_splits_load(tmp_path: Path) -> None:
    split = Path(__file__).resolve().parents[2] / "data/processed/extraction/test/records.jsonl"
    if not split.exists():
        pytest.skip("prepared data not present")

    records = load_records(split)

    assert records
    assert all(r.note and r.record_id for r in records)
