import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from anamnesislm.inference.extractor import ExtractionResult, parse_facts

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_extraction.py"

FACT = {
    "kind": "condition",
    "mention": "Asthma",
    "assertion": "affirmed",
    "experiencer": "patient",
    "temporality": "current",
}
FACTS_JSON = json.dumps({"facts": [FACT]})


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_extraction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeExtractor:
    """Answers every note with FACTS_JSON and records what it was asked."""

    model_version = "fake.pt@step1"
    max_new_tokens = 400
    loaded_from: list[tuple[Path, str]] = []
    notes: list[str] = []

    @classmethod
    def from_checkpoint(cls, path: Path, device: str) -> "FakeExtractor":
        cls.loaded_from.append((path, device))
        return cls()

    def extract(self, note: str) -> ExtractionResult:
        self.notes.append(note)
        return parse_facts(FACTS_JSON, self.model_version)


def _write_split(data_dir: Path, split: str, notes: list[str]) -> None:
    (data_dir / split).mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"record_id": f"{split}-{i}", "note": note, "expected_json": FACTS_JSON})
        for i, note in enumerate(notes)
    ]
    (data_dir / split / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script with data and reports under tmp_path and the real Extractor swapped for a fake."""
    module = _script()
    _write_split(tmp_path, "train", ["Seen in train."])
    _write_split(tmp_path, "test", ["Seen in train.", "Test one.", "Test two.", "Test three."])
    _write_split(tmp_path, "validation", ["Validation one."])
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "EVALUATIONS_DIR", tmp_path / "evaluations")
    monkeypatch.setattr(module, "Extractor", FakeExtractor)
    FakeExtractor.loaded_from = []
    FakeExtractor.notes = []
    return module


def _run(module: ModuleType, monkeypatch: pytest.MonkeyPatch, *argv: str) -> Any:
    monkeypatch.setattr("sys.argv", ["evaluate_extraction.py", *argv])
    return module.main()


# _parse_args


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate_extraction.py"])

    args = _script()._parse_args()

    assert args.split == "test"
    assert args.limit is None
    assert args.checkpoint.name == "best.pt"


@pytest.mark.parametrize("value", ["0", "-3"])
def test_non_positive_limit_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate_extraction.py", "--limit", value])

    with pytest.raises(SystemExit):
        _script()._parse_args()


def test_train_split_cannot_be_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate_extraction.py", "--split", "train"])

    with pytest.raises(SystemExit):
        _script()._parse_args()


# main


def test_scores_test_notes_not_seen_in_train(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(script, monkeypatch) == 0

    out = capsys.readouterr().out
    assert FakeExtractor.notes == ["Test one.", "Test two.", "Test three."]
    assert "Dropped 1 test records whose note is also in train" in out
    assert "Records scored: 3" in out
    assert "Overall: precision 100.0%, recall 100.0%, F1 100.0% (tp 3, fp 0, fn 0)" in out


def test_limit_applies_after_dropping_overlap(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(script, monkeypatch, "--limit", "2")

    assert FakeExtractor.notes == ["Test one.", "Test two."]


def test_validation_split_can_be_scored(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(script, monkeypatch, "--split", "validation")

    assert FakeExtractor.notes == ["Validation one."]
    assert "Dropped 0 validation records" in capsys.readouterr().out


def test_checkpoint_option_is_passed_to_the_extractor(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(script, monkeypatch, "--checkpoint", str(tmp_path / "other.pt"))

    [(path, device)] = FakeExtractor.loaded_from
    assert path == tmp_path / "other.pt"
    assert device in {"cuda", "cpu"}


def test_missing_split_file_fails_loudly(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "test/records.jsonl").unlink()

    with pytest.raises(FileNotFoundError):
        _run(script, monkeypatch)


# saved report


def _saved(tmp_path: Path) -> dict[str, Any]:
    [path] = (tmp_path / "evaluations").glob("*.json")
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def test_report_is_saved_with_run_details(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(script, monkeypatch, "--limit", "2")

    saved = _saved(tmp_path)
    assert saved["split"] == "test"
    assert saved["limit"] == 2
    assert saved["num_dropped_overlap"] == 1
    assert saved["max_new_tokens"] == 400
    assert saved["report"]["num_records"] == 2
    assert saved["report"]["overall"]["true_positives"] == 2
    assert set(saved) >= {"created_at", "git_commit", "git_dirty", "checkpoint", "device"}


def test_default_file_name_has_time_model_and_split(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(script, monkeypatch, "--split", "validation")

    [path] = (tmp_path / "evaluations").glob("*.json")
    assert path.name.endswith("Z-fake.pt@step1-validation.json")


def test_output_option_chooses_the_path(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "nested/run.json"

    _run(script, monkeypatch, "--output", str(output))

    assert json.loads(output.read_text(encoding="utf-8"))["report"]["num_records"] == 3
    assert not (tmp_path / "evaluations").exists()


def test_saved_report_holds_no_note_text(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(script, monkeypatch)

    text = next((tmp_path / "evaluations").glob("*.json")).read_text(encoding="utf-8")
    assert "Test one." not in text
    assert "Asthma" not in text


def test_checkpoint_inside_project_is_saved_as_relative_path(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(script, monkeypatch)

    assert _saved(tmp_path)["checkpoint"] == "checkpoints/extraction/best.pt"


def test_git_state_is_none_when_git_fails(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(script.subprocess, "run", fail)

    assert script._git_state() == (None, None)


# --show-errors


def test_failures_are_printed_only_with_show_errors(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_split(script.DATA_DIR, "validation", ["Note without asthma fact."])
    monkeypatch.setattr(
        FakeExtractor, "extract", lambda self, note: parse_facts("bad", self.model_version)
    )

    _run(script, monkeypatch, "--split", "validation")
    assert "Failed records" not in capsys.readouterr().out

    _run(script, monkeypatch, "--split", "validation", "--show-errors")
    out = capsys.readouterr().out
    assert "Failed records (1):" in out
    assert "Note: Note without asthma fact." in out
