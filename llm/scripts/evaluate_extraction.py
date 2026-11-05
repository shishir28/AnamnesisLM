"""Score a trained checkpoint on a prepared split (the test split by default).

Prints the report and saves it as JSON under experiments/evaluations/, so runs can be compared.
The JSON holds only aggregate counts, never note text, so it is safe to commit.

Example:
    uv run python scripts/evaluate_extraction.py --limit 100
    uv run python scripts/evaluate_extraction.py --show-errors | less
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from anamnesislm.data.records import load_records
from anamnesislm.evaluation import (
    evaluate,
    format_failures,
    format_report,
    remove_overlap,
    report_to_dict,
)
from anamnesislm.inference import Extractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/extraction"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/extraction/best.pt"
EVALUATIONS_DIR = PROJECT_ROOT / "experiments/evaluations"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--split",
        choices=["test", "test_holdout", "test_unseen_names", "validation"],
        default="test",
    )
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="folder holding <split>/records.jsonl, e.g. a frozen copy of the test sets so "
        "versions trained on different data are scored on the same notes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: experiments/evaluations/<time>-<model>-<split>.json)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="print every record the model got wrong (shows note text; not saved to the report)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def _git(*command: str) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_state() -> tuple[str | None, bool | None]:
    """(commit hash, has uncommitted changes); (None, None) when git is unavailable."""
    try:
        return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _display_path(path: Path) -> str:
    """Path relative to the project when inside it, so saved reports don't hold home folders."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT))
    return str(path)


def main() -> int:
    """Load records -> drop notes seen in train -> load the checkpoint -> score -> print, save."""
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = datetime.now(UTC)

    records = load_records(args.data_dir / args.split / "records.jsonl")
    train_records = load_records(DATA_DIR / "train/records.jsonl")
    kept = remove_overlap(records, train_records)
    num_dropped = len(records) - len(kept)
    print(f"Dropped {num_dropped} {args.split} records whose note is also in train")
    kept = kept[: args.limit]

    extractor = Extractor.from_checkpoint(args.checkpoint, device)
    report = evaluate(extractor, kept)
    print(format_report(report))
    if args.show_errors:
        print(f"\nFailed records ({len(report.failures)}):\n")
        print(format_failures(report))

    git_commit, git_dirty = _git_state()
    run: dict[str, Any] = {
        "created_at": started_at.isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint": _display_path(args.checkpoint),
        "split": args.split,
        "data_dir": _display_path(args.data_dir),
        "limit": args.limit,
        "num_dropped_overlap": num_dropped,
        "max_new_tokens": extractor.max_new_tokens,
        "device": device,
        "report": report_to_dict(report),
    }
    output = args.output or EVALUATIONS_DIR / (
        f"{started_at:%Y%m%dT%H%M%SZ}-{report.model_version}-{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"Saved report to {_display_path(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
