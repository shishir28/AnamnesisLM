"""Prepare extraction training data from a Synthea FHIR cohort.

Reads ``<cohort>/<State>/fhir/*.json`` one bundle at a time, assigns each patient to a
split by hashing their id, renders a note and expected facts, and writes:

    data/processed/extraction/
    ├── train/records.jsonl
    ├── validation/records.jsonl
    └── test/records.jsonl

Training reads only these JSONL files, never the raw FHIR folders.
"""

import argparse
import json
from pathlib import Path

import msgspec

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.record_builder import build_record
from anamnesislm.data.splits import SPLITS, assign_split
from anamnesislm.data.synthea import iter_bundle_files, read_patient_bundle
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHEA_DIR = PROJECT_ROOT / "data/raw/synthea"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/extraction"
# Room for the instruction prompt and EOS added at training time; replace with the
# real prompt length once processor.py defines it.
PROMPT_TOKEN_RESERVE = 32


def _latest_cohort() -> Path:
    cohorts = sorted(SYNTHEA_DIR.glob("cohort-*"), key=lambda path: path.stat().st_mtime)
    if not cohorts:
        raise SystemExit(f"No cohort-* directory in {SYNTHEA_DIR}; run tools/generate_synthea.sh.")
    return cohorts[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort-dir", type=Path, help="defaults to the newest cohort-* directory")
    cohort_dir: Path = parser.parse_args().cohort_dir or _latest_cohort()
    print(f"Reading {cohort_dir}")

    tokenizer = BasicTokenizer()
    token_limit = PipelineConfig().max_sequence_length - PROMPT_TOKEN_RESERVE
    counts = dict.fromkeys(SPLITS, 0)
    skipped = 0
    too_long = 0

    for split in SPLITS:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    with (
        (OUTPUT_DIR / "train/records.jsonl").open("w", encoding="utf-8") as train_file,
        (OUTPUT_DIR / "validation/records.jsonl").open("w", encoding="utf-8") as validation_file,
        (OUTPUT_DIR / "test/records.jsonl").open("w", encoding="utf-8") as test_file,
    ):
        writers = {"train": train_file, "validation": validation_file, "test": test_file}

        for path in iter_bundle_files(cohort_dir):
            try:
                patient_bundle = read_patient_bundle(path)
            except (OSError, msgspec.DecodeError, ValueError) as e:
                print(f"Could not read {path}: {e}")
                skipped += 1
                continue
            if patient_bundle is None:  # hospital / practitioner information files
                continue

            try:
                record = build_record(patient_bundle)
            except ValueError as e:  # e.g. missing birth date or gender
                print(f"Could not build record for {path}: {e}")
                skipped += 1
                continue
            # Longer records would be truncated in training, cutting off the JSON.
            if len(tokenizer.encode(record.note + record.expected_json)) > token_limit:
                too_long += 1
                continue
            split = assign_split(patient_bundle.patient_id)
            line = {
                "record_id": record.record_id,
                "note": record.note,
                "expected_json": record.expected_json,
            }
            writers[split].write(json.dumps(line) + "\n")
            counts[split] += 1

    print(f"Records per split: {counts}; skipped files: {skipped}")
    print(f"Records over {token_limit} tokens, dropped: {too_long}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
