"""Train the extraction model on the prepared JSONL splits.

Flow: load train/validation records -> encode -> DataLoader -> train, checking validation.
The test split is never loaded here; it is used once, by a separate evaluation script.
"""

import json
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import EncodedRecord, ExtractionRecord
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/extraction"


def load_records(path: Path) -> list[ExtractionRecord]:
    """Read one split's records.jsonl into ExtractionRecords. Move to records.py."""
    with path.open("r", encoding="utf-8") as f:
        records: list[ExtractionRecord] = []
        for line in f:
            if not (line := line.strip()):
                continue
            record_dict = json.loads(line)
            record = ExtractionRecord(
                note=record_dict["note"],
                expected_json=record_dict["expected_json"],
                record_id=record_dict["record_id"],
            )
            records.append(record)
    return records


def build_model(config: PipelineConfig) -> nn.Module:
    """Create the transformer. Belongs to model/."""
    raise NotImplementedError


def train(
    model: nn.Module,
    train_loader: DataLoader[EncodedRecord],
    validation_loader: DataLoader[EncodedRecord],
    config: PipelineConfig,
) -> None:
    """Training loop with validation and checkpoints. Belongs to training/."""
    raise NotImplementedError


def main() -> int:
    config = PipelineConfig()
    pipeline = ExtractionDataPipeline(
        config=config, tokenizer=BasicTokenizer()
    )  # Placeholder for tokenizer initialization

    train_records = load_records(DATA_DIR / "train/records.jsonl")
    validation_records = load_records(DATA_DIR / "validation/records.jsonl")

    train_loader = pipeline.build_dataloader(train_records, shuffle=True)
    validation_loader = pipeline.build_dataloader(validation_records, shuffle=False)

    # # model = build_model(config)
    # # train(model, train_loader, validation_loader, config)can
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
