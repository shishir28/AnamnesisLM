"""Train the extraction model on the prepared JSONL splits.

Flow: load train/validation records -> encode -> DataLoader -> train, checking validation.
The test split is never loaded here; it is used once, by a separate evaluation script.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import ExtractionRecord
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.tokenization.tokenizer import Tokenizer
from anamnesislm.training.config import TrainingConfig
from anamnesislm.training.trainer import train as trainer_train

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


def _check_configs(
    pipeline_config: PipelineConfig, model_config: ModelConfig, tokenizer: Tokenizer
) -> None:
    """Fail before training if data, tokenizer, and model settings disagree."""
    if pipeline_config.max_sequence_length > model_config.context_length:
        raise ValueError(
            f"Pipeline max_sequence_length {pipeline_config.max_sequence_length} "
            f"exceeds model context_length {model_config.context_length}"
        )
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Model vocab_size {model_config.vocab_size} "
            f"does not match tokenizer vocab_size {tokenizer.vocab_size}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        help="use only the first N train and validation records (e.g. 16 for the overfit check)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="override TrainingConfig.num_epochs (e.g. 100 with --limit 16)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    return args


def main() -> int:

    start_time = time.perf_counter()

    args = _parse_args()

    pipeline_config = PipelineConfig()
    tokenizer = BasicTokenizer()
    pipeline = ExtractionDataPipeline(config=pipeline_config, tokenizer=tokenizer)

    model_config = ModelConfig()
    _check_configs(pipeline_config, model_config, tokenizer)

    train_records = load_records(DATA_DIR / "train/records.jsonl")[: args.limit]
    validation_records = load_records(DATA_DIR / "validation/records.jsonl")[: args.limit]

    train_loader = pipeline.build_dataloader(train_records, shuffle=True)
    validation_loader = pipeline.build_dataloader(validation_records, shuffle=False)

    model = GPTModel(config=model_config)
    checkpoint_dir = PROJECT_ROOT / "checkpoints/extraction"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    if args.epochs is not None:
        training_config = replace(training_config, num_epochs=args.epochs)
    trainer_train(
        model,
        train_loader,
        validation_loader,
        training_config,
        ignore_index=pipeline_config.ignore_index,
    )
    print(f"Training completed in {time.perf_counter() - start_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
