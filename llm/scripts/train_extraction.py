"""Train the extraction model on the prepared JSONL splits.

Flow: load train/validation records -> encode -> DataLoader -> train, checking validation.
The test split is never loaded here; it is used once, by a separate evaluation script.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import load_records
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.tokenization.tokenizer import Tokenizer
from anamnesislm.training.config import TrainingConfig
from anamnesislm.training.trainer import train as trainer_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/extraction"


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
    parser.add_argument("--seed", type=int, help="override TrainingConfig.seed (default 42)")
    parser.add_argument("--num-layers", type=int, help="override ModelConfig.num_layers")
    parser.add_argument(
        "--compact-response",
        action="store_true",
        help="train the model to write only output_format.MODEL_FIELDS",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    if args.num_layers is not None and args.num_layers < 1:
        parser.error("--num-layers must be a positive integer")
    return args


def main() -> int:

    start_time = time.perf_counter()

    args = _parse_args()

    pipeline_config = PipelineConfig(compact_response=args.compact_response)
    tokenizer = BasicTokenizer()
    pipeline = ExtractionDataPipeline(config=pipeline_config, tokenizer=tokenizer)

    model_config = ModelConfig()
    if args.num_layers is not None:
        model_config = replace(model_config, num_layers=args.num_layers)
    _check_configs(pipeline_config, model_config, tokenizer)

    train_records = load_records(DATA_DIR / "train/records.jsonl")[: args.limit]
    validation_records = load_records(DATA_DIR / "validation/records.jsonl")[: args.limit]

    train_loader = pipeline.build_dataloader(train_records, shuffle=True)
    validation_loader = pipeline.build_dataloader(validation_records, shuffle=False)

    checkpoint_dir = PROJECT_ROOT / "checkpoints/extraction"
    training_config = TrainingConfig(checkpoint_dir=checkpoint_dir)
    if args.epochs is not None:
        training_config = replace(training_config, num_epochs=args.epochs)
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    # Seed before building the model, so its initial weights are reproducible too.
    torch.manual_seed(training_config.seed)
    model = GPTModel(config=model_config)
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
