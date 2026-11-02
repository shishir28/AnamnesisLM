from pathlib import Path

import pytest
import torch

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import ExtractionRecord
from anamnesislm.inference.extractor import Extractor, build_prompt, parse_facts
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.training.trainer import save_checkpoint

# build_prompt


@pytest.mark.parametrize(
    "note",
    [
        "Patient is a 54-year-old male. Denies chest pain.",
        "",
        "Line one.\nLine two.",
    ],
)
def test_prompt_matches_the_training_prompt(note: str) -> None:
    pipeline = ExtractionDataPipeline(PipelineConfig(), BasicTokenizer())
    training_prompt, _ = pipeline.serialize_record(
        ExtractionRecord(note=note, expected_json="{}", record_id="r1")
    )

    assert build_prompt(note) == training_prompt


# parse_facts


def test_valid_facts_are_returned() -> None:
    result = parse_facts('{"facts": [{"kind": "condition", "mention": "Asthma"}]}', "v1")

    assert result.facts == [{"kind": "condition", "mention": "Asthma"}]
    assert result.error is None
    assert result.model_version == "v1"


def test_empty_facts_list_is_valid() -> None:
    result = parse_facts('{"facts": []}', "v1")

    assert result.facts == []
    assert result.error is None


@pytest.mark.parametrize(
    ("raw_output", "error_fragment"),
    [
        ("not json", "JSONDecodeError"),
        ('{"facts": [', "JSONDecodeError"),
        ('{"other": 1}', '"facts" key'),
        ("[1, 2]", "not a JSON object"),
        ("null", "not a JSON object"),
        ('{"facts": "x"}', '"facts"'),
        ('{"facts": [1, "x"]}', '"facts"'),
    ],
)
def test_invalid_output_is_reported_not_raised(raw_output: str, error_fragment: str) -> None:
    result = parse_facts(raw_output, "v1")

    assert result.facts is None
    assert result.error is not None and error_fragment in result.error
    assert result.raw_output == raw_output
    assert result.model_version == "v1"


# Extractor.from_checkpoint

CONFIG = ModelConfig(vocab_size=50257, context_length=16, embedding_dim=16, num_layers=1)


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, GPTModel]:
    torch.manual_seed(0)
    model = GPTModel(CONFIG).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best.pt"
    save_checkpoint(
        path, model, optimizer, epoch=2, step=123, validation_loss=1.5, tokenizer="gpt2"
    )
    return path, model


def test_loaded_model_gives_the_same_logits(checkpoint: tuple[Path, GPTModel]) -> None:
    path, original = checkpoint

    extractor = Extractor.from_checkpoint(path, "cpu")

    tokens = torch.randint(0, 50257, (1, 8))
    assert torch.equal(extractor.model(tokens), original(tokens))


def test_loaded_model_is_in_eval_mode_on_the_requested_device(
    checkpoint: tuple[Path, GPTModel],
) -> None:
    extractor = Extractor.from_checkpoint(checkpoint[0], "cpu")

    assert not extractor.model.training
    assert next(extractor.model.parameters()).device == torch.device("cpu")
    assert extractor.device == "cpu"


def test_model_version_names_the_file_and_step(checkpoint: tuple[Path, GPTModel]) -> None:
    assert Extractor.from_checkpoint(checkpoint[0], "cpu").model_version == "best.pt@step123"


def test_tokenizer_comes_from_the_checkpoint(checkpoint: tuple[Path, GPTModel]) -> None:
    tokenizer = Extractor.from_checkpoint(checkpoint[0], "cpu").tokenizer

    assert tokenizer.vocab_size == 50257
    assert tokenizer.decode(tokenizer.encode("Denies fever.")) == "Denies fever."
