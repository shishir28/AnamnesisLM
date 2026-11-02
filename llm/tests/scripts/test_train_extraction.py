import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from anamnesislm.data.config import PipelineConfig
from anamnesislm.model.config import ModelConfig
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_extraction.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_extraction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_configs_agree() -> None:
    _script()._check_configs(PipelineConfig(), ModelConfig(), BasicTokenizer())


def test_sequence_longer_than_model_context_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="max_sequence_length 512 exceeds model context_length 256"
    ):
        _script()._check_configs(
            PipelineConfig(), ModelConfig(context_length=256), BasicTokenizer()
        )


def test_vocab_mismatch_with_tokenizer_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="vocab_size 1000 does not match tokenizer vocab_size 50257"
    ):
        _script()._check_configs(PipelineConfig(), ModelConfig(vocab_size=1000), BasicTokenizer())


@pytest.mark.parametrize("option", ["--limit", "--epochs"])
def test_non_positive_options_are_rejected(option: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_extraction.py", option, "0"])

    with pytest.raises(SystemExit):
        _script()._parse_args()


def test_options_default_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_extraction.py"])

    args = _script()._parse_args()

    assert (args.limit, args.epochs) == (None, None)
