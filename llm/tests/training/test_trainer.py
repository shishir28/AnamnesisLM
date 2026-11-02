import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from anamnesislm.data.dataset import EncodedDataset
from anamnesislm.data.records import EncodedRecord
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.training.config import TrainingConfig
from anamnesislm.training.trainer import compute_loss, evaluate_model, load_checkpoint, train

VOCAB = 20
IGNORE = -100
EOS = 0


class NextTokenOracle(nn.Module):
    """Stand-in model whose logits always point at the true next input token.

    Counts calls and records whether gradients were enabled, for evaluate_model checks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.grad_enabled: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        next_tokens = torch.roll(input_ids, shifts=-1, dims=1)
        return functional.one_hot(next_tokens, VOCAB).float() * 100.0


def _oracle() -> NextTokenOracle:
    return NextTokenOracle()


def _as_gpt(model: nn.Module) -> GPTModel:
    return cast(GPTModel, model)


def _record(prompt: list[int], response: list[int]) -> EncodedRecord:
    response = response + [EOS]
    return EncodedRecord(
        input_ids=prompt + response,
        labels=[IGNORE] * len(prompt) + response,
        loss_mask=[False] * len(prompt) + [True] * len(response),
    )


def _collate(records: list[EncodedRecord]) -> dict[str, torch.Tensor]:
    width = max(len(r.input_ids) for r in records)

    def pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (width - len(values))

    return {
        "input_ids": torch.tensor([pad(r.input_ids, EOS) for r in records]),
        "labels": torch.tensor([pad(r.labels, IGNORE) for r in records]),
    }


def _loader(records: list[EncodedRecord], batch_size: int = 2) -> DataLoader[EncodedRecord]:
    return DataLoader(EncodedDataset(records), batch_size=batch_size, collate_fn=_collate)


def _records(count: int) -> list[EncodedRecord]:
    """Fixed prompt/response pairs; the response repeats a pattern a tiny model can memorise."""
    return [_record([1, 2, 3 + i % 4], [5 + i % 3, 9, 10, 11]) for i in range(count)]


def _config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    settings: dict[str, object] = {
        "device": "cpu",
        "checkpoint_dir": tmp_path,
        "learning_rate": 1e-2,
        "eval_freq": 10_000,
        "eval_iter": 10,
    }
    settings.update(overrides)
    return TrainingConfig(**settings)  # type: ignore[arg-type]


def _tiny_model() -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(
        ModelConfig(
            vocab_size=VOCAB, context_length=16, embedding_dim=32, num_layers=2, drop_rate=0.0
        )
    )


# compute_loss


def test_loss_compares_each_position_with_the_next_label() -> None:
    batch = _collate([_record([1, 2], [3, 4])])

    loss = compute_loss(_as_gpt(_oracle()), batch, ignore_index=IGNORE)

    assert loss.item() < 1e-6


def test_loss_matches_manual_shifted_cross_entropy() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    batch = _collate([_record([1, 2, 3], [4, 5]), _record([1], [6, 7, 8])])

    logits = model(batch["input_ids"])
    expected = functional.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), batch["labels"][:, 1:].reshape(-1), ignore_index=IGNORE
    )

    assert torch.allclose(compute_loss(model, batch, ignore_index=IGNORE), expected)


def test_prompt_and_padding_positions_do_not_count() -> None:
    model = _tiny_model()
    batch = _collate([_record([1, 2, 3], [4, 5]), _record([1], [6])])
    only_last_response = batch["labels"].clone()
    only_last_response[:, :] = IGNORE
    only_last_response[0, 4] = 5

    loss = compute_loss(model, {**batch, "labels": only_last_response}, ignore_index=IGNORE)
    logits = model(batch["input_ids"])

    assert torch.allclose(loss, functional.cross_entropy(logits[0, 3:4], torch.tensor([5])))


# evaluate_model


def test_evaluation_stops_after_max_batches(tmp_path: Path) -> None:
    model = _oracle()

    evaluate_model(_as_gpt(model), _loader(_records(10)), _config(tmp_path), IGNORE, max_batches=2)

    assert model.calls == 2


def test_evaluation_without_max_batches_uses_the_whole_loader(tmp_path: Path) -> None:
    model = _oracle()

    evaluate_model(_as_gpt(model), _loader(_records(10)), _config(tmp_path), IGNORE)

    assert model.calls == 5


def test_evaluation_runs_without_gradients(tmp_path: Path) -> None:
    model = _oracle()

    evaluate_model(_as_gpt(model), _loader(_records(4)), _config(tmp_path), IGNORE)

    assert model.grad_enabled == [False, False]


@pytest.mark.parametrize("training", [True, False])
def test_evaluation_restores_the_previous_mode(tmp_path: Path, training: bool) -> None:
    model = _oracle()
    model.train(training)

    evaluate_model(_as_gpt(model), _loader(_records(4)), _config(tmp_path), IGNORE)

    assert model.training is training


def test_evaluation_with_no_batches_returns_nan(tmp_path: Path) -> None:
    loss = evaluate_model(
        _as_gpt(_oracle()), _loader(_records(4)), _config(tmp_path), IGNORE, max_batches=0
    )

    assert math.isnan(loss)


# train


def test_training_memorises_a_tiny_dataset(tmp_path: Path) -> None:
    model = _tiny_model()
    loader = _loader(_records(8), batch_size=4)
    config = _config(tmp_path, num_epochs=40)
    before = evaluate_model(model, loader, config, IGNORE)

    train(model, loader, loader, config, ignore_index=IGNORE)

    assert evaluate_model(model, loader, config, IGNORE) < 0.2 * before


def test_end_of_epoch_evaluation_saves_a_checkpoint(tmp_path: Path) -> None:
    model = _tiny_model()
    loader = _loader(_records(4), batch_size=2)

    train(model, loader, loader, _config(tmp_path, num_epochs=3), ignore_index=IGNORE)

    checkpoint = load_checkpoint(tmp_path / "best.pt", _tiny_model())
    assert checkpoint["epoch"] == 2
    assert checkpoint["step"] == 6


def test_periodic_and_end_of_epoch_evaluations_are_logged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _tiny_model()
    loader = _loader(_records(4), batch_size=2)

    train(model, loader, loader, _config(tmp_path, num_epochs=2, eval_freq=1), ignore_index=IGNORE)

    lines = [line for line in capsys.readouterr().out.splitlines() if "validation loss" in line]
    assert len(lines) == 4 + 2  # one per step, plus one at the end of each epoch
    assert all("train loss" in line for line in lines)
