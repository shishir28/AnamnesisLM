from pathlib import Path

import torch

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.training.trainer import load_checkpoint, save_checkpoint

CONFIG = ModelConfig(vocab_size=50, context_length=8, embedding_dim=16, drop_rate=0.0)


def _trained_model() -> tuple[GPTModel, torch.optim.Optimizer]:
    """A model and optimizer after one step, so the optimizer has state to save."""
    torch.manual_seed(0)
    model = GPTModel(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.autograd.backward(model(torch.randint(0, 50, (2, 6))).sum())
    optimizer.step()
    return model.eval(), optimizer


def _save(path: Path, model: GPTModel, optimizer: torch.optim.Optimizer) -> None:
    save_checkpoint(path, model, optimizer, epoch=1, step=7, validation_loss=2.5, tokenizer="gpt2")


def test_reloaded_model_gives_the_same_logits(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    _save(tmp_path / "best.pt", model, optimizer)

    reloaded = GPTModel(CONFIG).eval()
    load_checkpoint(tmp_path / "best.pt", reloaded)

    tokens = torch.randint(0, 50, (2, 6))
    assert torch.equal(model(tokens), reloaded(tokens))


def test_model_can_be_rebuilt_from_the_saved_config(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    _save(tmp_path / "best.pt", model, optimizer)

    saved_config = torch.load(tmp_path / "best.pt", map_location="cpu")["model_config"]

    assert ModelConfig(**saved_config) == CONFIG


def test_optimizer_state_is_restored(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    _save(tmp_path / "best.pt", model, optimizer)

    reloaded = GPTModel(CONFIG)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-2)
    load_checkpoint(tmp_path / "best.pt", reloaded, reloaded_optimizer)

    original, restored = optimizer.state_dict()["state"], reloaded_optimizer.state_dict()["state"]
    assert original.keys() == restored.keys()
    for key in original:
        assert torch.equal(original[key]["exp_avg"], restored[key]["exp_avg"])
        assert original[key]["step"] == restored[key]["step"]


def test_training_position_and_metadata_are_returned(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    _save(tmp_path / "best.pt", model, optimizer)

    checkpoint = load_checkpoint(tmp_path / "best.pt", GPTModel(CONFIG))

    assert (checkpoint["epoch"], checkpoint["step"]) == (1, 7)
    assert checkpoint["validation_loss"] == 2.5
    assert checkpoint["tokenizer"] == "gpt2"


def test_save_creates_the_folder_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    path = tmp_path / "nested" / "run" / "best.pt"

    _save(path, model, optimizer)

    assert [p.name for p in path.parent.iterdir()] == ["best.pt"]


def test_saving_again_replaces_the_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _trained_model()
    _save(tmp_path / "best.pt", model, optimizer)

    save_checkpoint(tmp_path / "best.pt", model, optimizer, epoch=2, step=9, validation_loss=1.0)

    assert load_checkpoint(tmp_path / "best.pt", GPTModel(CONFIG))["step"] == 9


def test_other_file_names_are_kept(tmp_path: Path) -> None:
    model, optimizer = _trained_model()

    _save(tmp_path / "last.pt", model, optimizer)

    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]
