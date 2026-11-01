import pytest
import torch

from anamnesislm.model.config import ModelConfig


@pytest.fixture
def config() -> ModelConfig:
    """Tiny model, dropout off, so outputs are deterministic and checks are exact."""
    return ModelConfig(vocab_size=50, context_length=8, embedding_dim=16, drop_rate=0.0)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
