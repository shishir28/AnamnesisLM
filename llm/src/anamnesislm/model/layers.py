import math

import torch
from torch import nn

from anamnesislm.model.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise MLP: expand to 4 x embedding_dim, GELU, project back."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        embedding_dim = config.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)."""
        out: torch.Tensor = self.layers(x)
        return out


class GELU(nn.Module):
    """GELU activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
