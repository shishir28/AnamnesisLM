import torch
from torch import nn

from anamnesislm.model.attention import CausalSelfAttention
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.layers import FeedForward


class TransformerBlock(nn.Module):
    """LayerNorm -> attention -> residual, then LayerNorm -> feed-forward -> residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention = CausalSelfAttention(config)
        self.feedforward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)."""
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
