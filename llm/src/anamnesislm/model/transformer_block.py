import torch
from torch import nn

from anamnesislm.model.attention import CausalSelfAttention, MultiHeadAttention
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.kv_cache import LayerCache
from anamnesislm.model.layers import FeedForward


class TransformerBlock(nn.Module):
    """LayerNorm -> attention -> residual, then LayerNorm -> feed-forward -> residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention: nn.Module = (
            MultiHeadAttention(config) if config.multi_head else CausalSelfAttention(config)
        )
        self.feedforward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim).

        cache: this block's KV cache during generation; only multi-head attention supports it.
        """
        shortcut = x
        x = self.layer_norm1(x)
        if cache is None:
            x = self.attention(x)
        elif isinstance(self.attention, MultiHeadAttention):
            x = self.attention(x, cache)
        else:
            raise ValueError("KV cache needs multi-head attention (ModelConfig.multi_head)")
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
