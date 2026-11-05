import torch
from torch import nn

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.kv_cache import LayerCache


class CausalSelfAttention(nn.Module):
    """Single-head self-attention: each position sees only itself and earlier positions.

    Used by checkpoints trained before multi-head attention (v1-v3, ModelConfig.multi_head False).
    """

    causal_mask: torch.Tensor  # registered buffer: True above the diagonal (future positions)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_in = config.embedding_dim
        self.d_out = config.embedding_dim
        self.max_seq_len = config.context_length
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=config.qkv_bias)
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=config.qkv_bias)
        self.W_value = nn.Linear(self.d_in, self.d_out, bias=config.qkv_bias)
        self.attention_dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)."""
        _, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        keys = self.W_key(x)  # (batch, seq_len, embedding_dim)
        queries = self.W_query(x)  # (batch, seq_len, embedding_dim)
        values = self.W_value(x)  # (batch, seq_len, embedding_dim)
        attn_scores = queries @ keys.transpose(-2, -1)  # (batch, seq_len, seq_len)

        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len], -torch.inf
        )  # (batch, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        context_vectors = torch.matmul(attn_weights, values)  # (batch, seq_len, embedding_dim)
        return context_vectors


class MultiHeadAttention(nn.Module):
    """Causal self-attention split into config.num_heads heads that attend independently.

    Each head works on head_dim = embedding_dim // num_heads dimensions, so one head can track,
    say, a negation word while another copies the mention. Heads are computed together in one
    matmul, not one module per head; out_proj then mixes their concatenated outputs.
    """

    causal_mask: torch.Tensor  # registered buffer: True above the diagonal (future positions)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                f"embedding_dim {config.embedding_dim} is not divisible by "
                f"num_heads {config.num_heads}"
            )
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.max_seq_len = config.context_length
        d = config.embedding_dim
        self.W_query = nn.Linear(d, d, bias=config.qkv_bias)
        self.W_key = nn.Linear(d, d, bias=config.qkv_bias)
        self.W_value = nn.Linear(d, d, bias=config.qkv_bias)
        self.out_proj = nn.Linear(d, d)
        self.attention_dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1
            ),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, num_heads, seq_len, head_dim)."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim).

        With a cache, x holds only the new positions: their keys and values are appended to the
        cache, and their queries attend to every cached position as well.
        """
        batch, seq_len, embedding_dim = x.shape
        past = 0 if cache is None else cache.length
        total = past + seq_len

        if total > self.max_seq_len:
            raise ValueError(f"Sequence length {total} exceeds maximum {self.max_seq_len}")

        queries = self._split_heads(self.W_query(x))  # (batch, heads, seq_len, head_dim)
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))
        if cache is not None:
            if cache.keys is not None and cache.values is not None:
                keys = torch.cat([cache.keys, keys], dim=2)  # (batch, heads, total, head_dim)
                values = torch.cat([cache.values, values], dim=2)
            cache.keys, cache.values = keys, values

        attn_scores = queries @ keys.transpose(-2, -1)  # (batch, heads, seq_len, total)
        # Rows are the new positions past..total-1; each may see columns up to itself.
        attn_scores = attn_scores.masked_fill(self.causal_mask[past:total, :total], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        context = attn_weights @ values  # (batch, heads, seq_len, head_dim)
        # Put the heads back side by side: (batch, seq_len, heads * head_dim).
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, embedding_dim)
        out: torch.Tensor = self.out_proj(context)
        return out
