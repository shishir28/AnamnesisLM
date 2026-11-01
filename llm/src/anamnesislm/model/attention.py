import torch
from torch import nn

from anamnesislm.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Self-attention where each position sees only itself and earlier positions.

    Single-head for now; config.num_heads is not used yet.
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


# class MultiHeadCausalSelfAttention(nn.Module):
#     """Multi-head self-attention where each position sees only itself and earlier positions."""

#     def __init__(self, config: ModelConfig) -> None:
#         super().__init__()
#         self.num_heads = config.num_heads
#         self.head_dim = config.embedding_dim // config.num_heads
#         self.attention_heads = nn.ModuleList(
#             [CausalSelfAttention(config) for _ in range(self.num_heads)]
#         )
#         self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
#         self.dropout = nn.Dropout(config.dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """(batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)."""
#         head_outputs = [head(x) for head in self.attention_heads]
#         concatenated = torch.cat(head_outputs, dim=-1)
#         output = self.output_projection(concatenated)
#         output = self.dropout(output)
#         return output
