from dataclasses import replace

import pytest
import torch

from anamnesislm.model.attention import CausalSelfAttention, MultiHeadAttention
from anamnesislm.model.config import ModelConfig


def _manual_heads(attention: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    """Reference: run each head separately on its slice of q/k/v, concatenate, project."""
    seq_len = x.shape[1]
    q, k, v = attention.W_query(x), attention.W_key(x), attention.W_value(x)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    outputs = []
    for h in range(attention.num_heads):
        part = slice(h * attention.head_dim, (h + 1) * attention.head_dim)
        scores = (q[..., part] @ k[..., part].transpose(-2, -1)).masked_fill(mask, -torch.inf)
        weights = torch.softmax(scores / attention.head_dim**0.5, dim=-1)
        outputs.append(weights @ v[..., part])
    projected: torch.Tensor = attention.out_proj(torch.cat(outputs, dim=-1))
    return projected


def test_output_has_the_input_shape(config: ModelConfig) -> None:
    assert MultiHeadAttention(config)(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_head_size_splits_the_embedding(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)

    assert (attention.num_heads, attention.head_dim) == (4, 4)


def test_matches_heads_computed_one_by_one(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)
    x = torch.randn(2, 6, 16)

    assert torch.allclose(attention(x), _manual_heads(attention, x), atol=1e-6)


def test_one_head_with_identity_projection_equals_single_head_attention(
    config: ModelConfig,
) -> None:
    one_head = replace(config, num_heads=1)
    single, multi = CausalSelfAttention(one_head), MultiHeadAttention(one_head)
    with torch.no_grad():
        for name in ("W_query", "W_key", "W_value"):
            getattr(multi, name).weight.copy_(getattr(single, name).weight)
        multi.out_proj.weight.copy_(torch.eye(16))
        multi.out_proj.bias.zero_()
    x = torch.randn(2, 6, 16)

    assert torch.allclose(multi(x), single(x), atol=1e-6)


def test_heads_attend_independently(config: ModelConfig) -> None:
    """Zeroing one head's value slice changes only that head's part of the context."""
    attention = MultiHeadAttention(config)
    with torch.no_grad():
        attention.out_proj.weight.copy_(torch.eye(16))
        attention.out_proj.bias.zero_()
    x = torch.randn(2, 6, 16)
    before = attention(x)
    with torch.no_grad():
        attention.W_value.weight[4:8].zero_()  # head 1's value dimensions

    after = attention(x)

    assert torch.allclose(after[..., 4:8], torch.zeros(2, 6, 4))
    assert torch.allclose(after[..., :4], before[..., :4])
    assert torch.allclose(after[..., 8:], before[..., 8:])


def test_changing_later_tokens_does_not_change_earlier_outputs(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)
    x = torch.randn(2, 6, 16)
    changed = x.clone()
    changed[:, 4:] = torch.randn(2, 2, 16)

    assert torch.allclose(attention(x)[:, :4], attention(changed)[:, :4], atol=1e-6)
    assert not torch.allclose(attention(x)[:, 4:], attention(changed)[:, 4:])


def test_embedding_not_divisible_by_heads_is_rejected(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="not divisible by num_heads 3"):
        MultiHeadAttention(replace(config, num_heads=3))


def test_sequence_longer_than_context_is_rejected(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="exceeds maximum 8"):
        MultiHeadAttention(config)(torch.randn(1, 9, 16))


def test_causal_mask_is_a_buffer_not_a_parameter(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)

    assert "causal_mask" in dict(attention.named_buffers())
    assert all("causal_mask" not in name for name, _ in attention.named_parameters())


def test_dropout_only_applies_in_training_mode() -> None:
    attention = MultiHeadAttention(ModelConfig(context_length=8, embedding_dim=16, drop_rate=0.5))
    x = torch.randn(2, 6, 16)

    attention.eval()
    assert torch.equal(attention(x), attention(x))
    attention.train()
    assert not torch.equal(attention(x), attention(x))


def test_gradients_reach_every_weight(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)

    attention(torch.randn(2, 5, 16)).sum().backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in attention.parameters())
