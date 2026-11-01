import pytest
import torch

from anamnesislm.model.attention import CausalSelfAttention
from anamnesislm.model.config import ModelConfig


def test_output_has_the_input_shape(config: ModelConfig) -> None:
    attention = CausalSelfAttention(config)

    assert attention(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_changing_later_tokens_does_not_change_earlier_outputs(config: ModelConfig) -> None:
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 6, 16)
    changed = x.clone()
    changed[:, 4:] = torch.randn(2, 2, 16)

    assert torch.allclose(attention(x)[:, :4], attention(changed)[:, :4], atol=1e-6)
    assert not torch.allclose(attention(x)[:, 4:], attention(changed)[:, 4:])


def test_first_position_attends_only_to_itself(config: ModelConfig) -> None:
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 5, 16)

    assert torch.allclose(attention(x)[:, 0], attention.W_value(x)[:, 0], atol=1e-6)


def test_sequence_longer_than_context_is_rejected(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="exceeds maximum 8"):
        CausalSelfAttention(config)(torch.randn(1, 9, 16))


def test_causal_mask_is_a_buffer_not_a_parameter(config: ModelConfig) -> None:
    attention = CausalSelfAttention(config)

    assert "causal_mask" in dict(attention.named_buffers())
    assert all("causal_mask" not in name for name, _ in attention.named_parameters())


def test_dropout_only_applies_in_training_mode() -> None:
    attention = CausalSelfAttention(ModelConfig(context_length=8, embedding_dim=16, drop_rate=0.5))
    x = torch.randn(2, 6, 16)

    attention.eval()
    assert torch.equal(attention(x), attention(x))
    attention.train()
    assert not torch.equal(attention(x), attention(x))


def test_gradients_reach_every_weight(config: ModelConfig) -> None:
    attention = CausalSelfAttention(config)

    attention(torch.randn(2, 5, 16)).sum().backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in attention.parameters())
