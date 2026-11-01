import torch
from torch import nn

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.transformer_block import TransformerBlock


def test_output_has_the_input_shape(config: ModelConfig) -> None:
    assert TransformerBlock(config)(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_block_is_causal(config: ModelConfig) -> None:
    block = TransformerBlock(config)
    x = torch.randn(2, 6, 16)
    changed = x.clone()
    changed[:, 4:] = torch.randn(2, 2, 16)

    assert torch.allclose(block(x)[:, :4], block(changed)[:, :4], atol=1e-5)


def test_residuals_pass_the_input_through_when_sublayers_output_zero(
    config: ModelConfig,
) -> None:
    block = TransformerBlock(config)
    last_feed_forward_layer = block.feedforward.layers[-1]
    assert isinstance(last_feed_forward_layer, nn.Linear)
    with torch.no_grad():
        block.attention.W_value.weight.zero_()
        last_feed_forward_layer.weight.zero_()
        last_feed_forward_layer.bias.zero_()
    x = torch.randn(2, 5, 16)

    assert torch.allclose(block(x), x)


def test_gradients_reach_every_weight(config: ModelConfig) -> None:
    block = TransformerBlock(config)

    block(torch.randn(2, 5, 16)).sum().backward()

    assert all(p.grad is not None for p in block.parameters())
