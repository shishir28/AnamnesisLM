import torch

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.layers import GELU, FeedForward


def test_gelu_matches_pytorch_tanh_approximation() -> None:
    x = torch.linspace(-6, 6, 241)

    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(GELU()(x), expected, atol=1e-6)


def test_gelu_limits() -> None:
    gelu = GELU()

    assert gelu(torch.tensor(0.0)).item() == 0.0
    assert torch.isclose(gelu(torch.tensor(10.0)), torch.tensor(10.0))
    assert abs(gelu(torch.tensor(-10.0)).item()) < 1e-6


def test_feed_forward_keeps_the_shape(config: ModelConfig) -> None:
    assert FeedForward(config)(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_feed_forward_hidden_layer_is_four_times_wider(config: ModelConfig) -> None:
    shapes = [tuple(p.shape) for p in FeedForward(config).parameters()]

    assert shapes == [(64, 16), (64,), (16, 64), (16,)]


def test_feed_forward_treats_each_position_independently(config: ModelConfig) -> None:
    feed_forward = FeedForward(config)
    x = torch.randn(1, 5, 16)
    order = torch.tensor([4, 2, 0, 3, 1])

    assert torch.allclose(feed_forward(x[:, order]), feed_forward(x)[:, order], atol=1e-6)
