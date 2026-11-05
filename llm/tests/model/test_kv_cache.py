from dataclasses import replace

import pytest
import torch

from anamnesislm.model.attention import MultiHeadAttention
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.model.kv_cache import KVCache, LayerCache
from anamnesislm.model.transformer_block import TransformerBlock


def _model(config: ModelConfig) -> GPTModel:
    return GPTModel(config).eval()


# LayerCache / KVCache


def test_new_cache_is_empty() -> None:
    cache = KVCache(num_layers=3)

    assert (len(cache.layers), cache.length) == (3, 0)


def test_cache_length_counts_stored_positions(config: ModelConfig) -> None:
    cache = KVCache(config.num_layers)

    _model(config)(torch.tensor([[1, 2, 3]]), cache)

    assert cache.length == 3
    assert all(layer.length == 3 for layer in cache.layers)


# attention


def test_attention_step_by_step_matches_full_sequence(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config).eval()
    x = torch.randn(2, 6, 16)
    cache = LayerCache()

    steps = [attention(x[:, :3], cache)] + [attention(x[:, i : i + 1], cache) for i in range(3, 6)]

    assert torch.allclose(torch.cat(steps, dim=1), attention(x), atol=1e-6)


def test_attention_cache_overflow_is_rejected(config: ModelConfig) -> None:
    attention = MultiHeadAttention(config)
    cache = LayerCache()
    attention(torch.randn(1, 6, 16), cache)

    with pytest.raises(ValueError, match="Sequence length 9 exceeds maximum 8"):
        attention(torch.randn(1, 3, 16), cache)


def test_single_head_block_rejects_a_cache(config: ModelConfig) -> None:
    block = TransformerBlock(replace(config, multi_head=False))

    with pytest.raises(ValueError, match="needs multi-head attention"):
        block(torch.randn(1, 3, 16), LayerCache())


# GPTModel


def test_model_logits_with_cache_match_without(config: ModelConfig) -> None:
    model = _model(config)
    tokens = torch.tensor([[5, 9, 2, 7, 1, 3, 8]])
    cache = KVCache(config.num_layers)

    prompt = model(tokens[:, :4], cache)
    steps = [model(tokens[:, i : i + 1], cache) for i in range(4, 7)]

    assert torch.allclose(torch.cat([prompt, *steps], dim=1), model(tokens), atol=1e-5)


def test_cached_steps_use_the_right_positions(config: ModelConfig) -> None:
    """Feeding token t at position 5 must differ from feeding it at position 0."""
    model = _model(config)
    cache = KVCache(config.num_layers)
    model(torch.tensor([[1, 2, 3, 4, 5]]), cache)

    at_position_5 = model(torch.tensor([[6]]), cache)

    assert not torch.allclose(at_position_5, model(torch.tensor([[6]])))


def test_model_cache_overflow_is_rejected(config: ModelConfig) -> None:
    model = _model(config)
    cache = KVCache(config.num_layers)
    model(torch.tensor([[1] * 8]), cache)

    with pytest.raises(ValueError, match="Sequence length 9 exceeds model context length 8"):
        model(torch.tensor([[1]]), cache)


def test_forward_without_cache_is_unchanged(config: ModelConfig) -> None:
    model = _model(config)
    tokens = torch.tensor([[4, 4, 2]])

    assert torch.equal(model(tokens), model(tokens, None))
