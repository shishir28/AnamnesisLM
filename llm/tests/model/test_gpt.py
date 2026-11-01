import pytest
import torch
from torch.nn import functional

from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel


def _tokens(batch: int, seq_len: int) -> torch.Tensor:
    return torch.randint(0, 50, (batch, seq_len))


def test_logits_shape(config: ModelConfig) -> None:
    logits = GPTModel(config)(_tokens(2, 5))

    assert logits.shape == (2, 5, 50)
    assert logits.dtype == torch.float32


def test_full_context_is_accepted(config: ModelConfig) -> None:
    assert GPTModel(config)(_tokens(1, 8)).shape == (1, 8, 50)


def test_sequence_longer_than_context_is_rejected(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="exceeds model context length 8"):
        GPTModel(config)(_tokens(1, 9))


def test_changing_later_tokens_does_not_change_earlier_logits(config: ModelConfig) -> None:
    model = GPTModel(config)
    tokens = _tokens(2, 6)
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 50

    assert torch.allclose(model(tokens)[:, :4], model(changed)[:, :4], atol=1e-5)
    assert not torch.allclose(model(tokens)[:, 4:], model(changed)[:, 4:])


def test_position_changes_the_logits(config: ModelConfig) -> None:
    logits = GPTModel(config)(torch.full((1, 3), 7))

    assert not torch.allclose(logits[0, 0], logits[0, 2])


def test_output_head_shares_the_token_embedding(config: ModelConfig) -> None:
    model = GPTModel(config)

    assert model.out_head.weight is model.token_embedding.weight


def test_num_parameters_counts_the_shared_table_once(config: ModelConfig) -> None:
    # Per block: attention 3 x 16 x 16 = 768, feed-forward 16 x 64 + 64 + 64 x 16 + 16 = 2128,
    # two LayerNorms 2 x 32 = 64 -> 2960; four blocks 11840.
    # Token table 50 x 16 = 800, positions 8 x 16 = 128, final LayerNorm 32. Head is tied.
    assert GPTModel(config).num_parameters() == 11840 + 800 + 128 + 32


def test_dropout_only_applies_in_training_mode() -> None:
    model = GPTModel(ModelConfig(vocab_size=50, context_length=8, embedding_dim=16, drop_rate=0.5))
    tokens = _tokens(2, 6)

    model.eval()
    assert torch.equal(model(tokens), model(tokens))
    model.train()
    assert not torch.equal(model(tokens), model(tokens))


def test_gradients_reach_every_weight(config: ModelConfig) -> None:
    model = GPTModel(config)

    model(_tokens(2, 5)).sum().backward()

    assert all(p.grad is not None for p in model.parameters())


def test_can_memorise_one_batch(config: ModelConfig) -> None:
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = _tokens(4, 8)

    def loss() -> torch.Tensor:
        logits = model(tokens[:, :-1])
        return functional.cross_entropy(logits.reshape(-1, 50), tokens[:, 1:].reshape(-1))

    initial = loss().item()
    for _ in range(50):
        optimizer.zero_grad()
        step_loss = loss()
        torch.autograd.backward(step_loss)
        optimizer.step()

    assert loss().item() < 0.1 * initial
