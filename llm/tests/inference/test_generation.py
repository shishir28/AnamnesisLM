"""Tests for generate() and Extractor.extract(), driven by a scripted stand-in model."""

from typing import cast

import pytest
import torch
from torch import nn

from anamnesislm.inference.extractor import Extractor, build_prompt
from anamnesislm.inference.generate import generate
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer


class _Config:
    def __init__(self, context_length: int) -> None:
        self.context_length = context_length
        self.multi_head = False  # no KV cache: generate re-sends the window each step


class ScriptedModel(nn.Module):
    """Stand-in model that 'writes' a fixed script: call n puts the top logit on script[n].

    Records every input it sees and whether gradients were enabled.
    """

    def __init__(self, script: list[int], vocab_size: int, context_length: int = 64) -> None:
        super().__init__()
        self.script = script
        self.vocab_size = vocab_size
        self.config = _Config(context_length)
        self.anchor = nn.Parameter(torch.zeros(1))  # gives generate() a device to read
        self.inputs: list[list[int]] = []
        self.grad_enabled: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        step = len(self.inputs)
        self.inputs.append(input_ids[0].tolist())
        self.grad_enabled.append(torch.is_grad_enabled())
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        logits[0, -1, self.script[step]] = 1.0
        return logits


def as_gpt(model: nn.Module) -> GPTModel:
    return cast(GPTModel, model)


# generate

GEN_EOS = 0
VOCAB = 20


def _run(
    script: list[int], prompt: list[int], max_new_tokens: int = 10, context_length: int = 64
) -> tuple[list[int], ScriptedModel]:
    model = ScriptedModel(script, VOCAB, context_length)
    return generate(as_gpt(model), prompt, max_new_tokens, GEN_EOS), model


def test_returns_only_new_tokens_and_stops_before_eos() -> None:
    new_ids, _ = _run([7, 8, 9, GEN_EOS, 5], prompt=[1, 2, 3])

    assert new_ids == [7, 8, 9]


def test_eos_first_returns_nothing() -> None:
    new_ids, model = _run([GEN_EOS], prompt=[1, 2, 3])

    assert new_ids == []
    assert len(model.inputs) == 1


def test_stops_after_max_new_tokens() -> None:
    new_ids, model = _run([5] * 10, prompt=[1], max_new_tokens=4)

    assert new_ids == [5, 5, 5, 5]
    assert len(model.inputs) == 4


def test_zero_max_new_tokens_does_not_call_the_model() -> None:
    new_ids, model = _run([5], prompt=[1], max_new_tokens=0)

    assert new_ids == []
    assert model.inputs == []


def test_each_step_sees_the_tokens_written_so_far() -> None:
    _, model = _run([7, 8, GEN_EOS], prompt=[1, 2])

    assert model.inputs == [[1, 2], [1, 2, 7], [1, 2, 7, 8]]


def test_input_is_cut_to_the_last_context_length_tokens() -> None:
    _, model = _run([7, 8, GEN_EOS], prompt=[1, 2, 3, 4, 5], context_length=4)

    assert model.inputs == [[2, 3, 4, 5], [3, 4, 5, 7], [4, 5, 7, 8]]


def test_prompt_list_is_not_changed() -> None:
    prompt = [1, 2, 3]

    _run([7, 8, GEN_EOS], prompt=prompt)

    assert prompt == [1, 2, 3]


def test_runs_without_gradients() -> None:
    _, model = _run([7, GEN_EOS], prompt=[1])

    assert model.grad_enabled == [False, False]


def test_real_model_is_deterministic() -> None:
    torch.manual_seed(0)
    model = GPTModel(ModelConfig(vocab_size=VOCAB, context_length=8, embedding_dim=16)).eval()

    first = generate(model, [1, 2, 3], max_new_tokens=12, eos_token_id=-1)
    second = generate(model, [1, 2, 3], max_new_tokens=12, eos_token_id=-1)

    assert first == second
    assert len(first) == 12


# Extractor.extract

TOKENIZER = BasicTokenizer()
EOS = TOKENIZER.eos_token_id


def _extractor(model_writes: str, max_new_tokens: int = 100) -> tuple[Extractor, ScriptedModel]:
    script = TOKENIZER.encode(model_writes) + [EOS]
    model = ScriptedModel(script, TOKENIZER.vocab_size, context_length=512)
    extractor = Extractor(
        as_gpt(model), TOKENIZER, "cpu", model_version="test@step1", max_new_tokens=max_new_tokens
    )
    return extractor, model


def test_valid_output_becomes_facts() -> None:
    extractor, _ = _extractor('{"facts": [{"kind": "condition", "mention": "fever"}]}')

    result = extractor.extract("Possible fever.")

    [fact] = result.facts or []
    assert (fact["kind"], fact["mention"]) == ("condition", "fever")
    assert result.error is None
    assert result.model_version == "test@step1"


def test_model_receives_the_training_prompt() -> None:
    extractor, model = _extractor('{"facts": []}')

    extractor.extract("Denies chest pain.")

    assert model.inputs[0] == TOKENIZER.encode(build_prompt("Denies chest pain."))


def test_raw_output_is_the_decoded_answer_only() -> None:
    extractor, _ = _extractor('{"facts": []}')

    result = extractor.extract("Denies chest pain.")

    assert result.raw_output == '{"facts": []}'


def test_invalid_output_is_reported_not_raised() -> None:
    extractor, _ = _extractor("this is not json")

    result = extractor.extract("Denies chest pain.")

    assert result.facts is None
    assert result.error is not None
    assert result.raw_output == "this is not json"


def test_output_cut_off_by_max_new_tokens_is_invalid() -> None:
    extractor, _ = _extractor('{"facts": [{"kind": "condition"}]}', max_new_tokens=3)

    result = extractor.extract("Denies chest pain.")

    assert result.facts is None
    assert result.error is not None and "JSONDecodeError" in result.error


# KV cache


def _generate_without_cache(
    model: GPTModel, prompt: list[int], max_new_tokens: int, eos: int
) -> list[int]:
    """The original loop: re-run the last context_length tokens every step."""
    tokens, new_ids = list(prompt), []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = torch.tensor([tokens[-model.config.context_length :]])
            next_id = int(model(window)[0, -1].argmax())
            if next_id == eos:
                break
            tokens.append(next_id)
            new_ids.append(next_id)
    return new_ids


@pytest.mark.parametrize("seed", range(5))
def test_cached_generation_matches_uncached(seed: int) -> None:
    torch.manual_seed(seed)
    model = GPTModel(ModelConfig(vocab_size=30, context_length=32, embedding_dim=16)).eval()
    prompt = torch.randint(1, 30, (6,)).tolist()

    assert generate(model, prompt, 20, eos_token_id=0) == _generate_without_cache(
        model, prompt, 20, eos=0
    )


def test_cached_generation_matches_uncached_past_the_context_length() -> None:
    """Once the sequence outgrows the context, generate falls back to the sliding window."""
    torch.manual_seed(0)
    model = GPTModel(ModelConfig(vocab_size=30, context_length=8, embedding_dim=16)).eval()
    prompt = [3, 1, 4, 1, 5]

    cached = generate(model, prompt, 12, eos_token_id=-1)  # -1: never stop early

    assert len(cached) == 12
    assert cached == _generate_without_cache(model, prompt, 12, eos=-1)


def test_single_head_models_still_generate() -> None:
    torch.manual_seed(0)
    model = GPTModel(
        ModelConfig(vocab_size=30, context_length=32, embedding_dim=16, multi_head=False)
    ).eval()

    assert generate(model, [1, 2, 3], 10, eos_token_id=0) == _generate_without_cache(
        model, [1, 2, 3], 10, eos=0
    )


# extract: normalised input, located evidence


def test_model_reads_the_note_with_normalised_whitespace() -> None:
    extractor, model = _extractor('{"facts": []}')

    extractor.extract("Possible\n  fever.")

    assert TOKENIZER.decode(model.inputs[0]) == build_prompt("Possible fever.")


def test_extract_sets_offsets_into_the_original_note() -> None:
    note = "Patient is a 40-year-old male. The patient\ndenies any fever."
    extractor, _ = _extractor(
        '{"facts": [{"kind": "condition", "mention": "fever", '
        '"evidence": "The patient denies any fever.", "evidence_start": 3, "evidence_end": 9}]}'
    )

    [fact] = extractor.extract(note).facts or []

    assert note[fact["evidence_start"] : fact["evidence_end"]] == fact["evidence"]
    assert fact["evidence"] == "The patient\ndenies any fever."


def test_compact_model_output_is_completed_to_the_full_fact() -> None:
    note = "Patient is a 40-year-old male. Denies fever."
    extractor, _ = _extractor(
        '{"facts": [{"kind": "condition", "mention": "fever", "assertion": "negated", '
        '"experiencer": "patient", "temporality": "current", "evidence": "Denies fever."}]}'
    )

    [fact] = extractor.extract(note).facts or []

    assert note[fact["evidence_start"] : fact["evidence_end"]] == "Denies fever."
    assert (fact["dose"], fact["unit"], fact["route"], fact["frequency"]) == (None,) * 4
