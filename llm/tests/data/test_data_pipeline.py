from collections.abc import Sequence

import pytest
import torch

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import EncodedRecord, ExtractionRecord
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer

EOS = 0


class CharTokenizer:
    """One token per character, so expected ids are easy to write by hand."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)

    @property
    def vocab_size(self) -> int:
        return 0x110000

    @property
    def eos_token_id(self) -> int:
        return EOS


def _pipeline(**config: int) -> ExtractionDataPipeline:
    return ExtractionDataPipeline(PipelineConfig(**config), CharTokenizer())


def _record(note: str = "Denies fever.", record_id: str = "r1") -> ExtractionRecord:
    return ExtractionRecord(note=note, expected_json='{"facts": []}', record_id=record_id)


def _encoded(length: int) -> EncodedRecord:
    """A record with a 2-token prompt and (length - 2) response tokens."""
    response = list(range(10, 10 + length - 2))
    return EncodedRecord(
        input_ids=[1, 2] + response,
        labels=[-100, -100] + response,
        loss_mask=[False, False] + [True] * len(response),
    )


# serialize_record


def test_prompt_holds_the_note_and_response_is_the_expected_json() -> None:
    prompt, response = _pipeline().serialize_record(_record("Denies fever."))

    assert "Denies fever." in prompt
    assert prompt.endswith("Response:")
    assert response == '{"facts": []}'


# encode_record


def test_input_ids_are_prompt_then_response_then_one_eos() -> None:
    encoded = _pipeline().encode_record("ab", "xyz")

    assert encoded.input_ids == [ord("a"), ord("b"), ord("x"), ord("y"), ord("z"), EOS]
    assert encoded.input_ids.count(EOS) == 1


def test_prompt_positions_are_ignored_in_labels_and_loss_mask() -> None:
    encoded = _pipeline().encode_record("ab", "xyz")

    assert encoded.labels == [-100, -100, ord("x"), ord("y"), ord("z"), EOS]
    assert encoded.loss_mask == [False, False, True, True, True, True]


def test_labels_use_the_configured_ignore_index() -> None:
    encoded = _pipeline(ignore_index=-1).encode_record("ab", "x")

    assert encoded.labels[:2] == [-1, -1]


def test_input_ids_labels_and_loss_mask_have_the_same_length() -> None:
    encoded = _pipeline().encode_record("a longer prompt", "{}")

    assert len(encoded.input_ids) == len(encoded.labels) == len(encoded.loss_mask)


def test_real_tokenizer_round_trips_the_response() -> None:
    pipeline = ExtractionDataPipeline(PipelineConfig(), BasicTokenizer())
    prompt, response = pipeline.serialize_record(_record())

    encoded = pipeline.encode_record(prompt, response)
    response_ids = [t for t, keep in zip(encoded.input_ids, encoded.loss_mask) if keep]

    assert response_ids[-1] == pipeline.tokenizer.eos_token_id
    assert pipeline.tokenizer.decode(response_ids[:-1]) == response


# validate_sequence_length / extract_record


def test_record_at_the_limit_is_accepted() -> None:
    _pipeline(max_sequence_length=5).validate_sequence_length(_encoded(5))


def test_record_over_the_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max sequence length of 5"):
        _pipeline(max_sequence_length=5).validate_sequence_length(_encoded(6))


def test_extract_record_rejects_a_record_that_is_too_long() -> None:
    with pytest.raises(ValueError, match="max sequence length"):
        _pipeline(max_sequence_length=10).extract_record(_record())


# collate_batch


def test_batch_is_padded_to_the_longest_record() -> None:
    batch = _pipeline().collate_batch([_encoded(3), _encoded(5)])

    assert isinstance(batch, dict)
    assert {k: tuple(v.shape) for k, v in batch.items()} == {
        "input_ids": (2, 5),
        "labels": (2, 5),
        "attention_mask": (2, 5),
        "loss_mask": (2, 5),
    }


def test_padding_values() -> None:
    batch = _pipeline().collate_batch([_encoded(3), _encoded(5)])
    assert isinstance(batch, dict)

    assert batch["input_ids"][0].tolist() == [1, 2, 10, EOS, EOS]
    assert batch["labels"][0].tolist() == [-100, -100, 10, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["loss_mask"][0].tolist() == [False, False, True, False, False]


def test_longest_record_is_unchanged() -> None:
    longest = _encoded(5)
    batch = _pipeline().collate_batch([_encoded(3), longest])
    assert isinstance(batch, dict)

    assert batch["input_ids"][1].tolist() == longest.input_ids
    assert batch["labels"][1].tolist() == longest.labels
    assert batch["attention_mask"][1].tolist() == [1] * 5


def test_batch_dtypes() -> None:
    batch = _pipeline().collate_batch([_encoded(3)])
    assert isinstance(batch, dict)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["loss_mask"].dtype == torch.bool


# build_dataloader


def test_dataloader_batches_records() -> None:
    records = [_record(record_id=f"r{i}") for i in range(5)]

    batches = list(_pipeline(batch_size=2).build_dataloader(records, shuffle=False))

    assert [len(b["input_ids"]) for b in batches] == [2, 2, 1]


def test_dataloader_without_shuffle_keeps_order() -> None:
    pipeline = _pipeline(batch_size=2)
    records = [_record("First note."), _record("Second note, which is longer.")]

    [batch] = list(pipeline.build_dataloader(records, shuffle=False))
    first = pipeline.extract_record(records[0])

    assert batch["input_ids"][0, : len(first.input_ids)].tolist() == first.input_ids


def test_dataloader_skips_records_that_are_too_long() -> None:
    pipeline = _pipeline(batch_size=8, max_sequence_length=100)
    records = [_record("Short."), _record("A note far too long. " * 10, record_id="long")]

    batches = list(pipeline.build_dataloader(records, shuffle=False))

    assert sum(len(b["input_ids"]) for b in batches) == 1
