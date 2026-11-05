from dataclasses import replace
from pathlib import Path

import pytest
import torch

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.records import ExtractionRecord
from anamnesislm.inference.extractor import (
    Extractor,
    build_prompt,
    locate_evidence,
    normalize_whitespace,
    parse_facts,
)
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.training.trainer import save_checkpoint

# build_prompt


@pytest.mark.parametrize(
    "note",
    [
        "Patient is a 54-year-old male. Denies chest pain.",
        "",
        "Line one.\nLine two.",
    ],
)
def test_prompt_matches_the_training_prompt(note: str) -> None:
    pipeline = ExtractionDataPipeline(PipelineConfig(), BasicTokenizer())
    training_prompt, _ = pipeline.serialize_record(
        ExtractionRecord(note=note, expected_json="{}", record_id="r1")
    )

    assert build_prompt(note) == training_prompt


# parse_facts


def test_valid_facts_are_returned() -> None:
    result = parse_facts('{"facts": [{"kind": "condition", "mention": "Asthma"}]}', "v1")

    assert result.facts == [{"kind": "condition", "mention": "Asthma"}]
    assert result.error is None
    assert result.model_version == "v1"


def test_empty_facts_list_is_valid() -> None:
    result = parse_facts('{"facts": []}', "v1")

    assert result.facts == []
    assert result.error is None


@pytest.mark.parametrize(
    ("raw_output", "error_fragment"),
    [
        ("not json", "JSONDecodeError"),
        ('{"facts": [', "JSONDecodeError"),
        ('{"other": 1}', '"facts" key'),
        ("[1, 2]", "not a JSON object"),
        ("null", "not a JSON object"),
        ('{"facts": "x"}', '"facts"'),
        ('{"facts": [1, "x"]}', '"facts"'),
    ],
)
def test_invalid_output_is_reported_not_raised(raw_output: str, error_fragment: str) -> None:
    result = parse_facts(raw_output, "v1")

    assert result.facts is None
    assert result.error is not None and error_fragment in result.error
    assert result.raw_output == raw_output
    assert result.model_version == "v1"


# Extractor.from_checkpoint

CONFIG = ModelConfig(vocab_size=50257, context_length=16, embedding_dim=16, num_layers=1)


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, GPTModel]:
    torch.manual_seed(0)
    model = GPTModel(CONFIG).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best.pt"
    save_checkpoint(
        path, model, optimizer, epoch=2, step=123, validation_loss=1.5, tokenizer="gpt2"
    )
    return path, model


def test_loaded_model_gives_the_same_logits(checkpoint: tuple[Path, GPTModel]) -> None:
    path, original = checkpoint

    extractor = Extractor.from_checkpoint(path, "cpu")

    tokens = torch.randint(0, 50257, (1, 8))
    assert torch.equal(extractor.model(tokens), original(tokens))


def test_loaded_model_is_in_eval_mode_on_the_requested_device(
    checkpoint: tuple[Path, GPTModel],
) -> None:
    extractor = Extractor.from_checkpoint(checkpoint[0], "cpu")

    assert not extractor.model.training
    assert next(extractor.model.parameters()).device == torch.device("cpu")
    assert extractor.device == "cpu"


def test_model_version_names_the_file_and_step(checkpoint: tuple[Path, GPTModel]) -> None:
    assert Extractor.from_checkpoint(checkpoint[0], "cpu").model_version == "best.pt@step123"


def test_tokenizer_comes_from_the_checkpoint(checkpoint: tuple[Path, GPTModel]) -> None:
    tokenizer = Extractor.from_checkpoint(checkpoint[0], "cpu").tokenizer

    assert tokenizer.vocab_size == 50257
    assert tokenizer.decode(tokenizer.encode("Denies fever.")) == "Denies fever."


def test_new_checkpoints_load_with_multi_head_attention(
    checkpoint: tuple[Path, GPTModel],
) -> None:
    assert Extractor.from_checkpoint(checkpoint[0], "cpu").model.config.multi_head


def test_checkpoint_from_before_multi_head_loads_single_head(tmp_path: Path) -> None:
    torch.manual_seed(0)
    old_model = GPTModel(replace(CONFIG, multi_head=False)).eval()
    path = tmp_path / "v1.pt"
    save_checkpoint(path, old_model, torch.optim.AdamW(old_model.parameters()), 0, 1, 1.0, "gpt2")
    saved = torch.load(path)
    del saved["model_config"]["multi_head"]  # as written by the code before multi-head existed
    torch.save(saved, path)

    extractor = Extractor.from_checkpoint(path, "cpu")

    tokens = torch.randint(0, 50257, (1, 8))
    assert not extractor.model.config.multi_head
    assert torch.equal(extractor.model(tokens), old_model(tokens))


# normalize_whitespace


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("One. Two.", "One. Two."),
        ("One.\nTwo.", "One. Two."),
        ("One.  \t\n Two.", "One. Two."),
        ("  One. Two.\n", "One. Two."),
        ("", ""),
    ],
)
def test_whitespace_runs_become_one_space(note: str, expected: str) -> None:
    assert normalize_whitespace(note)[0] == expected


def test_origin_maps_each_character_back_to_the_note() -> None:
    note = " Ab\n\ncd "
    normalised, origin = normalize_whitespace(note)

    assert normalised == "Ab cd"
    assert origin == [1, 2, 3, 5, 6]  # the space maps to the first "\n"


# locate_evidence


def _locate(facts: list[dict[str, object]], note: str) -> list[dict[str, object]]:
    return locate_evidence(facts, note, *normalize_whitespace(note))


def _evidence(text: str, start: int = 0, end: int = 0) -> dict[str, object]:
    return {"mention": "x", "evidence": text, "evidence_start": start, "evidence_end": end}


def test_wrong_model_offsets_are_replaced_by_exact_ones() -> None:
    note = "Patient is a 54-year-old male. Currently has Asthma."

    [fact] = _locate([_evidence("Currently has Asthma.", 31, 49)], note)

    assert (fact["evidence_start"], fact["evidence_end"]) == (31, 52)
    assert note[31:52] == fact["evidence"]


def test_evidence_across_a_line_break_maps_to_the_original_text() -> None:
    note = "Patient is a 54-year-old male. The patient\ndenies any chest pain."

    [fact] = _locate([_evidence("The patient denies any chest pain.")], note)

    start, end = fact["evidence_start"], fact["evidence_end"]
    assert isinstance(start, int) and isinstance(end, int)
    assert fact["evidence"] == "The patient\ndenies any chest pain."
    assert note[start:end] == fact["evidence"]


def test_repeated_sentence_maps_to_its_next_occurrence() -> None:
    note = "Patient. History of Asthma. History of Asthma."

    first, second = _locate(
        [_evidence("History of Asthma."), _evidence("History of Asthma.")], note
    )

    assert (first["evidence_start"], second["evidence_start"]) == (9, 28)


def test_evidence_not_in_the_note_is_left_unchanged() -> None:
    fact = _evidence("Invented sentence.", 3, 7)

    assert _locate([fact], "Patient. History of Asthma.") == [fact]


@pytest.mark.parametrize("evidence", [None, "", 5])
def test_missing_or_non_text_evidence_is_left_unchanged(evidence: object) -> None:
    fact = {"mention": "x", "evidence": evidence}

    assert _locate([fact], "Patient.") == [fact]


def test_other_fields_are_kept() -> None:
    fact = {**_evidence("Fever."), "kind": "condition", "assertion": "negated"}

    [located] = _locate([fact], "No. Fever.")

    assert (located["kind"], located["assertion"]) == ("condition", "negated")
