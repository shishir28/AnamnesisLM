import json

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.data_pipeline import ExtractionDataPipeline
from anamnesislm.data.output_format import (
    FACT_FIELDS,
    MODEL_FIELDS,
    compact_response,
    complete_facts,
)
from anamnesislm.data.records import ExtractionRecord
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer

FULL_FACT = {
    "kind": "condition",
    "mention": "Asthma",
    "assertion": "affirmed",
    "experiencer": "patient",
    "temporality": "current",
    "evidence": "Currently has Asthma.",
    "evidence_start": 31,
    "evidence_end": 52,
    "dose": None,
    "unit": None,
    "route": None,
    "frequency": None,
}
EXPECTED_JSON = json.dumps({"facts": [FULL_FACT]})


def test_full_fact_has_every_field() -> None:
    assert set(FULL_FACT) == set(FACT_FIELDS)


def test_compact_response_keeps_only_model_fields() -> None:
    [fact] = json.loads(compact_response(EXPECTED_JSON))["facts"]

    assert tuple(fact) == MODEL_FIELDS
    assert fact["evidence"] == "Currently has Asthma."


def test_compact_response_is_much_shorter() -> None:
    tokenizer = BasicTokenizer()

    full = len(tokenizer.encode(EXPECTED_JSON))

    assert len(tokenizer.encode(compact_response(EXPECTED_JSON))) < 0.75 * full


def test_complete_facts_adds_missing_fields_only() -> None:
    compact = {key: FULL_FACT[key] for key in MODEL_FIELDS}

    [completed] = complete_facts([compact])
    [unchanged] = complete_facts([FULL_FACT])

    assert set(completed) == set(FACT_FIELDS)
    assert completed["evidence_start"] == -1
    assert completed["dose"] is None
    assert unchanged == FULL_FACT


def test_pipeline_trains_on_the_compact_response_when_configured() -> None:
    record = ExtractionRecord(note="n", expected_json=EXPECTED_JSON, record_id="r")
    tokenizer = BasicTokenizer()

    _, full = ExtractionDataPipeline(PipelineConfig(), tokenizer).serialize_record(record)
    _, compact = ExtractionDataPipeline(
        PipelineConfig(compact_response=True), tokenizer
    ).serialize_record(record)

    assert full == EXPECTED_JSON
    assert compact == compact_response(EXPECTED_JSON)
