import json
from pathlib import Path


class ExtractionRecord:
    """Planned representation of one note and its expected JSON response.

    Planned fields:
    - note: original synthetic clinical note.
    - expected_json: canonical serialized extraction target.
    - record_id: stable identifier for provenance and debugging.
    """

    def __init__(self, note: str, expected_json: str, record_id: str) -> None:
        self.note = note
        self.expected_json = expected_json
        self.record_id = record_id


class EncodedRecord:
    """Planned tokenized representation of one extraction record.

    Planned fields:
    - input_ids: prompt, response, and one final EOS token.
    - labels: next-token targets with ignored prompt positions.
    - loss_mask: positions that contribute to response-only loss.
    """

    def __init__(self, input_ids: list[int], labels: list[int], loss_mask: list[bool]) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.loss_mask = loss_mask


def load_records(path: Path) -> list[ExtractionRecord]:
    """Read one split's records.jsonl into ExtractionRecords, skipping blank lines."""
    records: list[ExtractionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not (line := line.strip()):
                continue
            record_dict = json.loads(line)
            records.append(
                ExtractionRecord(
                    note=record_dict["note"],
                    expected_json=record_dict["expected_json"],
                    record_id=record_dict["record_id"],
                )
            )
    return records
