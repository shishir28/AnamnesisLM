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
