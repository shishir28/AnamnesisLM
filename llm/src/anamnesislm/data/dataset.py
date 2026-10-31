from torch.utils.data import Dataset

from anamnesislm.data.records import ExtractionRecord


class ExtractionDataset(Dataset[ExtractionRecord]):
    """A dataset of extraction records for model training or evaluation."""

    def __init__(self, records: list[ExtractionRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ExtractionRecord:
        return self.records[idx]
