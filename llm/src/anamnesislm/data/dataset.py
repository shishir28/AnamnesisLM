from torch.utils.data import Dataset

from anamnesislm.data.records import EncodedRecord


class EncodedDataset(Dataset[EncodedRecord]):
    """A dataset of encoded records for model training or evaluation."""

    def __init__(self, records: list[EncodedRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> EncodedRecord:
        return self.records[idx]
