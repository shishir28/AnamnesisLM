import torch
from torch.utils.data import DataLoader

from anamnesislm.data.config import PipelineConfig
from anamnesislm.data.dataset import EncodedDataset
from anamnesislm.data.records import EncodedRecord, ExtractionRecord
from anamnesislm.tokenization.tokenizer import Tokenizer


class ExtractionDataPipeline:
    """Planned conversion from extraction records to model-ready batches.

    Data flow:

        ExtractionRecord
            -> collate_batch()
            -> model-ready batch

    Synthea reading, patient splitting, and note rendering remain separate concerns.
    """

    def __init__(self, config: PipelineConfig, tokenizer: Tokenizer) -> None:
        """Initialize the pipeline with a configuration object and tokenizer."""
        self.config = config
        self.tokenizer = tokenizer

    def extract_record(self, record: ExtractionRecord) -> EncodedRecord:
        """Convert one extraction record to an encoded record."""
        prompt, response = self.serialize_record(record)
        encoded_record = self.encode_record(prompt, response)
        self.validate_sequence_length(encoded_record)
        return encoded_record

    def serialize_record(self, record: ExtractionRecord) -> tuple[str, str]:
        """Create prompt and response strings for one extraction record."""
        prompt = f"Extract facts from this note:\n{record.note}\n\nResponse:"
        response = record.expected_json
        return prompt, response

    def encode_record(self, prompt: str, response: str) -> EncodedRecord:
        """Tokenize prompt and response strings, append EOS once, and create labels and masks."""
        prompt_tokens = self.tokenizer.encode(prompt)
        response_tokens = self.tokenizer.encode(response)
        response_ids = response_tokens + [self.tokenizer.eos_token_id]

        return EncodedRecord(
            input_ids=prompt_tokens + response_ids,
            labels=[self.config.ignore_index] * len(prompt_tokens) + response_ids,
            loss_mask=[False] * len(prompt_tokens) + [True] * len(response_ids),
        )

    def validate_sequence_length(self, record: EncodedRecord) -> None:
        """Reject an encoded record that exceeds the configured token budget."""
        if len(record.input_ids) > self.config.max_sequence_length:
            raise ValueError(
                f"Encoded record exceeds max sequence length of {self.config.max_sequence_length} tokens."
            )

    def collate_batch(self, records: list[EncodedRecord]) -> object:
        """Pad encoded records and return a model-ready batch."""
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_loss_mask = []

        max_length = max(len(record.input_ids) for record in records)

        for record in records:
            padding_length = max_length - len(record.input_ids)
            batch_input_ids.append(
                record.input_ids + [self.tokenizer.eos_token_id] * padding_length
            )
            batch_labels.append(record.labels + [self.config.ignore_index] * padding_length)
            batch_attention_mask.append([1] * len(record.input_ids) + [0] * padding_length)
            batch_loss_mask.append(record.loss_mask + [False] * padding_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(batch_loss_mask, dtype=torch.bool),
        }

    def build_dataloader(
        self, records: list[ExtractionRecord], shuffle: bool
    ) -> DataLoader[EncodedRecord]:
        """Convert extraction records to a DataLoader for model training or evaluation."""
        encoded_records: list[EncodedRecord] = []
        for record in records:
            try:
                encoded_records.append(self.extract_record(record))
            except ValueError as e:
                print(f"Skipping record {record.record_id} due to error: {e}")

        encoded_dataset = EncodedDataset(encoded_records)
        return DataLoader(
            dataset=encoded_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_batch,
        )
