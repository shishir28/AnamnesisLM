from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    max_sequence_length: int = 512
    batch_size: int = 8
    num_workers: int = 0
    ignore_index: int = -100
    # Train the model to write only output_format.MODEL_FIELDS (offsets are computed in code).
    compact_response: bool = False
