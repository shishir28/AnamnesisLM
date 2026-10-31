from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    max_sequence_length: int = 512
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    ignore_index: int = -100
