from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser and loop settings. Starting values; tune after the tiny overfit check."""

    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    num_epochs: int = 3
    eval_freq: int = 500  # evaluate every N optimizer steps
    eval_iter: int = 50  # validation batches per periodic check; final score uses all
    device: str = "cuda"
    seed: int = 42
    grad_clip_norm: float = 1.0
    checkpoint_dir: Path = Path("checkpoints/extraction")  # git-ignored
