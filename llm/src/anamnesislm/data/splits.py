"""Deterministic patient-level train/validation/test assignment.

A patient's split depends only on a hash of their id, so it is stable across runs,
machines, and file order, and every record from one patient lands in the same split.
"""

import hashlib
from typing import Literal

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")


def assign_split(patient_id: str, train_percent: int = 80, validation_percent: int = 10) -> Split:
    """Map a patient id to a split; the remaining percentage goes to test."""
    if not 0 <= train_percent + validation_percent <= 100:
        raise ValueError("train_percent + validation_percent must be between 0 and 100")
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + validation_percent:
        return "validation"
    return "test"
