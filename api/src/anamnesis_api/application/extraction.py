import json
from dataclasses import replace
from threading import Lock
from typing import Protocol

from anamnesis_api.domain.extraction import ModelOutput


class InvalidNoteError(ValueError):
    pass


class ModelBusyError(Exception):
    pass


class ExtractionBackend(Protocol):
    def extract(self, note: str, max_new_tokens: int) -> ModelOutput: ...

    def info(self) -> dict[str, str | int]: ...


class ExtractionService:
    """Serialize generation without blocking a queue of incoming requests."""

    def __init__(self, backend: ExtractionBackend) -> None:
        self.backend = backend
        self._lock = Lock()

    def extract(self, note: str, max_new_tokens: int) -> ModelOutput:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError
        try:
            output = self.backend.extract(note, max_new_tokens)
            try:
                json.dumps(output.facts, allow_nan=False)
            except (TypeError, ValueError):
                # Preserve the raw generation; do not return NaN/Infinity as JSON values.
                return replace(
                    output, facts=None, error="Generated facts contain values not allowed in JSON."
                )
            return output
        finally:
            self._lock.release()
