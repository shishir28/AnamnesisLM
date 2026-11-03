from pathlib import Path
from typing import Any

from anamnesis_api.application.extraction import InvalidNoteError
from anamnesis_api.domain.extraction import ModelOutput


class LocalModelBackend:
    """Reuse one checkpoint; per-request generation settings never mutate it."""

    def __init__(self, extractor: Any, device: str) -> None:
        self.extractor = extractor
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: Path, device: str) -> "LocalModelBackend":
        import torch
        from anamnesislm.inference import Extractor

        if not checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint not found")
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(Extractor.from_checkpoint(checkpoint_path, resolved_device), resolved_device)

    def info(self) -> dict[str, str | int]:
        return {
            "model_version": self.extractor.model_version,
            "device": self.device,
            "context_length": self.extractor.model.config.context_length,
        }

    def extract(self, note: str, max_new_tokens: int) -> ModelOutput:
        from anamnesislm.inference import Extractor
        from anamnesislm.inference.extractor import build_prompt

        try:
            prompt_ids = self.extractor.tokenizer.encode(build_prompt(note))
        except ValueError as exc:
            raise InvalidNoteError("Note contains unsupported special-token text.") from exc
        available = self.extractor.model.config.context_length - len(prompt_ids)
        if available < 1:
            raise InvalidNoteError("Note is too long for the model context. Shorten it and retry.")
        token_limit = min(available, max_new_tokens)
        warnings = ()
        if token_limit < max_new_tokens:
            warnings = (
                f"Generation budget reduced to {token_limit} tokens to preserve the full note.",
            )
        # The loaded model and tokenizer are shared; only this lightweight wrapper is new.
        request_extractor = Extractor(
            model=self.extractor.model,
            tokenizer=self.extractor.tokenizer,
            device=self.device,
            model_version=self.extractor.model_version,
            max_new_tokens=token_limit,
        )
        result = request_extractor.extract(note)
        return ModelOutput(
            raw_output=result.raw_output,
            facts=result.facts,
            model_version=result.model_version,
            error=result.error,
            warnings=warnings,
            generation_token_limit=token_limit,
        )
