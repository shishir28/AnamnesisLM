"""Fact extraction with a trained checkpoint, shared by the CLI and the API.

Create one Extractor at startup and reuse it for every request; loading a checkpoint is slow.
Nothing here prints or logs note text.

Two steps around the model keep evidence offsets exact:
- The model reads a whitespace-normalised note (one space between words), the form it was
  trained on, so line breaks or double spaces in the caller's note do not confuse it.
- evidence_start / evidence_end are computed here by finding the evidence text in the note,
  never taken from the model: a small model copies text well (~98%) but counts characters badly
  (~26% exact). Offsets and evidence refer to the caller's original note.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from anamnesislm.data.output_format import complete_facts
from anamnesislm.inference.generate import generate
from anamnesislm.model.config import ModelConfig
from anamnesislm.model.gpt import GPTModel
from anamnesislm.tokenization.basic_tokenizer import BasicTokenizer
from anamnesislm.tokenization.tokenizer import Tokenizer


@dataclass(frozen=True)
class ExtractionResult:
    """What the model wrote, the facts if that text is valid JSON, and which model wrote it."""

    raw_output: str
    facts: list[dict[str, Any]] | None  # None when raw_output is not valid JSON
    model_version: str
    error: str | None = None


class Extractor:
    """A loaded model plus everything needed to turn a note into facts."""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: Tokenizer,
        device: str,
        model_version: str,
        max_new_tokens: int = 400,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_version = model_version
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str) -> "Extractor":
        """Rebuild GPTModel from the checkpoint's model_config, load weights, move to device, eval.

        Tokenizer comes from the checkpoint's "tokenizer" entry; model_version from the file
        name and training step, e.g. "best.pt@step8478".
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_config = dict(checkpoint["model_config"])
        # Checkpoints from before multi-head attention have no "multi_head" entry.
        saved_config.setdefault("multi_head", False)
        model_config = ModelConfig(**saved_config)
        model = GPTModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return cls(
            model=model,
            tokenizer=BasicTokenizer(checkpoint["tokenizer"]),
            device=device,
            model_version=f"{checkpoint_path.name}@step{checkpoint['step']}",
        )

    def extract(self, note: str) -> ExtractionResult:
        """normalise -> build_prompt -> encode -> generate -> decode -> parse_facts -> locate."""
        normalised, origin = normalize_whitespace(note)
        prompt = build_prompt(normalised)
        input_ids = self.tokenizer.encode(prompt)
        new_ids = generate(
            model=self.model,
            prompt_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        raw_output = self.tokenizer.decode(new_ids)
        result = parse_facts(raw_output, self.model_version)
        if result.facts is None:
            return result
        located = locate_evidence(result.facts, note, normalised, origin)
        return replace(result, facts=complete_facts(located))


def normalize_whitespace(note: str) -> tuple[str, list[int]]:
    """Collapse each run of whitespace to one space and trim the ends.

    Returns the normalised text and, for each of its characters, the index of the character in
    note it came from (a collapsed run maps to its first whitespace character).
    """
    chars: list[str] = []
    origin: list[int] = []
    gap_start: int | None = None
    for index, char in enumerate(note):
        if char.isspace():
            if chars and gap_start is None:
                gap_start = index
            continue
        if gap_start is not None:
            chars.append(" ")
            origin.append(gap_start)
            gap_start = None
        chars.append(char)
        origin.append(index)
    return "".join(chars), origin


def locate_evidence(
    facts: list[dict[str, Any]], note: str, normalised: str, origin: list[int]
) -> list[dict[str, Any]]:
    """Set each fact's evidence offsets from where its evidence text occurs in the note.

    Searches the normalised note, then maps back to note, so evidence becomes the exact original
    span (line breaks included). Facts are searched in order, after the previous match first, so
    a repeated sentence maps to its next occurrence. A fact whose evidence is not in the note is
    left as the model wrote it, so validation still reports it.
    """
    located: list[dict[str, Any]] = []
    cursor = 0
    for fact in facts:
        evidence = fact.get("evidence")
        if not isinstance(evidence, str) or not evidence:
            located.append(fact)
            continue
        start = normalised.find(evidence, cursor)
        if start == -1:
            start = normalised.find(evidence)
        if start == -1:
            located.append(fact)
            continue
        end = start + len(evidence)
        cursor = end
        original_start, original_end = origin[start], origin[end - 1] + 1
        located.append(
            {
                **fact,
                "evidence": note[original_start:original_end],
                "evidence_start": original_start,
                "evidence_end": original_end,
            }
        )
    return located


def build_prompt(note: str) -> str:
    """Prompt text for one note. Must match ExtractionDataPipeline.serialize_record exactly."""
    prompt = f"Extract facts from this note:\n{note}\n\nResponse:"
    return prompt


def parse_facts(raw_output: str, model_version: str) -> ExtractionResult:
    """Parse '{"facts": [...]}'; on json.JSONDecodeError or a missing "facts" key, keep the error."""
    try:
        json_data = json.loads(raw_output)  # raises json.JSONDecodeError if invalid
        if (not isinstance(json_data, dict)) or ("facts" not in json_data):
            return _invalid(
                raw_output, model_version, 'Output is not a JSON object with a "facts" key'
            )
        json_facts = json_data.get("facts")
        if not isinstance(json_facts, list) or not all(isinstance(f, dict) for f in json_facts):
            return _invalid(
                raw_output, model_version, 'Missing or invalid "facts" key in JSON output'
            )
    except json.JSONDecodeError as e:
        return _invalid(raw_output, model_version, f"JSONDecodeError: {str(e)}")
    return ExtractionResult(
        raw_output=raw_output,
        facts=json_facts,
        model_version=model_version,
        error=None,
    )


def _invalid(raw_output: str, model_version: str, error_message: str) -> ExtractionResult:
    """Return an ExtractionResult for invalid output."""
    return ExtractionResult(
        raw_output=raw_output,
        facts=None,
        model_version=model_version,
        error=error_message,
    )
