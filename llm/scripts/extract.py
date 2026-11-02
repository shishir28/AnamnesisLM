"""Extract facts from a clinical note with a trained checkpoint.

Example:
    uv run python scripts/extract.py "Patient is a 54-year-old male. Denies chest pain."
"""

import argparse
import json
from pathlib import Path

import torch

from anamnesislm.inference import Extractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/extraction/best.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("note", help="clinical note text")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor.from_checkpoint(args.checkpoint, device)
    extractor.max_new_tokens = args.max_new_tokens

    result = extractor.extract(args.note)

    if result.facts is None:
        print(f"Invalid output ({result.error}):\n{result.raw_output}")
        return 1
    print(json.dumps({"model_version": result.model_version, "facts": result.facts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
