# LLM — from-scratch learning implementation

Status: component directories only; no implemented model or training pipeline.

## Components

Under `src/anamnesislm/`:

| Folder | Purpose |
|---|---|
| `tokenization/` | Existing tiktoken BPE wrapper initially; custom tokenizer experiments later |
| `data/` | Readers, preprocessing, deterministic splits, rendering, batching |
| `model/` | Embeddings, causal attention, Transformer blocks, GPT architecture |
| `training/` | Training loops, response masks, optimizer handling, checkpoints; shared pretraining/fine-tuning mechanics |
| `extraction/` | Output schema, annotation rules, evidence validation, rule baseline |
| `evaluation/` | Task metrics, baseline comparison, error analysis |
| `inference/` | Checkpoint loading and generation, reusable by CLI and API |

## Supporting folders

- `tests/`: focused tests mirroring the components.
- `scripts/`: thin entry points; reusable behavior belongs in `src/`.
- `configs/`: versioned settings, without secrets or sensitive data.
- `data/raw/`, `data/interim/`: ignored source data and intermediate outputs.
- `data/processed/{pretraining,extraction}/{train,validation,test}/`: ignored prepared splits.
- `checkpoints/`: ignored weights and training state.
- `experiments/`: safe summaries; generated `experiments/runs/` is ignored.

Split patients before creating examples, and keep test material out of pretraining. Directory names alone do not guarantee isolation. Validate provenance, leakage, and evidence offsets.

Use the book mapping in [Agent.md](../Agent.md) and milestones in the [plan](../ANAMNESISLM_PLAN.md). Start with tiny correctness checks and extraction training from random weights; domain pretraining is a later measured experiment. No pretrained weights are required to reuse a tokenizer.

Credentialed clinical data must stay outside this repository, not in these ignored folders. Ignore rules do not prevent agent access.

Future packaging will make `anamnesislm` installable by the API. A training/CLI Dockerfile will be added when there is executable code and a measured device/dependency configuration.
