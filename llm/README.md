# LLM — from-scratch learning implementation

Status: data preparation, tokenizer, single-head decoder-only Transformer, training, checkpoint loading, and CLI inference are implemented. Task-level evaluation and a rule baseline remain incomplete.

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

The package is installed by the [FastAPI app](../api/README.md), which hosts the checkpoint for the [Next.js demo](../web/README.md). Run both through the root `compose.yaml`; the API image uses CPU PyTorch. A dedicated training container is not implemented.
