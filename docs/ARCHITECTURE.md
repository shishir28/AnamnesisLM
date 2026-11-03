# Architecture

```text
Browser → Next.js page
        → Next.js server routes (/api/model, /api/extract)
        → FastAPI (/model, /extract)
        → installed anamnesislm package → local PyTorch checkpoint
```

## Projects

- `web/`: domain-oriented extraction feature, HTTP client, and presentation.
- `api/`: domain validation, application service, LLM adapter, HTTP routes.
- `llm/`: model, training, data preparation, and reusable inference; independent of HTTP.

FastAPI loads one checkpoint at startup. The CLI and API reuse the existing LLM inference implementation. An API adapter bounds generation so the full prompt is retained, without modifying LLM training code. A nonblocking lock permits one generation request; another receives 429.

## Containers

| Service | Build context | Role |
|---|---|---|
| `web` | `web/` | Next.js standalone production server |
| `api` | repository root | FastAPI plus local LLM package; CPU PyTorch |
| `synthea` | repository root | Optional `tools` profile for synthetic records |

Run `docker compose up --build -d api web`. Bind ports are loopback-only: 3000 for web and 8000 for API. Remote access uses SSH forwarding. The web proxy resolves FastAPI through Docker's `api` service name; browsers do not need that hostname.

Mount `llm/checkpoints/extraction/` read-only at `/checkpoints`. Checkpoints, corpora, and secrets are excluded from image build contexts. Do not unpickle checkpoints from untrusted sources. GPU Docker serving is not yet configured. Native API serving supports CPU/CUDA through configuration.

Process health is independent of model readiness. Missing or incompatible weights leave `/health` working, `/ready` at 503, and the UI able to explain that the model is unavailable. After replacing a checkpoint, restart the API to reload it.

## Data and validation

No database or note persistence is implemented for this demo. PostgreSQL remains a later option for application records and extraction metadata; only the API should own its schema/migrations. Datasets and model weights remain file artifacts.

The API preserves generated facts and raw output. Schema and exact evidence-offset failures remain visible; no output is silently repaired. Evidence matching is not semantic support, clinical validation, or an evaluation score. Incomplete generation remains invalid JSON when applicable.

## Safety

Synthetic notes only. No authentication or public production hardening. Keep ports private; no raw note logging or persistence in application code. The web request size and generation budgets are bounded, but this is not a public multi-user serving platform. Training and inference remain separate processes to avoid resource contention.
