# API

FastAPI loads one local checkpoint at startup and reuses it for extraction. No database or note persistence is implemented.

## Docker

From the repository root:

```bash
docker compose up --build -d api web
curl http://localhost:8000/model
```

The default image uses CPU PyTorch, runs as a non-root user, and mounts `llm/checkpoints/extraction/` read-only at `/checkpoints`. It loads `v9-second-seed.pt` (set `ANAMNESIS_CHECKPOINT_FILE` to choose another file); weights are not bundled. Use trusted local checkpoints only.

Choose another directory before starting:

```bash
ANAMNESIS_CHECKPOINT_DIR=/absolute/path/to/checkpoints docker compose up -d --force-recreate api
```

Docker GPU serving is not configured. Native serving can use CUDA if installed and available.

## Native development

Requires Python 3.11+ and uv:

```bash
cd api
uv sync --extra dev
uv run uvicorn anamnesis_api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Optional settings in `api/.env` or the environment:

| Variable | Default |
|---|---|
| `ANAMNESIS_CHECKPOINT_PATH` | Repository `llm/checkpoints/extraction/v9-second-seed.pt`. Use a named checkpoint, not `best.pt`: every training run overwrites `best.pt`. |
| `ANAMNESIS_DEVICE` | `auto` (`cpu` or `cuda` can be selected) |
| `ANAMNESIS_MAX_NEW_TOKENS` | `400` |

Use one worker per model replica. Each additional process loads another model copy. Local dependency installation can download CUDA packages; the Docker image explicitly installs CPU wheels.

## HTTP contract

- `GET /health`: process liveness, even when the model failed to load.
- `GET /ready`: 200 when loaded, otherwise 503.
- `GET /model`: readiness, model version, device, context length, server generation limit.
- `POST /extract`: extraction and validation results.
- `GET /docs`: OpenAPI interface.

```bash
curl http://localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"note":"Patient denies chest pain.","max_new_tokens":400}'
```

Request: nonblank `note` up to 12,000 characters; integer `max_new_tokens` from 1 to 1,024. Original note whitespace is preserved. The server cap and remaining context can reduce generation length. Oversized tokenized prompts and special-token text are rejected rather than silently cropped.

Response: `raw_output`, `facts` (or null), `model_version`, `valid`, `validation_errors`, `warnings`, `elapsed_ms`, and `generation_token_limit`.

Invalid model JSON/schema/evidence returns HTTP 200 with `valid: false`, preserving output for inspection. This is not an HTTP transport failure. Validation checks schema, mention inclusion, and exact evidence offsets—not medical accuracy, omissions, or semantic support. Empty facts can pass structural validation.

Errors: 422 invalid request/note, 429 model busy, 503 model unavailable, 500 inference failure. Only one generation request runs at a time; concurrent requests are rejected rather than queued.

## Code and tests

- `domain/`: generated-fact contract and validation.
- `application/`: backend interface and concurrency control.
- `infrastructure/`: adapter to the existing LLM package.
- `presentation/`: HTTP validation and response mapping.

```bash
cd api
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests
```

Tests inject a fake backend; adapter tests mock generation. They do not require the checkpoint or GPU. The real checkpoint must be tested separately.

## Limits

Local synthetic-data demo only. No authentication, public deployment hardening, database, or record retention. Request validation responses omit rejected note values; application code does not log note content. Do not expose these ports publicly. Model-load failures leave the API live but not ready; check checkpoint availability, permissions, device compatibility, and `/model`.
