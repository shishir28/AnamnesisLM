# Architecture decisions

## Three projects, one repository

- `web/`: Next.js UI with domain-oriented feature modules.
- `api/`: FastAPI application, lightweight DDD, sole database owner initially.
- `llm/`: reusable Python package organized around ML components, not DDD layers.

The browser calls FastAPI; FastAPI invokes the local LLM package and uses PostgreSQL. Keep the LLM independent of HTTP and database frameworks. The CLI and API share the same inference path.

The UI can be developed independently against an agreed API contract. That contract, dependency manifests, and executable applications have not been implemented yet.

## Data ownership

The API owns database migrations and application persistence. The web app does not access database tables directly. Store training corpora, tokenized splits, and checkpoints outside PostgreSQL. Store only safe manifests and summaries in Git.

## Planned container layout

| Container | Build context | Role |
|---|---|---|
| synthea, available | repository root | Optional `tools` profile; generate synthetic patient data |
| web | `web/` | Next.js interface |
| api | repository root | FastAPI plus installed local LLM package |
| postgres | upstream image | Persistent application database |
| training, optional | repository root | Training/CLI environment; GPU access if supported |

The root `compose.yaml` currently contains only Synthea, built from `tools/synthea/Dockerfile`. Application Dockerfiles and services are deferred until entry points and dependencies exist; the web/API stack is not deployable yet. See [Synthea usage](../tools/synthea/README.md). When adding a `web/`-context image, add its own `.dockerignore` because the repository-root one does not apply to a different build context.

Use a persistent PostgreSQL volume and runtime mounts for weights and permitted local data. Avoid baking corpora, secrets, or checkpoints into images. Pin and validate compatible PyTorch/CUDA images for the actual hardware; GPU support is not assumed.

Load the checkpoint once at API startup. Initially use a single model-hosting process and bounded inference concurrency; multiple worker processes duplicate model memory. Keep training separate from serving to avoid resource contention.

## Demonstration boundaries

Use synthetic notes only. Show validation failures and model versions. Extraction results are experimental, not clinical guidance. Authentication, authorization, TLS, request limits, retention, and operational safeguards require explicit work before public exposure.
