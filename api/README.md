# API — FastAPI model hosting

Status: folders only; no FastAPI application, database schema, or migrations yet.

## Lightweight DDD structure

Under `src/anamnesis_api/`:

- `domain/`: application concepts and rules, independent of HTTP and persistence.
- `application/`: extraction use cases and interfaces for persistence/inference.
- `infrastructure/`: PostgreSQL repositories and adapters to the LLM package.
- `presentation/`: FastAPI routing, HTTP request/response models, error mapping.

`tests/` holds tests. `migrations/` will hold database migrations; the API owns them.

## Planned responsibilities

- Validate requests and invoke shared inference from `llm/` through an adapter.
- Load one checkpoint at startup, not per request.
- Expose an extraction endpoint and health/readiness endpoints; exact contracts are not defined yet.
- Persist application records and extraction metadata/results in PostgreSQL.
- Return explicit validation failures and model-version information.
- Avoid raw note logging; configure limits, concurrency, authentication, and retention before exposing the service beyond a trusted local environment.

Keep the initial model host in this process; do not introduce a separate inference microservice prematurely. One process per model replica is the initial assumption, because multiple API workers can duplicate model/GPU memory.

The planned API container uses repository-root build context to install both `api/` and the local `llm/` package. Mount weights at runtime. Add dependencies, packaging, Dockerfile, and Compose only alongside working code.
