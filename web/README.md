# Web — Next.js demonstration interface

Status: folders only; no Next.js application has been generated.

## Structure

- `src/app/`: planned Next.js App Router pages, layouts, and route composition.
- `src/modules/`: domain-oriented feature modules; introduce domain, application, infrastructure, and presentation boundaries within a module only when justified.
- `src/shared/`: genuinely shared UI and utilities, not a general dumping ground.
- `tests/`: unit and integration tests; browser tests can be added later.

## Planned behavior

Submit synthetic clinical notes to FastAPI and display extracted facts, JSON, supporting text, validation failures, and the model version. Never connect directly to PostgreSQL initially. Do not put secrets in browser-exposed configuration.

Define the API request/response contract before implementing the integration. Keep the interface separate from model inference logic. Vibe-coded UI should still be reviewed and tested, especially error handling and evidence highlighting.

A Dockerfile and dependency manifests will follow application initialization. Do not copy the Python project or datasets into the web image.
