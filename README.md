# AnamnesisLM

A small decoder-only Transformer, trained from scratch to extract evidence-backed facts from synthetic clinical notes.

**Status:** data preparation, a small Transformer, training, CLI inference, FastAPI hosting, and a Next.js demo are implemented. Extraction quality is experimental. Built for learning; not validated for clinical use.

## Structure

| Folder | Purpose |
|---|---|
| `llm/` | Data preparation, tokenizer, model, training, evaluation, inference |
| `api/` | FastAPI model hosting and output validation; no persistence yet |
| `web/` | Next.js interface |
| `tools/` | Synthea container and generation script |
| `docs/` | Architecture and data policies |

Request flow: Next.js → same-origin server proxy → FastAPI → PyTorch model. No database is required for the demo.

## Run the extraction app

Requires Docker Compose and a trusted checkpoint at `llm/checkpoints/extraction/best.pt`. Checkpoints are local artifacts, not included in Git.

```bash
docker compose up --build -d api web
```

- Web: http://localhost:3000
- API docs: http://localhost:8000/docs
- Model status: http://localhost:8000/model

Host ports are configurable; internal container ports remain unchanged:

```bash
WEB_PORT=3100 API_PORT=8100 docker compose up -d api web
```

This exposes the web at http://localhost:3100 and API docs at http://localhost:8100. Keep the same variable values for subsequent Compose commands, or set them in an ignored root `.env` file.

The API loads the checkpoint once. Containers serve on CPU; CUDA Docker serving is not configured. Invalid model output, schema errors, and incorrect evidence offsets remain visible in the UI. No notes are stored.

If accessing this host through SSH, run this on your local computer, then open http://localhost:3000:

```bash
ssh -N -L 3000:127.0.0.1:3000 user@host
```

```bash
docker compose logs --tail 50 api web
docker compose restart api  # Reload after replacing best.pt
docker compose stop api web
```

If the checkpoint is absent or incompatible, the UI reports the model as unavailable. Use [API instructions](api/README.md) for alternate checkpoint paths, native CUDA serving, and request contracts; see [web instructions](web/README.md) for development and tests.

## Generate synthetic data

### Requirements

- Bash: Linux, macOS, or WSL2.
- Running Docker with Compose v2 or later.
- Internet for the image build. No local Java or GPU required.
- Enough memory for the default 2 GB Java heap plus JVM overhead.

Verify Docker:

```bash
docker info
docker compose version
```

Run from the repository root:

```bash
# Small trial: 2 requested patients per state, 10 total
PATIENTS_PER_STATE=2 ./tools/generate_synthea.sh

# Default: 2,000 per state, 10,000 total
./tools/generate_synthea.sh
```

The script builds Synthea v4.0.0 and runs California, Texas, Florida, Massachusetts, and New York sequentially. It sets host UID/GID for file permissions and saves settings and per-state logs.

Age, sex, and race/ethnicity follow Synthea's distributions, not balanced quotas. Additional deceased patients can increase the exported count.

### Output

```text
llm/data/raw/synthea/<cohort>/
├── generation-settings.txt
├── California.log
├── California/fhir/*.json
└── ... other states and logs
```

Each run creates a new cohort directory. Files are bind-mounted at `/output` inside the container, persist after it exits, and are ignored by Git. Supporting FHIR resources mean file count is not patient count.

### Options

```bash
PATIENTS_PER_STATE=100 \
SEED=123 \
REFERENCE_DATE=20250101 \
COHORT_NAME=trial-500-v1 \
SYNTHEA_HEAP=4g \
./tools/generate_synthea.sh

./tools/generate_synthea.sh --help
```

- `PATIENTS_PER_STATE`: default `2000`; five states are fixed in the script.
- `SEED`: default `42`, used for patients and clinicians.
- `REFERENCE_DATE`: default `20250101`; `YYYYMMDD`, sets reference and end dates.
- `COHORT_NAME`: defaults to a timestamped name; must be new, using letters, digits, `_`, or `-`.
- `SYNTHEA_HEAP`: default `2g`; not a total container memory limit.

If execution permission is missing, use `bash tools/generate_synthea.sh`. On failure, inspect the printed state log. Partial output is retained; retry with a new cohort name. Check Docker access, output ownership, and available memory as needed.

Synthea outputs structured patient histories, not final note/JSON pairs. The preparation script in `llm/scripts/prepare_extraction_data.py` renders those pairs and assigns deterministic patient splits. Training and serving are separate from Synthea generation.

## References

- [Synthea commands and container details](tools/synthea/README.md)
- [Development plan](ANAMNESISLM_PLAN.md) and [diagram](ANAMNESISLM_PLAN.mmd)
- [Agent guidance](Agent.md) and [documentation style](Agents.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data strategy](docs/DATA_STRATEGY.md) and [policy](docs/DATA_POLICY.md)
- Component guides: [LLM](llm/README.md), [API](api/README.md), [web](web/README.md)

Learning reference: Sebastian Raschka's *Build a Large Language Model (From Scratch)*; the book is not bundled. Hand-written showcase examples are not model predictions or a blind test set.

## Containers and safety

Root `compose.yaml` provides `web`, `api`, and optional `synthea` (`tools` profile) services. Web/API ports bind to loopback only. PostgreSQL and training containers are not implemented. Do not expose this unauthenticated demo publicly.

Never expose restricted clinical data to online agents. Keep it outside the repository with verified access controls. Git/Docker ignore rules are not security boundaries. Use synthetic notes in demonstrations.

## License

[MIT](LICENSE): use, fork, modify, and redistribute while retaining the license notice. Third-party datasets, software, books, and model artifacts retain their own terms.
