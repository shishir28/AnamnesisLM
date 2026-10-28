# AnamnesisLM

A small decoder-only Transformer, trained from scratch to extract evidence-backed facts from synthetic clinical notes.

**Status:** Synthea data generation works. Model, API, and web app are not implemented. Built for learning; not validated for clinical use.

## Structure

| Folder | Purpose |
|---|---|
| `llm/` | Data preparation, tokenizer, model, training, evaluation, inference |
| `api/` | FastAPI model hosting and PostgreSQL access |
| `web/` | Next.js interface |
| `tools/` | Synthea container and generation script |
| `docs/` | Architecture and data policies |

Planned request flow: Next.js → FastAPI → PyTorch model. Only FastAPI accesses PostgreSQL. Datasets and checkpoints stay in files, not database rows.

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

Synthea outputs structured patient histories, not our final note/JSON pairs. Next: write the Python reader and renderer, validate evidence offsets, and split by patient before generating examples.

## References

- [Synthea commands and container details](tools/synthea/README.md)
- [Development plan](ANAMNESISLM_PLAN.md) and [diagram](ANAMNESISLM_PLAN.mmd)
- [Agent guidance](Agent.md) and [documentation style](Agents.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data strategy](docs/DATA_STRATEGY.md) and [policy](docs/DATA_POLICY.md)
- Component guides: [LLM](llm/README.md), [API](api/README.md), [web](web/README.md)

Learning reference: Sebastian Raschka's *Build a Large Language Model (From Scratch)*; the book is not bundled. Hand-written showcase examples are not model predictions or a blind test set.

## Containers and safety

Root `compose.yaml` currently contains only the optional Synthea `tools` service. Web, API, PostgreSQL, and training services will follow their implementations.

Never expose restricted clinical data to online agents. Keep it outside the repository with verified access controls. Git/Docker ignore rules are not security boundaries. Use synthetic notes in demonstrations.

## License

[MIT](LICENSE): use, fork, modify, and redistribute while retaining the license notice. Third-party datasets, software, books, and model artifacts retain their own terms.
