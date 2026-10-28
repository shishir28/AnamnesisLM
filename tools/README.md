# Development tools

Optional utilities used to prepare data or support development; not application services.

- [Synthea](synthea/README.md): containerized synthetic patient generation.
- `generate_synthea.sh`: generate 10,000 requested patients across five states, with configurable population and a fresh output folder on each run.

```bash
./tools/generate_synthea.sh

# Small trial: two requested patients per state
PATIENTS_PER_STATE=2 ./tools/generate_synthea.sh

# See all options
./tools/generate_synthea.sh --help
```

The script can also be invoked by absolute path from another directory. It builds the image, runs the states sequentially, and keeps settings and per-state logs under ignored `llm/data/raw/synthea/<cohort>/`. Age, sex, and race/ethnicity follow Synthea's demographic distributions; balanced quotas are not guaranteed. Inspect logs and records before using the output.

Use the repository-root `compose.yaml` as the shared Compose entry point. Web, API, and PostgreSQL services will be added as those components are implemented.
