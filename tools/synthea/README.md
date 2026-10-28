# Synthea — synthetic patient generator

This repository builds its own Java 17 container around the upstream **Synthea v4.0.0** release JAR. The download is verified against its published SHA-256. This is not an official upstream Docker image. Docker downloads Java and the JAR during the build; no local Java installation is needed.

## Generate a first sample

Run from the repository root with Docker Engine and Docker Compose installed:

```bash
# Create the mount as your own user before Compose starts.
mkdir -p llm/data/raw/synthea
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

docker compose --profile tools build synthea
docker compose --profile tools run --rm synthea
```

Defaults: 100 requested patients, patient and clinician seeds `42`, reference and end dates `20250101`, FHIR JSON enabled, CSV disabled. Fixed dates and timezone reduce time-dependent variation; keep the release, complete settings, and generated hashes in the data manifest. A seed alone is not a promise of byte-identical exports.

The container runs without runtime networking and writes to ignored `llm/data/raw/synthea/`, typically under `fhir/`. Compose bind-mounts this repository-relative host folder at `/output` in the container, so generated files remain available after the container exits. Records may include practitioner and organization resources as well as patients; do not equate file count with patient count. Synthea's population/overflow behavior may also produce additional records.

## Small smoke run or custom population

Arguments after the service name **replace**, rather than append to, the default command. Preserve the output-directory setting:

```bash
docker compose --profile tools run --rm synthea \
  -p 1 -s 42 -cs 42 -r 20250101 -e 20250101 \
  --exporter.baseDirectory=/output \
  --exporter.fhir.export=true \
  --exporter.csv.export=false
```

Change `-p 1` to your desired population. Use a separate output directory for each experiment to avoid mixing datasets; the tool does not clean previous exports. Repeating the same settings into the same directory can trigger `FileAlreadyExistsException`, even if the process exits successfully. Check logs and exported files, not just the exit code. To inspect upstream command-line options:

```bash
docker compose --profile tools run --rm synthea --help
```

## Resources and permissions

- Default Java maximum heap: `2g`. Override, for example, with `export SYNTHEA_HEAP=4g`. Heap is not the container's total memory limit; provide additional memory for the JVM and exports.
- Compose defaults to UID/GID `1000:1000`. On Linux export your actual IDs as shown above to avoid permission failures. Docker Desktop manages bind-mount permissions differently.
- The image itself defaults to non-root UID `10001`; Compose overrides this for local output ownership.
- This is an optional `tools` profile, not a continuously running service. `run --rm` removes the container, not generated files.
- The Java base image uses a version tag, not an immutable digest. Pin its digest as well if strict image reproducibility is needed.

## Next step

Inspect a handful of FHIR bundles. Then implement our own Python reader and renderer to create note/JSON extraction pairs. Synthea does not supply our final training paragraphs, annotation schema, or negation/family/uncertainty coverage automatically.

Record dataset provenance and applicable upstream terms. The project MIT License does not replace Synthea's or other third-party terms. See [data strategy](../../docs/DATA_STRATEGY.md).

Upstream: https://github.com/synthetichealth/synthea
