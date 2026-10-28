#!/usr/bin/env bash
# Generate a geographically varied cohort without requiring local Java.
set -Eeuo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: ./tools/generate_synthea.sh

Defaults: 2,000 requested patients per state (10,000 total), across
California, Texas, Florida, Massachusetts, and New York.

Optional environment variables:
  PATIENTS_PER_STATE=2000
  SEED=42
  REFERENCE_DATE=20250101       # YYYYMMDD; also used as the simulation end date
  COHORT_NAME=my-new-cohort    # Must be a new directory name
  SYNTHEA_HEAP=2g              # Maximum Java heap, not total container memory

Example small run:
  PATIENTS_PER_STATE=2 ./tools/generate_synthea.sh

Outputs: llm/data/raw/synthea/<cohort>/<state>/
A timestamped cohort name is generated unless COHORT_NAME is provided.
HELP
  exit 0
fi
if (( $# > 0 )); then
  echo 'Unknown argument. Use --help for usage.' >&2
  exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PATIENTS_PER_STATE="${PATIENTS_PER_STATE:-2000}"
SEED="${SEED:-42}"
REFERENCE_DATE="${REFERENCE_DATE:-20250101}"
[[ "$PATIENTS_PER_STATE" =~ ^[1-9][0-9]{0,5}$ ]] || {
  echo 'PATIENTS_PER_STATE must be an integer from 1 to 999999.' >&2; exit 2;
}
[[ "$SEED" =~ ^[0-9]{1,9}$ ]] || {
  echo 'SEED must contain 1–9 digits.' >&2; exit 2;
}
[[ "$REFERENCE_DATE" =~ ^[0-9]{8}$ ]] || {
  echo 'REFERENCE_DATE must use YYYYMMDD format.' >&2; exit 2;
}

states=(California Texas Florida Massachusetts 'New York')
total=$((10#$PATIENTS_PER_STATE * ${#states[@]}))
COHORT_NAME="${COHORT_NAME:-cohort-${total}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ "$COHORT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
  echo 'COHORT_NAME must contain only letters, digits, underscores, and hyphens.' >&2; exit 2;
}

command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"

OUTPUT_DIR="$ROOT_DIR/llm/data/raw/synthea/$COHORT_NAME"
mkdir -p "$ROOT_DIR/llm/data/raw/synthea"
# Do not reuse a cohort: identical reruns can trigger export collisions.
mkdir "$OUTPUT_DIR" || {
  echo 'Cannot create cohort directory. Choose a new COHORT_NAME; existing output is not overwritten.' >&2
  exit 1
}
trap 'echo "Generation failed. Partial output is retained at: $OUTPUT_DIR" >&2' ERR
printf 'synthea_version=4.0.0\nrequested_patients=%s\npatients_per_state=%s\npatient_seed=%s\nclinician_seed=%s\nreference_date=%s\nend_date=%s\nstates=%s\n' \
  "$total" "$PATIENTS_PER_STATE" "$SEED" "$SEED" "$REFERENCE_DATE" "$REFERENCE_DATE" \
  "${states[*]}" > "$OUTPUT_DIR/generation-settings.txt"

docker compose --profile tools build synthea
for state in "${states[@]}"; do
  folder="${state// /_}"
  log="$OUTPUT_DIR/$folder.log"
  echo "Generating $PATIENTS_PER_STATE requested patients in $state. Log: $log"
  docker compose --profile tools run --rm synthea \
    -p "$PATIENTS_PER_STATE" -s "$SEED" -cs "$SEED" \
    -r "$REFERENCE_DATE" -e "$REFERENCE_DATE" \
    "--exporter.baseDirectory=/output/$COHORT_NAME/$folder" \
    --exporter.fhir.export=true \
    --exporter.csv.export=false \
    "$state" > "$log" 2>&1
  # Upstream can report export failures without a nonzero exit status.
  if grep -Eq 'Exception|ERROR' "$log"; then
    echo "Possible generation/export failure; inspect $log. Partial output retained." >&2
    exit 1
  fi
  echo "Completed $state."
done

echo "Finished. Output: $OUTPUT_DIR"
echo 'Exported patient counts may exceed the requested population. Inspect records before training.'
