#!/usr/bin/env bash
# Generate a geographically varied cohort without requiring local Java.
set -Eeuo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: ./tools/generate_synthea.sh

Defaults: 500 requested patients per state across all 50 US states (25,000 total).
Each state gets its own seed (SEED + state index), so patients are unique across states.

Optional environment variables:
  PATIENTS_PER_STATE=500
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

PATIENTS_PER_STATE="${PATIENTS_PER_STATE:-500}"
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

states=(
  Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia
  Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland
  Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada
  'New Hampshire' 'New Jersey' 'New Mexico' 'New York' 'North Carolina' 'North Dakota'
  Ohio Oklahoma Oregon Pennsylvania 'Rhode Island' 'South Carolina' 'South Dakota'
  Tennessee Texas Utah Vermont Virginia Washington 'West Virginia' Wisconsin Wyoming
)
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
printf 'synthea_version=4.0.0\nrequested_patients=%s\npatients_per_state=%s\nbase_seed=%s\nstate_seed=base_seed+state_index\nreference_date=%s\nend_date=%s\nstates=%s\n' \
  "$total" "$PATIENTS_PER_STATE" "$SEED" "$REFERENCE_DATE" "$REFERENCE_DATE" \
  "${states[*]}" > "$OUTPUT_DIR/generation-settings.txt"

docker compose --profile tools build synthea
for i in "${!states[@]}"; do
  state="${states[$i]}"
  folder="${state// /_}"
  log="$OUTPUT_DIR/$folder.log"
  # A shared seed makes Synthea generate the same patient ids in every state.
  state_seed=$((10#$SEED + i))
  echo "Generating $PATIENTS_PER_STATE requested patients in $state (seed $state_seed). Log: $log"
  docker compose --profile tools run --rm synthea \
    -p "$PATIENTS_PER_STATE" -s "$state_seed" -cs "$state_seed" \
    -r "$REFERENCE_DATE" -e "$REFERENCE_DATE" \
    "--exporter.baseDirectory=/output/$COHORT_NAME/$folder" \
    --exporter.fhir.export=true \
    --exporter.csv.export=false \
    "$state" > "$log" 2>&1
  # Upstream can report export failures without a nonzero exit status.
  exported=$(find "$OUTPUT_DIR/$folder/fhir" -name '*.json' 2>/dev/null | wc -l)
  if (( exported == 0 )); then
    echo "No FHIR files exported for $state; inspect $log. Partial output retained." >&2
    exit 1
  fi
  # Synthea sometimes fails a single patient (e.g. a NullPointerException) and carries on.
  failures=$(grep -cE 'Exception|ERROR' "$log" || true)
  if (( failures > 0 )); then
    echo "Warning: $failures error line(s) in $log; $exported FHIR files exported anyway." >&2
  fi
  echo "Completed $state ($exported FHIR files)."
done

echo "Finished. Output: $OUTPUT_DIR"
echo 'Exported patient counts may exceed the requested population. Inspect records before training.'
