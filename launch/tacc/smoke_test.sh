#!/usr/bin/env bash
# Run from EE-119537 (local orchestration machine).
#
# Smoke-tests the TACC execution path end-to-end: sync repo -> submit a
# tiny job to gpu-a100-dev using the SAME launch/tacc/job.slurm, invoking
# the real s4dpc.sweep CLI (one case, one seed, few epochs, --wandb off).
# No separate smoke-test science code - this is the identical
# `python -m s4dpc.sweep` path production jobs use, just with small numbers.
#
# Authentication is interactive (TACC password + MFA at the ssh prompt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TACC_ENV_FILE:-$SCRIPT_DIR/tacc.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found." >&2
  echo "Copy launch/tacc/tacc.env.example -> launch/tacc/tacc.env and fill in your values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${TACC_USER:?tacc.env: TACC_USER not set}"
: "${TACC_HOST:?tacc.env: TACC_HOST not set}"
: "${TACC_ALLOCATION:?tacc.env: TACC_ALLOCATION not set}"
: "${TACC_REPO_DIR:?tacc.env: TACC_REPO_DIR not set}"

PARTITION="${TACC_SMOKE_PARTITION:-gpu-a100-dev}"

# Deliberately tiny: 1 case, 1 seed, 2 epochs, tiny architecture. M3 (not
# M1) so this actually exercises the S4/JAX training path, not just the
# least-squares baseline. Exercises the real sweep.py CLI and
# CSV-writing path, nothing more.
SMOKE_SWEEP_ARGS=(--variant M3 --cases 1 --n_seeds 1 --epochs 2 --d_model 8 --N 8 --n_layers 1 --wandb off)

echo "== 1/2: syncing TACC repo =="
"$SCRIPT_DIR/sync_tacc.sh"

echo
echo "== 2/2: submitting smoke job =="
echo "   partition:  $PARTITION"
echo "   allocation: $TACC_ALLOCATION"
echo "   sweep args: ${SMOKE_SWEEP_ARGS[*]} --out <computed on TACC under \$WORK/s4dpc-results/smoke/>"
echo "   (you will be prompted again for TACC password and MFA)"

JOB_OUTPUT="$(ssh "${TACC_USER}@${TACC_HOST}" bash -s <<EOF
set -euo pipefail
cd "$TACC_REPO_DIR"
TS=\$(date -u +%Y%m%dT%H%M%SZ)
OUT_CSV="\${TACC_RESULTS_DIR:-\$WORK/s4dpc-results}/smoke/smoke_\${TS}.csv"
echo "smoke --out: \$OUT_CSV"
sbatch -A "$TACC_ALLOCATION" -p "$PARTITION" -t 00:30:00 launch/tacc/job.slurm \
  ${SMOKE_SWEEP_ARGS[*]} --out "\$OUT_CSV"
EOF
)"

echo "$JOB_OUTPUT"

JOB_ID="$(echo "$JOB_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' || true)"
if [[ -n "$JOB_ID" ]]; then
  echo
  echo "== smoke job submitted: $JOB_ID =="
  echo "Check progress with: ./launch/tacc/status.sh"
  echo "Pull the CSV once finished with: ./launch/tacc/pull_results.sh"
else
  echo "WARNING: could not parse a job ID from sbatch output above." >&2
fi
