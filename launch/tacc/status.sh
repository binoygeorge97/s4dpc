#!/usr/bin/env bash
# Run from EE-119537 (local orchestration machine).
#
# One-shot status snapshot of your TACC jobs: squeue, squeue --start,
# recent sacct history (state/elapsed/exit code), and paths to the
# stdout/stderr files for recent jobs. Not a monitoring daemon - run it
# again whenever you want a fresh look.
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
: "${TACC_REPO_DIR:?tacc.env: TACC_REPO_DIR not set}"

ssh "${TACC_USER}@${TACC_HOST}" bash -s <<EOF
set -euo pipefail
echo "=== squeue -u $TACC_USER ==="
squeue -u "$TACC_USER"
echo
echo "=== squeue -u $TACC_USER --start (estimated start times for pending jobs) ==="
squeue -u "$TACC_USER" --start
echo
echo "=== sacct -u $TACC_USER (last 3 days) ==="
sacct -u "$TACC_USER" -S \$(date -u -d '3 days ago' +%Y-%m-%d) \
  --format=JobID,JobName%20,Partition,State,Elapsed,ExitCode,Start,End -X
echo
echo "=== recent stdout/stderr files in $TACC_REPO_DIR ==="
if [ -d "$TACC_REPO_DIR" ]; then
  cd "$TACC_REPO_DIR"
  ls -lt slurm-*.out slurm-*.err 2>/dev/null | head -20 || echo "(none found yet)"
else
  echo "(repo not present at $TACC_REPO_DIR - run sync_tacc.sh first)"
fi
EOF
