#!/usr/bin/env bash
# Run from EE-119537 (local orchestration machine).
#
# Retrieves durable CSVs/logs from $TACC_RESULTS_DIR on TACC into a locally
# gitignored results directory, preserving directory structure. Checkpoint
# .msgpack files are excluded by default (they can be large and are not
# needed to read CSV results); pass --with-checkpoints to include them.
# Uses `rsync --update`, which never overwrites a local file that is newer
# than the remote copy.
#
# Usage:
#   ./pull_results.sh [local_dest_dir] [--with-checkpoints]
# Default local_dest_dir: <repo root>/results/tacc (already covered by
# .gitignore's `results/` pattern).
#
# Authentication is interactive (TACC password + MFA at the ssh/rsync prompt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
: "${TACC_RESULTS_DIR:?tacc.env: TACC_RESULTS_DIR not set}"

LOCAL_DEST="$REPO_ROOT/results/tacc"
WITH_CKPT=0

for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CKPT=1 ;;
    *) LOCAL_DEST="$arg" ;;
  esac
done

mkdir -p "$LOCAL_DEST"

RSYNC_EXCLUDES=()
if [[ $WITH_CKPT -eq 0 ]]; then
  RSYNC_EXCLUDES=(--exclude '*.msgpack')
  echo "excluding checkpoint .msgpack files (pass --with-checkpoints to include them)"
fi

echo "== pulling ${TACC_USER}@${TACC_HOST}:${TACC_RESULTS_DIR%/}/ -> $LOCAL_DEST/ =="
echo "   (you will be prompted for your TACC password and MFA code)"
echo "   --update: never overwrites a local file newer than its remote copy"

rsync -avz --update "${RSYNC_EXCLUDES[@]}" \
  "${TACC_USER}@${TACC_HOST}:${TACC_RESULTS_DIR%/}/" "$LOCAL_DEST/"

echo
echo "done. results under: $LOCAL_DEST"
