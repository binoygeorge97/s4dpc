#!/usr/bin/env bash
# Run from EE-119537 (local orchestration machine).
#
# Submits approved production experiments to TACC (default: gpu-a100-small),
# using the SAME launch/tacc/job.slurm and the real s4dpc.sweep CLI. Scientific
# config is never embedded in this script or passed via `sbatch --export` -
# it is always the literal sweep CLI arguments you pass, printed back to you
# in the preview before anything is submitted.
#
# Two ways to call it:
#
#   1) One job:
#        ./submit_all.sh -- --variant M6 --cases 1,2,3,4,5,6,7 --n_seeds 5 \
#          --epochs 200 --d_model 16 --N 32 --n_layers 1 --wandb offline \
#          --out "$TACC_RESULTS_DIR/m6_full.csv"
#
#   2) Multiple independent configs/variants as separate Slurm jobs, one per
#      non-comment, non-blank line of a file you create (each line is the
#      full sweep-arg string for one job, e.g. a line reading
#      "--variant M3 --cases 1,2,3 --n_seeds 5 --epochs 200 --d_model 16 \
#       --N 32 --n_layers 1 --wandb offline --out $WORK/s4dpc-results/m3.csv"):
#        ./submit_all.sh --configs-file my_configs.txt
#
# Flags: --dry-run (print the plan, submit nothing), --partition NAME,
# --time HH:MM:SS (or D-HH:MM:SS).
#
# This script NEVER increases node count, partition, runtime, or concurrency
# beyond what you pass, and NEVER chains a follow-up experiment on its own.
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

PARTITION="${TACC_PROD_PARTITION:-gpu-a100-small}"
TIME_LIMIT=""
DRY_RUN=0
CONFIGS_FILE=""
SWEEP_ARGS=()

usage() {
  cat >&2 <<'USAGE'
Usage:
  submit_all.sh [--dry-run] [--partition NAME] [--time HH:MM:SS] -- <sweep args...>
  submit_all.sh [--dry-run] [--partition NAME] [--time HH:MM:SS] --configs-file FILE
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --configs-file) CONFIGS_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; SWEEP_ARGS=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

JOBS=()
if [[ -n "$CONFIGS_FILE" ]]; then
  [[ -f "$CONFIGS_FILE" ]] || { echo "ERROR: configs file not found: $CONFIGS_FILE" >&2; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" || "$line" == \#* ]] && continue
    JOBS+=("$line")
  done < "$CONFIGS_FILE"
elif [[ ${#SWEEP_ARGS[@]} -gt 0 ]]; then
  JOBS+=("${SWEEP_ARGS[*]}")
else
  echo "ERROR: pass sweep args after -- , or use --configs-file FILE" >&2
  usage
  exit 1
fi

echo "== 1/2: syncing TACC repo =="
"$SCRIPT_DIR/sync_tacc.sh"

echo
echo "== submission preview =="
echo "partition:   $PARTITION"
echo "allocation:  $TACC_ALLOCATION"
echo "time limit:  ${TIME_LIMIT:-<job.slurm default: 00:30:00 - pass --time to override>}"
echo "jobs:        ${#JOBS[@]}"
for i in "${!JOBS[@]}"; do
  echo "  [$i] sbatch -A $TACC_ALLOCATION -p $PARTITION ${TIME_LIMIT:+-t $TIME_LIMIT} launch/tacc/job.slurm ${JOBS[$i]}"
  if [[ "${JOBS[$i]}" != *"--out"* ]]; then
    echo "      WARNING: no --out in this job's args - s4dpc.sweep requires it and will fail fast."
  elif [[ "${JOBS[$i]}" != *'$WORK'* && "${JOBS[$i]}" != *"$TACC_RESULTS_DIR"* ]]; then
    echo "      NOTE: --out does not obviously point under \$WORK / \$TACC_RESULTS_DIR - durable"
    echo "            outputs must end up under \$WORK, not only \$SCRATCH (CLAUDE.md storage rule)."
  fi
done
echo

if [[ $DRY_RUN -eq 1 ]]; then
  echo "--dry-run: not submitting."
  exit 0
fi

read -r -p "Submit ${#JOBS[@]} job(s) to $PARTITION under allocation $TACC_ALLOCATION? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  echo "aborted."
  exit 1
fi

echo
echo "== 2/2: submitting =="
echo "   (you will be prompted again for TACC password and MFA)"

REMOTE_SCRIPT="set -euo pipefail
cd \"$TACC_REPO_DIR\""
for i in "${!JOBS[@]}"; do
  REMOTE_SCRIPT+=$'\n'"echo '--- job [$i] ---'"
  REMOTE_SCRIPT+=$'\n'"sbatch -A \"$TACC_ALLOCATION\" -p \"$PARTITION\" ${TIME_LIMIT:+-t \"$TIME_LIMIT\"} launch/tacc/job.slurm ${JOBS[$i]}"
done

JOB_OUTPUT="$(ssh "${TACC_USER}@${TACC_HOST}" bash -s <<EOF
$REMOTE_SCRIPT
EOF
)"

echo "$JOB_OUTPUT"

echo
echo "== submitted job IDs =="
echo "$JOB_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' || echo "(none parsed - check output above)"
