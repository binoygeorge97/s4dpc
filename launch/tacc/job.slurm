#!/bin/bash
# Platform/execution logic ONLY. All scientific config (variant, cases,
# seeds, epochs, architecture, --wandb mode, --out path) comes from the CLI
# arguments passed after this script - CLAUDE.md §3 rule 1: sweep.py is the
# only entry point, and there is no platform-specific training code.
#
# Usage (never invoke bare - allocation/partition/time are supplied by the
# caller, not hard-coded here):
#   sbatch -A <allocation> -p gpu-a100-dev  -t 00:30:00 launch/tacc/job.slurm --variant M1 --cases 1 ...
#   sbatch -A <allocation> -p gpu-a100-small -t 8:00:00  launch/tacc/job.slurm --variant M6 --cases 1,2,3,4,5,6,7 ...
#
# One node, one task: JAX vmap over (case x seed) happens inside this single
# process (CLAUDE.md §9) - no multiprocessing/Ray/joblib/per-seed jobs here.
#SBATCH --job-name=s4dpc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 00:30:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
# Deliberately NOT set here (per operator instructions):
#   -A / --account        : supplied by the caller (sbatch -A ...), never hard-coded
#   --gres, --gpus-per-task: not used - the gpu-a100-* partitions grant node-level GPU access

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${TACC_VENV_DIR:-$SCRATCH/venvs/s4dpc}"
RESULTS_ROOT="${TACC_RESULTS_DIR:-$WORK/s4dpc-results}"
LOG_DIR="$RESULTS_ROOT/logs/${SLURM_JOB_NAME:-s4dpc}_${SLURM_JOB_ID:-nojobid}"
mkdir -p "$LOG_DIR"
INFO_LOG="$LOG_DIR/job_info.txt"

GIT_SHA="$(git rev-parse --short HEAD)"
LOCKFILE_SHA="$(sha256sum requirements.lock | awk '{print $1}' | cut -c1-12)"

{
  echo "hostname:             $(hostname)"
  echo "date:                 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SLURM_JOB_ID:         ${SLURM_JOB_ID:-unset}"
  echo "partition:            ${SLURM_JOB_PARTITION:-unset}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git_sha:              $GIT_SHA"
  echo "lockfile_sha:         $LOCKFILE_SHA"
} | tee "$INFO_LOG"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR - run launch/tacc/build_env.sh first (inside a GPU allocation)." >&2
  exit 1
fi

# A batch job's inherited module/library environment is not guaranteed to
# match build_env.sh's - confirmed directly: the same venv reported Python
# 3.12.13 under a contaminated inherited environment vs. the correct
# 3.12.11 after this reset. Establish module state explicitly rather than
# trusting whatever the submitting shell happened to have loaded (also
# TACC's own documented recommendation for batch scripts).
module reset
module load python/3.12.11

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

{
  echo "python bin:           $(which python)"
  echo "python:               $(python --version 2>&1)"
  echo "jax devices:          $(python -c 'import jax; print(jax.devices())')"
} | tee -a "$INFO_LOG"

echo "--- launching: python -m s4dpc.sweep $* ---" | tee -a "$INFO_LOG"

# The only line that actually runs the experiment. Diverged runs are
# results, not failures (CLAUDE.md §3 rule 6) - do not swallow a non-zero
# exit here, and never delete anything sweep.py wrote on the way out.
python -m s4dpc.sweep "$@"

echo "job.slurm done." | tee -a "$INFO_LOG"
