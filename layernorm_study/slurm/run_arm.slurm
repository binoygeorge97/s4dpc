#!/bin/bash
# Platform/execution logic ONLY - mirrors launch/tacc/job.slurm's
# structure exactly (module reset + load, SLURM_SUBMIT_DIR as repo root,
# same venv). All scientific config (which arm(s), seeds) comes from the
# CLI arguments passed after this script.
#
# Usage (never invoke bare - allocation/partition/time supplied by the
# caller, same convention as launch/tacc/job.slurm):
#   sbatch -A <allocation> -p gpu-a100-dev -t 00:15:00 \
#       layernorm_study/slurm/run_arm.slurm --arms arm_2
#
# NOTE (2026-08-27): at this study's current toy scale (d_model=8, N=16,
# scalar plant, 20000 epochs), the entire ladder (8 arms) finishes in
# well under a minute on a laptop CPU - TACC is NOT currently necessary
# and local/Kaggle CPU is the right place to iterate (CLAUDE.md sec 13's
# "local GPU/CPU for development, Kaggle/TACC for results-that-matter"
# split still applies once this study scales to larger architectures,
# more seeds, or a nonlinear plant - this script exists for that point,
# not because it's needed today).
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/.git" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not a valid s4dpc git checkout: ${REPO_ROOT:-<unset>}" >&2
  exit 1
fi
cd "$REPO_ROOT"

VENV_DIR="${TACC_VENV_DIR:-$SCRATCH/venvs/s4dpc}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR - run launch/tacc/build_env.sh first (inside a GPU allocation)." >&2
  exit 1
fi

module reset
module load python/3.12.11
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "--- launching: python -m layernorm_study.experiments.exp2_train_ladder $* ---"
python -m layernorm_study.experiments.exp2_train_ladder "$@"
echo "run_arm.slurm done."
