#!/usr/bin/env bash
# Driver: submits ONE sbatch job per arm, all using the SAME
# run_arm.slurm batch script (never one hand-written script per arm -
# CLAUDE.md sec 9's "no per-run scripts" principle, applied here).
#
# Usage (run from the TACC login node, inside the s4dpc repo checkout):
#   layernorm_study/slurm/submit_all.sh <allocation> [partition] [arm1,arm2,...]
#
# Defaults: partition=gpu-a100-dev (dev queue - this study's jobs are all
# tiny), arms=all 8 registered in layernorm_study/src/arms.py.
set -euo pipefail

ALLOCATION="${1:?usage: submit_all.sh <allocation> [partition] [arms_csv]}"
PARTITION="${2:-gpu-a100-dev}"
ARMS_CSV="${3:-arm_0,arm_1,arm_2,arm_3,arm_4,arm_5,arm_6,arm_7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -ra ARMS <<< "$ARMS_CSV"
for arm in "${ARMS[@]}"; do
  echo "submitting $arm ..."
  sbatch -A "$ALLOCATION" -p "$PARTITION" -t 00:15:00 \
    --job-name "layernorm-${arm}" \
    "$SCRIPT_DIR/run_arm.slurm" --arms "$arm"
done
