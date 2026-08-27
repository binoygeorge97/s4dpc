# RUNNING

## Local (recommended for this study's current scale)

At the current toy scale (scalar plant, `d_model=8`, `N=16`, 20000 epochs),
the entire 8-arm ladder finishes in well under a minute on a laptop CPU.
**TACC is not currently necessary** — use it once this study scales to
larger architectures, more seeds, or a nonlinear plant (see `slurm/` below
for when that happens).

```bash
# from the repo root, with the s4dpc venv active (needs jax/flax/s4-nnx
# per the parent repo's requirements.lock)

# Experiment 1: skip/branch Jacobian decomposition on EXISTING s4dpc checkpoints
python -m layernorm_study.experiments.exp1_jacobian_decomposition

# Experiment 2, step 1: data sanity check (halts if least squares doesn't
# recover (A,B)=(3,1) to ~1e-9 - run this before trusting anything below)
python -m layernorm_study.experiments.exp2_data_sanity_check

# Experiment 2, step 2: the full complexity ladder (all 8 arms)
python -m layernorm_study.experiments.exp2_train_ladder

# ...or a subset (comma-separated, no spaces)
python -m layernorm_study.experiments.exp2_train_ladder --arms arm_0,arm_2
```

Outputs (all gitignored):
- `results/exp1_jacobian_decomposition.csv` — one row per (variant, case, seed)
- `results/exp2_ladder.csv` — one row per arm
- `results/exp2_<arm>_manifest.json` — full config + result for one arm
- `results/ckpt/<arm>.msgpack` — trained params for one arm (reloadable via
  `layernorm_study.src.arms.load_arm_model`, avoids retraining to rerun diagnostics)
- `figures/exp1_origin_sweep_<variant>_case1_seed0.png`
- `figures/exp2_<arm>_diagnostics.png`

## TACC (for when this study outgrows local/Kaggle CPU)

Same execution flow as the parent repo's `launch/tacc/` (see its README
and `job.slurm`): sync the repo (`launch/tacc/sync_tacc.sh`), then submit.

```bash
# submits one sbatch job per arm, all via the SAME run_arm.slurm script
layernorm_study/slurm/submit_all.sh <allocation> [partition] [arm1,arm2,...]

# check status
squeue -u $USER

# cancel a job
scancel <job_id>
```

Output lands in `$WORK/s4dpc-results/logs/layernorm-<arm>_<job_id>/` (same
`job_info.txt` convention as `launch/tacc/job.slurm`), plus this study's
own `results/`/`figures/` under whatever repo checkout the job ran from
(`$SLURM_SUBMIT_DIR`).

Per CLAUDE.md sec 12: never edit code on TACC, `python/3.12.11` via
`module load` only (never the default `python3`), and never split one
arm's seeds across two different backends.
