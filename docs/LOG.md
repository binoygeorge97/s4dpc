# LOG.md — s4dpc

Append-only, one entry per batch of runs. Three coordinates (sha, wandb
group, csv path), then what was concluded and any anomaly. See
`docs/DECISIONS.md` for the reasoning behind anything that changes what
enters the paper — this file is the run log, not the argument.

---

## 2026-08-12 — M3/M6 identification, all 7 cases x 3 seeds, float64, first real sweep.py batch

sha: 664964c | wandb: off | results/all_cases/M3.csv, results/all_cases/M6.csv

`python -m s4dpc.sweep --variant {M3,M6} --cases 1,2,3,4,5,6,7 --n_seeds 3
--epochs 40000 --d_model 16 --N 32 --n_layers 1 --wandb off --out
results/all_cases/{variant}.csv`. First time in the whole investigation
that the actual `sweep.py`/`identify.py` CSV+checkpoint path has been
exercised at any real scale (everything before this was standalone
`tools/diagnose_*.py` scripts) - required adding `@nnx.jit` to
`_train_one`/`_train_ensemble`'s per-step update first (eager dispatch
was fine at every previous epoch count, not at 40k x a 21-way vmapped
ensemble); verified safe via `tools/pilot_jit_check.py` before committing
to the full run (vmap vs `--no-vmap` agreement at 1e-6 to 1e-12 relative
precision on both variants).

Checkpoints -> `tools/diagnose_all_cases.py` -> Markov-parameter error
and equilibrium drift per (variant, case, seed) against that case's own
`(A_d, B_d)`, correlated against Kreiss-like transient amplification
(`tests/test_systems.py`) rather than spectral radius (flat ~1.02-1.04
across every case here, so it cannot explain per-case variation the way
Kreiss - spanning ~1 to 330 - could). Full reasoning and numbers in
`docs/DECISIONS.md`'s 2026-08-12 "Task 3" entry.

**Conclusion:** the raw all-cases-included correlation (+1.0000, both
variants) is misleading on its own - almost entirely driven by case 6
being a simultaneous extreme outlier on both axes. Recomputed with
case 6 excluded and using the seed-median (robust to a single diverged
run): M3 shows no relationship (-0.39) - its Markov error looks
optimization-noise-dominated rather than tied to plant dynamics, except
at the true extreme. M6 shows a real one (+0.96) that survives removing
the leverage point.

**Anomaly, tagged not deleted (CLAUDE.md sec 3 rule 6):** M3 case6/seed2
(markov_rel_err_mean=1.178e+11) and M6 case6/seed2
(markov_rel_err_mean=3.775e+11) are diverged runs, not low-quality
fits - 8-11 orders of magnitude beyond their own case's other two seeds,
matching this document's established Adam/overparameterization
divergence signature. Milder elevation also present: M6 case4/seed0
(~6.6x its case median) and case7/seed2 (~30x). Not rerun this batch;
3 seeds/case is not enough to separate "real effect" from "got unlucky
in a way that correlates with Kreiss" with high confidence - more seeds
or a targeted rerun of the case-6 outliers with fresh keys would
strengthen this either way, flagged as the natural next step.

GPU: 110.98s (M3 training) + 125.56s (M6 training) + diagnostics,
18.23 T4-min total for this kernel; 1.77 T4-min for the jit pilot that
gated it. Both logged in `gpu_ledger.csv`.
