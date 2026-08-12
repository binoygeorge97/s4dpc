# s4dpc

Research code for a conference paper on why differentiable predictive
control (DPC) fails when the plant model is a learned S4 surrogate, even
when one-step prediction MSE is low.

Core claim under test: **control-relevant derivative fidelity**, not
prediction error, determines DPC success. Suspected mechanism: LayerNorm
(degree-0 homogeneous, so it cannot represent a linear map) distorts the
learned Jacobian near the regulation setpoint.

The full operating guide — golden rules, Kaggle/TACC workflow, W&B
conventions, the variant ladder — is [CLAUDE.md](CLAUDE.md). This file is
just a status snapshot.

## Status: identification pipeline is float64 by default; diagnostics.py built and validated; the M3 investigation is closed (real finding, not a float32 artifact); the LayerNorm corollary test needs a different construction before it says anything

**Done:**

- End-to-end CLI/logging/Kaggle harness — `env_probe.py`, `s4dpc/sweep.py`,
  `s4dpc/logging.py`, `launch/kaggle/` — proven on both Kaggle T4 and CPU.
- `legacy/` — a verbatim port of the original notebook's S4 implementation,
  the bit-exact parity target for the `s4-nnx`-based refactor.
- `s4dpc/systems.py` — 7 discrete-time LTI plants (`A:(6,6)`, `B:(6,3)`),
  canonical bilinear/Tustin discretization at `dt=0.01` (see
  `docs/DECISIONS.md` for the ZOH comparison and case-specific notes).
- `s4dpc/data.py` — APRBS-driven trajectory data generation.
- `s4dpc/blocks.py` + `s4dpc/model.py` — the configurable S4 block
  (`BlockConfig`: norm/activation/glu/prenorm/residual) and the
  `M3`-`M6_fix` variant ladder, consuming `s4-nnx` v0.2.0 directly. M6 is
  verified architecturally bit-identical to `legacy` (`tests/test_parity.py`).
  `tests/test_decode_construction_parity.py` checks decode=True/False
  params match at init; `tests/test_control_decode_parity.py` checks their
  *forward output* matches on trained (not just initialized) params.
- `s4dpc/identify.py` — teacher-forced one-step MSE identification,
  vmapped over (case x seed), plus `fit_least_squares` (the closed-form
  M1 floor every neural variant is judged against). **float64 by default**
  as of 2026-08-12 (`s4dpc/sweep.py` sets `jax_enable_x64` before any JAX
  op; `--no-x64` to opt out) - this required an explicit, complex-aware
  param-dtype cast after construction (`nnx.Linear`/`LayerNorm` default to
  float32 regardless of the global flag) and fixed a real M6_fix pipeline
  bug along the way (StaticNorm's `nnx.Variable` state crashed
  `nnx.split`-based training - M6_fix had never successfully completed an
  identification run before this). Wired into `s4dpc/sweep.py`, the only
  CLI entry point (`--variant --cases --n_seeds --epochs --d_model --N
  --n_layers`, `--no-vmap` for debugging).
- `s4dpc/diagnostics.py` — `markov_parameters` (realization-invariant,
  via autodiff through a free-running multi-step unroll - not a raw
  one-step Jacobian), `equilibrium_drift`, `local_linearity_defect`, and
  `jacobian_sweep` (the kink-figure diagnostic). Validated
  (`tools/validate_diagnostics.py`) against a model forced to realize the
  TRUE `A_d`/`B_d` exactly: every diagnostic lands at or near float64
  machine precision, and a sanity control on a fresh nonlinear model
  confirms they detect real curvature rather than trivially returning 0.
- `s4dpc/control.py` — bounded GRU controller (`u =
  max_action * tanh(...)`, unlike the unbounded reference it was adapted
  from), DPC loss, rollout through either the TRUE linear plant or a
  trained S4 surrogate (`decode=True`, stepped), and an oracle discrete-LQR
  solve. A smoke test (`tools/smoke_control.py`) confirmed the identify → control
  pipeline is wired correctly end to end (case 3: oracle LQR cost 235.9,
  GRU-on-true-plant 415.9, GRU-on-M3-surrogate 895.8, GRU-on-M6-surrogate
  878.7) — **not a real M3-vs-M6 comparison** (float32, both under-converged,
  pre-dating every finding below) — needs re-running once the controller
  is back in scope.
- `tests/` — legacy/M6 parity, decode-mode parity (construction and
  forward), identify vmap-vs-no-vmap equivalence, systems-table sanity
  (SVD-based measures, deliberately not `np.linalg.eig` — see
  `docs/DECISIONS.md`).
- `gpu_ledger.csv` — running log of Kaggle GPU minutes against the weekly
  quota.

**The M3 investigation is closed — see `docs/DECISIONS.md` for the full
dated narrative.** Short version: M3 (and every variant) sits many orders
above its own least-squares floor at the standard 2000-step Adam budget,
not because of capacity, the S4 recurrence, or a special S4 pathology,
but because D-only's parameterization is ~89% overparameterized (a fixed
60-DOF affine map realized through 550 raw params — the earlier "32 =
2×d_model" reading of this was a float32 rounding artifact, corrected to
the true ~490-dim null space) and Adam's per-coordinate normalization
turns that redundancy into an active, unbounded random walk along the
null space — 22 orders of magnitude divergence in a *single step* from
the exact optimum, confirmed (not weakened) in float64, vanishing almost
entirely under plain SGD. **Tested and did NOT hold:** the natural
corollary that LayerNorm should make this worse for M5/M6 — redundancy at
a random init goes the *wrong* direction (LayerNorm slightly reduces it),
and the dynamical (Adam-vs-SGD) test as built is structurally blind to
norm choice (the LS-init construction needed to get an exact starting
point happens to permanently zero-gradient the entire S4-kernel-and-norm
sub-path — a real, explained construction artifact, not a null result).
A valid version of that test needs a different starting-point
construction (e.g. genuinely optimizing each variant to a good point via
L-BFGS first) — not attempted, flagged as a real piece of new
infrastructure rather than built without go-ahead.

**Not started yet:** a paper-scale identification sweep across the full
variant ladder using `sweep.py`'s CSV path (every run so far, including
all the M3 investigation and diagnostics validation, has been a
standalone `tools/diagnose_*.py`/`validate_*.py` script, not a batch
`sweep.py` run), correlating `diagnostics.py` against per-case DPC
behavior, the full-curriculum controller training run, and `docs/LOG.md`
(not needed yet for the same reason — nothing has gone through the
CSV-producing batch path).

## Quick start

```bash
python env_probe.py                          # env/device/model canary
pytest tests/                                 # parity + systems + identify + control tests
python tools/make_reference_checkpoint.py     # regenerate the legacy fixture
python -m s4dpc.sweep --variant M3 --cases 3 --n_seeds 1 --epochs 2000 \
    --d_model 16 --N 32 --n_layers 1 --wandb off --out out.csv
```

Kaggle kernels live under `launch/kaggle*/` — one directory per kernel
(smoke test, checkpoint generation, and one per M3/variant-ladder
diagnostic in `docs/DECISIONS.md`'s narrative). See CLAUDE.md §4 for the
push/poll/pull workflow.
