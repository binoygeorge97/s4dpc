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

## Status: identification pipeline built and diagnosed; controller pipeline wired and smoke-tested; M3's convergence behavior is an active, not-yet-closed investigation

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
  vmapped over (case × seed), plus `fit_least_squares` (the closed-form
  M1 floor every neural variant is judged against). Wired into
  `s4dpc/sweep.py`, the only CLI entry point (`--variant --cases --n_seeds
  --epochs --d_model --N --n_layers`, `--no-vmap` for debugging).
- `s4dpc/control.py` — bounded GRU controller (`u = max_action *
  tanh(...)`, unlike the unbounded reference it was adapted from), DPC
  loss, rollout through either the TRUE linear plant or a trained S4
  surrogate (`decode=True`, stepped), and an oracle discrete-LQR solve.
  A smoke test (`tools/smoke_control.py`) confirmed the identify → control
  pipeline is wired correctly end to end (case 3: oracle LQR cost 235.9,
  GRU-on-true-plant 415.9, GRU-on-M3-surrogate 895.8, GRU-on-M6-surrogate
  878.7) — **not yet a real M3-vs-M6 comparison**, see the confound note
  below.
- `tests/` — legacy/M6 parity, decode-mode parity (construction and
  forward), identify vmap-vs-no-vmap equivalence, systems-table sanity
  (SVD-based measures, deliberately not `np.linalg.eig` — see
  `docs/DECISIONS.md`).
- `gpu_ledger.csv` — running log of Kaggle GPU minutes against the weekly
  quota.

**Active investigation (see `docs/DECISIONS.md` for the full, dated
narrative — this is deliberately not summarized further here since it's
still moving):** M3, intended as the "capacity is fine" LTI control for
the variant ladder, does not reach anywhere near the least-squares floor
it can provably represent (~1e-14) under the standard identification
budget — it sits 9-13 orders of magnitude above it. Ruled out so far:
capacity, the S4 recurrence, target/input scale, and severe data
ill-conditioning. A plain unfactored linear layer under the same Adam
budget *also* fails to reach the floor, while D-only's Gauss-Newton
matrix is independently, exactly rank-deficient by 32 = 2×d_model (a
genuine parameter non-identifiability, not just slow conditioning) that
destabilizes training even when initialized exactly at the optimum. The
two mechanisms appear to compound rather than being alternatives to each
other; the chain of experiments establishing this is still open as of
the last DECISIONS.md entry. **Implication: any M3-vs-M4/M5/M6 comparison
run before this is resolved (including the controller smoke test above)
should not yet be read as evidence about the paper's central LayerNorm
hypothesis** — M3 itself isn't at a trustworthy reference point yet.

**Not started yet:** `s4dpc/diagnostics.py` (Markov params, gradient
cosine, equilibrium drift), a real paper-scale identification sweep
across the full variant ladder, the full-curriculum (not smoke-test)
controller training run, and `docs/LOG.md` (not needed yet — nothing has
gone through the CSV-producing `sweep.py` batch path so far; every run
so far has been a standalone `tools/diagnose_*.py`/`smoke_*.py` script).

## Quick start

```bash
python env_probe.py                          # env/device/model canary
pytest tests/                                 # parity + systems + identify + control tests
python tools/make_reference_checkpoint.py     # regenerate the legacy fixture
python -m s4dpc.sweep --variant M3 --cases 3 --n_seeds 1 --epochs 2000 \
    --d_model 16 --N 32 --n_layers 1 --wandb off --out out.csv
```

Kaggle kernels live under `launch/kaggle*/` — one directory per kernel
(smoke test, checkpoint generation, and one per M3 diagnostic/controller
run in the active investigation above). See CLAUDE.md §4 for the
push/poll/pull workflow.
