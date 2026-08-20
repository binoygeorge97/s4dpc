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

## TACC / Lonestar6 execution layer

Infrastructure-only integration, added and validated 2026-08-18/19. This
section is the durable, high-level record of how it was made to work and
what's been proven — script-by-script usage is
[launch/tacc/README.md](launch/tacc/README.md); operating rules Claude
Code must follow are [CLAUDE.md](CLAUDE.md) §12. Do not re-derive any of
this from scratch — read here first.

### 1. Machine roles

| Machine | Role | Repo path |
|---|---|---|
| **P53 ThinkPad / WSL** | Primary scientific development. Normal branch `main`. Scientific changes are developed and pushed here. | `~/projects/s4dpc` |
| **GitHub** | Canonical remote. `main` is the permanent branch — `infra/tacc` (used to develop/validate this integration) has already merged into it. | `https://github.com/binoygeorge97/s4dpc.git` |
| **Ubuntu lab desktop `EE-119537`** (user `dnc`) | Orchestration machine for TACC — runs sync/submit/status/pull. Must never become a separate scientific fork. | `~/Documents/Github/s4dpc` |
| **TACC Lonestar6** (user `binoygeorge`, host `ls6.tacc.utexas.edu`, allocation `IRI26016`) | Execution platform only. `$WORK=/work/11460/binoygeorge/ls6`, `$SCRATCH=/scratch/11460/binoygeorge`. Repo at `$WORK/s4dpc`, venv at `$SCRATCH/venvs/s4dpc`. Dev queue `gpu-a100-dev`, production queue `gpu-a100-small`. | `$WORK/s4dpc` |

**TACC is only an execution platform wrapper.** There is no separate
scientific-code fork for TACC — the same `s4dpc/` package that runs on
Kaggle and P53 runs there unmodified.

### 2. Single scientific entrypoint

```
python -m s4dpc.sweep
```

is the one experiment entrypoint across local, Kaggle, and TACC
workflows (CLAUDE.md §3 rule 1). `launch/tacc/job.slurm` is a
platform-only wrapper around this exact command — scientific logic
lives in the Python package, never duplicated into Slurm scripts.

### 3. Configuration

`launch/tacc/tacc.env` is local, machine-specific, and **gitignored** —
create it from the tracked template:

```bash
cp launch/tacc/tacc.env.example launch/tacc/tacc.env
```

Typical `EE-119537` values:

```
TACC_USER=binoygeorge
TACC_HOST=ls6.tacc.utexas.edu
TACC_ALLOCATION=IRI26016
TACC_REPO_DIR=/work/11460/binoygeorge/ls6/s4dpc
TACC_RESULTS_DIR=/work/11460/binoygeorge/ls6/s4dpc-results
TACC_VENV_DIR=/scratch/11460/binoygeorge/venvs/s4dpc
TACC_SMOKE_PARTITION=gpu-a100-dev
TACC_PROD_PARTITION=gpu-a100-small
TACC_GIT_REF=main
```

The integration was *developed and validated* on branch `infra/tacc`
(so it could be proven out on real hardware before touching `main`) —
`TACC_GIT_REF=infra/tacc` during that phase. `infra/tacc` has since
merged into `main`; the **permanent, intended** value going forward is
`TACC_GIT_REF=main`.

### 4. Standard workflow

From `EE-119537`:

```bash
./launch/tacc/sync_tacc.sh      # fetch + checkout + fast-forward-only TACC's $WORK/s4dpc to TACC_GIT_REF
./launch/tacc/smoke_test.sh     # sync, then submit a tiny M3/1-case/1-seed/2-epoch job to gpu-a100-dev
./launch/tacc/status.sh         # squeue / squeue --start / sacct snapshot for this user
./launch/tacc/pull_results.sh   # rsync durable CSVs/logs back (checkpoints excluded by default)
./launch/tacc/submit_all.sh -- --variant ... --cases ... ...   # production, gpu-a100-small
```

`submit_all.sh` never invents scientific configs — it only submits the
exact sweep CLI arguments you pass after `--`, previewed before
submission, and only after explicit confirmation. Use it only once
scientific configs (variant, cases, seeds, epochs, architecture,
`--wandb` mode) are explicitly chosen.

### 5. Authentication

TACC SSH requires password + a 6-digit MFA code, so TACC-touching
commands are normally run **by the human**, directly in the VS Code
integrated terminal — not from a non-interactive Claude Code shell,
which has no real TTY for the password/MFA prompt to appear on
(confirmed directly: a bare `ssh`/`sync_tacc.sh` invocation from the
agent's Bash tool fails with `ssh_askpass: exec(/usr/bin/ssh-askpass):
No such file or directory`).

**Transcript workaround**, so Claude Code can still see what happened
without holding the credentials itself:

```bash
script -q -f /tmp/tacc-session.log
# run ssh/TACC commands in this terminal as normal
```

Claude Code reads `/tmp/tacc-session.log` (`tail`/`Read`) after being
told a command completed, to check the result and diagnose failures —
it never runs the interactive command itself.

**GitHub authentication** on `EE-119537` was configured via the GitHub
CLI (no tokens or passwords recorded anywhere in this repo):

```bash
gh auth login
gh auth setup-git
```

This lets Claude Code's own sandbox `git push` through `gh`'s
credential helper — verified directly (`gh auth status` reports a
valid token; `git push --dry-run` and a real `git push origin
infra/tacc` both succeeded from the agent's own shell after this was
set up). Before this, `git push` failed with
`fatal: could not read Username for 'https://github.com'`.

### 6. Environment setup

Built on a **TACC compute node**, never the login node:

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00 -A IRI26016
bash launch/tacc/build_env.sh
```

`build_env.sh` refuses to run unless `$SLURM_JOB_ID` is set (i.e. you're
inside an `idev`/`sbatch` allocation, not the login node). It loads
`module load python/3.12.11`, creates/reuses `$SCRATCH/venvs/s4dpc`
idempotently, and installs exactly `requirements.lock`:

- Python `3.12.11`
- JAX `0.7.2`
- Flax `0.11.2`
- Optax `0.2.8`

GPU backend validated directly: JAX correctly enumerates 3 CUDA devices
on an A100 node (`CudaDevice(id=0/1/2)`).

### 7. Critical fix — Python module contamination

**Symptom:** the same venv could report **Python 3.12.13** inside a
batch job even though its `pyvenv.cfg` points at 3.12.11 — an inherited
module/library environment from the submitting shell contaminated the
job, overriding what the venv itself was built against.

**Fix, in `launch/tacc/job.slurm`, before activating the venv:**

```bash
module reset
module load python/3.12.11
```

This is intentional and should not be removed casually — it establishes
module state explicitly rather than trusting whatever the submitting
shell happened to have loaded, which is also TACC's own documented
recommendation for batch scripts.

### 8. Critical fix — Slurm spool path, not `BASH_SOURCE`

**Symptom:** the first real smoke job (`3377181`) failed in ~2 seconds
with:

```
fatal: not a git repository (or any parent up to mount point /)
```

**Cause:** `job.slurm` originally derived its repo root from
`BASH_SOURCE[0]`. Slurm executes a **spooled copy** of the batch script,
so `BASH_SOURCE[0]` pointed into Slurm's spool directory, not the actual
repo checkout the job was submitted from.

**Fix:** use `$SLURM_SUBMIT_DIR` instead, validated as a real git
checkout before `cd`-ing into it:

```bash
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/.git" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not a valid s4dpc git checkout: ${REPO_ROOT:-<unset>}" >&2
  exit 1
fi
cd "$REPO_ROOT"
```

Both `smoke_test.sh` and `submit_all.sh` already `cd
"$TACC_REPO_DIR"` before calling `sbatch`, so `SLURM_SUBMIT_DIR` is the
correct, intentional source of truth — never revert this to
`BASH_SOURCE`.

### 9. x64 precision requirement — not optional bookkeeping

Every real `s4dpc.sweep` invocation enables JAX x64
(`jax_enable_x64=True`, set in `sweep.py` before any JAX op). This is
load-bearing, not cosmetic: on the A100 nodes, the decode=False
(convolution)-vs-decode=True (recurrent) S4 parity test showed a large,
real discrepancy under **float32** (JAX's default, which is what
`pytest` runs under unless x64 is explicitly forced):

| | float32 `max_abs_diff` | x64 `max_abs_diff` (after the dtype fix below) |
|---|---|---|
| M3 | ≈ 0.67 | ≈ 1.09e-13 |
| M6 | ≈ 0.089 | ≈ 3.66e-14 |

Under x64, parity is essentially machine precision. **Future research
sweeps must preserve `jax_enable_x64=True` unless a specific experiment
is deliberately studying precision itself** — float32 is not a
supported production mode for this codebase's S4 path.

### 10. S4 recurrent-state dtype fix

`StackedModel.init_state()` (`s4dpc/model.py`) now follows
`jax.config.jax_enable_x64`:

- x64 disabled → `complex64`
- x64 enabled → `complex128`

Previously it hardcoded `complex64` unconditionally. Under x64, the S4
layer's own params (`Abar`/`Bbar`/`Cbar`) become `complex128`, so a
`complex64` recurrent-state carry made `jax.lax.scan` fail outright
(scan requires the carry's dtype to stay fixed across iterations) rather
than silently losing precision. `tests/test_control_decode_parity.py`
now explicitly verifies this (`test_init_state_dtype_follows_x64`,
parametrized over x64 on/off).

### 11. Validated test behavior (on TACC A100)

```
python -m pytest tests/test_control_decode_parity.py -v -s
```
→ **4 passed** (2 variants × the conv/recurrent parity test, 2 x64
on/off × the `init_state` dtype regression test).

```
python -m pytest tests/ -k "not test_parity" -v
```
→ **18 passed, 9 deselected** (the 9 deselected are `test_parity.py`'s
functions — see below).

```
python -m pytest tests/test_parity.py -v
```
→ **1 passed, 8 expected loud skips.** The skips are *expected, not
failures*: those 8 tests compare a legacy/M6 forward pass against a
fixture (`tests/fixtures/reference_model.msgpack`) generated on a CPU
backend, and float reduction order is not guaranteed identical across
backends (CLAUDE.md §5/§4 rule 4) — the fixtures' own loud-skip logic
correctly refuses to compare CPU-generated digests against GPU
execution rather than produce a meaningless pass/fail.

### 12. Fully validated end-to-end smoke test

| | |
|---|---|
| Slurm job | `3377245` |
| Git commit | `c0706cb` |
| Partition | `gpu-a100-dev` |
| Result | `COMPLETED`, ExitCode `0:0`, Elapsed `00:01:35` |
| Hostname | `c301-002.ls6.tacc.utexas.edu` |
| Python | `/scratch/11460/binoygeorge/venvs/s4dpc/bin/python`, `3.12.11` |
| Lockfile SHA | `aeaac02b0546` |
| JAX devices | `[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2)]` |
| `jax_enable_x64` | `True` |

Command actually run by `job.slurm`:

```bash
python -m s4dpc.sweep \
  --variant M3 --cases 1 --n_seeds 1 --epochs 2 \
  --d_model 8 --N 8 --n_layers 1 \
  --wandb off --out <TACC $WORK result path>
```

Wrote one CSV result row plus checkpoint metadata; `pull_results.sh`
then successfully copied the result back to `EE-119537`. **This
validated the entire chain end to end:** `EE-119537 → GitHub → TACC sync
→ Slurm → A100 → Python/JAX → real sweep → CSV/checkpoint → result
pullback.`

### 13. Result locations

| | |
|---|---|
| Remote repo | `$WORK/s4dpc` |
| Remote results | `$WORK/s4dpc-results` |
| Local pulled results | `~/Documents/Github/s4dpc/results/tacc` |

`pull_results.sh` excludes `.msgpack` checkpoints by default — pass
`--with-checkpoints` to include them.

### 14. Development safety rules — do not break these invariants

- Do not train on TACC login nodes — GPU work only inside an
  `idev`/`sbatch` allocation.
- Do not hard-reset the TACC checkout — sync is fetch + checkout +
  fast-forward-only merge, never `reset --hard`.
- Do not create scientific changes directly on TACC — develop on P53,
  push to GitHub, `sync_tacc.sh` pulls.
- Do not duplicate scientific logic into Slurm scripts —
  `python -m s4dpc.sweep` stays the single entrypoint.
- Do not silently change `requirements.lock`.
- Preserve JAX x64 unless intentionally testing precision (§9).
- Preserve the `module reset && module load python/3.12.11` step before
  venv activation in `job.slurm` (§7).
- Preserve the `SLURM_SUBMIT_DIR`-based repo-root logic in `job.slurm` —
  never revert to `BASH_SOURCE` (§8).
- `launch/tacc/tacc.env` must remain gitignored.
- Production jobs require explicit scientific configs from the user —
  scripts must never invent experiments.
- `python -m s4dpc.sweep` remains the single experiment entrypoint,
  everywhere.

### Context for future Claude / ChatGPT sessions

Quick-resume summary: **P53** develops science, pushes to **GitHub**
(`main`). **`EE-119537`** (`~/Documents/Github/s4dpc`) orchestrates TACC
via `launch/tacc/*.sh` but runs no science itself. **TACC Lonestar6**
(`$WORK/s4dpc`, venv at `$SCRATCH/venvs/s4dpc`) is execution-only, no
forked science. Standard loop from `EE-119537`: `sync_tacc.sh` →
`smoke_test.sh` (or `submit_all.sh` for production, gpu-a100-small,
explicit configs only) → `status.sh` → `pull_results.sh`. TACC auth
(password+MFA) is run by the human in their own terminal, optionally
recorded via `script -q -f /tmp/tacc-session.log` for Claude to inspect;
GitHub push from Claude's own sandbox works via `gh auth
login`/`setup-git` already configured on `EE-119537`. The environment
(Python 3.12.11, JAX 0.7.2/Flax 0.11.2/Optax 0.2.8, 3 A100 CUDA devices)
is validated and built via `build_env.sh` inside an `idev` allocation.
**JAX x64 is mandatory** for real sweeps — float32 showed a real
0.67/0.089 max-abs-diff S4 conv/recurrent mismatch that collapses to
~1e-13/~1e-14 under x64 after fixing `StackedModel.init_state()`'s
dtype. Two historical infra bugs, both fixed and must not regress: (1)
inherited module state could silently swap Python 3.12.11 for 3.12.13 —
fixed with `module reset && module load python/3.12.11` in `job.slurm`;
(2) `BASH_SOURCE` pointed into Slurm's spool dir, not the repo — fixed
by using `SLURM_SUBMIT_DIR`. Smoke job **3377245** (commit `c0706cb`)
is the proof the whole chain works end to end. The integration was
built and validated on branch `infra/tacc`, **which has since merged
into `main`** — `main` is the permanent branch going forward; don't
treat `infra/tacc` as a long-term fork, and never invent or submit a
production experiment without the user explicitly specifying its
scientific config first.
