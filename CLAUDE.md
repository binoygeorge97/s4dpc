# CLAUDE.md — s4dpc

Operating guide for Claude Code in this repository.

---

## 1. What this project is

Research code for a conference paper on **why differentiable predictive control (DPC)
fails when the plant model is a learned S4 surrogate**, even when one-step prediction
MSE is low.

Core claim under test: *control-relevant derivative fidelity*, not prediction error,
determines DPC success. Suspected mechanism: LayerNorm (degree-0 homogeneous, so it
cannot represent a linear map) distorts the learned Jacobian near the regulation
setpoint — the "kink".

Plants are 7 discrete-time linear systems, all `A: (6,6)`, `B: (6,3)`. Ground truth
`A_d`, `B_d`, Markov parameters and an oracle LQR controller are all computable, which
is the entire reason for using linear plants.

**Timeline: 7 days.** Bias every judgement call toward "ships this week".

---

## 2. Repo layout

```
s4dpc/
  requirements.lock          # EXACT pins. Do not edit without re-running parity.
  env_probe.py               # version + device + param-tree + numerics canary
  s4dpc/
    model.py                 # blocks + stacked model (consumes s4-nnx)
    blocks.py                # configurable block: norm / activation / glu
    systems.py               # get_discrete_system(case) -> A_d, B_d, name
    identify.py              # system identification, all variants
    control.py               # GRU controller + DPC loss (action IS bounded)
    diagnostics.py           # Markov params, gradient cosine, equilibrium drift
    sweep.py                 # THE entry point. argparse + vmapped harness + CSV
    logging.py               # Logger (online/offline/off), row schema, classifier
  tests/
    test_parity.py           # refactor == original, BIT-EXACT
    fixtures/reference_model.msgpack
  launch/
    kaggle/                  # kernel-metadata.json + runner notebook/script
    tacc/                    # job.slurm, build_env.sh, submit_all.sh
  docs/
    00_protocol.md           # frozen tables + figure captions + kill criteria
    DECISIONS.md             # append-only, dated
    LOG.md                   # append-only, one entry per batch of runs
  results/                   # gitignored, CSV output
  gpu_ledger.csv             # Kaggle GPU quota tracking
```

The S4 layer itself lives in a **separate pinned package** (`s4-nnx`, by git tag).
Never vendor or edit it from here — see §7.

---

## 3. Golden rules

1. **`sweep.py` is the only entry point.** Notebooks, Kaggle kernels and SLURM jobs
   all invoke `python -m s4dpc.sweep` with different flags. There is no
   platform-specific training code. If a platform seems to need its own version, the
   port is wrong — fix the port.
2. **Never edit code on TACC.** Develop locally/Kaggle, push, `git pull --ff-only` on
   the login node.
3. **Never bump a version in `requirements.lock`** without re-running `tests/test_parity.py`
   on all platforms.
4. **Never split one experiment arm across platforms.** All seeds of a variant run on
   the same hardware — XLA lowering differs by backend, and BPTT through unstable
   dynamics amplifies float differences.
5. **CSV is ground truth. W&B is a lens. Figures come from the CSV.**
6. **Diverged runs are results, not failures.** Tag, never delete.
7. Stamp `git_sha` and `lockfile_sha` into every CSV row.

---

## 4. Kaggle CLI workflow

Kaggle is for **development and parity testing**, not production sweeps. Fast feedback,
small configs, `--epochs 50`.

### Kernel metadata

`launch/kaggle/kernel-metadata.json`:

```json
{
  "id": "USERNAME/s4dpc-smoke",
  "title": "s4dpc smoke",
  "code_file": "runner.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

`enable_internet` must be true — the kernel clones the repo and pip-installs the lock file.

### Commands

```bash
# push
kaggle kernels push -p launch/kaggle

# poll until "complete" or "error"
kaggle kernels status USERNAME/s4dpc-smoke

# pull results + log
kaggle kernels output USERNAME/s4dpc-smoke -p ./out
```

### The runner cell

```python
!git clone https://github.com/USER/s4dpc.git /kaggle/working/s4dpc
!pip install -q -r /kaggle/working/s4dpc/requirements.lock
# RESTART REQUIRED if kaggle preinstalled a different jax
!cd /kaggle/working/s4dpc && python env_probe.py
!cd /kaggle/working/s4dpc && python -m s4dpc.sweep \
    --variant M6 --cases 3 --n_seeds 2 --epochs 50 \
    --wandb off --out /kaggle/working/out.csv
```

### GPU ledger — REQUIRED after every run

After every successful `kaggle kernels output`, parse the total runtime from the `.log`
in `./out` and append one line to `gpu_ledger.csv`:

```
date,kernel_slug,accelerator,minutes
```

**Before pushing any GPU job**, sum the current week's rows (quota resets **Saturday
00:00 UTC**) and report the running total to the user. Do not push if the remaining
budget looks tight without saying so first.

---

## 5. Parity testing — the current task

Goal: the refactored code (which calls `s4-nnx` for the SSM core) must reproduce the
original notebook code **exactly**.

**Acceptance criteria — bit-exact, `rtol=0, atol=0`:**

- **L1** — load `tests/fixtures/reference_model.msgpack` into both; same input →
  identical output.
- **L2** — identical gradients.
- **L3** — both `decode=False` (conv, `L == l_max`) and `decode=True` (stepped).

`rtol=1e-5` hides real bugs. Use zero tolerance. If the op sequence is unchanged, XLA
emits the same kernel and the outputs are bit-identical.

**Do not compare by training both and eyeballing loss curves.** `nnx.Rngs` is stateful,
so any change to key-draw order changes every parameter, and you cannot distinguish a
bug from a different random draw. Parity is tested against a **saved checkpoint**,
independent of initialization.

If param trees differ under the same seed, that is expected — record it in
`docs/DECISIONS.md` and note that pre/post-refactor runs are not seed-comparable.

---

## 6. Variant ladder

Set by config, never by branching code:

| Variant | `norm` | `activation` | `glu` | Isolates |
|---|---|---|---|---|
| M0 | — | — | — | true `(A_d, B_d)` oracle |
| M1 | — | — | — | least-squares `(Â, B̂)` |
| M3 | none | none | False | LTI S4 — is capacity the issue? |
| M4 | none | gelu | True | activation curvature |
| M5 | layer | none | False | **normalization (prime suspect)** |
| M6 | layer | gelu | True | current full model |
| M6_fix | static | gelu | True | fixed standardization (the proposed fix) |

**M3 is the pivotal cell.** A linear S4 is an LTI system and should recover the Markov
parameters to ~1e-6. If it does, capacity is not the problem and the nonlinearities are.
If it doesn't, the paper pivots to the spurious-memory/realization story — flag this
immediately.

---

## 7. Versions

Pinned in `requirements.lock`. Tested combination:

```
jax==0.7.2   flax==0.11.2   optax==0.2.8
numpy  scipy  pandas  matplotlib  tqdm  wandb
```

**Do not move to flax 0.12.x this week.** 0.12.0 introduced breaking NNX changes around
Modules holding Arrays; `StackedModelRegression` uses a plain `self.layers = []` +
`.append()` pattern that sits in that blast radius.

`venv` + pip only. **No conda** — the `jax[cuda12]` wheels bundle their own CUDA, and a
conda `cudatoolkit` shadows them and produces linker errors that look like hardware
faults.

`s4-nnx` is pinned by git tag:
```toml
dependencies = ["s4-nnx @ git+https://github.com/USER/s4-nnx.git@v0.2.0"]
```
It is frozen for the week. If you want to change it, move the code into `s4dpc` instead.

---

## 8. W&B

One flag, three modes: `--wandb {online,offline,off}`.

- `online` — Kaggle / Colab
- `offline` — TACC compute nodes (no outbound internet); `wandb sync` later from the login node
- `off` — tests, smoke runs, parity. **This is the default.**

Never scatter `if use_wandb:` through the code. Use the null-object `Logger` in
`s4dpc/logging.py`.

**Projects:**
- `s4dpc-dev` — all development, debugging, smoke runs. Default.
- `s4dpc` — the real sweep. Nothing enters until parity passes and the schema is frozen.

Runs are created **post-hoc from CSVs** by `push_to_wandb.py`, with a deterministic
`id` so re-pushing updates rather than duplicates. This is what makes ~35 vmapped
logical runs per job into 35 separately-filterable W&B runs.

**Never commit a W&B or GitHub token.** They load from the environment.

---

## 9. Parallelism

`jax.vmap` over `(case × seed)` inside one process; `sbatch` across configs. That is the
whole strategy.

**Do not add Ray, joblib, or multiprocessing.** All 7 systems share shapes, so 35 runs
fuse into single kernels — no process overhead, no VRAM contention, no second scheduler
fighting SLURM.

For readable tracebacks during debugging, `sweep.py` accepts `--no-vmap`, which loops in
Python. Same function, same outputs. That is the "single run" path — a flag, not a
second script.

---

## 10. When you finish a batch of runs

Append to `docs/LOG.md` with three coordinates — **sha, wandb group, csv path** — then
what was concluded and any anomaly:

```markdown
## 2026-08-11 — M3/M5/M6 identification, 18 archs × 10 seeds
sha: a3f91c2 | wandb: M5_case* | results/M5_*.csv

M3 recovers Markov params to ~1e-6 on all cases → capacity is not the issue.
M5 markov_err_h10 jumps 2-3 orders on cases 4/6 → supports the norm mechanism.
Anomaly: case 7 seed 3 diverged in identification, tagged bad_id, not rerun.
```

Any decision that changes what enters the paper goes in `docs/DECISIONS.md`, dated,
with the reasoning. Reviewers ask exactly these questions.

---

## 11. Things that will waste the week

- Debugging on a batch queue instead of Kaggle (20-min feedback loops)
- Refactoring `s4-nnx` mid-week
- Adding the nonlinear benchmark on day 3 because the compute is there — it costs the
  writing days; it is future work
- Manually transcribing numbers into LaTeX instead of generating `\input`-able tables
- Writing the research log retroactively on day 6
