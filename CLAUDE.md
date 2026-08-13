# CLAUDE.md — s4dpc

Operating guide for Claude Code in this repository.

---

## 1. What this project is

Research code for a conference paper on **why differentiable predictive control (DPC)
fails when the plant model is a learned S4 surrogate**, even when one-step prediction
MSE is low.

**Correction, 2026-08-13 (kept, not silently rewritten — see `docs/DECISIONS.md`'s
"REFUTATION" entry and the 10-seed/kink-magnitude entries around it for the full
evidence trail):** the paragraph below was this project's original central claim.
It has been tested directly on control-side data and **refuted**. M3 has exactly
zero kink by construction (no norm/activation/glu) and still fails DPC by
2-6 orders of magnitude on every case, including ones where it recovers Markov
parameters to ~1e-6; Spearman(kink magnitude, M6/M3 cost ratio) = -0.54 at n=6 —
wrong sign, not just non-significant. **Original claim, kept for the record:**
*control-relevant derivative fidelity*, not prediction error, determines DPC
success, via LayerNorm (degree-0 homogeneous, so it cannot represent a linear map)
distorting the learned Jacobian near the regulation setpoint — the "kink".

**Current working picture, updated 2026-08-13 with a decisive result (also
`docs/DECISIONS.md` — check there before citing a specific mechanism as settled,
since this remains under active investigation):** the S4/BPTT/`decode=True` control
machinery itself is innocent. "M0_S4" — an S4 hand-constructed (block-zeroed) to
realize the exact true `(A_d, B_d)` to ~1e-17, then deployed through the identical
`rollout_learned`/`jax.lax.scan` machinery M3/M6 controllers train through — matches
the true-plant training baseline essentially exactly, at every curriculum horizon
tested (5 to 200) and on every one of the 6 control cases (>=5 seeds each, most
agreeing to 3-4 decimal places). So the 300x-700,000x M3/M6 failure has nothing to
do with being an S4 surrogate, with BPTT through the stepped/scan machinery, or with
any training-mechanics candidate — it is entirely about what identification LEARNS
when fit to data instead of being handed exactly. The current best candidate for
what that learned difference is: M3's augmented state-transition operator (physical
state + S4 hidden state) has ~300 near-unit-circle modes (of 1030 dims) where the
true system has only 3-6, with real one-step leverage on the output and often
orders-of-magnitude-excess transient growth — present in every real M3 checkpoint,
absent by construction in M0_S4 (which was never asked to LEARN a realization).
This does not yet explain the case-by-case severity gradient (no tested spectral
quantity correlates with the per-case DPC ratio) or why the same spurious-mode
signature shows up: warm-starting the S4 hidden state on genuine burned-in history
makes M3's free-running prediction monotonically WORSE, not better, arguing against
a simpler inconsistent-initial-condition story and toward the internal modes being
fundamentally unmoored from the physical state rather than merely mis-initialized.

**Sharpened, 2026-08-13 (see `docs/DECISIONS.md`'s CORRECTION entry after Task 2
Part C):** M0_S4 changes TWO things relative to a real M3 checkpoint at once —
exact I/O (M3's is ~1e-6) and zero *observable* internal state (M3's `obs_norm`
is ~10-20, real). M0_S4's result therefore does NOT distinguish which one matters
— it only rules out the S4/BPTT machinery. Since M3's ~1e-6 Markov error was
already too small to explain a 300x+ blowup under ordinary error propagation
(`rho~1.02` over 200 steps compounds 1e-6 to only ~5e-5), realization/observable-
spurious-modes remains the standing, not-yet-refuted candidate.

**Tested directly via balanced truncation (Hankel-SVD/ERA, since the augmented
operator is marginally unstable so Gramian-based truncation doesn't apply) —
result decided NOTHING about the spurious-mode hypothesis (corrected
2026-08-13 in `docs/DECISIONS.md` — an earlier write-up called this "cleanly
falsified," which overstated it).** Truncating M3 to 6 states does NOT recover
M3's own ~1e-6 fidelity (lands at ~1e-2 instead) because M3's Hankel singular
values have no cliff at rank 6 the way the true system's do (verified:
self-check on the true system gives an EXACT cliff to 0 past index 6; M3's
spectrum decays smoothly) — a real, unconfounded finding about M3's own
realization. But truncated M3 then failing DPC 3-41x worse than full M3 is
NOT informative about spurious modes specifically: a ~1e-2-fidelity system
was already expected to fail badly given this project's own established
M1(~1e-14→~1x)-vs-M3(~1e-6→300x+) relationship, independent of any mode-based
story. The experiment never isolated the variable it needed to. A
fidelity-matched truncation (larger r, chosen so `err_vs_m3_markov` reaches
M3's own ~1e-6 floor) is the only experiment that would actually test the
hypothesis — not yet run.

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

### Colab CLI — for INDEPENDENT experiment arms only, when Kaggle's 2 slots are full

The `colab` CLI (`uv tool`-installed, symlinked at `~/.local/bin/colab` — add that to
`PATH` the same way `kaggle` needs it) provisions and drives real Google Colab VMs
from the shell: `colab new -s <name> --gpu T4`, then run code on it, then
`colab stop -s <name>`. Full usage is in `colab skill` / `colab readme`.

**Only for a genuinely separate experiment arm** — never to finish or split an arm
already started on Kaggle (CLAUDE.md §3 rule 4: XLA lowering differs by backend, and
BPTT through unstable dynamics amplifies float differences across platforms). If Task
X is running on Kaggle and Task Y is independent and ready to launch, and both Kaggle
slots are occupied, Colab is where Y goes — not a shortcut around the 2-slot limit for
splitting Task X.

**`colab exec` is broken as of CLI 0.6.0** (`AttributeError:
module 'jupyter_kernel_client' has no attribute 'KernelClient'` — a real bug in the
installed package, not a usage error; `colab update` reports already-latest).
Use `colab console -s <name>` instead, piping a full shell command in
(`echo "cd /content && git clone ... && pip install ... && python script.py" | colab
console -s <name>`), matching the Kaggle notebook cell pattern
(`!git clone && !pip install && !python`). Revisit `exec` once the CLI updates.

**Long jobs**: the piped command keeps running in the session's tmux pane even if the
local `colab console` connection drops (e.g. a Bash tool timeout) — reconnecting later
with another `colab console -s <name>` finds it still going. Do **not** poll
`colab status` to check progress on a `console`-launched job — that reflects the
*Jupyter kernel's* idle/busy state, not the tmux shell's, and reads IDLE the whole
time regardless of what's running in the pane. Instead have the job write its final
output to a known path and poll for that file with `colab download <remote> <local>`
(safe to call while the pane is busy — it's a separate file-transfer channel, not
console input). Do not pipe a status-check command into a *busy* console — with no
free shell prompt to run it, the keystrokes just queue silently instead of executing.

**Always `colab stop -s <name>` when the arm is done.** The keep-alive daemon prevents
idle timeout specifically so a forgotten session does *not* get auto-reclaimed — that
means a session left running keeps burning compute units indefinitely until someone
stops it.

---

## 5. Parity testing — DONE (2026-08-08, `docs/DECISIONS.md`); kept as reference

No longer the current task — the refactor passed L1/L2/L3 below and control-side work
(`s4dpc/control.py`, the M0/M1/M3/M6 DPC comparisons) has been running on top of it since.
Current focus is the control-failure mechanism itself (§1's correction) — see
`docs/DECISIONS.md`'s most recent entries for what's actually being worked on. This
section stays as reference for what to do if a future dependency bump (§7) requires
re-running parity.

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

**Correction, 2026-08-13: this decision rule fired, and the branch it pointed to was
wrong.** M3 DID recover Markov parameters to ~1e-6 (`docs/DECISIONS.md`'s 10-seed
identification entry) — by the rule as written, that meant "capacity is not the
problem, the nonlinearities are," which pointed straight at M5/M6's LayerNorm kink.
The kink was real (M6 has one, M3 provably can't) and looked like a clean confirmation
on identification-side data alone. But when actually tested on control-side data (M3
vs M6 through DPC), M3 fails just as catastrophically as M6 despite its exact
near-machine-precision Markov fidelity and zero kink — see §1's correction and
`docs/DECISIONS.md`'s "REFUTATION" entry. **The rule's premise was incomplete: "M3
recovers Markov parameters" answers whether M3's *local linearization* matches the
true plant, not whether M3 is *safe to backprop a controller through* — those turned
out to be different questions, and the ladder as designed had no cell that tested the
second one directly.** That gap is what the current working picture (§1) and the
session-by-session entries in `docs/DECISIONS.md` are now closing (M0_S4 — a
hand-constructed S4 forced to realize the true plant, deployed through the same
BPTT/control machinery — is the closest thing to a corrected version of this rule).

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
