# launch/tacc — Lonestar6 / TACC execution layer

Infrastructure only. This directory contains **no scientific code**. Every
job, smoke test, and production run invokes the same entry point as Kaggle:

```
python -m s4dpc.sweep
```

(CLAUDE.md §3 rule 1). If TACC ever seems to need its own training code, the
port is wrong — fix `s4dpc/sweep.py` instead, don't fork it here.

## Machines

| | |
|---|---|
| Local orchestration | `EE-119537` (Ubuntu 22.04.5), repo at `~/Documents/Github/s4dpc` |
| GitHub | `https://github.com/binoygeorge97/s4dpc`, `main` is approved science |
| TACC | `ls6.tacc.utexas.edu`, user `binoygeorge`, allocation `IRI26016` |

`$WORK` (`/work/11460/binoygeorge/ls6`) holds the persistent repo checkout
and all durable results. `$SCRATCH` (`/scratch/11460/binoygeorge`) holds
only the rebuildable Python venv and temporary/staging artifacts — never the
only copy of anything paper-critical.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `tacc.env.example` | — | Template for your local, gitignored `tacc.env` config |
| `build_env.sh` | TACC, **inside a GPU allocation** | Create/validate the `$SCRATCH/venvs/s4dpc` venv from `requirements.lock` |
| `job.slurm` | TACC, via `sbatch` | Platform-only Slurm wrapper around `python -m s4dpc.sweep` |
| `sync_tacc.sh` | EE-119537 → ssh → TACC | Clone/`git pull --ff-only` the persistent repo under `$WORK` |
| `smoke_test.sh` | EE-119537 | Sync + submit a tiny job to `gpu-a100-dev` using the real sweep CLI |
| `submit_all.sh` | EE-119537 | Sync + submit approved production job(s) to `gpu-a100-small` |
| `status.sh` | EE-119537 → ssh → TACC | One-shot `squeue`/`sacct` snapshot |
| `pull_results.sh` | EE-119537 → rsync → TACC | Pull durable CSVs/logs back (checkpoints excluded by default) |

## Validated status

Full narrative, exact numbers, and the "context for future AI sessions"
handoff summary live in the top-level [README.md](../../README.md)'s
"TACC / Lonestar6 execution layer" section — read that first if you're
picking this up cold. Short version:

- **End-to-end proven**: smoke job `3377245` (commit `c0706cb`)
  completed on `gpu-a100-dev` in 1m35s, wrote a real CSV row + checkpoint
  via `python -m s4dpc.sweep`, and `pull_results.sh` copied it back to
  `EE-119537`.
- **x64 is mandatory for real sweeps**, not optional bookkeeping — S4
  conv/recurrent parity was off by ~0.67 (M3) / ~0.089 (M6) under
  float32 on the A100, collapsing to ~1e-13/~1e-14 under x64 once
  `StackedModel.init_state()` was fixed to follow `jax_enable_x64`.
- **Two infra bugs, both fixed, do not regress either**: (1) a batch
  job's inherited module state could silently swap the venv's Python
  3.12.11 for 3.12.13 — `job.slurm` now runs `module reset && module
  load python/3.12.11` before activating the venv; (2) `job.slurm` used
  to derive its repo root from `BASH_SOURCE[0]`, which pointed into
  Slurm's spooled copy of the script rather than the real checkout —
  fixed by using `$SLURM_SUBMIT_DIR` instead (validated as a real git
  checkout before `cd`-ing in).
- **Network reachability from P53/WSL, confirmed 2026-08-25**:
  `nc -vz ls6.tacc.utexas.edu 22` succeeded from the P53 ThinkPad's WSL
  Ubuntu (resolved to `129.114.62.201`, port 22 open) with no campus VPN
  active. TACC's login nodes are reachable from that network as-is. This
  resolves the "untested off the lab network" caveat for P53
  specifically — it says nothing about the home PC, which has no WSL
  installed yet and remains completely untested.

## Authentication

TACC SSH login is always interactive: you type your password and 6-digit
MFA code directly at the terminal prompt when `ssh`/`rsync` asks. No script
here uses `sshpass`, stores a password, or automates MFA. Once `sbatch`
returns a job ID the ssh connection can close — Slurm owns the job from
there.

**In practice**, this means TACC-touching commands are normally run by
the human directly in their own terminal (e.g. the VS Code integrated
terminal) rather than through an agent's non-interactive shell, which
has no real TTY for the password/MFA prompt to land on. A working
pattern: start `script -q -f /tmp/tacc-session.log` in that terminal,
run the ssh/TACC commands there as usual, then let the agent `tail`/read
that log to check results and diagnose failures after being told a step
finished — the agent never handles the credentials itself.

Separately, **GitHub** push access from an agent's own sandbox is a
different credential path: configuring `gh auth login` followed by `gh
auth setup-git` once on `EE-119537` lets `git push` succeed through
`gh`'s credential helper from that sandbox — no token or password is
ever pasted into chat or stored in this repo.

## First-time TACC setup

From EE-119537:

```bash
cp launch/tacc/tacc.env.example launch/tacc/tacc.env
# edit launch/tacc/tacc.env if any path/partition differs from the defaults
```

```bash
ssh binoygeorge@ls6.tacc.utexas.edu
```

Then, on TACC (login node — clone only, no compute here):

```bash
cd $WORK
git clone https://github.com/binoygeorge97/s4dpc.git
cd s4dpc
```

Allocate a development GPU node (build_env.sh refuses to run without one):

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00 -A IRI26016
```

Once the interactive session starts on a compute node:

```bash
bash launch/tacc/build_env.sh
```

This loads `python/3.12.11` (never the default `python3`/3.9.7), creates
`$SCRATCH/venvs/s4dpc`, installs exactly `requirements.lock`, and runs
`env_probe.py`, printing Python/JAX/Flax/Optax versions, detected JAX
devices, git SHA, and the lockfile SHA.

Optionally, from inside the same allocation, run the local test suite /
parity checks (CLAUDE.md §5) before trusting the environment:

```bash
source $SCRATCH/venvs/s4dpc/bin/activate
cd $WORK/s4dpc
python -m pytest tests/ -k "not test_parity"   # GPU-agnostic subset
python -m pytest tests/test_parity.py          # will LOUD SKIP if the fixture's backend differs
```

## Routine workflow

On your dev machine: develop → test locally / Kaggle → commit → push to
`main`.

On EE-119537:

```bash
git pull --ff-only
```

then:

```bash
./launch/tacc/sync_tacc.sh
./launch/tacc/smoke_test.sh
./launch/tacc/submit_all.sh -- --variant M6 --cases 1,2,3,4,5,6,7 --n_seeds 5 \
  --epochs 200 --d_model 16 --N 32 --n_layers 1 --wandb offline \
  --out "$WORK/s4dpc-results/m6_full.csv"
./launch/tacc/status.sh
./launch/tacc/pull_results.sh
```

(`submit_all.sh` reads TACC-side `$WORK` values at submission time on the
remote host — the literal `$WORK` in the `--out` path above is intentional
and gets expanded on TACC, not locally. `--dry-run` previews without
submitting; `--configs-file FILE` submits several independent
variants/cases as separate jobs in one call.)

After a completed batch, append `docs/LOG.md` with the git SHA, W&B group
(if any), CSV path(s), and what was concluded — same convention as
Kaggle runs (CLAUDE.md §10).

### When you must enter your password/MFA

Every script that talks to TACC (`sync_tacc.sh`, `smoke_test.sh`,
`submit_all.sh`, `status.sh`, `pull_results.sh`) opens its own ssh (or
rsync) connection and will prompt you interactively each time it runs. In
particular `smoke_test.sh` and `submit_all.sh` each prompt twice — once for
the `sync_tacc.sh` step, once for the actual `sbatch` submission — since
each is a separate ssh session. There is no batching of prompts across
scripts; that's the deliberate tradeoff for keeping authentication fully
manual and un-automatable.

## Login-node rules (enforced by convention, not by these scripts)

Allowed on the TACC login node: `git clone`/`git pull --ff-only`, `sbatch`,
`squeue`, `sacct`, `scancel`, small file management, result transfers.

**Not allowed** on the login node: S4/DPC training, heavy analysis, GPU
computation, or environment installation that performs heavy computation.
`build_env.sh` enforces this itself — it refuses to run unless
`$SLURM_JOB_ID` is set (i.e., you are inside an `idev`/`sbatch` allocation).

## Storage discipline

- `$WORK`: persistent repo checkout, durable CSV/results (`$TACC_RESULTS_DIR`),
  important checkpoints/logs.
- `$SCRATCH`: Python venv (`$TACC_VENV_DIR`), staging, temporary artifacts only.
- Never `$HOME` for the venv, checkpoints, or datasets.
- `job.slurm` writes its own `stdout`/`stderr` (`slurm-%x-%j.out/.err`) into
  the repo checkout directory itself — which is under `$WORK` by
  construction (`$TACC_REPO_DIR`), so this is durable without extra wiring.
  A structured `job_info.txt` (hostname, date, job ID, partition,
  `CUDA_VISIBLE_DEVICES`, git SHA, lockfile SHA, JAX devices) is additionally
  written under `$TACC_RESULTS_DIR/logs/<job-name>_<job-id>/`.

## W&B

Same policy as everywhere else in this repo (CLAUDE.md §8) — nothing
TACC-specific. Use `--wandb offline` (sync later from the login node with
`wandb sync`) or `--wandb off`. CSV remains ground truth either way.

## Queues (do not self-escalate)

| Partition | Max walltime | Max nodes | Max running jobs/user |
|---|---|---|---|
| `gpu-a100-dev` (smoke) | 02:00:00 | 2 | 1 |
| `gpu-a100-small` (production) | 2-00:00:00 | 3 (1/job) | 3 |

`job.slurm` defaults to `gpu-a100-dev` and a 30-minute walltime as a safety
net if invoked bare; real submissions always pass `-p`/`-t` explicitly via
`smoke_test.sh`/`submit_all.sh`. Moving to `gpu-a100`/`gpu-h100`, or raising
node count/walltime/concurrency beyond what's requested, requires explicit
user approval — none of these scripts do it automatically.
