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
| Local orchestration (2) | P53 ThinkPad, Windows + WSL2 Ubuntu, repo at `~/projects/s4dpc` — set up 2026-08-25, see the P53/WSL section near the end of this file for machine-specific findings |
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

**Observed 2026-08-25**: `gpu-a100-dev` only has 2 nodes total (table
above), and both were occupied by OTHER users' long-running interactive
`idev` sessions when a smoke job was submitted from P53 — `squeue
--start` estimated a ~1 hour wait despite the job being first in this
user's own queue. Not a bug or misconfiguration, just real contention on
a small shared dev partition — if a smoke test seems to hang in
`PENDING`, check `squeue -p gpu-a100-dev` before assuming something is
wrong.

## P53/WSL-specific findings (2026-08-25)

Everything above this section is machine-independent — it applies the
same way regardless of which local machine is orchestrating. This
section is specifically what differed, or needed verifying, on the P53
ThinkPad (Windows + WSL2 Ubuntu, repo at `~/projects/s4dpc`) the first
time TACC was set up there. Nothing here contradicts the
machine-independent steps; treat this as an addendum for that one
machine, not a second procedure.

- **No VPN needed.** `nc -vz ls6.tacc.utexas.edu 22` succeeded directly
  from P53's WSL network with no campus VPN active (resolved to
  `129.114.62.201` for that specific check). TACC's login nodes are
  reachable from this network as-is.
- **`mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets` is mandatory,
  not optional, for the multiplexing setup below** — `ssh` will NOT
  create the `ControlPath` directory itself, and fails **silently**
  (falls back to a normal, non-multiplexed connection with no error)
  if that directory doesn't already exist. Create it before the first
  connection, not after something seems not to be working.
- **The `Host` line in `~/.ssh/config` must match the exact string
  used on the command line.** This repo's scripts all connect via
  `ssh "${TACC_USER}@${TACC_HOST}"` where `TACC_HOST=ls6.tacc.utexas.edu`
  (from `tacc.env`) — the `~/.ssh/config` entry must key on that exact
  hostname (`Host ls6.tacc.utexas.edu`), not an alias or shortened form,
  or the multiplexing config silently won't match and every script
  reverts to prompting for password/MFA again.
  ```
  Host ls6.tacc.utexas.edu
      User binoygeorge
      ControlMaster auto
      ControlPath ~/.ssh/sockets/%r@%h-%p
      ControlPersist 4h
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```
- **Socket verification (`ssh -O check ls6.tacc.utexas.edu`) must run
  on the LOCAL machine, in a second terminal — not inside the TACC ssh
  session itself.** The control socket is a local-machine artifact
  (`~/.ssh/sockets/binoygeorge@ls6.tacc.utexas.edu-22` on P53); checking
  for it or querying it from inside the remote TACC shell checks the
  wrong filesystem entirely and will report it missing even when it's
  working correctly.
- **Lonestar6 round-robins its login nodes.** The interactive login that
  established the control socket landed on `login2.ls6`
  (`129.114.62.202`) — a different node than the one the earlier plain
  reachability check (`nc -vz`) happened to hit (`.201`). This did not
  cause any problem: once the control master is established, EVERY
  subsequent multiplexed connection (confirmed directly — a
  non-interactive `ssh` call from Claude Code's own shell, 0.59s, no
  prompt) reuses that same already-open connection to whichever node was
  hit first, so round-robin variance only matters for the very first,
  interactive connection — it cannot cause a later script to land on a
  different, inconsistent node mid-session.
- **`$SCRATCH` is purge-eligible — venv presence must be checked every
  session, never assumed present from a prior session on a DIFFERENT
  local machine.** `$SCRATCH`/`$WORK` are tied to the TACC account, not
  to which local machine is orchestrating, so a venv built from
  `EE-119537` genuinely does persist and get reused from P53 — but only
  until `$SCRATCH` purges it, which can happen independent of any local
  machine's knowledge. Check with `ls -d $SCRATCH/venvs/s4dpc` before
  deciding whether `build_env.sh`/`idev` are needed; when the venv IS
  present, this is the difference between a ~2-minute start (activate
  and go) and a real `gpu-a100-dev` queue wait for a fresh `idev`
  allocation (see the observed ~1 hour wait noted above) — don't request
  `idev` reflexively.
- **Step 4 (`git clone`) is really "clone if absent, otherwise
  fast-forward"**, not a one-time step — `$WORK/s4dpc` is the SAME
  checkout shared across every local machine that orchestrates this
  TACC account, so a second (or third) machine setting up for the first
  time will find the repo already cloned and should `git fetch && git
  checkout main && git merge --ff-only origin/main` instead of cloning
  fresh (`sync_tacc.sh` already does exactly this — clone only if the
  `.git` directory is absent, fast-forward otherwise).
- **Clock drift after laptop sleep, corrected**: WSL2's clock lagging
  after a Windows host suspend/resume cannot break TOTP-based MFA
  itself — the 6-digit code is generated and validated against TACC's
  own clock via your separate authenticator device, not against
  anything on the WSL machine. What WSL clock drift CAN break is TLS
  certificate-validity checks for HTTPS operations this procedure does
  use (`git clone https://...`, `pip install`) — a sufficiently drifted
  local clock can make a valid remote certificate look expired or not-
  yet-valid. Check `date` after a laptop sleep/resume, before assuming
  an HTTPS failure is a network problem.

**Not yet done**: the home PC has no WSL installed and remains
completely untested for any part of this procedure.
