# ENVIRONMENTS.md — multi-machine provenance

Written 2026-08-25 once three local machines were in regular use for this
project, alongside Kaggle, Colab, and TACC. Companion to CLAUDE.md §3 rule 4
("never split one experiment arm across platforms") and §12 (TACC operational
flow) — this file is the map of WHERE things run; CLAUDE.md is the rules for
HOW. Read this before setting up a new machine, and re-run `env_probe.py`
(§5 below) on any machine whose result you're about to trust.

---

## 1. The machines

| Machine | OS | Role | Repo path | TACC status |
|---|---|---|---|---|
| **ThinkPad P53** (lab) | Windows + WSL2 Ubuntu, VS Code | Primary dev machine this session ran on. Kaggle CLI + Colab CLI installed and authenticated. | `~/projects/s4dpc` (WSL2 **ext4**, confirmed NOT `/mnt/c` — see §3) | Not yet configured (no `launch/tacc/tacc.env` here as of 2026-08-25) |
| **Lab PC** (`EE-119537` in `launch/tacc/README.md`) | Ubuntu 22.04.5, native | TACC orchestration — `sync_tacc.sh`/`smoke_test.sh`/`submit_all.sh` all run from here today | `~/Documents/Github/s4dpc` | Fully set up — `launch/tacc/tacc.env` present, validated end-to-end (smoke job `3377245`, `docs/DECISIONS.md`) |
| **Home PC** | Windows, no WSL yet | Not yet provisioned | — | Not yet configured |

`hostname` is what distinguishes machines in result-row provenance (§4) — on
this machine it is `LAPTOP-REIDHE1P`. Get each machine's own value with
`hostname` (Linux) before relying on it.

**Git identity is currently NOT uniform across machines** — this machine's
`git config user.name`/`user.email` is `phd-studies` /
`phdstudiescontrols@gmail.com`; the Lab PC's TACC-related commits in history
use `binoygeorge97`. Both push directly to `main` under the same GitHub
account (`binoygeorge97`, via `gh`'s stored credential — see §2). This is
not a problem for `git blame`/authorship, but don't assume `git log --author`
finds every commit from a given machine.

---

## 2. Credentials — what lives where, and what's per-machine

**Nothing that authenticates to TACC is stored anywhere.** TACC login is
interactive password + 6-digit MFA at the ssh/rsync prompt every time
(`launch/tacc/tacc.env.example`'s own header says this explicitly) — there
is no secret to provision on a new machine for TACC beyond the plain-text
host/path config in `tacc.env` (gitignored, `.example` template tracked).

Everything else IS per-machine and must be set up freshly on each one:

| Credential | Where it lives | Per-machine? | Set up with |
|---|---|---|---|
| GitHub (push/pull + `gh`) | `~/.config/gh/hosts.yml`, git configured to use `gh`'s HTTPS credential helper | Yes | `gh auth login` |
| Kaggle | `~/.kaggle/kaggle.json` (mode `600`) | Yes | Kaggle account settings → Create New API Token → save to that path |
| Colab CLI | `~/.config/colab-cli/` | Yes | `colab` CLI's own login flow (see CLAUDE.md §4) |
| TACC | none stored — interactive MFA every session | N/A | `launch/tacc/tacc.env` (paths/hostnames only, no secret) |
| W&B | `WANDB_API_KEY` env var, only if `--wandb online/offline` used (default is `off` — CLAUDE.md §8) | Yes, if used | `wandb login` |

**Cross-machine collision risk, not just a per-machine setup checklist:**
Kaggle's 2-concurrent-kernel limit (CLAUDE.md §4) is per-ACCOUNT, not
per-machine. If two machines are logged into the same Kaggle account and
both push kernels, they compete for the SAME 2 slots — check
`kaggle kernels list --mine` for anything `RUNNING` before pushing from a
second machine, don't assume a clean slate just because THIS machine hasn't
launched anything recently.

---

## 3. Bringing up a fresh machine

### Any Linux/WSL machine (the Home PC's target state)

1. **Windows machines: install WSL2 + Ubuntu first**, then do everything
   below INSIDE the WSL2 Ubuntu shell, never in PowerShell/cmd:
   ```powershell
   wsl --install -d Ubuntu
   ```
   Reboot when prompted, create the Ubuntu user, then open the Ubuntu app
   (not VS Code's default terminal, until VS Code's WSL extension is
   installed — see step 6).

2. **Clone into the Linux filesystem, not `/mnt/c`** — this is the single
   most important rule for a Windows-hosted machine (§3.1 below explains
   why). From inside WSL:
   ```bash
   mkdir -p ~/projects && cd ~/projects
   git clone https://github.com/binoygeorge97/s4dpc.git
   cd s4dpc
   ```

3. **Install `uv`** (no-root Python version manager — the system Python on
   a fresh Ubuntu/WSL image is very likely too old; this project pins
   `jax==0.7.2`/`flax==0.11.2`/`s4-nnx@v0.2.0`, all requiring Python>=3.11):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. **Build the venv from the exact pin**, never `pip install` loose:
   ```bash
   uv python install 3.11
   uv venv --python 3.11 .venv   # or any path outside the repo you prefer
   source .venv/bin/activate
   uv pip install -r requirements.lock
   ```
   Do not bump anything in `requirements.lock` without re-running
   `tests/test_parity.py` (CLAUDE.md §3 rule 3) — that rule now applies
   across three machines, not one.

5. **Set up credentials** per §2 above: `gh auth login`, Kaggle
   `~/.kaggle/kaggle.json`, and (if this machine will touch TACC)
   `cp launch/tacc/tacc.env.example launch/tacc/tacc.env` and fill in the
   real paths.

6. **VS Code users**: install the "WSL" extension, then `code .` from
   inside the WSL shell at the repo root — this opens VS Code running
   its server INSIDE WSL, so its own file operations stay on the Linux
   filesystem too, not just your terminal commands.

7. **Verify**: run `python env_probe.py` (§5) and sanity-check the output
   makes sense (a device list, no import errors) before trusting anything
   this machine produces.

8. **Install the sync-protocol git hooks** (CLAUDE.md §13) — tracked in
   `.githooks/`, but hooks don't clone with the repo, so this is a
   one-time, per-machine step:
   ```bash
   git config core.hooksPath .githooks
   ```
   This activates a `pre-push` guard (refuses to push `main` when local
   `main` is behind `origin/main`) and a `pre-commit` guard (flags a
   staged CSV whose `machine` column doesn't match this machine's own
   hostname). Both are skippable per-invocation
   (`S4DPC_SKIP_SYNC_CHECK=1 git push`, `S4DPC_SKIP_MACHINE_CHECK=1 git
   commit`, or git's own `--no-verify`) — they're meant as a speed bump
   against forgetting to sync, not a hard gate.

### 3.1 Why the Linux filesystem, not `/mnt/c`

Two independent reasons, both concrete, not theoretical:

- **Correctness**: WSL2's 9P-protocol bridge to `/mnt/c` is dramatically
  slower for the kind of many-small-file access pattern this repo has
  (hundreds of checkpoint files, `.git`'s own object store) — a `git
  status` or `pytest` run that takes seconds on ext4 can take minutes on
  `/mnt/c`. This isn't a style preference, it changes the Kaggle-vs-local
  feedback-loop economics CLAUDE.md §11 already warns about.
- **Isolation**: this machine's WSL shell inherits the FULL Windows `PATH`
  (confirmed directly — `/mnt/c/Users/.../Python310/`, a completely
  unrelated Python 3.10 install, sits on `$PATH` alongside the Linux
  tools). A repo living under `/mnt/c` makes it easy for a stray tool
  resolution to silently pick up the wrong Python/git/whatever. A repo
  under `~/projects` in the Linux filesystem keeps the working directory
  itself unambiguous even when `$PATH` isn't.

Confirmed on this machine: `~/projects/s4dpc` is `ext4` (`stat -f` reports
`Type: ext2/ext3`, mounted from `/dev/sdd`), not `drvfs` — already correct,
not merely intended.

---

## 4. Multi-machine result provenance

Every CSV row (`s4dpc.logging.write_csv`, the function `sweep.py` — the
one true entry point — actually calls) and every checkpoint sidecar JSON
(`s4dpc.identify.save_checkpoint`) now stamps FOUR fields, not two:
`git_sha`, `lockfile_sha`, `machine` (`platform.node()`, i.e. hostname),
and `backend` (`jax.default_backend()` — `"cpu"`/`"gpu"`/`"tpu"`). Added
2026-08-25 (`s4dpc/logging.py`'s `get_machine_id`/`get_jax_backend`) —
before that, `machine`/`backend` were not recorded anywhere in a result
row, only `git_sha`/`lockfile_sha`. This makes CLAUDE.md §3 rule 4 ("never
split one experiment arm across platforms — XLA lowering differs by
backend, and BPTT through unstable dynamics amplifies float differences")
mechanically checkable after the fact: `grep` a results CSV for more than
one distinct `(machine, backend)` pair within what's supposed to be a
single experiment arm is now a real audit, not something you have to
remember to check by hand.

---

## 5. `env_probe.py` — run this on every machine before trusting its output

`python env_probe.py` (repo root, inside the activated venv) prints package
versions, the JAX backend/device list, and two fixed-seed digests (a raw
matmul SHA-256, and an S4 model conv-vs-step forward/gradient digest). Not
asserted against a reference automatically — it's a fingerprint you compare
by eye against another machine's run, exactly as this project's own
docstring says ("Run this before trusting any result from a new machine").

**Three-way baseline captured 2026-08-25 — local CPU, local GPU (this
machine, the ThinkPad P53), and Kaggle T4** — full JSON in
`docs/env_probes/thinkpad_p53.json` (`.gpu`/`.cpu` keys) and
`docs/env_probes/kaggle_t4.json`. Headline digests (truncated to 12 hex
chars for readability — full values in the JSON):

```
              backend  matmul        fwd_digest_cnn  fwd_digest_rnn  grad_digest
local CPU     cpu      ee9f0ab0064f  b5a876b4c7fe    89bb8310be31    2719f9e9d3ac
local GPU     gpu      9b042a3047a2  e69ac8d8852d    8b614469bf83    5b669b8668c0
Kaggle T4     gpu      eb3fd0e19539  061c073f7d8c    8b614469bf83    4f16366eb9aa
```

`param_tree_sha`/`param_count` (3942, matching the external
reproduction's stated M6 count exactly) agree across all three — expected,
architecture-only. Everything numeric diverges between CPU and either
GPU, unsurprising. **The one genuinely interesting result: `fwd_digest_rnn`
(the step-mode/`scan` forward pass) is BIT-IDENTICAL between local GPU and
Kaggle T4** — different physical GPUs, different driver stacks, same
digest — while `fwd_digest_cnn` (the conv-mode/FFT forward pass) differs
between every pair, local-vs-local included (next paragraph). Read
narrowly: this is one fixed-seed, freshly-initialized, float32,
100-step canary — not a general portability guarantee, and specifically
NOT the x64 regime real sweeps run under (`cnn_rnn_max_abs_diff` here is
~5e-6, expected to be loose at float32 per CLAUDE.md's TACC x64 notes) —
but it's a genuine, reproducible directional signal that the recurrent
path travels better across GPU hardware than the convolutional one, on
top of everything else this project has already found about that same
conv/FFT path's sensitivity (`docs/DECISIONS.md`'s "TASK 0 EXTENSION").

**A second, independent local-GPU-specific hazard, confirmed directly
this round: this machine's local GPU is not deterministic with ITSELF
across separate process launches, isolated specifically to the
conv-mode path.** Running `env_probe.py` twice in a row (same code, same
seed) gave two different `fwd_digest_cnn`/`grad_digest` values but an
IDENTICAL `fwd_digest_rnn` and `matmul_sha256` both times. Confirmed this
is a process-boundary effect, not general GPU nondeterminism: building
the conv-mode model twice WITHIN one Python process gives bit-identical
output; local CPU gives bit-identical output across separate process
launches too. Measured magnitude: max abs diff `4.77e-07` (relative
`1.46e-07`) between two conv-mode outputs from separate launches — float32-
machine-epsilon scale, not large, but real. Likely cause (not
investigated further): XLA/cuFFT algorithm-selection autotuning picking
a different kernel on different cold starts. Practical consequence: if
identification (`identify.py`'s conv-mode training) ever ran on this
machine's local GPU across multiple process launches — e.g. a killed-
and-resumed job — the result would not be guaranteed bit-reproducible
even with identical code and seed, unlike Kaggle's presumably more
uniform per-job environment (not independently verified either way).

**One correction to a prior assumption, worth recording explicitly:** an
earlier ad-hoc venv on this SAME machine, built by hand-installing
`jax==0.7.2` outside of `requirements.lock`, silently fell back to CPU
("An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib
is not installed") — this machine DOES have working local GPU access, but
only when the venv is built the documented way (`uv pip install -r
requirements.lock`, §3 step 4). A machine reporting `jax_backend: cpu`
after a manual/improvised install is not evidence of missing hardware;
rebuild the venv from the lock file before concluding that.

**When the Lab PC and Home PC are set up**, run this same command there and
save `docs/env_probes/{machine}.json` next to this one — three of the
likely four+ backends this project touches (local CPU, local GPU, Kaggle
T4) are now on record; TACC's A100 and the other two local machines are
not yet.

---

## 6. Cross-reference

- CLAUDE.md §3 (golden rules), §4 (Kaggle), §12 (TACC operational flow),
  and the new "sync protocol" section added 2026-08-25 for the day-to-day
  pull/push discipline this file's setup work feeds into.
- `launch/tacc/README.md` for TACC-specific machine/file details (this
  file is the multi-machine map; that one is the TACC execution layer's
  own reference).
- `docs/DECISIONS.md`'s dated entries for the reasoning behind any
  specific infra fix mentioned here in passing.
