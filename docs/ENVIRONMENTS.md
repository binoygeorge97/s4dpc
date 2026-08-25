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

**Baseline captured on the ThinkPad P53 (this machine), 2026-08-25** — see
`docs/env_probes/thinkpad_p53.json` for the full JSON. Headline facts:

```
python_version: 3.11.15
platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
jax: 0.7.2, jaxlib: 0.7.2, flax: 0.11.2, optax: 0.2.8, s4_nnx: 0.1.0
jax_backend: gpu   jax_devices: ["cuda:0"]
model_canary.param_count: 3942   (matches the external reproduction's stated M6 count exactly)
cnn_rnn_max_abs_diff: 5.19e-06   (float32 default — NOT the x64 regime real sweeps run under; expected to be loose, see CLAUDE.md's TACC x64 notes)
```

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
save `docs/env_probes/{machine}.json` next to this one — the point of this
section is a three-way diff, not a single baseline sitting unused.

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
