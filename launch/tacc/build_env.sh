#!/usr/bin/env bash
# Create/validate the TACC Python environment for s4dpc.
#
# MUST be run on a GPU compute node obtained via idev or Slurm - NOT on a
# login node (CLAUDE.md: environment installation that performs heavy
# computation is not allowed on the login node). Run e.g.:
#
#   idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00 -A IRI26016
#   cd $WORK/.../s4dpc   # the persistent repo checkout (sync_tacc.sh keeps it current)
#   bash launch/tacc/build_env.sh
#
# Installs exactly requirements.lock into a venv under $SCRATCH/venvs/s4dpc.
# Never edits requirements.lock. venv + pip only - no conda (CLAUDE.md §7).
set -euo pipefail

trap 'echo "build_env.sh FAILED (see error above)." >&2' ERR

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: no \$SLURM_JOB_ID in the environment - this does not look like" >&2
  echo "a Slurm allocation (idev/sbatch). build_env.sh must run on a GPU" >&2
  echo "compute node, not the login node. Example:" >&2
  echo "  idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00 -A <allocation>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${TACC_VENV_DIR:-$SCRATCH/venvs/s4dpc}"

echo "== build_env.sh =="
echo "hostname:    $(hostname)"
echo "repo root:   $REPO_ROOT"
echo "venv dir:    $VENV_DIR"
echo

echo "== modules =="
module purge
module load TACC
module load python/3.12.11
module list

PY_BIN="$(command -v python3)"
echo "python3 resolved to: $PY_BIN"

# Verify Python >=3.11 before continuing (pyproject.toml requires >=3.10;
# this project targets 3.12.11 specifically per the operator's instructions).
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: python3 is $(python3 --version), need >=3.11. Check 'module load python/3.12.11' loaded correctly." >&2
  exit 1
fi

echo
echo "== venv (idempotent: reused if present) =="
if [[ -d "$VENV_DIR" ]]; then
  echo "venv already exists at $VENV_DIR, reusing."
else
  mkdir -p "$(dirname "$VENV_DIR")"
  python3 -m venv "$VENV_DIR"
  echo "created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "== installing exactly requirements.lock (never modified) =="
pip install --upgrade pip wheel
pip install --no-cache-dir -r "$REPO_ROOT/requirements.lock"

echo
echo "== env_probe.py =="
cd "$REPO_ROOT"
PROBE_JSON="$(python env_probe.py)"
echo "$PROBE_JSON"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
LOCKFILE_SHA="$(sha256sum "$REPO_ROOT/requirements.lock" | awk '{print $1}' | cut -c1-12)"

echo
echo "== summary =="
# PROBE_JSON goes through the environment, not stdin: `python3 -` already
# reads ITS OWN SCRIPT from stdin, so a heredoc attached to the same
# command consumes stdin for the script source - a piped `echo "$PROBE_JSON" |`
# into that same command never reaches json.load(sys.stdin) inside the
# script, which sees EOF immediately instead (found running this on TACC).
PROBE_JSON="$PROBE_JSON" python3 - "$GIT_SHA" "$LOCKFILE_SHA" <<'PYEOF'
import json, os, sys
d = json.loads(os.environ["PROBE_JSON"])
git_sha, lockfile_sha = sys.argv[1], sys.argv[2]
pkgs = d.get("packages", {})
print(f"Python version   : {d.get('python_version')}")
print(f"JAX version      : {pkgs.get('jax')}")
print(f"Flax version     : {pkgs.get('flax')}")
print(f"Optax version    : {pkgs.get('optax')}")
print(f"JAX devices      : {d.get('jax_devices')}")
print(f"git SHA          : {git_sha}")
print(f"requirements.lock SHA: {lockfile_sha}")
PYEOF

echo
echo "build_env.sh OK. venv ready at: $VENV_DIR"
echo "Activate it in future shells with: source $VENV_DIR/bin/activate"
