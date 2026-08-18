#!/usr/bin/env bash
# Run from EE-119537 (local orchestration machine).
#
# Ensures the persistent s4dpc repo checkout exists on TACC under $WORK, on
# the branch/ref named by TACC_GIT_REF (default: main), fast-forwarded to
# that ref's current tip. Setting TACC_GIT_REF to a feature branch (e.g.
# infra/tacc) lets that branch be validated on real TACC hardware before
# merging to main. Never edits scientific source on TACC (CLAUDE.md §2 rule
# 2) - this only fetches, checks out, and fast-forwards.
#
# Authentication is interactive: you will be prompted for your TACC password
# and 6-digit MFA code by ssh itself. Nothing here stores, logs, or
# automates that.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TACC_ENV_FILE:-$SCRIPT_DIR/tacc.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found." >&2
  echo "Copy launch/tacc/tacc.env.example -> launch/tacc/tacc.env and fill in your values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${TACC_USER:?tacc.env: TACC_USER not set}"
: "${TACC_HOST:?tacc.env: TACC_HOST not set}"
: "${TACC_REPO_DIR:?tacc.env: TACC_REPO_DIR not set}"
TACC_GIT_REF="${TACC_GIT_REF:-main}"

REPO_URL="https://github.com/binoygeorge97/s4dpc.git"

echo "== syncing $TACC_REPO_DIR on $TACC_HOST to ref '$TACC_GIT_REF' =="
echo "   (you will be prompted for your TACC password and MFA code)"

ssh "${TACC_USER}@${TACC_HOST}" bash -s <<EOF
set -euo pipefail
if [ -d "$TACC_REPO_DIR/.git" ]; then
  echo "repo exists at $TACC_REPO_DIR"
  cd "$TACC_REPO_DIR"
else
  echo "repo absent, cloning into $TACC_REPO_DIR..."
  mkdir -p "\$(dirname "$TACC_REPO_DIR")"
  git clone "$REPO_URL" "$TACC_REPO_DIR"
  cd "$TACC_REPO_DIR"
fi

echo "fetching origin..."
git fetch origin

echo "checking out $TACC_GIT_REF..."
git checkout "$TACC_GIT_REF"

echo "fast-forwarding $TACC_GIT_REF to origin/$TACC_GIT_REF (never a hard reset)..."
git merge --ff-only "origin/$TACC_GIT_REF"

echo "ref:     \$(git rev-parse --abbrev-ref HEAD)"
echo "git SHA: \$(git rev-parse --short HEAD)"
EOF

echo "== sync_tacc.sh done =="
