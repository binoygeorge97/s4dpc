"""Null-object W&B logger + CSV row writer (CLAUDE.md §3 rule 7, §8).

Never scatter `if use_wandb:` through experiment code: construct a Logger
for whichever --wandb mode was requested and call .log() unconditionally.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WANDB_MODES = ("online", "offline", "off")


def get_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_lockfile_sha(lockfile: Path | str | None = None) -> str:
    path = Path(lockfile) if lockfile is not None else _REPO_ROOT / "requirements.lock"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"


class Logger:
    """Null-object W&B logger. mode in {"online", "offline", "off"} (default "off", CLAUDE.md §8)."""

    def __init__(self, mode: str = "off", project: str = "s4dpc-dev", **wandb_kwargs: Any) -> None:
        if mode not in _WANDB_MODES:
            raise ValueError(f"--wandb must be one of {_WANDB_MODES}, got {mode!r}")
        self.mode = mode
        self._run = None
        if mode != "off":
            import wandb

            self._run = wandb.init(project=project, mode=mode, **wandb_kwargs)

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(dict(data), step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to `path`, stamping git_sha + lockfile_sha into every row."""
    if not rows:
        raise ValueError("write_csv: no rows to write")

    git_sha = get_git_sha()
    lockfile_sha = get_lockfile_sha()
    stamped = [{**row, "git_sha": git_sha, "lockfile_sha": lockfile_sha} for row in rows]

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stamped[0].keys()))
        writer.writeheader()
        writer.writerows(stamped)
