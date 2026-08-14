"""One-line status query for a running orchestrate.py job.

    python launch/colab/status.py <session>
"""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python launch/colab/status.py <session>")
        sys.exit(1)
    session = sys.argv[1]
    path = pathlib.Path(__file__).resolve().parent / f"{session}_status.json"
    if not path.exists():
        print(f"[{session}] no status file yet (orchestrator not started, or wrong session name)")
        return
    s = json.loads(path.read_text())
    print(f"[{s['session']}] {s['state']} | {s['n_verified_artifacts']} verified artifacts | "
          f"updated {s['updated_utc']}")
    if s["recent_verified_files"]:
        print("  recent: " + ", ".join(s["recent_verified_files"][-5:]))


if __name__ == "__main__":
    main()
