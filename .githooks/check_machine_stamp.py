"""Helper for .githooks/pre-commit: do any staged CSV files carry a
`machine` column (s4dpc.logging.write_csv's stamp) whose value disagrees
with THIS machine's own hostname? A mismatch usually means a CSV
produced on a different machine (or pulled from Kaggle/TACC output) is
about to be committed under a commit made on this one - not necessarily
wrong (cross-machine result collection is sometimes exactly the point),
but worth a deliberate look, not a silent commit.

Exit 0 if no staged CSV has a mismatched machine column (or no `machine`
column at all - older rows predate the stamp). Exit 1 and print the
offending files/values otherwise.
"""
from __future__ import annotations

import csv
import platform
import subprocess
import sys


def staged_csv_paths() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    )
    return [p for p in out.stdout.splitlines() if p.endswith(".csv")]


def staged_content(path: str) -> str:
    out = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True, check=True)
    return out.stdout


def main() -> int:
    this_machine = platform.node() or "unknown"
    problems: list[str] = []

    for path in staged_csv_paths():
        try:
            content = staged_content(path)
        except subprocess.CalledProcessError:
            continue
        if not content.strip():
            continue
        reader = csv.DictReader(content.splitlines())
        if reader.fieldnames is None or "machine" not in reader.fieldnames:
            continue
        machines_seen = {row["machine"] for row in reader if row.get("machine")}
        foreign = machines_seen - {this_machine}
        if foreign:
            problems.append(f"  {path}: stamped machine(s) {sorted(foreign)} != this machine ({this_machine!r})")

    if problems:
        print("pre-commit: staged CSV(s) carry a `machine` stamp from elsewhere:")
        print("\n".join(problems))
        print("If this is intentional (e.g. committing a pulled Kaggle/TACC result), "
              "skip with: S4DPC_SKIP_MACHINE_CHECK=1 git commit ...")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
