"""Task 1's oracle (M0/M1) baseline, unified and complete: all 6 control
cases (case 6 excluded - docs/DECISIONS.md, the oracle itself fails
there), each at its FINAL per-case max_action
(controller_oracles.CASE_MAX_ACTION - 50 for every case except case 5,
which needs 200; see that constant's docstring for the fixed-in-advance,
oracle-only rule and case 2's documented exception).

Why this re-trains M0/M1 for cases already computed at bound 50, rather
than reading back controller_oracles.py's original CSV: that CSV
predates s4dpc.control's saturation_frac field (added during the
2026-08-13 saturation check), so it's missing a column Task 1's table
now requires for every row, including the oracle's. One clean, complete
CSV with a consistent schema (cost, ratio, max|u|, saturation_frac, all
present for every row) beats merging three partial ones with different
histories. The original CSV is NOT superseded for its own purpose (the
Task 2 kill-criterion check, docs/DECISIONS.md) - it stays as the record
of that run; this is a separate, later, Task-1-specific artifact.

Writes docs/controller_oracles_final_summary.csv, columns matching
s4dpc.control._evaluate's dict plus oracle/case/seed/max_action/
oracle_lqr_cost/cost_ratio_to_oracle - the single source
tools/controller_surrogates.py reads for the M0/M1 side of Task 1's
combined table.

    python tools/controller_oracles_final.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
import controller_saturation as sat  # noqa: E402

CONTROL_CASES = [c for c in co.CASES if c != 6]
DOCS_DIR = _REPO_ROOT / "docs"


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES: {CONTROL_CASES}")
    print(f"per-case max_action: { {c: co.CASE_MAX_ACTION[c] for c in CONTROL_CASES} }\n")

    # group cases by their assigned max_action - one ensemble train per
    # (oracle, bound) group, same as controller_saturation.py's pattern
    by_bound: dict[float, list[int]] = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    rows: list[dict] = []
    for oracle_name in ["M0", "M1"]:
        for max_action, cases in by_bound.items():
            print(f"\n{'=' * 20} {oracle_name}, cases {cases}, max_action={max_action} {'=' * 20}")
            group_rows = sat._run_m0(cases, max_action, f"{oracle_name}@{max_action:.0f}", oracle_name=oracle_name)
            for r in group_rows:
                r["oracle"] = oracle_name
            rows.extend(group_rows)

    print("\n=== SUMMARY: median cost/ratio/saturation per (oracle, case) ===")
    print(f"{'oracle':6s} {'case':5s} {'max_action':>10s} {'median_ratio':>13s} {'median_sat_frac':>16s} "
          f"{'median_max|u|':>14s}")
    for oracle_name in ["M0", "M1"]:
        for case in CONTROL_CASES:
            these = [r for r in rows if r["oracle"] == oracle_name and r["case"] == case]
            med_ratio = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
            med_sat = float(np.median([r["saturation_frac"] for r in these]))
            med_maxu = float(np.median([r["max_abs_u"] for r in these]))
            print(f"{oracle_name:6s} {case:5d} {co.CASE_MAX_ACTION[case]:10.1f} {med_ratio:13.4e} "
                  f"{med_sat:16.4f} {med_maxu:14.3e}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "controller_oracles_final_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'controller_oracles_final_summary.csv'}")


if __name__ == "__main__":
    main()
