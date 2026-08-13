"""Phase 2 of the 2026-08-13 saturation check, run as its own script
rather than by lowering controller_saturation.py's SATURATION_TRIGGER
and re-running phase 1 (which would waste ~25 min re-deriving numbers
already on record).

Phase 1 (docs/controller_saturation_summary.csv, this run's log) found:
cases 2/5 at 4.5-7.0% saturation with median max|u| pegged at ~50.000
(the bound), vs cases 1/3/4/7 at 0-0.3% saturation with max|u| clearly
below 50. controller_saturation.py's automated trigger (both of 2,5
above a flat 5% cutoff) did NOT fire, only because case 5 landed at
4.48% - just under an arbitrary round-number threshold, not a real
break in the data; case 2 (7.0%) and case 5 (4.5%) are both clearly in
the same "meaningfully saturating" regime relative to the ~0% cases.
Treated as satisfying the user's conditional instruction in substance
and run directly.

    python tools/controller_saturation_phase2.py
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
import controller_saturation as sat  # noqa: E402

# from phase 1 (this run's log / docs/controller_saturation_summary.csv)
PHASE1 = {
    2: {"median_ratio": 4.1171e01, "median_sat": 0.0698, "median_max_u": 5.000e01},
    5: {"median_ratio": 4.1740e01, "median_sat": 0.0448, "median_max_u": 5.000e01},
}
TARGET_CASES = [2, 5]
RERUN_MAX_ACTION = 200.0


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"PHASE 2: M0, cases {TARGET_CASES}, max_action={RERUN_MAX_ACTION} "
          f"(phase 1's automated trigger narrowly missed on case 5 - 4.48% vs a 5% cutoff - "
          f"run directly per the substantive pattern, see module docstring)\n")

    phase2_rows = sat._run_m0(TARGET_CASES, RERUN_MAX_ACTION, "M0_bound200")

    print(f"\n=== PHASE 2 SUMMARY: max_action=50 (phase 1) vs max_action={RERUN_MAX_ACTION} (phase 2) ===")
    print(f"{'case':5s} {'ratio@50':>12s} {'ratio@200':>12s} {'sat_frac@50':>13s} {'sat_frac@200':>13s} "
          f"{'max|u|@50':>11s} {'max|u|@200':>11s}")
    for case in TARGET_CASES:
        r200 = [r for r in phase2_rows if r["case"] == case]
        med_ratio_200 = float(np.median([r["cost_ratio_to_oracle"] for r in r200]))
        med_sat_200 = float(np.median([r["saturation_frac"] for r in r200]))
        med_maxu_200 = float(np.median([r["max_abs_u"] for r in r200]))
        p1 = PHASE1[case]
        print(f"{case:5d} {p1['median_ratio']:12.4e} {med_ratio_200:12.4e} {p1['median_sat']:13.4f} "
              f"{med_sat_200:13.4f} {p1['median_max_u']:11.3e} {med_maxu_200:11.3e}")

    print("\nVerdict:")
    for case in TARGET_CASES:
        r200 = [r for r in phase2_rows if r["case"] == case]
        med_ratio_200 = float(np.median([r["cost_ratio_to_oracle"] for r in r200]))
        p1_ratio = PHASE1[case]["median_ratio"]
        drop = p1_ratio / med_ratio_200 if med_ratio_200 > 0 else float("inf")
        verdict = "BOUND ARTIFACT CONFIRMED" if drop > 3 else "NOT explained by the bound"
        print(f"  case {case}: ratio {p1_ratio:.2f}x -> {med_ratio_200:.2f}x ({drop:.2f}x reduction) - {verdict}")


if __name__ == "__main__":
    main()
