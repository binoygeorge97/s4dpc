"""Controller Task 2 (2026-08-13, docs/DECISIONS.md): before the M3/M6
surrogate comparison can be trusted on cases 2 and 5 (~41x oracle cost,
neither trivial nor catastrophic, unexplained), check whether
max_action=50 is actually binding for those two plants specifically -
if the GRU is saturated against its own action bound rather than
struggling with the dynamics, the ~41x is a controller-capacity
artifact, not a property of the plant, and would confound the surrogate
comparison the exact same way the ABSENT bound confounded case 4
originally (docs/DECISIONS.md's 2026-08-13 entries).

Re-trains M0 (true (A_d, B_d)) on all 7 cases x 3 seeds, same code path
and same PRNGKeys as controller_oracles.py's original run (deterministic
- this reproduces those exact controllers, it does not run a new
experiment), then evaluates with the now-extended _evaluate (adds
max_abs_u/saturation_frac - see controller_oracles.SATURATION_THRESHOLD
_FRAC) on every case, including case 6 for completeness (informative
even though case 6 is excluded from the Task 3 surrogate table).

PHASE 2 IS CONDITIONAL, encoded here rather than left for a follow-up
round-trip: if cases 2 AND 5 both show median saturation_frac above
SATURATION_TRIGGER while every other non-case-6 case stays below it,
this script automatically re-trains M0 on JUST cases 2 and 5 at
max_action=200 and reports whether the cost ratio drops. If the trigger
condition is not met, phase 2 is skipped and printed as such - the
~41x would then need a different explanation, not assumed to be a bound
artifact just because that was the working hypothesis.

    python tools/controller_saturation.py
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
from flax import nnx

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402

SATURATION_TRIGGER = 0.05  # median saturation_frac above this counts as "saturating"
RERUN_MAX_ACTION = 200.0
DOCS_DIR = _REPO_ROOT / "docs"


def _run_m0(cases: list[int], max_action: float, label: str, oracle_name: str = "M0") -> list[dict]:
    """One ensemble train+eval pass at the given max_action, restricted to
    `cases` x co.SEEDS. Name/default kept as _run_m0/"M0" since every
    existing caller (this module's phase 1/2) is M0-only and relies on
    that default; oracle_name="M1" (controller_oracles_final.py) routes
    through the least-squares (A_hat, B_hat) plant instead via
    _build_member_grid's own M0-vs-else branch. Mutates co.CASES/
    co.MAX_ACTION for the duration of the call (module-level, read by
    _build_member_grid/_train_ensemble/init_ensemble) and restores them
    afterward - same monkeypatch pattern tools/pilot_controller_ensemble.py
    already uses, just scoped to one function instead of a whole script."""
    orig_cases, orig_max_action = co.CASES, co.MAX_ACTION
    co.CASES, co.MAX_ACTION = cases, max_action
    try:
        grid = co._build_member_grid(oracle_name)
        ensemble_state = co._train_ensemble(grid, label)

        rows = []
        for i, (case, seed) in enumerate(zip(grid["cases"], grid["seeds"])):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            A_d, B_d = co.get_discrete_matrices(co.DT, case)
            Q = co.Q_X * np.eye(A_d.shape[0])
            R = co.R_U * np.eye(B_d.shape[1])
            K = co.solve_dlqr(A_d, B_d, Q, R)
            eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
            x0_eval_np = np.asarray(
                jax.random.uniform(eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
            )
            x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
            oracle_lqr_cost = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)

            result = co._evaluate(controller, A_d, B_d, eval_key)
            ratio = result["cost"] / oracle_lqr_cost if oracle_lqr_cost > 0 else float("inf")
            result.update(case=case, seed=seed, max_action=max_action, oracle_lqr_cost=oracle_lqr_cost,
                          cost_ratio_to_oracle=ratio)
            print(f"  [{label}/case{case}/seed{seed}] cost={result['cost']:.4e}  ratio={ratio:.4e}  "
                  f"max|u|={result['max_abs_u']:.3e}  saturation_frac={result['saturation_frac']:.4f}  "
                  f"finite={result['finite']}")
            rows.append(result)
        return rows
    finally:
        co.CASES, co.MAX_ACTION = orig_cases, orig_max_action


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"PHASE 1: M0, all 7 cases x {len(co.SEEDS)} seeds, max_action={co.MAX_ACTION} "
          f"(reproduces controller_oracles.py's original M0 run - same PRNGKeys, same code path)")
    print(f"saturation threshold: |u| >= {co.SATURATION_THRESHOLD_FRAC} * max_action counts as 'at saturation'; "
          f"a case 'saturates' below if median saturation_frac over its 3 seeds > {SATURATION_TRIGGER}\n")

    phase1_rows = _run_m0(co.CASES, co.MAX_ACTION, "M0")

    print("\n=== PHASE 1 SUMMARY: median max|u|, saturation_frac, cost ratio per case ===")
    print(f"{'case':5s} {'median_max|u|':>14s} {'median_sat_frac':>17s} {'median_ratio':>13s}")
    case_median_sat: dict[int, float] = {}
    for case in co.CASES:
        these = [r for r in phase1_rows if r["case"] == case]
        med_u = float(np.median([r["max_abs_u"] for r in these]))
        med_sat = float(np.median([r["saturation_frac"] for r in these]))
        med_ratio = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
        case_median_sat[case] = med_sat
        print(f"{case:5d} {med_u:14.3e} {med_sat:17.4f} {med_ratio:13.4e}")

    target_cases = [2, 5]
    other_cases = [c for c in co.CASES if c not in target_cases and c != 6]  # case 6 excluded: known-diverged, uninformative here
    targets_saturate = all(case_median_sat[c] > SATURATION_TRIGGER for c in target_cases)
    others_dont = all(case_median_sat[c] <= SATURATION_TRIGGER for c in other_cases)
    trigger = targets_saturate and others_dont

    print(f"\ncases {target_cases} saturate (median > {SATURATION_TRIGGER}): {targets_saturate} "
          f"({ {c: case_median_sat[c] for c in target_cases} })")
    print(f"other non-case-6 cases {other_cases} stay below {SATURATION_TRIGGER}: {others_dont} "
          f"({ {c: case_median_sat[c] for c in other_cases} })")
    print(f"case 6 median saturation_frac (reported, not used to gate the trigger): {case_median_sat[6]:.4f}")

    if not trigger:
        print(f"\n{'=' * 60}")
        print("TRIGGER CONDITION NOT MET: the pattern is not 'cases 2/5 saturate, others don't'.")
        print("PHASE 2 SKIPPED. The ~41x cost ratio on cases 2/5 is NOT explained by action")
        print("saturation under this test - it needs a different explanation, not assumed to be")
        print("a bound artifact. Reporting phase-1 numbers only.")
        print(f"{'=' * 60}")
        phase2_rows = []
    else:
        print(f"\n{'=' * 60}")
        print(f"TRIGGER CONDITION MET: re-running M0 on cases {target_cases} at max_action={RERUN_MAX_ACTION}")
        print(f"{'=' * 60}\n")
        phase2_rows = _run_m0(target_cases, RERUN_MAX_ACTION, "M0_bound200")

        print(f"\n=== PHASE 2 SUMMARY: max_action=50 vs max_action={RERUN_MAX_ACTION}, cases {target_cases} ===")
        print(f"{'case':5s} {'ratio@50':>12s} {'ratio@200':>12s} {'sat_frac@50':>13s} {'sat_frac@200':>13s}")
        for case in target_cases:
            r50 = [r for r in phase1_rows if r["case"] == case]
            r200 = [r for r in phase2_rows if r["case"] == case]
            med_ratio_50 = float(np.median([r["cost_ratio_to_oracle"] for r in r50]))
            med_ratio_200 = float(np.median([r["cost_ratio_to_oracle"] for r in r200]))
            med_sat_50 = float(np.median([r["saturation_frac"] for r in r50]))
            med_sat_200 = float(np.median([r["saturation_frac"] for r in r200]))
            print(f"{case:5d} {med_ratio_50:12.4e} {med_ratio_200:12.4e} {med_sat_50:13.4f} {med_sat_200:13.4f}")

    all_rows = phase1_rows + phase2_rows
    header = sorted({k for r in all_rows for k in r.keys()} | {"phase"})
    for r in phase1_rows:
        r.setdefault("phase", "1_bound50_all7")
    for r in phase2_rows:
        r.setdefault("phase", "2_bound200_case2-5")
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in all_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "controller_saturation_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'controller_saturation_summary.csv'}")


if __name__ == "__main__":
    main()
