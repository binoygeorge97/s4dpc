"""Task 6 (session brief, 2026-08-13): re-run the horizon sweep properly
for the TRUE plant and M1 (least-squares) - the cheap half (rollout_linear,
no surrogate identification needed). M3/M6 are the expensive half,
tools/horizon_sweep_surrogate.py.

The existing table (docs/task2a_m3_horizon_ablation.csv, this document's
2026-08-13 "Task 2a" entry) is single-seed, case-3-only, and explicitly
flagged as not to be trusted for its shape ("single seed, not a reliable
trend"). This reruns caps {5,20,50,100,200} - each cap = the ORIGINAL
curriculum's phases up through that N, matching
tools/task2a_m3_horizon_ablation.py's CAPPED_CURRICULUM convention
exactly, not a single-phase run at that N alone - for the TRUE plant
(M0) and M1, all 6 control cases (case 6 excluded, same reason
controller_surrogates.py excludes it), 5 seeds, vmapped per (cap,
oracle) over (case, seed).

    python tools/horizon_sweep_oracle.py
"""
from __future__ import annotations

import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402

CONTROL_CASES = [c for c in co.CASES if c != 6]
N_SEEDS = 5
CAPS = [5, 20, 50, 100, 200]
_FULL_CURRICULUM = list(co.CURRICULUM)
CAPPED_CURRICULUM = {
    5: _FULL_CURRICULUM[0:1], 20: _FULL_CURRICULUM[0:3], 50: _FULL_CURRICULUM[0:4],
    100: _FULL_CURRICULUM[0:5], 200: _FULL_CURRICULUM[0:6],
}
DOCS_DIR = _REPO_ROOT / "docs"


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}  CAPS={CAPS}")

    oracle_costs, eval_keys, true_AB = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = co.get_discrete_matrices(co.DT, case)
        true_AB[case] = (A_d, B_d)
        Q = co.Q_X * np.eye(A_d.shape[0])
        R = co.R_U * np.eye(B_d.shape[1])
        K = co.solve_dlqr(A_d, B_d, Q, R)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        eval_keys[case] = eval_key
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (co.EVAL_BATCH, A_d.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
        oracle_costs[case] = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)

    by_bound: dict[float, list[int]] = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    rows = []
    for oracle_name in ["M0", "M1"]:
        for cap in CAPS:
            co.CURRICULUM = CAPPED_CURRICULUM[cap]
            co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
            for max_action, cases in by_bound.items():
                print(f"\n{'=' * 10} {oracle_name} cap={cap} cases={cases} @ max_action={max_action} {'=' * 10}")
                co.CASES = cases
                co.SEEDS = list(range(N_SEEDS))
                grid = co._build_member_grid(oracle_name)
                t0 = time.time()
                ensemble_state = co._train_ensemble(grid, f"{oracle_name}@cap{cap}")
                print(f"  wall time: {time.time() - t0:.1f}s")

                for i, (case, seed) in enumerate(zip(grid["cases"], grid["seeds"])):
                    member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
                    nnx.update(controller, member_state)
                    A_d, B_d = true_AB[case]
                    result = co._evaluate(controller, A_d, B_d, eval_keys[case])
                    ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
                    print(f"  [{oracle_name}@cap{cap}/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  "
                          f"finite={result['finite']}")
                    rows.append({"oracle": oracle_name, "cap": cap, "case": case, "seed": seed,
                                 "cost": result["cost"], "cost_ratio_to_oracle": ratio, "finite": result["finite"]})

    co.CASES = list(range(1, 8))  # restore module-level default, in case anything else imports this process' state
    co.CURRICULUM = _FULL_CURRICULUM
    co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "horizon_sweep_oracle.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'horizon_sweep_oracle.csv'}")

    print("\n=== SUMMARY: median cost_ratio_to_oracle (with spread), per (oracle, cap, case) ===")
    print(f"{'oracle':6s} {'cap':5s} {'case':5s} {'median':>12s} {'min':>12s} {'max':>12s} {'n_finite':>9s}")
    for oracle_name in ["M0", "M1"]:
        for cap in CAPS:
            for case in CONTROL_CASES:
                these = [r for r in rows if r["oracle"] == oracle_name and r["cap"] == cap and r["case"] == case]
                if not these:
                    continue
                vals = [r["cost_ratio_to_oracle"] for r in these]
                n_finite = sum(1 for r in these if r["finite"])
                print(f"{oracle_name:6s} {cap:5d} {case:5d} {np.median(vals):12.4e} {np.min(vals):12.4e} "
                      f"{np.max(vals):12.4e} {n_finite:9d}/{len(these)}")


if __name__ == "__main__":
    main()
