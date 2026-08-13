"""Task 2a (docs/DECISIONS.md, post-verification): characterize M3's
training-time BPTT instability (Task 3 entry: mean loss through the M3
surrogate spiked to 6.9e13 at the N=200 curriculum phase) - is it the
HORIZON or the SURROGATE? Task 1's open-loop check already showed M3's
own free-running prediction diverges from the true plant under a
realistic, sustained control sequence (tracking error growing to ~205
by step 176) - this task asks whether that same divergence is what's
driving BPTT training itself unstable, by finding the SHORTEST horizon
at which training through M3 breaks and comparing directly against the
TRUE plant at the same horizons (case 3, same seed, same everything
else - the horizon and the surrogate/true choice are the only two
things that vary).

For 5 curriculum caps (N=5, 20, 50, 100, 200 - each cap = the ORIGINAL
curriculum's phases up through that N, not a single-phase run at that N
alone, so cap=50 still spends epochs at N=5/10/20 first, matching how
the real Task 3 curriculum actually reaches each horizon), trains ONE
controller (case 3, seed 0) through M3 AND ONE through the TRUE plant
(rollout_linear) - 10 runs total - then evaluates EVERY resulting
controller on the TRUE plant (the same honest-transfer test used
throughout) regardless of what it trained through, so degraded
real-world performance can be directly attributed to a training-time
break rather than assumed from the training loss alone.

    python tools/task2a_m3_horizon_ablation.py
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
import controller_surrogates as cs  # noqa: E402

CASE = 3
SEED = 0
MAX_ACTION = 50.0
DOCS_DIR = _REPO_ROOT / "docs"

_FULL_CURRICULUM = list(co.CURRICULUM)  # capture before any monkeypatching
CAPS = [5, 20, 50, 100, 200]
CAPPED_CURRICULUM = {
    5: _FULL_CURRICULUM[0:1],
    20: _FULL_CURRICULUM[0:3],
    50: _FULL_CURRICULUM[0:4],
    100: _FULL_CURRICULUM[0:5],
    200: _FULL_CURRICULUM[0:6],
}


def _train_true(cap: int) -> nnx.State:
    """One controller, case 3 seed 0, through the TRUE plant
    (rollout_linear), curriculum capped at `cap`."""
    co.CURRICULUM = CAPPED_CURRICULUM[cap]
    co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
    co.CASES = [CASE]  # _build_member_grid reads co.CASES globally - without this it trains all 7
    co.SEEDS = [SEED]
    grid = co._build_member_grid("M0")
    ensemble_state = co._train_ensemble(grid, f"true@cap{cap}")
    return jax.tree_util.tree_map(lambda x: x[0], ensemble_state)


def _train_m3(cap: int) -> nnx.State:
    """One controller, case 3 seed 0, through the M3 surrogate,
    curriculum capped at `cap`."""
    co.CURRICULUM = CAPPED_CURRICULUM[cap]
    co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
    cs.SEED_SELECTION[("M3", CASE)] = [SEED]
    ensemble_state, members = cs._train_ensemble_learned("M3", [CASE], MAX_ACTION)
    assert members == [(CASE, SEED)]
    return jax.tree_util.tree_map(lambda x: x[0], ensemble_state)


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CASE={CASE} SEED={SEED} MAX_ACTION={MAX_ACTION} CAPS={CAPS}")

    A_d, B_d = co.get_discrete_matrices(co.DT, CASE)
    Q = co.Q_X * np.eye(A_d.shape[0])
    R = co.R_U * np.eye(B_d.shape[1])
    K = co.solve_dlqr(A_d, B_d, Q, R)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    x0_eval_np = np.asarray(
        jax.random.uniform(eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
    )
    x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
    oracle_lqr_cost = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)
    print(f"oracle LQR cost (case {CASE}): {oracle_lqr_cost:.4f}\n")

    rows = []
    for cap in CAPS:
        for target, train_fn in [("true", _train_true), ("M3", _train_m3)]:
            print(f"{'=' * 20} target={target} cap={cap} (curriculum={[p['N'] for p in CAPPED_CURRICULUM[cap]]}) {'=' * 20}")
            t0 = time.time()
            try:
                controller_state = train_fn(cap)
                elapsed = time.time() - t0
                controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, MAX_ACTION, rngs=nnx.Rngs(0))
                nnx.update(controller, controller_state)
                result = co._evaluate(controller, A_d, B_d, eval_key)
                ratio = result["cost"] / oracle_lqr_cost if oracle_lqr_cost > 0 else float("inf")
                print(f"  [{target}@cap{cap}] wall={elapsed:.1f}s  true-plant eval: cost={result['cost']:.4e}  "
                      f"ratio_to_oracle={ratio:.4e}  max|u|={result['max_abs_u']:.3e}  "
                      f"saturation_frac={result['saturation_frac']:.4f}  finite={result['finite']}")
                rows.append({"target": target, "cap": cap, "wall_s": elapsed, "cost": result["cost"],
                             "ratio_to_oracle": ratio, "max_abs_u": result["max_abs_u"],
                             "saturation_frac": result["saturation_frac"], "finite": result["finite"],
                             "oracle_lqr_cost": oracle_lqr_cost, "failed": False})
            except Exception as e:
                import traceback
                print(f"  [{target}@cap{cap}] FAILED: {e}")
                traceback.print_exc()
                rows.append({"target": target, "cap": cap, "failed": True, "oracle_lqr_cost": oracle_lqr_cost})
            print()

    print("\n=== SUMMARY: true-plant eval cost/ratio, by cap, true vs M3 ===")
    print(f"{'cap':5s} {'target':6s} {'cost':>14s} {'ratio_to_oracle':>16s} {'finite':>7s}")
    for cap in CAPS:
        for target in ["true", "M3"]:
            these = [r for r in rows if r["cap"] == cap and r["target"] == target and not r.get("failed")]
            if not these:
                print(f"{cap:5d} {target:6s} {'FAILED':>14s}")
                continue
            r = these[0]
            print(f"{cap:5d} {target:6s} {r['cost']:14.4e} {r['ratio_to_oracle']:16.4e} {str(r['finite']):>7s}")

    ok_rows = [r for r in rows if not r.get("failed")]
    if ok_rows:
        header = sorted({k for r in ok_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in ok_rows]
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "task2a_m3_horizon_ablation.csv").write_text("\n".join(lines))
        print(f"\nwrote {DOCS_DIR / 'task2a_m3_horizon_ablation.csv'}")


if __name__ == "__main__":
    main()
