"""Controller Task 2 (docs/DECISIONS.md): train the GRU/DPC controller
through the M0 (true A_d, B_d) and M1 (least-squares A_hat, B_hat)
oracles, all 7 cases, 3 seeds, dpc_example's exact curriculum and LQR
weights, evaluated by the honest transfer test (closed-loop cost on the
TRUE plant).

KILL CRITERION, checked explicitly at the end, not just implied by the
numbers: if the GRU fails to stabilize cases 4 and 6 through the TRUE
plant with bounded actions, the failure is BPTT through unstable
dynamics, not the S4 surrogate, and Task 3 (the surrogates) should NOT
be built - this script only trains and reports; the calling agent
decides whether to proceed based on what's printed here.

Shares one @nnx.jit-compiled train_step PER DISTINCT CURRICULUM HORIZON
(6 total, cached - not per (case, seed, oracle) combination, which would
be 42x6=252 separate compilations for no reason: horizon_N controls a
Python-level loop-unroll count inside rollout_linear, so it must be
static/closed-over, but A/B/x0_batch are ordinary jit arguments and
differ freely across the 42 runs that reuse each cached compile).

    python tools/controller_oracles.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.control import (
    BoundedGRUController,
    evaluate_controller_on_true,
    make_controller_optimizer,
    rollout_linear,
    rollout_lqr_true,
    solve_dlqr,
    true_quadratic_cost,
)
from s4dpc.identify import fit_least_squares
from s4dpc.systems import get_discrete_matrices

CASES = list(range(1, 8))
SEEDS = [0, 1, 2]
DT = 0.01

# dpc_example's exact values, kept per instruction
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
HIDDEN_DIM = 64
CTRL_LR = 1e-3
TRAIN_X0_BATCH = 1000
TRAIN_X0_RANGE = 3.0
CURRICULUM = [
    {"N": 5, "epochs": 1000},
    {"N": 10, "epochs": 2000},
    {"N": 20, "epochs": 2000},
    {"N": 50, "epochs": 1000},
    {"N": 100, "epochs": 1000},
    {"N": 200, "epochs": 2000},
]
TOTAL_EPOCHS = sum(p["epochs"] for p in CURRICULUM)

# per instruction: max_action = 50 = 5x the APRBS training range of [-10,10]
APRBS_RANGE = 10.0
MAX_ACTION = 50.0

EVAL_BATCH = 100
EVAL_X0_RANGE = 5.0
EVAL_HORIZON = 200  # dpc_example's config['N'], the full horizon

KILL_CASES = (4, 6)
DIVERGED_COST_RATIO = 1e6  # cost / oracle_lqr_cost beyond this = failed to stabilize

DOCS_DIR = _REPO_ROOT / "docs"

_TRAIN_STEP_CACHE: dict[int, callable] = {}


def _get_train_step(horizon_n: int):
    if horizon_n not in _TRAIN_STEP_CACHE:
        @nnx.jit
        def train_step(controller, optimizer, x0_batch, A, B):
            def loss_fn(c):
                return rollout_linear(c, x0_batch, A, B, Q_X, R_U, Q_F, horizon_n)

            loss, grads = nnx.value_and_grad(loss_fn)(controller)
            optimizer.update(controller, grads)
            return loss

        _TRAIN_STEP_CACHE[horizon_n] = train_step
        print(f"  (compiled train_step for horizon_n={horizon_n})")
    return _TRAIN_STEP_CACHE[horizon_n]


def _train_one_controller(A: np.ndarray, B: np.ndarray, case: int, seed: int, label: str) -> BoundedGRUController:
    """Controller init key is fold_in(seed, case), not a bare seed -
    matching this session's established per-(case,seed) key derivation
    (s4dpc/identify.py etc) so "seed 0" doesn't mean the identical
    controller initialization reused across all 7 cases and both oracle
    types, which would confound controller-init variance with genuine
    case-difficulty variance."""
    d_x, d_u = A.shape[0], B.shape[1]
    init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    controller = BoundedGRUController(d_x, HIDDEN_DIM, d_u, MAX_ACTION, rngs=nnx.Rngs(init_key))
    optimizer = make_controller_optimizer(controller, CTRL_LR, 0.0, TOTAL_EPOCHS)

    x0_key = jax.random.fold_in(init_key, 999)
    x0_batch = jax.random.uniform(
        x0_key, (TRAIN_X0_BATCH, d_x), minval=-TRAIN_X0_RANGE, maxval=TRAIN_X0_RANGE, dtype=jnp.float64
    )
    A_j = jnp.asarray(A, dtype=jnp.float64)
    B_j = jnp.asarray(B, dtype=jnp.float64)

    for pi, phase in enumerate(CURRICULUM):
        N = phase["N"]
        train_step = _get_train_step(N)
        for epoch in range(phase["epochs"]):
            loss = train_step(controller, optimizer, x0_batch, A_j, B_j)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"    [{label}] phase {pi + 1}/{len(CURRICULUM)} (N={N}) epoch {epoch:4d} | "
                      f"DPC loss: {float(loss):.4f}")
    return controller


def _evaluate(controller: BoundedGRUController, A: np.ndarray, B: np.ndarray, eval_key: jax.Array) -> dict:
    d_x = A.shape[0]
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, d_x), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE, dtype=jnp.float64)
    x_hist, u_hist = evaluate_controller_on_true(controller, A, B, x0, EVAL_HORIZON)
    cost = true_quadratic_cost(x_hist, u_hist, Q_X, R_U, Q_F)
    init_norm = float(np.mean(np.linalg.norm(x_hist[0], axis=-1)))
    final_norm = float(np.mean(np.linalg.norm(x_hist[-1], axis=-1)))
    max_norm = float(np.max(np.linalg.norm(x_hist, axis=-1)))
    finite = bool(np.isfinite(cost) and np.all(np.isfinite(x_hist)) and np.all(np.isfinite(u_hist)))
    max_abs_u = float(np.max(np.abs(u_hist))) if finite else float("inf")
    return {
        "cost": cost if finite else float("inf"), "finite": finite,
        "init_norm": init_norm, "final_norm": final_norm, "max_norm": max_norm,
        "max_abs_u": max_abs_u,
    }


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"MAX_ACTION={MAX_ACTION}  TOTAL_EPOCHS={TOTAL_EPOCHS}  curriculum={[p['N'] for p in CURRICULUM]}")

    rows: list[dict] = []
    oracle_costs: dict[int, float] = {}

    for case in CASES:
        print(f"\n{'=' * 20} case {case} {'=' * 20}")
        A_d, B_d = get_discrete_matrices(DT, case)

        Q = Q_X * np.eye(A_d.shape[0])
        R = R_U * np.eye(B_d.shape[1])
        K = solve_dlqr(A_d, B_d, Q, R)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (EVAL_BATCH, A_d.shape[0]), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_d, B_d, K, x0_eval_np, EVAL_HORIZON)
        cost_lqr = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)
        oracle_costs[case] = cost_lqr
        print(f"  [oracle LQR] case{case} TRUE-plant cost: {cost_lqr:.4f}")

        ab_hat, ls_mse = fit_least_squares(case, l_max=100, aprbs_low=-APRBS_RANGE, aprbs_high=APRBS_RANGE)
        A_hat, B_hat = ab_hat[:, :6], ab_hat[:, 6:]
        print(f"  [M1 fit] case{case} LS mse={ls_mse:.4e}  "
              f"||A_hat-A_d||/||A_d||={np.linalg.norm(A_hat - A_d) / np.linalg.norm(A_d):.4e}  "
              f"||B_hat-B_d||/||B_d||={np.linalg.norm(B_hat - B_d) / np.linalg.norm(B_d):.4e}")

        for oracle_name, A_train, B_train in [("M0", A_d, B_d), ("M1", A_hat, B_hat)]:
            for seed in SEEDS:
                label = f"{oracle_name}/case{case}/seed{seed}"
                try:
                    controller = _train_one_controller(A_train, B_train, case, seed, label)
                    result = _evaluate(controller, A_d, B_d, eval_key)
                    ratio = result["cost"] / cost_lqr if cost_lqr > 0 else float("inf")
                    result.update(oracle=oracle_name, case=case, seed=seed,
                                  oracle_lqr_cost=cost_lqr, cost_ratio_to_oracle=ratio)
                    print(f"  [{label}] cost={result['cost']:.4e}  ratio_to_oracle={ratio:.4e}  "
                          f"init_norm={result['init_norm']:.3e}  final_norm={result['final_norm']:.3e}  "
                          f"max_norm={result['max_norm']:.3e}  max|u|={result['max_abs_u']:.3e}  "
                          f"finite={result['finite']}")
                except Exception as e:
                    import traceback
                    print(f"  [{label}] FAILED: {e}")
                    traceback.print_exc()
                    result = {"oracle": oracle_name, "case": case, "seed": seed, "failed": True,
                              "oracle_lqr_cost": cost_lqr}
                rows.append(result)

    # ============================================================
    # Summary table + KILL CRITERION check
    # ============================================================
    print("\n\n=== SUMMARY: median cost + ratio to oracle LQR, per (oracle, case) ===")
    print(f"{'oracle':6s} {'case':5s} {'oracle_lqr':>12s} {'median_cost':>14s} {'median_ratio':>13s} {'n_finite':>9s}")
    case_oracle_median: dict[tuple, float] = {}
    for oracle_name in ["M0", "M1"]:
        for case in CASES:
            these = [r for r in rows if not r.get("failed") and r["oracle"] == oracle_name and r["case"] == case]
            n_finite = sum(1 for r in these if r["finite"])
            if not these:
                continue
            median_cost = float(np.median([r["cost"] for r in these]))
            median_ratio = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
            case_oracle_median[(oracle_name, case)] = median_ratio
            print(f"{oracle_name:6s} {case:5d} {oracle_costs[case]:12.4f} {median_cost:14.4e} "
                  f"{median_ratio:13.4e} {n_finite:9d}/{len(these)}")

    print(f"\n=== KILL CRITERION CHECK: cases {KILL_CASES} (threshold: cost_ratio_to_oracle > {DIVERGED_COST_RATIO:.0e}) ===")
    kill_triggered = False
    for oracle_name in ["M0", "M1"]:
        for case in KILL_CASES:
            ratio = case_oracle_median.get((oracle_name, case))
            if ratio is None:
                print(f"  {oracle_name}/case{case}: NO DATA")
                continue
            failed = ratio > DIVERGED_COST_RATIO or not np.isfinite(ratio)
            kill_triggered = kill_triggered or failed
            print(f"  {oracle_name}/case{case}: median_ratio_to_oracle={ratio:.4e}  "
                  f"{'*** FAILED TO STABILIZE ***' if failed else 'stabilized'}")

    print(f"\n{'=' * 60}")
    if kill_triggered:
        print("KILL CRITERION TRIGGERED: at least one of M0/M1 failed to stabilize case 4 or case 6")
        print("through the TRUE plant. This means the failure is BPTT-through-unstable-dynamics,")
        print("not the S4 surrogate. STOP - do not build Task 3. Report this immediately.")
    else:
        print("KILL CRITERION NOT TRIGGERED: M0 and M1 both stabilize cases 4 and 6 through the")
        print("TRUE plant. Proceeding to Task 3 (M3/M6 surrogates) is appropriate.")
    print(f"{'=' * 60}")

    # ============================================================
    # CSV
    # ============================================================
    ok_rows = [r for r in rows if not r.get("failed")]
    if ok_rows:
        header = sorted({k for r in ok_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in ok_rows]
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "controller_oracles_summary.csv").write_text("\n".join(lines))
        print(f"\nwrote {DOCS_DIR / 'controller_oracles_summary.csv'}")


if __name__ == "__main__":
    main()
