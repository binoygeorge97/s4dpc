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

Trains all 21 (case, seed) members of a given oracle (M0 or M1) as ONE
nnx.vmap'd ensemble (mirrors s4dpc/identify.py's _train_ensemble
pattern exactly - construction via nnx.vmap, per-member forward/loss via
nnx.split + jax.vmap, one shared nnx.Optimizer over the whole batched
ensemble), NOT 21 sequential single-controller runs: a first pilot of
the sequential version measured 361.5s for ONE (case, seed) run at the
full 9000-epoch curriculum, projecting ~4.2 hours for all 42 runs (both
oracles) - the vmapped version pays 2x6=12 total @nnx.jit compilations
(one per curriculum phase per oracle) instead of 42x6=252, and lets the
GPU parallelize across members that a single controller + 1000-sample
batch does not come close to saturating.

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
D_X, D_U = 6, 3

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

# Per-case max_action for Task 1 (docs/DECISIONS.md, 2026-08-13 saturation
# entries). RULE, fixed in advance and referencing only the true plant and
# the oracle (M0) - never M3 or M6, so it cannot favour either surrogate:
# max_action is the smallest value in {50, 200} at which the ORACLE
# controller (M0, true A_d/B_d) does not saturate
# (docs/controller_saturation_summary.csv: fraction of timesteps with
# |u| >= 0.95*max_action).
#
# By that criterion alone, cases 2 and 5 both saturate at 50 (6.98% and
# 4.48%) while 1/3/4/7 don't (0-0.3%). EXCEPTION, still oracle-only
# evidence, never surrogate-referenced: case 2 stays at 50 anyway,
# because moving it to 200 does not serve the rule's purpose - M0's cost
# ratio barely improves (41.17x -> 30.27x, vs case 5's 41.74x -> 2.28x)
# and per-seed variance explodes (1.92x/30.27x/128.05x at 200, vs a tight
# 36.8-52.2x at 50) - see docs/DECISIONS.md's phase-2 entry. Saturation
# alone said "raise it"; cost-and-stability together said case 2's
# difficulty isn't the bound, so raising it just trades one confound for
# another without buying interpretability. Case 5 is the only case this
# script trains at 200.
CASE_MAX_ACTION: dict[int, float] = {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 200.0, 6: 50.0, 7: 50.0}

DOCS_DIR = _REPO_ROOT / "docs"


def _build_member_grid(oracle_name: str) -> dict:
    """Returns per-member A/B/x0/init_key stacks (leading axis = 21,
    order = [(case, seed) for case in CASES for seed in SEEDS]) for one
    oracle - A_batch/B_batch are the TRUE (A_d, B_d) for M0, or the
    least-squares (A_hat, B_hat) for M1; eval always happens against the
    TRUE plant regardless (see main()), so this only controls what the
    controller is TRAINED through."""
    member_cases, member_seeds = [], []
    A_list, B_list, x0_list, key_list = [], [], [], []
    for case in CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        if oracle_name == "M0":
            A_train, B_train = A_d, B_d
        else:
            ab_hat, _ = fit_least_squares(case, l_max=100, aprbs_low=-APRBS_RANGE, aprbs_high=APRBS_RANGE)
            A_train, B_train = ab_hat[:, :D_X], ab_hat[:, D_X:]
        for seed in SEEDS:
            init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
            x0_key = jax.random.fold_in(init_key, 999)
            x0 = jax.random.uniform(
                x0_key, (TRAIN_X0_BATCH, D_X), minval=-TRAIN_X0_RANGE, maxval=TRAIN_X0_RANGE, dtype=jnp.float64
            )
            member_cases.append(case)
            member_seeds.append(seed)
            A_list.append(jnp.asarray(A_train, dtype=jnp.float64))
            B_list.append(jnp.asarray(B_train, dtype=jnp.float64))
            x0_list.append(x0)
            key_list.append(init_key)
    return {
        "cases": member_cases, "seeds": member_seeds,
        "A_batch": jnp.stack(A_list), "B_batch": jnp.stack(B_list),
        "x0_batch": jnp.stack(x0_list), "keys": jnp.stack(key_list),
    }


def _train_ensemble(grid: dict, label: str) -> nnx.State:
    A_batch, B_batch, x0_batch, keys = grid["A_batch"], grid["B_batch"], grid["x0_batch"], grid["keys"]

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return BoundedGRUController(D_X, HIDDEN_DIM, D_U, MAX_ACTION, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = make_controller_optimizer(ensemble, CTRL_LR, 0.0, TOTAL_EPOCHS)

    for pi, phase in enumerate(CURRICULUM):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                graphdef, params = nnx.split(e, nnx.Param)

                def single_member(p, A, B, x0):
                    c = nnx.merge(graphdef, p)
                    return rollout_linear(c, x0, A, B, Q_X, R_U, Q_F, N)

                losses = jax.vmap(single_member)(params, A_batch, B_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"  [{label}] phase {pi + 1}/{len(CURRICULUM)} (N={N}) epoch {epoch:4d} | "
                      f"mean DPC loss: {float(loss):.4f}  per-member range: "
                      f"[{float(jnp.min(per_member)):.3f}, {float(jnp.max(per_member)):.3f}]")

    return nnx.state(ensemble, nnx.Param)


SATURATION_THRESHOLD_FRAC = 0.95  # |u| >= this fraction of max_action counts as "at saturation"


def _evaluate(controller: BoundedGRUController, A: np.ndarray, B: np.ndarray, eval_key: jax.Array) -> dict:
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE, dtype=jnp.float64)
    x_hist, u_hist = evaluate_controller_on_true(controller, A, B, x0, EVAL_HORIZON)
    cost = true_quadratic_cost(x_hist, u_hist, Q_X, R_U, Q_F)
    init_norm = float(np.mean(np.linalg.norm(x_hist[0], axis=-1)))
    final_norm = float(np.mean(np.linalg.norm(x_hist[-1], axis=-1)))
    max_norm = float(np.max(np.linalg.norm(x_hist, axis=-1)))
    finite = bool(np.isfinite(cost) and np.all(np.isfinite(x_hist)) and np.all(np.isfinite(u_hist)))
    max_abs_u = float(np.max(np.abs(u_hist))) if finite else float("inf")
    # controller.max_action, NOT the module MAX_ACTION constant - stays
    # correct when called on a controller trained at a different bound
    # (tools/controller_saturation.py's max_action=200 rerun).
    saturation_frac = (
        float(np.mean(np.abs(u_hist) >= SATURATION_THRESHOLD_FRAC * controller.max_action)) if finite else float("nan")
    )
    return {
        "cost": cost if finite else float("inf"), "finite": finite,
        "init_norm": init_norm, "final_norm": final_norm, "max_norm": max_norm,
        "max_abs_u": max_abs_u, "saturation_frac": saturation_frac,
    }


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"MAX_ACTION={MAX_ACTION}  TOTAL_EPOCHS={TOTAL_EPOCHS}  curriculum={[p['N'] for p in CURRICULUM]}")

    oracle_costs: dict[int, float] = {}
    eval_keys: dict[int, jax.Array] = {}
    true_AB: dict[int, tuple] = {}
    for case in CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        true_AB[case] = (A_d, B_d)
        Q = Q_X * np.eye(A_d.shape[0])
        R = R_U * np.eye(B_d.shape[1])
        K = solve_dlqr(A_d, B_d, Q, R)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        eval_keys[case] = eval_key
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (EVAL_BATCH, A_d.shape[0]), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_d, B_d, K, x0_eval_np, EVAL_HORIZON)
        cost_lqr = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)
        oracle_costs[case] = cost_lqr
        print(f"[oracle LQR] case{case} TRUE-plant cost: {cost_lqr:.4f}")

    rows: list[dict] = []
    for oracle_name in ["M0", "M1"]:
        print(f"\n{'=' * 20} {oracle_name} ensemble (21 members: 7 cases x 3 seeds) {'=' * 20}")
        grid = _build_member_grid(oracle_name)
        ensemble_state = _train_ensemble(grid, oracle_name)

        for i, (case, seed) in enumerate(zip(grid["cases"], grid["seeds"])):
            label = f"{oracle_name}/case{case}/seed{seed}"
            try:
                member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                controller = BoundedGRUController(D_X, HIDDEN_DIM, D_U, MAX_ACTION, rngs=nnx.Rngs(0))
                nnx.update(controller, member_state)
                A_d, B_d = true_AB[case]
                result = _evaluate(controller, A_d, B_d, eval_keys[case])
                cost_lqr = oracle_costs[case]
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
                          "oracle_lqr_cost": oracle_costs[case]}
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
