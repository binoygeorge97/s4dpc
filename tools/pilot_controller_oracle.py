"""One-off pilot (not a standing diagnostic): verifies
tools/controller_oracles.py's mechanics - jit-step caching across the
curriculum, BoundedGRUController's float64 cast, evaluate_controller_on_
true - on a SINGLE (case, seed, oracle) combination with the FULL
dpc_example curriculum, before committing to the real 42-run job. Reuses
controller_oracles.py's own functions directly (not a reimplementation),
so a pass here is evidence about the actual code path the real run uses,
and gives a real per-run timing estimate to project the full job's cost.

    python tools/pilot_controller_oracle.py
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

sys.path.insert(0, str(_REPO_ROOT / "tools"))
from controller_oracles import (  # noqa: E402
    CURRICULUM, EVAL_BATCH, EVAL_HORIZON, EVAL_X0_RANGE, MAX_ACTION, Q_F, Q_X, R_U,
    _evaluate, _train_one_controller,
)
from s4dpc.control import rollout_lqr_true, solve_dlqr, true_quadratic_cost  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402
import numpy as np  # noqa: E402

CASE = 3
SEED = 0


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"MAX_ACTION={MAX_ACTION}  curriculum={[p['N'] for p in CURRICULUM]}")

    A_d, B_d = get_discrete_matrices(0.01, CASE)
    Q = Q_X * np.eye(A_d.shape[0])
    R = R_U * np.eye(B_d.shape[1])
    K = solve_dlqr(A_d, B_d, Q, R)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    x0_eval_np = np.asarray(
        jax.random.uniform(eval_key, (EVAL_BATCH, A_d.shape[0]), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    )
    x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_d, B_d, K, x0_eval_np, EVAL_HORIZON)
    cost_lqr = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)
    print(f"[oracle LQR] case{CASE} TRUE-plant cost: {cost_lqr:.4f}")

    t0 = time.time()
    controller = _train_one_controller(A_d, B_d, CASE, SEED, f"PILOT/M0/case{CASE}/seed{SEED}")
    elapsed = time.time() - t0
    print(f"\ntraining wall time: {elapsed:.1f}s for {sum(p['epochs'] for p in CURRICULUM)} total epochs")

    result = _evaluate(controller, A_d, B_d, eval_key)
    ratio = result["cost"] / cost_lqr
    print(f"\n[pilot result] cost={result['cost']:.4e}  ratio_to_oracle={ratio:.4e}  "
          f"init_norm={result['init_norm']:.3e}  final_norm={result['final_norm']:.3e}  "
          f"max_norm={result['max_norm']:.3e}  max|u|={result['max_abs_u']:.3e}  finite={result['finite']}")
    print(f"\nPILOT {'PASS' if result['finite'] and ratio < 1e6 else 'FAIL'} - "
          f"projected full job (42 runs): ~{elapsed * 42 / 60:.1f} minutes")


if __name__ == "__main__":
    main()
