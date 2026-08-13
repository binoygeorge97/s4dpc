"""Task 1 (docs/DECISIONS.md, post-Task-3): verify the 310x/466,000x
numbers before building anything else on them. Three prior "results"
this project produced turned out to be bugs (unbounded action, float32
rank deficiency, StaticNorm never completing a run) - a catastrophic
number deserves one verification pass before it becomes a figure.

Trains ONE controller (M3, case 3, seed 0 - an easy case: oracle
1.02x, M3 measured at 310x in the full Task 3 run) and, from a single
shared x0:

  1. CLOSED-LOOP: roll the trained controller 200 steps on the TRUE
     plant and on the M3 SURROGATE it trained through, same x0. If the
     surrogate rollout is itself sensible (bounded, tracking toward the
     origin) while the true-plant rollout diverges, that's a genuine
     reality gap. If the SURROGATE rollout is nonsense (NaN, unbounded,
     not tracking), that's a bug in the decode=True rollout or state
     threading, not a control-transfer finding.

  2. OPEN-LOOP: take the oracle LQR's own closed-loop control sequence
     on the true plant (from the same x0), and feed that EXACT sequence
     into M3 open-loop (no controller, no feedback - just apply the
     recorded controls blindly). Compare M3's 200-step prediction
     against the true plant's actual trajectory under the same
     sequence. This isolates the surrogate's OWN free-running fidelity
     from anything about the trained controller - if M3 can't track
     open-loop despite ~1e-6 Markov error, that is itself the finding,
     independent of BPTT/training entirely.

Saves the trained controller checkpoint (so this doesn't need
re-training a third time) plus CSVs and PNG plots of all three
trajectories (closed-loop true, closed-loop surrogate, open-loop
surrogate-vs-true) for direct visual inspection.

    python tools/verify_m3_case3.py
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
import flax.serialization as serialization

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
import controller_surrogates as cs  # noqa: E402

from s4dpc.control import evaluate_controller_on_true, init_batched_state  # noqa: E402
from s4dpc.identify import _stringify_keys  # noqa: E402

CASE = 3
SEED = 0
VARIANT = "M3"
MAX_ACTION = 50.0  # case 3's assigned bound (controller_oracles.CASE_MAX_ACTION)
HORIZON = 200
DOCS_DIR = _REPO_ROOT / "docs"
FIG_DIR = _REPO_ROOT / "docs" / "verify_m3_case3"


def rollout_controller_on_surrogate(controller, model_graphdef, model_params, x0, states, horizon_N):
    """Same step logic as s4dpc.control.rollout_learned, but records the
    FULL per-step trajectory (x_hist, u_hist) instead of a scalar loss -
    for inspection, not training. x0: (batch, d_x)."""

    def apply_one(state, model_in):
        m = nnx.merge(model_graphdef, model_params)
        return m(model_in, state)

    x = x0
    h = controller.init_hidden(x.shape[0])
    xs, us = [x], []
    for _ in range(horizon_N):
        h, u = controller(h, x)
        model_in = jnp.concatenate([x, u], axis=-1)
        x_next, states = jax.vmap(apply_one, in_axes=(0, 0))(states, model_in)
        xs.append(x_next)
        us.append(u)
        x = x_next
    return np.asarray(jnp.stack(xs)), np.asarray(jnp.stack(us))


def rollout_surrogate_open_loop(model_graphdef, model_params, x0, states, u_sequence):
    """Feed a FIXED, externally-given control sequence through the
    surrogate - no controller, no feedback. x0: (batch, d_x).
    u_sequence: (horizon_N, batch, d_u). Returns x_hist: (horizon_N+1, batch, d_x)."""

    def apply_one(state, model_in):
        m = nnx.merge(model_graphdef, model_params)
        return m(model_in, state)

    x = x0
    xs = [x]
    for u in u_sequence:
        model_in = jnp.concatenate([x, u], axis=-1)
        x, states = jax.vmap(apply_one, in_axes=(0, 0))(states, model_in)
        xs.append(x)
    return np.asarray(jnp.stack(xs))


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"Verifying {VARIANT}/case{CASE}/seed{SEED}, max_action={MAX_ACTION}")

    # ---- train ONE controller (reuse controller_surrogates' exact code path) ----
    cs.SEED_SELECTION[(VARIANT, CASE)] = [SEED]  # restrict to a single member
    ensemble_state, members = cs._train_ensemble_learned(VARIANT, [CASE], MAX_ACTION)
    assert members == [(CASE, SEED)], f"expected a single ({CASE},{SEED}) member, got {members}"
    controller_state = jax.tree_util.tree_map(lambda x: x[0], ensemble_state)
    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, MAX_ACTION, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)

    # save the checkpoint - third time re-training the same result would be needless
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = FIG_DIR / f"controller_{VARIANT}_case{CASE}_seed{SEED}.msgpack"
    pure_dict = _stringify_keys(controller_state.to_pure_dict())
    ckpt_path.write_bytes(serialization.msgpack_serialize(pure_dict))
    print(f"saved controller checkpoint -> {ckpt_path}")

    # ---- shared x0, true plant, surrogate ----
    A_d, B_d = co.get_discrete_matrices(co.DT, CASE)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    x0_batch_eval = jax.random.uniform(
        eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE, dtype=jnp.float64
    )
    x0 = x0_batch_eval[0:1]  # ONE trajectory, batch axis kept at size 1
    print(f"x0 = {np.asarray(x0[0])}")

    surrogate = cs._build_surrogate(VARIANT, CASE, SEED)
    surrogate_graphdef, surrogate_params = nnx.split(surrogate, nnx.Param)
    states0 = init_batched_state(surrogate, 1)

    # ---- 1. CLOSED-LOOP: same controller, true plant vs M3 surrogate ----
    x_true, u_true = evaluate_controller_on_true(controller, A_d, B_d, x0, HORIZON)
    x_surr, u_surr = rollout_controller_on_surrogate(
        controller, surrogate_graphdef, surrogate_params, x0, states0, HORIZON
    )

    # same true_quadratic_cost the full Task 3 run scored every controller
    # with - directly comparable to the 310x number, not a re-derived formula.
    cost_true = co.true_quadratic_cost(x_true, u_true, co.Q_X, co.R_U, co.Q_F)
    cost_surr = co.true_quadratic_cost(x_surr, u_surr, co.Q_X, co.R_U, co.Q_F)
    finite_true = bool(np.all(np.isfinite(x_true)) and np.all(np.isfinite(u_true)))
    finite_surr = bool(np.all(np.isfinite(x_surr)) and np.all(np.isfinite(u_surr)))
    print(f"\nCLOSED-LOOP: cost_true={cost_true:.4e} (finite={finite_true})  "
          f"cost_surrogate={cost_surr:.4e} (finite={finite_surr})")
    print(f"  ||x_true(0)||={np.linalg.norm(x_true[0]):.3e}  ||x_true(200)||={np.linalg.norm(x_true[-1]):.3e}  "
          f"max||x_true||={np.max(np.linalg.norm(x_true, axis=-1)):.3e}")
    print(f"  ||x_surr(0)||={np.linalg.norm(x_surr[0]):.3e}  ||x_surr(200)||={np.linalg.norm(x_surr[-1]):.3e}  "
          f"max||x_surr||={np.max(np.linalg.norm(x_surr, axis=-1)):.3e}")
    print(f"  max|u_true|={np.max(np.abs(u_true)):.3e}  max|u_surr|={np.max(np.abs(u_surr)):.3e}")

    np.savez(FIG_DIR / "closed_loop.npz", x_true=x_true, u_true=u_true, x_surr=x_surr, u_surr=u_surr)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    t_x = np.arange(HORIZON + 1)
    t_u = np.arange(HORIZON)
    axes[0].plot(t_x, np.linalg.norm(x_true[:, 0, :], axis=-1), label="true plant")
    axes[0].plot(t_x, np.linalg.norm(x_surr[:, 0, :], axis=-1), label="M3 surrogate", linestyle="--")
    axes[0].set_yscale("symlog")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("||x(t)||")
    axes[0].set_title(f"Closed-loop: {VARIANT}/case{CASE}/seed{SEED} controller, true plant vs its own surrogate")
    axes[0].legend()
    axes[1].plot(t_u, np.linalg.norm(u_true[:, 0, :], axis=-1), label="true plant")
    axes[1].plot(t_u, np.linalg.norm(u_surr[:, 0, :], axis=-1), label="M3 surrogate", linestyle="--")
    axes[1].set_yscale("symlog")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("||u(t)||")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "closed_loop.png", dpi=130)
    plt.close(fig)

    # ---- 2. OPEN-LOOP: oracle LQR's control sequence, fed into M3 blind ----
    Q = co.Q_X * np.eye(A_d.shape[0])
    R = co.R_U * np.eye(B_d.shape[1])
    K = co.solve_dlqr(A_d, B_d, Q, R)
    x0_np = np.asarray(x0)
    x_lqr_true, u_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_np, HORIZON)  # (H+1,1,d_x), (H,1,d_u)

    x_lqr_on_surrogate = rollout_surrogate_open_loop(
        surrogate_graphdef, surrogate_params, jnp.asarray(x0_np), states0, jnp.asarray(u_lqr)
    )

    step_err = np.linalg.norm(x_lqr_on_surrogate[:, 0, :] - x_lqr_true[:, 0, :], axis=-1)
    finite_ol = bool(np.all(np.isfinite(x_lqr_on_surrogate)))
    print(f"\nOPEN-LOOP (oracle LQR's control sequence fed through M3 blind):")
    print(f"  finite={finite_ol}  ||err|| at step 1/10/50/100/200: "
          f"{step_err[1]:.3e} / {step_err[10]:.3e} / {step_err[50]:.3e} / {step_err[100]:.3e} / {step_err[200]:.3e}")
    print(f"  max ||err|| over trajectory: {np.max(step_err):.3e} at step {int(np.argmax(step_err))}")
    print(f"  ||x_true|| at step 200: {np.linalg.norm(x_lqr_true[-1]):.3e}  "
          f"||x_surrogate_predicted|| at step 200: {np.linalg.norm(x_lqr_on_surrogate[-1]):.3e}")

    np.savez(FIG_DIR / "open_loop.npz", x_lqr_true=x_lqr_true, u_lqr=u_lqr,
             x_lqr_on_surrogate=x_lqr_on_surrogate, step_err=step_err)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(t_x, np.linalg.norm(x_lqr_true[:, 0, :], axis=-1), label="true plant (actual)")
    axes[0].plot(t_x, np.linalg.norm(x_lqr_on_surrogate[:, 0, :], axis=-1), label="M3 prediction (same u sequence)", linestyle="--")
    axes[0].set_yscale("symlog")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("||x(t)||")
    axes[0].set_title(f"Open-loop: oracle LQR's control sequence, case {CASE}, true vs M3's blind prediction")
    axes[0].legend()
    axes[1].plot(t_x, step_err)
    axes[1].set_yscale("symlog")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("||x_M3(t) - x_true(t)||")
    axes[1].set_title("open-loop tracking error growth")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "open_loop.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote {FIG_DIR}/closed_loop.{{png,npz}}, {FIG_DIR}/open_loop.{{png,npz}}, and the controller checkpoint")


if __name__ == "__main__":
    main()
