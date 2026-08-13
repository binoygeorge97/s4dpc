"""Task 2b (docs/DECISIONS.md, post-verification): characterize M6's
reality gap. Task 3 showed M6's training loss stays bounded (~10-560)
through the same case group where M3's blows up, while M6's true-plant
transfer is just as catastrophic (67x-187,000x) - the controller found
something that works against the M6 surrogate specifically, without
that generalizing. The user's prediction: it exits the identification
support (the APRBS-driven trajectory identify.py trained the surrogate
on, aprbs range [-10,10], l_max=100). Checked, not assumed.

Trains ONE controller (M6, case 3, seed 0 - same case Task 1 used for
M3, for direct comparability), then for a batch of x0's rolls it out
BOTH on the M6 surrogate (what the controller was optimized against)
and the true plant (what it's actually judged on), recording every
(x, u) pair visited at every step of every trajectory. Compares both
clouds against the identification data's own per-dimension [min, max]
envelope (case_data's single APRBS trajectory, EXACTLY what M6 was
fit on) - fraction of samples outside the envelope, and by how much.

    python tools/task2b_m6_reality_gap.py
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

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
import controller_surrogates as cs  # noqa: E402

from s4dpc.control import evaluate_controller_on_true, init_batched_state  # noqa: E402
from s4dpc.identify import D_INPUT, case_data  # noqa: E402

CASE = 3
SEED = 0
VARIANT = "M6"
MAX_ACTION = 50.0
HORIZON = 200
APRBS_LOW, APRBS_HIGH, L_MAX = -10.0, 10.0, 100
DOCS_DIR = _REPO_ROOT / "docs"
FIG_DIR = _REPO_ROOT / "docs" / "task2b_m6_reality_gap"


def rollout_controller_on_surrogate_batch(controller, model_graphdef, model_params, x0_batch, states, horizon_N):
    """Same as verify_m3_case3's version, batched (x0_batch: (batch, d_x))."""

    def apply_one(state, model_in):
        m = nnx.merge(model_graphdef, model_params)
        return m(model_in, state)

    x = x0_batch
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


def _support_violation(samples: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """samples: (n, d). lo/hi: (d,) - the identification data's per-dim envelope."""
    outside = (samples < lo) | (samples > hi)
    any_dim_outside = np.any(outside, axis=-1)
    frac_outside = float(np.mean(any_dim_outside))
    excess = np.maximum(lo - samples, 0) + np.maximum(samples - hi, 0)  # per-dim, >=0
    return {
        "frac_samples_outside": frac_outside,
        "max_excess_per_dim": excess.max(axis=0).tolist(),
        "mean_excess_when_outside": float(excess[outside].mean()) if outside.any() else 0.0,
    }


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"Training {VARIANT}/case{CASE}/seed{SEED}, max_action={MAX_ACTION}")

    cs.SEED_SELECTION[(VARIANT, CASE)] = [SEED]
    ensemble_state, members = cs._train_ensemble_learned(VARIANT, [CASE], MAX_ACTION)
    assert members == [(CASE, SEED)]
    controller_state = jax.tree_util.tree_map(lambda x: x[0], ensemble_state)
    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, MAX_ACTION, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)

    A_d, B_d = co.get_discrete_matrices(co.DT, CASE)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    x0_batch = jax.random.uniform(
        eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE, dtype=jnp.float64
    )

    surrogate = cs._build_surrogate(VARIANT, CASE, SEED)
    surrogate_graphdef, surrogate_params = nnx.split(surrogate, nnx.Param)
    states0 = init_batched_state(surrogate, co.EVAL_BATCH)

    print("\nrolling out on the M6 surrogate (what the controller was optimized against)...")
    x_surr, u_surr = rollout_controller_on_surrogate_batch(
        controller, surrogate_graphdef, surrogate_params, x0_batch, states0, HORIZON
    )
    print("rolling out on the true plant (what it's actually judged on)...")
    x_true, u_true = evaluate_controller_on_true(controller, A_d, B_d, x0_batch, HORIZON)

    cost_surr = co.true_quadratic_cost(x_surr, u_surr, co.Q_X, co.R_U, co.Q_F)
    cost_true = co.true_quadratic_cost(x_true, u_true, co.Q_X, co.R_U, co.Q_F)
    print(f"cost on M6 surrogate: {cost_surr:.4e}  cost on true plant: {cost_true:.4e}")

    # identification support: the SINGLE APRBS trajectory M6 (and every
    # variant) was actually fit on - case_data's own (state, control) pairs
    ident_inputs, _ = case_data(CASE, L_MAX, APRBS_LOW, APRBS_HIGH)
    ident_inputs = np.asarray(ident_inputs)  # (L_MAX, D_INPUT=9): [state(6), control(3)]
    ident_x = ident_inputs[:, :co.D_X]
    ident_u = ident_inputs[:, co.D_X:]
    x_lo, x_hi = ident_x.min(axis=0), ident_x.max(axis=0)
    u_lo, u_hi = ident_u.min(axis=0), ident_u.max(axis=0)
    print(f"\nidentification data support (case {CASE}, l_max={L_MAX}, aprbs=[{APRBS_LOW},{APRBS_HIGH}]):")
    print(f"  state per-dim [min,max]: {list(zip(x_lo.round(3), x_hi.round(3)))}")
    print(f"  control per-dim [min,max]: {list(zip(u_lo.round(3), u_hi.round(3)))}")

    # flatten (time, batch, dim) -> (n, dim) for the support check
    x_surr_flat = x_surr.reshape(-1, co.D_X)
    u_surr_flat = u_surr.reshape(-1, co.D_U)
    x_true_flat = x_true.reshape(-1, co.D_X)
    u_true_flat = u_true.reshape(-1, co.D_U)

    results = {}
    for label, (xs, us) in [("surrogate-driven", (x_surr_flat, u_surr_flat)), ("true-plant", (x_true_flat, u_true_flat))]:
        x_viol = _support_violation(xs, x_lo, x_hi)
        u_viol = _support_violation(us, u_lo, u_hi)
        results[label] = {"x": x_viol, "u": u_viol}
        print(f"\n{label} trajectories vs identification support:")
        print(f"  state: {x_viol['frac_samples_outside']:.4f} of samples outside envelope on >=1 dim; "
              f"max excess per dim: {[round(v, 3) for v in x_viol['max_excess_per_dim']]}")
        print(f"  control: {u_viol['frac_samples_outside']:.4f} of samples outside envelope on >=1 dim; "
              f"max excess per dim: {[round(v, 3) for v in u_viol['max_excess_per_dim']]}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIG_DIR / "trajectories.npz", x_surr=x_surr, u_surr=u_surr, x_true=x_true, u_true=u_true,
             ident_x=ident_x, ident_u=ident_u)

    # plot: ||x|| and ||u|| over time (surrogate-driven vs true, mean +/- range across the eval batch)
    # vs the identification data's own ||x||/||u|| range, so the plot answers
    # "does the controller-driven trajectory leave the identification envelope" visually.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    t_x = np.arange(HORIZON + 1)
    ident_x_norm = np.linalg.norm(ident_x, axis=-1)
    x_surr_norm = np.linalg.norm(x_surr, axis=-1)  # (H+1, batch)
    x_true_norm = np.linalg.norm(x_true, axis=-1)
    axes[0].fill_between(t_x, x_surr_norm.min(axis=1), x_surr_norm.max(axis=1), alpha=0.3, label="M6 surrogate (range over 100 x0s)")
    axes[0].plot(t_x, x_surr_norm.mean(axis=1), color="C0")
    axes[0].fill_between(t_x, x_true_norm.min(axis=1), x_true_norm.max(axis=1), alpha=0.3, label="true plant (range over 100 x0s)", color="C1")
    axes[0].plot(t_x, x_true_norm.mean(axis=1), color="C1")
    axes[0].axhspan(ident_x_norm.min(), ident_x_norm.max(), color="gray", alpha=0.25, label="identification data ||x|| range")
    axes[0].set_yscale("symlog")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("||x(t)||")
    axes[0].set_title(f"{VARIANT}/case{CASE}: controller-driven ||x(t)|| vs identification data's range")
    axes[0].legend(fontsize=8)

    ident_u_norm = np.linalg.norm(ident_u, axis=-1)
    t_u = np.arange(HORIZON)
    u_surr_norm = np.linalg.norm(u_surr, axis=-1)
    u_true_norm = np.linalg.norm(u_true, axis=-1)
    axes[1].fill_between(t_u, u_surr_norm.min(axis=1), u_surr_norm.max(axis=1), alpha=0.3, label="M6 surrogate", color="C0")
    axes[1].plot(t_u, u_surr_norm.mean(axis=1), color="C0")
    axes[1].fill_between(t_u, u_true_norm.min(axis=1), u_true_norm.max(axis=1), alpha=0.3, label="true plant", color="C1")
    axes[1].plot(t_u, u_true_norm.mean(axis=1), color="C1")
    axes[1].axhspan(ident_u_norm.min(), ident_u_norm.max(), color="gray", alpha=0.25, label="identification data ||u|| range")
    axes[1].set_yscale("symlog")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("||u(t)||")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "support_comparison.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote {FIG_DIR}/support_comparison.png, {FIG_DIR}/trajectories.npz")


if __name__ == "__main__":
    main()
