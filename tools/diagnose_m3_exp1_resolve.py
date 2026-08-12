"""EXP 1 resolution (docs/DECISIONS.md): is the standardized single-layer
stall genuine-but-slow convergence to the unique convex optimum, or
something else? nnx.Linear(9,6) under MSE IS least squares - same convex
objective, same unique optimum - so there is nothing to tune here, only
two closed-form-adjacent checks:

1. ||W_adam - W_LS||_F / ||W_LS||_F, tracked over training (not just at
   step 2000) - if small and shrinking, Adam is moving toward the correct
   unique optimum, just slowly.
2. Refit the SAME convex objective with optax.lbfgs (falls back to scipy
   L-BFGS-B if the optax API isn't available/behaves unexpectedly at the
   pinned version - reported explicitly either way) for a few hundred
   iterations. A convex quadratic should reach ~1e-14 quickly, confirming
   this is an optimizer-choice/speed issue, not a harness bug.

Both LS and Adam/L-BFGS operate on the STANDARDIZED problem (fixed
mu/sigma from the training set, matching diagnose_m3_scale.py) - the
closed-form solution is recomputed here directly in standardized
coordinates (via the augmented design matrix) so it's directly
comparable to Adam's/L-BFGS's raw kernel/bias, no unstandardizing needed
for the parameter-distance comparison.

    python tools/diagnose_m3_exp1_resolve.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from s4dpc.identify import D_INPUT, D_OUTPUT, case_data

CASE = 3
L_MAX = 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0
LBFGS_ITERS = 300


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    inputs_np = np.asarray(inputs, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.float64)

    in_mu, in_sd = inputs_np.mean(axis=0), inputs_np.std(axis=0) + 1e-8
    out_mu, out_sd = targets_np.mean(axis=0), targets_np.std(axis=0) + 1e-8
    inputs_w_np = (inputs_np - in_mu) / in_sd
    targets_w_np = (targets_np - out_mu) / out_sd

    # closed-form LS in STANDARDIZED coordinates (augmented design matrix)
    Z = np.concatenate([inputs_w_np, np.ones((inputs_w_np.shape[0], 1))], axis=1)
    sol, _, _, _ = np.linalg.lstsq(Z, targets_w_np, rcond=None)
    W_LS = sol[:D_INPUT]
    b_LS = sol[D_INPUT]
    ls_standardized_mse = float(np.mean((Z @ sol - targets_w_np) ** 2))
    W_LS_full = np.concatenate([W_LS, b_LS[None, :]], axis=0)
    print(f"[closed-form LS, standardized] mse={ls_standardized_mse:.6e}")

    inputs_w = jnp.asarray(inputs_w_np)
    targets_w = jnp.asarray(targets_w_np)

    # === check 1: Adam trajectory vs closed-form LS ===
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = nnx.Linear(D_INPUT, D_OUTPUT, rngs=nnx.Rngs(params=key))
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = m(inputs_w)
        return jnp.mean((pred - targets_w) ** 2)

    print("\n[Adam vs closed-form LS, standardized]")
    for step in range(EPOCHS):
        if step % 400 == 0:
            W_adam = np.asarray(model.kernel.value, dtype=np.float64)
            b_adam = np.asarray(model.bias.value, dtype=np.float64)
            W_adam_full = np.concatenate([W_adam, b_adam[None, :]], axis=0)
            rel_err = np.linalg.norm(W_adam_full - W_LS_full) / np.linalg.norm(W_LS_full)
            loss_now = float(loss_fn(model))
            print(f"  step {step:4d}  loss={loss_now:.6e}  ||W_adam-W_LS||_F/||W_LS||_F={rel_err:.6e}")
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

    W_adam = np.asarray(model.kernel.value, dtype=np.float64)
    b_adam = np.asarray(model.bias.value, dtype=np.float64)
    W_adam_full = np.concatenate([W_adam, b_adam[None, :]], axis=0)
    rel_err_final = np.linalg.norm(W_adam_full - W_LS_full) / np.linalg.norm(W_LS_full)
    final_adam_loss = float(loss_fn(model))
    print(f"  step {EPOCHS:4d}  loss={final_adam_loss:.6e}  ||W_adam-W_LS||_F/||W_LS||_F={rel_err_final:.6e}  (final)")

    # === check 2: L-BFGS on the same convex objective, fresh init ===
    key2 = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model2 = nnx.Linear(D_INPUT, D_OUTPUT, rngs=nnx.Rngs(params=key2))
    params0 = {"W": jnp.asarray(model2.kernel.value), "b": jnp.asarray(model2.bias.value)}

    def lbfgs_loss(params):
        pred = inputs_w @ params["W"] + params["b"]
        return jnp.mean((pred - targets_w) ** 2)

    try:
        opt = optax.lbfgs()
        opt_state = opt.init(params0)
        params = params0
        value_and_grad_fun = optax.value_and_grad_from_state(lbfgs_loss)
        for _ in range(LBFGS_ITERS):
            value, grad = value_and_grad_fun(params, state=opt_state)
            updates, opt_state = opt.update(grad, opt_state, params, value=value, grad=grad, value_fn=lbfgs_loss)
            params = optax.apply_updates(params, updates)
        final_lbfgs_loss = float(lbfgs_loss(params))
        print(f"\n[optax.lbfgs, {LBFGS_ITERS} iters, path=optax.lbfgs] standardized loss={final_lbfgs_loss:.6e}")
    except Exception as e:
        print(f"\n[optax.lbfgs raised {type(e).__name__}: {e} - falling back to scipy L-BFGS-B]")
        from scipy.optimize import minimize

        def flat_loss_and_grad(flat_params):
            W = flat_params[: D_INPUT * D_OUTPUT].reshape(D_INPUT, D_OUTPUT)
            b = flat_params[D_INPUT * D_OUTPUT :]
            p = {"W": jnp.asarray(W), "b": jnp.asarray(b)}
            value, grad = jax.value_and_grad(lbfgs_loss)(p)
            flat_grad = np.concatenate([np.asarray(grad["W"]).ravel(), np.asarray(grad["b"]).ravel()])
            return float(value), flat_grad.astype(np.float64)

        x0 = np.concatenate([np.asarray(params0["W"]).ravel(), np.asarray(params0["b"]).ravel()]).astype(np.float64)
        res = minimize(flat_loss_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": LBFGS_ITERS})
        final_lbfgs_loss = float(res.fun)
        print(f"[scipy L-BFGS-B, up to {LBFGS_ITERS} iters, path=scipy fallback] standardized loss={final_lbfgs_loss:.6e}")

    print(f"\n[summary] closed-form LS (standardized) mse={ls_standardized_mse:.6e}")
    print(f"[summary] Adam {EPOCHS} steps: loss={final_adam_loss:.6e}  rel_W_err={rel_err_final:.6e}")
    print(f"[summary] L-BFGS {LBFGS_ITERS} iters: loss={final_lbfgs_loss:.6e}")


if __name__ == "__main__":
    main()
