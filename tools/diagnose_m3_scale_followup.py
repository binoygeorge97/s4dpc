"""Follow-up to diagnose_m3_scale.py (docs/DECISIONS.md): standardizing
inputs/targets did NOT fix EXP 1's stall (physical mse ~2.8e-2 either
way) - per the branch rule this means "something deeper than scale,"
before stopping the chain to report. Two cheap, targeted checks:

1. Condition number of the (standardized) design matrix Z=[x,u] via
   cond(Z^T Z) - directly tests "genuine ill-conditioning inherent to
   the regression" against "harness bug," independent of any S4-specific
   mechanism. (An earlier, uncommitted run before this session's context
   was compacted reported ~2809 on the unstandardized data - re-verified
   here, standardized, rather than trusted from memory.)
2. Gradient-norm trajectory over the standardized refit's first/last few
   steps - rules out a vanishing/exploding-gradient bug in the harness
   itself (as opposed to slow-but-nonzero-gradient genuine convergence).

    python tools/diagnose_m3_scale_followup.py
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


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    inputs_np = np.asarray(inputs, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.float64)

    in_mu, in_sd = inputs_np.mean(axis=0), inputs_np.std(axis=0) + 1e-8
    out_mu, out_sd = targets_np.mean(axis=0), targets_np.std(axis=0) + 1e-8
    inputs_w_np = (inputs_np - in_mu) / in_sd
    targets_w_np = (targets_np - out_mu) / out_sd

    # design matrix with a bias column, standardized units
    Z = np.concatenate([inputs_w_np, np.ones((inputs_w_np.shape[0], 1))], axis=1)
    ZtZ = Z.T @ Z
    cond_ZtZ = np.linalg.cond(ZtZ)
    cond_Z = np.linalg.cond(Z)
    eigvals = np.sort(np.linalg.eigvalsh(ZtZ))
    print(f"[design matrix, standardized] Z shape={Z.shape}")
    print(f"  cond(Z) = {cond_Z:.4e}   cond(Z^T Z) = {cond_ZtZ:.4e}  (= cond(Z)^2, as expected)")
    print(f"  eigenvalues of Z^T Z: smallest 3 = {eigvals[:3]}  largest 3 = {eigvals[-3:]}")

    # unstandardized, for comparison against the earlier (uncommitted, pre-compaction) ~2809 recollection
    Z_raw = np.concatenate([inputs_np, np.ones((inputs_np.shape[0], 1))], axis=1)
    cond_ZtZ_raw = np.linalg.cond(Z_raw.T @ Z_raw)
    print(f"[design matrix, physical units] cond(Z^T Z) = {cond_ZtZ_raw:.4e}")

    # gradient-norm trajectory on the standardized refit
    inputs_w = jnp.asarray(inputs_w_np)
    targets_w = jnp.asarray(targets_w_np)
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = nnx.Linear(D_INPUT, D_OUTPUT, rngs=nnx.Rngs(params=key))
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = m(inputs_w)
        return jnp.mean((pred - targets_w) ** 2)

    print("\n[gradient-norm trajectory, standardized refit]")
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        flat_grad = jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(grads)])
        grad_norm = float(jnp.linalg.norm(flat_grad))
        if step < 5 or step % 400 == 0 or step == EPOCHS - 1:
            print(f"  step {step:4d}  loss={float(loss):.6e}  grad_norm={grad_norm:.6e}")
        optimizer.update(model, grads)


if __name__ == "__main__":
    main()
