"""Diagnostic: EXP 1 follow-up (docs/DECISIONS.md). Does target/input
scale explain why a single nnx.Linear(9,6) - a convex, uniquely-solvable
problem - stalls at nmse=5.14e-3 instead of reaching ~1e-12 under Adam?

1. Report distributional stats of |x_{k+1}| (targets) and |u| (control
   inputs) over the case-3 training trajectory, plus per-timestep
   magnitude across k=0..99 - case 3 is unstable, so a heavy-tailed/
   growing target distribution (small early, huge late) would explain
   Adam struggling on raw physical-unit MSE even though least-squares
   (closed-form, scale-invariant) doesn't care.
2. Refit the single linear layer with inputs AND targets standardized
   (fixed mu/sigma from the training set - not per-batch, not touching
   test-time distribution assumptions), then unstandardize predictions
   to report physical-units MSE, comparable to the original EXP 1 number.

    python tools/diagnose_m3_scale.py
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

from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares

CASE = 3
L_MAX = 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0

PERCENTILES = [5, 25, 50, 75, 95, 99, 100]


def _report_stats(name: str, arr: np.ndarray) -> None:
    absarr = np.abs(arr)
    print(f"\n[{name}] shape={arr.shape}")
    print(f"  mean={absarr.mean():.4e}  std={arr.std():.4e}  min={absarr.min():.4e}  max={absarr.max():.4e}")
    pct = np.percentile(absarr, PERCENTILES)
    pct_str = "  ".join(f"p{p}={v:.4e}" for p, v in zip(PERCENTILES, pct))
    print(f"  percentiles(|.|): {pct_str}")


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    inputs_np = np.asarray(inputs)
    targets_np = np.asarray(targets)
    x_next = targets_np  # (L, 6): |x_{k+1}|
    u = inputs_np[:, D_OUTPUT:]  # (L, 3): |u_k|

    _report_stats("|x_{k+1}| (targets, all timesteps x all 6 states)", x_next.ravel())
    _report_stats("|u| (control inputs, all timesteps x all 3 channels)", u.ravel())

    print("\n[per-timestep max |x_{k+1}| across the 6 states, k=0..99]")
    per_t_max = np.max(np.abs(x_next), axis=1)
    per_t_norm = np.linalg.norm(x_next, axis=1)
    for k in range(0, L_MAX, 10):
        print(f"  k={k:3d}  max|x|={per_t_max[k]:.4e}  ||x||_2={per_t_norm[k]:.4e}")
    print(f"  k={L_MAX - 1:3d}  max|x|={per_t_max[-1]:.4e}  ||x||_2={per_t_norm[-1]:.4e}  (final step)")
    print(f"  growth ratio ||x(99)|| / ||x(0)|| = {per_t_norm[-1] / per_t_norm[0]:.4e}")

    # ---- standardized refit ----
    in_mu = inputs_np.mean(axis=0)
    in_sd = inputs_np.std(axis=0) + 1e-8
    out_mu = targets_np.mean(axis=0)
    out_sd = targets_np.std(axis=0) + 1e-8

    inputs_w = jnp.asarray((inputs_np - in_mu) / in_sd)
    targets_w = jnp.asarray((targets_np - out_mu) / out_sd)
    out_mu_j = jnp.asarray(out_mu)
    out_sd_j = jnp.asarray(out_sd)

    _, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    print(f"\nLS floor (physical units): mse={ls_mse:.6e}")

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = nnx.Linear(D_INPUT, D_OUTPUT, rngs=nnx.Rngs(params=key))
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn_standardized(m):
        pred_w = m(inputs_w)
        return jnp.mean((pred_w - targets_w) ** 2)

    def physical_mse(m):
        pred_w = m(inputs_w)
        pred_phys = pred_w * out_sd_j + out_mu_j
        return jnp.mean((pred_phys - jnp.asarray(targets_np)) ** 2)

    init_standardized_loss = float(loss_fn_standardized(model))
    init_physical_mse = float(physical_mse(model))
    for _ in range(EPOCHS):
        _, grads = nnx.value_and_grad(loss_fn_standardized)(model)
        optimizer.update(model, grads)
    final_standardized_loss = float(loss_fn_standardized(model))
    final_physical_mse = float(physical_mse(model))
    mean_target_sq = float(np.mean(targets_np**2))

    print(f"\n[standardized refit] init_standardized_loss={init_standardized_loss:.6e}  "
          f"init_physical_mse={init_physical_mse:.6e}")
    print(f"[standardized refit] final_standardized_loss={final_standardized_loss:.6e}  "
          f"final_physical_mse={final_physical_mse:.6e}  "
          f"nmse={final_physical_mse / mean_target_sq:.6e}  "
          f"ratio_to_LS_floor={final_physical_mse / ls_mse:.6e}")


if __name__ == "__main__":
    main()
