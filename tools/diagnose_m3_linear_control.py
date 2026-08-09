"""Diagnostic: EXP 1, the clean control for the M3 conditioning
investigation (docs/DECISIONS.md). Fits a SINGLE nnx.Linear(9 -> 6) on
case-3 data - same function class as the least-squares floor, no S4
factorization (no encoder/decoder/D/kernel(Lambda,P,B,step) chain at
all) - with the same Adam settings/budget as the M3 diagnosis runs
(lr=1e-3, wd=0, 2000 epochs).

Purpose: isolate whether the multiplicatively-factored S4 parameterization
is *itself* what makes Adam converge slowly on this near-instantaneous
linear-map target, or whether even the minimal, already-fully-identifiable
(full-rank) linear regression struggles under the same optimizer/data
scale. A single nnx.Linear's Gauss-Newton matrix (Z^T Z on the augmented
input) is generically full-rank for L=100 >> 10 parameters, unlike
D-only's - so this result can't be explained by the same exact-rank-
deficiency mechanism diagnose_m3_conditioning.py found.

    python tools/diagnose_m3_linear_control.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares

CASE = 3
L_MAX = 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    mean_target_sq = float(jnp.mean(targets**2))
    _, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    print(f"LS floor mse={ls_mse:.6e} nmse={ls_mse / mean_target_sq:.6e}")

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = nnx.Linear(D_INPUT, D_OUTPUT, rngs=nnx.Rngs(params=key))
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = m(inputs)
        return jnp.mean((pred - targets) ** 2)

    init_mse = float(loss_fn(model))
    for _ in range(EPOCHS):
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
    final_mse = float(loss_fn(model))

    print(f"single nnx.Linear(9,6): init_mse={init_mse:.6e}  final_mse={final_mse:.6e}  "
          f"nmse={final_mse / mean_target_sq:.6e}  ratio_to_LS_floor={final_mse / ls_mse:.6e}")


if __name__ == "__main__":
    main()
