"""Diagnostic: M3 identification loss curve, case 3, long run. Part of
the M3-underfitting diagnosis (docs/DECISIONS.md) - not part of the
regular sweep pipeline, but kept for reproducibility of that diagnosis.

    python tools/diagnose_m3_convergence.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import case_data, _build_model

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0

OUT_PATH = _REPO_ROOT / "docs" / "m3_case3_loss_curve.png"


def main() -> None:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    mean_target_sq = float(jnp.mean(targets**2))

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = _build_model(block_config, N_LAYERS, key)
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)
    states = model.init_state(N=STATE_SIZE)

    def loss_fn(m):
        pred, _ = m(inputs, states)
        return jnp.mean((pred - targets) ** 2)

    losses: list[float] = []
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        losses.append(float(loss))
        if step % 200 == 0 or step == EPOCHS - 1:
            print(f"step {step:5d}  mse {float(loss):.6f}  nmse {float(loss) / mean_target_sq:.4e}")

    final_mse = float(loss_fn(model))
    print(f"final mse (after all {EPOCHS} steps): {final_mse:.6f}")
    print(f"final nmse: {final_mse / mean_target_sq:.6e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("teacher-forced MSE (log scale)")
    ax.set_title(f"M3 identification, case {CASE}, {EPOCHS} steps, lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
