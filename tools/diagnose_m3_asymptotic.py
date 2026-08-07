"""Diagnostic: is M3's D-only underfit on case 3 asymptotically converging
(slow/conditioning) or stuck at a real floor? Part of the M3 diagnosis
(docs/DECISIONS.md) - not part of the regular pipeline.

50k steps, logging every 500, log-log plot: a straight line means
slow-but-converging; a bend/plateau means a real floor.

    python tools/diagnose_m3_asymptotic.py
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
import numpy as np
import optax
from flax import nnx
from s4_nnx import S4LayerEnsemble

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 50_000
LOG_EVERY = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0

OUT_PATH = _REPO_ROOT / "docs" / "m3_d_only_50k_loglog.png"


def _d_only_forward(model: StackedModel, inputs: jax.Array) -> jax.Array:
    """Same D-only ablation as tools/diagnose_m3_structure.py (kept as a
    small standalone copy here rather than importing a sibling script)."""
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x

    seq_graph, seq_params = nnx.split(block.seq)

    def run_one_channel(params_slice, u_slice):
        layer: S4LayerEnsemble = nnx.merge(seq_graph, params_slice)
        return layer.D.value * u_slice

    x = jax.vmap(run_one_channel, in_axes=(0, 1), out_axes=1)(seq_params, x)
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def main() -> None:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    mean_target_sq = float(jnp.mean(targets**2))
    _, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = _d_only_forward(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    steps: list[int] = []
    losses: list[float] = []
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        if step % LOG_EVERY == 0 or step == EPOCHS - 1:
            loss_v = float(loss)
            steps.append(step + 1)  # +1: log-log plot needs step > 0
            losses.append(loss_v)
            print(f"step {step:6d}  mse {loss_v:.6e}  nmse {loss_v / mean_target_sq:.6e}")

    final_mse = float(loss_fn(model))
    print(f"final mse (after {EPOCHS} steps): {final_mse:.6e}")
    print(f"final nmse: {final_mse / mean_target_sq:.6e}")
    print(f"LS floor mse: {ls_mse:.6e}  (ratio final/floor: {final_mse / ls_mse:.6e})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(steps, losses)
    ax.axhline(ls_mse, color="red", linestyle="--", label=f"LS floor ({ls_mse:.1e})")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("teacher-forced MSE (log scale)")
    ax.set_title(f"D-only, case {CASE}, {EPOCHS} steps, lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
    ax.legend()
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
