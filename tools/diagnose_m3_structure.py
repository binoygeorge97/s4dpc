"""Diagnostic: M3 structural ablations, case 3. Part of the M3-underfitting
diagnosis (docs/DECISIONS.md) - not part of the regular pipeline.

Reimplements S4LayerEnsemble.__call__'s CNN-mode (decode=False) branch for
one channel using s4-nnx's own EXPORTED kernel_dplr/causal_convolution
(never reimplemented internally, never vendored or edited), keeping only
the requested term (D-only or conv-only). "full" mode is a self-check:
it must reproduce the real layer's output bit-for-bit before any ablated
result is trusted. The surrounding structure (encoder, per-block out
projection, residual, decoder) is untouched - only the S4 layer's own
CNN computation is ablated, matching what M3 (norm=none, activation=none,
glu=False, residual=True) actually runs everywhere else.

    python tools/diagnose_m3_structure.py
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
from s4_nnx import S4LayerEnsemble, causal_convolution, kernel_dplr

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0


def _ablated_channel(layer: S4LayerEnsemble, u: jax.Array, mode: str) -> jax.Array:
    """One channel, CNN mode. u: (L,). mode: 'full' | 'd_only' | 'conv_only'."""
    step = jnp.clip(jnp.exp(layer.log_step.value), 0.001, 1.0)
    lambd = jnp.clip(layer.Lambda_re.value, None, -1e-4) + 1j * layer.Lambda_im.value
    c_vector = layer.C_real_imag.value[..., 0] + 1j * layer.C_real_imag.value[..., 1]

    d_term = layer.D.value * u
    if mode == "d_only":
        return d_term

    kernel = kernel_dplr(lambd, layer.P.value, layer.P.value, layer.B.value, c_vector, step, layer.l_max)
    conv_term = causal_convolution(u, kernel)
    if mode == "conv_only":
        return conv_term
    if mode == "full":
        return conv_term + d_term
    raise ValueError(mode)


def _forward(model: StackedModel, inputs: jax.Array, mode: str) -> jax.Array:
    """M3-shaped forward (single block: no norm, no activation, no glu,
    residual=True) with the S4 layer's CNN computation replaced by
    _ablated_channel. Assumes n_layers=1 (this diagnostic's config)."""
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x

    seq_graph, seq_params = nnx.split(block.seq)

    def run_one_channel(params_slice, u_slice):
        layer = nnx.merge(seq_graph, params_slice)
        return _ablated_channel(layer, u_slice, mode)

    x = jax.vmap(run_one_channel, in_axes=(0, 1), out_axes=1)(seq_params, x)

    x = block.out(x)  # glu=False for M3
    x = skip + x  # residual=True for M3

    return model.decoder(x)


def _train_ablated(model: StackedModel, inputs, targets, mode: str) -> tuple[float, list[float]]:
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = _forward(m, inputs, mode)
        return jnp.mean((pred - targets) ** 2)

    losses = []
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        losses.append(float(loss))

    return float(loss_fn(model)), losses


def main() -> None:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    mean_target_sq = float(jnp.mean(targets**2))

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )

    # self-check: "full" mode must reproduce the real model's forward pass
    # bit-for-bit before any ablated result is trustworthy.
    states = model.init_state(N=STATE_SIZE)
    real_out, _ = model(inputs, states)
    ablated_full_out = _forward(model, inputs, "full")
    max_diff = float(jnp.max(jnp.abs(real_out - ablated_full_out)))
    print(f"self-check (full mode vs real model), max abs diff: {max_diff:.3e}")
    if max_diff > 1e-5:
        print("SELF-CHECK FAILED - ablation reimplementation does not match the real model. Stopping.")
        return

    for mode, label in (("d_only", "b. D-only (feedthrough alone)"), ("conv_only", "c. conv-only (kernel alone)")):
        m = StackedModel(
            block_config=block_config,
            d_input=D_INPUT,
            d_output=D_OUTPUT,
            n_layers=N_LAYERS,
            decode=False,
            rngs=nnx.Rngs(params=key),
        )
        final_mse, losses = _train_ablated(m, inputs, targets, mode)
        print(f"{label}: final_mse={final_mse:.6f}  nmse={final_mse / mean_target_sq:.6e}")


if __name__ == "__main__":
    main()
