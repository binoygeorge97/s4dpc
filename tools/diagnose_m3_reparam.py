"""Diagnostic: does reparameterization/init or the optimizer close M3's
D-only gap to the least-squares floor? Part of the M3 diagnosis
(docs/DECISIONS.md) - not part of the regular pipeline.

Three D-only variants, case 3, same 2000-epoch budget:
  a. as-is       - default random init, adamw(lr=1e-3, wd=0)  (baseline)
  b. ls_init     - encoder/decoder hand-set so the D-only forward pass
                   computes the exact least-squares solution AT INIT
                   (D=0, out=0, so only the residual/skip path is live);
                   same optimizer as (a). Tests whether a good starting
                   point trains fine (confirms ill-conditioning: hard to
                   *reach*, not hard to *stay at*) or drifts away (would
                   point somewhere else entirely).
  c. clipped_adam - default random init, higher lr (1e-2) + global-norm
                   gradient clipping. Tests whether a cruder optimizer
                   fix alone closes most of the gap.

Self-checks before trusting any result: (1) the ls_init construction's
loss at step 0 must match fit_least_squares' own mse closely (confirms
the hand-set params actually realize the LS solution, not just "close");
(2) same raw-array-vs-nnx-module D-only forward as
tools/diagnose_m3_conditioning.py (already validated there at
max_abs_diff=0.0, reused verbatim here, not re-derived).

    python tools/diagnose_m3_reparam.py
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

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
SEED = 0


def _d_only_forward(model: StackedModel, inputs: jax.Array) -> jax.Array:
    """Same D-only ablation as diagnose_m3_conditioning.py's
    _d_only_forward_nnx (validated there against the real model,
    max_abs_diff=0.0 - reused verbatim, not re-derived)."""
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x
    d_vec = block.seq.D.value.squeeze(-1)
    x = x * d_vec[None, :]
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def _build_model(key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    return StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )


def _apply_ls_init(model: StackedModel, w_ls_in_out: np.ndarray) -> None:
    """Hand-set encoder/D/out/decoder so the D-only forward pass computes
    exactly `inputs @ w_ls_in_out` at init: zero the D-vector and the
    out-projection (so only the residual/skip path is live), then split
    w_ls_in_out across the first d_output encoder channels and route them
    straight through the decoder via an identity block.

    w_ls_in_out: (d_input, d_output) - x @ w gives the LS prediction.
    """
    block = model.layers[0]
    d_model = D_MODEL
    d_input, d_output = w_ls_in_out.shape

    encoder_kernel = jnp.zeros((d_input, d_model), dtype=model.encoder.kernel.value.dtype)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_ls_in_out))
    model.encoder.kernel.value = encoder_kernel
    model.encoder.bias.value = jnp.zeros_like(model.encoder.bias.value)

    block.seq.D.value = jnp.zeros_like(block.seq.D.value)
    block.out.kernel.value = jnp.zeros_like(block.out.kernel.value)
    block.out.bias.value = jnp.zeros_like(block.out.bias.value)

    decoder_kernel = jnp.zeros((d_model, d_output), dtype=model.decoder.kernel.value.dtype)
    decoder_kernel = decoder_kernel.at[:d_output, :].set(jnp.eye(d_output, dtype=decoder_kernel.dtype))
    model.decoder.kernel.value = decoder_kernel
    model.decoder.bias.value = jnp.zeros_like(model.decoder.bias.value)


def _train(model: StackedModel, inputs, targets, tx) -> tuple[float, float]:
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def loss_fn(m):
        pred = _d_only_forward(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    init_loss = float(loss_fn(model))
    for _ in range(EPOCHS):
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
    final_loss = float(loss_fn(model))
    return init_loss, final_loss


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    mean_target_sq = float(jnp.mean(targets**2))
    ab_hat, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)  # (d_output, d_input)
    w_ls_in_out = np.asarray(ab_hat.T)  # (d_input, d_output): inputs @ w -> LS prediction
    print(f"LS floor mse={ls_mse:.6e} nmse={ls_mse / mean_target_sq:.6e}")

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)

    # a. as-is
    m_a = _build_model(key)
    init_a, final_a = _train(m_a, inputs, targets, optax.adamw(1e-3, weight_decay=0.0))
    print(f"a. as-is        : init_mse={init_a:.6e}  final_mse={final_a:.6e}  nmse={final_a / mean_target_sq:.6e}")

    # b. ls_init - self-check the construction before trusting anything downstream
    m_b = _build_model(key)
    _apply_ls_init(m_b, w_ls_in_out)
    init_b, final_b = _train(m_b, inputs, targets, optax.adamw(1e-3, weight_decay=0.0))
    construction_ok = abs(init_b - ls_mse) / max(ls_mse, 1e-30) < 1e-2 or abs(init_b - ls_mse) < 1e-6
    print(f"b. ls_init self-check: init_mse={init_b:.6e} vs LS floor={ls_mse:.6e} (match: {construction_ok})")
    if not construction_ok:
        print("b. ls_init SELF-CHECK FAILED - construction does not realize the LS solution. Not trusting result.")
    else:
        print(f"b. ls_init      : init_mse={init_b:.6e}  final_mse={final_b:.6e}  nmse={final_b / mean_target_sq:.6e}")

    # c. clipped_adam - same random init as (a), cruder optimizer
    m_c = _build_model(key)
    tx_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2, weight_decay=0.0))
    init_c, final_c = _train(m_c, inputs, targets, tx_c)
    print(f"c. clipped_adam : init_mse={init_c:.6e}  final_mse={final_c:.6e}  nmse={final_c / mean_target_sq:.6e}")


if __name__ == "__main__":
    main()
