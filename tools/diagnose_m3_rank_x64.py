"""Follow-up to diagnose_m3_rank_sanity.py (docs/DECISIONS.md): the full
spectrum showed 60 "genuine" singular values (322.6 down to 0.49) then an
IMMEDIATE ~5-order-of-magnitude cliff to ~8e-6 at index 60 - exactly
matching the structural prediction (D-only's output is affine in inputs,
so it has exactly 60 effective degrees of freedom: W_eff (9,6) + b_eff
(6,)). The 518-vs-60 discrepancy from EXP3 looks like float32 rounding
noise: JAX defaults to float32, and the earlier script only cast the
jacfwd OUTPUT to float64 - the computation itself ran in float32, whose
~1e-7 relative precision on a top singular value of ~322 lands right in
the observed ~1e-5 to 1e-8 "null" band.

This re-runs the SAME Gauss-Newton computation with
`jax.config.update("jax_enable_x64", True)` set BEFORE any JAX op, so
the entire jacfwd computation - not just the final cast - happens in
float64. If the true rank drops from 518 to ~60 under x64, that
confirms EXP3's "32 = 2*d_model" finding was a float32 numerical
artifact, not a real mathematical signature - the true null space is
~490-dimensional (550-60), not 32-dimensional.

    python tools/diagnose_m3_rank_x64.py
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
import numpy as np
from flax import nnx
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
SEED = 0


def _extract_d_only_params(model: StackedModel) -> dict:
    block = model.layers[0]
    return {
        "encoder_kernel": model.encoder.kernel.value,
        "encoder_bias": model.encoder.bias.value,
        "D": block.seq.D.value,
        "out_kernel": block.out.kernel.value,
        "out_bias": block.out.bias.value,
        "decoder_kernel": model.decoder.kernel.value,
        "decoder_bias": model.decoder.bias.value,
    }


def _d_only_forward_nnx(model: StackedModel, inputs: jax.Array) -> jax.Array:
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x
    d_vec = block.seq.D.value.squeeze(-1)
    x = x * d_vec[None, :]
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def _predict_raw(params: dict, inputs: jax.Array) -> jax.Array:
    x = inputs @ params["encoder_kernel"] + params["encoder_bias"]
    skip = x
    d_vec = params["D"].squeeze(-1)
    x = x * d_vec[None, :]
    x = x @ params["out_kernel"] + params["out_bias"]
    x = skip + x
    return x @ params["decoder_kernel"] + params["decoder_bias"]


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    inputs = jnp.asarray(inputs, dtype=jnp.float64)

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    params_f32 = _extract_d_only_params(model)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params_f32)
    print(f"param dtypes after cast: {jax.tree_util.tree_leaves(params)[0].dtype}")

    real_out = _d_only_forward_nnx(model, inputs.astype(jnp.float32))
    raw_out = _predict_raw(params, inputs)
    max_diff = float(jnp.max(jnp.abs(real_out.astype(jnp.float64) - raw_out)))
    print(f"self-check (raw-array f64 vs real model f32), max abs diff: {max_diff:.3e}")

    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]
    print(f"flat_params dtype: {flat_params.dtype}, n_params={n_params}")

    def predict_flat(flat_p):
        return _predict_raw(unravel(flat_p), inputs).reshape(-1)

    j = jax.jacfwd(predict_flat)(flat_params)
    print(f"J dtype: {j.dtype}, shape: {j.shape}")
    j = np.asarray(j, dtype=np.float64)

    singular_values = np.sort(np.linalg.svd(j, compute_uv=False))[::-1]
    print(f"\n[x64 full singular value spectrum]")
    print("  around the predicted cliff (idx 55-65):")
    for i in range(55, 66):
        print(f"    idx {i:3d}: {singular_values[i]:.6e}")
    print("  bottom 10:")
    for v in singular_values[-10:]:
        print(f"    {v:.6e}")

    eps = np.finfo(np.float64).eps
    sv_tol = singular_values.max() * max(j.shape) * eps
    rank = int(np.sum(singular_values > sv_tol))
    print(f"\n  numerical rank (x64, tol={sv_tol:.3e}): {rank} / {n_params}")
    print(f"  null dimension: {n_params - rank}  (structural prediction: {n_params} - 60 = {n_params - 60})")


if __name__ == "__main__":
    main()
