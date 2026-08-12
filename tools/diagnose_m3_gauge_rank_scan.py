"""Part B.3 (docs/DECISIONS.md), corrected prediction: since D-only's
output is affine in inputs regardless of d_model (encoder -> elementwise
D-scale -> out -> residual -> decoder are all affine, for ANY d_model),
the effective degrees of freedom stays fixed at 60 (W_eff: 9x6, b_eff:
6) - the structural prediction is rank=60 EXACTLY for every d_model, not
"null = 2*d_model" (that was the float32-artifact reading of a
different, incorrect d_model=16 result - see diagnose_m3_rank_x64.py).
null_dim should instead equal n_params(d_model) - 60, growing
~quadratically with d_model (dominated by the out_kernel's d_model^2
term), not linearly.

Runs the SAME x64 Gauss-Newton rank computation as
diagnose_m3_rank_x64.py, at d_model in {8, 16, 32, 64}, on a FRESHLY
INITIALIZED (not trained) model each time - the rank bound is structural
(depends on the forward pass being affine, not on where in parameter
space we linearize), so init suffices and avoids an expensive re-train
at every d_model.

    python tools/diagnose_m3_gauge_rank_scan.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
STATE_SIZE, N_LAYERS, L_MAX = 32, 1, 100
SEED = 0
D_MODELS = [8, 16, 32, 64]


def _extract_params(model: StackedModel) -> dict:
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


def _predict_raw(params: dict, inputs: jax.Array) -> jax.Array:
    x = inputs @ params["encoder_kernel"] + params["encoder_bias"]
    skip = x
    d_vec = params["D"].squeeze(-1)
    x = x * d_vec[None, :]
    x = x @ params["out_kernel"] + params["out_bias"]
    x = skip + x
    return x @ params["decoder_kernel"] + params["decoder_bias"]


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    inputs64 = jnp.asarray(inputs, dtype=jnp.float64)

    print(f"{'d_model':>8s}  {'n_params':>9s}  {'rank':>6s}  {'null_dim':>9s}  {'predicted_null':>14s}  {'match':>6s}")
    for d_model in D_MODELS:
        block_config = BlockConfig(d_model=d_model, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
        key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
        model = StackedModel(
            block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
            decode=False, rngs=nnx.Rngs(params=key),
        )
        params_f32 = _extract_params(model)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params_f32)

        flat_params, unravel = ravel_pytree(params)
        n_params = flat_params.shape[0]

        def predict_flat(flat_p):
            return _predict_raw(unravel(flat_p), inputs64).reshape(-1)

        j = np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64)
        singular_values = np.linalg.svd(j, compute_uv=False)
        eps = np.finfo(np.float64).eps
        sv_tol = singular_values.max() * max(j.shape) * eps
        rank = int(np.sum(singular_values > sv_tol))
        null_dim = n_params - rank
        predicted_null = n_params - 60
        match = "YES" if rank == 60 else "NO"
        print(f"{d_model:8d}  {n_params:9d}  {rank:6d}  {null_dim:9d}  {predicted_null:14d}  {match:>6s}")


if __name__ == "__main__":
    main()
