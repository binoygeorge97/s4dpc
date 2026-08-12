"""Part B.1+B.2 (docs/DECISIONS.md), corrected for the true 490-dim null
space (not the earlier float32-artifact 32-dim one - see
diagnose_m3_rank_x64.py). All Gauss-Newton computation forces
jax_enable_x64 throughout, not just an output cast.

B.1: test TWO candidate per-channel symmetries for exact forward-output
invariance under a random scale c:
  (a) the user's proposed pairing: D_i -> c*D_i, encoder column i -> /c
  (b) the analytically-derived pairing (docs/DECISIONS.md): D_i -> c*D_i,
      out_kernel row i -> /c
Only (b) is expected to hold - (a) breaks the skip connection's
contribution unless something else compensates (skip = encoder(x) feeds
decoder directly, unaffected by D or out_kernel).

B.2: for whichever transform(s) pass B.1, build the analytic generator
vector (d/dc of the transform at c=1) in the SAME flat parameter
ordering as the Gauss-Newton Jacobian, and report how much of each
generator's norm lies within the null space (cosine similarity to its
own projection onto the null-space span) - this is a direct, symmetry-
independent way to confirm a candidate transform's tangent direction is
actually a flat direction of the loss.

    python tools/diagnose_m3_gauge_verify.py
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
import optax
from flax import nnx
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0
SCALE_C = 2.3


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


def _train_d_only(inputs, targets, key) -> dict:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = _d_only_forward_nnx(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    for _ in range(EPOCHS):
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

    return _extract_params(model)


def _test_transform(name: str, params: dict, inputs: jax.Array, c: float, transform_fn) -> None:
    real_out = _predict_raw(params, inputs)
    new_params = transform_fn(params, c)
    new_out = _predict_raw(new_params, inputs)
    max_diff = float(jnp.max(jnp.abs(real_out - new_out)))
    rel_diff = max_diff / (float(jnp.max(jnp.abs(real_out))) + 1e-12)
    print(f"[B.1: {name}] c={c}, max abs output diff = {max_diff:.3e}  (relative {rel_diff:.3e})")
    invariant = max_diff < 1e-4
    print(f"  {'INVARIANT (holds)' if invariant else 'NOT invariant (does not hold)'}")


def transform_D_encoder(params: dict, c: float, channel: int = 0) -> dict:
    """User's proposed pairing: D_i -> c*D_i, encoder column i -> /c."""
    new = dict(params)
    D = params["D"].at[channel, 0].multiply(c)
    enc_k = params["encoder_kernel"].at[:, channel].divide(c)
    enc_b = params["encoder_bias"].at[channel].divide(c)
    new["D"] = D
    new["encoder_kernel"] = enc_k
    new["encoder_bias"] = enc_b
    return new


def transform_D_outkernel(params: dict, c: float, channel: int = 0) -> dict:
    """Analytically-derived pairing: D_i -> c*D_i, out_kernel row i -> /c."""
    new = dict(params)
    D = params["D"].at[channel, 0].multiply(c)
    out_k = params["out_kernel"].at[channel, :].divide(c)
    new["D"] = D
    new["out_kernel"] = out_k
    return new


def _generator_D_outkernel(params: dict, channel: int) -> dict:
    """d/dc of transform_D_outkernel at c=1, as a pytree matching params'
    structure (zero everywhere except D[channel] and out_kernel[channel,:])."""
    gen = jax.tree_util.tree_map(jnp.zeros_like, params)
    gen["D"] = gen["D"].at[channel, 0].set(params["D"][channel, 0])
    gen["out_kernel"] = gen["out_kernel"].at[channel, :].set(-params["out_kernel"][channel, :])
    return gen


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    params = _train_d_only(inputs, targets, key)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params)
    inputs64 = jnp.asarray(inputs, dtype=jnp.float64)

    print("=== B.1: transform invariance ===")
    _test_transform("D_i & encoder column i (user's proposal)", params, inputs64, SCALE_C,
                     lambda p, c: transform_D_encoder(p, c, channel=3))
    _test_transform("D_i & out_kernel row i (derived)", params, inputs64, SCALE_C,
                     lambda p, c: transform_D_outkernel(p, c, channel=3))

    print("\n=== B.2: null-space alignment ===")
    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]

    def predict_flat(flat_p):
        return _predict_raw(unravel(flat_p), inputs64).reshape(-1)

    j = np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64)
    U, S, Vt = np.linalg.svd(j, full_matrices=True)
    eps = np.finfo(np.float64).eps
    sv_tol = S.max() * max(j.shape) * eps
    rank = int(np.sum(S > sv_tol))
    null_dim = n_params - rank
    print(f"J shape={j.shape}, rank={rank}, null_dim={null_dim} (expect 490)")
    # Vt rows [rank:] span the null space (right singular vectors with ~0 singular value)
    null_basis = Vt[rank:]  # (null_dim, n_params)

    for channel in range(D_MODEL):
        gen_pytree = _generator_D_outkernel(params, channel)
        gen_flat, _ = ravel_pytree(gen_pytree)
        gen_flat = np.asarray(gen_flat, dtype=np.float64)
        gen_norm = np.linalg.norm(gen_flat)
        if gen_norm < 1e-12:
            print(f"  channel {channel:2d}: generator norm ~0 (D_i or out_kernel row near zero), skipping")
            continue
        proj_coeffs = null_basis @ gen_flat  # (null_dim,)
        proj_norm = np.linalg.norm(proj_coeffs)
        cos_sim = proj_norm / gen_norm
        print(f"  channel {channel:2d}: ||generator||={gen_norm:.4e}  "
              f"cos_sim(generator, null-space projection)={cos_sim:.6f}")


if __name__ == "__main__":
    main()
