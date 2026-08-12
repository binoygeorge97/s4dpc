"""Critical sanity check before extending Part B (docs/DECISIONS.md): the
D-only forward pass is PROVABLY affine in `inputs` (encoder, elementwise
D-scale, out, residual add, decoder are all affine/linear ops composed),
so predictions = inputs_batch @ W_eff + b_eff for SOME (W_eff, b_eff)
that depend on the 550 raw params. W_eff is (9,6)=54 numbers, b_eff is
(6,)=6 numbers - only 60 effective degrees of freedom, GLOBALLY, no
matter how the 550 raw params vary. Since predictions are a FIXED LINEAR
function of (W_eff, b_eff) via the L=100 input batch, the Jacobian
d(predictions)/d(raw_params) factors through this 60-dim bottleneck, so
its rank must be AT MOST 60 - structurally, everywhere, not just at
special points.

diagnose_m3_conditioning.py's SVD-based rank computation reported rank
518 (null 32). That contradicts the <=60 structural bound UNLESS its
numerical-rank tolerance (`max_sv * max(dim) * eps`, ~1e-11 relative) is
far tighter than the actual gap in the singular spectrum - i.e., there
may be ~490 singular values that are small but not AT MACHINE-EPSILON
level, that got counted as "rank" instead of "null" by too strict a
threshold. This script prints the FULL singular value spectrum (all 550)
to check for a natural gap near index 60, and empirically verifies the
affine-in-inputs claim directly (not just by code inspection).

    python tools/diagnose_m3_rank_sanity.py
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
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    params = _extract_d_only_params(model)

    # self-check: raw-array reimplementation vs real model
    real_out = _d_only_forward_nnx(model, inputs)
    raw_out = _predict_raw(params, inputs)
    max_diff = float(jnp.max(jnp.abs(real_out - raw_out)))
    print(f"self-check (raw-array vs real model), max abs diff: {max_diff:.3e}")
    if max_diff > 1e-5:
        print("SELF-CHECK FAILED. Stopping.")
        return

    # --- empirical affine-in-inputs check ---
    # if y = inputs @ W_eff + b_eff, then for ANY two inputs rows u, v and
    # scalar a: y(a*u + (1-a)*v) should equal a*y(u) + (1-a)*y(v) exactly.
    rng = np.random.RandomState(0)
    u = jnp.asarray(rng.uniform(-5, 5, size=(D_INPUT,)))
    v = jnp.asarray(rng.uniform(-5, 5, size=(D_INPUT,)))
    a = 0.37
    y_u = _predict_raw(params, u[None, :])[0]
    y_v = _predict_raw(params, v[None, :])[0]
    y_mix = _predict_raw(params, (a * u + (1 - a) * v)[None, :])[0]
    affine_err = float(jnp.max(jnp.abs(y_mix - (a * y_u + (1 - a) * y_v))))
    print(f"\n[affine-in-inputs check] |y(a*u+(1-a)*v) - (a*y(u)+(1-a)*y(v))| max = {affine_err:.3e}")
    print("  (should be ~0 if D-only's output really is an affine function of the input,")
    print("   which the code structure implies regardless of raw-parameter values)")

    # --- extract the effective (W_eff, b_eff) directly by finite-difference-free means ---
    zero_in = jnp.zeros((1, D_INPUT))
    b_eff = _predict_raw(params, zero_in)[0]
    eye = jnp.eye(D_INPUT)
    y_basis = _predict_raw(params, eye)  # (9,6): rows are y(e_i)
    W_eff = y_basis - b_eff[None, :]  # (9,6)
    recon = eye @ W_eff + b_eff[None, :]
    recon_err = float(jnp.max(jnp.abs(recon - y_basis)))
    print(f"\n[effective (W_eff,b_eff)] W_eff shape={W_eff.shape} (54 numbers) + b_eff shape={b_eff.shape} (6 numbers)")
    print(f"  = 60 effective degrees of freedom total, reconstruction check: {recon_err:.3e}")

    # --- full singular value spectrum of the Gauss-Newton Jacobian ---
    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]

    def predict_flat(flat_p):
        return _predict_raw(unravel(flat_p), inputs).reshape(-1)

    j = np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64)
    singular_values = np.sort(np.linalg.svd(j, compute_uv=False))[::-1]  # descending
    print(f"\n[full singular value spectrum] J shape={j.shape}, n_params={n_params}")
    print(f"  top 65 singular values (descending):")
    for i in range(0, 65, 5):
        chunk = singular_values[i : i + 5]
        print(f"    idx {i:3d}-{i+4:3d}: " + "  ".join(f"{v:.4e}" for v in chunk))
    print(f"  ... (indices 65-{n_params - 6} omitted) ...")
    print(f"  bottom 10 singular values (should be ~0 if the 32-dim family from EXP3 is real):")
    for v in singular_values[-10:]:
        print(f"    {v:.4e}")

    eps = np.finfo(np.float64).eps
    sv_tol_tight = singular_values.max() * max(j.shape) * eps
    rank_tight = int(np.sum(singular_values > sv_tol_tight))
    print(f"\n  numerical rank (tight, machine-eps tolerance {sv_tol_tight:.3e}): {rank_tight} / {n_params}")

    # gap-based rank: largest ratio between consecutive singular values (log-scale jump)
    log_sv = np.log10(singular_values + 1e-300)
    gaps = log_sv[:-1] - log_sv[1:]
    biggest_gap_idx = int(np.argmax(gaps))
    print(f"  biggest log10-gap in the spectrum: between index {biggest_gap_idx} and {biggest_gap_idx + 1}"
          f" (values {singular_values[biggest_gap_idx]:.4e} -> {singular_values[biggest_gap_idx + 1]:.4e},"
          f" gap={gaps[biggest_gap_idx]:.2f} decades)")
    print(f"  ==> gap-based effective rank: {biggest_gap_idx + 1}  (structural prediction: <= 60)")


if __name__ == "__main__":
    main()
