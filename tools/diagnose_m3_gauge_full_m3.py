"""Part B.4 (docs/DECISIONS.md): does the FULL M3 model (S4 convolution
path active, not just D-only's instantaneous feedthrough) have MORE
null directions than D-only, or fewer?

Unlike D-only, this is NOT a case with a clean closed-form DOF bound:
the S4 kernel (via kernel_dplr) makes each channel's contribution a
CAUSAL LINEAR filter over the full L=100-step sequence, not a per-step
instantaneous map - the naive fully-general causal-linear-map DOF bound
(d_output * d_input * l_max = 6*9*100 = 5400) is far larger than D-only's
60, but S4's kernel is structurally CONSTRAINED to N=32 exponential
modes per channel (not a free 100-tap filter), so the *true* effective
DOF for the conv path is a genuinely open empirical question - measured
here directly, not assumed.

Reimplements the full ("D-only" + "conv") forward pass on raw arrays
using s4-nnx's own EXPORTED kernel_dplr/causal_convolution (never
reimplemented, matching diagnose_m3_structure.py's already-validated
"full" mode), self-checked against the real nnx model before any rank
result is trusted. Same x64 Gauss-Newton rank computation as
diagnose_m3_rank_x64.py.

    python tools/diagnose_m3_gauge_full_m3.py
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
from s4_nnx import S4LayerEnsemble, causal_convolution, kernel_dplr

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
SEED = 0
N_TRAJECTORIES = 8  # independent trajectories stacked so output samples (N_TRAJ*L*d_output) exceed n_params


def _extract_full_params(model: StackedModel) -> dict:
    block = model.layers[0]
    seq = block.seq
    return {
        "encoder_kernel": model.encoder.kernel.value,
        "encoder_bias": model.encoder.bias.value,
        "Lambda_re": seq.Lambda_re.value,
        "Lambda_im": seq.Lambda_im.value,
        "P": seq.P.value,
        "B": seq.B.value,
        "C_real_imag": seq.C_real_imag.value,
        "log_step": seq.log_step.value,
        "D": seq.D.value,
        "out_kernel": block.out.kernel.value,
        "out_bias": block.out.bias.value,
        "decoder_kernel": model.decoder.kernel.value,
        "decoder_bias": model.decoder.bias.value,
    }


def _full_forward_nnx(model: StackedModel, inputs: jax.Array) -> jax.Array:
    """Real nnx forward, full M3 (no ablation) - the reference this
    file's raw-array version is checked against."""
    states = model.init_state(N=STATE_SIZE)
    out, _ = model(inputs, states)
    return out


def _predict_raw(params: dict, inputs: jax.Array, l_max: int) -> jax.Array:
    x = inputs @ params["encoder_kernel"] + params["encoder_bias"]  # (L, d_model)
    skip = x

    lambd = jnp.clip(params["Lambda_re"], None, -1e-4) + 1j * params["Lambda_im"]  # (d_model, N)
    c_vector = params["C_real_imag"][..., 0] + 1j * params["C_real_imag"][..., 1]  # (d_model, N)
    step = jnp.clip(jnp.exp(params["log_step"]), 0.001, 1.0)  # (d_model,)

    def per_channel(lambd_i, p_i, b_i, c_i, step_i, d_i, u_i):
        kernel = kernel_dplr(lambd_i, p_i, p_i, b_i, c_i, step_i, l_max)
        conv_term = causal_convolution(u_i, kernel)
        return conv_term + d_i * u_i

    # vmap over the d_model channel axis; x is (L, d_model) -> transpose to (d_model, L)
    conv_plus_d = jax.vmap(per_channel, in_axes=(0, 0, 0, 0, 0, 0, 1), out_axes=1)(
        lambd, params["P"], params["B"], c_vector, step, params["D"].squeeze(-1), x
    )  # (L, d_model)

    x = conv_plus_d @ params["out_kernel"] + params["out_bias"]
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
    params_f32 = _extract_full_params(model)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params_f32)
    inputs64 = jnp.asarray(inputs, dtype=jnp.float64)

    # self-check: raw-array reimplementation vs real model (float32, since
    # the real model runs in float32 - self-check tolerance accounts for that)
    real_out = _full_forward_nnx(model, inputs)
    raw_out_f32 = _predict_raw(params_f32, inputs, L_MAX)
    max_diff = float(jnp.max(jnp.abs(real_out - raw_out_f32)))
    print(f"self-check (raw-array vs real model, full M3 with conv path), max abs diff: {max_diff:.3e}")
    if max_diff > 1e-3:
        print("SELF-CHECK FAILED - not trusting this Jacobian. Stopping.")
        return

    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]
    print(f"full M3 (conv path active): n_params={n_params}  (D-only alone was 550)")

    # a single L=100 trajectory gives only 600 output samples (100*d_output),
    # far fewer than n_params - the Jacobian's rank would then be trivially
    # capped at 600 regardless of any genuine parameter redundancy, which is
    # exactly what crashed the first version of this script (rank came back
    # essentially saturated at the 600 ceiling). Stack N_TRAJECTORIES
    # independent trajectories' Jacobians instead, so the output-sample count
    # (N_TRAJECTORIES*600) comfortably exceeds n_params and the TRUE
    # parameter-space redundancy (if any) has room to reveal itself.
    traj_inputs, _ = generate_microgrid_trajectory(
        batch_size=N_TRAJECTORIES, length=L_MAX, seed=123, system_case=CASE, dt=0.01,
        aprbs_low=-10.0, aprbs_high=10.0,
    )
    j_blocks = []
    for t in range(N_TRAJECTORIES):
        inputs_t = jnp.asarray(traj_inputs[t], dtype=jnp.float64)

        def predict_flat(flat_p, inputs_t=inputs_t):
            return _predict_raw(unravel(flat_p), inputs_t, L_MAX).reshape(-1)

        j_blocks.append(np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64))
    j = np.concatenate(j_blocks, axis=0)
    print(f"stacked J shape={j.shape} ({N_TRAJECTORIES} trajectories x 600 output samples each)")

    singular_values = np.sort(np.linalg.svd(j, compute_uv=False))[::-1]
    n_sv = singular_values.shape[0]
    eps = np.finfo(np.float64).eps
    sv_tol = singular_values.max() * max(j.shape) * eps
    rank = int(np.sum(singular_values > sv_tol))
    null_dim = n_params - rank

    print(f"\ntop 10 singular values: {singular_values[:10]}")
    print(f"around any cliff (idx max(0,rank-10) to min({n_sv},rank+10)):")
    lo, hi = max(0, rank - 10), min(n_sv, rank + 10)
    for i in range(lo, hi):
        print(f"  idx {i:4d}: {singular_values[i]:.6e}")
    print(f"bottom 10 singular values: {singular_values[-10:]}")
    print(f"\nnumerical rank (x64, stacked): {rank} / {n_params}  (n_sv={n_sv})")
    print(f"null dimension: {null_dim}")
    print(f"\ncomparison: D-only had rank=60, null=490 (out of 550 params, 89.1% redundant)")
    print(f"full M3:    rank={rank}, null={null_dim} (out of {n_params} params, "
          f"{100 * null_dim / n_params:.1f}% redundant)")


if __name__ == "__main__":
    main()
