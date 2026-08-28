"""Part C shared utilities: extracting the pre-final-norm activation
v(z) = b + M*z (to leading order) for a postnorm block, and the derived
quantities the boundedness theory's crossover radius depends on:
P (centering matrix), b = v(0), M = dv/dz|_0, r*_predicted = ||Pb|| /
sigma_max(PM).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.diagnostics import zero_states
from s4dpc.model import StackedModel


def pre_final_norm_activation(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    """v = skip + branch, exactly what the block computes just before
    its FINAL norm call (self.norm for postnorm, self.norm_post for
    postnorm_also) - mirrors ConfigurableBlock.__call__ up to but not
    including that call. Valid for n_layers==1 (this study's only case).
    Returns (v, new_states)."""
    block = model.layers[0]
    z = jnp.concatenate([x, u])
    skip = model.encoder(z[jnp.newaxis, :])
    x_block = skip

    if block.config.memoryless:
        from s4dpc.blocks import _memoryless_s4_branch
        seq_out = _memoryless_s4_branch(block.seq, x_block)
        new_state = states[0]
    else:
        from flax import nnx
        seq_graph, seq_params = nnx.split(block.seq)

        def run_one_channel(params_slice, u_slice, state_slice):
            single_channel_layer = nnx.merge(seq_graph, params_slice)
            return single_channel_layer(u_slice, state_slice)

        seq_out, new_state = jax.vmap(run_one_channel, in_axes=(0, 1, 0), out_axes=(1, 0))(
            seq_params, x_block, states[0]
        )

    if block.config.activation == "gelu":
        from flax import nnx
        seq_out = nnx.gelu(seq_out)

    if block.config.glu:
        gate = jax.nn.sigmoid(block.out2(seq_out))
        branch = block.out(seq_out) * gate
    else:
        branch = block.out(seq_out)

    v = skip + branch if block.config.residual else branch
    return v[0], [new_state]


# nnx.jit-wrapped once at module level - Experiment 1's lesson (unjitted
# per-checkpoint jacfwd calls took ~15 min each; jitted and reused across
# calls, seconds). b/M are recomputed once per checkpoint in
# predicted_r_star, but this is ALSO called repeatedly across many
# checkpoints in Part C's sweeps, so the same reuse-the-compiled-program
# benefit applies here too.
@nnx.jit
def _jit_v_and_M(model: StackedModel, z0: jax.Array, states: list[jax.Array]):
    d_x = model.d_output

    def v_fn(z):
        v, _ = pre_final_norm_activation(model, z[:d_x], z[d_x:], states)
        return v

    return v_fn(z0), jax.jacfwd(v_fn)(z0)


def centering_matrix(H: int) -> np.ndarray:
    return np.eye(H) - np.ones((H, H)) / H


def predicted_r_star(model: StackedModel, eps: float = 1e-6) -> dict:
    """b = v(0,0), M = dv/dz|_(0,0), P = centering matrix. Returns
    r*_predicted = ||Pb|| / sigma_max(PM), floored at
    sqrt(eps*H)/sigma_max(PM) (the theory's own floor when the bias is
    small)."""
    H = model.block_config.d_model
    states = zero_states(model, dtype=jnp.complex128)
    z0 = jnp.zeros((model.d_input,))
    b_j, M_j = _jit_v_and_M(model, z0, states)
    b, M = np.asarray(b_j), np.asarray(M_j)  # (H,), (H, d_input)

    P = centering_matrix(H)
    Pb = P @ b
    PM = P @ M
    sigma_max_PM = float(np.linalg.svd(PM, compute_uv=False)[0])
    Pb_norm = float(np.linalg.norm(Pb))

    r_star_raw = Pb_norm / sigma_max_PM if sigma_max_PM > 0 else float("inf")
    r_star_floor = float(np.sqrt(eps * H)) / sigma_max_PM if sigma_max_PM > 0 else float("inf")
    r_star_predicted = max(r_star_raw, r_star_floor)

    return {
        "Pb_norm": Pb_norm, "sigma_max_PM": sigma_max_PM,
        "r_star_raw": r_star_raw, "r_star_floor": r_star_floor, "r_star_predicted": r_star_predicted,
    }
