"""Analytic skip/branch Jacobian decomposition for a 1-layer StackedModel:

    F(z) = W_dec @ ( W_enc @ z + b_enc ) + b_dec  +  W_dec @ branch(z)
         = [ W_dec @ W_enc ]              @ z     +  [ W_dec @ branch(z) ]
             \\_______constant________/                \\______state-dependent______/

where z = concat([x, u]) (identify.py's D_INPUT layout) and branch(z) is
whatever the block computes between the encoder and the residual add
(norm -> S4 -> activation -> glu, per s4dpc/blocks.py's ConfigurableBlock,
under whichever flags the checkpoint was trained with). The constant term
is read directly off the encoder/decoder weights; the branch term is
whatever's left after subtracting it from the true (autodiff) Jacobian -
no LayerNorm-specific math is hand-derived here, so this decomposition is
valid for ANY block config (M3/M4/M5/M6/M0_S4), not just LayerNorm'd ones.

Only valid for n_layers == 1 (true of every s4dpc checkpoint used here -
identify.py's variant ladder never trains multi-layer stacks). All
Jacobians via jax.jacfwd - never finite-differenced.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.diagnostics import step, zero_states
from s4dpc.model import StackedModel


# nnx.jit-wrapped once at module level, not inside a per-checkpoint call:
# JAX's jit cache keys on the traced function object + the nnx graphdef's
# structural hash, so calling THESE (rather than re-deriving an equivalent
# closure per checkpoint) means every checkpoint of the same variant/shape
# reuses one compiled XLA program instead of re-tracing per call - measured
# directly, the unjitted version took ~15 minutes for a single checkpoint's
# full analysis (100-step trajectory + 2x100-step MSE rollout + 60-point
# origin sweep, ~360 jacfwd/step calls, each paying full eager-mode
# per-op dispatch through the S4 vmap+recurrence) - prohibitive for the
# ~150 checkpoints this experiment covers.
@nnx.jit
def _jit_full_jacobian(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]) -> jax.Array:
    d_x = x.shape[0]

    def f(z: jax.Array) -> jax.Array:
        x_next, _ = step(model, z[:d_x], z[d_x:], states)
        return x_next

    z = jnp.concatenate([x, u])
    return jax.jacfwd(f)(z)


@nnx.jit
def _jit_step(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    return step(model, x, u, states)


@nnx.jit
def _jit_branch_zeroed_step(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    z = jnp.concatenate([x, u])
    skip = model.encoder(z)  # (d_model,)
    x_next_ablated = model.decoder(skip)
    _, new_states = step(model, x, u, states)
    return x_next_ablated, new_states


@nnx.jit
def _jit_dfdx(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]) -> jax.Array:
    def f(xx: jax.Array) -> jax.Array:
        x_next, _ = step(model, xx, u, states)
        return x_next

    return jax.jacfwd(f)(x)


def encoder_decoder_matrices(model: StackedModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(W_enc, b_enc, W_dec, b_dec) in y = W @ z + b form (nnx.Linear
    stores kernel as (in, out), i.e. y = z @ kernel + bias, so W = kernel.T)."""
    W_enc = np.asarray(model.encoder.kernel.value).T  # (d_model, d_input)
    b_enc = np.asarray(model.encoder.bias.value)  # (d_model,)
    W_dec = np.asarray(model.decoder.kernel.value).T  # (d_output, d_model)
    b_dec = np.asarray(model.decoder.bias.value)  # (d_output,)
    return W_enc, b_enc, W_dec, b_dec


def constant_term(model: StackedModel) -> np.ndarray:
    """W_dec @ W_enc: (d_output, d_input) - the Jacobian contribution from
    the skip path alone, exactly constant (independent of z, s)."""
    W_enc, _, W_dec, _ = encoder_decoder_matrices(model)
    return W_dec @ W_enc


def equilibrium_offset(model: StackedModel) -> np.ndarray:
    """W_dec @ b_enc + b_dec: the skip path's contribution to F(0,0) even
    with the branch's own output zeroed - a purely bias-driven offset,
    distinct from anything LayerNorm does. See NOTES.md's bias-ablation
    open question (3.4)."""
    _, b_enc, W_dec, b_dec = encoder_decoder_matrices(model)
    return W_dec @ b_enc + b_dec


def full_jacobian(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]) -> np.ndarray:
    """dF/dz at (x, u, states), z = concat([x, u]). (d_output, d_input)."""
    return np.asarray(_jit_full_jacobian(model, x, u, states))


def decompose(
    model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (J_full, J_constant, J_branch) with J_full = J_constant + J_branch
    exactly (branch is defined as the residual, not measured independently)."""
    J_full = full_jacobian(model, x, u, states)
    J_const = constant_term(model)
    return J_full, J_const, J_full - J_const


def branch_zeroed_step(
    model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]
) -> tuple[jax.Array, list[jax.Array]]:
    """One step with the residual branch's contribution forced to zero at
    inference: the block's output becomes exactly `skip = encoder(z)`
    instead of `skip + branch(z)`, before decoding. Valid only for
    n_layers == 1 (asserted): the S4 hidden state is still updated
    normally via the real `step` (state evolution doesn't depend on
    whether the top-level residual sum keeps or discards the branch
    output computed FROM that same state), so a free-running rollout
    using this function evolves states identically to the un-ablated
    model - only the returned x_next differs."""
    if model.n_layers != 1:
        raise NotImplementedError("branch_zeroed_step assumes n_layers == 1")
    return _jit_branch_zeroed_step(model, x, u, states)


def teacher_forced_mse(
    model: StackedModel,
    inputs: jax.Array,
    targets: jax.Array,
    *,
    branch_zeroed: bool = False,
) -> float:
    """Re-derives one-step teacher-forced MSE (identify.py's training
    objective) from a decode=True model stepping through real (x, u) ->
    x_next data one step at a time - used both as a sanity check against
    the checkpoint's own recorded teacher_mse (branch_zeroed=False should
    reproduce it) and as the branch-zeroing ablation's MSE metric
    (branch_zeroed=True)."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    step_fn = _jit_branch_zeroed_step if branch_zeroed else _jit_step

    sq_errs = []
    for t in range(inputs.shape[0]):
        x_t = inputs[t, :d_x]
        u_t = inputs[t, d_x:]
        x_next, states = step_fn(model, x_t, u_t, states)
        sq_errs.append(np.asarray(jnp.sum((x_next - targets[t]) ** 2)))
    return float(np.mean(sq_errs) / d_x)


def trajectory_contamination(
    model: StackedModel, inputs: jax.Array, targets: jax.Array
) -> list[dict]:
    """Walks the real (teacher-forced) training trajectory, evolving the S4
    hidden state on the REAL data (not the model's own predictions - same
    convention as teacher_forced_mse/identify.py's training loss), and at
    each step records the full/constant/branch Jacobian decomposition.
    Returns one dict per timestep with Frobenius norms and the
    contamination ratio ||J_branch|| / ||J_constant||."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    J_const = constant_term(model)
    const_norm = float(np.linalg.norm(J_const))

    rows = []
    for t in range(inputs.shape[0]):
        x_t = inputs[t, :d_x]
        u_t = inputs[t, d_x:]
        J_full = full_jacobian(model, x_t, u_t, states)
        J_branch = J_full - J_const
        rows.append(
            {
                "t": t,
                "x_norm": float(jnp.linalg.norm(x_t)),
                "J_full_norm": float(np.linalg.norm(J_full)),
                "J_branch_norm": float(np.linalg.norm(J_branch)),
                "contamination_ratio": float(np.linalg.norm(J_branch) / const_norm) if const_norm > 0 else float("nan"),
            }
        )
        _, states = _jit_step(model, x_t, u_t, states)  # teacher-forced: real x_t, not model's own prediction

    return rows


def origin_sweep_decomposed(
    model: StackedModel,
    direction: np.ndarray,
    t_values: np.ndarray,
    u: jax.Array,
) -> list[dict]:
    """dF/dx at x = t * direction for each t in t_values (S4 state fixed
    at zero_states, u fixed) - decomposed into the constant skip-path
    columns (W_dec @ W_enc restricted to the x-block of z) and the
    remaining branch part. Reports the full-J - branch-J agreement as an
    exact-decomposition sanity check (should be ~0 to float precision;
    any nonzero residual would indicate a bug in this module, not a
    property of the model)."""
    d_x = direction.shape[0]
    states = zero_states(model, dtype=jnp.complex128)
    W_enc, _, W_dec, _ = encoder_decoder_matrices(model)
    C_x = W_dec @ W_enc[:, :d_x]  # (d_output, d_x): x-columns of the constant term

    rows = []
    for t in t_values:
        x = jnp.asarray(t * direction)
        J_x = np.asarray(_jit_dfdx(model, x, u, states))
        J_x_branch = J_x - C_x
        rows.append(
            {
                "t": float(t),
                "J_x_full_norm": float(np.linalg.norm(J_x)),
                "J_x_constant_norm": float(np.linalg.norm(C_x)),
                "J_x_branch_norm": float(np.linalg.norm(J_x_branch)),
                "decomposition_residual": float(np.linalg.norm(J_x - (C_x + J_x_branch))),
            }
        )
    return rows
