"""Diagnostics for Experiment 2's scalar test plant (x_next = 3x + u),
run against a decode=True model of any arm (layernorm_study.src.arms).

nnx.jit-wrapped once at module level (Experiment 1's
jacobian_decomposition.py lesson: unjitted per-checkpoint diagnostics
took ~15 minutes each on this machine; jitted and reused across calls,
~seconds) - see that module's longer comment for why this matters.
All Jacobians via jax.jacfwd, never finite-differenced.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.diagnostics import step, zero_states
from s4dpc.model import StackedModel

LAYERNORM_EPS = 1e-6  # flax.nnx.LayerNorm's own default - not overridden anywhere in this project


@nnx.jit
def _jit_step(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    return step(model, x, u, states)


@nnx.jit
def _jit_jxju(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    def f(z: jax.Array) -> jax.Array:
        x_next, _ = step(model, z[:1], z[1:], states)
        return x_next

    z = jnp.concatenate([x, u])
    J = jax.jacfwd(f)(z)  # (1, 2)
    return J[0, 0], J[0, 1]  # Jx, Ju (scalars, d_x=d_u=1 for this plant)


def teacher_forced_mse(model: StackedModel, inputs: jax.Array, targets: jax.Array) -> float:
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    sq_errs = []
    for t in range(inputs.shape[0]):
        x_t, u_t = inputs[t, :d_x], inputs[t, d_x:]
        x_next, states = _jit_step(model, x_t, u_t, states)
        sq_errs.append(np.asarray(jnp.sum((x_next - targets[t]) ** 2)))
    return float(np.mean(sq_errs))


def free_run_rmse(model: StackedModel, inputs: jax.Array, targets: jax.Array, k_stab: float) -> float:
    """Recursive (closed-loop) rollout: re-closes the SAME stabilizing
    feedback used to generate the data (u_hat_t = k_stab*x_hat_t + a_t,
    reusing the exact dither a_t = u_t - k_stab*x_t recovered from the
    recorded data) around the model's OWN x prediction, instead of
    teacher-forcing with the real x_t at every step.

    Deliberately NOT "replay the recorded u_t open-loop while feeding the
    model's own x forward": the true plant is unstable in open loop
    (rho=3), so ANY autoregressive rollout under an open-loop-recorded u
    sequence diverges as ~3^t regardless of model quality (verified
    directly - this was tried first and gave free_run_rmse ~3e39,
    reproducing the exact 3^100 blowup the task warns about for raw
    open-loop excitation, not a meaningful fidelity signal). Re-closing
    the loop around x_hat instead makes this a bounded diagnostic of the
    SURROGATE's fidelity specifically, matching how the data itself was
    generated."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    a_t = inputs[:, d_x:] - k_stab * inputs[:, :d_x]  # recover the dither exactly
    x_hat = inputs[0, :d_x]
    sq_errs = []
    for t in range(inputs.shape[0]):
        u_hat_t = k_stab * x_hat + a_t[t]
        x_hat, states = _jit_step(model, x_hat, u_hat_t, states)
        sq_errs.append(np.asarray(jnp.sum((x_hat - targets[t]) ** 2)))
    return float(np.sqrt(np.mean(sq_errs)))


def jx_ju_along_trajectory(model: StackedModel, inputs: jax.Array, targets: jax.Array) -> list[dict]:
    """Jx, Ju at every real (teacher-forced) trajectory point, S4 state
    evolved on the real data - same convention as Experiment 1's
    trajectory_contamination."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    rows = []
    for t in range(inputs.shape[0]):
        x_t, u_t = inputs[t, :d_x], inputs[t, d_x:]
        jx, ju = _jit_jxju(model, x_t, u_t, states)
        rows.append({"t": t, "x": float(x_t[0]), "Jx": float(jx), "Ju": float(ju)})
        _, states = _jit_step(model, x_t, u_t, states)
    return rows


def equilibrium_drift(model: StackedModel) -> float:
    """F(0, 0, s=0) - the true plant satisfies this exactly (LTI)."""
    states = zero_states(model, dtype=jnp.complex128)
    x0, u0 = jnp.zeros((1,)), jnp.zeros((1,))
    x_next, _ = step(model, x0, u0, states)
    return float(x_next[0])


def homogeneity_sweep(model: StackedModel, z0: np.ndarray, c_values: np.ndarray) -> list[dict]:
    """(F(c*z0) - F(0,0)) / c for c log-spaced through zero. z0=(x0, u0)
    a fixed nonzero direction. True plant: this ratio is A*x0 + B*u0
    exactly, CONSTANT for every c - the cleanest single statement of the
    degree-1 (true plant) vs. degree-0 (LayerNorm) mismatch.

    Subtracts F(0,0) (equilibrium_drift) BEFORE dividing - caught
    directly, not assumed: an earlier version divided raw F(c*z0) by c
    with no drift correction, and for every trained arm except arm_0/
    arm_1 (which converge to near-exactly-zero drift) this produced a
    dip/spike near c=0 that persisted even for arm_4 - an EXACTLY LTI
    model (autodiff-verified: its Jx(c) is flat to machine precision at
    every c) - proving that shape was the additive drift term divided by
    a vanishing c (F(c*z0)/c = J@z0 + drift/c -> diverges as c->0 for any
    drift != 0), not a Jacobian/curvature effect at all. This is the
    same class of drift-vs-derivative confound the parent repo's
    CLAUDE.md documents under its bias-term-round corrections."""
    states = zero_states(model, dtype=jnp.complex128)
    drift = equilibrium_drift(model)
    rows = []
    for c in c_values:
        z = c * z0
        x_next, _ = step(model, jnp.asarray(z[:1]), jnp.asarray(z[1:]), states)
        rows.append({"c": float(c), "F_over_c": (float(x_next[0]) - drift) / float(c)})
    return rows


def jx_vs_c_sweep(model: StackedModel, direction: np.ndarray, c_values: np.ndarray, u: jax.Array | None = None) -> list[dict]:
    """||Jx(c*direction)|| vs c, log-log - predicted slope -1 pre-epsilon
    (LayerNorm's 1/sigma term), flat after (epsilon floor saturates)."""
    states = zero_states(model, dtype=jnp.complex128)
    u = jnp.zeros((1,)) if u is None else u
    rows = []
    for c in c_values:
        x = jnp.asarray(c * direction)
        jx, ju = _jit_jxju(model, x, u, states)
        rows.append({"c": float(c), "Jx": float(jx), "Ju": float(ju)})
    return rows


def directional_asymmetry(model: StackedModel, eps_values: np.ndarray, key: jax.Array, n_directions: int = 4) -> list[dict]:
    """||Jx(+eps) - Jx(-eps)|| vs eps, at `n_directions` random FIXED u
    values (the state itself is 1-D, so "direction" varies through the
    (x, u) pair's u-component rather than an x-direction, which for a
    scalar state has only the trivial +1/-1 choice). PREDICTION (task):
    this GROWS as eps -> 0 for LayerNorm'd arms - the opposite of what a
    smooth-nonlinearity (GELU/GLU) explanation would predict, which is
    the discriminating test between the two mechanisms."""
    states = zero_states(model, dtype=jnp.complex128)
    u_dirs = jax.random.uniform(key, (n_directions,), minval=-1.0, maxval=1.0)
    rows = []
    for eps in eps_values:
        diffs = []
        for u_val in u_dirs:
            u = jnp.array([float(u_val)])
            jx_pos, _ = _jit_jxju(model, jnp.array([float(eps)]), u, states)
            jx_neg, _ = _jit_jxju(model, jnp.array([float(-eps)]), u, states)
            diffs.append(abs(float(jx_pos) - float(jx_neg)))
        rows.append({"eps": float(eps), "mean_abs_diff": float(np.mean(diffs)), "max_abs_diff": float(np.max(diffs))})
    return rows


def prenorm_sigma(model: StackedModel, z: np.ndarray) -> float | None:
    """The actual LayerNorm sigma = sqrt(var(pre_ln) + eps) at a given
    z=(x,u), recomputed externally (nnx.LayerNorm's __call__ doesn't
    expose it) from the SAME pre-LN vector (encoder(z)) the block's own
    prenorm branch normalizes. Returns None for any arm without a
    PRENORM LayerNorm (n_layers==0, norm!='layer', or prenorm=False) -
    deliberately scoped to the prenorm case only (closest to the
    "kink at the physical input" story); a postnorm/postnorm_also sigma
    would need the post-residual pre-LN value, which needs re-deriving
    part of the block's forward pass and is left out of this round."""
    if model.n_layers == 0:
        return None
    cfg = model.layers[0].config
    if cfg.norm != "layer" or not cfg.prenorm:
        return None
    pre_ln = np.asarray(model.encoder(jnp.asarray(z)[jnp.newaxis, :]))[0]
    return float(np.sqrt(np.var(pre_ln) + LAYERNORM_EPS))
