"""Control-relevant diagnostics for a trained identification model,
independent of one-step teacher-forced MSE (CLAUDE.md sec 1's central
claim under test).

All four diagnostics operate on an already-constructed `decode=True`
StackedModel with TRAINED params loaded in (identify.py trains
decode=False/conv; deploying one input at a time requires decode=True,
same params - see tests/test_control_decode_parity.py, and
s4dpc.control's rollout_learned, which this module's step function
mirrors). Building/loading that model is the caller's job, matching the
existing split (control.py doesn't train models either); diagnostics.py
only computes quantities given one.

    - equilibrium_drift: |F(0,0,s)| - does the model preserve the plant's
      true equilibrium at the physical origin?
    - markov_parameters: G_h = d x_{k+h} / d u_k, h=1..H, of the
      LINEARIZED AUGMENTED (physical state + S4 hidden state) system -
      realization-invariant, unlike a raw one-step d x_{k+1}/d x_k
      Jacobian (which only sees one step and ignores however the
      surrogate's own, generally non-A_d-shaped, internal realization
      carries information forward through the S4 hidden state). Computed
      by autodiff through an H-step UNROLLED free-running rollout w.r.t.
      u_0 - the chain rule through the unroll composes each step's local
      linearization exactly the way linearizing the augmented system by
      hand would, without needing to construct any augmented-state
      matrix explicitly.
    - local_linearity_defect: E|F(z+d) - F(z) - J(z)d| / |d| - how far a
      first-order Taylor expansion at z misses nearby points, normalized
      so it is comparable across perturbation scales.
    - jacobian_sweep: J(t*direction) for a sweep of t through 0 - the
      diagnostic behind the original kink figure (CLAUDE.md's suspected
      mechanism: LayerNorm is degree-0 homogeneous, so it cannot
      represent a linear map, and is expected to distort the Jacobian
      sharply near the regulation setpoint x=0).

See tools/validate_diagnostics.py for the ground-truth check these were
validated against before use on any trained surrogate: a model whose
S4Layer output is forced to exactly zero (D=0, C=0 - see docs/DECISIONS.md
Task 2 entry for the derivation) so its one-step map is EXACTLY
x_next = A_d @ x + B_d @ u; markov_parameters on that model must
reproduce A_d^(h-1) @ B_d to float64 precision, equilibrium_drift must be
exactly 0, and local_linearity_defect must be ~0 (the true system is
exactly linear).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from s4dpc.model import StackedModel


def zero_states(model: StackedModel, dtype: jnp.dtype = jnp.complex128) -> list[jax.Array]:
    """model.init_state() hardcodes complex64 (s4dpc/model.py) regardless
    of jax_enable_x64 - use this instead for float64 diagnostics work."""
    n = model.block_config.N
    shape = (model.block_config.d_model, n)
    return [jnp.zeros(shape, dtype=dtype) for _ in range(model.n_layers)]


def step(model: StackedModel, x: jax.Array, u: jax.Array, states: list[jax.Array]):
    """One augmented-system transition: (x, u, states) -> (x_next,
    states_next). x: (d_x,), u: (d_u,). Concatenation order matches
    s4dpc.data/identify.py's convention (state then control, D_INPUT =
    state_dim + control_dim) - the same input layout identify.py trained
    on and control.py's rollout_learned deploys."""
    model_in = jnp.concatenate([x, u], axis=-1)
    return model(model_in, states)


def equilibrium_drift(model: StackedModel, states: list[jax.Array]) -> jax.Array:
    """|F(0,0,s)|: the model's one-step prediction at the physical
    origin (x=0, u=0), given nominal S4 hidden state `states` (typically
    zero_states(model)). The TRUE plant satisfies F_true(0,0)=0 exactly
    (LTI: A_d@0 + B_d@0 = 0) for any case - this returns the model's raw
    deviation vector (d_x,), not yet reduced to a norm, so callers can
    inspect which state components drift."""
    d_x = model.d_output
    d_u = model.d_input - d_x
    x0 = jnp.zeros((d_x,))
    u0 = jnp.zeros((d_u,))
    x_next, _ = step(model, x0, u0, states)
    return x_next


def markov_parameters(model: StackedModel, states: list[jax.Array], H: int) -> jax.Array:
    """G_h = d x_{k+h} / d u_k for h=1..H, linearized around the
    equilibrium trajectory (x=0, u=0 at every step, S4 hidden state
    fixed at `states` for the FIRST step only - it evolves freely for
    steps 2..H exactly like a real rollout). An OPEN-LOOP, free-running
    multi-step unroll (x_{k+1} feeds back as the next step's state input,
    like control.py's rollout_learned) - NOT the teacher-forced one-step
    map identify.py trains on, where x_k is handed in explicitly at every
    step regardless of the model's own prior prediction.

    Forward-mode autodiff (jacfwd) w.r.t. u_0 only: cheap because d_u is
    small (3 for cases 1-7), regardless of how large H or the model is -
    cost scales with the INPUT dimension being differentiated, not H.

    Returns (H, d_x, d_u)."""
    d_x = model.d_output
    d_u = model.d_input - d_x

    def rollout(u0: jax.Array) -> jax.Array:
        x, s = step(model, jnp.zeros((d_x,)), u0, states)
        xs = [x]
        for _ in range(H - 1):
            x, s = step(model, x, jnp.zeros((d_u,)), s)
            xs.append(x)
        return jnp.stack(xs)  # (H, d_x)

    return jax.jacfwd(rollout)(jnp.zeros((d_u,)))  # (H, d_x, d_u)


def local_linearity_defect(
    model: StackedModel,
    states: list[jax.Array],
    z: jax.Array,
    u: jax.Array,
    key: jax.Array,
    n_samples: int = 64,
    delta_scale: float = 1e-3,
) -> jax.Array:
    """E|F(z+d) - F(z) - J(z)@d| / |d|, Monte Carlo over `n_samples`
    random small state-space perturbations d (control fixed at `u`, S4
    hidden state fixed at `states`) - normalized so the result is
    comparable across choices of delta_scale; large values indicate F is
    far from its own local (Gauss-Newton) linear approximation near z -
    the signature CLAUDE.md's LayerNorm/kink hypothesis predicts should
    be sharply elevated for LayerNorm'd variants near x=0 specifically."""

    def f(x: jax.Array) -> jax.Array:
        x_next, _ = step(model, x, u, states)
        return x_next

    f_z = f(z)
    j_z = jax.jacfwd(f)(z)  # (d_x, d_x)
    deltas = delta_scale * jax.random.normal(key, (n_samples, z.shape[0]))

    def defect_one(d: jax.Array) -> jax.Array:
        actual = f(z + d) - f_z
        predicted = j_z @ d
        return jnp.linalg.norm(actual - predicted) / jnp.linalg.norm(d)

    return jnp.mean(jax.vmap(defect_one)(deltas))


def jacobian_sweep(
    model: StackedModel,
    states: list[jax.Array],
    direction: jax.Array,
    t_values: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """J(t*direction) = d F/d x at x=t*direction, for each t in
    t_values (S4 hidden state fixed at `states`, control fixed at `u`) -
    the diagnostic behind the original kink figure: sweeping t through 0
    and inspecting how the Jacobian changes reveals whether F is smooth
    there (norm=none/static) or has a sharp transition (LayerNorm,
    degree-0 homogeneous, cannot represent a linear map).

    Returns (len(t_values), d_x, d_x)."""

    def f(x: jax.Array) -> jax.Array:
        x_next, _ = step(model, x, u, states)
        return x_next

    def jac_at(t: jax.Array) -> jax.Array:
        return jax.jacfwd(f)(t * direction)

    return jax.vmap(jac_at)(jnp.asarray(t_values))
