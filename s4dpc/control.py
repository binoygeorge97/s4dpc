"""GRU controller + DPC loss (CLAUDE.md sec 2: repo layout).

Reference templates provided by the user, not ported verbatim:
  - dpc_example: GRU architecture, quadratic DPC cost, curriculum-horizon
    training, true-plant transfer test. Its `StandaloneGRUController`
    computes `max_action` but never applies it ("unbounded (matches your
    working setup)"); ours DOES bound the action (tanh * max_action) -
    that is the one deliberate architectural change CLAUDE.md calls for
    ("action IS bounded"). Its `LinearDynamicsModel` (LayerNorm-based
    system id) is NOT reused here either - see
    s4dpc.identify.fit_least_squares's docstring and docs/DECISIONS.md.
  - training_code: optimizer/schedule conventions (adamw + cosine
    one-cycle), reused for `make_controller_optimizer`.

Two rollout paths, both unrolling the SAME controller/loss:
  - rollout_linear: through a raw (A, B) - the TRUE oracle plant, or an
    M0/M1 linear surrogate.
  - rollout_learned: through a trained StackedModel surrogate (M3/M4/M5/
    M6/M6_fix), built with decode=True and stepped one input at a time -
    training identifies in conv mode (decode=False, CLAUDE.md/
    tests/test_decode_construction_parity.py), control deploys in step
    mode, same params. Model input at each step is concat([x, u]) (state
    THEN control - s4dpc.data's convention, D_INPUT = state_dim +
    control_dim), matching what identify.py trained on.

Model params are split ONCE outside the step loop (nnx.split) and closed
over inside jax.vmap - shared/broadcast across the batch of independent
trajectories, while per-trajectory S4 recurrent state and controller
hidden state vary via vmap's batch axis. This is the same
"static graph + vmapped state" shape as blocks.py's channel-vmap and
identify.py's ensemble-vmap, applied here to batch-of-trajectories
instead of batch-of-channels/batch-of-ensemble-members.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from scipy.linalg import solve_discrete_are

from s4dpc.model import StackedModel


def _target_param_dtype() -> jnp.dtype:
    """Follows the global jax_enable_x64 flag, same reasoning as
    s4dpc/identify.py's version (duplicated, not imported - both are
    permanent library modules, this is a two-line helper, and importing
    a private underscore-prefixed name across modules is worse than a
    small duplication): one source of truth is the global flag itself,
    set once by the entry point before any JAX op."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _cast_params(module: nnx.Module, dtype: jnp.dtype) -> None:
    """nnx.Linear defaults param_dtype=float32 regardless of the global
    x64 flag (docs/DECISIONS.md) - BoundedGRUController is built entirely
    from nnx.Linear, so without this it would silently train at float32
    precision even with jax_enable_x64=True. No complex params here
    (unlike identify.py's S4LayerEnsemble case), so a flat real cast is
    sufficient."""
    if dtype == jnp.float32:
        return  # already the construction default; skip the no-op tree_map
    state = nnx.state(module, nnx.Param)
    state = jax.tree_util.tree_map(lambda x: x.astype(dtype), state)
    nnx.update(module, state)


class BoundedGRUController(nnx.Module):
    """GRU-cell controller: u = max_action * tanh(head(h)).

    Bounded, unlike dpc_example's StandaloneGRUController (which stores
    max_action but never applies it - dpc_example's own controls reached
    ~1e7 against APRBS training inputs in [-10,10], contaminating every
    result produced with it). Casts its own params to match the global
    jax_enable_x64 flag after construction (see _cast_params) - this is
    NOT optional the way it might look, since plain nnx.Linear silently
    stays float32 otherwise."""

    def __init__(self, d_x: int, hidden_dim: int, d_u: int, max_action: float, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.W_z = nnx.Linear(d_x + hidden_dim, hidden_dim, rngs=rngs)
        self.W_r = nnx.Linear(d_x + hidden_dim, hidden_dim, rngs=rngs)
        self.W_h = nnx.Linear(d_x + hidden_dim, hidden_dim, rngs=rngs)
        self.head = nnx.Linear(hidden_dim, d_u, rngs=rngs)
        _cast_params(self, _target_param_dtype())

    def __call__(self, h_prev: jax.Array, x_curr: jax.Array) -> tuple[jax.Array, jax.Array]:
        hx = jnp.concatenate([x_curr, h_prev], axis=-1)
        z = jax.nn.sigmoid(self.W_z(hx))
        r = jax.nn.sigmoid(self.W_r(hx))
        rhx = jnp.concatenate([x_curr, r * h_prev], axis=-1)
        h_tilde = jnp.tanh(self.W_h(rhx))
        h_new = (1 - z) * h_prev + z * h_tilde
        u = self.max_action * jnp.tanh(self.head(h_new))
        return h_new, u

    def init_hidden(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_dim))


def quadratic_stage_cost(x: jax.Array, u: jax.Array, Q_x: float, R_u: float) -> jax.Array:
    return jnp.sum((x**2) * Q_x, axis=-1) + jnp.sum((u**2) * R_u, axis=-1)


def rollout_linear(
    controller: BoundedGRUController,
    x0: jax.Array,
    A: jax.Array,
    B: jax.Array,
    Q_x: float,
    R_u: float,
    Q_f: float,
    horizon_N: int,
) -> jax.Array:
    """Unroll through a raw linear (A, B) plant - the TRUE system, or a
    least-squares-identified (M0/M1) linear surrogate. x0: (batch, d_x)."""
    x = x0
    h = controller.init_hidden(x.shape[0])
    trajectory_loss = 0.0
    for _ in range(horizon_N):
        h, u = controller(h, x)
        x_next = x @ A.T + u @ B.T
        trajectory_loss += jnp.mean(quadratic_stage_cost(x, u, Q_x, R_u))
        x = x_next
    terminal_cost = jnp.mean(jnp.sum((x**2) * Q_f, axis=-1))
    return (trajectory_loss + terminal_cost) / horizon_N


def init_batched_state(model: StackedModel, batch_size: int) -> list[jax.Array]:
    """Per-trajectory S4 recurrent state, batch axis prepended.

    Not redundant with model.init_state() even now that init_state()
    itself follows jax_enable_x64 (s4dpc/model.py): init_state()'s shape
    is (d_model, N) - no batch_size parameter exists or is planned there,
    since the unbatched shape is also what identify.py's conv-mode
    training and diagnostics.zero_states need. This function's reason to
    exist has narrowed to exactly the batch-axis prepending
    ((batch_size, d_model, N)) that rollout_learned's jax.vmap over a
    batch of independent trajectories requires - the dtype-matching half
    of its original docstring is now redundant with init_state()'s own
    fix, kept here only because building the batched zeros directly is
    simpler than calling init_state() and re-broadcasting."""
    complex_dtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    n = model.block_config.N
    shape = (batch_size, model.block_config.d_model, n)
    return [jnp.zeros(shape, dtype=complex_dtype) for _ in range(model.n_layers)]


def rollout_learned(
    controller: BoundedGRUController,
    model_graphdef,
    model_params,
    x0: jax.Array,
    states: list[jax.Array],
    Q_x: float,
    R_u: float,
    Q_f: float,
    horizon_N: int,
) -> tuple[jax.Array, list[jax.Array]]:
    """Unroll through a trained StackedModel surrogate (decode=True,
    stepped). model_graphdef/model_params: from nnx.split(model) on a
    StackedModel built with decode=True - params are shared (broadcast,
    closed over) across the batch; only `states` and the per-step model
    input vary per trajectory via jax.vmap. x0: (batch, d_x).

    `states` MUST be real per-trajectory state (e.g. from
    init_batched_state), never None: StackedModel.__call__'s own
    `states=None` default silently falls back to a FRESH init_state()
    every call - inert in decode=False/conv mode (the state is threaded
    through unchanged there regardless), but fatal here, since it would
    reset the S4 recurrence to zero every single step instead of
    carrying it forward, and jax.vmap over a `None` in place of a real
    batched array fails outright rather than silently doing the wrong
    thing. Asserted explicitly rather than left as a silent footgun."""
    assert states is not None, (
        "rollout_learned: states=None would silently reset the S4 recurrence to zero "
        "every step (decode=True has no safe default, unlike decode=False's inert one) "
        "- pass init_batched_state(model, batch_size) or the evolving state from a prior call"
    )

    def apply_one(state, model_in):
        m = nnx.merge(model_graphdef, model_params)
        return m(model_in, state)

    x = x0
    h = controller.init_hidden(x.shape[0])
    trajectory_loss = 0.0
    for _ in range(horizon_N):
        h, u = controller(h, x)
        model_in = jnp.concatenate([x, u], axis=-1)
        x_next, states = jax.vmap(apply_one, in_axes=(0, 0))(states, model_in)
        trajectory_loss += jnp.mean(quadratic_stage_cost(x, u, Q_x, R_u))
        x = x_next
    terminal_cost = jnp.mean(jnp.sum((x**2) * Q_f, axis=-1))
    return (trajectory_loss + terminal_cost) / horizon_N, states


def make_controller_optimizer(
    controller: BoundedGRUController, base_lr: float, weight_decay: float, total_steps: int
) -> nnx.Optimizer:
    """adamw + cosine one-cycle, per training_code's create_optimizer
    convention (total_steps<=0 falls back to a constant schedule)."""
    if total_steps > 0:
        schedule = optax.cosine_onecycle_schedule(peak_value=base_lr, transition_steps=total_steps, pct_start=0.1)
    else:
        schedule = optax.constant_schedule(base_lr)
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    return nnx.Optimizer(controller, tx, wrt=nnx.Param)


def solve_dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Oracle discrete-time LQR gain K (u = -Kx) via the discrete
    algebraic Riccati equation. CLAUDE.md sec 1: ground-truth (A_d, B_d)
    make an oracle LQR controller directly computable - the upper-bound
    baseline GRU controllers are compared against."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def rollout_lqr_true(
    A: np.ndarray, B: np.ndarray, K: np.ndarray, x0: np.ndarray, horizon_N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop rollout of the oracle LQR gain on the TRUE plant.
    x0: (batch, d_x). Returns (x_hist: (N+1,batch,d_x), u_hist: (N,batch,d_u))."""
    x = x0
    xs = [x]
    us = []
    for _ in range(horizon_N):
        u = -x @ K.T
        x = x @ A.T + u @ B.T
        xs.append(x)
        us.append(u)
    return np.stack(xs), np.stack(us)


def true_quadratic_cost(x_hist: np.ndarray, u_hist: np.ndarray, Q_x: float, R_u: float, Q_f: float) -> float:
    """Same cost convention as quadratic_stage_cost/rollout_linear's
    terminal term, applied to a fixed (numpy) rollout - dpc_example's
    true_lqr_cost, generalized to scalar Q_x/R_u/Q_f weights."""
    stage = np.sum(x_hist[:-1] ** 2, axis=-1) * Q_x + np.sum(u_hist**2, axis=-1) * R_u
    term = np.sum(x_hist[-1] ** 2, axis=-1) * Q_f
    return float((stage.sum(axis=0) + term).mean() / x_hist.shape[0])


def evaluate_controller_on_true(
    controller: BoundedGRUController,
    A: np.ndarray,
    B: np.ndarray,
    x0: jax.Array,
    horizon_N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The "honest transfer test" (CLAUDE.md sec 1): roll a TRAINED GRU
    controller (trained through the true plant, an M0/M1 linear
    surrogate, or a learned M3/M6 surrogate - doesn't matter which)
    forward on the TRUE (A, B) plant from a batch of initial states,
    open-loop with respect to training (the controller only ever sees
    x_curr, never A/B directly). x0: (batch, d_x). Returns (x_hist:
    (N+1,batch,d_x), u_hist: (N,batch,d_u)) - same shape convention as
    rollout_lqr_true, so both feed the same true_quadratic_cost - this is
    dpc_example's rollout_true, generalized to any BoundedGRUController
    regardless of what it was trained through."""
    A_j, B_j = jnp.asarray(A, dtype=jnp.float64), jnp.asarray(B, dtype=jnp.float64)
    h = controller.init_hidden(x0.shape[0])
    x = x0
    xs, us = [x], []
    for _ in range(horizon_N):
        h, u = controller(h, x)
        x = x @ A_j.T + u @ B_j.T
        xs.append(x)
        us.append(u)
    x_hist = np.asarray(jnp.stack(xs))
    u_hist = np.asarray(jnp.stack(us))
    return x_hist, u_hist
