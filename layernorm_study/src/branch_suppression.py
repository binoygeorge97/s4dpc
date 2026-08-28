"""Round 1.5, Task 1/2: does the optimizer make arm_2's branch
homogeneous, or does it just suppress it?

LayerNorm is homogeneous of degree ZERO by construction: no setting of
gamma/beta makes LN(c*z) = c*LN(z). A prenorm block containing a real
LayerNorm therefore CANNOT represent an exactly homogeneous (degree-1)
map unless the branch's contribution is driven toward zero - it cannot
be "trained into" linearity, only trained away. This module measures
which one round 1's arm_2 actually did.

Reuses layernorm_study.src.jacobian_decomposition's constant_term/
full_jacobian (Experiment 1 tooling - dimension-agnostic, so it applies
unchanged to the scalar plant's d_input=2/d_output=1 models here) rather
than re-deriving the same decomposition a second time.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from s4dpc.diagnostics import step, zero_states
from s4dpc.model import StackedModel

from layernorm_study.src.arms import ArmSpec, build_model
from layernorm_study.src.jacobian_decomposition import constant_term, full_jacobian


def gamma_beta_norms(model: StackedModel) -> tuple[float, float]:
    """||gamma||, ||beta|| of the block's LayerNorm (whichever one is
    primary - `config.norm == 'layer'`). Raises if this arm has none."""
    if model.n_layers == 0:
        raise ValueError("this arm has no block/LayerNorm (n_layers == 0)")
    norm = model.layers[0].norm
    if norm is None:
        raise ValueError("this arm's block has no LayerNorm (config.norm != 'layer')")
    return float(jnp.linalg.norm(norm.scale.value)), float(jnp.linalg.norm(norm.bias.value))


def skip_only_output(model: StackedModel, x: jax.Array, u: jax.Array) -> jax.Array:
    """decoder(encoder(z)) - the block's residual/skip contribution
    alone, with the branch's contribution excluded entirely (NOT the
    same as branch_zeroed_step in jacobian_decomposition.py, which also
    returns S4 state; this returns just the output vector)."""
    z = jnp.concatenate([x, u])
    return model.decoder(model.encoder(z[jnp.newaxis, :]))[0]


def branch_to_skip_ratio_along_trajectory(
    model: StackedModel, inputs: jax.Array, targets: jax.Array
) -> list[dict]:
    """||branch_output|| / ||skip_output|| = ||W_dec @ G(LN(.))|| /
    ||W_dec @ W_enc @ z|| at every real trajectory point (S4 state
    evolved teacher-forced on the real data, matching Experiment 1's
    trajectory_contamination convention)."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    rows = []
    for t in range(inputs.shape[0]):
        x_t, u_t = inputs[t, :d_x], inputs[t, d_x:]
        skip = np.asarray(skip_only_output(model, x_t, u_t))
        full, states = step(model, x_t, u_t, states)
        branch = np.asarray(full) - skip
        skip_norm = float(np.linalg.norm(skip))
        rows.append(
            {
                "t": t,
                "skip_norm": skip_norm,
                "branch_norm": float(np.linalg.norm(branch)),
                "ratio": float(np.linalg.norm(branch) / skip_norm) if skip_norm > 0 else float("nan"),
            }
        )
    return rows


def jacobian_decomposition_along_trajectory(
    model: StackedModel, inputs: jax.Array, targets: jax.Array, AB_true: np.ndarray
) -> list[dict]:
    """J(z) = C + R(z), C = W_dec@W_enc constant. Reports ||C - AB_true||
    (should -> 0 if the branch is suppressed, since then J -> C exactly)
    and ||R(z)|| (should -> 0 likewise) at every real trajectory point."""
    d_x = targets.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    C = constant_term(model)
    c_err = float(np.linalg.norm(C - AB_true) / np.linalg.norm(AB_true))
    rows = []
    for t in range(inputs.shape[0]):
        x_t, u_t = inputs[t, :d_x], inputs[t, d_x:]
        J = full_jacobian(model, x_t, u_t, states)
        R = J - C
        rows.append({"t": t, "C_err_rel": c_err, "R_norm": float(np.linalg.norm(R))})
        _, states = step(model, x_t, u_t, states)
    return rows


def freeze_layernorm_affine(model: StackedModel) -> None:
    """In-place: converts the block's LayerNorm scale/bias from nnx.Param
    to plain nnx.Variable, at whatever value they currently hold (called
    right after construction, before training, so this freezes them at
    flax's own init - gamma=1, beta=0, verified directly). nnx.Optimizer(
    ..., wrt=nnx.Param) then skips them entirely (they're no longer
    nnx.Param instances) - same "Variable, not Param" pattern
    s4dpc/blocks.py's StaticNorm already uses for its own frozen mu/sigma,
    applied here via nnx's public attribute-reassignment mechanism rather
    than a parent-repo code change (LayerNorm's use_scale/use_bias flags
    would need s4dpc/blocks.py to pass them through, which is a much
    smaller change - IF this experiment's result recommends actually
    disabling gamma/beta permanently, that flag is the follow-up)."""
    norm = model.layers[0].norm
    if norm is None:
        raise ValueError("this arm's block has no LayerNorm (config.norm != 'layer')")
    norm.scale = nnx.Variable(norm.scale.value)
    norm.bias = nnx.Variable(norm.bias.value)


def train_arm_frozen_ln(
    arm: ArmSpec,
    inputs: jax.Array,
    targets: jax.Array,
    *,
    d_model: int,
    N: int,
    l_max: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.0,
) -> tuple[nnx.State, float]:
    """Same as arms.train_arm, except gamma/beta are frozen at init
    (freeze_layernorm_affine, called right after construction, before
    the optimizer is built) - LN's SCALING is still fully active every
    forward pass (1/sigma, mean-subtraction), the optimizer just cannot
    shrink gamma or shift beta to suppress its effect. The decisive test
    for whether round 1's "trainable away" reading of arm_2 was actually
    branch suppression: if error jumps back to percent-level here, yes;
    if it stays near machine precision, no - something else let arm_2
    train to near-perfect fidelity despite LN's active nonlinearity."""
    key = jax.random.PRNGKey(seed)
    model = build_model(
        arm, d_model=d_model, N=N, l_max=l_max,
        d_input=inputs.shape[-1], d_output=targets.shape[-1], decode=False, key=key,
    )
    freeze_layernorm_affine(model)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)
    states = model.init_state(N=N)

    def loss_fn(m):
        pred, _ = m(inputs, states)
        return jnp.mean((pred - targets) ** 2)

    @nnx.jit
    def train_step(m, opt):
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    for _ in range(epochs):
        train_step(model, optimizer)

    final_mse = float(loss_fn(model))
    return nnx.state(model, nnx.Param), final_mse
