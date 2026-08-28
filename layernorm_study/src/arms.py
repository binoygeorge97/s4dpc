"""Experiment 2's complexity ladder: one BlockConfig per arm, holding
data/seed/d_model/N/l_max/optimizer/epochs fixed and varying ONLY the
block's flags - so any difference between arms is attributable to the
flag difference alone, not a confound.

arm_0 is special: n_layers=0 means StackedModel has no ConfigurableBlock
at all (the layer list is empty), so F(z) = decoder(encoder(z)) exactly -
an exactly affine map, with no "branch" to zero out. This needs no new
parent-repo code: n_layers=0 is already handled generically by
StackedModel's existing `for layer, state in zip(self.layers, ...)` loop
(vacuous when self.layers == []).

arm_1/2/3's `memoryless=True` and arm_7's `postnorm_also=True` use the
two new BlockConfig flags added to s4dpc/blocks.py for this project
(both default off, verified not to affect M3-M6/M6_fix - see the git
history for the parity-test verification of each).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from s4dpc.blocks import BlockConfig
from s4dpc.model import StackedModel


@dataclass(frozen=True)
class ArmSpec:
    name: str
    description: str
    n_layers: int
    block_kwargs: dict = field(default_factory=dict)


ARMS: dict[str, ArmSpec] = {
    "arm_0": ArmSpec(
        "arm_0", "zero branch (skip only) - exactly linear, no memory",
        n_layers=0, block_kwargs={},
    ),
    "arm_1": ArmSpec(
        "arm_1", "linear branch, no LN, no memory - exactly linear",
        n_layers=1, block_kwargs=dict(norm="none", activation="none", glu=False, memoryless=True),
    ),
    "arm_2": ArmSpec(
        "arm_2", "linear + LayerNorm, no memory - KEY ARM",
        n_layers=1, block_kwargs=dict(norm="layer", activation="none", glu=False, memoryless=True, prenorm=True),
    ),
    "arm_3": ArmSpec(
        "arm_3", "linear + GELU + GLU, no LN, no memory",
        n_layers=1, block_kwargs=dict(norm="none", activation="gelu", glu=True, memoryless=True),
    ),
    "arm_4": ArmSpec(
        "arm_4", "S4, no LN - nonlinear (via memory), no LayerNorm",
        n_layers=1, block_kwargs=dict(norm="none", activation="none", glu=False, memoryless=False),
    ),
    "arm_5": ArmSpec(
        "arm_5", "S4 + LN prenorm - full model",
        n_layers=1, block_kwargs=dict(norm="layer", activation="gelu", glu=True, memoryless=False, prenorm=True),
    ),
    "arm_6": ArmSpec(
        "arm_6", "S4 + LN postnorm - full model, placement swapped",
        n_layers=1, block_kwargs=dict(norm="layer", activation="gelu", glu=True, memoryless=False, prenorm=False),
    ),
    "arm_7": ArmSpec(
        "arm_7", "S4 + LN prenorm AND postnorm combined (user-requested)",
        n_layers=1,
        block_kwargs=dict(
            norm="layer", activation="gelu", glu=True, memoryless=False, prenorm=True, postnorm_also=True,
        ),
    ),
}


def build_model(
    arm: ArmSpec,
    *,
    d_model: int,
    N: int,
    l_max: int,
    d_input: int,
    d_output: int,
    decode: bool,
    key: jax.Array,
) -> StackedModel:
    block_config = BlockConfig(d_model=d_model, N=N, l_max=l_max, **arm.block_kwargs)
    return StackedModel(
        block_config=block_config,
        d_input=d_input,
        d_output=d_output,
        n_layers=arm.n_layers,
        decode=decode,
        rngs=nnx.Rngs(params=key),
    )


def train_arm(
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
    """Teacher-forced one-step MSE identification, mirroring
    s4dpc.identify._train_one's pattern (adamw, jitted per-step update,
    plain Python epoch loop) but for the scalar test plant's
    (d_input=2, d_output=1) shapes and arms.ARMS's block configs instead
    of s4dpc.blocks.VARIANTS. Returns (trained param state, final MSE)."""
    key = jax.random.PRNGKey(seed)
    model = build_model(
        arm, d_model=d_model, N=N, l_max=l_max,
        d_input=inputs.shape[-1], d_output=targets.shape[-1], decode=False, key=key,
    )
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


def train_arm_weighted(
    arm: ArmSpec,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    *,
    d_model: int,
    N: int,
    l_max: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.0,
) -> tuple[nnx.State, float]:
    """Same as train_arm, but the per-timestep loss is weighted:
    loss = sum_t weights[t]*(pred_t-target_t)^2 / sum_t weights[t].
    `weights` (L,), nonnegative, need not sum to 1 (normalized here).
    The SAME 100 real (inputs, targets) pairs are used in their ORIGINAL
    temporal order and positions (S4's conv-mode kernel is position-
    dependent, so reordering/resampling into a shorter/longer/shuffled
    sequence would corrupt it) - only which timesteps' errors matter for
    the gradient changes, for round 2's Task A10 (does arm_5's
    seed-to-seed spread track conditioning, with persistence-of-
    excitation order held fixed rather than confounded with it, as
    round 2's segment_length experiment was)."""
    key = jax.random.PRNGKey(seed)
    model = build_model(
        arm, d_model=d_model, N=N, l_max=l_max,
        d_input=inputs.shape[-1], d_output=targets.shape[-1], decode=False, key=key,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)
    states = model.init_state(N=N)
    w_norm = weights / jnp.sum(weights)

    def loss_fn(m):
        pred, _ = m(inputs, states)
        sq_err = jnp.sum((pred - targets) ** 2, axis=-1)  # (L,)
        return jnp.sum(w_norm * sq_err)

    @nnx.jit
    def train_step(m, opt):
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    for _ in range(epochs):
        train_step(model, optimizer)

    final_mse = float(loss_fn(model))
    return nnx.state(model, nnx.Param), final_mse


def load_arm_model(
    arm: ArmSpec,
    param_state: nnx.State,
    *,
    d_model: int,
    N: int,
    l_max: int,
    d_input: int,
    d_output: int,
    seed: int,
    decode: bool = True,
) -> StackedModel:
    """Builds a fresh (decode=True by default, for step-mode diagnostics)
    model of the same architecture and loads `param_state` (from
    train_arm, a decode=False model's params - identical shapes, decode
    only affects S4LayerEnsemble's OWN forward-mode branch, not its
    param shapes) into it."""
    key = jax.random.PRNGKey(seed)
    model = build_model(
        arm, d_model=d_model, N=N, l_max=l_max, d_input=d_input, d_output=d_output, decode=decode, key=key,
    )
    nnx.update(model, param_state)
    return model
