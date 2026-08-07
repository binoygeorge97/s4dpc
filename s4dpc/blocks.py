"""Configurable sequence block, consuming s4-nnx v0.2.0 (never legacy).

The channel-vmap pattern (nnx.split / nnx.merge / jax.vmap with
in_axes=(0,1,0), out_axes=(1,0)) is lifted from legacy's
SequenceBlockNNX.__call__: s4-nnx's S4LayerEnsemble holds one S4 SSM per
d_model channel in a single NNX module (params stacked on axis 0), so
running it per-channel means splitting into (static graph, per-channel
param pytree), vmapping a single-channel call over that param axis plus
the input's channel axis, then remerging.

M6 (norm="layer", activation="gelu", glu=True, prenorm=True,
residual=True) is built to be architecturally identical to legacy's
SequenceBlockNNX with its own defaults - see tests/test_parity.py for the
empirical bit-exact check against the legacy checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from s4_nnx import S4LayerEnsemble


@dataclass(frozen=True)
class BlockConfig:
    d_model: int
    N: int
    l_max: int
    norm: str = "layer"  # "layer" | "static" | "none"
    activation: str = "gelu"  # "gelu" | "none"
    glu: bool = True
    prenorm: bool = True
    residual: bool = True


VARIANTS = {
    "M3": dict(norm="none", activation="none", glu=False),
    "M4": dict(norm="none", activation="gelu", glu=True),
    "M5": dict(norm="layer", activation="none", glu=False),
    "M6": dict(norm="layer", activation="gelu", glu=True),
    "M6_fix": dict(norm="static", activation="gelu", glu=True),
}


class StaticNorm(nnx.Module):
    """Fixed (x - mu) / sigma, computed once and frozen - NOT LayerNorm.

    LayerNorm is degree-0 homogeneous (LN(cz) = LN(z) for any scalar c):
    it divides by a per-instance standard deviation computed from the
    same input it's normalizing, so it cannot represent a linear map no
    matter what its (trainable) scale/bias become. Static standardization
    instead uses FIXED mu/sigma (constants w.r.t. the current input), so
    y = (x - mu) / sigma is affine in x - it preserves the underlying
    linear dynamics' Jacobian structure, which LayerNorm provably cannot.

    mu/sigma are nnx.Variable, not nnx.Param: `nnx.Optimizer(..., wrt=
    nnx.Param)` never touches them, so calibrate() really is a one-time,
    frozen-afterward operation, not something training can undo. Default
    to identity (mu=0, sigma=1) until calibrate() is called explicitly.
    """

    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        del rngs  # unused: mu/sigma start at fixed values, not randomly
        self.mu = nnx.Variable(jnp.zeros((d_model,)))
        self.sigma = nnx.Variable(jnp.ones((d_model,)))

    def calibrate(self, x: jax.Array) -> None:
        """x: (n_samples, d_model) - e.g. a batch of training-set
        activations at this point in the network. Call once before
        training."""
        self.mu.value = jnp.mean(x, axis=0)
        self.sigma.value = jnp.std(x, axis=0) + 1e-6

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.mu.value) / self.sigma.value


class ConfigurableBlock(nnx.Module):
    """One S4 sequence block: per-channel S4Layer (vmapped) + configurable
    norm/activation/glu/prenorm/residual. decode is fixed at construction
    (s4-nnx's S4LayerEnsemble requirement), so a decode=True and
    decode=False instance built from the same seed are needed for
    conv-mode vs. step-mode use - same as legacy and the s4-nnx port."""

    def __init__(self, config: BlockConfig, *, decode: bool = False, rngs: nnx.Rngs):
        self.config = config
        self.decode = decode

        self.seq = S4LayerEnsemble(
            N=config.N,
            l_max=config.l_max,
            D_MODEL=config.d_model,
            decode=decode,
            rngs=rngs,
        )

        keys = jax.random.split(rngs.params(), 3)
        if config.norm == "layer":
            self.norm = nnx.LayerNorm(config.d_model, rngs=nnx.Rngs(params=keys[0]))
        elif config.norm == "static":
            self.norm = StaticNorm(config.d_model, rngs=nnx.Rngs(params=keys[0]))
        elif config.norm == "none":
            self.norm = None
        else:
            raise ValueError(f"unknown norm {config.norm!r}")

        self.out = nnx.Linear(config.d_model, config.d_model, rngs=nnx.Rngs(params=keys[1]))
        if config.glu:
            self.out2 = nnx.Linear(config.d_model, config.d_model, rngs=nnx.Rngs(params=keys[2]))

    def __call__(self, x: jax.Array, s4_state: jax.Array) -> tuple[jax.Array, jax.Array]:
        skip = x

        if self.norm is not None and self.config.prenorm:
            x = self.norm(x)

        seq_graph, seq_params = nnx.split(self.seq)

        def run_one_channel(params_slice, u_slice, state_slice):
            single_channel_layer = nnx.merge(seq_graph, params_slice)
            return single_channel_layer(u_slice, state_slice)

        x, new_s4_state = jax.vmap(
            run_one_channel,
            in_axes=(0, 1, 0),
            out_axes=(1, 0),
        )(seq_params, x, s4_state)

        if self.config.activation == "gelu":
            x = nnx.gelu(x)
        elif self.config.activation == "none":
            pass
        else:
            raise ValueError(f"unknown activation {self.config.activation!r}")

        if self.config.glu:
            gate = jax.nn.sigmoid(self.out2(x))
            x = self.out(x) * gate
        else:
            x = self.out(x)

        if self.config.residual:
            x = skip + x

        if self.norm is not None and not self.config.prenorm:
            x = self.norm(x)

        return x, new_s4_state
