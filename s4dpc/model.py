"""Stacked model: encoder -> ConfigurableBlock x n_layers -> decoder.

RNG key-splitting order matches legacy.StackedModelRegression /
SequenceBlockNNX exactly (encoder key, decoder key, then one fresh
per-layer nnx.Rngs each; within a block, the S4 layer consumes rngs.params()
first, then norm/out/out2 consume a second, separate split) so that M6
(BlockConfig(norm="layer", activation="gelu", glu=True, prenorm=True,
residual=True)) built from the same seed reproduces legacy bit-exactly -
see tests/test_parity.py.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from s4dpc.blocks import BlockConfig, ConfigurableBlock


class StackedModel(nnx.Module):
    def __init__(
        self,
        block_config: BlockConfig,
        d_input: int,
        d_output: int,
        n_layers: int,
        *,
        decode: bool = False,
        rngs: nnx.Rngs,
    ):
        self.block_config = block_config
        self.d_input = d_input
        self.d_output = d_output
        self.n_layers = n_layers
        self.decode = decode

        keys = jax.random.split(rngs.params(), 3)
        self.encoder = nnx.Linear(d_input, block_config.d_model, rngs=nnx.Rngs(params=keys[0]))
        self.decoder = nnx.Linear(block_config.d_model, d_output, rngs=nnx.Rngs(params=keys[1]))

        layer_keys = jax.random.split(keys[2], n_layers)
        self.layers = [
            ConfigurableBlock(block_config, decode=decode, rngs=nnx.Rngs(params=layer_keys[i]))
            for i in range(n_layers)
        ]

    def init_state(self, N: int | None = None) -> list[jax.Array]:
        n = N if N is not None else self.block_config.N
        shape = (self.block_config.d_model, n)
        return [jnp.zeros(shape, dtype=jnp.complex64) for _ in range(self.n_layers)]

    def __call__(
        self, x: jax.Array, states: list[jax.Array] | None = None
    ) -> tuple[jax.Array, list[jax.Array]]:
        """x: (L, d_input), or (d_input,) for one decode=True step."""
        was_1d = x.ndim == 1
        if was_1d:
            x = x[jnp.newaxis, :]
        elif x.ndim != 2:
            raise ValueError(f"expected x with shape (L, d_input) or (d_input,); got {x.shape}")

        x = self.encoder(x)
        current_states = states if states is not None else self.init_state()

        new_states = []
        for layer, state in zip(self.layers, current_states):
            x, new_state = layer(x, state)
            new_states.append(new_state)

        output = self.decoder(x)
        if was_1d:
            output = output.squeeze(0)

        return output, new_states
