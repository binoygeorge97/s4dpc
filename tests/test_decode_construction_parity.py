"""decode=True and decode=False models built from the same seed must have
bit-identical params: the identify-then-control pipeline trains in conv
mode (decode=False) and deploys in step mode (decode=True), so a mismatch
here would silently train one model and deploy another.
"""
from __future__ import annotations

import jax
import pytest
from flax import nnx
from s4_nnx import S4Config, create_model

_CONFIG_KWARGS = dict(
    d_input=9,
    d_output=6,
    d_model=16,
    n_layers=1,
    state_size=32,
    l_max=100,
)


def _build(decode: bool, seed: int):
    return create_model(S4Config(decode=decode, **_CONFIG_KWARGS), seed=seed)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decode_construction_parity(seed: int):
    model_cnn = _build(decode=False, seed=seed)
    model_rnn = _build(decode=True, seed=seed)

    params_cnn = jax.tree_util.tree_leaves(nnx.state(model_cnn, nnx.Param))
    params_rnn = jax.tree_util.tree_leaves(nnx.state(model_rnn, nnx.Param))

    assert len(params_cnn) == len(params_rnn)
    for p_cnn, p_rnn in zip(params_cnn, params_rnn):
        assert p_cnn.shape == p_rnn.shape
        assert bool(jax.numpy.array_equal(p_cnn, p_rnn)), (
            "decode=True/False params diverged for the same seed"
        )
