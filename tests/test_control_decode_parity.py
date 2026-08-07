"""s4dpc.control.rollout_learned steps a StackedModel one input at a time
(decode=True) using params trained in conv mode (decode=False,
s4dpc.identify). Conv and step form are mathematically equivalent for an
LTI SSM by construction, but go through different code paths (FFT/matrix
causal convolution vs sequential recurrence) - this must be checked
empirically, not assumed, before rollout_learned's control-loop
predictions can be trusted. test_decode_construction_parity.py already
checks the two modes' PARAMS match at init; this checks their FORWARD
OUTPUT matches, on trained (not just freshly-initialized) params, for
both M3 (no norm/activation) and M6 (the full nonlinear stack) - the two
poles of the variant ladder.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, _train_one, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 50  # just enough to move params off their random init
SEED = 0


@pytest.mark.parametrize("variant", ["M3", "M6"])
def test_decode_true_step_matches_decode_false_conv(variant: str):
    import jax

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)

    trained_state, _ = _train_one(
        block_config, N_LAYERS, EPOCHS, learning_rate=1e-3, key=key, inputs=inputs, targets=targets
    )

    model_cnn = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS, decode=False,
        rngs=nnx.Rngs(params=key),
    )
    nnx.update(model_cnn, trained_state)
    conv_out, _ = model_cnn(inputs, model_cnn.init_state(N=STATE_SIZE))

    model_rnn = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS, decode=True,
        rngs=nnx.Rngs(params=key),
    )
    nnx.update(model_rnn, trained_state)
    state = model_rnn.init_state(N=STATE_SIZE)
    step_outs = []
    for t in range(inputs.shape[0]):
        out_t, state = model_rnn(inputs[t], state)
        step_outs.append(out_t)
    step_out = jnp.stack(step_outs, axis=0)

    max_abs_diff = float(jnp.max(jnp.abs(conv_out - step_out)))
    max_target_scale = float(jnp.max(jnp.abs(targets))) + 1e-8
    print(f"[{variant}] decode=False vs decode=True, max abs diff: {max_abs_diff:.3e} (target scale {max_target_scale:.3e})")
    assert max_abs_diff / max_target_scale < 1e-3, (
        f"{variant}: decode=True stepping diverges from decode=False conv output "
        f"(max_abs_diff={max_abs_diff:.3e}) - rollout_learned would deploy a "
        f"different function than identify.py trained"
    )
