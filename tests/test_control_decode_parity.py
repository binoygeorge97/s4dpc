"""s4dpc.control.rollout_learned steps a StackedModel one input at a time
(decode=True) using params trained in conv mode (decode=False,
s4dpc.identify). Conv and step form are mathematically equivalent for an
LTI SSM by construction (see legacy/s4.py's discrete_DPLR: the recurrent
path's Cbar is deliberately pre-scaled by (I - Abar^L)^-1 specifically so
that its zero-state scan output matches the conv path's roots-of-unity
kernel exactly), but they go through different code paths (FFT/Cauchy-
kernel causal convolution vs sequential recurrence) - this must be
checked empirically, not assumed, before rollout_learned's control-loop
predictions can be trusted. test_decode_construction_parity.py already
checks the two modes' PARAMS match at init; this checks their FORWARD
OUTPUT matches, on trained (not just freshly-initialized) params, for
both M3 (no norm/activation) and M6 (the full nonlinear stack) - the two
poles of the variant ladder.

Runs under jax_enable_x64=True, not JAX's float32 default: every real
s4dpc.sweep invocation runs at x64 (sweep.py sets it before any JAX op;
CLAUDE.md/docs/DECISIONS.md), so that is the precision regime this test
is meant to certify parity under. Under float32 (JAX's pytest default,
since nothing in the pytest path enables x64 the way sweep.py's argv
parsing does), the Cauchy-kernel evaluation in kernel_DPLR is known to be
numerically sensitive (roots-of-unity evaluation points can land close to
a channel's poles), and a large decode=False-vs-decode=True mismatch
there is expected, not a bug - confirmed directly: M3 ~1.0e-1 and
M6 ~1.3e-2 relative mismatch under float32, both collapsing to ~1e-14/
~1e-15 under x64 with everything else unchanged. Float32 parity is
deliberately NOT a required invariant here; only x64 (the regime every
real experiment actually runs in) is asserted.
"""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, _train_one, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 50  # just enough to move params off their random init
SEED = 0

# ~8 orders of magnitude above the observed x64 floor (M3 1.6e-14, M6
# 5.4e-15 relative - dominated by float64 machine epsilon, ~2.2e-16,
# compounded over L_MAX=100 S4 recursion/Cauchy-kernel steps), and ~1000x
# stricter than the old float32-era 1e-3 - that threshold was really only
# a "not catastrophically diverging" check, not a precision requirement.
# 1e-6 leaves enough headroom to stay robust across different trained-
# parameter draws, cases, or hardware backends without being loosened
# just to pass.
X64_RELATIVE_TOLERANCE = 1e-6


@contextlib.contextmanager
def _x64_scope(enabled: bool):
    """Temporarily set jax.config.jax_enable_x64, restoring the previous
    value on exit (even on failure). jax_enable_x64 is a process-global
    flag, not scoped to one test - a plain `jax.config.update(...)` left
    set here would leak into whichever test runs next in the same pytest
    session. Tests in this directory collect in file order, so a bare
    `pytest tests/` run reaches this file before test_parity.py's
    bit-exact (rtol=0/atol=0) checks against a float32-generated
    checkpoint - forcing x64 on without restoring it would silently
    change what that later file sees."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("variant", ["M3", "M6"])
def test_decode_true_step_matches_decode_false_conv(variant: str):
    with _x64_scope(True):
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
        # init_state() now follows jax_enable_x64 itself (s4dpc/model.py) -
        # no manual dtype cast needed here.
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
        relative_error = max_abs_diff / max_target_scale
        print(
            f"[{variant}] decode=False vs decode=True (x64): max_abs_diff={max_abs_diff:.3e} "
            f"target_scale={max_target_scale:.3e} relative_error={relative_error:.3e}"
        )
        assert relative_error < X64_RELATIVE_TOLERANCE, (
            f"{variant}: decode=True stepping diverges from decode=False conv output under x64 "
            f"(max_abs_diff={max_abs_diff:.3e}, relative_error={relative_error:.3e}) - "
            f"rollout_learned would deploy a different function than identify.py trained"
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_init_state_dtype_follows_x64(enable_x64: bool):
    """Regression test for the bug this file's parity test surfaced:
    StackedModel.init_state() used to hardcode complex64 regardless of
    jax_enable_x64, which made jax.lax.scan fail once decode=True's
    Abar/Bbar/Cbar became complex128 under x64 (lax.scan requires the
    carry's dtype to stay fixed across iterations)."""
    with _x64_scope(enable_x64):
        block_config = BlockConfig(d_model=4, N=4, l_max=8, **VARIANTS["M3"])
        model = StackedModel(
            block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=1, decode=False,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
        )
        states = model.init_state()
        expected_dtype = jnp.complex128 if enable_x64 else jnp.complex64
        assert len(states) == 1
        for s in states:
            assert s.dtype == expected_dtype, f"enable_x64={enable_x64}: expected {expected_dtype}, got {s.dtype}"
