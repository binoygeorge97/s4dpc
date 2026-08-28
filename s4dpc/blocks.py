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
from s4_nnx import S4LayerEnsemble, discrete_dplr


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
    memoryless: bool = False  # layernorm_study/CLAUDE.md: truncate the S4 branch to
    # its zero-lag impulse response only (no memory) - see
    # _memoryless_s4_branch below for why this belongs here rather than in s4-nnx.
    postnorm_also: bool = False  # layernorm_study Exp2's combined prenorm+postnorm
    # arm: an INDEPENDENT second LayerNorm applied at the very end (after the
    # residual add, after the primary prenorm/postnorm norm if any) - default
    # False leaves every existing M3-M6/M6_fix variant's forward pass, and
    # its RNG key-splitting order (test_m6_init_params_match_legacy), untouched.
    layer_norm_eps: float = 1e-6  # layernorm_study's epsilon sweep (Part B Task 5):
    # flax.nnx.LayerNorm's own default - passed through explicitly rather than left
    # implicit, so this can be varied without a second, parallel norm implementation.


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


def _memoryless_s4_branch(seq: S4LayerEnsemble, x: jax.Array) -> jax.Array:
    """Truncates the S4 branch to ONLY its zero-lag impulse-response tap
    (h_0 = C_bar @ B_bar, the memoryless instantaneous gain per channel),
    for layernorm_study's Experiment 2 complexity ladder (arm_1/2/3:
    "linear/nonlinear branch, NO MEMORY" - isolating LayerNorm/activation
    effects from S4's own recurrence). Composes s4-nnx's own PUBLIC
    `discrete_dplr` function (never forked/copied - CLAUDE.md sec 7:
    s4-nnx is pinned/frozen for the week, changes belong in s4dpc instead)
    rather than S4LayerEnsemble.__call__'s conv/decode branches directly,
    for one deliberate reason: this project has an independently
    documented conv-mode-vs-decode-mode numerics gap in the FULL S4
    branch (CLAUDE.md's "M6's conv/step parity gap"), because conv mode's
    kernel_dplr (FFT/Cauchy) and decode mode's discrete_dplr+scan take
    different numerical paths to nominally the same answer. Using
    discrete_dplr's h_0 = C_bar@B_bar as the SAME single formula in BOTH
    modes (rather than kernel_dplr's kernel[0] in conv mode and
    discrete_dplr's h_0 in decode mode) makes the memoryless arms exactly
    conv/step-consistent by construction, instead of inheriting a second,
    unrelated numerics discrepancy on top of the one already being
    isolated for study.

    x: (L, d_model) - same shape ConfigurableBlock's normal S4 path
    consumes, for either L=1 (decode/step mode) or L=l_max (conv mode);
    the elementwise gain below is L-agnostic, so no mode branch is needed
    here at all. Returns (L, d_model)."""
    step = jnp.clip(jnp.exp(seq.log_step.value), 0.001, 1.0)  # (d_model, 1)
    lambd = jnp.clip(seq.Lambda_re.value, None, -1e-4) + 1j * seq.Lambda_im.value  # (d_model, N)
    c_vector = seq.C_real_imag.value[..., 0] + 1j * seq.C_real_imag.value[..., 1]  # (d_model, N)

    def h0_one_channel(lambd_c, p_c, b_c, c_c, step_c):
        _, b_bar, c_bar = discrete_dplr(lambd_c, p_c, p_c, b_c, c_c, step_c, seq.l_max)
        return (c_bar @ b_bar).reshape(()).real

    h0 = jax.vmap(h0_one_channel)(lambd, seq.P.value, seq.B.value, c_vector, step)  # (d_model,)
    gain = h0 + seq.D.value.reshape(-1)  # (d_model,) - same +D feedthrough S4LayerEnsemble adds
    return x * gain[jnp.newaxis, :]


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
            self.norm = nnx.LayerNorm(config.d_model, epsilon=config.layer_norm_eps, rngs=nnx.Rngs(params=keys[0]))
        elif config.norm == "static":
            self.norm = StaticNorm(config.d_model, rngs=nnx.Rngs(params=keys[0]))
        elif config.norm == "none":
            self.norm = None
        else:
            raise ValueError(f"unknown norm {config.norm!r}")

        self.out = nnx.Linear(config.d_model, config.d_model, rngs=nnx.Rngs(params=keys[1]))
        if config.glu:
            self.out2 = nnx.Linear(config.d_model, config.d_model, rngs=nnx.Rngs(params=keys[2]))

        # fold_in (not a 4th split() slot): keeps `keys` a fixed 3-way split
        # of rngs.params() regardless of postnorm_also, so M3-M6/M6_fix
        # (postnorm_also=False always) keep the EXACT key-derivation
        # sequence test_m6_init_params_match_legacy checks bit-for-bit -
        # widening the split() count would change keys[0..2] themselves,
        # not just add a 4th, since split()'s outputs depend on the
        # requested count.
        self.norm_post = None
        if config.postnorm_also:
            post_key = jax.random.fold_in(keys[0], 1)
            self.norm_post = nnx.LayerNorm(config.d_model, epsilon=config.layer_norm_eps, rngs=nnx.Rngs(params=post_key))

    def __call__(self, x: jax.Array, s4_state: jax.Array) -> tuple[jax.Array, jax.Array]:
        skip = x

        if self.norm is not None and self.config.prenorm:
            x = self.norm(x)

        if self.config.memoryless:
            x = _memoryless_s4_branch(self.seq, x)
            new_s4_state = s4_state  # untouched: memoryless branch carries no state forward
        else:
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

        if self.norm_post is not None:
            x = self.norm_post(x)

        return x, new_s4_state
