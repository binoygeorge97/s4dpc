"""Part C, C1: bias ablation - the decisive test of the boundedness/
local-linearization theory.

Freezes (or scales) every additive constant in the model - encoder
bias, decoder bias, the block's out/out2 (GLU) biases, and LayerNorm's
beta - at zero (or at a scaled multiple of their TRAINED values, for
the r*-scaling converse test). With every bias zero, F(0)=0 exactly and
the only source of scale-dependence left is LayerNorm's own 1/sigma
prefactor - the map is then exactly degree-0 through the branch (no
bias-determined operating point to linearize about), so the theory
predicts the "good region" vanishes entirely: ||J|| ~ 1/r everywhere,
no plateau at any radius.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from s4dpc.model import StackedModel


def _bias_modules(model: StackedModel) -> list:
    """Every nnx.Linear/nnx.LayerNorm module in the model that has a
    `.bias` attribute contributing an additive constant to the forward
    pass. n_layers==1 throughout this study."""
    mods = [model.encoder, model.decoder]
    if model.n_layers >= 1:
        block = model.layers[0]
        mods.append(block.out)
        if block.config.glu:
            mods.append(block.out2)
        if block.norm is not None:
            mods.append(block.norm)
        if block.norm_post is not None:
            mods.append(block.norm_post)
    return mods


def freeze_all_biases_at_zero(model: StackedModel) -> None:
    """In-place: sets every bias/beta to exactly zero and converts it to
    a plain nnx.Variable (not nnx.Param), so nnx.Optimizer(...,
    wrt=nnx.Param) never updates it - same "Variable, not Param" pattern
    already used for freeze_layernorm_affine. Must be called BEFORE
    constructing the optimizer."""
    for mod in _bias_modules(model):
        mod.bias = nnx.Variable(jnp.zeros_like(mod.bias.value))


def scale_encoder_bias(model: StackedModel, k: float) -> None:
    """In-place: scales the (already-trained) encoder bias by k, for
    C1's converse test (does r* scale linearly with ||P b_enc||). Only
    the ENCODER bias is scaled - decoder/out/out2/beta are left as
    trained, isolating b_enc's own contribution to the crossover radius
    as the theory's r* ~ ||P b|| / sigma_max(P M) formula specifically
    concerns (b = the pre-norm activation's bias, v(z)=b+Mz, which for
    a prenorm-free/postnorm block traces back to b_enc plus whatever
    constant the branch itself contributes at z=0 - b_enc is the
    directly controllable, cleanly interpretable piece)."""
    model.encoder.bias.value = model.encoder.bias.value * k
