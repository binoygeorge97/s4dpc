"""Part B Task 4 / Part C C2: postnorm's output-ceiling.

LayerNorm maps R^H into a compact ball (||zhat|| <= sqrt(H) for every
v, equality at eps=0), so a postnorm block - where LN is the LAST
operation before the decoder, with no skip bypassing it - has a
uniformly bounded output regardless of input magnitude:

    ||F(z)|| <= ||W_dec|| * (||gamma||_inf * sqrt(H) + ||beta||) + ||b_dec|| =: Y_max

(||gamma||_inf, not ||gamma||_2: LN's output is gamma (elementwise) *
zhat + beta, so the WORST CASE over unit-ball zhat bounds each
component's contribution by |gamma_i|, and ||diag(gamma)zhat|| is
maximized, for ||zhat||<=sqrt(H), by putting all of zhat's norm on the
single largest-|gamma_i| component - giving ||gamma||_inf * sqrt(H),
not ||gamma||_2 * sqrt(H) as an early informal statement of this bound
said. Using the inf-norm is the CORRECT, tight bound.)
"""
from __future__ import annotations

import numpy as np
from flax import nnx

from s4dpc.model import StackedModel


def which_norm_bounds_output(model: StackedModel):
    """Returns the nnx.LayerNorm module that is the LAST operation
    before the decoder (the one whose gamma/beta actually bound the
    output), or None if no such norm exists (prenorm-only or no-norm
    blocks have no output bound at all). For postnorm_also (arm_7),
    this is norm_post (applied after norm/prenorm, right before
    decoding) - NOT the primary prenorm norm."""
    if model.n_layers == 0:
        return None
    block = model.layers[0]
    if block.norm_post is not None:
        return block.norm_post
    if block.norm is not None and not block.config.prenorm:
        return block.norm
    return None  # prenorm-only or no-norm: output is NOT bounded


def compute_y_max(model: StackedModel) -> float | None:
    norm = which_norm_bounds_output(model)
    if norm is None:
        return None
    H = model.block_config.d_model
    gamma_inf = float(np.max(np.abs(np.asarray(norm.scale.value))))
    beta_norm = float(np.linalg.norm(np.asarray(norm.bias.value)))
    W_dec_norm = float(np.linalg.norm(np.asarray(model.decoder.kernel.value)))
    b_dec_norm = float(np.linalg.norm(np.asarray(model.decoder.bias.value)))
    return W_dec_norm * (gamma_inf * np.sqrt(H) + beta_norm) + b_dec_norm
