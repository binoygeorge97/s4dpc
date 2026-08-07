"""Verbatim port of the original notebook S4 implementation (base_code).

This is the parity target: the refactored code (s4dpc, via the s4-nnx
package) must reproduce what legacy/s4.py computes, bugs and all. Import
from here (or from legacy._shim) rather than legacy.s4 directly, so the
missing-jnp patch in _shim.py is guaranteed to have run first.
"""
from legacy._shim import (
    S4LayerEnsemble,
    SequenceBlockNNX,
    StackedModelRegression,
    cauchy,
    causal_convolution,
    discrete_DPLR,
    hippo_initializer,
    kernel_DPLR,
    log_step_initializer,
    make_DPLR_HiPPO,
    make_HiPPO,
    make_NPLR_HiPPO,
    scan_SSM,
)

__all__ = [
    "S4LayerEnsemble",
    "SequenceBlockNNX",
    "StackedModelRegression",
    "cauchy",
    "causal_convolution",
    "discrete_DPLR",
    "hippo_initializer",
    "kernel_DPLR",
    "log_step_initializer",
    "make_DPLR_HiPPO",
    "make_HiPPO",
    "make_NPLR_HiPPO",
    "scan_SSM",
]
