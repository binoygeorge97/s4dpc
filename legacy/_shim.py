"""Makes legacy/s4.py importable without editing it.

legacy/s4.py (base_code, verbatim) is missing `import jax.numpy as jnp`.
`jnp` is only ever referenced inside function/method bodies there, never at
module level, so `import legacy.s4` succeeds on its own — the NameError only
fires when a function that touches `jnp` is actually called. That means the
fix can be applied after import, as a plain module-attribute assignment:
functions resolve free variables against their module's __dict__ at call
time, and `s4.jnp = jnp` writes straight into that dict. legacy/s4.py itself
is never read for writing, never patched in place, never re-exec'd.

This is the only place that bug is worked around. Every other quirk in
legacy/s4.py (e.g. `StackedModelRegression.__call__` crashes if `states` is
left at its default None, because it forwards `[None] * n_layers` into a
`jax.vmap` that expects real arrays) is left exactly as it is — callers must
route around those, not this shim.
"""
from __future__ import annotations

import jax.numpy as jnp

from legacy import s4 as _s4

_s4.jnp = jnp

from legacy.s4 import (  # noqa: E402  (must follow the jnp patch above)
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
