"""Discrete-time LTI plants (CLAUDE.md §1, §2): 7 systems, all A:(6,6), B:(6,3).

PLACEHOLDER — no real plant matrices have been provided to this codebase
anywhere (not in CLAUDE.md, not in base_code). The systems below are
fixed-seed, stable-by-construction (symmetric A with eigenvalues drawn in
(0.5, 0.95), so spectral radius < 1 exactly) and exist only so callers like
tools/make_reference_checkpoint.py have *something* concrete and
reproducible to run against. They are not the paper's actual plants.
Replace get_discrete_system's body with the real matrices before any
scientific run, and delete this docstring warning once that happens.
"""
from __future__ import annotations

import numpy as np

N_CASES = 7
STATE_DIM = 6
INPUT_DIM = 3


def get_discrete_system(case: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (A_d, B_d, name) for `case` in 1..7. See module docstring:
    placeholder matrices, not the paper's real plants."""
    if not 1 <= case <= N_CASES:
        raise ValueError(f"case must be in 1..{N_CASES}, got {case}")

    rng = np.random.default_rng(seed=1000 + case)

    q, _ = np.linalg.qr(rng.normal(size=(STATE_DIM, STATE_DIM)))
    eigenvalues = rng.uniform(0.5, 0.95, size=STATE_DIM)
    a_d = q @ np.diag(eigenvalues) @ q.T  # symmetric => eigenvalues are exactly `eigenvalues`

    b_d = 0.5 * rng.normal(size=(STATE_DIM, INPUT_DIM))

    return a_d, b_d, f"placeholder_case_{case}"
