"""Part C's test plant: x_{k+1} = 1.03 x_k + 0.01 u_k.

Near-unity gain (rho=1.03, barely unstable) and weak control authority
(B=0.01) - the system on which round 2's postnorm S4 was observed to
learn the correct Jacobian only in a local region around the origin,
motivating the boundedness theory tested in this round's "Postnorm
boundedness" section.

Uses A9/A11's short-horizon open-loop-with-resets scheme for
excitation (conditioning.open_loop_with_resets), NOT the closed-loop-
plus-dither scheme scalar_system.py uses for the rho=3 plant - per
this round's own instruction ("no reason to carry the closed-loop
conditioning artifact into these tests") and A9's own finding that
this plant's earlier closed-loop attempt hit a severe B-scale
numerical pathology (cond=1.6e14) that open-loop-with-resets avoids
entirely (cond_standardized=1.38 at reset_every=20, near the ideal
floor).
"""
from __future__ import annotations

import numpy as np

from layernorm_study.src.conditioning import open_loop_with_resets

A_TRUE = 1.03
B_TRUE = 0.01
L_MAX = 100
RESET_EVERY = 20  # 1.03^20 ~ 1.8, negligible growth per window (A9's own bound argument)


def generate_data(
    seed: int = 42,
    length: int = L_MAX,
    reset_every: int = RESET_EVERY,
    aprbs_low: float = -10.0,
    aprbs_high: float = 10.0,
    x0_range: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    return open_loop_with_resets(
        A_TRUE, B_TRUE, length, reset_every, seed=seed,
        aprbs_low=aprbs_low, aprbs_high=aprbs_high, x0_range=x0_range,
    )


def generate_data_at_radius(
    seed: int, length: int, x0_center: float, x0_spread: float,
    aprbs_low: float, aprbs_high: float, reset_every: int = RESET_EVERY,
) -> tuple[np.ndarray, np.ndarray]:
    """Same open-loop-with-resets scheme, but x_0 is redrawn from
    Uniform(x0_center-x0_spread, x0_center+x0_spread) each reset - for
    C5 (two-shell) and C6 (operating-point shift), which need excitation
    concentrated at specific radii rather than centered at 0."""
    import numpy.random as npr

    from s4dpc.data import fast_vectorized_aprbs

    rng = npr.RandomState(seed)
    a_signal = fast_vectorized_aprbs(
        batch_size=1, length=length, low=aprbs_low, high=aprbs_high, hold_prob=0.8, rng=rng, Nu=1,
    )[0, 0]
    inputs = np.zeros((length, 2))
    targets = np.zeros((length, 1))
    x = rng.uniform(x0_center - x0_spread, x0_center + x0_spread)
    for t in range(length):
        if t % reset_every == 0:
            x = rng.uniform(x0_center - x0_spread, x0_center + x0_spread)
        u = a_signal[t]
        inputs[t, 0] = x
        inputs[t, 1] = u
        x_next = A_TRUE * x + B_TRUE * u
        targets[t, 0] = x_next
        x = x_next
    return inputs, targets
