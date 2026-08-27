"""The scalar test plant for Experiment 2: x_{k+1} = 3*x_k + u_k.

Unstable (rho=3): open-loop APRBS excitation (s4dpc.data's convention for
cases 1-7) diverges as 3^100 over a length-100 trajectory. Training data
must come from a STABILIZED closed loop instead: u_k = K_STAB*x_k + a_k,
with a_k an independent APRBS dither providing the excitation the closed
loop itself can't (a pure state-feedback loop with no external dither is
not identifiable - u would be an exact linear function of x, so [x, u]
would be rank-deficient). K_STAB = -2.7 gives closed-loop pole
A_TRUE + B_TRUE*K_STAB = 3.0 + 1.0*(-2.7) = 0.3.

Reuses s4dpc.data.fast_vectorized_aprbs for the dither signal (imported,
not reforked) - everything else here is new, since s4dpc.data's own
generate_microgrid_trajectory is open-loop-only (see its module
docstring) and would diverge on this plant.
"""
from __future__ import annotations

import numpy as np

from s4dpc.data import fast_vectorized_aprbs

A_TRUE = 3.0
B_TRUE = 1.0
K_STAB = -2.7
CLOSED_LOOP_POLE = A_TRUE + B_TRUE * K_STAB  # 0.3


def generate_scalar_trajectory(
    length: int = 100,
    seed: int = 42,
    aprbs_low: float = -10.0,
    aprbs_high: float = 10.0,
    hold_prob: float = 0.8,
    x0_range: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (inputs, targets): inputs (length, 2) = [x_k, u_k],
    targets (length, 1) = x_{k+1}, under the closed-loop-plus-dither
    excitation described above. float64 throughout (np.random.RandomState
    already returns float64), matching identify.fit_least_squares's own
    float64 convention for the ~1e-12-level precision this is meant to
    support."""
    rng = np.random.RandomState(seed)
    a_signal = fast_vectorized_aprbs(
        batch_size=1, length=length, low=aprbs_low, high=aprbs_high, hold_prob=hold_prob, rng=rng, Nu=1,
    )[0, 0]  # (length,)

    inputs = np.zeros((length, 2))
    targets = np.zeros((length, 1))
    x = rng.uniform(-x0_range, x0_range)
    for k in range(length):
        u = K_STAB * x + a_signal[k]
        inputs[k, 0] = x
        inputs[k, 1] = u
        x_next = A_TRUE * x + B_TRUE * u
        targets[k, 0] = x_next
        x = x_next

    return inputs, targets


def fit_least_squares_scalar(inputs: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    """Closed-form [A_hat, B_hat] = Y @ Z^dagger on the SAME (x, u) ->
    x_next data trained models will see, mirroring
    s4dpc.identify.fit_least_squares's convention exactly (float64,
    np.linalg.lstsq rather than forming the pseudo-inverse explicitly).
    Returns (A_hat, B_hat, mse)."""
    z = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    ab_hat_t, _, _, _ = np.linalg.lstsq(z, y, rcond=None)  # (2, 1)
    pred = z @ ab_hat_t
    mse = float(np.mean((pred - y) ** 2))
    a_hat, b_hat = float(ab_hat_t[0, 0]), float(ab_hat_t[1, 0])
    return a_hat, b_hat, mse
