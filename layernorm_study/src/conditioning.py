"""Round 2, A9/A10: structural conditioning analysis.

A9: under closed-loop feedback u=k*x+a (any stabilizing k), u lies in
span(x) plus bounded excitation, so cond(E[zz^T]) has a floor no choice
of k can beat (verified directly in round 2's A8 - every stabilizing k
or randomization of it landed at cond~76-380, never lower). The actual
fix is to decouple u from x structurally: short-horizon OPEN-LOOP
excitation with periodic resets (x_0 redrawn every `reset_every` steps,
u pure APRBS - x and u independent by construction), kept SHORT enough
that the plant's own growth (rho^reset_every) stays bounded.

A10: a "reweight one fixed dataset to isolate conditioning" attempt
(round 2's own instruction, correcting A8's segment_length confound).
Found a genuine structural limit, not a parametrization failure: an
unconstrained numerical search for minimum-cond weights collapses
effective sample size (1/sum(w^2)) to ~1 - achieving low cond by
concentrating nearly all loss-weight on one or two points, which is
itself a confound (a dataset that is EFFECTIVELY 1-2 points has far
less identification signal than one that is effectively 100, regardless
of conditioning). This module's `min_cond_at_eff_n_floor` traces out
the achievable (cond, eff_n) trade-off curve explicitly, so the
retraining sweep can report both quantities at every level rather than
pretending they are separable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from s4dpc.data import fast_vectorized_aprbs


def open_loop_with_resets(
    a_true: float,
    b_true: float,
    total_length: int,
    reset_every: int,
    seed: int,
    aprbs_low: float,
    aprbs_high: float,
    x0_range: float,
    hold_prob: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """x_0 redrawn fresh from Uniform(-x0_range,x0_range) every
    `reset_every` steps; u is PURE APRBS (no feedback at all) - x and u
    are independent by construction. Growth within one reset window is
    a_true^reset_every, so reset_every must be chosen small enough to
    keep that bounded for unstable a_true (verified per-plant in A9,
    not assumed)."""
    rng = np.random.RandomState(seed)
    a_signal = fast_vectorized_aprbs(
        batch_size=1, length=total_length, low=aprbs_low, high=aprbs_high, hold_prob=hold_prob, rng=rng, Nu=1,
    )[0, 0]
    inputs = np.zeros((total_length, 2))
    targets = np.zeros((total_length, 1))
    x = rng.uniform(-x0_range, x0_range)
    for t in range(total_length):
        if t % reset_every == 0:
            x = rng.uniform(-x0_range, x0_range)
        u = a_signal[t]
        inputs[t, 0] = x
        inputs[t, 1] = u
        x_next = a_true * x + b_true * u
        targets[t, 0] = x_next
        x = x_next
    return inputs, targets


def standardized_condition_number(z: np.ndarray) -> float:
    """cond of the CORRELATION matrix (each column standardized to unit
    variance first) - scale-invariant, unlike raw cond(E[zz^T]), which
    depends on the arbitrary choice of physical units for x vs u (a
    genuine issue when B is very small/large, e.g. B=0.01: u must be
    ~100x x in raw units for comparable dynamical influence, so the raw
    Gram matrix is dominated by that unit mismatch, not by genuine
    ill-conditioning). Report ALONGSIDE the raw number, never alone -
    state which convention a given cond figure uses."""
    std = z.std(axis=0)
    z_std = (z - z.mean(axis=0)) / np.where(std > 0, std, 1.0)
    cov = z_std.T @ z_std / len(z_std)
    return float(np.linalg.cond(cov))


def effective_sample_size(w: np.ndarray) -> float:
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def _cond_of_log_weights(theta: np.ndarray, z: np.ndarray) -> float:
    w = np.exp(theta - theta.max())
    w = w / w.sum()
    M = (z * w[:, None]).T @ z
    eigvals = np.linalg.eigvalsh(M)
    return float(eigvals[-1] / max(eigvals[0], 1e-12))


def min_cond_at_eff_n_floor(z: np.ndarray, eff_n_floor: float, n_restarts: int = 3, maxiter: int = 30000) -> dict:
    """Numerically searches for loss-weights w (over the SAME L rows of
    z, in their original order/positions - only the per-row weight
    varies) minimizing cond(sum_t w_t z_t z_t^T) subject to
    effective_sample_size(w) >= eff_n_floor (soft-enforced via a
    quadratic penalty, since scipy.optimize.minimize's simplex method
    used here doesn't take constraints directly). Returns the best
    (cond, eff_n, weights) found across `n_restarts` random inits."""
    L = z.shape[0]

    def objective(theta):
        c = _cond_of_log_weights(theta, z)
        w = np.exp(theta - theta.max())
        w = w / w.sum()
        e = effective_sample_size(w)
        penalty = max(0.0, eff_n_floor - e) ** 2 * 1e6
        return c + penalty

    best = None
    for trial in range(n_restarts):
        theta0 = np.random.RandomState(trial).normal(0, 0.5, L)
        res = minimize(objective, theta0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-9})
        w = np.exp(res.x - res.x.max())
        w = w / w.sum()
        e = effective_sample_size(w)
        c = _cond_of_log_weights(res.x, z)
        if e >= eff_n_floor * 0.9 and (best is None or c < best["cond"]):
            best = {"cond": c, "eff_n": e, "weights": w}
    if best is None:
        # fall back to the closest-to-constraint result found
        theta0 = np.random.RandomState(0).normal(0, 0.5, L)
        res = minimize(objective, theta0, method="Nelder-Mead", options={"maxiter": maxiter})
        w = np.exp(res.x - res.x.max())
        w = w / w.sum()
        best = {"cond": _cond_of_log_weights(res.x, z), "eff_n": effective_sample_size(w), "weights": w}
    return best
