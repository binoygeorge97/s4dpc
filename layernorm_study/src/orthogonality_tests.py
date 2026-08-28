"""Round 2, Part A: null tests for round 1.5's "W_dec learns a direction
nearly orthogonal to the branch (cos~0.15)" claim.

Objection (correct, and checked here rather than dismissed): for two
INDEPENDENT RANDOM vectors in R^H, the typical |cosine| is ~1/sqrt(H) -
near-orthogonality in high dimensions is the DEFAULT, not something
learned. At this project's d_model=8, that baseline is 1/sqrt(8)=0.354,
even higher than the cos~0.15-0.16 spot-checked in round 1.5 - meaning
the observed value may be BELOW the random baseline, which would need
explaining on its own terms (over-orthogonalization is a claim too, but
a different one), not just failing to be "more orthogonal than random."

Three tests, all cheap (existing checkpoints, no retraining):
  A1. cosine at init vs convergence, per seed.
  A2. null distribution from >=1000 random unit W_dec directions, percentile
      of the trained value.
  A3. PROJECTED CONTRIBUTION |<w, branch>| / |<w, skip_linear>| (scale-
      sensitive, unlike cosine) at init and convergence, per seed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from s4dpc.diagnostics import step, zero_states
from s4dpc.model import StackedModel


def decoder_vector(model: StackedModel) -> np.ndarray:
    """W_dec as a plain (d_model,) vector - only valid for d_output==1
    (true throughout this scalar-plant study)."""
    if model.d_output != 1:
        raise ValueError("decoder_vector assumes d_output == 1")
    return np.asarray(model.decoder.kernel.value)[:, 0]


def pre_decoder_vectors_along_trajectory(
    model: StackedModel, inputs: jax.Array, targets: jax.Array
) -> list[dict]:
    """skip_dmodel = encoder(z) (LINEAR part only, no b_dec), branch_dmodel
    = the block's raw (pre-decoder) residual contribution, at every real
    trajectory point (S4 state teacher-forced on the real data).

    Computed as block_output - skip_dmodel, calling the REAL block
    forward pass directly (model.layers[0](skip_dmodel, state)) rather
    than reimplementing its internals - exact for residual=True blocks
    where the residual add is a clean sum, i.e. prenorm or no-norm
    blocks (`x = skip + branch`). NOT valid for postnorm blocks, where
    `x = norm(skip + branch)` mixes skip and branch irreversibly before
    this function ever sees them - raises if config.prenorm is False and
    a norm is present, rather than silently returning a meaningless
    decomposition."""
    block = model.layers[0]
    if block.norm is not None and not block.config.prenorm:
        raise ValueError(
            "pre_decoder_vectors_along_trajectory is only exact for prenorm/no-norm "
            "blocks - this model's residual add is norm(skip + branch), not skip + branch"
        )

    d_x = targets.shape[-1]
    state = zero_states(model, dtype=jnp.complex128)[0]  # n_layers==1: unwrap the one-element list
    rows = []
    for t in range(inputs.shape[0]):
        x_t, u_t = inputs[t, :d_x], inputs[t, d_x:]
        z = jnp.concatenate([x_t, u_t])
        skip_dmodel_j = model.encoder(z[jnp.newaxis, :])[0]
        block_output, state = block(skip_dmodel_j[jnp.newaxis, :], state)
        branch_dmodel = np.asarray(block_output[0]) - np.asarray(skip_dmodel_j)
        rows.append({"t": t, "skip_dmodel": np.asarray(skip_dmodel_j), "branch_dmodel": branch_dmodel})
    return rows


def cosine_with_decoder(w: np.ndarray, branch_dmodel: np.ndarray) -> float:
    nw, nb = np.linalg.norm(w), np.linalg.norm(branch_dmodel)
    if nw == 0 or nb == 0:
        return float("nan")
    return float(np.dot(w, branch_dmodel) / (nw * nb))


def null_cosine_percentile(w: np.ndarray, branch_dmodel: np.ndarray, key: jax.Array, n_samples: int = 2000) -> dict:
    """Samples n_samples random unit vectors in R^{len(w)}, computes
    cos(random_dir, branch_dmodel), and reports where the ACTUAL
    cos(w, branch_dmodel) falls in that null distribution (by |cosine|,
    since sign is arbitrary for a random direction)."""
    h = w.shape[0]
    random_dirs = jax.random.normal(key, (n_samples, h))
    random_dirs = random_dirs / jnp.linalg.norm(random_dirs, axis=1, keepdims=True)
    b_unit = branch_dmodel / np.linalg.norm(branch_dmodel)
    null_cos = np.abs(np.asarray(random_dirs @ jnp.asarray(b_unit)))
    actual_cos = abs(cosine_with_decoder(w, branch_dmodel))
    percentile = float(np.mean(null_cos <= actual_cos) * 100)
    return {
        "actual_abs_cos": actual_cos,
        "null_mean_abs_cos": float(np.mean(null_cos)),
        "null_std_abs_cos": float(np.std(null_cos)),
        "theoretical_baseline_1_over_sqrtH": 1.0 / np.sqrt(h),
        "percentile_of_actual_in_null": percentile,
    }


def projected_contribution_ratio(w: np.ndarray, skip_dmodel: np.ndarray, branch_dmodel: np.ndarray) -> float:
    """|<w, branch>| / |<w, skip>| - scale-sensitive (unlike cosine),
    directly measures how much each contributes to the SCALAR output."""
    skip_proj = abs(np.dot(w, skip_dmodel))
    branch_proj = abs(np.dot(w, branch_dmodel))
    return branch_proj / skip_proj if skip_proj > 0 else float("inf")


def dominant_direction(vectors: np.ndarray) -> np.ndarray:
    """Leading right singular vector of `vectors` ((L, H)) - "the"
    direction a collection of vectors (e.g. branch_dmodel across a
    trajectory, which varies per t and has no single fixed direction)
    predominantly points in. Sign is arbitrary (SVD doesn't fix it);
    callers must compare via |cos|, not signed cos."""
    _, _, vt = np.linalg.svd(vectors, full_matrices=False)
    return vt[0]


def kink_amplitude_and_r_star(sweep: list[dict]) -> tuple[float, float]:
    """Operational (amplitude, r*) from an origin sweep (list of dicts
    with 'c' and 'Jx', e.g. scalar_diagnostics.jx_vs_c_sweep's output):
    amplitude = |Jx| at the point closest to the origin minus the
    far-field |Jx| (median of the 5 largest-|c| points on each side);
    r* = smallest |c| (moving inward from the far side) at which |Jx|
    first crosses halfway between the far-field value and the peak -
    the theory's crossover radius between "near field, J~const" and
    "far field, J~1/r" (or, for a well-behaved prenorm/no-norm arm with
    amplitude~0, this returns a large/undefined r* by construction,
    which callers should interpret as "no kink found", not a small one).
    Used by Part B Task 5 (epsilon sweep) and Part C C1 (bias ablation)."""
    jx_abs = np.array([abs(r["Jx"]) for r in sweep])
    c_abs = np.array([abs(r["c"]) for r in sweep])
    jx_far = float(np.median(np.concatenate([jx_abs[:5], jx_abs[-5:]])))
    peak_idx = int(np.argmin(c_abs))
    jx_peak = float(jx_abs[peak_idx])
    amplitude = jx_peak - jx_far
    half = jx_far + amplitude / 2

    pos_mask = np.array([r["c"] > 0 for r in sweep])
    pos_c = c_abs[pos_mask]
    pos_jx = jx_abs[pos_mask]
    order = np.argsort(pos_c)[::-1]
    pos_c, pos_jx = pos_c[order], pos_jx[order]
    r_star = pos_c[-1]
    for i in range(len(pos_c)):
        if amplitude > 0 and pos_jx[i] >= half:
            r_star = pos_c[i]
            break
        elif amplitude < 0 and pos_jx[i] <= half:
            r_star = pos_c[i]
            break
    return amplitude, r_star


def angular_displacement_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """arccos(|cos(v1,v2)|) in degrees - sign-agnostic (both W_dec and
    an SVD-derived dominant direction can flip sign between two
    independently-obtained snapshots without that being a meaningful
    "rotation")."""
    c = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    c = min(1.0, max(-1.0, c))
    return float(np.degrees(np.arccos(c)))
