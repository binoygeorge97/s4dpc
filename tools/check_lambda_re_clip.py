"""TASK B (part 2, user 2026-08-15): does the Lambda_re clip at -1e-4 actually
keep the S4 layer's OWN per-channel discrete dynamics stable?

s4-nnx (src/s4_nnx/s4.py) clips ONLY the diagonal continuous-time real part:
    lambd = clip(Lambda_re, None, -1e-4) + 1j*Lambda_im
before discretizing via discrete_dplr(lambd, P, P, B, C, step, l_max), which
builds a DIAGONAL-PLUS-LOW-RANK (DPLR) state matrix diag(lambd) - P @ P^H (P is
a FREE, unclipped trainable parameter) and bilinear-discretizes THAT, not
diag(lambd) alone. diag(lambd) is not Hermitian (lambd is complex), so Weyl's
inequality (which would guarantee a negative-semidefinite Hermitian correction
like -P P^H can only move eigenvalues toward more-stable) does not apply - the
correction can shift eigenvalues in either direction for a non-normal matrix.
This calls discrete_dplr DIRECTLY (the exact function trained checkpoints run
through, cloned read-only from the pinned v0.2.0 tag - no re-derivation) on
the loaded checkpoint's raw parameters, to see whether the per-CHANNEL discrete
transition itself is already unstable (proving the clip does not protect what
it looks like it protects), separate from any further instability introduced
by the (unconstrained) encoder/decoder feedback composition captured in the
full augmented operator.

    python tools/check_lambda_re_clip.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Needs s4-nnx's source on the path (read-only, NOT pip-installed - its
# pyproject.toml pins flax>=0.11.2 which conflicts with what could be
# installed locally at investigation time). Reproduce with:
#   git clone --quiet https://github.com/binoygeorge97/s4-nnx.git /tmp/s4-nnx-src --branch v0.2.0
sys.path.insert(0, "/tmp/s4-nnx-src/src")

import jax.numpy as jnp
import numpy as np
import flax.serialization as serialization
from s4_nnx.s4 import discrete_dplr

CKPT_DIR = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
CLIP = -1e-4


def load_s4_params(case: int, seed: int):
    path = CKPT_DIR / f"M3_case{case}_seed{seed}.msgpack"
    if not path.exists():
        return None
    d = serialization.msgpack_restore(path.read_bytes())
    seq = d["layers"]["0"]["seq"]
    return {k: np.asarray(v) for k, v in seq.items()}


def per_channel_check(p: dict) -> dict:
    """p: raw params, Lambda_re/Lambda_im/P/B: (D_MODEL, N), log_step: (D_MODEL,1).
    Returns per-channel diagnostics."""
    D_MODEL, N = p["Lambda_re"].shape
    step = np.clip(np.exp(p["log_step"][:, 0]), 0.001, 1.0)
    lambda_re_raw = p["Lambda_re"]
    lambda_re_used = np.clip(lambda_re_raw, None, CLIP)
    n_at_clip_boundary = int(np.sum(np.isclose(lambda_re_used, CLIP) & (lambda_re_raw > CLIP - 1e-12)))
    n_clipped = int(np.sum(lambda_re_raw > CLIP))  # raw value was ABOVE the clip (clip was binding)
    max_raw = float(np.max(lambda_re_raw))

    max_discrete_pole = 0.0
    n_channel_unstable_discrete = 0
    n_channel_unstable_continuous_dplr = 0
    for ch in range(D_MODEL):
        lambd = lambda_re_used[ch] + 1j * p["Lambda_im"][ch]
        # continuous-time DPLR matrix actually used: diag(lambd) - P P^H
        Pc = p["P"][ch]
        A_dplr = np.diag(lambd) - np.outer(Pc, Pc.conj())
        eig_cont = np.linalg.eigvals(A_dplr)
        if np.max(eig_cont.real) > 0:
            n_channel_unstable_continuous_dplr += 1

        a_bar, b_bar, c_bar = discrete_dplr(
            jnp.asarray(lambd), jnp.asarray(Pc), jnp.asarray(Pc), jnp.asarray(p["B"][ch]),
            jnp.asarray(p["C_real_imag"][ch, :, 0] + 1j * p["C_real_imag"][ch, :, 1]),
            jnp.asarray(step[ch]), l_max=100,
        )
        eig_disc = np.linalg.eigvals(np.asarray(a_bar))
        rho = float(np.max(np.abs(eig_disc)))
        max_discrete_pole = max(max_discrete_pole, rho)
        if rho > 1.0:
            n_channel_unstable_discrete += 1

    return {
        "D_MODEL": D_MODEL, "N": N,
        "n_clipped_raw_gt_boundary": n_clipped, "max_raw_lambda_re": max_raw,
        "n_channel_unstable_continuous_dplr": n_channel_unstable_continuous_dplr,
        "n_channel_unstable_discrete": n_channel_unstable_discrete,
        "max_discrete_pole_radius": max_discrete_pole,
    }


def main() -> None:
    print(f"{'case':5s} {'seed':5s} {'raw_gt_-1e-4':>13s} {'max_raw_Lre':>12s} "
          f"{'unstable_cont_dplr':>19s} {'unstable_disc_ch':>17s} {'max_disc_rho':>13s}")
    for case in CASES:
        for seed in range(N_SEEDS):
            p = load_s4_params(case, seed)
            if p is None:
                continue
            r = per_channel_check(p)
            print(f"{case:<5d} {seed:<5d} {r['n_clipped_raw_gt_boundary']:>13d}/{r['D_MODEL']:<2d} "
                  f"{r['max_raw_lambda_re']:>12.4f} {r['n_channel_unstable_continuous_dplr']:>19d} "
                  f"{r['n_channel_unstable_discrete']:>17d} {r['max_discrete_pole_radius']:>13.6f}")


if __name__ == "__main__":
    main()
