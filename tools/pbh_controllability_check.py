"""NEW HYPOTHESIS (user, 2026-08-16): if M3's spurious unstable modes are
UNCONTROLLABLE from u, M3 is not stabilizable and b=0 is FORCED for every
possible controller - explaining every negative result at once (warm start,
k=10, k=1000 identification, the controller-side horizon extension) without
implicating the GRU, the horizon, or the optimizer. Markov parameters are the
transfer function; by the Kalman decomposition the transfer function sees
only the controllable-and-observable subsystem, so an uncontrollable mode
contributes EXACTLY ZERO to the impulse response - M3's ~1e-6 Markov error
is fully consistent with unstable garbage sitting in a subspace
identification could never have seen or constrained.

PBH (Popov-Belevitch-Hautus) test, for every eigenvalue lambda of each
checkpoint's augmented (A, B, C) with |lambda|>1:
    controllable  iff rank[lambda*I - A, B]   == n  (full ROW rank)
    observable    iff rank[[lambda*I - A]; C] == n  (full COLUMN rank)
Reports the smallest singular value of each PBH matrix directly (rank is
numerically fragile near-zero isn't the same as exactly zero - no silent
thresholding) for every unstable mode, per (variant, case, seed), for M3
(the test) and M1/M0_S4 (controls - M0_S4 in particular: 1030-dimensional
like M3, but should be stabilizable if the whole reading is right, since it
was constructed to have zero spurious modes by design).

Reads tools/nu_gap_export.py's existing .npz exports directly - zero new GPU
work, pure numpy/scipy.

    python tools/pbh_controllability_check.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
VARIANTS = ["M1", "M0_S4", "fullM3"]


def pbh_smallest_singular_values(A: np.ndarray, B: np.ndarray, C: np.ndarray, lam: complex) -> tuple[float, float, float, float]:
    """Returns (sv_ctrl, sv_obs, sv_ctrl_relative, sv_obs_relative) - the
    relative versions divide by the PBH matrix's own Frobenius norm, since
    M0_S4's (hand-constructed) and M3's (learned) A/B live at very
    different absolute scales (confirmed directly: ||A||_F ~370 vs ~35,
    ||B||_F ~2.98 vs ~0.34 for case 3) - an absolute-singular-value
    comparison between them is apples-to-oranges without this."""
    n = A.shape[0]
    lamI = lam * np.eye(n, dtype=complex)
    M_ctrl = np.hstack([lamI - A, B.astype(complex)])
    M_obs = np.vstack([lamI - A, C.astype(complex)])
    sv_ctrl = float(np.linalg.svd(M_ctrl, compute_uv=False)[-1])
    sv_obs = float(np.linalg.svd(M_obs, compute_uv=False)[-1])
    return sv_ctrl, sv_obs, sv_ctrl / float(np.linalg.norm(M_ctrl)), sv_obs / float(np.linalg.norm(M_obs))


def reference_scale(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """A representative-controllable-mode smallest-singular-value, for scale:
    the STABLE mode with the smallest |lambda| (deep in the disk, least
    likely to be a spurious/degenerate direction) as an in-checkpoint
    reference for what a "clearly controllable" PBH singular value looks
    like. NOTE: found unreliable in practice - for a block-zeroed
    construction like M0_S4, the smallest-|lambda| mode can itself be a
    near-exactly-zero degenerate eigenvalue, giving a meaningless
    near-zero reference. Kept only as a diagnostic column, NOT used for
    the controllability verdict - see the RELATIVE (matrix-norm-normalized)
    singular values instead, which is the metric actually load-bearing here."""
    eigvals = np.linalg.eigvals(A)
    stable = eigvals[np.abs(eigvals) < 0.9]
    if len(stable) == 0:
        return float("nan")
    lam = stable[np.argmin(np.abs(stable))]
    sv_ctrl, _, _, _ = pbh_smallest_singular_values(A, B, C, lam)
    return sv_ctrl


def main() -> None:
    print(f"{'variant':8s} {'case':5s} {'seed':5s} {'n_unstable':>10s} {'min_sv_ctrl':>12s} "
          f"{'min_sv_obs':>12s} {'min_REL_ctrl':>13s} {'min_REL_obs':>13s} {'ref_scale(diag)':>16s}")
    rows = []
    for variant in VARIANTS:
        for case in CASES:
            for seed in range(N_SEEDS):
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A, B, C = data["A"], data["B"], data["C"]
                eigvals = np.linalg.eigvals(A)
                unstable = eigvals[np.abs(eigvals) > 1.0]
                if len(unstable) == 0:
                    print(f"{variant:8s} {case:<5d} {seed:<5d} {0:>10d} {'-':>12s} {'-':>12s} "
                          f"{'-':>13s} {'-':>13s} {'-':>16s}")
                    rows.append({"variant": variant, "case": case, "seed": seed, "n_unstable": 0})
                    continue

                ref = reference_scale(A, B, C)
                sv_ctrls, sv_obss, rel_ctrls, rel_obss = [], [], [], []
                for lam in unstable:
                    sv_c, sv_o, rel_c, rel_o = pbh_smallest_singular_values(A, B, C, lam)
                    sv_ctrls.append(sv_c)
                    sv_obss.append(sv_o)
                    rel_ctrls.append(rel_c)
                    rel_obss.append(rel_o)
                    rows.append({"variant": variant, "case": case, "seed": seed,
                                 "lambda_re": float(lam.real), "lambda_im": float(lam.imag),
                                 "abs_lambda": float(abs(lam)), "sv_ctrl": sv_c, "sv_obs": sv_o,
                                 "rel_sv_ctrl": rel_c, "rel_sv_obs": rel_o, "ref_scale_diag": ref})
                print(f"{variant:8s} {case:<5d} {seed:<5d} {len(unstable):>10d} {min(sv_ctrls):>12.4e} "
                      f"{min(sv_obss):>12.4e} {min(rel_ctrls):>13.4e} {min(rel_obss):>13.4e} {ref:>16.4e}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "pbh_check.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
