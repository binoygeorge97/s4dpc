"""TASK B (user, 2026-08-18): show, don't assert, which part of the state
space the transfer construction's instability actually lives in.

For each unstable eigenvalue (|lambda|>1) of the COMBINED closed-loop matrix
(tools/lqr_transfer_to_true_plant.py's A_open + B_open@K_direct - true-plant
x-dynamics + M3's s-dynamics + M3's own LQR gain), computes the eigenvector's
energy fraction in the x-block (first D_X=6 rows) vs the s-block (the
remaining ~1024 rows): frac_x = ||v[:D_X]||^2 / ||v||^2. A mode close to
frac_x=1 is x-block-dominated (a physical-state instability); close to 0 is
s-block-dominated (the surrogate's fictitious internal state driving it).

Reuses docs/lqr_cache/*.npz - no new DARE solves, pure CPU, seconds.

    python tools/eigenmode_decomposition.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.systems import get_discrete_matrices

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01


def main() -> None:
    rows = []
    print(f"{'case':5s} {'seed':5s} {'n_unstable':>10s} {'median_frac_x':>14s} "
          f"{'min_frac_x':>11s} {'max_frac_x':>11s}")
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            cache_path = CACHE_DIR / f"fullM3_{case}_{seed}.npz"
            if not (path.exists() and cache_path.exists()):
                continue
            data = np.load(path)
            A, B = data["A"], data["B"]
            K_lqr = np.load(cache_path)["K_lqr"]
            K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
            n_s = A.shape[0] - D_X
            Asx, Ass, Bs = A[D_X:, :D_X], A[D_X:, D_X:], B[D_X:, :]

            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])
            K_direct = np.hstack([-K_x, -K_s])
            Acl = A_open + B_open @ K_direct

            eigvals, eigvecs = np.linalg.eig(Acl)
            unstable_idx = np.where(np.abs(eigvals) > 1.0)[0]
            if len(unstable_idx) == 0:
                continue

            fracs = []
            for i in unstable_idx:
                v = eigvecs[:, i]
                frac_x = float(np.sum(np.abs(v[:D_X]) ** 2) / np.sum(np.abs(v) ** 2))
                fracs.append(frac_x)
                rows.append({"case": case, "seed": seed, "lambda_re": float(eigvals[i].real),
                             "lambda_im": float(eigvals[i].imag), "abs_lambda": float(abs(eigvals[i])),
                             "frac_x": frac_x})

            print(f"{case:<5d} {seed:<5d} {len(unstable_idx):>10d} {np.median(fracs):>14.6f} "
                  f"{min(fracs):>11.6f} {max(fracs):>11.6f}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "eigenmode_decomposition.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path} ({len(rows)} unstable eigenvalues total)")

    all_fracs = [r["frac_x"] for r in rows]
    print(f"\nACROSS ALL {len(all_fracs)} unstable eigenvalues, all 30 checkpoints:")
    print(f"  median frac_x (physical-state energy fraction): {np.median(all_fracs):.6f}")
    print(f"  mean:   {np.mean(all_fracs):.6f}")
    print(f"  90th pctile: {np.percentile(all_fracs, 90):.6f}")
    print(f"  max:    {max(all_fracs):.6f}")
    n_mostly_s = sum(1 for f in all_fracs if f < 0.5)
    print(f"  {n_mostly_s}/{len(all_fracs)} unstable eigenvalues are majority-s-block "
          f"(frac_x < 0.5)")


if __name__ == "__main__":
    main()
