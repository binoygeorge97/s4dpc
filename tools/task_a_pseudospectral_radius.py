"""Correct predictor for coupling-induced instability (user, 2026-08-26):
kappa_max * ||A_xs|| (docs/task_a_coupling_quantify.csv) runs 1e10-1e14 -
far outside first-order perturbation theory's validity radius (valid only
while eps*kappa << 1), so its weak/non-significant correlation with
n_unstable(Abar) (Spearman 0.22, p=0.24) is what "linear theory doesn't
apply" looks like, not an unexplained miss. The corrected predictor is the
eps-pseudospectral radius of A_ss at eps=||A_xs|| - rho_eps(A_ss) =
max{|z| : sigma_min(zI-A_ss) <= eps} - which does not assume small
perturbations.

COARSE approximation only, as instructed ("a coarse grid ... suffice"):
for a modest subsample of checkpoints, evaluate sigma_min(zI-A_ss) via
dense SVD (n=1024 is tractable for a full SVD per grid point, just not
for a fine grid or the full population) on a grid of z values at several
radii x several angles, and report the max radius at which the grid found
sigma_min <= eps anywhere. This is a lower bound on the true pseudospectral
radius (finer grids can only find MORE such points, never fewer), stated
as such - not the exact continuous pseudospectral radius.

    python tools/task_a_pseudospectral_radius.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import csv
import numpy as np

D_X = 6

RADII = [1.00, 2.0, 5.0, 10.0, 20.0]
N_ANGLES = 2

# subsample: mix across cases/seeds/B-values, kept modest for runtime
# (dense complex SVD at n=1024 is the bottleneck - this grid trades
# resolution for tractability, see module docstring)
SUBSAMPLE = [
    ("B1", 1, 0), ("B1", 2, 0), ("B320", 3, 3), ("B320", 4, 0),
]


def load_A(label: str, case: int, seed: int):
    if label == "B1":
        path = _REPO_ROOT / "docs" / "nu_gap_export" / f"fullM3_{case}_{seed}.npz"
    else:
        path = _REPO_ROOT / "docs" / "b320" / f"M3_b320_{case}_{seed}.npz"
    return np.load(path)["A"]


def sigma_min(M: np.ndarray) -> float:
    return float(np.linalg.svd(M, compute_uv=False)[-1])


def pseudospectral_radius_lower_bound(A_ss: np.ndarray, eps: float) -> float:
    n = A_ss.shape[0]
    I = np.eye(n)
    best = 0.0
    for r in RADII:
        for k in range(N_ANGLES):
            theta = 2 * np.pi * k / N_ANGLES
            z = r * np.exp(1j * theta)
            smin = sigma_min(z * I - A_ss)
            if smin <= eps and r > best:
                best = r
    return best


def main() -> None:
    n_axs = {}
    with open(_REPO_ROOT / "docs" / "task_a_coupling_quantify.csv") as f:
        for row in csv.DictReader(f):
            if row["axs_norm"]:
                n_axs[(row["label"], int(row["case"]), int(row["seed"]))] = float(row["axs_norm"])
    n_unstable_full = {}
    with open(_REPO_ROOT / "docs" / "n_unstable_characterization.csv") as f:
        for row in csv.DictReader(f):
            n_unstable_full[(row["label"], int(row["case"]), int(row["seed"]))] = int(row["n_unstable_full"])

    label_map_axs = {"B1": "M3_B1", "B320": "M3_B320"}

    results = []
    for label, case, seed in SUBSAMPLE:
        A = load_A(label, case, seed)
        A_ss = A[D_X:, D_X:]
        eps = n_axs[(label_map_axs[label], case, seed)]
        rho_eps_lb = pseudospectral_radius_lower_bound(A_ss, eps)
        nu_full = n_unstable_full[(label, case, seed)]
        results.append({"label": label, "case": case, "seed": seed, "eps_axs_norm": eps,
                         "rho_eps_lower_bound": rho_eps_lb, "n_unstable_full": nu_full})
        print(f"[{label}/case{case}/seed{seed}] eps=||A_xs||={eps:.4f}  "
              f"rho_eps(A_ss) lower bound (coarse grid) = {rho_eps_lb}  n_unstable(Abar)={nu_full}")

    header = sorted({k for r in results for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in results]
    out_path = _REPO_ROOT / "docs" / "task_a_pseudospectral_radius.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    from scipy.stats import spearmanr
    xs = [r["rho_eps_lower_bound"] for r in results]
    ys = [r["n_unstable_full"] for r in results]
    rho, p = spearmanr(xs, ys)
    print(f"\nSpearman(rho_eps_lower_bound, n_unstable_full), n={len(results)}: rho={rho:.3f} p={p:.3f}")
    print(f"rho_eps_lower_bound values: {sorted(set(xs))}")


if __name__ == "__main__":
    main()
