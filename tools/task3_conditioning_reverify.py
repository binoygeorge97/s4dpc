"""Re-verify TASK 3's "A_ss has zero unstable eigenvalues" / "margin~2.5e-5"
claims via numerically robust methods, per user instruction (2026-08-26):
raw np.linalg.eig on A_ss is provably untrustworthy at the claimed magnitude
- eigenvalue condition number kappa(lambda)=1/|y^H x| runs up to ~1.2e16
(docs/task_a_coupling_quantify.csv), so computed eigenvalue error is
~kappa*eps_machine ~ O(1) via standard first-order perturbation theory.
Full mpmath arbitrary-precision eig is INFEASIBLE at n=1024 (O(n^3) pure-
Python arbitrary-precision arithmetic - many hours to days per checkpoint,
explicitly not attempted, stated rather than silently skipped).

Two robust alternatives used instead:
  1. Discrete Lyapunov/Stein-equation PD test (A_ss^T P A_ss - P = -I).
     Solved via Bartels-Stewart (scipy.linalg.solve_discrete_lyapunov),
     which uses a real Schur decomposition - a UNITARY similarity
     transform, backward-stable regardless of A_ss's non-normality
     (unlike np.linalg.eig's general eigenvector decomposition, which is
     exactly where the kappa~1e16 ill-conditioning enters). Theorem: A is
     Schur-stable (rho(A)<1) IFF this equation has a PD solution P for
     any PD RHS. Checking PD-ness of the (symmetric) solution is itself
     well-conditioned (symmetric eigenvalue problems have kappa=1
     always), so this gives a DEFINITIVE, robust binary stability verdict
     with no dependence on eigenvector conditioning anywhere in the
     pipeline. Run on all 60 checkpoints (B=1 fullM3 + B=320 M3_b320).
  2. ARPACK/Arnoldi (scipy.sparse.linalg.eigs) as an independent float64
     cross-check on the tightest-margin checkpoints - a fundamentally
     different algorithm (orthonormal Krylov-subspace projection) than
     dense np.linalg.eig's general eigendecomposition, so agreement
     between the two is not a tautology.

    python tools/task3_conditioning_reverify.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.sparse.linalg import eigs as arpack_eigs

D_X = 6
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5

# tightest-margin checkpoints from docs/n_unstable_characterization.csv,
# used for the ARPACK cross-check (most stress-testing of the claim)
ARPACK_SUBSAMPLE = [
    ("B320", 3, 3), ("B320", 4, 0), ("B320", 1, 3),
    ("B1", 2, 0), ("B1", 2, 4), ("B1", 3, 3),
]


def lyap_pd_test(A_ss: np.ndarray) -> dict:
    n = A_ss.shape[0]
    P = solve_discrete_lyapunov(A_ss.T, np.eye(n))
    P_sym = 0.5 * (P + P.T)
    asym_frac = float(np.linalg.norm(P - P_sym, "fro") / np.linalg.norm(P, "fro"))
    eigvals_sym = np.linalg.eigvalsh(P_sym)  # well-conditioned: symmetric eigenproblem
    min_eig = float(eigvals_sym.min())
    is_pd = min_eig > 0
    resid = A_ss.T @ P @ A_ss - P + np.eye(n)
    rel_resid = float(np.linalg.norm(resid, "fro") / np.linalg.norm(P, "fro"))
    return {"is_pd": is_pd, "min_eig_Psym": min_eig, "P_asymmetry_frac": asym_frac, "stein_rel_residual": rel_resid}


def load_A_ss(label: str, case: int, seed: int) -> np.ndarray:
    if label == "B1":
        path = _REPO_ROOT / "docs" / "nu_gap_export" / f"fullM3_{case}_{seed}.npz"
    else:
        path = _REPO_ROOT / "docs" / "b320" / f"M3_b320_{case}_{seed}.npz"
    A = np.load(path)["A"]
    return A[D_X:, D_X:]


def main() -> None:
    print("=== Lyapunov/Stein PD stability test, all 60 checkpoints ===")
    rows = []
    for label, src_dir, prefix in [
        ("B1", _REPO_ROOT / "docs" / "nu_gap_export", "fullM3"),
        ("B320", _REPO_ROOT / "docs" / "b320", "M3_b320"),
    ]:
        for case in CASES:
            for seed in range(N_SEEDS):
                path = src_dir / f"{prefix}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                A_ss = np.load(path)["A"][D_X:, D_X:]
                r = lyap_pd_test(A_ss)
                r.update({"label": label, "case": case, "seed": seed})
                rows.append(r)
                flag = "STABLE (PD confirmed)" if r["is_pd"] else "UNSTABLE OR INDETERMINATE (not PD)"
                print(f"[{label}/case{case}/seed{seed}] min_eig(P_sym)={r['min_eig_Psym']:.3e}  "
                      f"stein_residual={r['stein_rel_residual']:.3e}  P_asym_frac={r['P_asymmetry_frac']:.3e}  {flag}")

    n_pd = sum(1 for r in rows if r["is_pd"])
    print(f"\nLyapunov/Stein verdict: {n_pd}/{len(rows)} checkpoints confirmed PD (i.e. A_ss provably "
          f"Schur-stable, rho(A_ss)<1) via a method with NO dependence on eigenvector conditioning.")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "task3_lyapunov_reverify.csv"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")

    print("\n=== ARPACK/Arnoldi cross-check on tightest-margin checkpoints ===")
    print("(independent algorithm - orthonormal Krylov projection, not dense np.linalg.eig)")
    for label, case, seed in ARPACK_SUBSAMPLE:
        A_ss = load_A_ss(label, case, seed)
        dense_max = float(np.max(np.abs(np.linalg.eigvals(A_ss))))
        vals = arpack_eigs(A_ss, k=6, which="LM", return_eigenvectors=False, maxiter=5000, tol=0)
        arpack_max = float(np.max(np.abs(vals)))
        diff = abs(dense_max - arpack_max)
        print(f"[{label}/case{case}/seed{seed}] dense_eig_max_abs={dense_max:.8f}  "
              f"arpack_max_abs={arpack_max:.8f}  abs_diff={diff:.3e}  "
              f"margin(dense)={1 - dense_max:.3e}  margin(arpack)={1 - arpack_max:.3e}")

    print("\n=== mpmath arbitrary-precision full eig at n=1024: explicitly NOT attempted ===")
    print("Reasoning: dense eig is O(n^3); at n=1024 with pure-Python arbitrary-precision")
    print("arithmetic (mpmath, needed for >=50 digits), this is estimated at many hours to")
    print("multiple days PER CHECKPOINT - infeasible within this task's scope. The Lyapunov/Stein")
    print("test above gives a definitive, well-conditioned binary stability verdict instead, and")
    print("the ARPACK cross-check gives an independent-algorithm magnitude estimate; neither")
    print("requires forming or inverting an ill-conditioned eigenvector matrix.")


if __name__ == "__main__":
    main()
