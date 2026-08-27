"""Riccati-residual audit (user, 2026-08-26): for every DARE solve behind
a reported cost ratio this session, recompute the discrete-time algebraic
Riccati equation residual, cond(R + B'PB), and rho(A - B@K_lqr) on the
DESIGN model (not the transfer target). Flags any checkpoint whose
residual is not near machine precision as unreliable.

Covers, retroactively:
  - fullM3, M1, M0_S4 (B=1, n=30 each): augmented 1030-dim (or 6-dim
    for M1) DARE, cached in docs/lqr_cache/{variant}_{case}_{seed}.npz,
    A/B/C from docs/nu_gap_export/{variant}_{case}_{seed}.npz.
  - fullM3_axszeroed (B=1, n=30, TASK A2): same cache convention, but
    the DARE was solved on A with the A_xs block zeroed - reconstructed
    identically here (A_xx/A_xs/A_sx/A_ss layout, D_X=6).
  - M3_b320 (n=30, TASK 4(a)/A4 control): augmented DARE, A/B/C from
    docs/b320/M3_b320_{case}_{seed}.npz.
  - TASK A4 physical-only 6-dim solves (M3_b320, M1, M0_S4): re-solved
    inline here (cheap, no cache existed for these).

    python tools/task_riccati_residuals.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import numpy as np
from scipy.linalg import solve_discrete_are

CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
D_X, D_U = 6, 3
Q_X, R_U = 5.0, 0.1
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5


def dare_residual(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> dict:
    S = R + B.T @ P @ B
    resid = A.T @ P @ A - P - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A) + Q
    rel_resid = float(np.linalg.norm(resid, "fro") / np.linalg.norm(P, "fro"))
    cond_S = float(np.linalg.cond(S))
    K = np.linalg.solve(S, B.T @ P @ A)
    rho_design = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    return {"rel_residual": rel_resid, "cond_R_BtPB": cond_S, "rho_design": rho_design}


def audit_augmented(variant: str, a_dir: pathlib.Path, a_prefix: str, zero_axs: bool = False) -> list[dict]:
    rows = []
    for case in CASES:
        for seed in range(N_SEEDS):
            cache_path = CACHE_DIR / f"{variant}_{case}_{seed}.npz"
            src_path = a_dir / f"{a_prefix}_{case}_{seed}.npz"
            if not cache_path.exists() or not src_path.exists():
                continue
            cache = np.load(cache_path)
            src = np.load(src_path)
            A, B, C = src["A"], src["B"], src["C"]
            if zero_axs:
                A = A.copy()
                A[:D_X, D_X:] = 0.0
            Q = C.T @ (Q_X * np.eye(D_X)) @ C
            R = R_U * np.eye(D_U)
            P = cache["P"]
            r = dare_residual(A, B, Q, R, P)
            r.update({"variant": variant, "case": case, "seed": seed, "dim": A.shape[0]})
            rows.append(r)
            flag = "" if r["rel_residual"] < 1e-8 else "  <-- FLAG: not machine precision"
            print(f"[{variant}/case{case}/seed{seed}] dim={A.shape[0]} rel_residual={r['rel_residual']:.3e} "
                  f"cond(S)={r['cond_R_BtPB']:.3e} rho_design={r['rho_design']:.6f}{flag}")
    return rows


def audit_physical_only(variant: str, a_dir: pathlib.Path, a_prefix: str) -> list[dict]:
    rows = []
    for case in CASES:
        for seed in range(N_SEEDS):
            src_path = a_dir / f"{a_prefix}_{case}_{seed}.npz"
            if not src_path.exists():
                continue
            src = np.load(src_path)
            A, B = src["A"], src["B"]
            Axx = A[:D_X, :D_X]
            Bx = B[:D_X, :]
            Q = Q_X * np.eye(D_X)
            R = R_U * np.eye(D_U)
            P = solve_discrete_are(Axx, Bx, Q, R)
            r = dare_residual(Axx, Bx, Q, R, P)
            r.update({"variant": f"{variant}_physonly", "case": case, "seed": seed, "dim": D_X})
            rows.append(r)
            flag = "" if r["rel_residual"] < 1e-8 else "  <-- FLAG: not machine precision"
            print(f"[{variant}_physonly/case{case}/seed{seed}] dim=6 rel_residual={r['rel_residual']:.3e} "
                  f"cond(S)={r['cond_R_BtPB']:.3e} rho_design={r['rho_design']:.6f}{flag}")
    return rows


def main() -> None:
    all_rows = []
    print("=== augmented DARE solves (cached K_lqr/P, recomputing residual) ===")
    all_rows += audit_augmented("fullM3", _REPO_ROOT / "docs" / "nu_gap_export", "fullM3")
    all_rows += audit_augmented("M1", _REPO_ROOT / "docs" / "nu_gap_export", "M1")
    all_rows += audit_augmented("M0_S4", _REPO_ROOT / "docs" / "nu_gap_export", "M0_S4")
    all_rows += audit_augmented("fullM3_axszeroed", _REPO_ROOT / "docs" / "nu_gap_export", "fullM3", zero_axs=True)
    all_rows += audit_augmented("M3_b320", _REPO_ROOT / "docs" / "b320", "M3_b320")

    print("\n=== TASK A4 physical-only 6-dim DARE solves (re-solved, no cache existed) ===")
    all_rows += audit_physical_only("M3_b320", _REPO_ROOT / "docs" / "b320", "M3_b320")
    all_rows += audit_physical_only("M1", _REPO_ROOT / "docs" / "nu_gap_export", "M1")
    all_rows += audit_physical_only("M0_S4", _REPO_ROOT / "docs" / "nu_gap_export", "M0_S4")

    header = sorted({k for r in all_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in all_rows]
    out_path = _REPO_ROOT / "docs" / "task_riccati_residuals.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    import statistics as st
    variants = sorted({r["variant"] for r in all_rows})
    print("\n=== summary ===")
    for v in variants:
        these = [r for r in all_rows if r["variant"] == v]
        resids = [r["rel_residual"] for r in these]
        conds = [r["cond_R_BtPB"] for r in these]
        rhos = [r["rho_design"] for r in these]
        n_flagged = sum(1 for x in resids if x >= 1e-8)
        n_design_unstable = sum(1 for x in rhos if x >= 1.0)
        print(f"{v} (n={len(these)}): rel_residual median={st.median(resids):.3e} max={max(resids):.3e}  "
              f"cond(S) median={st.median(conds):.3e} max={max(conds):.3e}  "
              f"rho_design median={st.median(rhos):.6f} max={max(rhos):.6f}  "
              f"n_flagged={n_flagged}  n_design_unstable(rho_design>=1)={n_design_unstable}")


if __name__ == "__main__":
    main()
