"""TASK A4 (user, 2026-08-25, verbatim spec, HIGHEST PRIORITY): does K_x
from the AUGMENTED synthesis carry contamination from latent
participation in the Riccati solution, even though A_xx itself is
already accurate (B=320, ~3.5e-3 rel err)? Test: solve the 6-dimensional
DARE on (A_xx, B_x) ALONE (discard the latent block entirely - not
"zero A_xs and resynthesize the augmented problem" like TASK A2, a
genuinely smaller, physical-only design), transfer that gain to the
true plant.

For all 30 M3 checkpoints at B=320: report rho(A_t + B_t@K_x^phys), the
cost ratio, and ||K_x^phys - K_x^aug|| / ||K_x^phys|| (K_x^aug = the
ORIGINAL, full-augmented-synthesis K_x, already cached). Controls: M1
(B=1, no augmented state at all - this synthesis IS its own gain by
definition, a sanity check not a real comparison), M0_S4 (B=1, Axs=0 by
hand construction), and the true (A_d, B_d) directly (confirms the DARE
path reproduces the oracle gain exactly - a check on this SCRIPT, not
on any model).

Pure linear algebra, 6-dimensional DARE solves - genuinely cheap (no
1030-dim solves at all).

    python tools/task_a4_physical_only_synthesis.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(_REPO_ROOT / "tools"))
from lqr_transfer_to_true_plant import (  # noqa: E402
    get_x0_batch, solve_dlqr, rollout_lqr_true, true_quadratic_cost, simulate_cost,
)

from s4dpc.systems import get_discrete_matrices  # noqa: E402

CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
DT = 0.01


def solve_physical_only(Axx: np.ndarray, Bx: np.ndarray) -> np.ndarray:
    Q = Q_X * np.eye(D_X)
    R = R_U * np.eye(D_U)
    P = solve_discrete_are(Axx, Bx, Q, R)
    K = np.linalg.solve(R + Bx.T @ P @ Bx, Bx.T @ P @ Axx)
    return K


def run_population(npz_dir: pathlib.Path, npz_prefix: str, label: str, K_x_aug_cache_prefix: str | None) -> list[dict]:
    rows = []
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, 200)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            npz_path = npz_dir / f"{npz_prefix}_{case}_{seed}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            A, B = data["A"], data["B"]
            Axx = A[:D_X, :D_X]
            Bx = B[:D_X, :]
            n_s = A.shape[0] - D_X

            K_x_phys = solve_physical_only(Axx, Bx)

            if n_s > 0:
                Asx = A[D_X:, :D_X]
                Ass = A[D_X:, D_X:]
                Bs = B[D_X:, :]
                A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
                B_open = np.vstack([B_true, Bs])
                K_direct = np.hstack([-K_x_phys, np.zeros((D_U, n_s))])
            else:
                A_open, B_open = A_true, B_true
                K_direct = -K_x_phys

            rho = float(np.max(np.abs(np.linalg.eigvals(A_open + B_open @ K_direct))))
            res = simulate_cost(A_open, B_open, K_direct, x0_eval, oracle_cost)

            gain_diff = None
            if K_x_aug_cache_prefix is not None:
                cache_path = _REPO_ROOT / "docs" / "lqr_cache" / f"{K_x_aug_cache_prefix}_{case}_{seed}.npz"
                if cache_path.exists():
                    K_aug = np.load(cache_path)["K_lqr"]
                    K_x_aug = K_aug[:, :D_X]
                    gain_diff = float(np.linalg.norm(K_x_phys - K_x_aug) / np.linalg.norm(K_x_phys))

            row = {
                "label": label, "case": case, "seed": seed, "rho": rho,
                "cost_ratio": res["cost_ratio"], "finite": res["finite"],
                "gain_diff_rel": gain_diff,
            }
            rows.append(row)
            print(f"[{label}/case{case}/seed{seed}] rho={rho:.6f}  cost_ratio={res['cost_ratio']:.4e}  "
                  f"gain_diff_rel={gain_diff}")
    return rows


def check_oracle_reproduction() -> None:
    print("=== sanity check: does this DARE path reproduce the oracle gain exactly on (A_d,B_d)? ===")
    max_diff = 0.0
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        K_check = solve_physical_only(A_true, B_true)
        diff = float(np.max(np.abs(K_oracle - K_check)))
        max_diff = max(max_diff, diff)
        print(f"  case{case}: max abs diff (oracle vs this script's DARE path) = {diff:.3e}")
    print(f"max over all cases: {max_diff:.3e}  (should be ~0, both paths solve the identical DARE)")


def main() -> None:
    check_oracle_reproduction()
    print()

    all_rows = []
    all_rows += run_population(_REPO_ROOT / "docs" / "b320", "M3_b320", "M3_B320", "fullM3")
    all_rows += run_population(_REPO_ROOT / "docs" / "nu_gap_export", "M1", "M1", None)
    all_rows += run_population(_REPO_ROOT / "docs" / "nu_gap_export", "M0_S4", "M0_S4", "M0_S4")

    header = sorted({k for r in all_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in all_rows]
    out_path = _REPO_ROOT / "docs" / "task_a4_physical_only_synthesis.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    import statistics as st
    for label in ["M3_B320", "M1", "M0_S4"]:
        these = [r for r in all_rows if r["label"] == label]
        if not these:
            continue
        n_stable = sum(1 for r in these if r["rho"] < 1.0)
        finite_costs = [r["cost_ratio"] for r in these if r["finite"] and np.isfinite(r["cost_ratio"])]
        print(f"\n{label} (n={len(these)}): n_stable={n_stable}/{len(these)}")
        if finite_costs:
            print(f"  cost_ratio: median={st.median(finite_costs):.4e} min={min(finite_costs):.4e} "
                  f"max={max(finite_costs):.4e}")
        gd = [r["gain_diff_rel"] for r in these if r["gain_diff_rel"] is not None]
        if gd:
            print(f"  gain_diff_rel (||Kx_phys-Kx_aug||/||Kx_phys||): median={st.median(gd):.4f} "
                  f"min={min(gd):.4f} max={max(gd):.4f}")


if __name__ == "__main__":
    main()
