"""TASK A2 (user, 2026-08-25, verbatim spec, PRIORITY): rerun TASK 4(b)'s
warm-started latent initialisation, but with A_xs zeroed at synthesis
(Axx, Asx, Ass, B unchanged - only A_xs zeroed before solving DARE, "no
retraining, no other weight touched", matching the external group's
"converts 356x into 1.0013x" construction). Same 30 fullM3 checkpoints,
same cost function, same CORRECTED divisor (/201, not the buggy /200 -
docs/DECISIONS.md's TASK 1 entry) throughout, for both the cold and warm
initial conditions.

Prediction: harmless, landing near the reference 1.0013x zero-coupling
number. If cost stays inflated under a warm start even with A_xs
zeroed, there is a second injection path from latent to physical state,
and the single-block (A_xs-only) mechanism is incomplete.

Pure linear algebra on saved checkpoints - no training, no GPU. Needs
30 fresh 1030-dim DARE solves (~100s each) for the A_xs-zeroed gain.

    python tools/task_a2_warmstart_axs_zeroed.py
"""
from __future__ import annotations

import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(_REPO_ROOT / "tools"))
from lqr_transfer_to_true_plant import get_x0_batch, solve_dlqr, rollout_lqr_true  # noqa: E402
from lqr_transfer_warmstart import warm_start_s  # noqa: E402

from s4dpc.systems import get_discrete_matrices  # noqa: E402

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
DT = 0.01
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200
CORRECT_DIVISOR = EVAL_HORIZON + 1  # 201, matching true_quadratic_cost's own (always-correct) convention


def true_quadratic_cost_corrected(x_hist: np.ndarray, u_hist: np.ndarray) -> float:
    stage = np.sum(x_hist[:-1] ** 2, axis=-1) * Q_X + np.sum(u_hist ** 2, axis=-1) * R_U
    term = np.sum(x_hist[-1] ** 2, axis=-1) * Q_F
    return float((stage.sum(axis=0) + term).mean() / x_hist.shape[0])  # already /201, unchanged from baseline


def solve_axs_zeroed(case: int, seed: int, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    path = CACHE_DIR / f"fullM3_axszeroed_{case}_{seed}.npz"
    if path.exists():
        data = np.load(path)
        return data["K_lqr"], float(data["dare_time"])
    A_zeroed = A.copy()
    A_zeroed[:D_X, D_X:] = 0.0
    Q = C.T @ (Q_X * np.eye(D_X)) @ C
    R = R_U * np.eye(D_U)
    t0 = time.time()
    P = solve_discrete_are(A_zeroed, B, Q, R)
    dare_time = time.time() - t0
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A_zeroed)
    np.savez(path, K_lqr=K_lqr, P=P, dare_time=dare_time)
    return K_lqr, dare_time


def simulate_cost_from_z0_corrected(A_open, B_open, K_direct, z0_batch, oracle_cost):
    """CORRECTED divisor (/201, matching CORRECT_DIVISOR = EVAL_HORIZON+1) -
    per instruction, both cold and warm use this from the start, not the
    established (buggy, /200) simulate_cost."""
    Acl = A_open + B_open @ K_direct
    stage = 0.0
    z = z0_batch.copy()
    for _ in range(EVAL_HORIZON):
        x_k = z[:, :D_X]
        u_k = z @ K_direct.T
        stage += np.mean(np.sum(x_k ** 2, axis=-1)) * Q_X + np.mean(np.sum(u_k ** 2, axis=-1)) * R_U
        z = z @ Acl.T
        if not np.all(np.isfinite(z)):
            return {"cost_ratio": float("inf"), "finite": False}
    x_final = z[:, :D_X]
    terminal = np.mean(np.sum(x_final ** 2, axis=-1)) * Q_F
    cost = (stage + terminal) / CORRECT_DIVISOR
    ratio = cost / oracle_cost if oracle_cost > 0 else float("inf")
    return {"cost_ratio": ratio, "finite": bool(np.all(np.isfinite(z)))}


def main() -> None:
    rows = []
    print(f"{'case':5s} {'seed':5s} {'rho':>10s} {'ratio_cold_zeroed':>18s} {'ratio_warm_zeroed':>18s}  {'||s_warm||':>10s}")
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost_corrected(x_hist_lqr, u_hist_lqr)

        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A, B, C = data["A"], data["B"], data["C"]
            Asx = A[D_X:, :D_X]
            Ass = A[D_X:, D_X:]
            Bs = B[D_X:, :]
            n_s = Ass.shape[0]
            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])

            K_zeroed, dare_time = solve_axs_zeroed(case, seed, A, B, C)
            K_x_z, K_s_z = K_zeroed[:, :D_X], K_zeroed[:, D_X:]
            K_direct_z = np.hstack([-K_x_z, -K_s_z])

            Acl = A_open + B_open @ K_direct_z
            rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))

            z0_cold = np.zeros((EVAL_BATCH, D_X + n_s))
            z0_cold[:, :D_X] = x0_eval
            res_cold = simulate_cost_from_z0_corrected(A_open, B_open, K_direct_z, z0_cold, oracle_cost)

            s_warm = warm_start_s(Asx, Ass, Bs, case)
            z0_warm = np.zeros((EVAL_BATCH, D_X + n_s))
            z0_warm[:, :D_X] = x0_eval
            z0_warm[:, D_X:] = s_warm[None, :]
            res_warm = simulate_cost_from_z0_corrected(A_open, B_open, K_direct_z, z0_warm, oracle_cost)

            print(f"{case:<5d} {seed:<5d} {rho:>10.6f} {res_cold['cost_ratio']:>18.6e} "
                  f"{res_warm['cost_ratio']:>18.6e}  {float(np.linalg.norm(s_warm)):>10.4f}  (DARE {dare_time:.0f}s)")

            rows.append({
                "case": case, "seed": seed, "rho": rho,
                "ratio_cold_zeroed": res_cold["cost_ratio"], "finite_cold": res_cold["finite"],
                "ratio_warm_zeroed": res_warm["cost_ratio"], "finite_warm": res_warm["finite"],
                "s_warm_norm": float(np.linalg.norm(s_warm)),
            })

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "task_a2_warmstart_axs_zeroed.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    import statistics as st
    cold = [r["ratio_cold_zeroed"] for r in rows if r["finite_cold"] and np.isfinite(r["ratio_cold_zeroed"])]
    warm = [r["ratio_warm_zeroed"] for r in rows if r["finite_warm"] and np.isfinite(r["ratio_warm_zeroed"])]
    n_stable = sum(1 for r in rows if r["rho"] < 1.0)
    print(f"\nn_stable (rho<1, A_xs zeroed): {n_stable}/{len(rows)}")
    if cold:
        print(f"cold (A_xs zeroed): median={st.median(cold):.6e} min={min(cold):.6e} max={max(cold):.6e}")
    if warm:
        print(f"warm (A_xs zeroed): median={st.median(warm):.6e} min={min(warm):.6e} max={max(warm):.6e}")


if __name__ == "__main__":
    main()
