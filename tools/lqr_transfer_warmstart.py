"""TASK 4(b) (user, 2026-08-25): rival hypothesis check - "transient,
not margin." With l_max=100 and modes at 1-1e-2, an initial-condition
transient from a cold (s0=0) start decays on roughly the horizon length
this project has always used for identification - so the LQR-transfer
construction's 0/30-stable, catastrophic-cost result COULD in principle
be an artifact of starting the internal state cold rather than a real,
persistent stability-margin problem. This reruns the exact same
lqr_transfer_to_true_plant.py construction, but with the internal state
WARM-STARTED from a genuine model rollout on the true identification
trajectory (Asx/Ass/Bs driven by the REAL (x_t,u_t) from case_data for
100 steps, s0=0 -> s_100) instead of z_0=[x0_eval; 0].

Pure linear algebra on saved matrices - no training, no GPU.

    python tools/lqr_transfer_warmstart.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "tools"))
from lqr_transfer_to_true_plant import (  # noqa: E402
    robust_margin_and_rho, solve_lqr_cached, get_x0_batch, solve_dlqr,
    rollout_lqr_true, true_quadratic_cost,
)

from s4dpc.identify import case_data, DATA_SEED  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
DT = 0.01
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200
L_MAX = 100
APRBS_LOW, APRBS_HIGH = -10.0, 10.0


def warm_start_s(Asx: np.ndarray, Ass: np.ndarray, Bs: np.ndarray, case: int) -> np.ndarray:
    """Roll the internal-state recursion forward over the REAL, l_max=100
    identification trajectory (the exact same (x_t,u_t) case_data trains
    on), s0=0 -> s_100. Independent of x0_eval - one canonical warm state
    per (case, checkpoint), shared across the whole eval batch, matching
    "a model that has already been running, not one just switched on"."""
    inputs, _ = case_data(case, L_MAX, APRBS_LOW, APRBS_HIGH)
    inputs = np.asarray(inputs)  # (L_MAX, D_X+D_U)
    x_traj = inputs[:, :D_X]
    u_traj = inputs[:, D_X:]
    n_s = Ass.shape[0]
    s = np.zeros(n_s)
    for t in range(L_MAX):
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return s


def simulate_cost_from_z0(A_open, B_open, K_direct, z0_batch, oracle_cost):
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
    cost = (stage + terminal) / EVAL_HORIZON
    ratio = cost / oracle_cost if oracle_cost > 0 else float("inf")
    return {"cost_ratio": ratio, "finite": bool(np.all(np.isfinite(z)))}


def main() -> None:
    rows = []
    print(f"{'case':5s} {'seed':5s} {'rho_cold':>10s} {'ratio_cold':>12s} {'rho_warm':>10s} {'ratio_warm':>12s}  {'||s_warm||':>10s}")
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A, B, C = data["A"], data["B"], data["C"]
            K_lqr, _ = solve_lqr_cached("fullM3", case, seed, A, B, C)
            K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]

            Asx = A[D_X:, :D_X]
            Ass = A[D_X:, D_X:]
            Bs = B[D_X:, :]
            n_s = Ass.shape[0]
            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])
            K_direct = np.hstack([-K_x, -K_s])

            # cold start (the existing, established construction)
            b_cold, rho_cold = robust_margin_and_rho(A_open, B_open, K_direct)
            z0_cold = np.zeros((EVAL_BATCH, D_X + n_s))
            z0_cold[:, :D_X] = x0_eval
            res_cold = simulate_cost_from_z0(A_open, B_open, K_direct, z0_cold, oracle_cost)

            # warm start: s from a real model rollout on the true identification trajectory
            s_warm = warm_start_s(Asx, Ass, Bs, case)
            z0_warm = np.zeros((EVAL_BATCH, D_X + n_s))
            z0_warm[:, :D_X] = x0_eval
            z0_warm[:, D_X:] = s_warm[None, :]
            res_warm = simulate_cost_from_z0(A_open, B_open, K_direct, z0_warm, oracle_cost)
            # rho is unchanged by initial condition (property of Acl alone) - same b_cold/rho_cold value

            print(f"{case:<5d} {seed:<5d} {rho_cold:>10.6f} {res_cold['cost_ratio']:>12.4e} "
                  f"{rho_cold:>10.6f} {res_warm['cost_ratio']:>12.4e}  {float(np.linalg.norm(s_warm)):>10.4f}")
            rows.append({
                "case": case, "seed": seed, "rho": rho_cold, "b": b_cold,
                "ratio_cold": res_cold["cost_ratio"], "finite_cold": res_cold["finite"],
                "ratio_warm": res_warm["cost_ratio"], "finite_warm": res_warm["finite"],
                "s_warm_norm": float(np.linalg.norm(s_warm)),
            })

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "lqr_transfer_warmstart.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    n_stable = sum(1 for r in rows if r["b"] > 0)
    finite_cold = [r["ratio_cold"] for r in rows if r["finite_cold"]]
    finite_warm = [r["ratio_warm"] for r in rows if r["finite_warm"]]
    print(f"\nn_stable (rho<1, same for both since rho is IC-independent): {n_stable}/{len(rows)}")
    if finite_cold:
        import statistics as st
        print(f"cold: median={st.median(finite_cold):.4e}  min={min(finite_cold):.4e}  max={max(finite_cold):.4e}")
    if finite_warm:
        import statistics as st
        print(f"warm: median={st.median(finite_warm):.4e}  min={min(finite_warm):.4e}  max={max(finite_warm):.4e}")


if __name__ == "__main__":
    main()
