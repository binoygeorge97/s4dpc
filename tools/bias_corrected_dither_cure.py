"""TASK A (user, 2026-08-19, seventh round): re-verify the dither cure's
30/30 near-oracle claim with the affine offset properly modeled.

TASK C already established (separately) that the dither-corrected
(Axx,Axs,Bx) re-fit, WITH an intercept column added, drives the
recovered c0_x to ~2e-15 (exactly zero) at n_dither=2000 - this script
redoes that SAME fit (confirming it inline) and carries it through to
the actual closed-loop DARE synthesis, z* fixed point, and corrected
cost ratio TASK A actually asked for. The residual bias this construction
CANNOT remove is c0_s (the unchanged Asx/Ass/Bs path's own small bias,
from the real M3 checkpoint's encoder feeding the untouched S4
recursion) - included here, not assumed away.

    python tools/bias_corrected_dither_cure.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.linalg import solve_discrete_are

from s4dpc.data import generate_microgrid_trajectory
from s4dpc.systems import get_discrete_matrices

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
L_MAX = 100
DATA_SEED = 42
APRBS_LOW, APRBS_HIGH = -10.0, 10.0
X_SYNTH_RANGE = 5.0
N_DITHER = 2000
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200


def get_training_trajectory(case):
    inputs, targets = generate_microgrid_trajectory(
        batch_size=1, length=L_MAX, seed=DATA_SEED, system_case=case, dt=DT,
        aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH)
    return inputs[0, :, :D_X], inputs[0, :, D_X:], targets[0]


def simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj):
    n_s = Ass.shape[0]
    S = np.zeros((L_MAX, n_s))
    s = np.zeros(n_s)
    for t in range(L_MAX):
        S[t] = s
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return S


def dither_refit_with_intercept(x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, n_dither, rng, s_scale):
    n_s = s_traj.shape[1]
    ones_on = np.ones((L_MAX, 1))
    Z_on = np.hstack([ones_on, x_traj, s_traj, u_traj])
    Y_on = x_next_traj
    if n_dither > 0:
        x_s = rng.uniform(-X_SYNTH_RANGE, X_SYNTH_RANGE, size=(n_dither, D_X))
        u_s = rng.uniform(APRBS_LOW, APRBS_HIGH, size=(n_dither, D_U))
        s_s = rng.standard_normal((n_dither, n_s)) * s_scale
        y_s = x_s @ A_true.T + u_s @ B_true.T
        ones_s = np.ones((n_dither, 1))
        Z = np.vstack([Z_on, np.hstack([ones_s, x_s, s_s, u_s])])
        Y = np.vstack([Y_on, y_s])
    else:
        Z, Y = Z_on, Y_on
    theta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    theta = theta.T
    c0_x = theta[:, 0]
    Axx = theta[:, 1:1 + D_X]
    Axs = theta[:, 1 + D_X:1 + D_X + n_s]
    Bx = theta[:, 1 + D_X + n_s:]
    return c0_x, Axx, Axs, Bx


def solve_dlqr(A, B, Q, R):
    return np.linalg.solve(R + B.T @ solve_discrete_are(A, B, Q, R) @ B,
                            B.T @ solve_discrete_are(A, B, Q, R) @ A)


def robust_margin_and_rho(A, B, K_direct):
    n = A.shape[0]
    Acl = A + B @ K_direct
    rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    return rho, Acl


def simulate_cost_biased(A_open, B_open, K_direct, c_open, x0_batch, oracle_cost):
    Acl = A_open + B_open @ K_direct
    n = Acl.shape[0]
    z = np.zeros((x0_batch.shape[0], n))
    z[:, :D_X] = x0_batch
    stage = 0.0
    for _ in range(EVAL_HORIZON):
        x_k = z[:, :D_X]
        u_k = z @ K_direct.T
        stage += np.mean(np.sum(x_k ** 2, axis=-1)) * Q_X + np.mean(np.sum(u_k ** 2, axis=-1)) * R_U
        z = z @ Acl.T + c_open
        if not np.all(np.isfinite(z)):
            return float("inf"), False
    x_final = z[:, :D_X]
    terminal = np.mean(np.sum(x_final ** 2, axis=-1)) * Q_F
    cost = (stage + terminal) / EVAL_HORIZON
    ratio = cost / oracle_cost if oracle_cost > 0 else float("inf")
    return ratio, bool(np.all(np.isfinite(z)))


def rollout_lqr_true(A, B, K, x0, horizon_N):
    x = x0
    xs, us = [x], []
    for _ in range(horizon_N):
        u = -x @ K.T
        x = x @ A.T + u @ B.T
        xs.append(x)
        us.append(u)
    return np.stack(xs), np.stack(us)


def true_quadratic_cost(x_hist, u_hist, Q_x, R_u, Q_f):
    stage = np.sum(x_hist[:-1] ** 2, axis=-1) * Q_x + np.sum(u_hist ** 2, axis=-1) * R_u
    term = np.sum(x_hist[-1] ** 2, axis=-1) * Q_f
    return float((stage.sum(axis=0) + term).mean() / x_hist.shape[0])


def get_x0_batch(case):
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


def main() -> None:
    rows = []
    rng = np.random.RandomState(0)

    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        x_traj, u_traj, x_next_traj = get_training_trajectory(case)
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A = data["A"]
            n_s = A.shape[0] - D_X
            Asx, Ass = A[D_X:, :D_X], A[D_X:, D_X:]
            Bs = data["B"][D_X:, :]
            C = data["C"]

            s_traj = simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj)
            s_scale = float(np.sqrt(np.mean(s_traj ** 2)))

            c0_x, Axx, Axs, Bx = dither_refit_with_intercept(
                x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, N_DITHER, rng, s_scale)

            Abar_new = np.block([[Axx, Axs], [Asx, Ass]])
            Bbar_new = np.vstack([Bx, Bs])
            # residual bias this construction CANNOT remove: c0_s, from the
            # UNCHANGED Asx/Ass/Bs path's own encoder-driven bias - measured
            # directly from the real M3 checkpoint's OWN full augmented c0
            # (computed once via Task 0's extraction; reused here rather than
            # re-deriving, since Asx/Ass/Bs are literally unchanged).
            c0_full_path = DOCS / "task0_c0_cache" / f"fullM3_{case}_{seed}.npy"
            if c0_full_path.exists():
                c0_full = np.load(c0_full_path)
                c0_s = c0_full[D_X:]
            else:
                c0_s = None

            Q = C.T @ (Q_X * np.eye(D_X)) @ C
            R = R_U * np.eye(D_U)
            try:
                P = solve_discrete_are(Abar_new, Bbar_new, Q, R)
                K_lqr = np.linalg.solve(R + Bbar_new.T @ P @ Bbar_new, Bbar_new.T @ P @ Abar_new)
            except np.linalg.LinAlgError:
                rows.append({"case": case, "seed": seed, "note": "DARE failed",
                             "c0_x_recovered_norm": float(np.linalg.norm(c0_x))})
                continue

            K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])
            K_direct = np.hstack([-K_x, -K_s])
            rho, Acl = robust_margin_and_rho(A_open, B_open, K_direct)

            c_open = np.concatenate([np.zeros(D_X), c0_s]) if c0_s is not None else np.zeros(D_X + n_s)
            z_star = None
            z_star_x_norm = float("nan")
            if rho < 1.0:
                try:
                    z_star = np.linalg.solve(np.eye(Acl.shape[0]) - Acl, c_open)
                    z_star_x_norm = float(np.linalg.norm(z_star[:D_X]))
                except np.linalg.LinAlgError:
                    pass

            ratio_corrected, finite = simulate_cost_biased(A_open, B_open, K_direct, c_open, x0_eval, oracle_cost)

            print(f"  [case{case}/seed{seed}] c0_x_recovered_norm={np.linalg.norm(c0_x):.2e}  rho={rho:.6f}  "
                  f"||z*_x||={z_star_x_norm:.6f}  ratio_corrected={ratio_corrected:.4e}  "
                  f"c0_s_available={c0_s is not None}")

            rows.append({"case": case, "seed": seed, "c0_x_recovered_norm": float(np.linalg.norm(c0_x)),
                         "rho": rho, "z_star_x_norm": z_star_x_norm,
                         "ratio_corrected": ratio_corrected, "finite": finite,
                         "c0_s_available": c0_s is not None})

    out_path = DOCS / "bias_corrected_dither_cure.csv"
    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    ratios = [r["ratio_corrected"] for r in rows if r.get("finite")]
    n_success = sum(1 for r in ratios if r < 100)
    z_norms = [r["z_star_x_norm"] for r in rows if not np.isnan(r.get("z_star_x_norm", float("nan")))]
    print(f"\n=== SUMMARY: {n_success}/{len(rows)} transfer-stable at ratio<100x  "
          f"median ratio={np.median(ratios):.4e}  median ||z*_x||={np.median(z_norms) if z_norms else float('nan'):.6f}")


if __name__ == "__main__":
    main()
