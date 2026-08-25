"""TASK 4 (user, 2026-08-25), second half: run the SAME LQR-transfer
construction as tools/lqr_transfer_to_true_plant.py against the B=320
M3 checkpoints (tools/identify_b320.py), reporting stable count and
cost ratio.

Deliberately reuses the EXACT SAME simulate_cost/true_quadratic_cost
pair as the original script, bug included (docs/DECISIONS.md's TASK 1
entry - the numerator/denominator normalization mismatch, ~1.005x) -
per instruction, not fixed yet, and fixing it here would make the B=320
numbers not directly comparable to the B=1 numbers already on record.

    python tools/lqr_transfer_b320.py
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

from s4dpc.systems import get_discrete_matrices

EXPORT_DIR = _REPO_ROOT / "docs" / "b320"
CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
DT = 0.01
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200
VARIANT = "M3_b320"


def get_x0_batch(case: int) -> np.ndarray:
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


def solve_lqr_cached(variant: str, case: int, seed: int, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{variant}_{case}_{seed}.npz"
    if path.exists():
        data = np.load(path)
        return data["K_lqr"], float(data["dare_time"])
    Q = C.T @ (Q_X * np.eye(D_X)) @ C
    R = R_U * np.eye(D_U)
    t0 = time.time()
    P = solve_discrete_are(A, B, Q, R)
    dare_time = time.time() - t0
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    np.savez(path, K_lqr=K_lqr, P=P, dare_time=dare_time)
    return K_lqr, dare_time


def robust_margin_and_rho(A: np.ndarray, B: np.ndarray, K_direct: np.ndarray):
    n, m = A.shape[0], B.shape[1]
    Acl = A + B @ K_direct
    rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    if rho >= 1.0:
        return 0.0, rho
    import control
    Bcl = np.hstack([B @ K_direct, B])
    Ccl = np.vstack([K_direct, np.eye(n)])
    Dcl = np.vstack([np.hstack([K_direct, np.zeros((m, m))]), np.hstack([np.eye(n), np.zeros((n, m))])])
    sys = control.StateSpace(Acl, Bcl, Ccl, Dcl, dt=1)
    try:
        gamma = control.norm(sys, p="inf", print_warning=False)
        b = 1.0 / gamma
    except Exception:
        b = None
    return b, rho


def simulate_cost(A_open: np.ndarray, B_open: np.ndarray, K_direct: np.ndarray, x0_batch: np.ndarray,
                   oracle_cost: float) -> dict:
    Acl = A_open + B_open @ K_direct
    n = Acl.shape[0]
    batch = x0_batch.shape[0]
    z = np.zeros((batch, n))
    z[:, :D_X] = x0_batch
    stage = 0.0
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


def solve_dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def rollout_lqr_true(A: np.ndarray, B: np.ndarray, K: np.ndarray, x0: np.ndarray, horizon_N: int):
    x = x0
    xs, us = [x], []
    for _ in range(horizon_N):
        u = -x @ K.T
        x = x @ A.T + u @ B.T
        xs.append(x)
        us.append(u)
    return np.stack(xs), np.stack(us)


def true_quadratic_cost(x_hist: np.ndarray, u_hist: np.ndarray, Q_x: float, R_u: float, Q_f: float) -> float:
    stage = np.sum(x_hist[:-1] ** 2, axis=-1) * Q_x + np.sum(u_hist ** 2, axis=-1) * R_u
    term = np.sum(x_hist[-1] ** 2, axis=-1) * Q_f
    return float((stage.sum(axis=0) + term).mean() / x_hist.shape[0])


def main() -> None:
    rows = []
    print(f"{'case':5s} {'seed':5s} {'b':>10s} {'rho':>10s} {'cost_ratio':>12s} {'finite':>7s}")
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"{VARIANT}_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A, B, C = data["A"], data["B"], data["C"]
            K_lqr, dare_time = solve_lqr_cached(VARIANT, case, seed, A, B, C)

            b, rho = robust_margin_and_rho(A, B, -K_lqr)
            res = simulate_cost(A, B, -K_lqr, x0_eval, oracle_cost)
            print(f"{case:<5d} {seed:<5d} {b:>10.4f} {rho:>10.6f} "
                  f"{res['cost_ratio']:>12.4e} {str(res['finite']):>7s}  (DARE {dare_time:.0f}s)")
            rows.append({"variant": VARIANT, "case": case, "seed": seed, "b": b, "rho": rho,
                         "cost_ratio": res["cost_ratio"], "finite": res["finite"]})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = EXPORT_DIR / "lqr_transfer_b320.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    n_stable = sum(1 for r in rows if r["b"] > 0)
    finite_ratios = [r["cost_ratio"] for r in rows if r["finite"] and np.isfinite(r["cost_ratio"])]
    print(f"\n{VARIANT}: {n_stable}/{len(rows)} stable when transferred to the true plant")
    if finite_ratios:
        print(f"cost_ratio: median={np.median(finite_ratios):.4e}  min={min(finite_ratios):.4e}  "
              f"max={max(finite_ratios):.4e}")


if __name__ == "__main__":
    main()
