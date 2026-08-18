"""TASK A (user, 2026-08-16, second round): does a controller built for M3
transfer to the TRUE plant when equipped with the most generous possible
observer for M3's own internal state?

M3 is exactly LTI: Abar = [[Axx, Axs], [Asx, Ass]], Bbar = [Bx; Bs]
(block-decomposed by physical-state rows/cols (first D_X) vs S4-hidden-state
rows/cols (the rest)). Free-running M3 diverges because it feeds its OWN
(compounding-error) x-prediction back into itself. Instead, drive the SAME
internal recursion with the TRUE, measured x_k and u_k at every step - an
exact, ideal linear predictor implied by M3's own identified dynamics, not
a learned approximation:

    s_hat_{k+1} = Asx @ x_k_true + Ass @ s_hat_k + Bs @ u_k

Then apply the SAME full-state LQR gain that provably stabilizes M3's own
closed loop (tools/lqr_on_m3.py) to the TRUE plant using this estimate:

    u_k = -K_x @ x_k_true - K_s @ s_hat_k
    x_{k+1}_true = A_true @ x_k_true + B_true @ u_k

Stacking z = [x_true; s_hat] gives ONE (D_X + n_s)-dim LINEAR system;
eigenvalues of its closed-loop matrix answer stability by pure linear
algebra - no neural net, no BPTT, no optimizer, no capacity limit anywhere
in the construction. Runs for M3 (the test) and M0_S4/M1 (controls - M1 has
no extra internal state at all, so its version of this construction is the
degenerate/trivial case: apply M1's own full-state LQR gain directly to the
true plant, no observer needed).

Caches each checkpoint's solved (P, K_lqr) to docs/lqr_cache/ - the DARE
solve is the expensive part (~100s per 1030-dim checkpoint) and tools/
lqr_on_m3.py did not save it, so this recomputes M3's from scratch and adds
M0_S4's for the first time. Resumable: skips a (variant,case,seed) whose
cache file already exists.

    python tools/lqr_transfer_to_true_plant.py
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
from s4dpc.systems import get_discrete_matrices  # noqa: E402

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
DT = 0.01
CASE_MAX_ACTION = {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 200.0, 7: 50.0}
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200


def get_x0_batch(case: int) -> np.ndarray:
    """Exact match to controller_oracles.py's eval_key convention, so
    cost_ratio_to_oracle is directly comparable to every other ratio
    reported this session - the only place jax is used in this whole
    script, purely to reproduce this one PRNG draw."""
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
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)  # u = -K_lqr @ z
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
    """z_k = [x_true_k; s_hat_k] (or just x_true_k for M1's trivial case).
    Acl = A_open + B_open@K_direct; propagate z_{k+1}=Acl@z_k, extract
    x_k = z_k[:, :D_X], u_k = K_direct@z_k, matching s4dpc.control's cost
    convention exactly (stage sum over k=0..N-1 + terminal Q_f term, /N)."""
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
    """Inlined from s4dpc.control (identical formula) - that module pulls in
    s4dpc.model -> s4_nnx, which isn't installed locally (flax version
    conflict, docs/DECISIONS.md), and isn't needed for this tiny piece."""
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
    print(f"{'variant':8s} {'case':5s} {'seed':5s} {'b':>10s} {'rho':>10s} {'cost_ratio':>12s} {'finite':>7s}")
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            # ---- M1: trivial case, no extra state - LQR on M1's own fitted (A,B), applied to true plant ----
            m1_path = EXPORT_DIR / f"M1_{case}_{seed}.npz"
            if m1_path.exists():
                m1 = np.load(m1_path)
                A1, B1 = m1["A"], m1["B"]
                K1_direct, _ = solve_lqr_cached("M1", case, seed, A1, B1, m1["C"])
                b1, rho1 = robust_margin_and_rho(A_true, B_true, -K1_direct)
                res1 = simulate_cost(A_true, B_true, -K1_direct, x0_eval, oracle_cost)
                print(f"{'M1':8s} {case:<5d} {seed:<5d} {b1:>10.4f} {rho1:>10.6f} "
                      f"{res1['cost_ratio']:>12.4e} {str(res1['finite']):>7s}")
                rows.append({"variant": "M1", "case": case, "seed": seed, "b": b1, "rho": rho1,
                             "cost_ratio": res1["cost_ratio"], "finite": res1["finite"]})

            # ---- M3 and M0_S4: observer construction ----
            for variant in ["fullM3", "M0_S4"]:
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A, B, C = data["A"], data["B"], data["C"]
                K_lqr, dare_time = solve_lqr_cached(variant, case, seed, A, B, C)
                K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]

                Asx = A[D_X:, :D_X]
                Ass = A[D_X:, D_X:]
                Bs = B[D_X:, :]
                n_s = Ass.shape[0]

                A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
                B_open = np.vstack([B_true, Bs])
                K_direct = np.hstack([-K_x, -K_s])

                b, rho = robust_margin_and_rho(A_open, B_open, K_direct)
                res = simulate_cost(A_open, B_open, K_direct, x0_eval, oracle_cost)
                print(f"{variant:8s} {case:<5d} {seed:<5d} {b:>10.4f} {rho:>10.6f} "
                      f"{res['cost_ratio']:>12.4e} {str(res['finite']):>7s}  (DARE {dare_time:.0f}s)")
                rows.append({"variant": variant, "case": case, "seed": seed, "b": b, "rho": rho,
                             "cost_ratio": res["cost_ratio"], "finite": res["finite"]})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "lqr_transfer_to_true_plant.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    for variant in ["M1", "fullM3", "M0_S4"]:
        these = [r for r in rows if r["variant"] == variant]
        if not these:
            continue
        n_stable = sum(1 for r in these if r["b"] > 0)
        print(f"{variant}: {n_stable}/{len(these)} stable when transferred to the true plant")


if __name__ == "__main__":
    main()
