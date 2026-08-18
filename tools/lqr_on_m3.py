"""TASK A (user, 2026-08-16): does a stabilizing controller for M3 exist,
and can a controller that only observes the 6-dim physical state x (never
the 1024-dim S4 hidden state s - the GRU's actual information constraint,
matching every controller in this project) hold it?

Synthesizes the FULL-STATE-FEEDBACK discrete LQR for M3's augmented
(A, B, C) directly (state = [x; s], 1030-dim; Q = C^T (Q_x I_6) C, R = R_u I_3
- same weights the DPC cost itself uses). Reports b (robust margin, same
convention as docs/nu_gap_analysis.csv) and closed-loop spectral radius for:
  - the FULL LQR gain K (acts on all 1030 dims - an oracle that knows s)
  - the SAME gain with its s-columns zeroed (K_xonly - the only form of
    linear gain a GRU's static K_eff linearization could structurally
    represent, since the GRU only ever receives x)
plus ||K||_op split into its x-block and s-block operator norms, and the
control magnitude K_xonly@x would demand at typical/max training-range
||x|| against CASE_MAX_ACTION (the third reading: does the REQUIRED gain
exceed what a bounded max_action*tanh(...) output could deliver, even
before asking whether the function class could represent it).

Reads tools/nu_gap_export.py's existing fullM3_*.npz exports directly - the
DARE solve is the only new computation, pure CPU/scipy, no GPU.

    python tools/lqr_on_m3.py
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

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
Q_X, R_U = 5.0, 0.1
CASE_MAX_ACTION = {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 200.0, 7: 50.0}
TRAIN_X0_RANGE = 3.0


def robust_margin(A: np.ndarray, B: np.ndarray, K_direct: np.ndarray):
    """b_{K,P}, same convention as tools/nu_gap_analysis.py: K_direct such
    that u = K_direct @ x (NOT the u=-Kx LQR convention - caller negates)."""
    n, m = A.shape[0], B.shape[1]
    Acl = A + B @ K_direct
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1.0:
        return 0.0, float(np.max(np.abs(np.linalg.eigvals(Acl))))
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
    rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    return b, rho


def main() -> None:
    rows = []
    print(f"{'case':5s} {'seed':5s} {'DARE_s':>7s} {'b_full':>8s} {'rho_full':>9s} "
          f"{'b_xonly':>8s} {'rho_xonly':>10s} {'||Kx||':>9s} {'||Ks||':>10s} "
          f"{'req_u_at_edge':>13s} {'max_action':>10s}")
    for case in CASES:
        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A, B, C = data["A"], data["B"], data["C"]
            n = A.shape[0]
            Q = C.T @ (Q_X * np.eye(D_X)) @ C
            R = R_U * np.eye(D_U)

            t0 = time.time()
            P = solve_discrete_are(A, B, Q, R)
            dare_time = time.time() - t0
            K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)  # u = -K_lqr @ z

            b_full, rho_full = robust_margin(A, B, -K_lqr)

            K_xonly = np.zeros_like(K_lqr)
            K_xonly[:, :D_X] = K_lqr[:, :D_X]
            b_xonly, rho_xonly = robust_margin(A, B, -K_xonly)

            norm_Kx = float(np.linalg.svd(K_lqr[:, :D_X], compute_uv=False)[0])
            norm_Ks = float(np.linalg.svd(K_lqr[:, D_X:], compute_uv=False)[0])

            max_action = CASE_MAX_ACTION[case]
            # required ||u|| = ||K_xonly @ x|| at the EDGE of the training
            # x0 distribution (||x||=TRAIN_X0_RANGE*sqrt(D_X), a corner of
            # the uniform box) - the state K_xonly would actually need to
            # act on if it were the whole story
            x_edge_norm = TRAIN_X0_RANGE * np.sqrt(D_X)
            req_u_at_edge = norm_Kx * x_edge_norm

            print(f"{case:<5d} {seed:<5d} {dare_time:>7.1f} {b_full:>8.4f} {rho_full:>9.6f} "
                  f"{b_xonly:>8.4f} {rho_xonly:>10.6f} {norm_Kx:>9.3f} {norm_Ks:>10.3f} "
                  f"{req_u_at_edge:>13.3f} {max_action:>10.1f}")

            rows.append({"case": case, "seed": seed, "dare_time_s": dare_time,
                         "b_full": b_full, "rho_full": rho_full,
                         "b_xonly": b_xonly, "rho_xonly": rho_xonly,
                         "norm_Kx": norm_Kx, "norm_Ks": norm_Ks,
                         "req_u_at_edge": req_u_at_edge, "max_action": max_action})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path = _REPO_ROOT / "docs" / "lqr_on_m3.csv"
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    n_b_full_zero = sum(1 for r in rows if r["b_full"] == 0.0)
    print(f"\nSTOP CHECK: {n_b_full_zero}/{len(rows)} checkpoints have b_full == 0.0 "
          f"(full-state LQR itself unstable){'  <<<< STOP, something is wrong with b' if n_b_full_zero else ' - none, proceeding as expected'}")


if __name__ == "__main__":
    main()
