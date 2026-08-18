"""TASK C (user, 2026-08-18, fifth round): the dither cure for closed-loop
identifiability (Gustavsson/Ljung/Soderstrom - when one signal is a
deterministic function of the others' history, regressors are collinear
and only a combination is identifiable; cures are dither, loop noise, or
switching the feedback law). Teacher forcing puts s in exactly that
structural position relative to x (TASK A this round: for every training
timestep t>=1, S's column space exactly reproduces any specific physical-
state perturbation pattern - s_0=0 at t=0 is the only sample that pins
Axx down at all).

IMPLEMENTABILITY, decided before writing this, not assumed: is
"present genuinely independent (x, s_random, u)" actually implementable,
or only a weaker "perturb s" version? M3's S4 layer state is a LINEAR
recursion's hidden vector (no gating nonlinearity - M3 has no norm/
activation/glu at all) - unlike a GRU/LSTM's tanh-gated state, a linear
SSM's state space has NO implicit manifold constraint; any point in that
vector space is a mathematically valid state for the SAME step function.
`s4dpc/diagnostics.py`'s own `zero_states` helper already treats this
state as a plain array to overwrite (that IS how "cold start" s0=0 is
implemented everywhere in this project). So: YES, genuinely independent
random sampling of s is implementable here, not merely perturbation - the
one thing that has to be chosen with care is the SCALE (drawing s from an
unrealistic range would make the test uninformative about anything a real
rollout could encounter), calibrated below from the actual empirical
scale of s reached during real teacher-forced training.

METHOD - no GPU, no gradient descent needed: M3's x-prediction is EXACTLY
LINEAR in (Axx, Axs, Bx) (established in this round's TASK A entry), so
re-fitting this specific sub-block under a modified data presentation is
an ordinary least-squares problem, not a training run. Build a combined
design matrix: the REAL on-manifold (x_t, s_t, u_t) -> x_true_{t+1} data
(same trajectory M3 actually trained on) PLUS N_DITHER synthetic
off-manifold samples (x_synth, s_synth, u_synth) -> x_true_next, where
x_synth/u_synth are drawn from realistic ranges, s_synth is drawn
INDEPENDENTLY from a distribution matching real s's empirical RMS, and
the target is computed from the TRUE plant directly (A_true@x_synth +
B_true@u_synth) - well-defined regardless of s_synth precisely because
the true plant has no internal state for s to stand in for. Re-solve OLS
for (Axx, Axs, Bx) on the combined data; keep the REAL M3's own
(Asx, Ass, Bs, C) unchanged (this only re-identifies the READOUT block,
not the S4 recursion itself - deliberately narrow, matching what the
theory predicts the dither should fix). Plug the corrected (Axx, Axs, Bx)
back into the augmented operator and rerun the SAME LQR-transfer
construction used everywhere else this session.

Pre-registered prediction: Axs -> 0, Axx -> A_d as N_DITHER grows.
Reports both, teacher_mse on the REAL on-manifold data (fairness check -
does fixing identifiability cost real-data fit?), and the LQR-transfer
outcome. Sweeps N_DITHER (0/on-manifold-only baseline, then increasing)
rather than picking one arbitrary amount, since "how much dither is
enough" is itself part of the answer.

    python tools/dither_cure_test.py
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
X_SYNTH_RANGE = 5.0  # matches EVAL_X0_RANGE convention used throughout this session
N_DITHER_SWEEP = [0, 2000]  # reduced from a 5-point sweep after timing: a single DARE solve at
# d1030 did not finish within 100s locally - a 5-point x 30-checkpoint sweep (150 solves) would
# take multiple hours. Two points (no-dither OLS-refit baseline vs large dither) still answers
# the pre-registered question; a finer sweep is a natural follow-up if this shows an effect.
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200


def get_training_trajectory(case: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, targets = generate_microgrid_trajectory(
        batch_size=1, length=L_MAX, seed=DATA_SEED, system_case=case, dt=DT,
        aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH)
    return inputs[0, :, :D_X], inputs[0, :, D_X:], targets[0]


def simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj) -> np.ndarray:
    n_s = Ass.shape[0]
    S = np.zeros((L_MAX, n_s))
    s = np.zeros(n_s)
    for t in range(L_MAX):
        S[t] = s
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return S


def dither_refit(x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, n_dither, rng, s_scale):
    n_s = s_traj.shape[1]
    Z_on = np.hstack([x_traj, s_traj, u_traj])  # (100, D_X+n_s+D_U)
    Y_on = x_next_traj

    if n_dither > 0:
        x_synth = rng.uniform(-X_SYNTH_RANGE, X_SYNTH_RANGE, size=(n_dither, D_X))
        u_synth = rng.uniform(APRBS_LOW, APRBS_HIGH, size=(n_dither, D_U))
        s_synth = rng.standard_normal((n_dither, n_s)) * s_scale
        y_synth = x_synth @ A_true.T + u_synth @ B_true.T
        Z = np.vstack([Z_on, np.hstack([x_synth, s_synth, u_synth])])
        Y = np.vstack([Y_on, y_synth])
    else:
        Z, Y = Z_on, Y_on

    theta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)  # (D_X+n_s+D_U, D_X)
    theta = theta.T  # (D_X, D_X+n_s+D_U)
    Axx_new = theta[:, :D_X]
    Axs_new = theta[:, D_X:D_X + n_s]
    Bx_new = theta[:, D_X + n_s:]

    on_manifold_mse = float(np.mean((Z_on @ theta.T - Y_on) ** 2))
    return Axx_new, Axs_new, Bx_new, on_manifold_mse


def get_x0_batch(case: int) -> np.ndarray:
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


def solve_dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def robust_margin_and_rho(A, B, K_direct):
    n, m = A.shape[0], B.shape[1]
    Acl = A + B @ K_direct
    rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
    if rho >= 1.0:
        return 0.0, rho
    import control
    Bcl = np.hstack([B @ K_direct, B])
    Ccl = np.vstack([K_direct, np.eye(n)])
    Dcl = np.vstack([np.hstack([K_direct, np.zeros((m, m))]), np.hstack([np.eye(n), np.zeros((n, m))])])
    sys_ = control.StateSpace(Acl, Bcl, Ccl, Dcl, dt=1)
    try:
        gamma = control.norm(sys_, p="inf", print_warning=False)
        b = 1.0 / gamma
    except Exception:
        b = None
    return b, rho


def simulate_cost(A_open, B_open, K_direct, x0_batch, oracle_cost):
    Acl = A_open + B_open @ K_direct
    n = Acl.shape[0]
    z = np.zeros((x0_batch.shape[0], n))
    z[:, :D_X] = x0_batch
    stage = 0.0
    for _ in range(EVAL_HORIZON):
        x_k = z[:, :D_X]
        u_k = z @ K_direct.T
        stage += np.mean(np.sum(x_k ** 2, axis=-1)) * Q_X + np.mean(np.sum(u_k ** 2, axis=-1)) * R_U
        z = z @ Acl.T
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
            teacher_mse_orig = float(data["teacher_mse"])

            s_traj = simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj)
            s_scale = float(np.sqrt(np.mean(s_traj ** 2)))

            for n_dither in N_DITHER_SWEEP:
                Axx_new, Axs_new, Bx_new, on_mse = dither_refit(
                    x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, n_dither, rng, s_scale)
                axx_rel_err = float(np.linalg.norm(Axx_new - A_true, "fro") / np.linalg.norm(A_true, "fro"))
                axs_norm = float(np.linalg.norm(Axs_new, "fro"))

                Abar_new = np.block([[Axx_new, Axs_new], [Asx, Ass]])
                Bbar_new = np.vstack([Bx_new, Bs])

                Q = C.T @ (Q_X * np.eye(D_X)) @ C
                R = R_U * np.eye(D_U)
                try:
                    P = solve_discrete_are(Abar_new, Bbar_new, Q, R)
                    K_lqr = np.linalg.solve(R + Bbar_new.T @ P @ Bbar_new, Bbar_new.T @ P @ Abar_new)
                    K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
                    A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
                    B_open = np.vstack([B_true, Bs])
                    K_direct = np.hstack([-K_x, -K_s])
                    b, rho = robust_margin_and_rho(A_open, B_open, K_direct)
                    ratio, finite = simulate_cost(A_open, B_open, K_direct, x0_eval, oracle_cost)
                    dare_ok = True
                except np.linalg.LinAlgError:
                    b, rho, ratio, finite, dare_ok = None, None, float("inf"), False, False

                rows.append({
                    "case": case, "seed": seed, "n_dither": n_dither,
                    "axx_rel_err": axx_rel_err, "axs_norm": axs_norm,
                    "on_manifold_mse": on_mse, "teacher_mse_orig": teacher_mse_orig,
                    "b": b, "rho": rho, "ratio_transfer": ratio, "finite": finite, "dare_ok": dare_ok,
                })
                print(f"  case{case} seed{seed} n_dither={n_dither:5d}: "
                      f"axx_rel_err={axx_rel_err:.4f}  ||Axs||={axs_norm:.3f}  "
                      f"on_mse={on_mse:.3e} (orig teacher_mse={teacher_mse_orig:.3e})  "
                      f"ratio_transfer={ratio}  finite={finite}")

    out_path = DOCS / "dither_cure_test.csv"
    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    print("\n=== SUMMARY by n_dither ===")
    for n_dither in N_DITHER_SWEEP:
        these = [r for r in rows if r["n_dither"] == n_dither]
        axx_errs = [r["axx_rel_err"] for r in these]
        axs_norms = [r["axs_norm"] for r in these]
        n_success = sum(1 for r in these if r["finite"] and r["ratio_transfer"] < 100.0)
        ratios = [r["ratio_transfer"] for r in these if r["finite"]]
        med_ratio = np.median(ratios) if ratios else float("nan")
        print(f"  n_dither={n_dither:5d}: median axx_rel_err={np.median(axx_errs):.4f}  "
              f"median ||Axs||={np.median(axs_norms):.3f}  {n_success}/{len(these)} transfer-stable  "
              f"median ratio={med_ratio:.4e}")


if __name__ == "__main__":
    main()
