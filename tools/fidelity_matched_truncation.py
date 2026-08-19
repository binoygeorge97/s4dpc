"""TASK D (user, 2026-08-18, sixth round): the fidelity-matched
truncation, deferred since the balanced-truncation entry (2026-08-13)
flagged it as "the only experiment that would actually test the
[spurious-mode] hypothesis - not yet run." With the gauge-freedom theorem
(this round's TASK A) now established, this answers the last open
question precisely: is LQR-transfer failure caused by excess realization
content per se, or specifically by the GAUGE the objective happened to
land on (which TASK C already showed is fixable by dither)?

  - truncation ALSO transfers -> excess content (regardless of which
    gauge produced it) is the culprit, and truncation is a SECOND,
    independent cure - strictly more practical than TASK C's (post-hoc,
    no retraining, no known-simulator/synthetic-target requirement).
  - truncation FAILS at matched fidelity -> the gauge is the whole
    explanation; removing excess content without addressing identifiability
    doesn't help, and truncation is cleanly ruled out.

Method: Hankel-SVD/ERA (tools/balanced_truncation.py's exact machinery -
required because Abar is marginally unstable, so Gramian-based balanced
truncation doesn't apply; reimplemented here directly on numpy, not
imported, since balanced_truncation.py's module-level imports pull in
flax/s4-nnx which aren't reliably available locally - not needed anyway,
since Abar/Bbar are already extracted in docs/nu_gap_export/). For each
fullM3 checkpoint: compute M3's OWN Markov-parameter error vs the true
plant at FULL dimension (this checkpoint's actual achieved fidelity, not
a generic "~1e-6" for everyone) as the fidelity TARGET; sweep truncation
order r=6..60 (N_HANKEL=20 caps the reachable order at min(120,60)=60);
report the Hankel singular-value spectrum; take the smallest r whose
truncated-model error vs the TRUE plant reaches that checkpoint's own
target; run the SAME observer/LQR-transfer construction used everywhere
else this session on the truncated (A,B) (now small - DARE solves are
fast, unlike the ~100-150s d1030 solves this session's other experiments
needed). M1 (control, already 6-dim, r=6 trivially) and M0_S4 (control,
Abar[:6,6:]=0 exactly by construction, so its own full-dimension fidelity
vs true is already near machine precision and truncation to r=6 should
recover it trivially) run through the identical pipeline.

    python tools/fidelity_matched_truncation.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.linalg import solve_discrete_are

from s4dpc.systems import get_discrete_matrices

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
N_HANKEL = 20
R_SWEEP = list(range(6, 61))
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200


def markov_from_augmented(Abar, Bbar, Cout, H):
    Gs = []
    M = np.eye(Abar.shape[0])
    for _ in range(H):
        Gs.append(Cout @ M @ Bbar)
        M = Abar @ M
    return Gs


def era(markov_params, r, n_hankel):
    p, m = markov_params[0].shape
    H0 = np.block([[markov_params[i + j] for j in range(n_hankel)] for i in range(n_hankel)])
    H1 = np.block([[markov_params[i + j + 1] for j in range(n_hankel)] for i in range(n_hankel)])
    U, S, Vt = np.linalg.svd(H0, full_matrices=False)
    Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
    Sr_sqrt = np.sqrt(Sr)
    Sr_inv_sqrt = 1.0 / Sr_sqrt
    Ar = np.diag(Sr_inv_sqrt) @ Ur.T @ H1 @ Vtr.T @ np.diag(Sr_inv_sqrt)
    Br = np.diag(Sr_sqrt) @ Vtr[:, :m]
    Cr = Ur[:p, :] @ np.diag(Sr_sqrt)
    return Ar, Br, Cr, S


def to_partial_output_normal_form(Ar, Br, Cr, d_x):
    """balanced_truncation.py's to_output_normal_form only handles Cr
    square (r == d_x) - this generalizes to r >= d_x: find a similarity
    transform T such that the FIRST d_x coordinates of the new state read
    out directly as the physical output (Cr' = [I_dx | 0]) and the
    remaining r-d_x coordinates are whatever basis completes it, exactly
    matching M3's own [x-role; s-role] augmented-state structure but with
    an (r-d_x)-dim "internal" block instead of 1024. T^-1 = [Cr^+ | N]
    where Cr^+ is Cr's right pseudo-inverse (Cr @ Cr^+ = I) and N's
    columns span Cr's null space (via SVD) - standard construction."""
    r = Ar.shape[0]
    if r == d_x:
        cond = np.linalg.cond(Cr)
        Cr_inv = np.linalg.inv(Cr)
        return Cr @ Ar @ Cr_inv, Cr @ Br, cond
    Cr_pinv = Cr.T @ np.linalg.inv(Cr @ Cr.T)  # (r, d_x), Cr @ Cr_pinv = I
    _, _, Vt_full = np.linalg.svd(Cr, full_matrices=True)  # Vt_full: (r, r)
    N = Vt_full[d_x:, :].T  # (r, r-d_x), spans null(Cr)
    T_inv = np.hstack([Cr_pinv, N])  # (r, r)
    cond = np.linalg.cond(T_inv)
    T = np.linalg.inv(T_inv)
    A_final = T @ Ar @ T_inv
    B_final = T @ Br
    return A_final, B_final, cond


def verify_markov_match(A, B, C, true_markov):
    max_err = 0.0
    M = np.eye(A.shape[0])
    for h in range(len(true_markov)):
        recon = C @ M @ B
        max_err = max(max_err, float(np.max(np.abs(recon - true_markov[h]))))
        M = A @ M
    return max_err


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


def get_x0_batch(case):
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


def main() -> None:
    rows = []
    spectrum_rows = []
    H = 2 * N_HANKEL

    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        true_markov = [np.linalg.matrix_power(A_true, h) @ B_true for h in range(H)]
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for variant in ["fullM3", "M0_S4", "M1"]:
            for seed in range(N_SEEDS):
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A_full, B_full, C_full = data["A"], data["B"], data["C"]

                if variant == "M1":
                    # already 6-dim; "truncation" is a no-op, report directly
                    target_fidelity = verify_markov_match(A_full, B_full, C_full, true_markov)
                    r_chosen = D_X
                    A_r, B_r = A_full, B_full
                    hsv = np.array([])
                    achieved_fidelity = target_fidelity
                else:
                    m3_markov = markov_from_augmented(A_full, B_full, C_full, H)
                    target_fidelity = verify_markov_match(A_full, B_full, C_full, true_markov)

                    r_chosen, A_r, B_r, achieved_fidelity, hsv = None, None, None, None, None
                    for r in R_SWEEP:
                        Ar, Br, Cr, hsv_r = era(m3_markov, r, N_HANKEL)
                        cond = np.linalg.cond(Cr)
                        if cond > 1e8:
                            continue
                        A_final, B_final, _ = to_partial_output_normal_form(Ar, Br, Cr, D_X)
                        C_r_eye = np.eye(D_X, r)  # [I_6 | 0] when r>D_X, plain I_6 when r==D_X
                        err_vs_true = verify_markov_match(A_final, B_final, C_r_eye, true_markov)
                        if hsv is None:
                            hsv = hsv_r  # record spectrum at the FIRST (smallest r) attempt for reporting
                        if err_vs_true <= max(target_fidelity, 1e-9):
                            r_chosen, A_r, B_r, achieved_fidelity = r, A_final, B_final, err_vs_true
                            break
                    if r_chosen is None:
                        rows.append({"variant": variant, "case": case, "seed": seed, "r_chosen": None,
                                     "target_fidelity": target_fidelity, "achieved_fidelity": None,
                                     "ratio_transfer": None, "rho_transfer": None, "finite": False,
                                     "note": "no r in [6,60] reached target fidelity"})
                        print(f"  {variant} case{case} seed{seed}: NO r in [6,60] reached "
                              f"target_fidelity={target_fidelity:.3e}")
                        continue

                n_r = A_r.shape[0]
                Q = Q_X * np.eye(n_r) if variant == "M1" else np.block(
                    [[Q_X * np.eye(D_X), np.zeros((D_X, n_r - D_X))], [np.zeros((n_r - D_X, n_r))]])
                R = R_U * np.eye(D_U)
                try:
                    P = solve_discrete_are(A_r, B_r, Q, R)
                    K_lqr = np.linalg.solve(R + B_r.T @ P @ B_r, B_r.T @ P @ A_r)
                except np.linalg.LinAlgError:
                    rows.append({"variant": variant, "case": case, "seed": seed, "r_chosen": r_chosen,
                                 "target_fidelity": target_fidelity, "achieved_fidelity": achieved_fidelity,
                                 "ratio_transfer": None, "rho_transfer": None, "finite": False,
                                 "note": "DARE failed"})
                    continue

                if variant == "M1":
                    b, rho = robust_margin_and_rho(A_true, B_true, -K_lqr)
                    ratio, finite = simulate_cost(A_true, B_true, -K_lqr, x0_eval, oracle_cost)
                else:
                    n_s_r = n_r - D_X
                    K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
                    Asx_r, Ass_r = A_r[D_X:, :D_X], A_r[D_X:, D_X:]
                    Bs_r = B_r[D_X:, :]
                    A_open = np.block([[A_true, np.zeros((D_X, n_s_r))], [Asx_r, Ass_r]])
                    B_open = np.vstack([B_true, Bs_r])
                    K_direct = np.hstack([-K_x, -K_s])
                    b, rho = robust_margin_and_rho(A_open, B_open, K_direct)
                    ratio, finite = simulate_cost(A_open, B_open, K_direct, x0_eval, oracle_cost)

                rows.append({"variant": variant, "case": case, "seed": seed, "r_chosen": r_chosen,
                             "target_fidelity": target_fidelity, "achieved_fidelity": achieved_fidelity,
                             "b": b, "rho_transfer": rho, "ratio_transfer": ratio, "finite": finite, "note": ""})
                print(f"  {variant} case{case} seed{seed}: r={r_chosen}  target_fid={target_fidelity:.3e}  "
                      f"achieved_fid={achieved_fidelity:.3e}  rho={rho:.6f}  ratio={ratio:.4e}  finite={finite}")

                if hsv is not None:
                    for i, sv in enumerate(hsv[:20]):
                        spectrum_rows.append({"variant": variant, "case": case, "seed": seed,
                                               "sv_index": i, "sv_value": float(sv)})

    for name, data_rows in [("fidelity_matched_truncation.csv", rows),
                             ("fidelity_matched_truncation_spectrum.csv", spectrum_rows)]:
        out_path = DOCS / name
        header = sorted({k for r in data_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in data_rows]
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path} ({len(data_rows)} rows)")

    print("\n=== SUMMARY ===")
    for variant in ["fullM3", "M0_S4", "M1"]:
        these = [r for r in rows if r["variant"] == variant and r["ratio_transfer"] is not None]
        n_success = sum(1 for r in these if r["finite"] and r["ratio_transfer"] < 100.0)
        ratios = [r["ratio_transfer"] for r in these if r["finite"]]
        r_chosens = [r["r_chosen"] for r in these if r["r_chosen"] is not None]
        med_ratio = np.median(ratios) if ratios else float("nan")
        med_r = np.median(r_chosens) if r_chosens else float("nan")
        print(f"  {variant:8s}: {n_success}/{len(these)} transfer-stable  median r_chosen={med_r}  "
              f"median ratio={med_ratio:.4e}")


if __name__ == "__main__":
    main()
