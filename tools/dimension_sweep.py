"""TASK C (user, 2026-08-18): does the reality gap scale with M3's
over-parameterization? Re-identify M3 at internal dimensions ~64 and ~256
(vs the standard ~1030 - d_model=16, N=32 already on record), all 6 cases,
5 seeds each, and run the SAME PBH-unstable-count + LQR-synthesis +
observer-transfer-to-true-plant pipeline as the decisive result
(tools/lqr_transfer_to_true_plant.py) at each scale. If the gap shrinks or
vanishes at minimal dimension, over-parameterization isn't just a
description of the failure - it's a mitigation ("don't over-parameterize
your identifier" becomes actionable, not just a warning).

Dimension configs (d_model x N, real internal dim = 2*d_model*N):
    d64:   d_model=4, N=8   -> 64 real internal dims  (70 augmented total)
    d256:  d_model=8, N=16  -> 256 real internal dims (262 augmented total)
    d1030: d_model=16, N=32 -> 1024 real internal dims (existing exports,
           docs/nu_gap_export/fullM3_*.npz - NOT re-identified here)

Same identification protocol as standard M3 (teacher-forced one-step MSE,
40000 epochs) - also reports teacher_mse per checkpoint as a fairness
check: if the smaller models fit worse, a shrinking spurious-mode count
would be confounded with "just a worse fit," not "less over-parameterized."

GPU: identification only. Augmented-operator extraction (jacfwd) is cheap
at these dimensions and stays in the same jax session. Everything after
that (PBH unstable count, DARE/LQR synthesis, observer-transfer stability
and cost) is pure numpy/scipy - fast at these smaller dims (DARE cost
scales close to cubically with state dimension, so ~70/~262-dim solves
should take seconds, not the ~100s the 1030-dim ones needed).

    python tools/dimension_sweep.py
"""
from __future__ import annotations

import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.linalg import solve_discrete_are

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

DIM_CONFIGS = [("d64", 4, 8), ("d256", 8, 16)]
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
EPOCHS = 40000
N_LAYERS, L_MAX = 1, 100
D_X, D_U = 6, 3
DT = 0.01
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_X0_RANGE = 5.0
DOCS_DIR = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS_DIR / "dimension_sweep"


def make_pack_unpack(d_model: int, N: int):
    s_dim = d_model * N

    def _pack(x, s):
        return jnp.concatenate([x, s.real.ravel(), s.imag.ravel()])

    def _unpack(z):
        x = z[:D_X]
        s_re = z[D_X:D_X + s_dim].reshape(d_model, N)
        s_im = z[D_X + s_dim:].reshape(d_model, N)
        return x, s_re + 1j * s_im

    return _pack, _unpack, D_X + 2 * s_dim


def augmented_operator_generic(graphdef, params, d_model: int, N: int):
    _pack, _unpack, z_dim = make_pack_unpack(d_model, N)

    def f(z, u):
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        x_next, (s_next,) = m(jnp.concatenate([x, u]), [s])
        return _pack(x_next, s_next)

    z0 = jnp.zeros((z_dim,), dtype=jnp.float64)
    u0 = jnp.zeros((D_U,), dtype=jnp.float64)
    Abar = jax.jacfwd(f, argnums=0)(z0, u0)
    Bbar = jax.jacfwd(f, argnums=1)(z0, u0)
    return np.asarray(Abar), np.asarray(Bbar)


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


def solve_dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


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


def simulate_transfer_cost(A_true, B_true, A_open, B_open, K_direct, x0_batch, oracle_cost, horizon_N=200):
    Acl = A_open + B_open @ K_direct
    n = Acl.shape[0]
    batch = x0_batch.shape[0]
    z = np.zeros((batch, n))
    z[:, :D_X] = x0_batch
    stage = 0.0
    for _ in range(horizon_N):
        x_k = z[:, :D_X]
        u_k = z @ K_direct.T
        stage += np.mean(np.sum(x_k ** 2, axis=-1)) * Q_X + np.mean(np.sum(u_k ** 2, axis=-1)) * R_U
        z = z @ Acl.T
        if not np.all(np.isfinite(z)):
            return float("inf"), False
    x_final = z[:, :D_X]
    terminal = np.mean(np.sum(x_final ** 2, axis=-1)) * Q_F
    cost = (stage + terminal) / horizon_N
    ratio = cost / oracle_cost if oracle_cost > 0 else float("inf")
    return ratio, bool(np.all(np.isfinite(z)))


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    true_AB, oracle_costs, x0_batches = {}, {}, {}
    for case in CASES:
        A_d, B_d = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        true_AB[case] = (A_d, B_d)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        x0_batch = np.asarray(jax.random.uniform(eval_key, (100, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE),
                               dtype=np.float64)
        x0_batches[case] = x0_batch
        K_oracle = solve_dlqr(A_d, B_d, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        xh, uh = rollout_lqr_true(A_d, B_d, K_oracle, x0_batch, 200)
        oracle_costs[case] = true_quadratic_cost(xh, uh, Q_X, R_U, Q_F)

    for label, d_model, N in DIM_CONFIGS:
        real_internal_dim = 2 * d_model * N
        print(f"\n{'=' * 20} {label}: d_model={d_model} N={N} -> {real_internal_dim} real internal dims "
              f"({D_X + real_internal_dim} augmented total) {'=' * 20}")
        t0 = time.time()
        id_rows = run_identify(
            variant="M3", cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
            d_model=d_model, N=N, n_layers=N_LAYERS, l_max=L_MAX,
        )
        print(f"  identification wall time: {time.time() - t0:.1f}s")
        diverged = {(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0}
        print(f"  diverged: {sorted(diverged)}  ({len(diverged)}/{len(id_rows)})")

        block_config = BlockConfig(d_model=d_model, N=N, l_max=L_MAX, **VARIANTS["M3"])
        graphdef, _ = nnx.split(
            StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                         decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
            nnx.Param,
        )

        for r in id_rows:
            case, seed = r["case"], r["seed"]
            if (case, seed) in diverged:
                continue
            Abar, Bbar = augmented_operator_generic(graphdef, r["param_state"], d_model, N)
            n_total = Abar.shape[0]

            eigvals_full = np.linalg.eigvals(Abar)
            n_unstable = int(np.sum(np.abs(eigvals_full) > 1.0))

            A_true, B_true = true_AB[case]
            C = np.concatenate([np.eye(D_X), np.zeros((D_X, n_total - D_X))], axis=1)
            Q = C.T @ (Q_X * np.eye(D_X)) @ C
            R = R_U * np.eye(D_U)
            t_dare = time.time()
            K_lqr = solve_dlqr(Abar, Bbar, Q, R)
            dare_time = time.time() - t_dare
            K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]

            b_own, rho_own = robust_margin_and_rho(Abar, Bbar, -K_lqr)
            # robust_margin_and_rho returns b=None when control.norm's H-inf
            # computation itself fails to converge (a real, expected edge
            # case, not a bug) - coerce to NaN HERE, once, so every
            # downstream use (print, save, CSV, summary aggregation
            # including "> 0" comparisons) is uniformly safe. NaN compares
            # False to everything in both python and numpy, so "> 0" never
            # raises, unlike None.
            b_own = np.nan if b_own is None else b_own

            n_s = n_total - D_X
            Asx, Ass, Bs = Abar[D_X:, :D_X], Abar[D_X:, D_X:], Bbar[D_X:, :]
            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])
            K_direct = np.hstack([-K_x, -K_s])
            b_transfer, rho_transfer = robust_margin_and_rho(A_open, B_open, K_direct)
            b_transfer = np.nan if b_transfer is None else b_transfer
            ratio_transfer, finite = simulate_transfer_cost(
                A_true, B_true, A_open, B_open, K_direct, x0_batches[case], oracle_costs[case]
            )

            print(f"  [{label}/case{case}/seed{seed}] n_unstable={n_unstable}/{n_total}  "
                  f"teacher_mse={r['teacher_mse']:.3e}  DARE={dare_time:.2f}s  "
                  f"b_own={b_own:.4f}  b_transfer={b_transfer:.4f}  rho_transfer={rho_transfer:.6f}  "
                  f"ratio_transfer={ratio_transfer:.4e}")

            np.savez(EXPORT_DIR / f"{label}_{case}_{seed}.npz",
                     Abar=Abar, Bbar=Bbar, K_lqr=K_lqr, n_unstable=n_unstable, n_total=n_total,
                     teacher_mse=r["teacher_mse"], b_own=b_own, rho_own=rho_own,
                     b_transfer=b_transfer, rho_transfer=rho_transfer, ratio_transfer=ratio_transfer,
                     finite=finite)
            rows.append({"label": label, "d_model": d_model, "N": N, "real_internal_dim": real_internal_dim,
                         "n_total": n_total, "case": case, "seed": seed, "n_unstable": n_unstable,
                         "teacher_mse": r["teacher_mse"], "b_own": b_own, "rho_own": rho_own,
                         "b_transfer": b_transfer, "rho_transfer": rho_transfer,
                         "ratio_transfer": ratio_transfer, "finite": finite})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    (DOCS_DIR / "dimension_sweep_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'dimension_sweep_summary.csv'}")

    print(f"\n{'=' * 20} SUMMARY: unstable-mode count and transfer outcome vs internal dimension {'=' * 20}")
    for label, d_model, N in DIM_CONFIGS:
        these = [r for r in rows if r["label"] == label]
        if not these:
            continue
        n_stable_transfer = sum(1 for r in these if r["b_transfer"] > 0)
        print(f"  {label} ({these[0]['real_internal_dim']} real internal dims): "
              f"median n_unstable={np.median([r['n_unstable'] for r in these]):.1f}  "
              f"median teacher_mse={np.median([r['teacher_mse'] for r in these]):.3e}  "
              f"{n_stable_transfer}/{len(these)} stable when transferred to true plant  "
              f"median ratio={np.median([r['ratio_transfer'] for r in these]):.4e}")


if __name__ == "__main__":
    main()
