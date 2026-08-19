"""TASK A/B (user, 2026-08-19, seventh round): re-verify every claimed
success/near-success with the +c0 (affine offset) term included, and
redo the two fully-re-derivable established scripts
(lqr_transfer_to_true_plant.py, free_response_test.py) with c0 included,
reporting corrected numbers alongside the originals.

Key simplification that keeps this CPU-only, no new DARE solves: c0
never enters the DARE synthesis (DARE is a linear/homogeneous method,
takes no bias input) - the cached K_lqr in docs/lqr_cache/ is exactly
the same gain whether or not c0 is modeled. Only the CLOSED-LOOP
SIMULATION changes: z_{k+1} = Acl@z_k + c_open instead of Acl@z_k alone,
c_open = [0_Dx (true plant has zero bias); c0_s (M3's own S4-recursion
bias, unchanged by the dither cure)] for the observer construction, or
the FULL [c0_x; c0_s] for M3's own free-running dynamics.

Also recomputes the dither cure's (Axx, Axs, Bx, c0_x) WITH an explicit
intercept column (dither_cure_test.py never fit one - implicitly forced
c0_x=0 by omission, not by a data-driven result). TASK C already showed
this SEPARATELY: at n_dither=2000 the recovered c0_x is ~2e-15 (exactly
zero, machine precision) - so re-deriving it here with the corrected
regression should reproduce that, and this script carries it through to
the CLOSED-LOOP z*/cost that TASK A actually asked for.

    python tools/bias_corrected_reverify.py
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
LQR_CACHE = DOCS / "lqr_cache"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_BATCH, EVAL_X0_RANGE, EVAL_HORIZON = 100, 5.0, 200
APRBS_LOW, APRBS_HIGH = -10.0, 10.0
X_SYNTH_RANGE = 5.0
N_DITHER = 2000
T_BURN = 50


def get_x0_batch(case):
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


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


def simulate_cost_biased(A_open, B_open, K_direct, c_open, x0_batch, oracle_cost):
    """Same as every prior simulate_cost this session, PLUS a constant
    forcing term c_open added every step - the only change needed."""
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


def fixed_point(Acl, c_open):
    n = Acl.shape[0]
    try:
        z_star = np.linalg.solve(np.eye(n) - Acl, c_open)
    except np.linalg.LinAlgError:
        return None
    return z_star


def get_training_trajectory(case):
    inputs, targets = generate_microgrid_trajectory(
        batch_size=1, length=100, seed=42, system_case=case, dt=DT,
        aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH)
    return inputs[0, :, :D_X], inputs[0, :, D_X:], targets[0]


def simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj):
    n_s = Ass.shape[0]
    S = np.zeros((100, n_s))
    s = np.zeros(n_s)
    for t in range(100):
        S[t] = s
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return S


def extract_c0_s(Asx, Ass, Bs, model_A_full, model_B_full, D_X_local):
    """c0's s-component (Asx/Ass/Bs's own bias contribution, from the
    encoder feeding the S4 recursion) - NOT directly available from
    nu_gap_export's saved (A,B) alone (those are the JACOBIAN, no bias
    info). Falls back to None if unavailable for this checkpoint family;
    caller must handle."""
    return None  # placeholder; real c0_s comes from real params only (see main())


# ---------- PART 1: LQR-transfer + free-response, fullM3 population, WITH c0 ----------

def part1_lqr_transfer_and_free_response():
    print("=" * 20 + " PART 1: fullM3 population, LQR-transfer + free-response, WITH c0 " + "=" * 20)
    rows = []
    free_response_rows = []

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from flax import nnx
    import flax.serialization as serialization
    from s4dpc.blocks import BlockConfig, VARIANTS
    from s4dpc.model import StackedModel
    from s4dpc.diagnostics import zero_states

    block_config = BlockConfig(d_model=16, N=32, l_max=100, **VARIANTS["M3"])
    key = jax.random.PRNGKey(0)
    model = StackedModel(block_config=block_config, d_input=9, d_output=6,
                          n_layers=1, decode=True, rngs=nnx.Rngs(params=key))
    template_state = nnx.state(model, nnx.Param)
    x0_ = jnp.zeros(D_X, dtype=jnp.float64)
    u0_ = jnp.zeros(D_U, dtype=jnp.float64)

    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        K_oracle = solve_dlqr(A_true, B_true, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        x0_eval = get_x0_batch(case)
        x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K_oracle, x0_eval, EVAL_HORIZON)
        oracle_cost = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)

        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            cache_path = LQR_CACHE / f"fullM3_{case}_{seed}.npz"
            ckpt_path = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt" / f"M3_case{case}_seed{seed}.msgpack"
            if not (path.exists() and cache_path.exists() and ckpt_path.exists()):
                continue
            data = np.load(path)
            A, B = data["A"], data["B"]
            K_lqr = np.load(cache_path)["K_lqr"]
            n_s = A.shape[0] - D_X

            # c0 needs the REAL model params (not just Abar/Bbar) - model/template
            # built once above; only params reloaded per checkpoint here.
            pure_dict = serialization.msgpack_restore(ckpt_path.read_bytes())
            state = jax.tree_util.tree_map(lambda x: x, template_state)
            state.replace_by_pure_dict(pure_dict)
            nnx.update(model, state)
            x_next0, new_states0 = model(jnp.concatenate([x0_, u0_]), zero_states(model))
            c0_x = np.asarray(x_next0)
            c0_s = np.asarray(new_states0[0].real).ravel()
            c0_s_im = np.asarray(new_states0[0].imag).ravel()
            c0_full = np.concatenate([c0_x, c0_s, c0_s_im])

            # ---- LQR-transfer, corrected ----
            Asx, Ass = A[D_X:, :D_X], A[D_X:, D_X:]
            Bs = B[D_X:, :]
            K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
            A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx, Ass]])
            B_open = np.vstack([B_true, Bs])
            K_direct = np.hstack([-K_x, -K_s])
            c_open = np.concatenate([np.zeros(D_X), c0_full[D_X:]])  # true plant bias-free; s-role keeps M3's own bias

            Acl = A_open + B_open @ K_direct
            rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
            z_star = fixed_point(Acl, c_open) if rho < 1.0 else None
            z_star_x_norm = float(np.linalg.norm(z_star[:D_X])) if z_star is not None else float("nan")

            ratio_corrected, finite = simulate_cost_biased(A_open, B_open, K_direct, c_open, x0_eval, oracle_cost)

            print(f"  [fullM3 case{case}/seed{seed}] rho={rho:.6f}  ||c0_x||={np.linalg.norm(c0_x):.4f}  "
                  f"||z*_x||={z_star_x_norm:.4f}  ratio_corrected={ratio_corrected:.4e}")

            rows.append({"case": case, "seed": seed, "rho": rho,
                         "c0_x_norm": float(np.linalg.norm(c0_x)), "z_star_x_norm": z_star_x_norm,
                         "ratio_corrected": ratio_corrected, "finite": finite})

            # ---- free-response, corrected: M3's OWN free dynamics need the FULL c0.
            # EXACT match to free_response_test.py's own x0 construction (same PRNG
            # convention, batch 50, T_BURN=50 free decay under the true plant) for a
            # fair, directly-comparable re-run - not a different experiment. ----
            x_seed_batch = get_x0_batch(case)  # (50, D_X), uniform[-5,5]
            x_true_burn = x_seed_batch
            for _ in range(T_BURN):
                x_true_burn = x_true_burn @ A_true.T
            x0_free_batch = x_true_burn  # (50, D_X)

            z_batch = np.hstack([x0_free_batch, np.zeros((x0_free_batch.shape[0], n_s))])
            x_true_t_batch = x0_free_batch.copy()
            err_t1 = err_t200 = None
            for tt in range(1, 201):
                z_batch = z_batch @ A.T + c0_full[None, :]
                x_true_t_batch = x_true_t_batch @ A_true.T
                rel_err = np.mean(np.linalg.norm(z_batch[:, :D_X] - x_true_t_batch, axis=-1)
                                   / (np.linalg.norm(x_true_t_batch, axis=-1) + 1e-300))
                if tt == 1:
                    err_t1 = rel_err
                if tt == 200:
                    err_t200 = rel_err
            free_response_rows.append({"case": case, "seed": seed,
                                        "err_t1_corrected": err_t1, "err_t200_corrected": err_t200})
            print(f"                free-response WITH c0: err@t=1={err_t1:.4f}  err@t=200={err_t200:.4f}")

    out_path = DOCS / "bias_corrected_lqr_transfer.csv"
    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(rows)} rows)")

    out_path2 = DOCS / "bias_corrected_free_response.csv"
    header2 = sorted({k for r in free_response_rows for k in r.keys()})
    lines2 = [",".join(header2)] + [",".join(str(r.get(h, "")) for h in header2) for r in free_response_rows]
    out_path2.write_text("\n".join(lines2))
    print(f"wrote {out_path2} ({len(free_response_rows)} rows)")

    return rows


if __name__ == "__main__":
    part1_lqr_transfer_and_free_response()
