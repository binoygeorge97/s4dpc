"""TASK B (user, 2026-08-18): is this S4-specific, or does it generalize?
Five consecutive negatives (warm-start, k-step identification, horizon
extension, dimension reduction being at best a partial mitigation, and the
decisive LQR-transfer failure itself) is a pattern worth being suspicious
of - this sets what the paper is allowed to claim, not scaffolding.

A plain, fully unconstrained linear state-space model - dense A/B/C/D, no
per-channel diagonal structure, no HiPPO initialization, no Lambda_re clip,
no complex parameterization at all - trained by gradient descent on the
IDENTICAL teacher-forced one-step MSE loss and the SAME APRBS data as every
M3 run this session. n_h=64 real hidden states (D_X+n_h=70 augmented total -
matching tools/dimension_sweep.py's d64 S4 config exactly, for a direct,
dimension-matched comparison).

Model: x_{k+1} = D @ [x_k; u_k] + C @ h_k,  h_{k+1} = B @ [x_k; u_k] + A @ h_k.
Deliberately the SAME [x;u] input convention S4 uses (D_INPUT=9), so the
augmented (physical + hidden) realization assembles directly from the
trained (A, B, C, D) blocks - no jacfwd extraction needed, this model is
linear/exact by construction the same way M3 is, just without any special
structure imposed on how it gets there.

Runs the identical PBH-unstable-count + LQR-synthesis + observer-transfer
pipeline as every other checkpoint this session, with M1 and M0_S4 as the
same controls (already on record, not re-run).

    python tools/linear_ssm_baseline.py [--smoke]
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
import optax
from flax import nnx
from scipy.linalg import solve_discrete_are

from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.systems import get_discrete_matrices

CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
EPOCHS = 40000
LEARNING_RATE = 1e-3
N_H = 64  # hidden dim - matches dimension_sweep.py's d64 (D_X+64=70 total)
D_X, D_U = 6, 3
L_MAX = 100
DT = 0.01
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_X0_RANGE = 5.0
DOCS_DIR = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS_DIR / "linear_ssm_baseline"


class LinearSSM(nnx.Module):
    """Fully unconstrained linear SSM - no diagonal/DPLR structure, no
    HiPPO init, no stability-oriented clip anywhere. Inputs are [x; u]
    (D_INPUT=9,) per step, matching S4's own convention exactly, so the
    trained (A,B,C,D) assemble directly into an augmented [x;h] realization
    the same shape as M3's (Abar,Bbar) - see main()."""

    def __init__(self, n_h: int, d_input: int, d_output: int, rngs: nnx.Rngs):
        k1, k2, k3, k4 = jax.random.split(rngs.params(), 4)
        self.n_h = n_h
        self.A = nnx.Param(jax.random.normal(k1, (n_h, n_h), dtype=jnp.float64) / jnp.sqrt(n_h))
        self.B = nnx.Param(jax.random.normal(k2, (n_h, d_input), dtype=jnp.float64) / jnp.sqrt(d_input))
        self.C = nnx.Param(jax.random.normal(k3, (d_output, n_h), dtype=jnp.float64) / jnp.sqrt(n_h))
        self.D = nnx.Param(jax.random.normal(k4, (d_output, d_input), dtype=jnp.float64) * 0.01)

    def init_state(self) -> jax.Array:
        return jnp.zeros((self.n_h,), dtype=jnp.float64)

    def __call__(self, inputs: jax.Array, h0: jax.Array) -> tuple[jax.Array, jax.Array]:
        def step(h, u):
            x_pred = self.C.value @ h + self.D.value @ u
            h_next = self.A.value @ h + self.B.value @ u
            return h_next, x_pred

        h_final, outputs = jax.lax.scan(step, h0, inputs)
        return outputs, h_final


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
    sys = control.StateSpace(Acl, Bcl, Ccl, Dcl, dt=1)
    try:
        gamma = control.norm(sys, p="inf", print_warning=False)
        b = 1.0 / gamma
    except Exception:
        b = None
    return (np.nan if b is None else b), rho


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


def train_ensemble(keys, inputs_grid, targets_grid, epochs):
    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return LinearSSM(N_H, D_INPUT, D_OUTPUT, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = nnx.Optimizer(ensemble, optax.adamw(LEARNING_RATE, weight_decay=0.0), wrt=nnx.Param)

    def loss_fn(model):
        graphdef, params = nnx.split(model, nnx.Param)

        def single_member(p, inp, tgt):
            m = nnx.merge(graphdef, p)
            h0 = m.init_state()
            pred, _ = m(inp, h0)
            return jnp.mean((pred - tgt) ** 2)

        losses = jax.vmap(single_member)(params, inputs_grid, targets_grid)
        return jnp.mean(losses), losses

    @nnx.jit
    def train_step(ens, opt):
        (_, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
        opt.update(ens, grads)
        return per_member

    for epoch in range(epochs):
        per_member = train_step(ensemble, optimizer)
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    epoch {epoch + 1}/{epochs}  mean_mse={float(jnp.mean(per_member)):.4e}  "
                  f"max_mse={float(jnp.max(per_member)):.4e}")

    _, final_mse = loss_fn(ensemble)
    return nnx.state(ensemble, nnx.Param), final_mse


def main() -> None:
    smoke = "--smoke" in sys.argv
    cases = [3] if smoke else CASES
    n_seeds = 2 if smoke else N_SEEDS
    epochs = 2000 if smoke else EPOCHS
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}  smoke={smoke}  cases={cases}  "
          f"n_seeds={n_seeds}  epochs={epochs}  N_H={N_H}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = {c: case_data(c, L_MAX, -10.0, 10.0) for c in cases}
    flat_cases, flat_seeds = [], []
    for c in cases:
        for s in range(n_seeds):
            flat_cases.append(c)
            flat_seeds.append(s)
    inputs_grid = jnp.stack([data[c][0] for c in flat_cases])
    targets_grid = jnp.stack([data[c][1] for c in flat_cases])
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), c) for c, s in zip(flat_cases, flat_seeds)])

    t0 = time.time()
    ens_state, final_mse = train_ensemble(keys, inputs_grid, targets_grid, epochs)
    print(f"  training wall time: {time.time() - t0:.1f}s")

    diverged = {i for i in range(len(flat_cases)) if float(final_mse[i]) > 10.0}
    if diverged:
        print(f"  WARNING: {len(diverged)}/{len(flat_cases)} members diverged (teacher_mse > 10), excluded")

    # per-member extraction: the established pattern this session (e.g.
    # tools/nu_gap_export.py) - tree_map-slice the ensemble state, nnx.update
    # a template instance, read arrays off the CONCRETE module. Not direct
    # dict-style indexing into the State object, which isn't this project's
    # proven access pattern.
    template = LinearSSM(N_H, D_INPUT, D_OUTPUT, rngs=nnx.Rngs(0))

    true_AB, oracle_costs, x0_batches = {}, {}, {}
    for case in cases:
        A_d, B_d = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        true_AB[case] = (A_d, B_d)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        x0_batch = np.asarray(jax.random.uniform(eval_key, (100, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE),
                               dtype=np.float64)
        x0_batches[case] = x0_batch
        K_oracle = solve_dlqr(A_d, B_d, Q_X * np.eye(D_X), R_U * np.eye(D_U))
        xh, uh = rollout_lqr_true(A_d, B_d, K_oracle, x0_batch, 200)
        oracle_costs[case] = true_quadratic_cost(xh, uh, Q_X, R_U, Q_F)

    rows = []
    for i, (case, seed) in enumerate(zip(flat_cases, flat_seeds)):
        if i in diverged:
            continue
        member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
        nnx.update(template, member_state)
        A_ssm = np.asarray(template.A.value)
        B_ssm = np.asarray(template.B.value)
        C_ssm = np.asarray(template.C.value)
        D_ssm = np.asarray(template.D.value)

        Axx, Bx = D_ssm[:, :D_X], D_ssm[:, D_X:]
        Axs = C_ssm
        Asx, Bs = B_ssm[:, :D_X], B_ssm[:, D_X:]
        Ass = A_ssm

        Abar = np.block([[Axx, Axs], [Asx, Ass]])
        Bbar = np.vstack([Bx, Bs])
        n_total = Abar.shape[0]

        eigvals = np.linalg.eigvals(Abar)
        n_unstable = int(np.sum(np.abs(eigvals) > 1.0))

        A_true, B_true = true_AB[case]
        C_out = np.concatenate([np.eye(D_X), np.zeros((D_X, n_total - D_X))], axis=1)
        Q = C_out.T @ (Q_X * np.eye(D_X)) @ C_out
        R = R_U * np.eye(D_U)
        K_lqr = solve_dlqr(Abar, Bbar, Q, R)
        K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
        b_own, rho_own = robust_margin_and_rho(Abar, Bbar, -K_lqr)

        n_s = n_total - D_X
        Asx2, Ass2, Bs2 = Abar[D_X:, :D_X], Abar[D_X:, D_X:], Bbar[D_X:, :]
        A_open = np.block([[A_true, np.zeros((D_X, n_s))], [Asx2, Ass2]])
        B_open = np.vstack([B_true, Bs2])
        K_direct = np.hstack([-K_x, -K_s])
        b_transfer, rho_transfer = robust_margin_and_rho(A_open, B_open, K_direct)
        ratio_transfer, finite = simulate_transfer_cost(
            A_true, B_true, A_open, B_open, K_direct, x0_batches[case], oracle_costs[case]
        )

        teacher_mse = float(final_mse[i])
        print(f"  [case{case}/seed{seed}] teacher_mse={teacher_mse:.4e}  n_unstable={n_unstable}/{n_total}  "
              f"b_own={b_own:.4f}  b_transfer={b_transfer:.4f}  rho_transfer={rho_transfer:.6f}  "
              f"ratio_transfer={ratio_transfer:.4e}")

        np.savez(EXPORT_DIR / f"case{case}_seed{seed}.npz", Abar=Abar, Bbar=Bbar, K_lqr=K_lqr,
                 teacher_mse=teacher_mse, n_unstable=n_unstable, b_own=b_own, rho_own=rho_own,
                 b_transfer=b_transfer, rho_transfer=rho_transfer, ratio_transfer=ratio_transfer, finite=finite)
        rows.append({"case": case, "seed": seed, "teacher_mse": teacher_mse, "n_unstable": n_unstable,
                     "b_own": b_own, "rho_own": rho_own, "b_transfer": b_transfer, "rho_transfer": rho_transfer,
                     "ratio_transfer": ratio_transfer, "finite": finite})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    (DOCS_DIR / "linear_ssm_baseline_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'linear_ssm_baseline_summary.csv'}")

    if rows:
        n_stable = sum(1 for r in rows if r["b_transfer"] > 0)
        print(f"\nSUMMARY: {n_stable}/{len(rows)} stable when transferred to true plant  "
              f"median teacher_mse={np.median([r['teacher_mse'] for r in rows]):.4e}  "
              f"median n_unstable={np.median([r['n_unstable'] for r in rows]):.1f}  "
              f"median ratio={np.median([r['ratio_transfer'] for r in rows]):.4e}")


if __name__ == "__main__":
    main()
