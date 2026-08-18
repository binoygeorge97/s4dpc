"""TASK A (user, 2026-08-18): the one fix that follows from the "objective,
not architecture" reading (docs/DECISIONS.md, same date). If teacher-forced
one-step MSE cannot see a mode that is unstable but weakly expressed within
one step, add a term to the loss that CAN see it: a hinge penalty on the
augmented operator's spectral radius.

Full eigendecomposition of the (Z_DIM, Z_DIM) augmented operator every
gradient step (needed for an exact spectral-radius penalty) is both too
expensive at 40000 epochs and numerically fragile to differentiate through
for a general non-symmetric matrix with near-degenerate eigenvalues (jax's
`jnp.linalg.eig` has no reverse-mode rule at all). Uses POWER ITERATION
instead: `v <- (Abar @ v)/||Abar @ v||`, repeated, converges (or for a
dominant complex-conjugate pair, oscillates around) the dominant
eigenvalue's magnitude - computed via `jax.jvp` on the model's own step
function (a Jacobian-VECTOR product, never materializing the full (Z_DIM,
Z_DIM) matrix), so cost is ~N_POWER_ITERS extra forward passes per member
per step, not a full Jacobian. Validated locally against exact
eigendecomposition on a real M3 checkpoint before use: ~1.5-2% relative
error at 20 iterations - oscillation from a dominant complex pair, not
divergence, and plenty accurate for a training SIGNAL rather than a
precise number (the final reported n_unstable/rho below still use exact
`np.linalg.eigvals` on the trained model, never the power-iteration
estimate).

Evaluated at z=0, u=0: M3 (no norm/activation/glu) is exactly affine, so
its Jacobian - and therefore Abar and its spectral radius - is identical at
every point; this is the SAME fact `augmented_operator` already relies on.

Limitation stated up front, not discovered later: single-vector power
iteration tracks only the CURRENTLY-dominant eigenvalue. With ~8.5 (median)
unstable modes typically present, this can only push down one at a time -
after the current dominant mode is suppressed, gradient descent should
start tracking whichever is now largest, but this isn't guaranteed to
address all of them within a fixed epoch budget. Reported honestly below,
not assumed away.

After training: extracts the augmented operator EXACTLY (jacfwd, matching
every other script this session), reports teacher_mse (fairness check - if
this degrades badly relative to standard M3, the comparison is confounded),
n_unstable (exact eigendecomposition), and runs the identical LQR-synthesis
+ observer-transfer-to-true-plant pipeline as the decisive result.

    python tools/identify_stability_constrained.py [--smoke]
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

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, _build_model, case_data
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
EPOCHS = 40000
LEARNING_RATE = 1e-3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
D_X, D_U = 6, 3
S_DIM = D_MODEL * STATE_SIZE
Z_DIM = D_X + 2 * S_DIM
DT = 0.01
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
EVAL_X0_RANGE = 5.0
N_POWER_ITERS = 20
PENALTY_WEIGHT = 1.0
DOCS_DIR = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS_DIR / "stability_constrained"


def _pack(x, s):
    return jnp.concatenate([x, s.real.ravel(), s.imag.ravel()])


def _unpack(z):
    x = z[:D_X]
    s_re = z[D_X:D_X + S_DIM].reshape(D_MODEL, STATE_SIZE)
    s_im = z[D_X + S_DIM:].reshape(D_MODEL, STATE_SIZE)
    return x, s_re + 1j * s_im


def augmented_operator(graphdef, params):
    def f(z, u):
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        x_next, (s_next,) = m(jnp.concatenate([x, u]), [s])
        return _pack(x_next, s_next)

    z0 = jnp.zeros((Z_DIM,), dtype=jnp.float64)
    u0 = jnp.zeros((D_U,), dtype=jnp.float64)
    Abar = jax.jacfwd(f, argnums=0)(z0, u0)
    Bbar = jax.jacfwd(f, argnums=1)(z0, u0)
    return np.asarray(Abar), np.asarray(Bbar)


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
    b = np.nan if b is None else b
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


def train_ensemble_with_stability_penalty(block_config, keys, inputs_grid, targets_grid, epochs, penalty_weight):
    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return _build_model(block_config, N_LAYERS, key)

    ensemble = init_ensemble(keys)
    optimizer = nnx.Optimizer(ensemble, optax.adamw(LEARNING_RATE, weight_decay=0.0), wrt=nnx.Param)

    def loss_fn(model, power_key):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        power_keys = jax.random.split(power_key, inputs_grid.shape[0])

        def single_member(p, r, inp, tgt, pkey):
            m = nnx.merge(graphdef, p, r)
            states = m.init_state(N=block_config.N)
            pred, _ = m(inp, states)
            mse = jnp.mean((pred - tgt) ** 2)

            def f(z):
                x, s = _unpack(z)
                x_next, (s_next,) = m(jnp.concatenate([x, jnp.zeros((D_U,), dtype=jnp.float64)]), [s])
                return _pack(x_next, s_next)

            z0 = jnp.zeros((Z_DIM,), dtype=jnp.float64)
            v0 = jax.random.normal(pkey, (Z_DIM,), dtype=jnp.float64)
            v0 = v0 / jnp.linalg.norm(v0)

            def body(v, _):
                _, Av = jax.jvp(f, (z0,), (v,))
                return Av / (jnp.linalg.norm(Av) + 1e-12), None

            v_final, _ = jax.lax.scan(body, v0, None, length=N_POWER_ITERS)
            _, Av_final = jax.jvp(f, (z0,), (v_final,))
            rho_estimate = jnp.linalg.norm(Av_final)
            penalty = jax.nn.relu(rho_estimate - 1.0) ** 2
            return mse + penalty_weight * penalty, (mse, rho_estimate)

        losses, aux = jax.vmap(single_member)(params, rest, inputs_grid, targets_grid, power_keys)
        return jnp.mean(losses), aux

    @nnx.jit
    def train_step(ens, opt, power_key):
        (_, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens, power_key)
        opt.update(ens, grads)
        return aux

    master_key = jax.random.PRNGKey(999)
    for epoch in range(epochs):
        master_key, step_key = jax.random.split(master_key)
        mse, rho_est = train_step(ensemble, optimizer, step_key)
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    epoch {epoch + 1}/{epochs}  mean_mse={float(jnp.mean(mse)):.4e}  "
                  f"mean_rho_est={float(jnp.mean(rho_est)):.4f}  max_rho_est={float(jnp.max(rho_est)):.4f}")

    _, (final_mse, final_rho_est) = loss_fn(ensemble, master_key)
    return nnx.state(ensemble, nnx.Param), final_mse, final_rho_est


def main() -> None:
    smoke = "--smoke" in sys.argv
    cases = [3] if smoke else CASES
    n_seeds = 2 if smoke else N_SEEDS
    epochs = 2000 if smoke else EPOCHS
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}  smoke={smoke}  cases={cases}  "
          f"n_seeds={n_seeds}  epochs={epochs}  N_POWER_ITERS={N_POWER_ITERS}  PENALTY_WEIGHT={PENALTY_WEIGHT}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
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
    ens_state, final_mse, final_rho_est = train_ensemble_with_stability_penalty(
        block_config, keys, inputs_grid, targets_grid, epochs, PENALTY_WEIGHT
    )
    print(f"  training wall time: {time.time() - t0:.1f}s")

    graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )

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
        member_params = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
        Abar, Bbar = augmented_operator(graphdef, member_params)
        eigvals = np.linalg.eigvals(Abar)
        n_unstable = int(np.sum(np.abs(eigvals) > 1.0))

        A_true, B_true = true_AB[case]
        C = np.concatenate([np.eye(D_X), np.zeros((D_X, Z_DIM - D_X))], axis=1)
        Q = C.T @ (Q_X * np.eye(D_X)) @ C
        R = R_U * np.eye(D_U)
        K_lqr = solve_dlqr(Abar, Bbar, Q, R)
        K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
        b_own, rho_own = robust_margin_and_rho(Abar, Bbar, -K_lqr)

        Asx, Ass, Bs = Abar[D_X:, :D_X], Abar[D_X:, D_X:], Bbar[D_X:, :]
        A_open = np.block([[A_true, np.zeros((D_X, Z_DIM - D_X))], [Asx, Ass]])
        B_open = np.vstack([B_true, Bs])
        K_direct = np.hstack([-K_x, -K_s])
        b_transfer, rho_transfer = robust_margin_and_rho(A_open, B_open, K_direct)
        ratio_transfer, finite = simulate_transfer_cost(
            A_true, B_true, A_open, B_open, K_direct, x0_batches[case], oracle_costs[case]
        )

        teacher_mse = float(final_mse[i])
        print(f"  [case{case}/seed{seed}] teacher_mse={teacher_mse:.4e}  n_unstable={n_unstable}/{Z_DIM}  "
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
    (DOCS_DIR / "stability_constrained_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'stability_constrained_summary.csv'}")

    n_stable = sum(1 for r in rows if r["b_transfer"] > 0)
    print(f"\nSUMMARY: {n_stable}/{len(rows)} stable when transferred to true plant  "
          f"median teacher_mse={np.median([r['teacher_mse'] for r in rows]):.4e}  "
          f"median n_unstable={np.median([r['n_unstable'] for r in rows]):.1f}  "
          f"median ratio={np.median([r['ratio_transfer'] for r in rows]):.4e}")


if __name__ == "__main__":
    main()
