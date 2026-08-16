"""Falsifiable prediction from docs/DECISIONS.md's horizon-blindness mechanism
entry (2026-08-15): tools/mode_contribution_vs_horizon.py showed M3 case 3's
dominant unstable closed-loop mode contributes negligibly to the DPC cost
early in the curriculum but grows to dominate it entirely by N=200 (3.2e7)
and beyond (1.5e85 at N=2000) - if that mechanism is real, training LONGER
(more horizon, so the mode's cost is no longer negligible during training
itself) should give the optimizer a real chance to fix it. If it doesn't,
either the model can't represent a fix (capacity), or training resists moving
there regardless of gradient pressure (optimization landscape).

Trains M3 case 3 (5 seeds) through the STANDARD curriculum first (baseline,
reproduces the known ~300-700000x-scale failure and b=0 result), snapshots
that state, then continues training the SAME ensemble for one more phase at
N=1000 (2000 epochs, a FRESH cosine LR schedule sized to those 2000 epochs -
reusing the baseline's already-decayed-to-~0 schedule would confound "more
training" with "training at a near-zero learning rate"). Reports, for both
snapshots: DPC cost ratio to oracle, closed-loop spectral radius, count of
genuinely unstable (|lambda|>1) eigenvalues, and b (robust margin) - same
metrics as docs/nu_gap_analysis.csv, computed inline (pure numpy on the
extracted matrices) so this kernel is self-contained.

    python tools/test_n1000_case3.py
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

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
from m3_spurious_modes import D_MODEL, STATE_SIZE, N_LAYERS, L_MAX, D_X, D_U, augmented_operator  # noqa: E402

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import (
    BoundedGRUController, init_batched_state, make_controller_optimizer, rollout_learned,
)  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

CASE = 3
N_SEEDS = 5
DT = 0.01
DOCS_DIR = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS_DIR / "n1000_case3"
EXTENDED_N = 1000
EXTENDED_EPOCHS = 2000
COUT = np.concatenate([np.eye(D_X), np.zeros((D_X, 1030 - D_X))], axis=1)


def closed_loop_diagnostics(A: np.ndarray, B: np.ndarray, C: np.ndarray, K_eff: np.ndarray) -> dict:
    n = A.shape[0]
    K_pad = K_eff if n == D_X else np.hstack([K_eff, np.zeros((D_U, n - D_X))])
    Acl = A + B @ K_pad
    eig = np.linalg.eigvals(Acl)
    rho = float(np.max(np.abs(eig)))
    n_unstable = int(np.sum(np.abs(eig) > 1.0))
    if rho >= 1.0:
        b = 0.0
    else:
        import control
        n_, m_ = A.shape[0], B.shape[1]
        Bcl = np.hstack([B @ K_pad, B])
        Ccl = np.vstack([K_pad, np.eye(n_)])
        Dcl = np.vstack([np.hstack([K_pad, np.zeros((m_, m_))]), np.hstack([np.eye(n_), np.zeros((n_, m_))])])
        sys = control.StateSpace(Acl, Bcl, Ccl, Dcl, dt=1)
        try:
            gamma = control.norm(sys, p="inf", print_warning=False)
            b = 1.0 / gamma
        except Exception:
            b = None
    return {"rho": rho, "n_unstable": n_unstable, "b": b}


def k_eff_from_controller(controller_state, max_action: float) -> np.ndarray:
    controller = BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)

    def u_of_x(x):
        h0 = jnp.zeros((co.HIDDEN_DIM,))
        _, u = controller(h0, x)
        return u

    return np.asarray(jax.jacfwd(u_of_x)(jnp.zeros((co.D_X,))))


def evaluate_and_report(label, member_states, max_action, A_true, B_true, oracle_cost, eval_key,
                         m3_graphdef, m3_params_by_seed):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    rows = []
    for seed in range(N_SEEDS):
        member_state = member_states[seed]
        K_eff = k_eff_from_controller(member_state, max_action)
        Abar, Bbar = augmented_operator(m3_graphdef, m3_params_by_seed[seed])
        diag = closed_loop_diagnostics(Abar, Bbar, COUT, K_eff)

        controller = BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
        nnx.update(controller, member_state)
        result = co._evaluate(controller, A_true, B_true, eval_key)
        ratio = result["cost"] / oracle_cost if oracle_cost > 0 else float("inf")

        print(f"  seed{seed}: ratio={ratio:.4e}  finite={result['finite']}  "
              f"closed_loop_rho={diag['rho']:.6f}  n_unstable_eig={diag['n_unstable']}  b={diag['b']}")
        rows.append({"label": label, "seed": seed, "ratio": ratio, "finite": result["finite"],
                     "rho": diag["rho"], "n_unstable": diag["n_unstable"], "b": diag["b"]})

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(EXPORT_DIR / f"{label}_{seed}.npz", A=Abar, B=Bbar, C=COUT, K_eff=K_eff,
                 ratio=ratio, finite=result["finite"], rho=diag["rho"],
                 n_unstable=diag["n_unstable"], b=diag["b"] if diag["b"] is not None else np.nan)
    return rows


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CASE={CASE}  N_SEEDS={N_SEEDS}  EXTENDED_N={EXTENDED_N}  EXTENDED_EPOCHS={EXTENDED_EPOCHS}")

    A_d, B_d = get_discrete_matrices(DT, CASE)
    A_true, B_true = np.asarray(A_d), np.asarray(B_d)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    Q = co.Q_X * np.eye(A_true.shape[0])
    R = co.R_U * np.eye(B_true.shape[1])
    K_lqr = co.solve_dlqr(A_true, B_true, Q, R)
    x0_eval_np = np.asarray(
        jax.random.uniform(eval_key, (co.EVAL_BATCH, A_true.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
    )
    x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_true, B_true, K_lqr, x0_eval_np, co.EVAL_HORIZON)
    oracle_cost = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)
    max_action = co.CASE_MAX_ACTION[CASE]

    print(f"\n{'=' * 20} identifying M3 case {CASE}, {N_SEEDS} seeds {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant="M3", cases=[CASE], n_seeds=N_SEEDS, epochs=40000,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")
    for r in id_rows:
        print(f"    seed{r['seed']}: teacher_mse={r['teacher_mse']:.3e}")
    diverged = {r["seed"] for r in id_rows if r["teacher_mse"] > 10.0}
    m3_params_by_seed = {r["seed"]: r["param_state"] for r in id_rows if r["seed"] not in diverged}
    members = sorted(m3_params_by_seed.keys())
    if diverged:
        print(f"  WARNING: seeds {sorted(diverged)} diverged during identification, excluded")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    m3_graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    surrogate_params_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[m3_params_by_seed[s] for s in members])

    x0_list, key_list = [], []
    for seed in members:
        init_key = jax.random.fold_in(jax.random.PRNGKey(seed), CASE)
        x0_key = jax.random.fold_in(init_key, 999)
        x0 = jax.random.uniform(
            x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE, dtype=jnp.float64
        )
        x0_list.append(x0)
        key_list.append(init_key)
    x0_batch, keys = jnp.stack(x0_list), jnp.stack(key_list)

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    ref_states = init_batched_state(StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    ), co.TRAIN_X0_BATCH)

    def make_train_step(N):
        @nnx.jit
        def train_step(ens, opt):
            def loss_fn(e):
                cg, cp = nnx.split(e, nnx.Param)

                def single_member(p, sp, x0):
                    c = nnx.merge(cg, p)
                    loss, _ = rollout_learned(c, m3_graphdef, sp, x0, ref_states, co.Q_X, co.R_U, co.Q_F, N)
                    return loss

                losses = jax.vmap(single_member)(cp, surrogate_params_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        return train_step

    # ---- BASELINE: standard curriculum, cap N=200 ----
    print(f"\n{'=' * 20} BASELINE: standard curriculum (cap N=200) {'=' * 20}")
    optimizer = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)
    t0 = time.time()
    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]
        train_step = make_train_step(N)
        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
        print(f"    phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")
    print(f"  wall time: {time.time() - t0:.1f}s")

    baseline_ens_state = nnx.state(ensemble, nnx.Param)
    baseline_states = {seed: jax.tree_util.tree_map(lambda x, i=i: x[i], baseline_ens_state)
                        for i, seed in enumerate(members)}
    baseline_rows = evaluate_and_report("baseline_N200", baseline_states, max_action, A_true, B_true,
                                          oracle_cost, eval_key, m3_graphdef, m3_params_by_seed)

    # ---- EXTENDED: continue training the SAME ensemble at N=1000, fresh LR schedule ----
    print(f"\n{'=' * 20} EXTENDED: +1 phase at N={EXTENDED_N}, {EXTENDED_EPOCHS} epochs, fresh cosine LR {'=' * 20}")
    optimizer_ext = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, EXTENDED_EPOCHS)
    train_step_ext = make_train_step(EXTENDED_N)
    t0 = time.time()
    for epoch in range(EXTENDED_EPOCHS):
        loss, per_member = train_step_ext(ensemble, optimizer_ext)
        if (epoch + 1) % 200 == 0:
            print(f"    epoch {epoch + 1}/{EXTENDED_EPOCHS}  mean DPC loss: {float(loss):.4e}")
    print(f"  wall time: {time.time() - t0:.1f}s")

    extended_ens_state = nnx.state(ensemble, nnx.Param)
    extended_states = {seed: jax.tree_util.tree_map(lambda x, i=i: x[i], extended_ens_state)
                        for i, seed in enumerate(members)}
    extended_rows = evaluate_and_report(f"extended_N{EXTENDED_N}", extended_states, max_action, A_true, B_true,
                                          oracle_cost, eval_key, m3_graphdef, m3_params_by_seed)

    # ---- summary ----
    print(f"\n{'=' * 20} SUMMARY: baseline (N=200) vs extended (+N={EXTENDED_N}) {'=' * 20}")
    print(f"{'seed':5s} {'baseline_ratio':>15s} {'baseline_rho':>13s} {'baseline_b':>11s}  "
          f"{'extended_ratio':>15s} {'extended_rho':>13s} {'extended_b':>11s}")
    for seed in members:
        br = next(r for r in baseline_rows if r["seed"] == seed)
        er = next(r for r in extended_rows if r["seed"] == seed)
        print(f"{seed:<5d} {br['ratio']:>15.4e} {br['rho']:>13.6f} {str(br['b']):>11s}  "
              f"{er['ratio']:>15.4e} {er['rho']:>13.6f} {str(er['b']):>11s}")

    header = sorted({k for r in (baseline_rows + extended_rows) for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in (baseline_rows + extended_rows)]
    (DOCS_DIR / "n1000_case3_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'n1000_case3_summary.csv'}")


if __name__ == "__main__":
    main()
