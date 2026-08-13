"""GPU-only phase of the nu-gap analysis: identify M3, train controllers
for M1/full-M3/truncated-M3/M0_S4, extract (A,B)/(Abar,Bbar) and K_eff,
and EXPORT raw matrices - no nu-gap/Hankel-SVD/robust-margin math here.
That's all pure numpy/scipy given these matrices (seconds on CPU,
verified locally) and belongs on the machine that will read the
results, not queued behind a scarce 2-slot GPU scheduler for work that
doesn't need a GPU at all.

M0_S4's controller is trained via rollout_linear through the TRUE
(A_d,B_d) directly, not rollout_learned through the S4-stepped
construction - exploiting Task 2's own already-established, 5-seed-
per-case-confirmed result that the two are indistinguishable (identical
to 3-4 decimal places on 5/6 cases). This is a well-justified shortcut,
not a new assumption: re-deriving it via the expensive S4-stepping path
would just reproduce a result already on record at high statistical
power, for a fraction of the compute this export needs to be cheap.

Full M3 is the only genuinely expensive part (rollout_learned, BPTT
through the decode=True/S4 machinery) - everything else here is
rollout_linear, cheap.

Output: docs/nu_gap_export/{variant}_{case}_{seed}.npz, each containing
A, B, C (state-space realization used for delta_nu), K_eff (controller
linearization), teacher_mse (where applicable), plus a summary CSV
with the already-computed DPC cost ratio for cross-reference.

    python tools/nu_gap_export.py
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
from balanced_truncation import build_truncated_m3  # noqa: E402
from controller_m0_s4 import _build_m0_s4  # noqa: E402
from m3_spurious_modes import D_MODEL, STATE_SIZE, N_LAYERS, L_MAX, D_X, D_U, augmented_operator  # noqa: E402

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import (
    BoundedGRUController, init_batched_state, make_controller_optimizer,
    rollout_learned, rollout_linear,
)  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT, fit_least_squares, run_identify  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

CONTROL_CASES = [c for c in co.CASES if c != 6]
N_SEEDS = 5
DT = 0.01
DOCS_DIR = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS_DIR / "nu_gap_export"
COUT = np.concatenate([np.eye(D_X), np.zeros((D_X, 1030 - D_X))], axis=1)  # (6,1030), augmented-state readout


def k_eff_from_controller(controller_state, max_action: float) -> np.ndarray:
    controller = BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)

    def u_of_x(x):
        h0 = jnp.zeros((co.HIDDEN_DIM,))
        _, u = controller(h0, x)
        return u

    return np.asarray(jax.jacfwd(u_of_x)(jnp.zeros((co.D_X,))))


def train_linear_ensemble(AB_by_key, members, max_action, key_fn):
    A_list, B_list, x0_list, key_list = [], [], [], []
    for m in members:
        A, B = AB_by_key[key_fn(m)]
        A_list.append(jnp.asarray(A, dtype=jnp.float64))
        B_list.append(jnp.asarray(B, dtype=jnp.float64))
        case, seed = m
        init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
        x0_key = jax.random.fold_in(init_key, 999)
        x0 = jax.random.uniform(
            x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE, dtype=jnp.float64
        )
        x0_list.append(x0)
        key_list.append(init_key)
    A_batch, B_batch = jnp.stack(A_list), jnp.stack(B_list)
    x0_batch, keys = jnp.stack(x0_list), jnp.stack(key_list)

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)
    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                cg, cp = nnx.split(e, nnx.Param)

                def single_member(p, A, B, x0):
                    c = nnx.merge(cg, p)
                    return rollout_linear(c, x0, A, B, co.Q_X, co.R_U, co.Q_F, N)

                losses = jax.vmap(single_member)(cp, A_batch, B_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
        print(f"    phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")
    return nnx.state(ensemble, nnx.Param)


def evaluate_and_save(variant, case, seed, A_true, B_true, oracle_cost, eval_key, controller_state, max_action,
                       A_export, B_export, C_export, teacher_mse, rows):
    K_eff = k_eff_from_controller(controller_state, max_action)
    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)
    result = co._evaluate(controller, A_true, B_true, eval_key)
    ratio = result["cost"] / oracle_cost if oracle_cost > 0 else float("inf")
    print(f"    [{variant}/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EXPORT_DIR / f"{variant}_{case}_{seed}.npz",
              A=A_export, B=B_export, C=C_export, K_eff=K_eff,
              teacher_mse=teacher_mse if teacher_mse is not None else np.nan)
    rows.append({"variant": variant, "case": case, "seed": seed, "cost_ratio_to_oracle": ratio,
                 "finite": result["finite"], "teacher_mse": teacher_mse})


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    true_AB, oracle_costs, eval_keys = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        true_AB[case] = (np.asarray(A_d), np.asarray(B_d))
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        eval_keys[case] = eval_key
        Q = co.Q_X * np.eye(A_d.shape[0])
        R = co.R_U * np.eye(B_d.shape[1])
        K = co.solve_dlqr(A_d, B_d, Q, R)
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (co.EVAL_BATCH, A_d.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
        oracle_costs[case] = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)

    rows = []
    by_bound = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    # ---- M1 ----
    print(f"\n{'=' * 20} M1 {'=' * 20}")
    m1_AB = {}
    for case in CONTROL_CASES:
        ab_hat, _ = fit_least_squares(case, l_max=100, aprbs_low=-10.0, aprbs_high=10.0)
        m1_AB[case] = (ab_hat[:, :D_X], ab_hat[:, D_X:])
    for max_action, cases in by_bound.items():
        members = [(c, s) for c in cases for s in range(N_SEEDS)]
        print(f"  M1 max_action={max_action}, {len(members)} members")
        t0 = time.time()
        ens_state = train_linear_ensemble(m1_AB, members, max_action, key_fn=lambda m: m[0])
        print(f"  wall time: {time.time() - t0:.1f}s")
        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            A, B = m1_AB[case]
            evaluate_and_save("M1", case, seed, *true_AB[case], oracle_costs[case], eval_keys[case],
                               member_state, max_action, A, B, np.eye(D_X), None, rows)

    # ---- M0_S4 (via true-plant rollout_linear, per Task 2's established equivalence) ----
    print(f"\n{'=' * 20} M0_S4 (trained via rollout_linear through the true plant) {'=' * 20}")
    for max_action, cases in by_bound.items():
        members = [(c, s) for c in cases for s in range(N_SEEDS)]
        print(f"  M0_S4 max_action={max_action}, {len(members)} members")
        t0 = time.time()
        ens_state = train_linear_ensemble(true_AB, members, max_action, key_fn=lambda m: m[0])
        print(f"  wall time: {time.time() - t0:.1f}s")
        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            # Abar/Bbar for the ACTUAL M0_S4 construction (for delta_nu against the
            # augmented realization, even though the controller itself was trained
            # via the cheaper equivalent path)
            m0s4_model = _build_m0_s4(case, seed)
            m0s4_graphdef, m0s4_params = nnx.split(m0s4_model, nnx.Param)
            Abar, Bbar = augmented_operator(m0s4_graphdef, m0s4_params)
            evaluate_and_save("M0_S4", case, seed, *true_AB[case], oracle_costs[case], eval_keys[case],
                               member_state, max_action, Abar, Bbar, COUT, None, rows)

    # ---- M3 identification (shared by full-M3 and truncated-M3) ----
    print(f"\n{'=' * 20} identifying M3, cases {CONTROL_CASES} x {N_SEEDS} seeds {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant="M3", cases=CONTROL_CASES, n_seeds=N_SEEDS, epochs=40000,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")
    diverged = {(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0}
    print(f"  diverged: {sorted(diverged)}")
    teacher_mse_by_cs = {(r["case"], r["seed"]): r["teacher_mse"] for r in id_rows}

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    m3_graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    m3_param_by_cs = {(r["case"], r["seed"]): r["param_state"] for r in id_rows if (r["case"], r["seed"]) not in diverged}

    # ---- truncated M3 (ERA, r=6) ----
    print(f"\n{'=' * 20} truncated M3 (ERA) {'=' * 20}")
    trunc_AB = {}
    for (case, seed), param_state in m3_param_by_cs.items():
        res = build_truncated_m3(case, seed, param_state, m3_graphdef)
        if res.get("ok"):
            trunc_AB[(case, seed)] = (res["A"], res["B"])
    for max_action, cases in by_bound.items():
        members = [(c, s) for c in cases for s in range(N_SEEDS) if (c, s) in trunc_AB]
        if not members:
            continue
        print(f"  truncM3 max_action={max_action}, {len(members)} members")
        t0 = time.time()
        ens_state = train_linear_ensemble(trunc_AB, members, max_action, key_fn=lambda m: m)
        print(f"  wall time: {time.time() - t0:.1f}s")
        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            A, B = trunc_AB[(case, seed)]
            evaluate_and_save("truncM3", case, seed, *true_AB[case], oracle_costs[case], eval_keys[case],
                               member_state, max_action, A, B, np.eye(D_X), teacher_mse_by_cs.get((case, seed)), rows)

    # ---- full M3 (rollout_learned - the expensive part) ----
    print(f"\n{'=' * 20} full M3 (rollout_learned) {'=' * 20}")
    for max_action, cases in by_bound.items():
        members = [(c, s) for c in cases for s in range(N_SEEDS) if (c, s) in m3_param_by_cs]
        if not members:
            continue
        print(f"  fullM3 max_action={max_action}, {len(members)} members")
        surrogate_params_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[m3_param_by_cs[m] for m in members])
        x0_list, key_list = [], []
        for case, seed in members:
            init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
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
        optimizer = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)
        ref_states = init_batched_state(StackedModel(
            block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
            decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
        ), co.TRAIN_X0_BATCH)

        t0 = time.time()
        for pi, phase in enumerate(co.CURRICULUM):
            N = phase["N"]

            @nnx.jit
            def train_step(ens, opt, N=N):
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

            for epoch in range(phase["epochs"]):
                loss, per_member = train_step(ensemble, optimizer)
            print(f"    phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")
        print(f"  wall time: {time.time() - t0:.1f}s")
        ens_state = nnx.state(ensemble, nnx.Param)

        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            Abar, Bbar = augmented_operator(m3_graphdef, m3_param_by_cs[(case, seed)])
            evaluate_and_save("fullM3", case, seed, *true_AB[case], oracle_costs[case], eval_keys[case],
                               member_state, max_action, Abar, Bbar, COUT, teacher_mse_by_cs.get((case, seed)), rows)

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    (DOCS_DIR / "nu_gap_export_summary.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'nu_gap_export_summary.csv'}")
    print(f"wrote {len(rows)} .npz files to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
