"""Task 2 (session brief, 2026-08-13): the exact-realization control.

Builds an S4 model (M6 architecture - norm=layer, activation=gelu,
glu=True, so LayerNorm/GELU/GLU are architecturally PRESENT, not absent)
forced to realize a case's TRUE (A_d, B_d) to machine precision - the
same block-zeroing construction as tools/validate_diagnostics.py's
_apply_exact_linear, generalized here to build one such model per (case,
seed) rather than validating diagnostics.py once on case 3. Call this
"M0_S4": the true dynamics, deployed through the EXACT SAME decode=True
stepped StackedModel machinery (s4dpc.control.rollout_learned) that
M3/M6 controllers train through in tools/controller_surrogates.py.

Question this answers: does training a BoundedGRUController via BPTT
through THIS model - which computes x_next = A_d@x + B_d@u exactly, to
~1e-17, for every case - reproduce any of the 300x-700,000x
control-transfer failure Task 3 found for M3/M6? If yes, the failure is
in the S4-stepped-BPTT graph itself, not in what identification learned.
If no (M0_S4 lands near oracle, like M0/M1 do in
docs/controller_oracles_final_summary.csv), the S4 realization and its
backprop graph are innocent and the failure is genuinely about
identification quality.

Three parts, one kernel:

  A. CONSTRUCTION DIAGNOSTIC (seconds). Self-check against A_d@x+B_d@u
     directly. Then a direct empirical check of a claim readable off
     blocks.py's ConfigurableBlock.__call__: out.kernel=0/out.bias=0
     forces the block's post-GLU contribution to exactly 0 regardless of
     what the S4 layer computes, so with residual=True the block output
     is exactly `skip = encoder(x,u)`, independent of s4_state. Cold
     start (s0=0) and a burned-in warm start MUST therefore give
     bit-identical rollouts - UNLESS the (causally dead) state itself
     overflows to inf/nan, in which case `0 * nan = nan` would poison
     the supposedly-dead branch anyway. Checked directly, not assumed:
     diff a cold vs warm rollout, and separately check state finiteness
     over a 200-step rollout at each case's own max_action.

  B. HORIZON ABLATION (case 3, seed 0, same curriculum caps as
     tools/task2a_m3_horizon_ablation.py: {5,20,50,100,200}) - a direct,
     apples-to-apples comparison against that script's M3 numbers
     (357.75x at cap=5, docs/task2a_m3_horizon_ablation.csv) and the
     true-plant numbers (4.16x at cap=5) already on record. Answers
     "does even 5-step BPTT through the EXACT plant, deployed via the S4
     stepping machinery, transfer this badly" directly and cheaply
     before the full ensemble runs.

  C. FULL ENSEMBLE (all 6 control cases - case 6 excluded, same reason
     controller_surrogates.py excludes it: the oracle itself fails there
     under BPTT, so no result would be interpretable - 5 seeds, full
     standard curriculum, per-case max_action from
     controller_oracles.CASE_MAX_ACTION). Reuses controller_surrogates'
     exact training/eval shape (_train_ensemble_learned, co._evaluate)
     with ONLY the model construction swapped (M0_S4 in place of a
     loaded M3/M6 checkpoint), so any resulting gap can't be attributed
     to a different curriculum, cost, hidden_dim, or evaluation.

    python tools/controller_m0_s4.py
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

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import init_batched_state, rollout_learned  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100  # matches the 10-seed sweep's architecture
D_X, D_U = 6, 3

CONTROL_CASES = [c for c in co.CASES if c != 6]
N_SEEDS = 5
HORIZON_CASE, HORIZON_SEED = 3, 0
CAPS = [5, 20, 50, 100, 200]
_FULL_CURRICULUM = list(co.CURRICULUM)
CAPPED_CURRICULUM = {
    5: _FULL_CURRICULUM[0:1], 20: _FULL_CURRICULUM[0:3], 50: _FULL_CURRICULUM[0:4],
    100: _FULL_CURRICULUM[0:5], 200: _FULL_CURRICULUM[0:6],
}

DOCS_DIR = _REPO_ROOT / "docs"


def _build_m0_s4(case: int, seed: int) -> StackedModel:
    """M6-architecture StackedModel, decode=True, with the block zeroed
    so its one-step map is EXACTLY x_next = A_d@x + B_d@u for `case`
    (tools/validate_diagnostics.py's _apply_exact_linear, generalized
    over case/seed). `seed` only affects the S4 layer's OWN (Lambda_re,
    Lambda_im, P, B, log_step) params, which this construction leaves
    untouched - those params drive the internal state recurrence but,
    per the docstring above, have zero causal path to the output. Kept
    varying by seed anyway: it's free (one extra PRNGKey draw) and lets
    the finiteness check in run_construction_diagnostics sample multiple
    random SSM parameterizations rather than just one."""
    A_d, B_d = get_discrete_matrices(co.DT, case)
    w_true = np.concatenate([np.asarray(A_d), np.asarray(B_d)], axis=1).T  # (9,6) = [A_d|B_d]^T

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M6"])
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=key),
    )

    def _cast(x):
        return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)

    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(_cast, state)
    nnx.update(model, state)

    block = model.layers[0]
    d_input, d_output = w_true.shape  # (9, 6)

    encoder_kernel = jnp.zeros((d_input, D_MODEL), dtype=jnp.float64)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_true, dtype=jnp.float64))
    model.encoder.kernel.value = encoder_kernel
    model.encoder.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    block.seq.D.value = jnp.zeros_like(block.seq.D.value, dtype=jnp.float64)
    block.seq.C_real_imag.value = jnp.zeros_like(block.seq.C_real_imag.value, dtype=jnp.float64)
    block.out.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)
    block.out.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)
    block.out2.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)  # M6 has glu=True
    block.out2.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    decoder_kernel = jnp.zeros((D_MODEL, d_output), dtype=jnp.float64)
    decoder_kernel = decoder_kernel.at[:d_output, :].set(jnp.eye(d_output, dtype=jnp.float64))
    model.decoder.kernel.value = decoder_kernel
    model.decoder.bias.value = jnp.zeros((d_output,), dtype=jnp.float64)

    return model


def run_construction_diagnostics() -> float:
    """Returns the max self-check error across all control cases - the
    caller gates Parts B/C on this before spending real GPU time on a
    construction that might be wrong."""
    print("\n" + "=" * 20 + " PART A: construction diagnostics " + "=" * 20)
    rng = np.random.RandomState(0)
    max_self_check_err = 0.0
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(co.DT, case)
        model = _build_m0_s4(case, seed=0)
        graphdef, params = nnx.split(model, nnx.Param)

        # self-check: forward vs A_d@x + B_d@u directly
        x_test = jnp.asarray(rng.uniform(-3, 3, size=(D_X,)), dtype=jnp.float64)
        u_test = jnp.asarray(rng.uniform(-3, 3, size=(D_U,)), dtype=jnp.float64)
        states0 = init_batched_state(model, 1)
        states0_unbatched = [s[0] for s in states0]
        model_in = jnp.concatenate([x_test, u_test])
        x_next_model, _ = model(model_in, states0_unbatched)
        x_next_true = jnp.asarray(A_d) @ x_test + jnp.asarray(B_d) @ u_test
        self_check_err = float(jnp.max(jnp.abs(x_next_model - x_next_true)))
        max_self_check_err = max(max_self_check_err, self_check_err)

        # cold vs warm start: burn the model in on a random (x,u) trajectory
        # to build a non-trivial s4_state, then compare a rollout from a
        # fresh x0 under cold (s0=0) vs warm (s_burned) state, same actions.
        burn_len = 40
        max_action = co.CASE_MAX_ACTION[case]
        u_burn = jnp.asarray(rng.uniform(-max_action, max_action, size=(burn_len, D_U)), dtype=jnp.float64)
        x_burn0 = jnp.asarray(rng.uniform(-5, 5, size=(D_X,)), dtype=jnp.float64)

        def apply_one(state, model_in):
            m = nnx.merge(graphdef, params)
            return m(model_in, state)

        x, s = x_burn0, states0_unbatched
        for t in range(burn_len):
            x, s = apply_one(s, jnp.concatenate([x, u_burn[t]]))
        states_warm = s

        rollout_len = 200
        u_roll = jnp.asarray(rng.uniform(-max_action, max_action, size=(rollout_len, D_U)), dtype=jnp.float64)
        x0_probe = jnp.asarray(rng.uniform(-5, 5, size=(D_X,)), dtype=jnp.float64)

        def rollout(states_init):
            x, s = x0_probe, states_init
            xs, s_norms = [x], []
            for t in range(rollout_len):
                x, s = apply_one(s, jnp.concatenate([x, u_roll[t]]))
                xs.append(x)
                s_norms.append(float(jnp.linalg.norm(jnp.concatenate([jnp.ravel(si) for si in s]))))
            return jnp.stack(xs), s_norms

        x_cold, s_norms_cold = rollout(states0_unbatched)
        x_warm, s_norms_warm = rollout(states_warm)
        cold_warm_diff = float(jnp.max(jnp.abs(x_cold - x_warm)))
        state_finite_cold = bool(np.all(np.isfinite(s_norms_cold)))
        state_finite_warm = bool(np.all(np.isfinite(s_norms_warm)))

        print(
            f"case{case}: self_check_err={self_check_err:.3e}  cold_vs_warm_output_diff={cold_warm_diff:.3e}  "
            f"state_norm_range_cold=[{min(s_norms_cold):.3e},{max(s_norms_cold):.3e}]  "
            f"state_finite_cold={state_finite_cold}  state_finite_warm={state_finite_warm}"
        )
    return max_self_check_err


def _train_true(cap: int) -> nnx.State:
    co.CURRICULUM = CAPPED_CURRICULUM[cap]
    co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
    co.CASES = [HORIZON_CASE]
    co.SEEDS = [HORIZON_SEED]
    grid = co._build_member_grid("M0")
    ensemble_state = co._train_ensemble(grid, f"true@cap{cap}")
    return jax.tree_util.tree_map(lambda x: x[0], ensemble_state)


def _train_ensemble_m0s4(cases: list[int], seeds: list[int], max_action: float) -> tuple[nnx.State, list[tuple[int, int]]]:
    """Mirrors controller_surrogates._train_ensemble_learned exactly,
    substituting _build_m0_s4 for a loaded checkpoint - same curriculum,
    cost, hidden_dim, optimizer, x0 sampling, evaluation shape."""
    members = [(case, seed) for case in cases for seed in seeds]
    print(f"  members ({len(members)}) @ max_action={max_action}: {members}")

    surrogate_models = [_build_m0_s4(case, seed) for case, seed in members]
    surrogate_graphdef, _ = nnx.split(surrogate_models[0], nnx.Param)
    surrogate_params_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[nnx.state(m, nnx.Param) for m in surrogate_models]
    )

    x0_list, key_list = [], []
    for case, seed in members:
        init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
        x0_key = jax.random.fold_in(init_key, 999)
        x0 = jax.random.uniform(
            x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE, dtype=jnp.float64
        )
        x0_list.append(x0)
        key_list.append(init_key)
    x0_batch = jnp.stack(x0_list)
    keys = jnp.stack(key_list)

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = co.make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)
    ref_states = init_batched_state(surrogate_models[0], co.TRAIN_X0_BATCH)

    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                controller_graphdef, controller_params = nnx.split(e, nnx.Param)

                def single_member(cp, sp, x0):
                    c = nnx.merge(controller_graphdef, cp)
                    loss, _ = rollout_learned(c, surrogate_graphdef, sp, x0, ref_states, co.Q_X, co.R_U, co.Q_F, N)
                    return loss

                losses = jax.vmap(single_member)(controller_params, surrogate_params_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"  [M0_S4] phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) epoch {epoch:4d} | "
                      f"mean DPC loss: {float(loss):.4f}  per-member range: "
                      f"[{float(jnp.min(per_member)):.3f}, {float(jnp.max(per_member)):.3f}]")

    return nnx.state(ensemble, nnx.Param), members


def run_horizon_ablation() -> None:
    print("\n" + "=" * 20 + f" PART B: horizon ablation, case {HORIZON_CASE} seed {HORIZON_SEED} " + "=" * 20)
    A_d, B_d = get_discrete_matrices(co.DT, HORIZON_CASE)
    Q = co.Q_X * np.eye(A_d.shape[0])
    R = co.R_U * np.eye(B_d.shape[1])
    K = co.solve_dlqr(A_d, B_d, Q, R)
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), HORIZON_CASE)
    x0_eval_np = np.asarray(
        jax.random.uniform(eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
    )
    x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
    oracle_lqr_cost = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)
    print(f"oracle LQR cost (case {HORIZON_CASE}): {oracle_lqr_cost:.4f}\n")

    max_action = co.CASE_MAX_ACTION[HORIZON_CASE]
    rows = []
    for cap in CAPS:
        # true plant
        co.CURRICULUM = CAPPED_CURRICULUM[cap]
        co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
        t0 = time.time()
        controller_state = _train_true(cap)
        elapsed = time.time() - t0
        controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
        nnx.update(controller, controller_state)
        result = co._evaluate(controller, A_d, B_d, eval_key)
        ratio = result["cost"] / oracle_lqr_cost if oracle_lqr_cost > 0 else float("inf")
        print(f"  [true@cap{cap}] wall={elapsed:.1f}s  ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
        rows.append({"target": "true", "cap": cap, "wall_s": elapsed, "cost": result["cost"],
                     "ratio_to_oracle": ratio, "oracle_lqr_cost": oracle_lqr_cost})

        # M0_S4
        co.CURRICULUM = CAPPED_CURRICULUM[cap]
        co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)
        t0 = time.time()
        ensemble_state, members = _train_ensemble_m0s4([HORIZON_CASE], [HORIZON_SEED], max_action)
        assert members == [(HORIZON_CASE, HORIZON_SEED)]
        controller_state = jax.tree_util.tree_map(lambda x: x[0], ensemble_state)
        elapsed = time.time() - t0
        controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
        nnx.update(controller, controller_state)
        result = co._evaluate(controller, A_d, B_d, eval_key)
        ratio = result["cost"] / oracle_lqr_cost if oracle_lqr_cost > 0 else float("inf")
        print(f"  [M0_S4@cap{cap}] wall={elapsed:.1f}s  ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
        rows.append({"target": "M0_S4", "cap": cap, "wall_s": elapsed, "cost": result["cost"],
                     "ratio_to_oracle": ratio, "oracle_lqr_cost": oracle_lqr_cost})

    co.CURRICULUM = _FULL_CURRICULUM
    co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)

    print("\n=== HORIZON ABLATION SUMMARY (case 3, seed 0) ===")
    print(f"{'cap':5s} {'target':7s} {'ratio_to_oracle':>16s}")
    for cap in CAPS:
        for target in ["true", "M0_S4"]:
            r = next(r for r in rows if r["cap"] == cap and r["target"] == target)
            print(f"{cap:5d} {target:7s} {r['ratio_to_oracle']:16.4e}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "controller_m0_s4_horizon_ablation.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'controller_m0_s4_horizon_ablation.csv'}")


def run_full_ensemble() -> None:
    print("\n" + "=" * 20 + " PART C: full ensemble, all control cases, "
          f"{N_SEEDS} seeds " + "=" * 20)
    seeds = list(range(N_SEEDS))

    oracle_costs, eval_keys, true_AB = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(co.DT, case)
        true_AB[case] = (A_d, B_d)
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

    by_bound: dict[float, list[int]] = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    rows = []
    for max_action, cases in by_bound.items():
        n_members = len(cases) * N_SEEDS
        print(f"\n{'=' * 10} M0_S4 ensemble, cases {cases} @ max_action={max_action} ({n_members} members) {'=' * 10}")
        t0 = time.time()
        ensemble_state, members = _train_ensemble_m0s4(cases, seeds, max_action)
        print(f"  ensemble training wall time: {time.time() - t0:.1f}s")

        for i, (case, seed) in enumerate(members):
            label = f"M0_S4/case{case}/seed{seed}"
            try:
                member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
                nnx.update(controller, member_state)
                A_d, B_d = true_AB[case]
                result = co._evaluate(controller, A_d, B_d, eval_keys[case])
                cost_lqr = oracle_costs[case]
                ratio = result["cost"] / cost_lqr if cost_lqr > 0 else float("inf")
                result.update(oracle="M0_S4", case=case, seed=seed, max_action=max_action,
                              oracle_lqr_cost=cost_lqr, cost_ratio_to_oracle=ratio)
                print(f"  [{label}] cost={result['cost']:.4e}  ratio_to_oracle={ratio:.4e}  "
                      f"max|u|={result['max_abs_u']:.3e}  saturation_frac={result['saturation_frac']:.4f}  "
                      f"finite={result['finite']}")
            except Exception as e:
                import traceback
                print(f"  [{label}] FAILED: {e}")
                traceback.print_exc()
                result = {"oracle": "M0_S4", "case": case, "seed": seed, "max_action": max_action,
                          "failed": True, "oracle_lqr_cost": oracle_costs[case]}
            rows.append(result)

    print("\n\n=== M0_S4 SUMMARY: median cost + ratio to oracle LQR, per case ===")
    print(f"{'case':5s} {'max_action':>10s} {'oracle_lqr':>12s} {'median_cost':>14s} {'median_ratio':>13s} {'n_finite':>9s}")
    for case in CONTROL_CASES:
        these = [r for r in rows if not r.get("failed") and r["case"] == case]
        if not these:
            continue
        n_finite = sum(1 for r in these if r["finite"])
        median_cost = float(np.median([r["cost"] for r in these]))
        median_ratio = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
        print(f"{case:5d} {co.CASE_MAX_ACTION[case]:10.1f} {oracle_costs[case]:12.4f} "
              f"{median_cost:14.4e} {median_ratio:13.4e} {n_finite:9d}/{len(these)}")

    ok_rows = [r for r in rows if not r.get("failed")]
    if ok_rows:
        header = sorted({k for r in ok_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in ok_rows]
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "controller_m0_s4_summary.csv").write_text("\n".join(lines))
        print(f"\nwrote {DOCS_DIR / 'controller_m0_s4_summary.csv'}")


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}  per-case max_action="
          f"{ {c: co.CASE_MAX_ACTION[c] for c in CONTROL_CASES} }")
    max_self_check_err = run_construction_diagnostics()
    if max_self_check_err > 1e-9:
        print(f"\nSELF-CHECK FAILED (max err {max_self_check_err:.3e} > 1e-9) - construction does not "
              f"realize the true system for at least one case. Stopping before Parts B/C.")
        return
    run_horizon_ablation()
    run_full_ensemble()


if __name__ == "__main__":
    main()
