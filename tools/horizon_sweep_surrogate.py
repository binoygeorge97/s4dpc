"""Task 6 (session brief, 2026-08-13): re-run the horizon sweep properly
for M3 and M6 - the expensive half (tools/horizon_sweep_oracle.py has
the cheap M0/M1 half). See that script's docstring for why this rerun
matters (docs/task2a_m3_horizon_ablation.csv is single-seed, case-3-only,
explicitly flagged as not to be trusted for its shape).

Fresh M3+M6 identification (all 6 control cases - case 6 excluded from
control per this document's standing convention, same reason
controller_surrogates.py excludes it - N_SEEDS seeds, 40k epochs), then
for each cap in {5,20,50,100,200} (CAPPED_CURRICULUM, same convention as
tools/task2a_m3_horizon_ablation.py: the ORIGINAL curriculum's phases up
through that N, not a single-phase run at N alone), trains a controller
through the trained surrogate and evaluates on the true plant.

    python tools/horizon_sweep_surrogate.py
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
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

CONTROL_CASES = [c for c in co.CASES if c != 6]
N_SEEDS = 5
VARIANTS_TO_RUN = ["M3", "M6"]
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS_ID = 40000

CAPS = [5, 20, 50, 100, 200]
_FULL_CURRICULUM = list(co.CURRICULUM)
CAPPED_CURRICULUM = {
    5: _FULL_CURRICULUM[0:1], 20: _FULL_CURRICULUM[0:3], 50: _FULL_CURRICULUM[0:4],
    100: _FULL_CURRICULUM[0:5], 200: _FULL_CURRICULUM[0:6],
}
DOCS_DIR = _REPO_ROOT / "docs"


def _train_ensemble_capped(
    variant: str, surrogate_graphdef, surrogate_params_batch, members: list[tuple[int, int]],
    max_action: float, cap: int,
) -> nnx.State:
    curriculum = CAPPED_CURRICULUM[cap]
    total_epochs = sum(p["epochs"] for p in curriculum)

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
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
    optimizer = co.make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, total_epochs)
    ref_states = init_batched_state(StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    ), co.TRAIN_X0_BATCH)

    for pi, phase in enumerate(curriculum):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                cg, cp = nnx.split(e, nnx.Param)

                def single_member(p, sp, x0):
                    c = nnx.merge(cg, p)
                    loss, _ = rollout_learned(c, surrogate_graphdef, sp, x0, ref_states, co.Q_X, co.R_U, co.Q_F, N)
                    return loss

                losses = jax.vmap(single_member)(cp, surrogate_params_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
        print(f"    [{variant}@cap{cap}] phase {pi + 1}/{len(curriculum)} (N={N}) final mean DPC loss: {float(loss):.4f}")

    return nnx.state(ensemble, nnx.Param)


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}  CAPS={CAPS}")

    oracle_costs, eval_keys, true_AB = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(co.DT, case)
        true_AB[case] = (A_d, B_d)
        Q = co.Q_X * np.eye(A_d.shape[0])
        R = co.R_U * np.eye(B_d.shape[1])
        K = co.solve_dlqr(A_d, B_d, Q, R)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        eval_keys[case] = eval_key
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (co.EVAL_BATCH, A_d.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
        oracle_costs[case] = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)

    by_bound: dict[float, list[int]] = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    rows = []
    for variant in VARIANTS_TO_RUN:
        print(f"\n{'=' * 20} identifying {variant}, cases {CONTROL_CASES} x {N_SEEDS} seeds, {EPOCHS_ID} epochs {'=' * 20}")
        id_rows = run_identify(
            variant=variant, cases=CONTROL_CASES, n_seeds=N_SEEDS, epochs=EPOCHS_ID,
            d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
        )
        diverged = {(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0}
        print(f"  diverged (teacher_mse > 10): {sorted(diverged)}")

        block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
        surrogate_graphdef, _ = nnx.split(
            StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                         decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
            nnx.Param,
        )
        by_case_seed = {(r["case"], r["seed"]): r["param_state"] for r in id_rows if (r["case"], r["seed"]) not in diverged}

        for cap in CAPS:
            for max_action, cases in by_bound.items():
                members = [(c, s) for c in cases for s in range(N_SEEDS) if (c, s) in by_case_seed]
                if not members:
                    continue
                print(f"\n  --- {variant} cap={cap} cases={cases} @ max_action={max_action} "
                      f"({len(members)} members) ---")
                surrogate_params_batch = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *[by_case_seed[m] for m in members]
                )
                t0 = time.time()
                ensemble_state = _train_ensemble_capped(
                    variant, surrogate_graphdef, surrogate_params_batch, members, max_action, cap
                )
                print(f"  wall time: {time.time() - t0:.1f}s")

                for i, (case, seed) in enumerate(members):
                    member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
                    nnx.update(controller, member_state)
                    A_d, B_d = true_AB[case]
                    result = co._evaluate(controller, A_d, B_d, eval_keys[case])
                    ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
                    print(f"    [{variant}@cap{cap}/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  "
                          f"finite={result['finite']}")
                    rows.append({"variant": variant, "cap": cap, "case": case, "seed": seed,
                                 "cost": result["cost"], "cost_ratio_to_oracle": ratio, "finite": result["finite"]})

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "horizon_sweep_surrogate.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'horizon_sweep_surrogate.csv'}")

    print("\n=== SUMMARY: median cost_ratio_to_oracle (with spread), per (variant, cap, case) ===")
    print(f"{'variant':8s} {'cap':5s} {'case':5s} {'median':>12s} {'min':>12s} {'max':>12s} {'n_finite':>9s}")
    for variant in VARIANTS_TO_RUN:
        for cap in CAPS:
            for case in CONTROL_CASES:
                these = [r for r in rows if r["variant"] == variant and r["cap"] == cap and r["case"] == case]
                if not these:
                    continue
                vals = [r["cost_ratio_to_oracle"] for r in these]
                n_finite = sum(1 for r in these if r["finite"])
                print(f"{variant:8s} {cap:5d} {case:5d} {np.median(vals):12.4e} {np.min(vals):12.4e} "
                      f"{np.max(vals):12.4e} {n_finite:9d}/{len(these)}")


if __name__ == "__main__":
    main()
