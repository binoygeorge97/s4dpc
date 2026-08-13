"""Task 5 (session brief, 2026-08-13): k-step free-running (scheduled
sampling) identification loss, as a fix candidate for M3's control
failure.

s4dpc/identify.py trains fully teacher-forced: at every step t, the
model receives the TRUE x_t as part of its input (conv mode, decode=
False, one shot over the whole L=100 sequence), regardless of the
model's own prior prediction. But control.py's rollout_learned - and
therefore every DPC gradient - is fully FREE-RUNNING: the model's own
x_{t+1} feeds back as the next step's input, the true x is never seen
again after t=0. This function trains with the SAME architecture and
data but a chunked loss: reset to the TRUE x every k steps (teacher-
forced anchor), and for the k-1 steps after each anchor, feed the
model's OWN prediction forward, computing loss against the true target
at every one of those steps. k=1 reduces to exactly today's teacher-
forced loss (every step is its own anchor) - used as a sanity check
that this new code path reproduces the existing one before trusting
k>1.

Requires decode=True (stepped), unlike identify.py's decode=False/conv
training - the free-running steps need the model's own output fed back
as the next input, which conv mode's one-shot-over-the-sequence call
cannot do. This makes k-step training more expensive per epoch than
teacher-forced (a 100-step Python loop vs one conv call), so this uses
fewer epochs than the 40k-epoch standard sweep (see EPOCHS below) -
flagged plainly, not hidden, since it means k-step's absolute teacher_mse
values are not directly comparable to the standard sweep's.

    python tools/identify_kstep.py
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

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
from m3_spurious_modes import augmented_operator, K_GRID as SPECTRUM_K_GRID  # noqa: E402

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import BoundedGRUController, init_batched_state, rollout_learned  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
D_X, D_U = 6, 3
K_LIST = [1, 5, 10, 20]
N_SEEDS = 5  # identification + open-loop/spectrum diagnostics (cheap)
EPOCHS_KSTEP = 2000  # fewer than the standard 40k - see module docstring
CONTROL_CASES = [c for c in co.CASES if c != 6]

# DPC control training is the expensive part (4 k-values x a curriculum x
# an ensemble): meets the >=5-seed floor for identification/diagnostics
# above, but uses fewer seeds and a HALVED per-phase epoch budget here
# specifically, to keep 4 k-values' worth of control training tractable
# in one kernel. The brief's own framing ("a clean negative is stronger
# than a partial success... I am not asking you to make this work") does
# not require full curriculum depth to see whether k-step training moves
# the DPC ratio at all - flagged plainly as a lighter budget than every
# other DPC run in this project, not hidden in the numbers.
N_SEEDS_CONTROL = 3
CONTROL_CURRICULUM = [{"N": p["N"], "epochs": p["epochs"] // 2} for p in co.CURRICULUM]

DOCS_DIR = _REPO_ROOT / "docs"


def _cast_params(model: StackedModel, dtype: jnp.dtype) -> None:
    """Duplicated from s4dpc.identify (not imported - both are permanent
    modules; a small duplication beats importing a private name across
    files, matching control.py's own stated precedent). Complex-aware:
    S4LayerEnsemble's P/B params are genuinely complex."""
    complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64

    def _cast_leaf(x: jax.Array) -> jax.Array:
        return x.astype(complex_dtype if jnp.iscomplexobj(x) else dtype)

    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(_cast_leaf, state)
    nnx.update(model, state)


def _build_decode_model(block_config: BlockConfig, key: jax.Array) -> StackedModel:
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=key),
    )
    _cast_params(model, jnp.float64)
    return model


def _kstep_loss_single(model: StackedModel, inputs: jax.Array, targets: jax.Array, k: int) -> jax.Array:
    """inputs: (L, d_input) = [x_true_t, u_t]. targets: (L, d_output) =
    x_true_{t+1}. Chunked free-running loss: teacher-forced anchor every
    k steps, model's own prediction feeds forward within each chunk."""
    L = inputs.shape[0]
    x_true = inputs[:, :D_OUTPUT]
    u = inputs[:, D_OUTPUT:]
    states = model.init_state(N=STATE_SIZE)

    total = 0.0
    count = 0
    for seg_start in range(0, L, k):
        seg_len = min(k, L - seg_start)
        x = x_true[seg_start]  # teacher-forced anchor
        for i in range(seg_len):
            t = seg_start + i
            x_next, states = model(jnp.concatenate([x, u[t]]), states)
            total = total + jnp.sum((x_next - targets[t]) ** 2)
            count += D_OUTPUT
            x = x_next  # free-running within the chunk
    return total / count


def run_identify_kstep(
    k: int, cases: list[int], n_seeds: int, epochs: int, learning_rate: float = 1e-3,
    seed_base: int = 0,
) -> list[dict]:
    """Mirrors s4dpc.identify.run_identify's return shape (one row per
    (case, seed): variant='M3', case, seed, teacher_mse, param_state).

    Even though the chunked loss needs a 100-step-per-epoch sequential
    Python loop (unlike identify.py's one-shot conv call), that loop's
    SEQUENTIAL step count is the same whether training 1 member or all
    of them at once - vmapping the ensemble turns "cases*seeds separate
    100-step loops" into "one 100-step loop where each step does
    cases*seeds-way parallel work", the same order-of-magnitude speedup
    _train_ensemble gets in identify.py. An unvmapped first draft of
    this function was estimated (not run) at several hours for this
    module's K_LIST/N_SEEDS/EPOCHS_KSTEP; vmapped, it matches the
    established ensemble pattern (nnx.vmap for per-member construction,
    nnx.split/jax.vmap for the per-member forward+loss, one shared
    nnx.Optimizer) used throughout this repo."""
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])

    flat_cases, flat_seeds = [], []
    for case in cases:
        for seed in range(n_seeds):
            flat_cases.append(case)
            flat_seeds.append(seed)

    data = {c: case_data(c, L_MAX, -10.0, 10.0) for c in cases}
    inputs_grid = jnp.stack([data[c][0] for c in flat_cases])
    targets_grid = jnp.stack([data[c][1] for c in flat_cases])
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(seed_base + s), c) for c, s in zip(flat_cases, flat_seeds)])

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return _build_decode_model(block_config, key)

    ensemble = init_ensemble(keys)
    optimizer = nnx.Optimizer(ensemble, optax.adamw(learning_rate, weight_decay=0.0), wrt=nnx.Param)

    def loss_fn(model):
        graphdef, params = nnx.split(model, nnx.Param)

        def single_member(p, inp, tgt):
            m = nnx.merge(graphdef, p)
            return _kstep_loss_single(m, inp, tgt, k)

        losses = jax.vmap(single_member)(params, inputs_grid, targets_grid)
        return jnp.mean(losses), losses

    @nnx.jit
    def train_step(ens, opt):
        (_, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
        opt.update(ens, grads)
        return per_member

    t0 = time.time()
    for _ in range(epochs):
        train_step(ensemble, optimizer)
    _, final_per_member = loss_fn(ensemble)
    print(f"  [M3-k{k}] {len(flat_cases)} members, {epochs} epochs: wall={time.time() - t0:.1f}s")

    ensemble_state = nnx.state(ensemble, nnx.Param)
    rows = []
    for i, (case, seed) in enumerate(zip(flat_cases, flat_seeds)):
        member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
        rows.append({
            "variant": "M3", "case": case, "seed": seed, "k": k,
            "teacher_mse": float(final_per_member[i]), "param_state": member_state,
        })
    return rows


def _openloop_rmse(model: StackedModel, inputs: jax.Array, targets: jax.Array) -> float:
    """Cold-start free-run over the SAME L=100 training trajectory (no
    new data needed): x0 = the trajectory's own true x(0), then the
    model's own prediction feeds back every step - unlike teacher_mse,
    which is one-step (true x handed in at every step)."""
    x_true = inputs[:, :D_OUTPUT]
    u = inputs[:, D_OUTPUT:]
    states = model.init_state(N=STATE_SIZE)
    x = x_true[0]
    errs = []
    for t in range(inputs.shape[0]):
        x_next, states = model(jnp.concatenate([x, u[t]]), states)
        errs.append(jnp.sum((x_next - targets[t]) ** 2))
        x = x_next
    return float(jnp.sqrt(jnp.mean(jnp.stack(errs))))


def _train_ensemble_kstep_control(k_rows: list[dict], cases: list[int], max_action: float) -> tuple[nnx.State, list[tuple[int, int]]]:
    """Same shape as controller_surrogates._train_ensemble_learned, but
    the surrogate params come from k_rows (this script's own in-memory
    k-step checkpoints) instead of a loaded checkpoint file."""
    by_case_seed = {(r["case"], r["seed"]): r["param_state"] for r in k_rows}
    members = [(case, seed) for case in cases for seed in range(N_SEEDS_CONTROL) if (case, seed) in by_case_seed]
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    surrogate_graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    surrogate_params_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[by_case_seed[m] for m in members]
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
        return BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

    total_epochs = sum(p["epochs"] for p in CONTROL_CURRICULUM)
    ensemble = init_ensemble(keys)
    optimizer = co.make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, total_epochs)
    ref_states = init_batched_state(StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    ), co.TRAIN_X0_BATCH)

    for pi, phase in enumerate(CONTROL_CURRICULUM):
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
            if epoch == phase["epochs"] - 1:
                print(f"    phase {pi + 1}/{len(CONTROL_CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")

    return nnx.state(ensemble, nnx.Param), members


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"K_LIST={K_LIST}  N_SEEDS={N_SEEDS}  EPOCHS_KSTEP={EPOCHS_KSTEP}  CONTROL_CASES={CONTROL_CASES}")

    all_id_rows = {}
    diag_rows = []
    for k in K_LIST:
        print(f"\n{'=' * 20} k={k}: identification, all 7 cases x {N_SEEDS} seeds, {EPOCHS_KSTEP} epochs {'=' * 20}")
        id_rows = run_identify_kstep(k, cases=co.CASES, n_seeds=N_SEEDS, epochs=EPOCHS_KSTEP)
        all_id_rows[k] = id_rows

        block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
        graphdef, _ = nnx.split(
            StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                         decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
            nnx.Param,
        )
        for r in id_rows:
            inputs, targets = case_data(r["case"], L_MAX, -10.0, 10.0)
            model = _build_decode_model(block_config, jax.random.fold_in(jax.random.PRNGKey(r["seed"]), r["case"]))
            nnx.update(model, r["param_state"])
            openloop_rmse = _openloop_rmse(model, inputs, targets)

            Abar, _ = augmented_operator(graphdef, r["param_state"])
            Abar_j = jnp.asarray(Abar)
            rho_abar = float(np.max(np.abs(np.linalg.eigvals(Abar))))
            obs_norm = float(np.linalg.norm(Abar[:D_X, D_X:], ord=2))
            M = jnp.eye(Abar_j.shape[0], dtype=Abar_j.dtype)
            for _ in range(SPECTRUM_K_GRID[-1]):
                M = M @ Abar_j
            growth_k200 = float(jnp.linalg.norm(M, ord=2))

            print(f"  [k={k}/case{r['case']}/seed{r['seed']}] teacher_mse={r['teacher_mse']:.3e}  "
                  f"openloop_rmse={openloop_rmse:.3e}  rho(Abar)={rho_abar:.4f}  obs_norm={obs_norm:.3e}  "
                  f"||Abar^200||={growth_k200:.3e}")
            diag_rows.append({
                "k": k, "case": r["case"], "seed": r["seed"], "teacher_mse": r["teacher_mse"],
                "openloop_rmse": openloop_rmse, "rho_abar": rho_abar, "obs_norm": obs_norm,
                "growth_k200": growth_k200,
            })

    header = sorted({kk for r in diag_rows for kk in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in diag_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "identify_kstep_diagnostics.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'identify_kstep_diagnostics.csv'}")

    print("\n=== DIAGNOSTIC SUMMARY: median over seeds, per (k, case) ===")
    print(f"{'k':4s} {'case':5s} {'teacher_mse':>12s} {'openloop_rmse':>14s} {'rho(Abar)':>10s} {'obs_norm':>10s} {'growth_k200':>12s}")
    for k in K_LIST:
        for case in co.CASES:
            these = [r for r in diag_rows if r["k"] == k and r["case"] == case]
            if not these:
                continue
            print(f"{k:4d} {case:5d} "
                  f"{np.median([r['teacher_mse'] for r in these]):12.4e} "
                  f"{np.median([r['openloop_rmse'] for r in these]):14.4e} "
                  f"{np.median([r['rho_abar'] for r in these]):10.4f} "
                  f"{np.median([r['obs_norm'] for r in these]):10.4e} "
                  f"{np.median([r['growth_k200'] for r in these]):12.4e}")

    # ---- DPC control, all 6 control cases, per k ----
    print(f"\n{'=' * 20} DPC control training, all k, {CONTROL_CASES} {'=' * 20}")
    oracle_costs, true_AB, eval_keys = {}, {}, {}
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

    control_rows = []
    for k in K_LIST:
        for max_action, cases in by_bound.items():
            print(f"\n  --- k={k}, cases {cases} @ max_action={max_action} ---")
            t0 = time.time()
            ensemble_state, members = _train_ensemble_kstep_control(all_id_rows[k], cases, max_action)
            print(f"  wall time: {time.time() - t0:.1f}s  members: {members}")
            for i, (case, seed) in enumerate(members):
                member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
                nnx.update(controller, member_state)
                A_d, B_d = true_AB[case]
                result = co._evaluate(controller, A_d, B_d, eval_keys[case])
                ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
                print(f"    [k={k}/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
                control_rows.append({"k": k, "case": case, "seed": seed, "cost": result["cost"],
                                      "cost_ratio_to_oracle": ratio, "finite": result["finite"]})

    header = sorted({kk for r in control_rows for kk in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in control_rows]
    (DOCS_DIR / "identify_kstep_control.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'identify_kstep_control.csv'}")

    print("\n=== DPC SUMMARY: median cost_ratio_to_oracle, per (k, case) ===")
    print(f"{'k':4s} {'case':5s} {'median_ratio':>13s} {'n_finite':>9s}")
    for k in K_LIST:
        for case in CONTROL_CASES:
            these = [r for r in control_rows if r["k"] == k and r["case"] == case]
            if not these:
                continue
            n_finite = sum(1 for r in these if r["finite"])
            med = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
            print(f"{k:4d} {case:5d} {med:13.4e} {n_finite:9d}/{len(these)}")


if __name__ == "__main__":
    main()
