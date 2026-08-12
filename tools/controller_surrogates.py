"""Controller Task 3 (docs/DECISIONS.md): train the GRU/DPC controller
through the M3 and M6 TRAINED SURROGATE checkpoints from the 10-seed
identification sweep, evaluated by the same honest transfer test as
Task 2 (s4dpc.control.evaluate_controller_on_true: closed-loop cost on
the TRUE plant, regardless of what the controller trained through).

GATED on Task 2's kill criterion NOT triggering (docs/DECISIONS.md,
controller_oracles.py's module docstring) - this script assumes that
gate has already been checked and cleared; it does not re-check it and
must not be run before that check has been done and reported.

CASE 6 EXCLUDED (docs/DECISIONS.md's 2026-08-13 entries): the kill
criterion DID trigger on case 6 specifically - the oracle (M0, exact
true dynamics) fails there at 1.3e7x, so a case-6 surrogate result would
be uninterpretable (any failure is indistinguishable from "BPTT through
a kreiss_like=330 plant is just unstable regardless of what's being
controlled through"). This is a control, not a convenience - the M0
number is what justifies dropping it, and case 6 stays IN all
identification-side results elsewhere, where it is informative. See
CONTROL_CASES below.

Final report is a SINGLE combined table (M0/M1/M3/M6 x case, median
cost + ratio to oracle) - M0/M1's numbers are NOT re-run here; they are
read back from docs/controller_oracles_summary.csv (already committed,
produced by controller_oracles.py's earlier run). Re-running would be
wasteful and would not change anything: each ensemble member's loss/
gradient is independent per-member under vmap (jnp.mean(losses)'s
gradient w.r.t. member i's params depends only on member i's own loss),
so cases 1/2/3/4/5/7's M0/M1 numbers are identical whether or not case 6
was included in that training batch.

Checkpoint selection: 3 seeds per (variant, case), lowest-index
non-diverged seed first (diverged = markov_err_mean > 1e3 in the 10-seed
sweep, docs/DECISIONS.md's 2026-08-12 10-seed entry, sourced from
docs/diagnose_all_cases_10seeds_raw.csv's `diverged` flag). Most
(variant, case) cells use seeds [0,1,2] untouched; six cells substitute
a later seed because one of 0/1/2 diverged - both the selection AND the
excluded seeds are printed explicitly at the top of main(), not just
implied by the table (SEED_SELECTION/EXCLUDED_SEEDS below).

Checkpoints ship via a Kaggle dataset
(mlresearch42/s4dpc-controller-task3-ckpt: the 42 (variant,case,seed)
.msgpack+.json pairs SEED_SELECTION names, nothing else), not git:
results/ is gitignored (CLAUDE.md sec 2/5), and these are
identification-run byproducts, not sweep.py CSV output - the transport
CLAUDE.md documents for platform-independent code (git clone + pip
install) has no analogue for binary artifacts, so this uses the
mechanism kernel-metadata.json already has a field for
(dataset_sources) rather than committing msgpack files to the repo.

Structurally mirrors controller_oracles.py's ensemble design (one
nnx.vmap'd controller ensemble, nnx.split/jax.vmap per-member forward),
reusing its constants/curriculum/_evaluate rather than re-deriving them
(`import controller_oracles as co`) - the one real difference is what
each member rolls through: rollout_learned (a trained StackedModel
surrogate, decode=True/stepped, s4dpc.control's docstring) in place of
rollout_linear's raw (A, B) matmul. The surrogate ensemble is built the
same shape as the controller ensemble (one shared graphdef + a
member-stacked Param-state pytree) but is NEVER passed to
nnx.value_and_grad - only the controller's params are the differentiated
argument, so the surrogate stays exactly as trained, frozen data the
loss is computed through, never updated by the controller's optimizer.

    python tools/controller_surrogates.py
"""
from __future__ import annotations

import csv
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
import flax.serialization as serialization

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402 - reuse constants/_evaluate, not re-derive them

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import init_batched_state, rollout_learned  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

# Must match the 10-seed sweep's architecture exactly (docs/LOG.md,
# tools/diagnose_all_cases_10seeds.py) - a mismatch would make
# state.replace_by_pure_dict() load trained weights onto the wrong
# shapes silently rather than raising.
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100

VARIANTS_TO_RUN = ["M3", "M6"]

# case 6 excluded throughout this script - see module docstring
CONTROL_CASES = [c for c in co.CASES if c != 6]

ORACLE_CSV = _REPO_ROOT / "docs" / "controller_oracles_summary.csv"  # M0/M1, from controller_oracles.py

# user-supplied M6 kink magnitudes (docs/DECISIONS.md's 10-seed entry),
# for the "does the M6-vs-M3 cost gap track kink" question - reporting
# only, not consumed by the training/eval logic above.
M6_KINK_MAGNITUDE = {1: 7.90, 2: 11.2, 3: 9.56, 4: 15.8, 5: 11.4, 7: 6.62}

# lowest-index non-diverged seed, first 3, per (variant, case) - derived
# from docs/diagnose_all_cases_10seeds_raw.csv's `diverged` flag
# (markov_err_mean > 1e3, docs/DECISIONS.md's 10-seed entry). Regenerate
# via the snippet in that entry if the checkpoint set ever changes.
SEED_SELECTION: dict[tuple[str, int], list[int]] = {
    ("M3", 1): [0, 1, 2], ("M3", 2): [0, 1, 2], ("M3", 3): [0, 1, 2],
    ("M3", 4): [0, 1, 2], ("M3", 5): [0, 1, 2], ("M3", 6): [0, 1, 4],
    ("M3", 7): [0, 1, 2],
    ("M6", 1): [0, 1, 2], ("M6", 2): [0, 1, 2], ("M6", 3): [0, 1, 2],
    ("M6", 4): [1, 2, 3], ("M6", 5): [0, 1, 2], ("M6", 6): [0, 1, 7],
    ("M6", 7): [0, 1, 3],
}
EXCLUDED_SEEDS: dict[tuple[str, int], list[int]] = {
    ("M3", 3): [8], ("M3", 6): [2, 3, 5, 6, 8],
    ("M6", 4): [0], ("M6", 5): [9], ("M6", 6): [2, 3, 4, 5, 6, 9], ("M6", 7): [2],
}

# Kaggle attaches dataset_sources under /kaggle/input/<slug>/; locally,
# populate from out-10seeds/s4dpc/results/all_cases/ckpt (not committed
# - see module docstring) by copying the needed files here.
CKPT_DIR = _REPO_ROOT / "results" / "all_cases" / "ckpt"

DOCS_DIR = _REPO_ROOT / "docs"


def _build_surrogate(variant: str, case: int, seed: int) -> StackedModel:
    """decode=True (rollout_learned needs stepped, not conv) - identify.py
    trains decode=False/conv; deploying one input at a time for control
    requires decode=True with the SAME trained params
    (tests/test_control_decode_parity.py, control.py's module docstring)."""
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=key),
    )
    path = CKPT_DIR / f"{variant}_case{case}_seed{seed}.msgpack"
    pure_dict = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)
    return model


def _train_ensemble_learned(variant: str) -> tuple[nnx.State, list[tuple[int, int]]]:
    """Same shape as controller_oracles._train_ensemble, but each member
    rolls its trajectory batch through ITS OWN trained surrogate
    checkpoint (rollout_learned) instead of a shared linear (A, B).
    Returns (trained controller param state, member (case, seed) order)."""
    members = [(case, seed) for case in CONTROL_CASES for seed in SEED_SELECTION[(variant, case)]]
    print(f"  members ({len(members)}): {members}")

    surrogate_models = [_build_surrogate(variant, case, seed) for case, seed in members]
    surrogate_graphdef, _ = nnx.split(surrogate_models[0], nnx.Param)
    surrogate_params_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[nnx.state(m, nnx.Param) for m in surrogate_models]
    )

    x0_list, key_list = [], []
    for case, seed in members:
        init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
        x0_key = jax.random.fold_in(init_key, 999)
        x0 = jax.random.uniform(
            x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE,
            dtype=jnp.float64,
        )
        x0_list.append(x0)
        key_list.append(init_key)
    x0_batch = jnp.stack(x0_list)
    keys = jnp.stack(key_list)

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, co.MAX_ACTION, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = co.make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)

    # State shape depends only on architecture (shared within one variant
    # ensemble) and batch size, not on trained weights - build once,
    # reused (fresh, i.e. zero) at the start of every member's rollout
    # every training step, same as controller_oracles' x0-only design.
    ref_states = init_batched_state(surrogate_models[0], co.TRAIN_X0_BATCH)

    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                controller_graphdef, controller_params = nnx.split(e, nnx.Param)

                def single_member(cp, sp, x0):
                    c = nnx.merge(controller_graphdef, cp)
                    loss, _ = rollout_learned(
                        c, surrogate_graphdef, sp, x0, ref_states, co.Q_X, co.R_U, co.Q_F, N
                    )
                    return loss

                losses = jax.vmap(single_member)(controller_params, surrogate_params_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"  [{variant}] phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) epoch {epoch:4d} | "
                      f"mean DPC loss: {float(loss):.4f}  per-member range: "
                      f"[{float(jnp.min(per_member)):.3f}, {float(jnp.max(per_member)):.3f}]")

    return nnx.state(ensemble, nnx.Param), members


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"MAX_ACTION={co.MAX_ACTION}  TOTAL_EPOCHS={co.TOTAL_EPOCHS}  "
          f"curriculum={[p['N'] for p in co.CURRICULUM]}")
    print("\nSeed selection (lowest-index non-diverged, first 3 per (variant, case)):")
    for (variant, case), seeds in SEED_SELECTION.items():
        excl = EXCLUDED_SEEDS.get((variant, case))
        note = f"  <- excluded {excl} (diverged, markov_err_mean > 1e3)" if excl else ""
        print(f"  {variant} case{case}: {seeds}{note}")

    print(f"\nCONTROL_CASES (case 6 excluded - oracle fails there, see module docstring): {CONTROL_CASES}")

    oracle_costs: dict[int, float] = {}
    eval_keys: dict[int, jax.Array] = {}
    true_AB: dict[int, tuple] = {}
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

    rows: list[dict] = []
    for variant in VARIANTS_TO_RUN:
        print(f"\n{'=' * 20} {variant} surrogate ensemble "
              f"({sum(len(v) for k, v in SEED_SELECTION.items() if k[0] == variant)} members) {'=' * 20}")
        t0 = time.time()
        ensemble_state, members = _train_ensemble_learned(variant)
        print(f"  [{variant}] ensemble training wall time: {time.time() - t0:.1f}s")

        for i, (case, seed) in enumerate(members):
            label = f"{variant}/case{case}/seed{seed}"
            try:
                member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
                controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, co.MAX_ACTION, rngs=nnx.Rngs(0))
                nnx.update(controller, member_state)
                A_d, B_d = true_AB[case]
                result = co._evaluate(controller, A_d, B_d, eval_keys[case])
                cost_lqr = oracle_costs[case]
                ratio = result["cost"] / cost_lqr if cost_lqr > 0 else float("inf")
                result.update(oracle=variant, case=case, seed=seed,
                              oracle_lqr_cost=cost_lqr, cost_ratio_to_oracle=ratio)
                print(f"  [{label}] cost={result['cost']:.4e}  ratio_to_oracle={ratio:.4e}  "
                      f"init_norm={result['init_norm']:.3e}  final_norm={result['final_norm']:.3e}  "
                      f"max_norm={result['max_norm']:.3e}  max|u|={result['max_abs_u']:.3e}  "
                      f"finite={result['finite']}")
            except Exception as e:
                import traceback
                print(f"  [{label}] FAILED: {e}")
                traceback.print_exc()
                result = {"oracle": variant, "case": case, "seed": seed, "failed": True,
                          "oracle_lqr_cost": oracle_costs[case]}
            rows.append(result)

    print("\n\n=== M3/M6 SUMMARY: median cost + ratio to oracle LQR, per (variant, case) ===")
    print(f"{'variant':8s} {'case':5s} {'oracle_lqr':>12s} {'median_cost':>14s} {'median_ratio':>13s} {'n_finite':>9s}")
    for variant in VARIANTS_TO_RUN:
        for case in CONTROL_CASES:
            these = [r for r in rows if not r.get("failed") and r["oracle"] == variant and r["case"] == case]
            if not these:
                continue
            n_finite = sum(1 for r in these if r["finite"])
            median_cost = float(np.median([r["cost"] for r in these]))
            median_ratio = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
            print(f"{variant:8s} {case:5d} {oracle_costs[case]:12.4f} {median_cost:14.4e} "
                  f"{median_ratio:13.4e} {n_finite:9d}/{len(these)}")

    ok_rows = [r for r in rows if not r.get("failed")]
    if ok_rows:
        header = sorted({k for r in ok_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in ok_rows]
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "controller_surrogates_summary.csv").write_text("\n".join(lines))
        print(f"\nwrote {DOCS_DIR / 'controller_surrogates_summary.csv'}")

    # ============================================================
    # Combined M0/M1/M3/M6 table - the one the user actually asked for.
    # M0/M1 read back from the already-committed oracle run (not
    # re-trained - see module docstring for why that's valid).
    # ============================================================
    if not ORACLE_CSV.exists():
        print(f"\nWARNING: {ORACLE_CSV} not found - skipping the combined M0/M1/M3/M6 table. "
              f"(controller_oracles.py must be run and its CSV committed first.)")
        return

    with open(ORACLE_CSV, newline="") as f:
        oracle_rows = [r for r in csv.DictReader(f) if int(r["case"]) in CONTROL_CASES]

    print("\n\n=== COMBINED TABLE: M0 / M1 / M3 / M6, median cost + ratio to oracle LQR, per case ===")
    print(f"{'oracle':8s} {'case':5s} {'oracle_lqr':>12s} {'median_cost':>14s} {'median_ratio':>13s} "
          f"{'n_finite':>9s} {'m6_kink':>8s}")
    combined: list[dict] = []
    for oracle_name in ["M0", "M1", "M3", "M6"]:
        for case in CONTROL_CASES:
            if oracle_name in ("M0", "M1"):
                these_costs = [float(r["cost"]) for r in oracle_rows if r["oracle"] == oracle_name and int(r["case"]) == case]
                these_ratios = [float(r["cost_ratio_to_oracle"]) for r in oracle_rows if r["oracle"] == oracle_name and int(r["case"]) == case]
                these_finite = [r["finite"] == "True" for r in oracle_rows if r["oracle"] == oracle_name and int(r["case"]) == case]
            else:
                these = [r for r in rows if not r.get("failed") and r["oracle"] == oracle_name and r["case"] == case]
                these_costs = [r["cost"] for r in these]
                these_ratios = [r["cost_ratio_to_oracle"] for r in these]
                these_finite = [r["finite"] for r in these]
            if not these_costs:
                continue
            median_cost = float(np.median(these_costs))
            median_ratio = float(np.median(these_ratios))
            n_finite = sum(1 for f in these_finite if f)
            kink = M6_KINK_MAGNITUDE.get(case, float("nan")) if oracle_name == "M6" else float("nan")
            kink_str = f"{kink:8.2f}" if np.isfinite(kink) else " " * 8
            print(f"{oracle_name:8s} {case:5d} {oracle_costs[case]:12.4f} {median_cost:14.4e} "
                  f"{median_ratio:13.4e} {n_finite:9d}/{len(these_costs)} {kink_str}")
            combined.append({
                "oracle": oracle_name, "case": case, "oracle_lqr_cost": oracle_costs[case],
                "median_cost": median_cost, "median_ratio_to_oracle": median_ratio,
                "n_finite": n_finite, "n_total": len(these_costs),
                "m6_kink_magnitude": kink if np.isfinite(kink) else "",
            })

    if combined:
        header = sorted({k for r in combined for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in combined]
        (DOCS_DIR / "controller_comparison_summary.csv").write_text("\n".join(lines))
        print(f"\nwrote {DOCS_DIR / 'controller_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
