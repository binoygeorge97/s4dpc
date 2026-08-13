"""Housekeeping (user brief, 2026-08-13, item 3): re-run k-step
free-running identification with a budget matched to the standard
teacher-forced baseline, since Task 5's original negative result
(docs/DECISIONS.md) was confounded by k-step training itself failing to
converge at a 2000-epoch budget (teacher_mse degrading from ~1e-2 at
k=1 to ~1e4-1e5 at k=20 on several cases).

SCOPING NOTE, stated plainly rather than silently decided: matching the
standard 40k-epoch budget exactly is not feasible in one Kaggle
session. The k-step chunked loss needs a genuinely sequential 100-step
Python loop per epoch (unlike identify.py's one parallel conv call per
epoch) - measured directly from Task 5's own run, ~2.37s/epoch for a
35-member (7 cases x 5 seeds) vmapped batch, so 40k epochs would take
~26 hours for even ONE k value, well past Kaggle's session limit. This
runs k=10 ALONE (the representative middle value from Task 5's {1,5,
10,20} sweep) at 10,000 epochs - 5x Task 5's original budget, ~6.6
hours estimated - the largest single-k budget that fits safely in one
session. If this still doesn't converge cleanly, that itself is
informative (the objective may be fundamentally harder to optimize,
not just under-provisioned); if it converges well and DPC still fails,
that's the clean negative the brief is after. Extending to k=5/20 at
the same budget, or to k=10 at a still-larger budget via a checkpointed
multi-session run, is a natural follow-up if this run doesn't settle
the question - not attempted here.

The k=1 comparison point is NOT re-derived here - it's read directly
from the already-established, already-trusted standard 40k-epoch M3
teacher-forced identification (tools/identify.py's own path, matching
every other M3 number in docs/DECISIONS.md), not from Task 5's own
bespoke k=1 chunked-loss run (which was a different code path at the
same reduced budget as every other k in that run).

    python tools/identify_kstep_matched.py
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

import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import identify_kstep as ik  # noqa: E402
import controller_oracles as co  # noqa: E402
from s4dpc.identify import run_identify  # noqa: E402

K = 10
EPOCHS = 10000
N_SEEDS = 5
DOCS_DIR = _REPO_ROOT / "docs"


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"K={K}  EPOCHS={EPOCHS} (5x Task 5's original 2000)  N_SEEDS={N_SEEDS}  cases={co.CASES}")

    print(f"\n{'=' * 20} standard teacher-forced M3 baseline (k=1 reference, 40k epochs) {'=' * 20}")
    t0 = time.time()
    baseline_rows = run_identify(
        variant="M3", cases=co.CASES, n_seeds=N_SEEDS, epochs=40000,
        d_model=ik.D_MODEL, N=ik.STATE_SIZE, n_layers=ik.N_LAYERS, l_max=ik.L_MAX,
    )
    print(f"  wall time: {time.time() - t0:.1f}s")
    baseline_by_case = {}
    for r in baseline_rows:
        baseline_by_case.setdefault(r["case"], []).append(r["teacher_mse"])

    print(f"\n{'=' * 20} k={K} identification, {EPOCHS} epochs, all 7 cases x {N_SEEDS} seeds {'=' * 20}")
    t0 = time.time()
    id_rows = ik.run_identify_kstep(K, cases=co.CASES, n_seeds=N_SEEDS, epochs=EPOCHS)
    print(f"  wall time: {time.time() - t0:.1f}s")

    block_config = ik.BlockConfig(d_model=ik.D_MODEL, N=ik.STATE_SIZE, l_max=ik.L_MAX, **ik.VARIANTS["M3"])
    graphdef, _ = ik.nnx.split(
        ik.StackedModel(block_config=block_config, d_input=ik.D_INPUT, d_output=ik.D_OUTPUT, n_layers=ik.N_LAYERS,
                         decode=True, rngs=ik.nnx.Rngs(params=jax.random.PRNGKey(0))),
        ik.nnx.Param,
    )

    print(f"\n{'=' * 20} diagnostics: teacher_mse, openloop_rmse, spectrum vs the k=1 baseline {'=' * 20}")
    diag_rows = []
    for r in id_rows:
        inputs, targets = ik.case_data(r["case"], ik.L_MAX, -10.0, 10.0)
        model = ik._build_decode_model(block_config, jax.random.fold_in(jax.random.PRNGKey(r["seed"]), r["case"]))
        ik.nnx.update(model, r["param_state"])
        openloop_rmse = ik._openloop_rmse(model, inputs, targets)

        Abar, _ = ik.augmented_operator(graphdef, r["param_state"])
        rho_abar = float(np.max(np.abs(np.linalg.eigvals(Abar))))

        baseline_median = float(np.median(baseline_by_case[r["case"]]))
        print(f"  [k={K}/case{r['case']}/seed{r['seed']}] teacher_mse={r['teacher_mse']:.3e}  "
              f"(k=1 baseline median for this case: {baseline_median:.3e})  "
              f"openloop_rmse={openloop_rmse:.3e}  rho(Abar)={rho_abar:.4f}")
        diag_rows.append({
            "k": K, "case": r["case"], "seed": r["seed"], "teacher_mse": r["teacher_mse"],
            "k1_baseline_median_teacher_mse": baseline_median, "openloop_rmse": openloop_rmse, "rho_abar": rho_abar,
        })

    header = sorted({k for r in diag_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in diag_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "identify_kstep_matched_diagnostics.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'identify_kstep_matched_diagnostics.csv'}")

    print(f"\n{'=' * 20} DPC control through k={K} checkpoints (halved curriculum, {ik.N_SEEDS_CONTROL} seeds - "
          f"unchanged from Task 5, this rerun targets the IDENTIFICATION budget confound specifically) {'=' * 20}")
    control_rows = []
    for max_action, cases in [(50.0, [c for c in ik.CONTROL_CASES if co.CASE_MAX_ACTION[c] == 50.0]),
                               (200.0, [c for c in ik.CONTROL_CASES if co.CASE_MAX_ACTION[c] == 200.0])]:
        if not cases:
            continue
        t0 = time.time()
        ensemble_state, members = ik._train_ensemble_kstep_control(id_rows, cases, max_action)
        print(f"  wall time: {time.time() - t0:.1f}s  members: {members}")
        for i, (case, seed) in enumerate(members):
            member_state = ik.jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=ik.nnx.Rngs(0))
            ik.nnx.update(controller, member_state)
            A_d, B_d = ik.get_discrete_matrices(co.DT, case)
            Q = co.Q_X * np.eye(A_d.shape[0])
            R = co.R_U * np.eye(B_d.shape[1])
            K_gain = co.solve_dlqr(A_d, B_d, Q, R)
            eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
            x0_eval_np = np.asarray(
                jax.random.uniform(eval_key, (co.EVAL_BATCH, A_d.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
            )
            x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K_gain, x0_eval_np, co.EVAL_HORIZON)
            oracle_cost = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)
            result = co._evaluate(controller, A_d, B_d, eval_key)
            ratio = result["cost"] / oracle_cost if oracle_cost > 0 else float("inf")
            print(f"    [k={K}/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
            control_rows.append({"k": K, "case": case, "seed": seed, "cost_ratio_to_oracle": ratio,
                                  "finite": result["finite"]})

    header = sorted({k for r in control_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in control_rows]
    (DOCS_DIR / "identify_kstep_matched_control.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'identify_kstep_matched_control.csv'}")

    print(f"\n=== SUMMARY: k={K}@{EPOCHS}epochs vs Task 5's k=10@2000epochs vs k=1 baseline, per case ===")
    print(f"{'case':5s} {'median_ratio':>13s} {'n_finite':>9s}")
    for case in ik.CONTROL_CASES:
        these = [r for r in control_rows if r["case"] == case]
        if not these:
            continue
        n_finite = sum(1 for r in these if r["finite"])
        med = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
        print(f"{case:5d} {med:13.4e} {n_finite:9d}/{len(these)}")


if __name__ == "__main__":
    main()
