"""Task 3 (session brief, 2026-08-13): is the S4 hidden state's cold start
(s0=0 at every rollout, regardless of what physical x0 != 0 actually is)
the - or a - cause of the open-loop free-running divergence
tools/verify_m3_case3.py already found for M3/case3?

No controller anywhere in this script - free-running PREDICTION only,
open-loop, comparing:
  - cold start:  s0 = 0 at time t0 (current behaviour everywhere else in
    this project - control.py's rollout_learned, diagnostics.py).
  - warm start:  burn the model in TEACHER-FORCED (fed the TRUE x, not
    its own prediction) on the true (x, u) trajectory for B steps
    immediately preceding t0, carrying s forward, so s at t0 reflects
    the model's own recurrence given genuine recent history instead of
    an all-zero initial condition inconsistent with x(t0) != 0.

Fresh identification (M3 + M6, all 7 cases - case 6 stays IN, per
CLAUDE.md/docs/DECISIONS.md: identification-side results always keep
case 6, only CONTROL results exclude it), n_seeds=10 (not the >=5 floor
alone - case 6's ~50-60% divergence rate at this epoch budget,
docs/DECISIONS.md's 10-seed entry, needs the extra margin so at least 5
non-diverged seeds survive there too), 40k epochs - same architecture/
budget as the established all-cases-10seeds sweep, run fresh here (no
persisted checkpoint dataset covers a full 10-seed x 7-case x 2-variant
set) rather than loaded from disk, since everything downstream happens
in the same process/kernel.

For 3 restart points t0 (giving x0 magnitudes that grow naturally with
each case's own -generally unstable- open-loop dynamics, rather than
artificially constructing states: t0=50 samples "near the APRBS training
range", t0=150/250 sample increasingly large ||x|| as the case's own
instability compounds) and 4 burn-in lengths B in {0, 5, 20, 50} (B=0 is
exactly today's cold-start behaviour), every (case, seed) is rolled
forward B steps of teacher-forced burn-in then H_FREE=150 steps of
genuine free-running prediction (its own x feeds back, only u is
externally given) - vmapped over (case, seed) per (variant, B, t0)
combination (B and t0 are Python-level loops since every member within
one vmap call shares the same burn-in length and restart point; case/
seed vary within the batch), same "split params, vmap the forward pass"
shape as every other ensemble script in this repo.

Decision, per the session brief: if warm-starting substantially reduces
free-run error relative to cold, the inconsistent-initial-condition
mechanism is confirmed as (part of) the cause and warrants a follow-on
DPC re-run; if it changes little, that mechanism is dead and Task 4
(spurious internal modes) carries the investigation forward.

    python tools/openloop_warmstart.py
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

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify
from s4dpc.model import StackedModel

CASES = list(range(1, 8))
N_SEEDS = 10
EPOCHS = 40000
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
VARIANTS_TO_RUN = ["M3", "M6"]
DIVERGED_MSE_THRESHOLD = 10.0  # teacher_mse above this is flagged, not trusted (CLAUDE.md sec 3: tag, don't delete)

EVAL_SEED = 777  # deliberately different from identify.py's DATA_SEED=42
EVAL_LENGTH = 405
T0_LIST = [50, 150, 250]
B_LIST = [0, 5, 20, 50]
H_FREE = 150
D_X, D_U = 6, 3

DOCS_DIR = _REPO_ROOT / "docs"


def _true_trajectory(case: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x: (EVAL_LENGTH+1, d_x), u: (EVAL_LENGTH, d_u)) - a single
    long true rollout, seeded independently of identification data."""
    batch_inputs, batch_targets = generate_microgrid_trajectory(
        batch_size=1, length=EVAL_LENGTH, seed=EVAL_SEED, system_case=case, dt=DT,
        aprbs_low=-10.0, aprbs_high=10.0,
    )
    x = np.concatenate([batch_inputs[0, :, :D_X], batch_targets[0, -1:, :]], axis=0)  # (EVAL_LENGTH+1, d_x)
    u = batch_inputs[0, :, D_X:]  # (EVAL_LENGTH, d_u)
    return x, u


def run_variant(variant: str, true_traj: dict) -> list[dict]:
    print(f"\n{'=' * 20} identifying {variant}, all 7 cases x {N_SEEDS} seeds, {EPOCHS} epochs {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant=variant, cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")

    diverged = [(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > DIVERGED_MSE_THRESHOLD]
    print(f"  diverged (teacher_mse > {DIVERGED_MSE_THRESHOLD}): {diverged}")

    # decode=True/False share identical Param structure/shapes
    # (tests/test_decode_construction_parity.py) - `decode` only affects
    # the static graphdef's control flow (conv vs step), so id_rows'
    # param_state (trained decode=False) merges directly into a
    # decode=True graphdef with no rebuild needed. id_rows is already in
    # [(case, seed) for case in CASES for seed in range(N_SEEDS)] order
    # (s4dpc.identify.run_identify's flat_cases/flat_seeds construction),
    # matching `members` below exactly.
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    members = [(c, s) for c in CASES for s in range(N_SEEDS)]
    assert [(r["case"], r["seed"]) for r in id_rows] == members, "run_identify's row order changed"
    params_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r["param_state"] for r in id_rows])

    def apply_one(p, state, model_in):
        m = nnx.merge(graphdef, p)
        return m(model_in, state)

    out_rows = []
    for B in B_LIST:
        # rollout is defined once per B (its burn-in loop is unrolled over
        # B, a Python-level constant) and jitted once here - reused across
        # all 3 t0 values below via a stable function identity, instead of
        # being redefined (and silently recompiled) 3x per B.
        def rollout(p, x0, ub, xbt, uf, xt_free, B=B):
            # ub/xbt/uf/xt_free arrive here as (time, d) - member axis (0)
            # already stripped by vmap.
            state = [jnp.zeros((D_MODEL, STATE_SIZE), dtype=jnp.complex128)]
            x = x0
            for t in range(B):
                _, state = apply_one(p, state, jnp.concatenate([x, ub[t]]))
                x = xbt[t]
            errs = []
            for t in range(H_FREE):
                x_next, state = apply_one(p, state, jnp.concatenate([x, uf[t]]))
                errs.append(jnp.linalg.norm(x_next - xt_free[t]))
                x = x_next
            return jnp.stack(errs)

        vmapped_rollout = jax.jit(jax.vmap(rollout, in_axes=(0, 0, 0, 0, 0, 0)))

        for t0_val in T0_LIST:
            x_start_list, u_burn_list, u_free_list, x_true_free_list, x_burn_targets_list = [], [], [], [], []
            for case, seed in members:
                x_true, u_true = true_traj[case]
                x_start_list.append(x_true[t0_val - B])
                u_burn_list.append(u_true[t0_val - B:t0_val] if B > 0 else np.zeros((0, D_U)))
                u_free_list.append(u_true[t0_val:t0_val + H_FREE])
                x_true_free_list.append(x_true[t0_val + 1:t0_val + 1 + H_FREE])
                x_burn_targets_list.append(x_true[t0_val - B + 1:t0_val + 1] if B > 0 else np.zeros((0, D_X)))

            # all kept as (n_members, time, d) - member axis at 0, matching
            # params_batch/x_start, so a single uniform in_axes=0 vmap is
            # correct with no transposing.
            x_start = jnp.asarray(np.stack(x_start_list), dtype=jnp.float64)
            u_burn = jnp.asarray(np.stack(u_burn_list), dtype=jnp.float64)
            u_free = jnp.asarray(np.stack(u_free_list), dtype=jnp.float64)
            x_true_free = jnp.asarray(np.stack(x_true_free_list), dtype=jnp.float64)
            x_burn_targets = jnp.asarray(np.stack(x_burn_targets_list), dtype=jnp.float64)

            t_start = time.time()
            errs = vmapped_rollout(params_batch, x_start, u_burn, x_burn_targets, u_free, x_true_free)
            print(f"  [{variant}] t0={t0_val} B={B}: wall={time.time() - t_start:.1f}s  "
                  f"err@h=1/10/50/150 median across all members: "
                  f"{float(jnp.median(errs[:, 0])):.3e} / {float(jnp.median(errs[:, 9])):.3e} / "
                  f"{float(jnp.median(errs[:, 49])):.3e} / {float(jnp.median(errs[:, -1])):.3e}")

            for i, (case, seed) in enumerate(members):
                out_rows.append({
                    "variant": variant, "case": case, "seed": seed, "t0": t0_val, "B": B,
                    "diverged": (case, seed) in diverged,
                    "x0_norm": float(jnp.linalg.norm(x_start[i])),
                    "err_h1": float(errs[i, 0]), "err_h10": float(errs[i, 9]),
                    "err_h50": float(errs[i, 49]), "err_h150": float(errs[i, -1]),
                })
    return out_rows


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CASES={CASES}  N_SEEDS={N_SEEDS}  T0_LIST={T0_LIST}  B_LIST={B_LIST}  H_FREE={H_FREE}")

    true_traj = {case: _true_trajectory(case) for case in CASES}
    for case in CASES:
        x, _ = true_traj[case]
        norms = [float(np.linalg.norm(x[t0])) for t0 in T0_LIST]
        print(f"  case{case} ||x(t0)|| at t0={T0_LIST}: {[f'{n:.2f}' for n in norms]}")

    all_rows = []
    for variant in VARIANTS_TO_RUN:
        all_rows.extend(run_variant(variant, true_traj))

    header = sorted({k for r in all_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in all_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "openloop_warmstart.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'openloop_warmstart.csv'}")

    print("\n=== SUMMARY: median err_h150 (non-diverged only), cold(B=0) vs warmest(B=50), by variant/case ===")
    print(f"{'variant':8s} {'case':5s} {'t0':5s} {'cold(B=0)':>12s} {'warm(B=50)':>12s} {'ratio warm/cold':>16s}")
    for variant in VARIANTS_TO_RUN:
        for case in CASES:
            for t0_val in T0_LIST:
                these = [r for r in all_rows if r["variant"] == variant and r["case"] == case
                         and r["t0"] == t0_val and not r["diverged"]]
                cold = [r["err_h150"] for r in these if r["B"] == 0]
                warm = [r["err_h150"] for r in these if r["B"] == 50]
                if not cold or not warm:
                    continue
                med_cold = float(np.median(cold))
                med_warm = float(np.median(warm))
                ratio = med_warm / med_cold if med_cold > 0 else float("nan")
                print(f"{variant:8s} {case:5d} {t0_val:5d} {med_cold:12.4e} {med_warm:12.4e} {ratio:16.4f}")


if __name__ == "__main__":
    main()
