"""TASK B (user, 2026-08-19, eighth round): the conv/step parity break
for M6 (docs/DECISIONS.md's TASK 0 EXTENSION entry) was measured
teacher-forced along the identification data's own APRBS trajectory -
exactly the inputs M6 was fit on. A real DPC rollout is chosen by an
actively-optimizing controller, not by APRBS, and CAN exit the
identification support (tools/task2b_m6_reality_gap.py already showed
this for cost specifically). "Too small to explain M6's failure"
assumed the discrepancy stays bounded outside that support - not yet
checked directly.

Trains ONE real GRU-DPC controller (case=3, seed=0, same case this
project has used for M3/M6 comparisons before) against the ACTUAL
committed M6 checkpoint (docs/nu_gap_export/ckpt/M6_case3_seed0.msgpack -
verified NOT to match tools/task2b_m6_reality_gap.py's old,
uncommitted results/all_cases/ckpt weights, so that trajectory could
not be reused for a self-consistent parity check). Same training
recipe as tools/controller_surrogates.py/tools/task2b_m6_reality_gap.py
(single-member ensemble via a SEED_SELECTION override), same curriculum,
same eval batch. Saves the resulting 200-step closed-loop (x, u)
trajectory with explicit checkpoint provenance for the local (CPU)
parity analysis to consume.

    python tools/task_b_m6_closedloop_trajectory.py
"""
from __future__ import annotations

import json
import pathlib
import sys

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
import controller_surrogates as cs  # noqa: E402

from s4dpc.control import init_batched_state  # noqa: E402
from s4dpc.logging import get_git_sha, get_lockfile_sha  # noqa: E402

CASE = 3
SEED = 0
VARIANT = "M6"
MAX_ACTION = 50.0  # CASE_MAX_ACTION[3] per controller_oracles.py
HORIZON = 200
DOCS_DIR = _REPO_ROOT / "docs"
OUT_DIR = DOCS_DIR / "task_b_m6_closedloop"

# Point at THIS session's committed checkpoints, not the old, uncommitted
# results/all_cases/ckpt (controller_surrogates.py's default) - verified
# directly (CPU-only replay check) that the old trajectories.npz does
# NOT match docs/nu_gap_export/ckpt/M6_case3_seed0.msgpack (max abs diff
# ~4-6 at step 1, nowhere near conv/step-parity scale), so reusing it
# would not have been a self-consistent check.
cs.CKPT_DIR = DOCS_DIR / "nu_gap_export" / "ckpt"


def rollout_controller_on_surrogate_batch(controller, model_graphdef, model_params, x0_batch, states, horizon_N):
    def apply_one(state, model_in):
        m = nnx.merge(model_graphdef, model_params)
        return m(model_in, state)

    x = x0_batch
    h = controller.init_hidden(x.shape[0])
    xs, us = [x], []
    for _ in range(horizon_N):
        h, u = controller(h, x)
        model_in = jnp.concatenate([x, u], axis=-1)
        x_next, states = jax.vmap(apply_one, in_axes=(0, 0))(states, model_in)
        xs.append(x_next)
        us.append(u)
        x = x_next
    return np.asarray(jnp.stack(xs)), np.asarray(jnp.stack(us))


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    ckpt_path = cs.CKPT_DIR / f"{VARIANT}_case{CASE}_seed{SEED}.msgpack"
    assert ckpt_path.exists(), f"missing checkpoint: {ckpt_path}"
    print(f"training controller against {ckpt_path}")

    cs.SEED_SELECTION[(VARIANT, CASE)] = [SEED]
    ensemble_state, members = cs._train_ensemble_learned(VARIANT, [CASE], MAX_ACTION)
    assert members == [(CASE, SEED)]
    controller_state = jax.tree_util.tree_map(lambda x: x[0], ensemble_state)
    controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, MAX_ACTION, rngs=nnx.Rngs(0))
    nnx.update(controller, controller_state)

    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), CASE)
    x0_batch = jax.random.uniform(
        eval_key, (co.EVAL_BATCH, co.D_X), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE, dtype=jnp.float64
    )

    surrogate = cs._build_surrogate(VARIANT, CASE, SEED)
    surrogate_graphdef, surrogate_params = nnx.split(surrogate, nnx.Param)
    states0 = init_batched_state(surrogate, co.EVAL_BATCH)

    print("rolling out the trained controller on the M6 surrogate, 200 steps...")
    x_surr, u_surr = rollout_controller_on_surrogate_batch(
        controller, surrogate_graphdef, surrogate_params, x0_batch, states0, HORIZON
    )
    cost_surr = co.true_quadratic_cost(x_surr, u_surr, co.Q_X, co.R_U, co.Q_F)
    print(f"cost on M6 surrogate: {cost_surr:.4e}")
    print(f"x_surr shape {x_surr.shape}  u_surr shape {u_surr.shape}")
    print(f"||x|| range over trajectory: [{np.linalg.norm(x_surr, axis=-1).min():.3f}, "
          f"{np.linalg.norm(x_surr, axis=-1).max():.3f}]")
    print(f"||u|| range over trajectory: [{np.linalg.norm(u_surr, axis=-1).min():.3f}, "
          f"{np.linalg.norm(u_surr, axis=-1).max():.3f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "trajectory.npz", x_surr=x_surr, u_surr=u_surr)
    sidecar = {
        "variant": VARIANT, "case": CASE, "seed": SEED, "max_action": MAX_ACTION, "horizon": HORIZON,
        "surrogate_checkpoint": str(ckpt_path.relative_to(_REPO_ROOT)),
        "cost_on_surrogate": float(cost_surr),
        "controller_recipe": "tools/controller_surrogates.py's _train_ensemble_learned, single-member "
                              "ensemble via SEED_SELECTION override (same pattern as task2b_m6_reality_gap.py)",
        "git_sha": get_git_sha(), "lockfile_sha": get_lockfile_sha(),
    }
    (OUT_DIR / "trajectory.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {OUT_DIR / 'trajectory.npz'}")


if __name__ == "__main__":
    main()
