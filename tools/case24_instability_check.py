"""Housekeeping (session brief, 2026-08-13): resolve the flagged training-
instability item on cases 2/4.

Task 3's original M3 {1,2,3,4,7}@50 ensemble (docs/DECISIONS.md,
2026-08-13 "Task 3 results" entry) showed mean DPC loss through M3
spiking to 6.9e13 (per-member max 1.04e15) at the N=200 curriculum
phase - reported as aggregate ensemble statistics (mean/range over 15
members), which cannot distinguish "every member is unstable" from "one
member exploded and dragged the mean/max with it." Task 2a's follow-up
(sha cf86e51, this same document) trained case3/seed0 ALONE at the same
N=200 phase and found a completely calm loss curve (262.9 -> 37.3),
concluding the instability signature likely belongs to specific OTHER
cases in that ensemble (most likely case 2 and/or case 4, going by their
own eval-ratio outliers in Task 3's per-member table: case2's worst seed
reached 131,980x, case4's 486,120x) - a hypothesis, not yet directly
checked by looking at case 2 or case 4's own per-seed training curves.

This script checks that hypothesis directly: trains M3 controllers for
cases 2 AND 4 specifically (fresh M3 identification for those two cases,
then the FULL standard curriculum through control_surrogates' own
per-case max_action, matching Task 3's exact training setup), 5 seeds
each, recording PER-SEED (not just aggregate) DPC loss at every epoch of
the N=200 phase - the phase where the blowup was originally seen.
Answers directly: is the blowup a reproducible property of cases 2/4
(most or all seeds blow up), or one exploding member each time (a
minority of seeds)?

    python tools/case24_instability_check.py
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

CASES = [2, 4]
N_SEEDS = 5
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS_ID = 40000  # matches the established identification budget
DOCS_DIR = _REPO_ROOT / "docs"


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CASES={CASES}  N_SEEDS={N_SEEDS}")

    print(f"\n{'=' * 20} identifying M3, cases {CASES} x {N_SEEDS} seeds, {EPOCHS_ID} epochs {'=' * 20}")
    id_rows = run_identify(
        variant="M3", cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS_ID,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    diverged = [(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0]
    print(f"diverged (teacher_mse > 10): {diverged}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    surrogate_graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    members = [(c, s) for c in CASES for s in range(N_SEEDS)]
    assert [(r["case"], r["seed"]) for r in id_rows] == members
    surrogate_params_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r["param_state"] for r in id_rows])

    max_action = co.CASE_MAX_ACTION[2]
    assert co.CASE_MAX_ACTION[2] == co.CASE_MAX_ACTION[4] == 50.0, "cases 2/4 must share one bound to batch together"

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
    ref_states = init_batched_state(StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    ), co.TRAIN_X0_BATCH)

    per_epoch_member_loss = []  # only recorded during the N=200 phase
    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]
        is_last_phase = (pi == len(co.CURRICULUM) - 1)

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

        t0 = time.time()
        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
            if is_last_phase:
                per_epoch_member_loss.append(np.asarray(per_member))
            if epoch % max(1, phase["epochs"] // 5) == 0 or epoch == phase["epochs"] - 1:
                print(f"  phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) epoch {epoch:4d} | "
                      f"mean={float(loss):.4e}  per-member range=[{float(jnp.min(per_member)):.4e}, "
                      f"{float(jnp.max(per_member)):.4e}]")
        print(f"  phase {pi + 1} wall time: {time.time() - t0:.1f}s")

    per_epoch_member_loss = np.stack(per_epoch_member_loss)  # (n_epochs_last_phase, n_members)
    print(f"\n=== N=200 phase: per-seed loss trajectory, cases {CASES} x {N_SEEDS} seeds ===")
    print(f"{'case':5s} {'seed':5s} {'first_epoch':>14s} {'max_over_phase':>16s} {'final_epoch':>14s} {'exploded(>1e6)':>15s}")
    rows = []
    for i, (case, seed) in enumerate(members):
        traj = per_epoch_member_loss[:, i]
        first, mx, final = float(traj[0]), float(np.max(traj)), float(traj[-1])
        exploded = mx > 1e6
        print(f"{case:5d} {seed:5d} {first:14.4e} {mx:16.4e} {final:14.4e} {str(exploded):>15s}")
        rows.append({"case": case, "seed": seed, "first_epoch_loss": first, "max_loss": mx,
                      "final_epoch_loss": final, "exploded": exploded})

    n_exploded_by_case = {c: sum(1 for r in rows if r["case"] == c and r["exploded"]) for c in CASES}
    print(f"\nexploded-member count: {n_exploded_by_case} (out of {N_SEEDS} seeds each)")
    for c in CASES:
        frac = n_exploded_by_case[c] / N_SEEDS
        verdict = "REPRODUCIBLE (most/all seeds)" if frac >= 0.6 else (
            "MINORITY (one/few exploding members)" if frac > 0 else "NOT REPRODUCED here")
        print(f"  case{c}: {n_exploded_by_case[c]}/{N_SEEDS} exploded -> {verdict}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "case24_instability_check.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'case24_instability_check.csv'}")

    # also save the full per-epoch trajectory for plotting if useful later
    np.savez(DOCS_DIR / "case24_instability_trajectories.npz",
              per_epoch_member_loss=per_epoch_member_loss,
              members=np.array(members))
    print(f"wrote {DOCS_DIR / 'case24_instability_trajectories.npz'}")


if __name__ == "__main__":
    main()
