"""One-off pilot (not a standing diagnostic): verifies
tools/controller_surrogates.py's structurally NEW piece before
committing to the real, gated-behind-Task-2 run - training a controller
ensemble where each member rolls through a DIFFERENT trained S4
surrogate checkpoint (rollout_learned + a member-stacked, frozen,
NEVER-differentiated surrogate param pytree, nested inside the same
per-member jax.vmap that controller_oracles.py already proved works for
the simple rollout_linear case). Checks: checkpoint loading + graphdef
sharing across members, the nested vmap (member-vmap wrapping
rollout_learned's own trajectory-vmap) traces and runs, gradients flow
ONLY to the controller (loss decreases, surrogate untouched by
construction since it's never passed to nnx.value_and_grad), and
per-member post-training evaluation on the TRUE plant is finite.

Overrides controller_surrogates' SEED_SELECTION/co.CASES/co.CURRICULUM
to a small, fast scope (1 variant x 2 cases x 2 seeds = 4 members, 2
short phases) - same functions, same code path, just less of it.

    python tools/pilot_controller_surrogates.py
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

from flax import nnx  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
import controller_surrogates as cs  # noqa: E402

# shrink the scope - same functions/code path, less of it
co.CASES = [3, 6]
cs.SEED_SELECTION = {("M3", 3): [0, 1], ("M3", 6): [0, 1]}
co.CURRICULUM = [{"N": 5, "epochs": 100}, {"N": 10, "epochs": 100}]
co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"PILOT scope: variant=M3 members={list(cs.SEED_SELECTION.items())} "
          f"curriculum={[p['N'] for p in co.CURRICULUM]}")

    t0 = time.time()
    ensemble_state, members = cs._train_ensemble_learned("M3")
    elapsed = time.time() - t0
    print(f"\nensemble training wall time: {elapsed:.1f}s for {len(members)} members x {co.TOTAL_EPOCHS} epochs")

    print("\nper-member post-training evaluation:")
    all_ok = True
    for i, (case, seed) in enumerate(members):
        try:
            import jax.numpy as jnp
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, co.MAX_ACTION, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            A_d, B_d = co.get_discrete_matrices(co.DT, case)
            eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
            result = co._evaluate(controller, A_d, B_d, eval_key)
            print(f"  M3/case{case}/seed{seed}: cost={result['cost']:.4e}  finite={result['finite']}  "
                  f"init_norm={result['init_norm']:.3e}  final_norm={result['final_norm']:.3e}")
            all_ok = all_ok and result["finite"]
        except Exception as e:
            import traceback
            print(f"  M3/case{case}/seed{seed}: FAILED: {e}")
            traceback.print_exc()
            all_ok = False

    print(f"\nPILOT {'PASS' if all_ok else 'FAIL'}")
    print(f"~{elapsed / len(co.CURRICULUM):.2f}s per (phase, {len(members)}-member ensemble) at this small scope - "
          f"the real run is up to 21 members per variant x 2 variants and the full curriculum "
          f"(9000 vs {co.TOTAL_EPOCHS} epochs), and EACH step is a full StackedModel forward pass per "
          f"trajectory (much more expensive than rollout_linear's matmul) - not a direct time projection.")


if __name__ == "__main__":
    main()
