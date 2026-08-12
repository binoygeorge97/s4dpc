"""One-off pilot (not a standing diagnostic): verifies the vmapped-
ensemble rewrite of tools/controller_oracles.py (nnx.vmap construction +
nnx.split/jax.vmap per-member forward, mirroring s4dpc/identify.py's
_train_ensemble - structurally proven, but not yet exercised for
BoundedGRUController specifically, nor for training through 21 DIFFERENT
(A,B) plant matrices at once) before committing to the real, ~4-hour-if-
sequential job. Overrides the module's CURRICULUM/CASES/SEEDS to a small,
fast scope (3 cases x 2 seeds = 6 members, 2 short phases) - same
functions, same code path, just less of it, so a pass here is direct
evidence about the real run's mechanics, and gives a per-phase timing
data point for the vmapped design specifically.

    python tools/pilot_controller_ensemble.py
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

# shrink the scope - same functions/code path, less of it
co.CASES = [3, 4, 6]
co.SEEDS = [0, 1]
co.CURRICULUM = [{"N": 5, "epochs": 100}, {"N": 10, "epochs": 100}]
co.TOTAL_EPOCHS = sum(p["epochs"] for p in co.CURRICULUM)


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"PILOT scope: cases={co.CASES} seeds={co.SEEDS} curriculum={[p['N'] for p in co.CURRICULUM]}")

    t0 = time.time()
    grid = co._build_member_grid("M0")
    print(f"grid built: A_batch.shape={grid['A_batch'].shape}  B_batch.shape={grid['B_batch'].shape}  "
          f"x0_batch.shape={grid['x0_batch'].shape}  keys.shape={grid['keys'].shape}")
    print(f"member order: {list(zip(grid['cases'], grid['seeds']))}")

    ensemble_state = co._train_ensemble(grid, "PILOT/M0")
    elapsed = time.time() - t0
    print(f"\nensemble training wall time: {elapsed:.1f}s for {len(co.CASES) * len(co.SEEDS)} members x "
          f"{co.TOTAL_EPOCHS} epochs")

    print("\nper-member post-training evaluation:")
    all_ok = True
    for i, (case, seed) in enumerate(zip(grid["cases"], grid["seeds"])):
        try:
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, co.MAX_ACTION, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            A_d, B_d = co.get_discrete_matrices(co.DT, case)
            eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
            result = co._evaluate(controller, A_d, B_d, eval_key)
            print(f"  case{case}/seed{seed}: cost={result['cost']:.4e}  finite={result['finite']}  "
                  f"init_norm={result['init_norm']:.3e}  final_norm={result['final_norm']:.3e}")
            all_ok = all_ok and result["finite"]
        except Exception as e:
            import traceback
            print(f"  case{case}/seed{seed}: FAILED: {e}")
            traceback.print_exc()
            all_ok = False

    per_phase_time = elapsed / len(co.CURRICULUM)
    print(f"\nPILOT {'PASS' if all_ok else 'FAIL'}")
    print(f"~{per_phase_time:.2f}s per (phase, 6-member ensemble) at this small scope - "
          f"real run is 21 members (3.5x) and the full curriculum (9000 vs {co.TOTAL_EPOCHS} epochs, "
          f"{9000 / co.TOTAL_EPOCHS:.1f}x), so this is a lower bound on relative speedup, not a direct "
          f"time projection (N=100/200 phases cost much more per-epoch than N=5/10).")


if __name__ == "__main__":
    main()
