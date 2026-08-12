"""One-off pilot (not a standing diagnostic): sanity-checks the @nnx.jit
added to identify.py's _train_one/_train_ensemble (docs/DECISIONS.md,
Task 3 prep) before committing a large 2x40k-epoch x 7-case x 3-seed run
to it. Checks that jit doesn't silently break anything by:
  1. running vmap and --no-vmap on the same tiny config and comparing
     final teacher_mse (mirrors tests/test_identify.py's own check,
     which cannot be run directly here - no local jax);
  2. confirming the loss actually decreases (a jit bug that silently
     no-ops the update would still "run" without error).

    python tools/pilot_jit_check.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

from s4dpc.identify import run_identify

CASES = [3, 6]
N_SEEDS = 2
EPOCHS = 300
D_MODEL, STATE_SIZE, N_LAYERS = 16, 32, 1


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    for variant in ["M3", "M6"]:
        print(f"\n=== {variant} ===")
        rows_vmap = run_identify(
            variant=variant, cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
            d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, use_vmap=True,
        )
        rows_novmap = run_identify(
            variant=variant, cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
            d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, use_vmap=False,
        )
        assert len(rows_vmap) == len(rows_novmap) == len(CASES) * N_SEEDS

        all_ok = True
        for rv, rn in zip(rows_vmap, rows_novmap):
            assert rv["case"] == rn["case"] and rv["seed"] == rn["seed"]
            rel_diff = abs(rv["teacher_mse"] - rn["teacher_mse"]) / max(rn["teacher_mse"], 1e-300)
            ok = rel_diff < 1e-4
            all_ok &= ok
            print(f"  case={rv['case']} seed={rv['seed']}  vmap={rv['teacher_mse']:.6e}  "
                  f"no-vmap={rn['teacher_mse']:.6e}  rel_diff={rel_diff:.3e}  {'OK' if ok else 'MISMATCH'}")

        print(f"  [{variant}] vmap == no-vmap (within 1e-4 rel): {'PASS' if all_ok else 'FAIL'}")

        # sanity: loss actually moved from a fresh random init (not silently a no-op)
        # M3/M6 at 300 epochs, d_model=16: known-bad-init teacher_mse is O(1-30)
        # (docs/DECISIONS.md's very first M3 smoke-test entry, teacher_mse=3.46
        # at 50 epochs) - after 300 REAL jitted steps every row should be well
        # under 1.0, not still at random-init scale.
        max_mse = max(r["teacher_mse"] for r in rows_vmap)
        moved = max_mse < 1.0
        print(f"  [{variant}] max teacher_mse over all rows = {max_mse:.6e}  "
              f"(should be << 1.0 after {EPOCHS} real steps): {'PASS' if moved else 'FAIL - jit may be a no-op'}")


if __name__ == "__main__":
    main()
