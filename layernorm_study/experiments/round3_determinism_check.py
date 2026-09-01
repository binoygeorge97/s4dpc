"""Round 3 audit follow-up: is the 2-4x train_mse discrepancy seen when
rerunning round 3's checkpoints under Python 3.12 (vs. the original 3.11
numbers) evidence of general nondeterminism in the training path itself
(XLA reduction order, threading), or specific to crossing environments?

Test: retrain the IDENTICAL (arm, seed) TWICE in THIS SAME environment
(two separate process invocations, matching how the original CSVs were
actually produced - separate script runs, not two calls in one
process), and check whether train_mse is bit-identical. If yes, the
training path is deterministic WITHIN a fixed environment and the 2-4x
gap is purely an environment artifact (wheel build / BLAS / XLA
codegen tied to the Python-version-specific jaxlib wheel), not evidence
that seed-indexed results are untrustworthy in general. If no, that is
a materially different and more serious finding.

Run TWICE, separately:
    python -m layernorm_study.experiments.round3_determinism_check
    python -m layernorm_study.experiments.round3_determinism_check
then diff the two printed values (and the two JSON files this writes).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from layernorm_study.src.arms import ARMS, train_arm
from layernorm_study.src.plant2 import L_MAX, generate_data

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
SEED = 0


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    t0 = time.time()
    param_state, mse = train_arm(
        ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
        epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=SEED,
    )
    elapsed = time.time() - t0

    # full bit pattern, not just the printed float - repr() on a Python
    # float already round-trips exactly, but be explicit about it.
    out_path = RESULTS_DIR / f"round3_determinism_check_run{int(time.time() * 1000)}.json"
    with out_path.open("w") as f:
        json.dump({"seed": SEED, "arm": "arm_6", "epochs": EPOCHS, "train_mse": mse,
                    "train_mse_hex": mse.hex(), "elapsed_s": elapsed}, f, indent=2)
    print(f"arm_6 seed={SEED}: train_mse={mse!r} (hex={mse.hex()}) elapsed={elapsed:.1f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
