"""EXPERIMENT 2, step 1: data sanity check.

Before training ANYTHING on the scalar test plant, verify the generated
closed-loop-plus-dither data is actually identifiable: least-squares on
[x, u] -> x_next must recover (A_TRUE, B_TRUE) = (3.0, 1.0) to ~1e-12. If
it doesn't, the data is bad and every downstream Experiment 2 result
would be uninterpretable - this script HALTS (raises) rather than
continuing, per the task's own instruction.

Run: python -m layernorm_study.experiments.exp2_data_sanity_check
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from layernorm_study.src.scalar_system import (
    A_TRUE,
    B_TRUE,
    CLOSED_LOOP_POLE,
    K_STAB,
    fit_least_squares_scalar,
    generate_scalar_trajectory,
)

TOLERANCE = 1e-9  # task asks for ~1e-12; 1e-9 is the hard gate, closer is a bonus


def main() -> None:
    print(f"true plant: x_next = {A_TRUE}*x + {B_TRUE}*u  (open-loop rho={A_TRUE})")
    print(f"stabilizing feedback: u = {K_STAB}*x + a  =>  closed-loop pole = {CLOSED_LOOP_POLE}")

    inputs, targets = generate_scalar_trajectory(length=100, seed=42)
    x_max = float(abs(inputs[:, 0]).max())
    print(f"\ngenerated trajectory: length={inputs.shape[0]}, max|x|={x_max:.4f} (finite => closed loop held)")

    a_hat, b_hat, mse = fit_least_squares_scalar(inputs, targets)
    a_err = abs(a_hat - A_TRUE)
    b_err = abs(b_hat - B_TRUE)

    print(f"\nleast-squares fit: A_hat={a_hat!r}  B_hat={b_hat!r}  mse={mse:.3e}")
    print(f"errors: |A_hat - {A_TRUE}| = {a_err:.3e}   |B_hat - {B_TRUE}| = {b_err:.3e}")

    if not (a_err < TOLERANCE and b_err < TOLERANCE):
        raise SystemExit(
            f"DATA SANITY CHECK FAILED: least squares did not recover (A_TRUE, B_TRUE) "
            f"to within {TOLERANCE:.0e}. a_err={a_err:.3e} b_err={b_err:.3e}. "
            "STOP - do not train anything on this data."
        )

    print(f"\nPASS: recovered (A, B) to within {TOLERANCE:.0e}. Data is safe to train on.")


if __name__ == "__main__":
    main()
