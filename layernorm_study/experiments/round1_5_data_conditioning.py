"""Round 1.5, Task 3: is arm_0's slow convergence (needing 60000 epochs,
failing at 20000 on one seed for a 2-parameter exactly-affine fit) a
data-conditioning artifact rather than a model-expressiveness one?

cond(E[z z^T]) is the condition number of the (up to a constant factor)
Hessian of the teacher-forced MSE loss - it governs gradient-descent
convergence rate directly, independent of model capacity. Computed for
round 1's data (k_stab=-2.7, pole=0.3): cond=156, corr(x,u)=-0.977 -
u=k_stab*x+a with a independent of x drives this (closed-form:
corr = k_stab/sqrt(k_stab^2+1-pole^2), pole=A_TRUE+B_TRUE*k_stab).

A parameter scan (k_stab in the stabilizing range (-4,-2), several APRBS
amplitudes/hold_probs) found APRBS amplitude barely moves cond (the
closed-form doesn't depend on it), but k_stab does: k_stab=-3.5
(pole=-0.5) empirically minimizes it at cond~82, a real ~47% reduction,
not a dramatic one - there appears to be a floor to how decorrelated a
PURELY PROPORTIONAL stabilizing loop can make (x, u) for this specific
unstable plant, without changing the excitation paradigm entirely (e.g.
an independent reference dither, out of scope for this round).

Re-runs arm_0 and arm_5 on the k_stab=-3.5 data at the SAME PINNED 60000
epochs (per this round's own methodological note: do not re-tune the
epoch budget again) and same 8 seeds, to check whether the conclusions
move.

Run: python -m layernorm_study.experiments.round1_5_data_conditioning
"""
from __future__ import annotations

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.scalar_diagnostics import jx_ju_along_trajectory, teacher_forced_mse
from layernorm_study.src.scalar_system import (
    A_TRUE,
    B_TRUE,
    K_STAB,
    fit_least_squares_scalar,
    generate_scalar_trajectory,
    regressor_condition_number,
)
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX, EPOCHS, LEARNING_RATE

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = list(range(8))
K_STAB_DECORRELATED = -3.5  # pole=-0.5; empirically minimizes cond within the stabilizing range - see module docstring


def run_arm_on_data(arm_name: str, inputs: np.ndarray, targets: np.ndarray, seed: int) -> dict:
    arm = ARMS[arm_name]
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    param_state, train_mse = train_arm(
        arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
        epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
    )
    model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
    teacher_mse = teacher_forced_mse(model, inputs_j, targets_j)
    traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
    jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
    return {"arm": arm_name, "seed": seed, "teacher_mse_train_final": train_mse,
            "teacher_mse": teacher_mse, "jx_err_mean": float(np.mean(jx_errs)), "jx_err_max": float(np.max(jx_errs))}


def main() -> None:
    print("=== condition number: round 1's data (k_stab={:.2f}) ===".format(K_STAB))
    inputs_orig, targets_orig = generate_scalar_trajectory(length=L_MAX, seed=42, k_stab=K_STAB)
    cond_orig = regressor_condition_number(inputs_orig)
    a_hat, b_hat, ls_mse = fit_least_squares_scalar(inputs_orig, targets_orig)
    print(f"cond(E[zz^T]) = {cond_orig:.2f}   least-squares (A,B)=({a_hat!r},{b_hat!r}) mse={ls_mse:.3e}")

    print(f"\n=== condition number: decorrelated data (k_stab={K_STAB_DECORRELATED}) ===")
    inputs_new, targets_new = generate_scalar_trajectory(length=L_MAX, seed=42, k_stab=K_STAB_DECORRELATED)
    cond_new = regressor_condition_number(inputs_new)
    a_hat2, b_hat2, ls_mse2 = fit_least_squares_scalar(inputs_new, targets_new)
    print(f"cond(E[zz^T]) = {cond_new:.2f}   least-squares (A,B)=({a_hat2!r},{b_hat2!r}) mse={ls_mse2:.3e}")
    print(f"reduction: {cond_orig:.1f} -> {cond_new:.1f} ({100*(1-cond_new/cond_orig):.1f}% lower)")

    if abs(a_hat2 - A_TRUE) > 1e-9 or abs(b_hat2 - B_TRUE) > 1e-9:
        raise SystemExit("DATA SANITY CHECK FAILED on decorrelated data - STOP, do not train on it.")
    print("PASS: decorrelated data's least-squares fit still recovers (A,B) to <1e-9.")

    rows = []
    for arm_name in ["arm_0", "arm_5"]:
        print(f"\n=== retraining {arm_name} on decorrelated data (k_stab={K_STAB_DECORRELATED}), pinned {EPOCHS} epochs ===")
        for seed in SEEDS:
            row = run_arm_on_data(arm_name, inputs_new, targets_new, seed)
            rows.append(row)
            print(f"  seed={seed}: teacher_mse={row['teacher_mse']:.3e} jx_err_mean={row['jx_err_mean']:.3e} "
                  f"jx_err_max={row['jx_err_max']:.3e}", flush=True)

    new_df = pd.DataFrame(rows)
    new_df.to_csv(RESULTS_DIR / "round1_5_decorrelated_data_results.csv", index=False)

    ladder_df = pd.read_csv(RESULTS_DIR / "exp2_ladder.csv")
    print("\n=== comparison: original vs decorrelated data ===")
    for arm_name in ["arm_0", "arm_5"]:
        orig = ladder_df[ladder_df["arm"] == arm_name][["seed", "jx_err_mean"]].rename(columns={"jx_err_mean": "jx_err_mean_orig_data"})
        new = new_df[new_df["arm"] == arm_name][["seed", "jx_err_mean"]].rename(columns={"jx_err_mean": "jx_err_mean_decorrelated_data"})
        merged = orig.merge(new, on="seed")
        print(f"\n{arm_name}:")
        print(merged.to_string(index=False))
        print(f"  median (orig data): {merged['jx_err_mean_orig_data'].median():.3e}")
        print(f"  median (decorrelated data): {merged['jx_err_mean_decorrelated_data'].median():.3e}")
        merged.to_csv(RESULTS_DIR / f"round1_5_{arm_name}_data_comparison.csv", index=False)


if __name__ == "__main__":
    main()
