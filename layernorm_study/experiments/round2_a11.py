"""Round 2, A11: close the conditioning range properly.

A10's sweep only reached cond~41 (via reweighting, at the cost of
eff_n~15), so the "flat variance" finding spanned ~4x in cond, not the
~30x the test was designed for. Fix: use A9's short-horizon open-loop-
with-resets data DIRECTLY as a training set (cond~10 at FULL eff_n=100,
PE held fixed by construction - no reweighting needed at all), and add
it as the low-cond anchor to A10's sweep.

Run: python -m layernorm_study.experiments.round2_a11
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.conditioning import open_loop_with_resets
from layernorm_study.src.scalar_diagnostics import jx_ju_along_trajectory
from layernorm_study.src.scalar_system import A_TRUE, B_TRUE, regressor_condition_number
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX, EPOCHS, LEARNING_RATE

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== A11: reset-based data (cond~10, full eff_n) as the low-cond anchor ===")
    inputs, targets = open_loop_with_resets(A_TRUE, B_TRUE, L_MAX, reset_every=3, seed=42, aprbs_low=-10, aprbs_high=10, x0_range=2.0)
    cond = regressor_condition_number(inputs)
    print(f"reset_every=3 data: cond={cond:.2f}, eff_n=100 (uniform weights, full dataset used, no reweighting)")

    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    rows = []
    for seed in SEEDS:
        param_state, train_mse = train_arm(
            ARMS["arm_5"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_5"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
        jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
        row = {"label": "reset_based_cond10", "cond": cond, "eff_n": 100.0, "seed": seed,
               "teacher_mse_train_final": train_mse, "jx_err_mean": float(np.mean(jx_errs))}
        rows.append(row)
        print(f"  [reset_based seed={seed}] jx_err_mean={row['jx_err_mean']:.4e}", flush=True)

    new_df = pd.DataFrame(rows)
    new_df.to_csv(RESULTS_DIR / "round2_A11_arm5_resetbased_results.csv", index=False)

    a10_df = pd.read_csv(RESULTS_DIR / "round2_A10_arm5_reweighted_results.csv")
    combined = pd.concat([new_df, a10_df], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "round2_A11_combined_results.csv", index=False)

    print("\n=== combined summary: does the spread track cond across the FULL ~15x range now? ===")
    summary = combined.groupby("label").agg(cond=("cond", "first"), eff_n=("eff_n", "first"),
                                             median_err=("jx_err_mean", "median"), std_err=("jx_err_mean", "std"),
                                             min_err=("jx_err_mean", "min"), max_err=("jx_err_mean", "max"))
    summary = summary.sort_values("cond")
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "round2_A11_combined_summary.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label in summary.index:
        sub = combined[combined["label"] == label]
        ax.scatter([summary.loc[label, "cond"]] * len(sub), sub["jx_err_mean"], alpha=0.7, s=55, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cond(E[zz^T]) (log)")
    ax.set_ylabel("jx_err_mean per seed (log)")
    ax.set_title("arm_5: full conditioning range, cond~10 to 156")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_A11_arm5_full_range.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
