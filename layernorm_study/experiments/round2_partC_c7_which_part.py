"""Part C, C7: which part of LayerNorm does the damage.

Three arms, all postnorm, otherwise identical to arm_6 (GELU+GLU, real
S4 memory), varying only the norm type:
  - rmsnorm: drops mean-centering, keeps degree-0 scaling -> should
    fail IDENTICALLY to arm_6/LayerNorm if 1/sigma is the sole culprit.
  - frozen_sigma: per-input mean-centering, FIXED denominator -> breaks
    degree-0 -> should FIX it.
  - centering_only: same construction as frozen_sigma, labeled
    separately per the task's own spec.

Run: python -m layernorm_study.experiments.round2_partC_c7_which_part
"""
from __future__ import annotations

import dataclasses
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

from layernorm_study.src.arms import ARMS, ArmSpec, load_arm_model, train_arm
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data
from layernorm_study.src.scalar_diagnostics import jx_ju_along_trajectory

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3

NORM_VARIANTS = ["rmsnorm", "frozen_sigma", "centering_only"]


def arm6_with_norm(norm_type: str) -> ArmSpec:
    base = ARMS["arm_6"]
    kwargs = dict(base.block_kwargs)
    kwargs["norm"] = norm_type
    return dataclasses.replace(base, block_kwargs=kwargs)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for norm_type in NORM_VARIANTS:
        arm = arm6_with_norm(norm_type)
        print(f"=== {norm_type} ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
            jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
            row = {"norm_type": norm_type, "seed": seed, "train_mse": mse,
                   "jx_err_mean": float(np.mean(jx_errs)), "jx_err_max": float(np.max(jx_errs))}
            rows.append(row)
            print(f"  seed={seed}: train_mse={mse:.3e} jx_err_mean={row['jx_err_mean']:.3e} "
                  f"jx_err_max={row['jx_err_max']:.3e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C7_which_part_results.csv", index=False)

    print("\n=== summary (median across 8 seeds) ===")
    summary = df.groupby("norm_type")[["jx_err_mean", "jx_err_max"]].median()
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "round2_C7_which_part_summary.csv")

    baseline_path = RESULTS_DIR / "round2_C1a_withbias_baseline.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        print(f"\n(for comparison, arm_6/real LayerNorm baseline from C1(a) - if available)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = NORM_VARIANTS
    data = [df[df["norm_type"] == n]["jx_err_mean"].values for n in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_yscale("log")
    ax.set_ylabel("jx_err_mean (log)")
    ax.set_title("C7: which part of LN does the damage")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C7_which_part.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
