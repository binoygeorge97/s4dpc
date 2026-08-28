"""Part C, C1: bias ablation - the decisive test.

(a) With-bias baseline (arm_6/postnorm, plant2, 8 seeds): confirms the
    "good region near the origin, drifting/decaying away from it"
    finding this whole section explains.
(b) No-bias ablation (bias_ablation.freeze_all_biases_at_zero, same
    arm/plant/seeds): the theory predicts the good region VANISHES
    entirely (||J|| ~ 1/r everywhere, no plateau at any radius).
(c) Converse: scale the TRAINED encoder bias by k in {0.25,0.5,1,2,4}
    (post-hoc weight surgery, no retraining) and check whether the
    measured r* (from a real origin sweep on the modified model) AND
    the theory's own predicted r* (postnorm_geometry.predicted_r_star,
    recomputed fresh after each scaling) both track k, and agree with
    each other.

Run: python -m layernorm_study.experiments.round2_partC_c1_bias_ablation
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
from scipy import stats

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm, train_arm_no_bias
from layernorm_study.src.orthogonality_tests import kink_amplitude_and_r_star
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data
from layernorm_study.src.postnorm_geometry import predicted_r_star
from layernorm_study.src.scalar_diagnostics import jx_vs_c_sweep

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
C_VALUES = np.concatenate([-np.logspace(3, -6, 60), np.logspace(-6, 3, 60)])
BIAS_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    print("=== C1(a): with-bias baseline, arm_6/postnorm, plant2 ===")
    baseline_rows = []
    baseline_models = {}
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        baseline_models[seed] = model
        sweep = jx_vs_c_sweep(model, np.array([1.0]), C_VALUES)
        amp, rstar = kink_amplitude_and_r_star(sweep)
        pred = predicted_r_star(model)
        row = {"seed": seed, "train_mse": mse, "amplitude": amp, "r_star_measured": rstar, **pred}
        baseline_rows.append(row)
        print(f"  seed={seed}: train_mse={mse:.3e} amplitude={amp:.4f} r_star_measured={rstar:.4e} "
              f"r_star_predicted={pred['r_star_predicted']:.4e}", flush=True)
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(RESULTS_DIR / "round2_C1a_withbias_baseline.csv", index=False)

    print("\n=== C1(b): no-bias ablation, same arm/plant/seeds ===")
    nobias_rows = []
    for seed in SEEDS:
        param_state, mse = train_arm_no_bias(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        sweep = jx_vs_c_sweep(model, np.array([1.0]), C_VALUES)
        amp, rstar = kink_amplitude_and_r_star(sweep)
        row = {"seed": seed, "train_mse": mse, "amplitude": amp, "r_star_measured": rstar}
        nobias_rows.append(row)
        print(f"  seed={seed}: train_mse={mse:.3e} amplitude={amp:.4f} r_star_measured={rstar:.4e}", flush=True)
    nobias_df = pd.DataFrame(nobias_rows)
    nobias_df.to_csv(RESULTS_DIR / "round2_C1b_nobias_ablation.csv", index=False)

    print("\n=== C1(c): bias-scale converse test (post-hoc weight surgery on the ALREADY-TRAINED "
          "C1(a) baseline models - no retraining) ===")
    scale_rows = []
    for seed in SEEDS:
        original_bias = jnp.array(baseline_models[seed].encoder.bias.value)  # save once, before any scaling
        for k in BIAS_SCALES:
            model = baseline_models[seed]
            model.encoder.bias.value = original_bias * k  # absolute scale from the ORIGINAL, not compounding
            sweep = jx_vs_c_sweep(model, np.array([1.0]), C_VALUES)
            amp, rstar = kink_amplitude_and_r_star(sweep)
            pred = predicted_r_star(model)
            row = {"seed": seed, "k": k, "amplitude": amp, "r_star_measured": rstar, **pred}
            scale_rows.append(row)
            print(f"  seed={seed} k={k}: r_star_measured={rstar:.4e} r_star_predicted={pred['r_star_predicted']:.4e} "
                  f"Pb_norm={pred['Pb_norm']:.4f}", flush=True)
        baseline_models[seed].encoder.bias.value = original_bias  # restore for cleanliness
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(RESULTS_DIR / "round2_C1c_bias_scale_converse.csv", index=False)

    print("\n=== C1 summary ===")
    print(f"C1(a) with-bias: median amplitude={baseline_df['amplitude'].median():.4f}, "
          f"median r*_measured={baseline_df['r_star_measured'].median():.4e}")
    print(f"C1(b) no-bias:   median amplitude={nobias_df['amplitude'].median():.4f}, "
          f"median r*_measured={nobias_df['r_star_measured'].median():.4e}")

    # predicted vs measured r*, and linear-scaling check, across seeds
    log_Pb = np.log10(scale_df["Pb_norm"].values)
    log_rmeas = np.log10(scale_df["r_star_measured"].values)
    slope, intercept, r, p, se = stats.linregress(log_Pb, log_rmeas)
    print(f"\nC1(c) log-log fit: r_star_measured ~ Pb_norm^{slope:.3f} (predicted slope=1.0 if r* scales linearly "
          f"with ||Pb||) r^2={r**2:.3f} p={p:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(scale_df["Pb_norm"], scale_df["r_star_predicted"], alpha=0.5, label="predicted", marker="x")
    axes[0].scatter(scale_df["Pb_norm"], scale_df["r_star_measured"], alpha=0.5, label="measured")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("||Pb|| (log)")
    axes[0].set_ylabel("r* (log)")
    axes[0].set_title(f"predicted vs measured r* (slope={slope:.2f})")
    axes[0].legend()

    axes[1].scatter([0] * len(baseline_df), baseline_df["r_star_measured"], alpha=0.6, label="with-bias")
    axes[1].scatter([1] * len(nobias_df), nobias_df["r_star_measured"], alpha=0.6, label="no-bias")
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["with-bias", "no-bias"])
    axes[1].set_ylabel("r* measured (log)")
    axes[1].set_title("does the good region vanish without bias?")
    axes[1].legend()

    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C1_bias_ablation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    with (RESULTS_DIR / "round2_C1_summary.txt").open("w") as f:
        f.write(f"withbias_median_rstar={baseline_df['r_star_measured'].median()}\n")
        f.write(f"nobias_median_rstar={nobias_df['r_star_measured'].median()}\n")
        f.write(f"scale_slope={slope}, r2={r**2}, p={p}\n")


if __name__ == "__main__":
    main()
