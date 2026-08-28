"""Part C, C6-revised.

C6 (original) is WITHDRAWN - see NOTES.md. The prediction that
postnorm's good region RELOCATES to a training center x_ref was wrong
on the theory's own terms: the near-field ball ||PMz||<<||Pb|| is
always centered at z=0; increasing the bias enlarges that ball, it
never moves it to an annulus elsewhere. Getting good behavior AT
||z||=50 needs ||Pb|| >> 50*sigma_max(PM), which gives a ball that
ALSO contains the origin.

This script tests what the observed C6 numbers actually were evidence
of instead:
  (a-d) C6-revised: is J_x~0 at x=50 actually output SATURATION
        (Theorem 2 - Y_max too small to represent the ~51.5 target),
        not a relocation failure? Compares Y_max (from trained
        weights) against the actual required output magnitude, plots
        predicted vs true target across the training data, reports
        max|F(z)|/Y_max.
  (e)   ENLARGEMENT test - the theory's ACTUAL prediction: scale
        b_enc up by 10x/50x/200x on ORIGIN-centered data (not x_ref=50
        data) and confirm r* grows, with the good region always
        containing the origin (Jx at z=0 stays close to true
        throughout, not just far from it).

Run: python -m layernorm_study.experiments.round2_partC_c6_revised
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

from s4dpc.diagnostics import step, zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.output_ceiling import compute_y_max
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data, generate_data_at_radius
from layernorm_study.src.postnorm_geometry import predicted_r_star
from layernorm_study.src.scalar_diagnostics import _jit_jxju

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
X_REF = 50.0


def teacher_forced_predictions(model, inputs_j, targets_j):
    d_x = targets_j.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    preds = []
    for t in range(inputs_j.shape[0]):
        x_next, states = step(model, inputs_j[t, :d_x], inputs_j[t, d_x:], states)
        preds.append(float(x_next[0]))
    return np.array(preds)


def c6_revised_ceiling_check() -> pd.DataFrame:
    inputs, targets = generate_data_at_radius(seed=42, length=L_MAX, x0_center=X_REF, x0_spread=8.0,
                                               aprbs_low=-10.0, aprbs_high=10.0, reset_every=20)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    required_at_ref = A_TRUE * X_REF  # ~51.5
    target_min, target_max = float(np.abs(targets).min()), float(np.abs(targets).max())
    print(f"required output at x_ref: {required_at_ref:.2f}; actual target range: [{target_min:.2f}, {target_max:.2f}]")

    rows = []
    preds_seed0 = None
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        y_max = compute_y_max(model)
        preds = teacher_forced_predictions(model, inputs_j, targets_j)
        max_pred = float(np.max(np.abs(preds)))
        row = {
            "seed": seed, "train_mse": mse, "y_max": y_max, "required_at_ref": required_at_ref,
            "target_max": target_max, "max_pred": max_pred,
            "ratio_ymax_to_required": y_max / required_at_ref,
            "ratio_maxpred_to_ymax": max_pred / y_max if y_max else float("nan"),
        }
        rows.append(row)
        print(f"  seed={seed}: Y_max={y_max:.4f} required={required_at_ref:.2f} target_max={target_max:.2f} "
              f"max|pred|={max_pred:.4f} Y_max/required={row['ratio_ymax_to_required']:.4f} "
              f"max_pred/Y_max={row['ratio_maxpred_to_ymax']:.4f}", flush=True)
        if seed == 0:
            preds_seed0 = preds

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C6revised_ceiling_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    t = np.arange(len(preds_seed0))
    ax.plot(t, np.abs(targets[:, 0]), color="tab:red", label="|true target|")
    ax.plot(t, np.abs(preds_seed0), color="black", label="|postnorm prediction|")
    ax.axhline(df["y_max"].iloc[0], color="tab:blue", linestyle=":", label=f"Y_max={df['y_max'].iloc[0]:.2f}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("|output|")
    ax.set_title("C6-revised: predicted output saturates at Y_max while targets keep growing")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C6revised_ceiling.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return df


def c6_revised_enlargement_test() -> pd.DataFrame:
    inputs, targets = generate_data(seed=42)  # origin-centered, plant2's default
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    scales = [1.0, 10.0, 50.0, 200.0]

    rows = []
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        original_bias = jnp.array(model.encoder.bias.value)
        states0 = zero_states(model, dtype=jnp.complex128)
        u0 = jnp.zeros((1,))
        for k in scales:
            model.encoder.bias.value = original_bias * k
            pred = predicted_r_star(model)
            jx_at_origin, _ = _jit_jxju(model, jnp.zeros((1,)), u0, states0)
            row = {"seed": seed, "k": k, "r_star_predicted": pred["r_star_predicted"],
                   "Pb_norm": pred["Pb_norm"], "jx_at_origin": float(jx_at_origin)}
            rows.append(row)
            print(f"  seed={seed} k={k}: r*_predicted={pred['r_star_predicted']:.4e} "
                  f"Pb_norm={pred['Pb_norm']:.4f} Jx_at_origin={float(jx_at_origin):.4f} (true={A_TRUE})", flush=True)
        model.encoder.bias.value = original_bias

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C6revised_enlargement_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for seed in SEEDS:
        sub = df[df["seed"] == seed].sort_values("k")
        ax.plot(sub["k"], sub["r_star_predicted"], alpha=0.5, marker="o", color="black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bias scale k (log)")
    ax.set_ylabel("predicted r* (log)")
    ax.set_title("C6-revised enlargement test: does r* grow with bias scale?")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C6revised_enlargement.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== C6-revised (a-d): output ceiling check at x_ref=50 ===")
    ceiling_df = c6_revised_ceiling_check()
    print("\n=== C6-revised (e): enlargement test, origin-centered data ===")
    enlargement_df = c6_revised_enlargement_test()

    print("\n=== C6-revised summary ===")
    print(f"median Y_max/required_at_ref: {ceiling_df['ratio_ymax_to_required'].median():.4f} "
          f"(< 1 means the model structurally CANNOT reach the required output)")
    print(f"median max|pred|/Y_max: {ceiling_df['ratio_maxpred_to_ymax'].median():.4f} "
          f"(close to 1 means predictions are pinned near the ceiling)")

    slope_rows = []
    for seed in SEEDS:
        sub = enlargement_df[enlargement_df["seed"] == seed].sort_values("k")
        log_k = np.log10(sub["k"].values)
        log_r = np.log10(sub["r_star_predicted"].values)
        slope = np.polyfit(log_k, log_r, 1)[0]
        slope_rows.append(slope)
    print(f"median log-log slope of r* vs k: {np.median(slope_rows):.3f} (predicted 1.0 if r* scales linearly with bias scale)")
    print(f"median Jx_at_origin across all (seed,k): {enlargement_df['jx_at_origin'].median():.4f} (true={A_TRUE}) "
          f"- does the good region keep containing the origin as bias grows?")


if __name__ == "__main__":
    main()
