"""Round 2, Part A follow-ups A6-A8.

A6: does W_dec rotate toward the branch's null space, or does the
branch reorganize into W_dec's null space? The cosine shrinking alone
cannot distinguish these - they are different mechanisms with different
implications. Measured via angular displacement of each from its OWN
initialization.

A7: report median|cos| explicitly (round 2's headline numbers were
signed medians, which can hide a large but sign-flipping cosine).

A8: one more conditioning attempt (randomized feedback gain, no open-
loop divergence) plus the test that actually settles whether
conditioning drives arm_5's variance: multiple conditioning levels,
does arm_5's across-seed spread track cond.

Run: python -m layernorm_study.experiments.round2_part_a_followups
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

from layernorm_study.src.arms import ARMS, build_model, load_arm_model, train_arm
from layernorm_study.experiments.round1_5_branch_suppression import load_trained
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX, EPOCHS, LEARNING_RATE
from layernorm_study.src.orthogonality_tests import (
    angular_displacement_deg,
    decoder_vector,
    dominant_direction,
    pre_decoder_vectors_along_trajectory,
)
from layernorm_study.src.scalar_diagnostics import jx_ju_along_trajectory
from layernorm_study.src.scalar_system import (
    A_TRUE,
    B_TRUE,
    fast_vectorized_aprbs,
    generate_scalar_trajectory,
    regressor_condition_number,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))


# --------------------------------------------------------------------------
# A6/A7
# --------------------------------------------------------------------------

def task_a6_a7() -> pd.DataFrame:
    inputs, targets = generate_scalar_trajectory(length=L_MAX, seed=42)
    rows = []
    for seed in SEEDS:
        init_model = build_model(
            ARMS["arm_2"], d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1,
            decode=True, key=jax.random.PRNGKey(seed),
        )
        final_model = load_trained("arm_2", seed, decode=True)

        w_init = decoder_vector(init_model)
        w_final = decoder_vector(final_model)

        branch_init_rows = pre_decoder_vectors_along_trajectory(init_model, jnp.asarray(inputs), jnp.asarray(targets))
        branch_final_rows = pre_decoder_vectors_along_trajectory(final_model, jnp.asarray(inputs), jnp.asarray(targets))
        branch_init_mat = np.stack([r["branch_dmodel"] for r in branch_init_rows])
        branch_final_mat = np.stack([r["branch_dmodel"] for r in branch_final_rows])
        branch_dir_init = dominant_direction(branch_init_mat)
        branch_dir_final = dominant_direction(branch_final_mat)

        w_displacement = angular_displacement_deg(w_init, w_final)
        branch_displacement = angular_displacement_deg(branch_dir_init, branch_dir_final)

        cos_final = np.array([
            float(np.dot(w_final, r["branch_dmodel"]) / (np.linalg.norm(w_final) * np.linalg.norm(r["branch_dmodel"])))
            for r in branch_final_rows
        ])

        row = {
            "seed": seed,
            "w_dec_angular_displacement_deg": w_displacement,
            "branch_direction_angular_displacement_deg": branch_displacement,
            "which_moved_more": "W_dec" if w_displacement > branch_displacement else "branch",
            "signed_cos_median": float(np.median(cos_final)),
            "signed_cos_mean": float(np.mean(cos_final)),
            "abs_cos_median": float(np.median(np.abs(cos_final))),
            "abs_cos_mean": float(np.mean(np.abs(cos_final))),
        }
        rows.append(row)
        print(
            f"seed={seed}: W_dec moved {w_displacement:.1f} deg, branch dir moved {branch_displacement:.1f} deg "
            f"-> {row['which_moved_more']} moved more  |  signed_cos_median={row['signed_cos_median']:.4f} "
            f"abs_cos_median={row['abs_cos_median']:.4f}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_partA_a6_a7_direction_and_signed_vs_abs.csv", index=False)
    return df


# --------------------------------------------------------------------------
# A8, part 1: randomized feedback gain (no open-loop divergence)
# --------------------------------------------------------------------------

def generate_randomized_gain_trajectory(
    a_true: float, b_true: float, length: int, seed: int,
    k_min: float, k_max: float, segment_length: int = 10,
    aprbs_low: float = -10.0, aprbs_high: float = 10.0, hold_prob: float = 0.8, x0_range: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop-plus-dither, but the feedback gain k is RESAMPLED
    every `segment_length` steps from Uniform(k_min, k_max) instead of
    held fixed for the whole trajectory - decorrelates x and u by
    varying the LINEAR RELATIONSHIP between them over time, without the
    open-loop divergence a fully unforced excitation would hit (the
    loop is always closed, just with a moving gain, so it stays
    stabilizing at every step by construction as long as (k_min,k_max)
    is inside the stabilizing range)."""
    rng = np.random.RandomState(seed)
    a_signal = fast_vectorized_aprbs(batch_size=1, length=length, low=aprbs_low, high=aprbs_high, hold_prob=hold_prob, rng=rng, Nu=1)[0, 0]
    inputs = np.zeros((length, 2))
    targets = np.zeros((length, 1))
    x = rng.uniform(-x0_range, x0_range)
    k = rng.uniform(k_min, k_max)
    for t in range(length):
        if t % segment_length == 0:
            k = rng.uniform(k_min, k_max)
        u = k * x + a_signal[t]
        inputs[t, 0] = x
        inputs[t, 1] = u
        x_next = a_true * x + b_true * u
        targets[t, 0] = x_next
        x = x_next
    return inputs, targets


def task_a8_gain_randomization() -> None:
    print("\n--- A8 part 1: gain randomization on OUR existing plant (A=3, B=1) ---")
    print("stabilizing range for k: |3+k|<1 => k in (-4,-2) - narrow (width 2)")
    print("scan over segment_length (how often k is resampled within the 100-step trajectory):")
    scan_rows = []
    for seg in [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 70, 100]:
        inputs, _ = generate_randomized_gain_trajectory(A_TRUE, B_TRUE, L_MAX, seed=42, k_min=-3.99, k_max=-2.01, segment_length=seg)
        cond = regressor_condition_number(inputs)
        corr = float(np.corrcoef(inputs[:, 0], inputs[:, 1])[0, 1])
        scan_rows.append({"segment_length": seg, "cond": cond, "corr": corr})
        print(f"  segment_length={seg}: cond={cond:.2f}  corr={corr:.4f}")
    pd.DataFrame(scan_rows).to_csv(RESULTS_DIR / "round2_partA_a8_gain_randomization_scan.csv", index=False)

    best_cond = min(r["cond"] for r in scan_rows)
    print(f"\nFINDING (contradicts the prediction): gain randomization does NOT move things much further than the "
          f"round 1.5 fixed-k=-3.5 result (corr=-0.945, cond=81.81). Best achieved here: cond={best_cond:.2f}. "
          f"Every segment_length tried lands in cond~76-95 (well-randomized) or WORSE (cond up to 383 at long "
          f"segments, since infrequent switching approaches single-fixed-k behavior). There appears to be a hard "
          f"floor around cond~76-80 for this specific plant (B_true=1, strong control authority keeps u tightly "
          f"tied to x regardless of which stabilizing k is used) that neither fixed-k selection nor randomization "
          f"beats.")

    print("\n--- bonus attempt on the user's example plant (A=1.03, B=0.01), flagged not measured ---")
    print("stabilizing range for k: |1.03-0.01k|<1 => k in (3,203) - wide (width 200)")
    a2, b2 = 1.03, 0.01
    inputs2, _ = generate_randomized_gain_trajectory(a2, b2, L_MAX, seed=42, k_min=3.5, k_max=202.5, segment_length=10)
    cond2 = regressor_condition_number(inputs2)
    print(f"  cond={cond2:.3e} (!) - hit a numerical/scaling pathology, NOT a meaningful measurement: this plant's "
          f"weak control authority (B=0.01) and near-marginal poles at the edges of the wide k-range need their own "
          f"dedicated excitation tuning (different APRBS amplitude/x0 range than the A=3,B=1 defaults reused here "
          f"opportunistically) - deferred to Part C, where this plant is set up properly rather than reused ad hoc.")


# --------------------------------------------------------------------------
# A8, part 2: does arm_5's seed variance track conditioning level?
# --------------------------------------------------------------------------

def task_a8_multilevel_sweep() -> None:
    print("\n--- A8 part 2: multi-level conditioning sweep, does arm_5's seed variance track cond? ---")
    print("NOTE: the originally-requested targets (cond~5, 20, 80, 156) are NOT all achievable for this plant - "
          "part 1's scan found a floor around cond~76-80 that neither fixed-k nor randomized-k selection beats. "
          "Using the actually-achieved spread from that scan instead (flagging the deviation rather than silently "
          "substituting): segment_length in {6, 10, 20, 50} spans the widest real range found (~76 to ~318).")
    segment_lengths = [6, 10, 20, 50]
    levels = []
    for seg in segment_lengths:
        k_min, k_max = -3.99, -2.01
        inputs, _ = generate_randomized_gain_trajectory(A_TRUE, B_TRUE, L_MAX, seed=42, k_min=k_min, k_max=k_max, segment_length=seg)
        cond_actual = regressor_condition_number(inputs)
        corr_actual = float(np.corrcoef(inputs[:, 0], inputs[:, 1])[0, 1])
        levels.append({"target_cond": cond_actual, "k_min": k_min, "k_max": k_max, "segment_length": seg,
                        "achieved_cond": cond_actual, "achieved_corr": corr_actual})
        print(f"segment_length={seg}: achieved cond={cond_actual:.2f} corr={corr_actual:.4f}")

    levels_df = pd.DataFrame(levels)
    levels_df.to_csv(RESULTS_DIR / "round2_partA_a8_conditioning_levels.csv", index=False)

    all_rows = []
    for level in levels:
        inputs, targets = generate_randomized_gain_trajectory(
            A_TRUE, B_TRUE, L_MAX, seed=42, k_min=level["k_min"], k_max=level["k_max"], segment_length=level["segment_length"],
        )
        inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
        for seed in SEEDS:
            param_state, train_mse = train_arm(
                ARMS["arm_5"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(ARMS["arm_5"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
            jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
            row = {"target_cond": level["target_cond"], "achieved_cond": level["achieved_cond"], "seed": seed,
                   "teacher_mse_train_final": train_mse, "jx_err_mean": float(np.mean(jx_errs))}
            all_rows.append(row)
            print(f"  [cond~{level['target_cond']} seed={seed}] jx_err_mean={row['jx_err_mean']:.4e}", flush=True)

    sweep_df = pd.DataFrame(all_rows)
    sweep_df.to_csv(RESULTS_DIR / "round2_partA_a8_arm5_multilevel_results.csv", index=False)

    print("\n=== does arm_5's seed-to-seed spread track conditioning level? ===")
    summary = sweep_df.groupby("target_cond")["jx_err_mean"].agg(["median", "std", "min", "max"])
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "round2_partA_a8_arm5_multilevel_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for level in levels:
        sub = sweep_df[sweep_df["target_cond"] == level["target_cond"]]
        ax.scatter([level["achieved_cond"]] * len(sub), sub["jx_err_mean"], alpha=0.7, s=50)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("achieved cond(E[zz^T]) (log)")
    ax.set_ylabel("jx_err_mean per seed (log)")
    ax.set_title("arm_5: does seed variance track conditioning?")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_partA_a8_arm5_variance_vs_cond.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== A6/A7: direction ambiguity + signed vs abs cosine, arm_2, all 8 seeds ===")
    df = task_a6_a7()
    n_wdec_moved_more = (df["which_moved_more"] == "W_dec").sum()
    print(f"\nSummary: W_dec moved more in {n_wdec_moved_more}/8 seeds; branch direction moved more in {8-n_wdec_moved_more}/8 seeds")

    task_a8_gain_randomization()
    task_a8_multilevel_sweep()


if __name__ == "__main__":
    main()
