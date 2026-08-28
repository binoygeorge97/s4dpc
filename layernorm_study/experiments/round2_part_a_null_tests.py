"""Round 2, Part A: null tests for round 1.5's orthogonality claim.

Objection under test: for two independent random vectors in R^H, typical
|cosine| ~ 1/sqrt(H) (0.354 at this study's d_model=8) - near-
orthogonality in high dimensions is the DEFAULT. Round 1.5's single-point
spot-check (cos~0.15-0.16) was BELOW that baseline but computed at one
arbitrary trajectory point, not systematically. This script redoes it
properly: median over the full trajectory, at both init and convergence,
against an explicit null distribution, plus the scale-sensitive
projected-contribution measure cosine cannot provide.

A4/A5 (Task 2 robustness, Task 3 further decorrelation attempt) are
appended at the end of main().

Run: python -m layernorm_study.experiments.round2_part_a_null_tests
"""
from __future__ import annotations

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layernorm_study.src.arms import ARMS, build_model
from layernorm_study.experiments.round1_5_branch_suppression import load_trained
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX
from layernorm_study.src.orthogonality_tests import (
    cosine_with_decoder,
    decoder_vector,
    null_cosine_percentile,
    pre_decoder_vectors_along_trajectory,
    projected_contribution_ratio,
)
from layernorm_study.src.scalar_system import A_TRUE, B_TRUE, generate_scalar_trajectory, regressor_condition_number
from s4dpc.data import fast_vectorized_aprbs

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))


def analyze_model(model, inputs, targets, null_key) -> dict:
    rows = pre_decoder_vectors_along_trajectory(model, jnp.asarray(inputs), jnp.asarray(targets))
    w = decoder_vector(model)
    cos_vals = np.array([cosine_with_decoder(w, r["branch_dmodel"]) for r in rows])
    abs_cos_vals = np.abs(cos_vals)
    ratios = np.array([projected_contribution_ratio(w, r["skip_dmodel"], r["branch_dmodel"]) for r in rows])

    # null test against the MEDIAN-magnitude branch vector (representative,
    # not an arbitrary single t) - use the trajectory point whose |branch|
    # norm is closest to the trajectory's own median norm
    branch_norms = np.array([np.linalg.norm(r["branch_dmodel"]) for r in rows])
    rep_idx = int(np.argmin(np.abs(branch_norms - np.median(branch_norms))))
    null = null_cosine_percentile(w, rows[rep_idx]["branch_dmodel"], null_key, n_samples=2000)

    return {
        "cos_median": float(np.median(cos_vals)),
        "cos_mean": float(np.mean(cos_vals)),
        "abs_cos_median": float(np.median(abs_cos_vals)),
        "projected_ratio_median": float(np.median(ratios)),
        "projected_ratio_mean": float(np.mean(ratios)),
        **null,
    }


def task_a1_a2_a3() -> pd.DataFrame:
    inputs, targets = generate_scalar_trajectory(length=L_MAX, seed=42)
    rows = []
    for seed in SEEDS:
        init_model = build_model(
            ARMS["arm_2"], d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1,
            decode=True, key=jax.random.PRNGKey(seed),
        )
        final_model = load_trained("arm_2", seed, decode=True)

        init_stats = analyze_model(init_model, inputs, targets, jax.random.PRNGKey(seed + 5000))
        final_stats = analyze_model(final_model, inputs, targets, jax.random.PRNGKey(seed + 6000))

        row = {"seed": seed}
        row.update({f"init_{k}": v for k, v in init_stats.items()})
        row.update({f"final_{k}": v for k, v in final_stats.items()})
        row["projected_ratio_shrink_factor"] = (
            row["init_projected_ratio_median"] / row["final_projected_ratio_median"]
            if row["final_projected_ratio_median"] > 0 else float("inf")
        )
        rows.append(row)
        print(
            f"seed={seed}: cos init={row['init_cos_median']:.4f} (pctl={row['init_percentile_of_actual_in_null']:.1f}) "
            f"-> final={row['final_cos_median']:.4f} (pctl={row['final_percentile_of_actual_in_null']:.1f})  "
            f"proj_ratio init={row['init_projected_ratio_median']:.4f} -> final={row['final_projected_ratio_median']:.4f} "
            f"(shrink {row['projected_ratio_shrink_factor']:.1f}x)",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_partA_arm2_orthogonality.csv", index=False)
    return df


def task_a4_spearman() -> None:
    branch_df = pd.read_csv(RESULTS_DIR / "round1_5_arm5_branch_ratio_vs_error.csv")
    pearson_r = float(np.corrcoef(branch_df["branch_skip_ratio_median"], branch_df["jx_err_mean"])[0, 1])
    spearman_r, spearman_p = stats.spearmanr(branch_df["branch_skip_ratio_median"], branch_df["jx_err_mean"])
    print(f"\nTask 2 robustness: Pearson r={pearson_r:.4f}  Spearman rho={spearman_r:.4f} (p={spearman_p:.4f})")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(branch_df["branch_skip_ratio_median"], branch_df["jx_err_mean"], s=60)
    for _, r in branch_df.iterrows():
        ax.annotate(f"seed{int(r['seed'])}", (r["branch_skip_ratio_median"], r["jx_err_mean"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("median branch/skip ratio along trajectory")
    ax.set_ylabel("jx_err_mean")
    ax.set_yscale("log")
    ax.set_title(f"arm_5: Pearson r={pearson_r:.3f}, Spearman rho={spearman_r:.3f}")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_partA_arm5_correlation_labeled.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    with (RESULTS_DIR / "round2_partA_task2_correlations.txt").open("w") as f:
        f.write(f"pearson_r={pearson_r}\nspearman_r={spearman_r}\nspearman_p={spearman_p}\n")


def task_a5_conditioning() -> None:
    inputs_orig, _ = generate_scalar_trajectory(length=L_MAX, seed=42)
    corr_orig = float(np.corrcoef(inputs_orig[:, 0], inputs_orig[:, 1])[0, 1])
    inputs_decorr, _ = generate_scalar_trajectory(length=L_MAX, seed=42, k_stab=-3.5)
    corr_decorr = float(np.corrcoef(inputs_decorr[:, 0], inputs_decorr[:, 1])[0, 1])
    print(f"\nTask 3: corr(x,u) original (k_stab=-2.7) = {corr_orig:.4f}")
    print(f"Task 3: corr(x,u) decorrelated (k_stab=-3.5) = {corr_decorr:.4f}")

    # Genuine attempt at decorrelation: OPEN-LOOP excitation over a short
    # horizon. rho=3 makes a length-100 open-loop trajectory diverge as
    # 3^100 (the original task's own warning) - but a SHORT horizon keeps
    # 3^L bounded (float64 safely handles 3^L up to L~600 before overflow,
    # but the excitation needs to stay small enough not to saturate any
    # downstream diagnostic scale well before that). Try a few short
    # lengths and report cond/corr - open-loop x and u are independent BY
    # CONSTRUCTION (u is pure APRBS, x depends only on past u), so this
    # should decorrelate much more thoroughly than any closed-loop k_stab
    # choice can.
    rng = np.random.RandomState(42)
    print("\nopen-loop, short-horizon attempt (x,u independent by construction):")
    for length in [8, 12, 16, 20]:
        a_signal = fast_vectorized_aprbs(batch_size=1, length=length, low=-1.0, high=1.0, hold_prob=0.8, rng=rng, Nu=1)[0, 0]
        x = 0.1  # small initial condition - open loop, so this matters a lot for staying bounded
        xs, us = [], []
        for k in range(length):
            u = a_signal[k]
            xs.append(x)
            us.append(u)
            x = A_TRUE * x + B_TRUE * u
        z = np.stack([xs, us], axis=1)
        cond = regressor_condition_number(z)
        corr = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) if np.std(z[:, 1]) > 0 else float("nan")
        print(f"  length={length}: max|x|={np.abs(z[:,0]).max():.3e}  cond={cond:.2f}  corr(x,u)={corr:.4f}")

    with (RESULTS_DIR / "round2_partA_task3_conditioning.txt").open("w") as f:
        f.write(f"corr_orig={corr_orig}\ncorr_decorrelated={corr_decorr}\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== A1/A2/A3: orthogonality null tests, arm_2, init vs final, all 8 seeds ===")
    df = task_a1_a2_a3()
    print("\n" + df[["seed", "init_cos_median", "final_cos_median",
                       "init_percentile_of_actual_in_null", "final_percentile_of_actual_in_null",
                       "init_projected_ratio_median", "final_projected_ratio_median"]].to_string(index=False))

    print("\n=== A4: Task 2 robustness (Spearman) ===")
    task_a4_spearman()

    print("\n=== A5: Task 3 further decorrelation attempt ===")
    task_a5_conditioning()


if __name__ == "__main__":
    main()
