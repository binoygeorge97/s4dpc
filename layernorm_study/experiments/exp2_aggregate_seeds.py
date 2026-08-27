"""Aggregates results/exp2_ladder.csv (multi-seed) into per-arm summary
statistics and a compact multi-panel figure, to answer the standing
caveat in NOTES.md's "Results, round 1" section: is round 1's ranking
(flat < mild < kinked < severely broken) and each arm's approximate
severity a property of the ARCHITECTURE, or a fluke of seed=0?

Run: python -m layernorm_study.experiments.exp2_aggregate_seeds
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layernorm_study.src.arms import ARMS

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"

METRICS = ["teacher_mse", "jx_err_mean", "jx_err_max", "jx_origin_spike_ratio", "equilibrium_drift"]


def main() -> None:
    csv_path = RESULTS_DIR / "exp2_ladder.csv"
    df = pd.read_csv(csv_path)
    arm_order = [a for a in ARMS.keys() if a in df["arm"].unique()]
    n_seeds = df.groupby("arm")["seed"].nunique()

    print(f"loaded {csv_path}: {len(df)} rows, arms={list(arm_order)}, seeds/arm={dict(n_seeds)}")
    print()

    summary_rows = []
    for arm in arm_order:
        sub = df[df["arm"] == arm]
        row = {"arm": arm, "n_seeds": len(sub)}
        for metric in METRICS:
            vals = sub[metric].values
            row[f"{metric}_median"] = float(np.median(vals))
            row[f"{metric}_min"] = float(np.min(vals))
            row[f"{metric}_max"] = float(np.max(vals))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "exp2_ladder_seed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary[["arm", "n_seeds", "jx_err_mean_median", "jx_err_mean_min", "jx_err_mean_max",
                    "jx_origin_spike_ratio_median", "teacher_mse_median"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data_jx = [df[df["arm"] == a]["jx_err_mean"].values for a in arm_order]
    axes[0].boxplot(data_jx, tick_labels=arm_order, showfliers=True)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("jx_err_mean (relative, log)")
    axes[0].set_title(f"Jx error by arm, across {n_seeds.iloc[0]} seeds each")
    axes[0].tick_params(axis="x", rotation=45)

    data_mse = [df[df["arm"] == a]["teacher_mse"].values for a in arm_order]
    axes[1].boxplot(data_mse, tick_labels=arm_order, showfliers=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("teacher_mse (log)")
    axes[1].set_title("Teacher-forced MSE by arm, across seeds")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp2_seed_variance_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
