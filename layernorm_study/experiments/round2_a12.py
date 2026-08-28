"""Round 2, A12: tighten the closed-form cond formula.

The stated formula cond ~ (1+k^2)*var(x)/var(a) was off by a "roughly
constant 7-14x" factor in round 2's A8/A9. Re-derived from the exact
2x2 eigenvalue product/sum of E[zz^T] (not assumed): in the small-
var(a) asymptotic regime, lambda_max ~ Mxx*(1+k^2) and
lambda_min ~ Maa/(1+k^2) (an extra factor of (1+k^2) was missing from
the stated formula) - so cond ~ (1+k^2)^2 * var(x)/var(a).

Checks whether the residual (actual/predicted) correlates with k
(rather than calling it constant), and separately isolates how much of
that residual is finite-sample noise (varying seed at fixed k) versus
genuine k-dependence (varying k at fixed seed).

Run: python -m layernorm_study.experiments.round2_a12
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from layernorm_study.src.scalar_system import generate_scalar_trajectory

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def cond_and_predictions(k: float, seed: int = 42) -> dict:
    inputs, _ = generate_scalar_trajectory(length=100, seed=seed, k_stab=k)
    x, u = inputs[:, 0], inputs[:, 1]
    a = u - k * x
    Mxx, Maa = float(np.mean(x ** 2)), float(np.mean(a ** 2))
    cov = inputs.T @ inputs / len(inputs)
    actual_cond = float(np.linalg.cond(cov))
    orig_pred = (1 + k ** 2) * Mxx / Maa
    refined_pred = (1 + k ** 2) ** 2 * Mxx / Maa
    return {
        "k": k, "seed": seed, "Mxx": Mxx, "Maa": Maa, "actual_cond": actual_cond,
        "orig_pred": orig_pred, "orig_residual": actual_cond / orig_pred,
        "refined_pred": refined_pred, "refined_residual": actual_cond / refined_pred,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== A12: k-scan at fixed seed=42, original vs refined formula ===")
    k_scan_rows = [cond_and_predictions(k, seed=42) for k in np.arange(-3.99, -2.01, 0.05)]
    k_scan_df = pd.DataFrame(k_scan_rows)
    k_scan_df.to_csv(RESULTS_DIR / "round2_A12_k_scan.csv", index=False)

    orig_res = k_scan_df["orig_residual"].values
    refined_res = k_scan_df["refined_residual"].values
    print(f"original formula residual: range=[{orig_res.min():.3f},{orig_res.max():.3f}] mean={orig_res.mean():.3f}")
    print(f"refined formula residual:  range=[{refined_res.min():.3f},{refined_res.max():.3f}] mean={refined_res.mean():.3f}")

    r, p = stats.pearsonr(k_scan_df["k"].values, refined_res)
    print(f"\nrefined residual vs k: pearson r={r:.4f} p={p:.4g}")

    print("\n=== seed scan at fixed k=-2.7 (isolates finite-sample noise from k-dependence) ===")
    seed_scan_rows = [cond_and_predictions(-2.7, seed=s) for s in range(10)]
    seed_scan_df = pd.DataFrame(seed_scan_rows)
    seed_scan_df.to_csv(RESULTS_DIR / "round2_A12_seed_scan.csv", index=False)
    seed_res = seed_scan_df["refined_residual"].values
    print(f"refined residual across seeds at fixed k: range=[{seed_res.min():.3f},{seed_res.max():.3f}] "
          f"mean={seed_res.mean():.3f} std={seed_res.std():.3f}  (vs k-scan std={refined_res.std():.3f})")

    with (RESULTS_DIR / "round2_A12_summary.txt").open("w") as f:
        f.write(f"orig_formula: cond ~ (1+k^2)*var(x)/var(a), residual mean={orig_res.mean():.3f} range=[{orig_res.min():.3f},{orig_res.max():.3f}]\n")
        f.write(f"refined_formula: cond ~ (1+k^2)^2*var(x)/var(a), residual mean={refined_res.mean():.3f} range=[{refined_res.min():.3f},{refined_res.max():.3f}]\n")
        f.write(f"refined_residual_vs_k: pearson_r={r:.4f}, p={p:.4g}\n")
        f.write(f"seed_noise_at_fixed_k: std={seed_res.std():.3f}, k_scan_std={refined_res.std():.3f}\n")


if __name__ == "__main__":
    main()
