"""Part B, Task 5: epsilon sweep on arm_5 - the direct mechanism probe.

LayerNorm's Jacobian has a 1/sigma prefactor, sigma=sqrt(||Pv||^2/H+eps)
- near the origin (Pv small), sigma saturates at sqrt(eps) rather than
diverging, capping the Jacobian's magnitude and setting the WIDTH of
the region where that cap applies. If the kink is really this
regularized-singularity mechanism, both its characteristic width
(where sigma stops being ~const and starts tracking ||z||) and,
plausibly, its peak amplitude (bounded by ~1/sqrt(eps)) should scale
with sqrt(eps) as eps is swept over orders of magnitude, holding
everything else (data, seed, d_model, optimizer, epochs) fixed.

Run: python -m layernorm_study.experiments.round2_partB_task5_epsilon_sweep
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
from scipy import stats

from layernorm_study.src.arms import ARMS, ArmSpec, load_arm_model, train_arm
from layernorm_study.src.scalar_diagnostics import jx_vs_c_sweep
from layernorm_study.src.scalar_system import generate_scalar_trajectory
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX, EPOCHS, LEARNING_RATE

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
EPS_VALUES = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
C_VALUES = np.concatenate([-np.logspace(3, -8, 60), np.logspace(-8, 3, 60)])


def arm5_with_eps(eps: float) -> ArmSpec:
    base = ARMS["arm_5"]
    kwargs = dict(base.block_kwargs)
    kwargs["layer_norm_eps"] = eps
    return dataclasses.replace(base, block_kwargs=kwargs)


def kink_amplitude_and_width(sweep: list[dict]) -> tuple[float, float]:
    """amplitude = |Jx_peak(near origin)| - |Jx_far|; width = smallest
    |c| (measured inward from the far side) at which |Jx(c)| first
    deviates from Jx_far by more than half the peak deviation."""
    n = len(sweep)
    jx_abs = np.array([abs(r["Jx"]) for r in sweep])
    c_abs = np.array([abs(r["c"]) for r in sweep])
    jx_far = float(np.median(np.concatenate([jx_abs[:5], jx_abs[-5:]])))
    peak_idx = int(np.argmin(c_abs))  # closest to the origin
    jx_peak = float(jx_abs[peak_idx])
    amplitude = jx_peak - jx_far
    half = jx_far + amplitude / 2

    pos_mask = np.array([r["c"] > 0 for r in sweep])
    pos_c = c_abs[pos_mask]
    pos_jx = jx_abs[pos_mask]
    order = np.argsort(pos_c)[::-1]  # descending c, from far to near
    pos_c, pos_jx = pos_c[order], pos_jx[order]
    width = pos_c[-1]
    for i in range(len(pos_c)):
        if amplitude > 0 and pos_jx[i] >= half:
            width = pos_c[i]
            break
        elif amplitude < 0 and pos_jx[i] <= half:
            width = pos_c[i]
            break
    return amplitude, width


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_scalar_trajectory(length=L_MAX, seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for eps in EPS_VALUES:
        arm = arm5_with_eps(eps)
        print(f"=== eps={eps:.0e} ===")
        for seed in SEEDS:
            param_state, train_mse = train_arm(
                arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            sweep = jx_vs_c_sweep(model, np.array([1.0]), C_VALUES)
            amplitude, width = kink_amplitude_and_width(sweep)
            row = {"eps": eps, "seed": seed, "teacher_mse_train_final": train_mse,
                   "kink_amplitude": amplitude, "kink_width": width}
            rows.append(row)
            print(f"  seed={seed}: train_mse={train_mse:.3e} kink_amplitude={amplitude:.4f} kink_width={width:.3e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_partB_task5_results.csv", index=False)

    summary = df.groupby("eps").agg(
        median_amplitude=("kink_amplitude", "median"), median_width=("kink_width", "median"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "round2_partB_task5_summary.csv", index=False)
    print("\n=== summary (median across 8 seeds) ===")
    print(summary.to_string(index=False))

    log_eps = np.log10(summary["eps"].values)
    log_amp = np.log10(np.abs(summary["median_amplitude"].values))
    log_width = np.log10(summary["median_width"].values)
    slope_amp, intercept_amp, r_amp, p_amp, se_amp = stats.linregress(log_eps, log_amp)
    slope_width, intercept_width, r_width, p_width, se_width = stats.linregress(log_eps, log_width)
    print(f"\nlog-log fit: kink_amplitude ~ eps^{slope_amp:.3f} (predicted 0.5 if amplitude~sqrt(eps), "
          f"-0.5 if amplitude~1/sqrt(eps)) r^2={r_amp**2:.3f}")
    print(f"log-log fit: kink_width ~ eps^{slope_width:.3f} (predicted 0.5 if width~sqrt(eps)) r^2={r_width**2:.3f}")

    with (RESULTS_DIR / "round2_partB_task5_fits.txt").open("w") as f:
        f.write(f"amplitude_slope={slope_amp}, r2={r_amp**2}, se={se_amp}\n")
        f.write(f"width_slope={slope_width}, r2={r_width**2}, se={se_width}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for eps in EPS_VALUES:
        sub = df[df["eps"] == eps]
        axes[0].scatter([eps] * len(sub), np.abs(sub["kink_amplitude"]), alpha=0.6)
        axes[1].scatter([eps] * len(sub), sub["kink_width"], alpha=0.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("eps"); axes[0].set_ylabel("|kink amplitude|")
    axes[0].set_title(f"amplitude ~ eps^{slope_amp:.2f}")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("eps"); axes[1].set_ylabel("kink width (c)")
    axes[1].set_title(f"width ~ eps^{slope_width:.2f}")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_partB_task5_epsilon_sweep.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
