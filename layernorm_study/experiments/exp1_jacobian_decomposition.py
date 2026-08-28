"""EXPERIMENT 1: skip/branch Jacobian decomposition on EXISTING s4dpc
checkpoints (docs/nu_gap_export/ckpt/ by default) - no new training.

For each checkpoint, decomposes F(z) = W_dec@W_enc@z + W_dec@branch(z) and
reports:
  - how much of the true [A_d | B_d] the constant skip term alone explains
  - the branch/constant Jacobian norm ratio along the real training trajectory
  - an origin sweep of dF/dx, decomposed into constant + branch
  - a branch-zeroing ablation: force the branch to 0 at inference, re-measure
    teacher-forced one-step MSE

Run: python -m layernorm_study.experiments.exp1_jacobian_decomposition
     (from the repo root, so `s4dpc` and `layernorm_study` are both importable)
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op (CLAUDE.md convention)

import csv
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s4dpc.identify import DT, case_data
from s4dpc.systems import get_discrete_matrices

from layernorm_study.src.checkpoints import available_checkpoints, load_model
from layernorm_study.src.jacobian_decomposition import (
    constant_term,
    origin_sweep_decomposed,
    teacher_forced_mse,
    trajectory_contamination,
)

CKPT_DIR = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"

VARIANTS = ["M3", "M4", "M5", "M6"]  # M5/M6 are the prenorm+LN arms the task asks about;
# M3/M4 (no LN) are included as a free contrast - same checkpoints, same loader/analysis.
# M0_S4 deliberately excluded: it's hand-constructed via a different code path
# (tools/controller_m0_s4.py), not s4dpc.blocks.VARIANTS/identify.py's ladder, so it
# doesn't fit this module's generic (BlockConfig, VARIANTS[variant]) loader.

T_VALUES = np.concatenate(
    [-np.logspace(2, -6, 30), np.logspace(-6, 2, 30)]
)  # symmetric log-spaced sweep through the origin, both sides


def analyze_one(variant: str, case: int, seed: int) -> dict:
    model, sidecar = load_model(CKPT_DIR, variant, case, seed)
    A_d, B_d = get_discrete_matrices(DT, case)
    AB_true = np.concatenate([A_d, B_d], axis=1)  # (d_x, d_x+d_u), matches [x, u] input layout
    true_norm = float(np.linalg.norm(AB_true))

    C = constant_term(model)
    skip_alone_rel_err = float(np.linalg.norm(C - AB_true) / true_norm)

    inputs, targets = case_data(case, sidecar["config"]["l_max"], -10.0, 10.0)
    contamination_rows = trajectory_contamination(model, inputs, targets)
    ratios = np.array([r["contamination_ratio"] for r in contamination_rows])

    mse_baseline = teacher_forced_mse(model, inputs, targets, branch_zeroed=False)
    mse_branch_zeroed = teacher_forced_mse(model, inputs, targets, branch_zeroed=True)

    d_x = targets.shape[-1]
    direction = np.zeros(d_x)
    direction[0] = 1.0
    sweep_rows = origin_sweep_decomposed(model, direction, T_VALUES, u=jnp.zeros(inputs.shape[-1] - d_x))
    max_decomposition_residual = max(r["decomposition_residual"] for r in sweep_rows)

    return {
        "variant": variant,
        "case": case,
        "seed": seed,
        "recorded_teacher_mse": sidecar["teacher_mse"],
        "recomputed_teacher_mse": mse_baseline,
        "mse_sanity_rel_err": abs(mse_baseline - sidecar["teacher_mse"]) / max(sidecar["teacher_mse"], 1e-300),
        "skip_alone_rel_err_vs_AB_true": skip_alone_rel_err,
        "contamination_ratio_median": float(np.median(ratios)),
        "contamination_ratio_max": float(np.max(ratios)),
        "mse_branch_zeroed": mse_branch_zeroed,
        "branch_zeroing_mse_ratio": mse_branch_zeroed / mse_baseline if mse_baseline > 0 else float("inf"),
        "max_decomposition_residual": max_decomposition_residual,
        "sweep_rows": sweep_rows,  # kept for the representative plot only, stripped before CSV
    }


def plot_origin_sweep(rows: list[dict], variant: str, case: int, seed: int) -> pathlib.Path:
    t = [r["t"] for r in rows]
    const = [r["J_x_constant_norm"] for r in rows]
    branch = [r["J_x_branch_norm"] for r in rows]
    full = [r["J_x_full_norm"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, full, label="||J_x full||", color="black", linewidth=1.5)
    ax.plot(t, const, label="||J_x constant (skip)||", linestyle="--", color="tab:blue")
    ax.plot(t, branch, label="||J_x branch||", linestyle=":", color="tab:red")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_xlabel("x (state, symlog)")
    ax.set_ylabel("Jacobian Frobenius norm (log)")
    ax.set_title(f"{variant} case{case} seed{seed}: origin sweep, skip/branch decomposition")
    ax.legend()
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"exp1_origin_sweep_{variant}_case{case}_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    plotted_variants = set()

    # Written incrementally (one row appended per checkpoint, not batched at
    # the end) - a crash partway through (e.g. an unloadable variant) must
    # not discard already-computed checkpoints, which a single end-of-run
    # write would do. Fieldnames are fixed up front so every row's key
    # order matches, regardless of which variant/case produced it.
    csv_path = RESULTS_DIR / "exp1_jacobian_decomposition.csv"
    fieldnames = [
        "variant", "case", "seed", "recorded_teacher_mse", "recomputed_teacher_mse",
        "mse_sanity_rel_err", "skip_alone_rel_err_vs_AB_true", "contamination_ratio_median",
        "contamination_ratio_max", "mse_branch_zeroed", "branch_zeroing_mse_ratio",
        "max_decomposition_residual",
    ]
    csv_file = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for variant in VARIANTS:
        pairs = available_checkpoints(CKPT_DIR, variant)
        print(f"[{variant}] {len(pairs)} checkpoints found", flush=True)
        for case, seed in sorted(pairs):
            result = analyze_one(variant, case, seed)
            sweep_rows = result.pop("sweep_rows")
            all_rows.append(result)
            writer.writerow(result)
            csv_file.flush()
            print(
                f"    case{case} seed{seed}: contam_med={result['contamination_ratio_median']:.3g} "
                f"skip_err={result['skip_alone_rel_err_vs_AB_true']:.3g} "
                f"branch_zero_ratio={result['branch_zeroing_mse_ratio']:.3g}",
                flush=True,
            )

            if variant not in plotted_variants and case == 1 and seed == 0:
                fig_path = plot_origin_sweep(sweep_rows, variant, case, seed)
                print(f"    wrote {fig_path}")
                plotted_variants.add(variant)

    csv_file.close()
    print(f"\nwrote {csv_path} ({len(all_rows)} rows)")

    print("\n=== per-variant summary (median across case x seed) ===")
    header = (
        f"{'variant':8s} {'n':>3s} {'skip_alone_err':>15s} {'contam_ratio_med':>17s} "
        f"{'contam_ratio_max':>17s} {'mse_sanity_err':>15s} {'branch_zero_mse_ratio':>22s}"
    )
    print(header)
    for variant in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == variant]
        if not rows:
            continue
        n = len(rows)
        skip_err = np.median([r["skip_alone_rel_err_vs_AB_true"] for r in rows])
        contam_med = np.median([r["contamination_ratio_median"] for r in rows])
        contam_max = np.median([r["contamination_ratio_max"] for r in rows])
        mse_sanity = np.median([r["mse_sanity_rel_err"] for r in rows])
        branch_zero_ratio = np.median([r["branch_zeroing_mse_ratio"] for r in rows])
        print(
            f"{variant:8s} {n:3d} {skip_err:15.4e} {contam_med:17.4e} "
            f"{contam_max:17.4e} {mse_sanity:15.4e} {branch_zero_ratio:22.4e}"
        )

    max_residual = max(r["max_decomposition_residual"] for r in all_rows)
    print(f"\nmax decomposition residual (||J - (C + branch)||, should be ~0): {max_residual:.3e}")


if __name__ == "__main__":
    main()
