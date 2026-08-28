"""Part B, Task 4: postnorm output-ceiling check, on this study's
original plant (x_next=3x+u) - the existing arm_6 (postnorm) and arm_7
(prenorm+postnorm) checkpoints, 8 seeds each, no retraining.

Computes Y_max from trained weights (output_ceiling.compute_y_max),
sweeps the model's raw output magnitude ||F(c*z0)|| across a wide range
of c, and checks whether it plateaus at Y_max while the TRUE plant's
output grows without bound (|A*x0+B*u0|*c, linear in c). If predicted
and measured ceilings agree, "postnorm is bad" upgrades to "postnorm
cannot represent a homogeneous map with gain > 1."

Run: python -m layernorm_study.experiments.round2_partB_task4_output_ceiling
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
from layernorm_study.experiments.round1_5_branch_suppression import load_trained
from layernorm_study.src.output_ceiling import compute_y_max
from layernorm_study.src.scalar_system import A_TRUE, B_TRUE

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
C_VALUES = np.logspace(-3, 4, 60)  # positive only - ||F|| is what's bounded, direction doesn't matter for the ceiling


def raw_output_norm_sweep(model, direction: np.ndarray, c_values: np.ndarray) -> list[dict]:
    states = zero_states(model, dtype=jnp.complex128)
    rows = []
    for c in c_values:
        z = c * direction
        x_next, _ = step(model, jnp.asarray(z[:1]), jnp.asarray(z[1:]), states)
        rows.append({"c": float(c), "output_norm": float(jnp.linalg.norm(x_next))})
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    direction = np.array([1.0, 1.0])
    true_slope = abs(A_TRUE * direction[0] + B_TRUE * direction[1])

    all_rows = []
    for arm_name in ["arm_6", "arm_7"]:
        print(f"=== {arm_name} ===")
        for seed in SEEDS:
            model = load_trained(arm_name, seed, decode=True)
            y_max = compute_y_max(model)
            sweep = raw_output_norm_sweep(model, direction, C_VALUES)
            plateau_measured = float(np.median([r["output_norm"] for r in sweep[-10:]]))  # last 10 (largest c) points
            all_rows.append({"arm": arm_name, "seed": seed, "y_max_predicted": y_max,
                              "plateau_measured": plateau_measured,
                              "ratio_measured_over_predicted": plateau_measured / y_max if y_max else float("nan")})
            print(f"  seed={seed}: Y_max_predicted={y_max:.4f}  plateau_measured={plateau_measured:.4f}  "
                  f"ratio={plateau_measured/y_max:.3f}", flush=True)

            if seed == 0:
                fig, ax = plt.subplots(figsize=(7, 5.5))
                cs = [r["c"] for r in sweep]
                outs = [r["output_norm"] for r in sweep]
                ax.plot(cs, outs, color="black", label="postnorm |F(c*z0)|")
                ax.plot(cs, [true_slope * c for c in cs], color="tab:red", linestyle="--", label="true plant (unbounded, linear)")
                ax.axhline(y_max, color="tab:blue", linestyle=":", label=f"predicted Y_max={y_max:.3f}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("c (||z||, log)")
                ax.set_ylabel("||F(z)|| (log)")
                ax.set_title(f"{arm_name} seed0: output ceiling")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig_path = FIGURES_DIR / f"round2_partB_task4_{arm_name}_ceiling.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
                print(f"    wrote {fig_path}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "round2_partB_task4_results.csv", index=False)
    print("\n=== summary ===")
    print(df.groupby("arm")["ratio_measured_over_predicted"].agg(["median", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
