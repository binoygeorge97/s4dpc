"""Part C, C8: gain sweep.

Sweeps true rho in {0.5, 0.9, 1.03, 1.5, 3}, B_true FIXED at 1 across
the whole sweep (isolating rho as the swept variable - plant2's own
B=0.01 is NOT reused here, since mixing B values would confound the
gain sweep with a second varying quantity). Measures the radius at
which relative Jacobian error first crosses a fixed threshold (10%),
moving outward from the origin - the "failure radius." Prediction:
failure radius shrinks monotonically as rho rises, with STABLE systems
(rho<1) on the SAME curve rather than a separate case.

Uses open-loop-with-resets (A9's scheme) at a per-rho reset_every
chosen so rho^reset_every stays in a comparable, moderate range across
the sweep (not held exactly fixed, since a UNIFORM reset_every would
give wildly different growth factors at rho=0.5 vs rho=3).

Run: python -m layernorm_study.experiments.round2_partC_c8_gain_sweep
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

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.conditioning import open_loop_with_resets
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from s4dpc.diagnostics import zero_states

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N, L_MAX = 8, 16, 100
EPOCHS, LEARNING_RATE = 60000, 1e-3
B_FIXED = 1.0
RHO_VALUES = [0.5, 0.9, 1.03, 1.5, 3.0]
RESET_EVERY = {0.5: 20, 0.9: 20, 1.03: 20, 1.5: 8, 3.0: 3}  # keeps rho^reset_every in a comparable moderate range
THRESHOLD = 0.10
C_VALUES = np.logspace(-6, 3, 60)  # positive only; failure radius is symmetric enough for this SISO plant


def failure_radius(model, rho: float) -> float:
    states = zero_states(model, dtype=jnp.complex128)
    u = jnp.zeros((1,))
    for c in C_VALUES:
        x = jnp.array([float(c)])
        jx, _ = _jit_jxju(model, x, u, states)
        rel_err = abs(float(jx) - rho) / abs(rho)
        if rel_err > THRESHOLD:
            return float(c)
    return float(C_VALUES[-1])  # never crossed within the tested range


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rho in RHO_VALUES:
        reset_every = RESET_EVERY[rho]
        inputs, targets = open_loop_with_resets(rho, B_FIXED, L_MAX, reset_every, seed=42, aprbs_low=-10, aprbs_high=10, x0_range=2.0)
        inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
        print(f"=== rho={rho} (reset_every={reset_every}, max|x|={np.abs(inputs[:,0]).max():.2f}) ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            r_fail = failure_radius(model, rho)
            row = {"rho": rho, "seed": seed, "train_mse": mse, "failure_radius": r_fail}
            rows.append(row)
            print(f"  seed={seed}: train_mse={mse:.3e} failure_radius={r_fail:.4e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C8_gain_sweep_results.csv", index=False)

    print("\n=== summary (median failure radius per rho) ===")
    summary = df.groupby("rho")["failure_radius"].median()
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "round2_C8_gain_sweep_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for rho in RHO_VALUES:
        sub = df[df["rho"] == rho]
        ax.scatter([rho] * len(sub), sub["failure_radius"], alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("true rho")
    ax.set_ylabel("failure radius (log)")
    ax.set_title("C8: failure radius vs gain")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C8_gain_sweep.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
