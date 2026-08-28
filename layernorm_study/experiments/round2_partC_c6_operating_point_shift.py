"""Part C, C6: operating-point shift.

Excites postnorm (arm_6) around x_ref=50 instead of 0. Prediction:
postnorm RELOCATES its correct region to that radius by adapting the
bias (v(0) shifts so the near-field/plateau region of the theory's
r*~||Pb||/sigma_max(PM) crossover sits near x_ref, not near the
physical origin) - establishing postnorm as a local-linearization
machine centered wherever it was trained, not a linear-system learner
that would be correct everywhere regardless of training distribution.

Run: python -m layernorm_study.experiments.round2_partC_c6_operating_point_shift
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
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data_at_radius
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from s4dpc.diagnostics import zero_states

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
X_REF = 50.0
SWEEP_CENTERS = np.concatenate([[0.0], np.linspace(-20, 120, 29)])  # includes the ORIGINAL origin and the new x_ref region


def jx_at_centers(model, centers: np.ndarray) -> list[dict]:
    states = zero_states(model, dtype=jnp.complex128)
    u = jnp.zeros((1,))
    rows = []
    for c in centers:
        x = jnp.array([float(c)])
        jx, ju = _jit_jxju(model, x, u, states)
        rows.append({"x_center": float(c), "Jx": float(jx)})
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data_at_radius(seed=42, length=L_MAX, x0_center=X_REF, x0_spread=8.0,
                                               aprbs_low=-10.0, aprbs_high=10.0, reset_every=20)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        sweep = jx_at_centers(model, SWEEP_CENTERS)
        for r in sweep:
            r["seed"] = seed
            r["train_mse"] = mse
        rows.extend(sweep)
        jx_at_0 = [r["Jx"] for r in sweep if abs(r["x_center"]) < 1e-9][0]
        jx_at_ref = min(sweep, key=lambda r: abs(r["x_center"] - X_REF))["Jx"]
        print(f"  seed={seed}: train_mse={mse:.3e} Jx_at_origin={jx_at_0:.4f} Jx_at_x_ref={jx_at_ref:.4f} "
              f"(true={A_TRUE})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C6_operating_point_shift.csv", index=False)

    print("\n=== summary: median Jx by sweep center, across 8 seeds ===")
    summary = df.groupby("x_center")["Jx"].median()
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in SEEDS:
        sub = df[df["seed"] == seed].sort_values("x_center")
        ax.plot(sub["x_center"], sub["Jx"], alpha=0.4, color="black")
    ax.axhline(A_TRUE, color="tab:red", linestyle="--", label=f"true Jx={A_TRUE}")
    ax.axvline(0.0, color="tab:blue", linestyle=":", label="original origin")
    ax.axvline(X_REF, color="tab:green", linestyle=":", label=f"training x_ref={X_REF}")
    ax.set_xlabel("x (sweep center)")
    ax.set_ylabel("Jx")
    ax.set_title("C6: does postnorm's good region relocate to x_ref?")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C6_operating_point_shift.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
