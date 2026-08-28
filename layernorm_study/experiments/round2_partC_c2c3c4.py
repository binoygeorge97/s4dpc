"""Part C, C2/C3/C4: output ceiling, decay slope, rank deficiency.

Trains its OWN arm_6/plant2 checkpoints (8 seeds) rather than reusing
C1's - C1's script (already launched before this one) does not persist
checkpoints to disk, only CSV summary statistics, so there is nothing
to load from. A real, acknowledged inefficiency (8 extra trainings),
not a design choice.

C2: Y_max from trained weights vs. the observed plateau in a FREE-RUN
rollout (the model's own predictions fed back recursively, no re-
driving from real data) - overlays the true plant's unbounded 1.03^k
growth on the same axes.

C3: log-log slope of ||J|| vs ||z|| in the far field - predicted -1.

C4: Euler identity / rank deficiency of the postnorm block's OWN
LayerNorm Jacobian (H x H, dLN/dv - not the overall 1x2 input-output
Jacobian, which has only one singular value and no rank-deficiency
structure to speak of). Checks ||J_LN(v) v|| / (||J_LN(v)|| ||v||) -> 0
far from the origin, that the null direction aligns with vhat, and that
sigma_min collapses before sigma_max as ||v|| grows.

Run: python -m layernorm_study.experiments.round2_partC_c2c3c4
"""
from __future__ import annotations

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from s4dpc.diagnostics import step, zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.output_ceiling import compute_y_max, which_norm_bounds_output
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data
from layernorm_study.src.postnorm_geometry import pre_final_norm_activation
from layernorm_study.src.scalar_diagnostics import _jit_jxju

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
FAR_FIELD_C = np.logspace(1, 3, 15)  # for the decay-slope fit specifically
C4_C_VALUES = np.logspace(-3, 3, 25)


@nnx.jit
def _jit_ln_jacobian(norm_module, v: jax.Array):
    return jax.jacfwd(norm_module)(v)


def free_run_rollout(model, x0: float, n_steps: int) -> np.ndarray:
    states = zero_states(model, dtype=jnp.complex128)
    x = jnp.array([x0])
    u = jnp.zeros((1,))
    xs = [x0]
    for _ in range(n_steps):
        x, states = step(model, x, u, states)
        xs.append(float(x[0]))
    return np.array(xs)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    c2_rows, c3_rows, c4_rows = [], [], []
    seed0_model, seed0_rollout, seed0_y_max = None, None, None
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
        print(f"seed={seed}: train_mse={mse:.3e}", flush=True)

        # C2
        y_max = compute_y_max(model)
        rollout = free_run_rollout(model, x0=10.0, n_steps=100)
        plateau = float(np.median(np.abs(rollout[-20:])))
        c2_rows.append({"seed": seed, "y_max_predicted": y_max, "plateau_measured": plateau,
                         "ratio": plateau / y_max if y_max else float("nan")})
        print(f"  C2: Y_max={y_max:.4f} plateau={plateau:.4f} ratio={plateau/y_max:.3f}", flush=True)
        if seed == 0:
            seed0_rollout, seed0_y_max = rollout, y_max

        # C3: far-field |Jx| vs |c|, log-log slope
        states0 = zero_states(model, dtype=jnp.complex128)
        u0 = jnp.zeros((1,))
        jx_far = []
        for c in FAR_FIELD_C:
            jx, _ = _jit_jxju(model, jnp.array([float(c)]), u0, states0)
            jx_far.append(abs(float(jx)))
        slope, intercept, r, p, se = stats.linregress(np.log10(FAR_FIELD_C), np.log10(jx_far))
        c3_rows.append({"seed": seed, "slope": slope, "r2": r ** 2, "se": se})
        print(f"  C3: far-field slope={slope:.3f} (predicted -1) r2={r**2:.3f}", flush=True)

        # C4: LayerNorm's own H x H Jacobian, Euler identity + rank structure
        norm = which_norm_bounds_output(model)
        states1 = zero_states(model, dtype=jnp.complex128)
        for c in [1.0, 1e3]:  # near vs far
            x = jnp.array([float(c)])
            v, _ = pre_final_norm_activation(model, x, u0, states1)
            J_ln = np.asarray(_jit_ln_jacobian(norm, v))
            Jv = J_ln @ np.asarray(v)
            euler_ratio = float(np.linalg.norm(Jv) / (np.linalg.norm(J_ln) * np.linalg.norm(v))) if np.linalg.norm(v) > 0 else float("nan")
            U, S, Vt = np.linalg.svd(J_ln)
            null_dir = Vt[-1]  # smallest singular value's right singular vector
            v_hat = np.asarray(v) / np.linalg.norm(v)
            null_alignment = float(abs(np.dot(null_dir, v_hat)))
            c4_rows.append({"seed": seed, "c": c, "euler_ratio": euler_ratio, "null_alignment": null_alignment,
                             "sigma_min": float(S[-1]), "sigma_max": float(S[0])})
            print(f"  C4 (c={c}): euler_ratio={euler_ratio:.4f} null_alignment={null_alignment:.4f} "
                  f"sigma_min={S[-1]:.4e} sigma_max={S[0]:.4e}", flush=True)

    c2_df, c3_df, c4_df = pd.DataFrame(c2_rows), pd.DataFrame(c3_rows), pd.DataFrame(c4_rows)
    c2_df.to_csv(RESULTS_DIR / "round2_C2_output_ceiling.csv", index=False)
    c3_df.to_csv(RESULTS_DIR / "round2_C3_decay_slope.csv", index=False)
    c4_df.to_csv(RESULTS_DIR / "round2_C4_rank_deficiency.csv", index=False)

    print("\n=== C2 summary ===")
    print(f"median ratio (plateau/Y_max): {c2_df['ratio'].median():.3f}")
    print("\n=== C3 summary ===")
    print(f"median slope: {c3_df['slope'].median():.3f} (predicted -1.0)")
    print(f"95% CI on median slope (via seed spread): [{c3_df['slope'].quantile(0.025):.3f}, {c3_df['slope'].quantile(0.975):.3f}]")
    print("\n=== C4 summary ===")
    print(c4_df.groupby("c")[["euler_ratio", "null_alignment", "sigma_min", "sigma_max"]].median().to_string())

    # C2 figure: one representative rollout (seed 0, reused from the main loop above - not retrained)
    rollout0, y_max0 = seed0_rollout, seed0_y_max
    ks = np.arange(len(rollout0))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(ks, np.abs(rollout0), color="black", label="postnorm free-run |x_k|")
    ax.plot(ks, 10.0 * A_TRUE ** ks, color="tab:red", linestyle="--", label=f"true plant {A_TRUE}^k * x0")
    ax.axhline(y_max0, color="tab:blue", linestyle=":", label=f"predicted Y_max={y_max0:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("step k")
    ax.set_ylabel("|x_k| (log)")
    ax.set_title("C2: free-run rollout vs predicted ceiling")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C2_output_ceiling.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
