"""Part C, C5: two-shell test - the cleanest impossibility demonstration.

Trains on data at TWO radii, ||z||~1 and ||z||~50. The true plant has
the SAME Jacobian (A_true, B_true) at both; postnorm's bounded output
(C2) means it cannot represent that Jacobian faithfully at both scales
at once - it must compromise. The theoretical worst-case error floor
along a direction w, splitting the difference between the two targets,
is (r2-r1)*|L.w|/2 where L=(A_true,B_true) (this plant is scalar, so
L.w = A_true*w_x + B_true*w_u for a UNIT direction w=(w_x,w_u)).

Prenorm (arm_5) is the control - it should handle both shells fine
(no boundedness constraint forces a compromise). If postnorm's actual
error lands ON the predicted floor, the network is provably at its
best and its best is provably bad.

Run: python -m layernorm_study.experiments.round2_partC_c5_two_shell
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

from s4dpc.data import fast_vectorized_aprbs
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX
from layernorm_study.src.scalar_diagnostics import teacher_forced_mse

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
R1, R2 = 1.0, 50.0
W_DIRECTION = np.array([1.0, 0.0])  # unit direction (w_x, w_u): pure x-excitation, u carries only dither


def generate_two_shell_data(seed: int, length: int = L_MAX, segment: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Alternates blocks of `segment` steps between x drawn near R1 and
    near R2 (open-loop, u pure small-amplitude APRBS dither on top of
    the shell's own excitation direction - x itself is reset to the
    target shell every `segment` steps, u independent, matching A9's
    open-loop-with-resets scheme so this inherits its good
    conditioning rather than introducing a new artifact)."""
    rng = np.random.RandomState(seed)
    dither = fast_vectorized_aprbs(batch_size=1, length=length, low=-0.5, high=0.5, hold_prob=0.8, rng=rng, Nu=1)[0, 0]
    inputs = np.zeros((length, 2))
    targets = np.zeros((length, 1))
    x = 0.0
    for t in range(length):
        if t % segment == 0:
            shell = R1 if (t // segment) % 2 == 0 else R2
            sign = rng.choice([-1.0, 1.0])
            x = sign * shell * W_DIRECTION[0]
        u = dither[t] + (rng.uniform(-1, 1) * shell * W_DIRECTION[1] if t % segment == 0 else 0.0)
        inputs[t, 0] = x
        inputs[t, 1] = u
        x_next = A_TRUE * x + B_TRUE * u
        targets[t, 0] = x_next
        x = x_next
    return inputs, targets


def theoretical_floor() -> float:
    Lw = A_TRUE * W_DIRECTION[0] + B_TRUE * W_DIRECTION[1]
    return (R2 - R1) * abs(Lw) / 2


def shell_wise_error(model, inputs_j, targets_j, shell_mask: np.ndarray) -> float:
    """RMSE restricted to timesteps belonging to one shell."""
    from s4dpc.diagnostics import step, zero_states
    d_x = targets_j.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    sq_errs = []
    for t in range(inputs_j.shape[0]):
        x_next, states = step(model, inputs_j[t, :d_x], inputs_j[t, d_x:], states)
        if shell_mask[t]:
            sq_errs.append(float(jnp.sum((x_next - targets_j[t]) ** 2)))
    return float(np.sqrt(np.mean(sq_errs))) if sq_errs else float("nan")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    floor = theoretical_floor()
    print(f"theoretical worst-case floor (splitting the difference): {floor:.4f}")

    inputs, targets = generate_two_shell_data(seed=42)
    shell_mask_r2 = np.abs(inputs[:, 0]) > (R1 + R2) / 2  # crude split by |x|
    print(f"data: {shell_mask_r2.sum()} points near r2, {(~shell_mask_r2).sum()} points near r1")

    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for arm_name in ["arm_5", "arm_6"]:
        print(f"=== {arm_name} ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                ARMS[arm_name], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(ARMS[arm_name], param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            err_r1 = shell_wise_error(model, inputs_j, targets_j, ~shell_mask_r2)
            err_r2 = shell_wise_error(model, inputs_j, targets_j, shell_mask_r2)
            row = {"arm": arm_name, "seed": seed, "train_mse": mse, "rmse_r1": err_r1, "rmse_r2": err_r2,
                   "theoretical_floor": floor}
            rows.append(row)
            print(f"  seed={seed}: train_mse={mse:.3e} rmse_r1={err_r1:.4f} rmse_r2={err_r2:.4f} "
                  f"(floor={floor:.4f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round2_C5_two_shell_results.csv", index=False)

    print("\n=== summary ===")
    summary = df.groupby("arm")[["rmse_r1", "rmse_r2"]].median()
    print(summary.to_string())
    print(f"theoretical floor: {floor:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arm_name, marker in [("arm_5", "o"), ("arm_6", "s")]:
        sub = df[df["arm"] == arm_name]
        ax.scatter(sub["rmse_r1"], sub["rmse_r2"], label=arm_name, marker=marker, alpha=0.7, s=60)
    ax.axhline(floor, color="red", linestyle="--", label=f"theoretical floor ({floor:.2f})")
    ax.axvline(floor, color="red", linestyle="--")
    ax.set_xlabel("RMSE at shell r1=1")
    ax.set_ylabel("RMSE at shell r2=50")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend()
    ax.set_title("C5: two-shell test, postnorm vs prenorm")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round2_C5_two_shell.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
