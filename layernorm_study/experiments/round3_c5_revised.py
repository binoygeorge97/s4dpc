"""Round 3, C5-revised: two-shell test with BOTH shells verified in the far field.

Round 2's C5 was MIS-SPECIFIED, not falsified (user's own correction):
the floor (r2-r1)||Lw||/2 assumes the map is EXACTLY degree-0, so that
F(r1 w) ~ F(r2 w). But C1 established a real bias-determined operating
point with measured r* ~ 0.375, and round 2's C5 used r1=1 - which sits
at or INSIDE that r*, i.e. in the NEAR field where the model fits fine.
The two shells were never both in the degree-0 regime, so the floor
never applied and the test could not have landed either way.

This version places both shells above the measured r* and VERIFIES
that empirically before interpreting the fit, rather than trusting the
r* number alone:

  1. Pick candidate radii from C1's r* (r1 = 10*r*, r2 = 500*r*).
  2. Train postnorm (arm_6) and prenorm (arm_5, control) on
     alternating-shell data at those radii.
  3. MEASURE the local log-log Jacobian slope in a window around each
     shell radius on the resulting checkpoints. Plateau (near-field)
     gives slope ~0; far field gives clearly negative slope. Report
     both, so the test's own applicability is auditable from the
     numbers rather than asserted.
  4. Compare achieved per-shell RMSE against the predicted floor.

Also reports Y_max and the floor-optimal constant at each shell, since
if Y_max < 1.03*(r1+r2)/2 the model cannot even reach the optimal
constant and the error would EXCEED the floor for a ceiling reason
rather than a degree-0-compromise reason - a distinction the raw
"did it hit the floor" number alone would hide.

Run: python -m layernorm_study.experiments.round3_c5_revised
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
from scipy import stats

from s4dpc.data import fast_vectorized_aprbs
from s4dpc.diagnostics import step, zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.output_ceiling import compute_y_max
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX
from layernorm_study.src.scalar_diagnostics import _jit_jxju

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3

R_STAR_REF = 0.5  # from round 2's C1: median r*_measured=0.375, median r*_predicted=0.479
R1 = 10 * R_STAR_REF   # 5.0
R2 = 500 * R_STAR_REF  # 250.0
W_DIRECTION = np.array([1.0, 0.0])  # unit direction: pure x, u carries dither only
SEGMENT = 10


def generate_two_shell_data(seed: int, r1: float, r2: float, length: int = L_MAX, segment: int = SEGMENT):
    """Alternates blocks of `segment` steps between |x| ~ r1 and |x| ~ r2
    (open-loop, x reset to the target shell each block, u independent
    small APRBS dither) - same construction as round 2's C5, only the
    radii change."""
    rng = np.random.RandomState(seed)
    dither = fast_vectorized_aprbs(batch_size=1, length=length, low=-0.5, high=0.5, hold_prob=0.8, rng=rng, Nu=1)[0, 0]
    inputs = np.zeros((length, 2))
    targets = np.zeros((length, 1))
    x = 0.0
    for t in range(length):
        if t % segment == 0:
            shell = r1 if (t // segment) % 2 == 0 else r2
            x = rng.choice([-1.0, 1.0]) * shell
        u = dither[t]
        inputs[t, 0] = x
        inputs[t, 1] = u
        x_next = A_TRUE * x + B_TRUE * u
        targets[t, 0] = x_next
        x = x_next
    return inputs, targets


def local_jacobian_slope(model, radius: float, window: float = 1.3, n_pts: int = 9) -> float:
    """log-log slope of |Jx| vs |x| in a multiplicative window around
    `radius`. ~0 => plateau (near field); clearly negative => decay
    regime (far field). This is the empirical far-field VERIFICATION
    the round-2 version of this test lacked."""
    states = zero_states(model, dtype=jnp.complex128)
    u0 = jnp.zeros((1,))
    xs = radius * np.logspace(-np.log10(window), np.log10(window), n_pts)
    jx = []
    for x in xs:
        j, _ = _jit_jxju(model, jnp.array([float(x)]), u0, states)
        jx.append(abs(float(j)))
    jx = np.array(jx)
    good = jx > 0
    if good.sum() < 3:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log10(xs[good]), np.log10(jx[good]))
    return float(slope)


def shell_rmse(model, inputs_j, targets_j, mask: np.ndarray) -> float:
    d_x = targets_j.shape[-1]
    states = zero_states(model, dtype=jnp.complex128)
    sq = []
    for t in range(inputs_j.shape[0]):
        x_next, states = step(model, inputs_j[t, :d_x], inputs_j[t, d_x:], states)
        if mask[t]:
            sq.append(float(jnp.sum((x_next - targets_j[t]) ** 2)))
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Lw = A_TRUE * W_DIRECTION[0] + B_TRUE * W_DIRECTION[1]
    floor = (R2 - R1) * abs(Lw) / 2
    optimal_const = abs(Lw) * (R1 + R2) / 2

    print(f"r*_ref (from C1)={R_STAR_REF}  ->  r1={R1} (10x r*), r2={R2} (500x r*)")
    print(f"||Lw||={abs(Lw):.4f}  predicted floor=(r2-r1)||Lw||/2 = {floor:.4f}")
    print(f"floor-optimal constant output = ||Lw||(r1+r2)/2 = {optimal_const:.4f}")
    print("  (if Y_max < that constant, the model cannot even reach the optimal compromise -")
    print("   error would EXCEED the floor for a ceiling reason, not a degree-0 reason)")

    inputs, targets = generate_two_shell_data(seed=42, r1=R1, r2=R2)
    mask_r2 = np.abs(inputs[:, 0]) > (R1 + R2) / 2
    print(f"\ndata: {mask_r2.sum()} pts near r2, {(~mask_r2).sum()} near r1; "
          f"|target| range [{np.abs(targets).min():.2f}, {np.abs(targets).max():.2f}]")
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for arm_name in ["arm_6", "arm_5"]:
        print(f"\n=== {arm_name} ({'postnorm' if arm_name == 'arm_6' else 'prenorm control'}) ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                ARMS[arm_name], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(ARMS[arm_name], param_state, d_model=D_MODEL, N=N,
                                    l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            slope_r1 = local_jacobian_slope(model, R1)
            slope_r2 = local_jacobian_slope(model, R2)
            y_max = compute_y_max(model)  # None for prenorm (no bounding norm) - reported as NaN
            rows.append({
                "arm": arm_name, "seed": seed, "train_mse": mse,
                "r1": R1, "r2": R2, "floor": floor, "optimal_const": optimal_const,
                "y_max": y_max if y_max is not None else float("nan"),
                "local_slope_at_r1": slope_r1, "local_slope_at_r2": slope_r2,
                "rmse_r1": shell_rmse(model, inputs_j, targets_j, ~mask_r2),
                "rmse_r2": shell_rmse(model, inputs_j, targets_j, mask_r2),
            })
            r = rows[-1]
            print(f"  seed={seed}: train_mse={mse:.3e} | far-field check: slope@r1={slope_r1:+.2f} "
                  f"slope@r2={slope_r2:+.2f} | rmse_r1={r['rmse_r1']:.3f} rmse_r2={r['rmse_r2']:.3f} "
                  f"(floor={floor:.2f}) | Y_max={r['y_max']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round3_C5revised_results.csv", index=False)

    print("\n=== FAR-FIELD APPLICABILITY CHECK (does the test apply at all?) ===")
    post = df[df["arm"] == "arm_6"]
    print(f"postnorm median local slope at r1={R1}: {post['local_slope_at_r1'].median():+.3f}")
    print(f"postnorm median local slope at r2={R2}: {post['local_slope_at_r2'].median():+.3f}")
    print("  (~0 => still in the near-field plateau, test does NOT apply;")
    print("   clearly negative => in the decay regime, test DOES apply)")

    print("\n=== summary: achieved error vs predicted floor ===")
    summ = df.groupby("arm")[["rmse_r1", "rmse_r2", "y_max"]].median()
    print(summ.to_string())
    print(f"predicted floor: {floor:.4f}")
    for arm_name in ["arm_6", "arm_5"]:
        sub = df[df["arm"] == arm_name]
        print(f"  {arm_name}: rmse_r2/floor median = {(sub['rmse_r2'] / floor).median():.4f}")
    summ.to_csv(RESULTS_DIR / "round3_C5revised_summary.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arm_name, marker in [("arm_6", "s"), ("arm_5", "o")]:
        sub = df[df["arm"] == arm_name]
        ax.scatter(sub["rmse_r1"], sub["rmse_r2"], label=arm_name, marker=marker, alpha=0.7, s=65)
    ax.axhline(floor, color="red", linestyle="--", label=f"predicted floor ({floor:.1f})")
    ax.axvline(floor, color="red", linestyle="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"RMSE at inner shell r1={R1}")
    ax.set_ylabel(f"RMSE at outer shell r2={R2}")
    ax.set_title("C5-revised: both shells in the far field")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "round3_C5revised_two_shell.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'round3_C5revised_two_shell.png'}")


if __name__ == "__main__":
    main()
