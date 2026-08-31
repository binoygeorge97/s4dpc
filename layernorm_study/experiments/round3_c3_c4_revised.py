"""Round 3: C3-revised (decay-slope ablation) and C4-fixed (2D null space).

C3-revised. Round 2 measured a far-field slope of -2.33 (CI excluding
-1) and recorded it as a bare FAIL. The likely cause, per the user's
correction: GLU is a PRODUCT a(v)*sigmoid(b(v)). If both arguments are
post-LN activations that each decay like 1/r, the product picks up a
SECOND factor of 1/r, giving 1/r^2 - composed degree-0-ish stages
compound, so -1 is LayerNorm's law ALONE, not the whole block's.
Tested by peeling the nonlinearities off, all postnorm, all else equal:

    LN + GELU + GLU  (arm_6)      -> predicted ~ -2
    LN + GELU, no GLU             -> predicted between -1 and -2
    LN + linear (no GELU, no GLU) -> predicted ~ -1

C4-fixed. Round 2's null-direction sub-check compared the single
smallest right singular vector against one target vector and got
~0.28-0.31 (at/below the R^8 random baseline of 0.354) - flagged then
as a measurement-definition problem, now corrected. LayerNorm's
Jacobian (1/sigma) diag(gamma) [P - zhat zhat^T / H] has TWO exact null
directions by construction: the all-ones vector 1 (killed by P) and
zhat (P leaves it, the outer-product term cancels it). So the correct
test is rank(J) <= H-2 plus the PRINCIPAL ANGLES between the measured
2D null space and span{1, zhat} - not alignment with any single vector.

Note: span{1, v} == span{1, zhat} exactly, since zhat is proportional
to Pv = v - mean(v)*1, so each of v, zhat is in the other's span once
1 is included. The user's "span{1, vhat}" and the derivation's
span{1, zhat} are the same 2D subspace; this script uses Pv (i.e. zhat's
direction) explicitly.

Run: python -m layernorm_study.experiments.round3_c3_c4_revised
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
from flax import nnx
from scipy import stats
from scipy.linalg import subspace_angles

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import ARMS, ArmSpec, load_arm_model, train_arm
from layernorm_study.src.output_ceiling import which_norm_bounds_output
from layernorm_study.src.plant2 import A_TRUE, L_MAX, generate_data
from layernorm_study.src.postnorm_geometry import pre_final_norm_activation
from layernorm_study.src.scalar_diagnostics import _jit_jxju

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
FAR_FIELD_C = np.logspace(1, 3, 15)
C4_RADII = [1.0, 1e3]

C3_ARMS = {
    "LN+GELU+GLU": dict(norm="layer", activation="gelu", glu=True, memoryless=False, prenorm=False),
    "LN+GELU": dict(norm="layer", activation="gelu", glu=False, memoryless=False, prenorm=False),
    "LN+linear": dict(norm="layer", activation="none", glu=False, memoryless=False, prenorm=False),
}


def make_arm(name: str, kwargs: dict) -> ArmSpec:
    return dataclasses.replace(ARMS["arm_6"], name=name, description=f"postnorm {name}", block_kwargs=kwargs)


@nnx.jit
def _jit_ln_jacobian(norm_module, v: jax.Array):
    return jax.jacfwd(norm_module)(v)


def far_field_slope(model) -> tuple[float, float, float]:
    """Returns (slope, stderr, r2) of log|Jx| vs log|x| in the far field."""
    states = zero_states(model, dtype=jnp.complex128)
    u0 = jnp.zeros((1,))
    jx = []
    for c in FAR_FIELD_C:
        j, _ = _jit_jxju(model, jnp.array([float(c)]), u0, states)
        jx.append(abs(float(j)))
    jx = np.array(jx)
    good = jx > 0
    slope, _, r, _, se = stats.linregress(np.log10(FAR_FIELD_C[good]), np.log10(jx[good]))
    return float(slope), float(se), float(r ** 2)


def c4_null_space_check(model, radius: float) -> dict:
    """rank(J_LN) and principal angles between its measured 2D null
    space and span{1, zhat}."""
    norm = which_norm_bounds_output(model)
    states = zero_states(model, dtype=jnp.complex128)
    v, _ = pre_final_norm_activation(model, jnp.array([radius]), jnp.zeros((1,)), states)
    v_np = np.asarray(v)
    J = np.asarray(_jit_ln_jacobian(norm, v))
    H = J.shape[0]

    U, S, Vt = np.linalg.svd(J)
    # Rank at TWO tolerances, not one. The all-ones null direction is
    # exact to machine precision (P kills it identically), but the zhat
    # direction is annihilated exactly only in the eps->0 limit: with
    # LayerNorm's eps=1e-6, sigma=sqrt(||Pv||^2/H + eps), so that second
    # singular value is small-but-not-machine-zero. Reporting only a
    # machine-eps rank would therefore say "rank H-1" and hide the real
    # 2D structure; reporting both makes the eps dependence visible
    # instead of asserting the cleaner number.
    tol_machine = max(J.shape) * np.finfo(float).eps * S[0]
    tol_rel = 1e-4 * S[0]
    rank_machine = int((S > tol_machine).sum())
    rank_rel = int((S > tol_rel).sum())

    measured_null = Vt[-2:].T  # (H, 2): the two smallest right singular vectors
    ones = np.ones(H)
    Pv = v_np - v_np.mean()  # zhat's direction (span{1,v} == span{1,Pv} == span{1,zhat})
    base = {
        "radius": radius, "H": H,
        "rank_machine_tol": rank_machine, "rank_rel_tol_1e-4": rank_rel,
        "rank_le_H_minus_2_at_rel_tol": rank_rel <= H - 2,
        "sigma_max": float(S[0]), "sigma_H_minus_1": float(S[-2]), "sigma_H": float(S[-1]),
        "sigma_H_minus_1_over_max": float(S[-2] / S[0]), "sigma_H_over_max": float(S[-1] / S[0]),
    }
    if np.linalg.norm(Pv) < 1e-300:
        return {**base, "angle1_deg": float("nan"), "angle2_deg": float("nan")}
    target = np.stack([ones / np.linalg.norm(ones), Pv / np.linalg.norm(Pv)], axis=1)  # (H, 2)
    angles = np.degrees(subspace_angles(measured_null, target))
    return {**base, "angle1_deg": float(min(angles)), "angle2_deg": float(max(angles))}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    c3_rows, c4_rows = [], []
    for arm_label, kwargs in C3_ARMS.items():
        arm = make_arm(arm_label, kwargs)
        print(f"=== C3-revised: {arm_label} ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX,
                                    d_input=2, d_output=1, seed=seed, decode=True)
            slope, se, r2 = far_field_slope(model)
            c3_rows.append({"arm": arm_label, "seed": seed, "train_mse": mse,
                             "slope": slope, "stderr": se, "r2": r2})
            print(f"  seed={seed}: train_mse={mse:.3e} far-field slope={slope:+.3f} "
                  f"(se={se:.3f}, r2={r2:.3f})", flush=True)

            # C4 on the FULL block only (the architecture round 2 tested)
            if arm_label == "LN+GELU+GLU":
                for radius in C4_RADII:
                    row = c4_null_space_check(model, radius)
                    row["seed"] = seed
                    c4_rows.append(row)
                    print(f"    C4 r={radius:g}: H={row['H']} rank(machine)={row['rank_machine_tol']} "
                          f"rank(rel 1e-4)={row['rank_rel_tol_1e-4']} | angles=("
                          f"{row['angle1_deg']:.2e}, {row['angle2_deg']:.2e}) deg | "
                          f"s_H-1/s_max={row['sigma_H_minus_1_over_max']:.2e}", flush=True)

    c3_df, c4_df = pd.DataFrame(c3_rows), pd.DataFrame(c4_rows)
    c3_df.to_csv(RESULTS_DIR / "round3_C3revised_slopes.csv", index=False)
    c4_df.to_csv(RESULTS_DIR / "round3_C4fixed_null_space.csv", index=False)

    print("\n=== C3-revised summary: does the slope walk toward -1 as nonlinearities come off? ===")
    for arm_label in C3_ARMS:
        sub = c3_df[c3_df["arm"] == arm_label]["slope"]
        lo, hi = np.percentile(sub, [2.5, 97.5])
        print(f"  {arm_label:<14} median slope={sub.median():+.3f}  95% CI across seeds [{lo:+.3f}, {hi:+.3f}]")
    c3_df.groupby("arm")["slope"].describe().to_csv(RESULTS_DIR / "round3_C3revised_summary.csv")

    print("\n=== C4-fixed summary (2D null space, H=8 so predicted rank <= 6) ===")
    for radius in C4_RADII:
        sub = c4_df[c4_df["radius"] == radius]
        print(f"  r={radius:g}: median rank(machine tol)={sub['rank_machine_tol'].median():.1f}, "
              f"median rank(rel 1e-4)={sub['rank_rel_tol_1e-4'].median():.1f} "
              f"({sub['rank_le_H_minus_2_at_rel_tol'].mean()*100:.0f}% of seeds rank<=H-2 at rel tol)")
        print(f"        principal angles to span{{1,zhat}}: median "
              f"({sub['angle1_deg'].median():.2e}, {sub['angle2_deg'].median():.2e}) deg  "
              f"| median s_H-1/s_max={sub['sigma_H_minus_1_over_max'].median():.2e} "
              f"s_H/s_max={sub['sigma_H_over_max'].median():.2e}")
    c4_df.groupby("radius")[["rank_machine_tol", "rank_rel_tol_1e-4", "angle1_deg", "angle2_deg"]].median().to_csv(
        RESULTS_DIR / "round3_C4fixed_summary.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    order = list(C3_ARMS)
    ax.boxplot([c3_df[c3_df["arm"] == a]["slope"].values for a in order], tick_labels=order)
    ax.axhline(-1, color="tab:red", linestyle="--", label="LayerNorm-alone prediction (-1)")
    ax.axhline(-2, color="tab:blue", linestyle=":", label="LN+GLU compounding prediction (-2)")
    ax.set_ylabel("far-field log-log slope of |Jx|")
    ax.set_title("C3-revised: slope vs block nonlinearity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "round3_C3revised_slopes.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'round3_C3revised_slopes.png'}")


if __name__ == "__main__":
    main()
