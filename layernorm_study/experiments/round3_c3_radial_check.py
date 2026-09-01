"""Round 3, C3 supplementary: is the far-field slope -2 (not -1) because
the swept direction is RADIAL, via the Euler identity C4 just confirmed?

C3-revised peeled off GLU and then GELU and the slope did NOT walk to
-1: LN+linear still measures ~-1.9. Rather than record that as a second
bare FAIL, here is the mechanism that predicts it, and a test that
discriminates.

For a postnorm block with a purely linear branch, v(z) = b + Mz is
exactly affine, so sigma ~ ||PMz||/sqrt(H) ~ c|z| in the far field and
dLN/dv ~ 1/sigma ~ 1/|z| - which naively gives slope -1. BUT the
bracket [P - zhat zhat^T/H] ANNIHILATES zhat (C4's Euler identity,
confirmed to principal angles ~1e-14 deg), and zhat is by definition
the normalized centered direction of v = b + Mz. Sweeping the SCALAR
state x along a ray moves v essentially radially, so the perturbation
direction M*e_x is (asymptotically) parallel to zhat itself - exactly
the direction the bracket kills. The leading 1/|z| term therefore
CANCELS, and dF/dx is set by the subleading term: 1/|z|^2, i.e. slope
-2.

Discriminating prediction: the u-direction is NOT the swept radial
direction, so dF/du should retain the uncancelled 1/|z| term.

    sweep |x| -> infinity, measure both:
        d(F)/dx  (radial)     -> predicted slope ~ -2
        d(F)/du  (non-radial) -> predicted slope ~ -1

If Ju decays as -1 while Jx decays as -2 on the SAME checkpoints, the
-1 law is confirmed as LayerNorm's own, and the -2 seen throughout C3
is the Euler-identity cancellation along the swept direction - not a
failure of the theory and not GLU compounding.

Run: python -m layernorm_study.experiments.round3_c3_radial_check
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

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import load_arm_model, train_arm
from layernorm_study.src.plant2 import L_MAX, generate_data
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from layernorm_study.experiments.round3_c3_c4_revised import C3_ARMS, make_arm

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
FAR_FIELD_C = np.logspace(1, 3, 15)


def both_slopes(model) -> tuple[float, float, float, float]:
    """Far-field log-log slopes of |Jx| (radial, the swept direction)
    and |Ju| (non-radial) on the same checkpoint, same sweep."""
    states = zero_states(model, dtype=jnp.complex128)
    u0 = jnp.zeros((1,))
    jx_vals, ju_vals = [], []
    for c in FAR_FIELD_C:
        jx, ju = _jit_jxju(model, jnp.array([float(c)]), u0, states)
        jx_vals.append(abs(float(jx)))
        ju_vals.append(abs(float(ju)))
    out = []
    for vals in (jx_vals, ju_vals):
        vals = np.array(vals)
        good = vals > 0
        s, _, r, _, se = stats.linregress(np.log10(FAR_FIELD_C[good]), np.log10(vals[good]))
        out += [float(s), float(r ** 2)]
    return tuple(out)  # (jx_slope, jx_r2, ju_slope, ju_r2)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    rows = []
    for arm_label in ["LN+linear", "LN+GELU+GLU"]:
        arm = make_arm(arm_label, C3_ARMS[arm_label])
        print(f"=== {arm_label} ===")
        for seed in SEEDS:
            param_state, mse = train_arm(arm, inputs_j, targets_j, d_model=D_MODEL, N=N,
                                          l_max=L_MAX, epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed)
            model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX,
                                    d_input=2, d_output=1, seed=seed, decode=True)
            jx_s, jx_r2, ju_s, ju_r2 = both_slopes(model)
            rows.append({"arm": arm_label, "seed": seed, "train_mse": mse,
                          "jx_slope_radial": jx_s, "jx_r2": jx_r2,
                          "ju_slope_nonradial": ju_s, "ju_r2": ju_r2})
            print(f"  seed={seed}: Jx(radial) slope={jx_s:+.3f} (r2={jx_r2:.3f}) | "
                  f"Ju(non-radial) slope={ju_s:+.3f} (r2={ju_r2:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round3_C3_radial_check.csv", index=False)

    print("\n=== summary: radial vs non-radial far-field slope ===")
    for arm_label in ["LN+linear", "LN+GELU+GLU"]:
        sub = df[df["arm"] == arm_label]
        jl, jh = np.percentile(sub["jx_slope_radial"], [2.5, 97.5])
        ul, uh = np.percentile(sub["ju_slope_nonradial"], [2.5, 97.5])
        print(f"  {arm_label:<14} Jx(radial)     median={sub['jx_slope_radial'].median():+.3f} CI[{jl:+.3f},{jh:+.3f}]")
        print(f"  {'':<14} Ju(non-radial) median={sub['ju_slope_nonradial'].median():+.3f} CI[{ul:+.3f},{uh:+.3f}]")
    print("\npredicted: radial ~ -2 (Euler cancellation), non-radial ~ -1 (LayerNorm's own law)")
    df.groupby("arm")[["jx_slope_radial", "ju_slope_nonradial"]].median().to_csv(
        RESULTS_DIR / "round3_C3_radial_check_summary.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels, data = [], []
    for arm_label in ["LN+linear", "LN+GELU+GLU"]:
        sub = df[df["arm"] == arm_label]
        labels += [f"{arm_label}\nJx (radial)", f"{arm_label}\nJu (non-radial)"]
        data += [sub["jx_slope_radial"].values, sub["ju_slope_nonradial"].values]
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(-1, color="tab:red", linestyle="--", label="LayerNorm's own law (-1)")
    ax.axhline(-2, color="tab:blue", linestyle=":", label="Euler-cancelled radial (-2)")
    ax.set_ylabel("far-field log-log slope")
    ax.set_title("C3 supplementary: radial vs non-radial decay")
    ax.legend()
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "round3_C3_radial_check.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'round3_C3_radial_check.png'}")


if __name__ == "__main__":
    main()
