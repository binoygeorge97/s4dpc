"""Round 3, C8-revised: gain sweep with SCALE-MATCHED, validated excitation.

Round 2's C8 was AMBIGUOUS and the cause is now identified precisely.
The per-rho excitation was not validated before use, and it produced
wildly different DATA SCALES across the sweep: max|x| ranged 19.5
(rho=0.5) to 311 (rho=1.5). Since postnorm's output ceiling Y_max is
~120-150 for this architecture (round 2's C2/C6-revised), the rho=1.5
dataset required outputs of ~467 - the model could not represent that
data AT ALL, so its "failure radius" reflected a total fit failure,
not a gain-dependent effect. That confound made every rho report the
same floor value (1e-6) with no discrimination.

Fixed here by choosing (reset_every, x0_range, aprbs_range) per rho so
that max|x| ~ 20 for EVERY gain - a controlled comparison in which the
required output magnitude is held roughly constant and only rho varies.
Each dataset is validated BEFORE training (least-squares recovery of
(A,B), and condition number), and training convergence is reported per
run so a repeat of round 2's silent non-convergence is visible.

Run: python -m layernorm_study.experiments.round3_c8_revised
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

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.conditioning import open_loop_with_resets
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from layernorm_study.src.scalar_system import regressor_condition_number

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
SEEDS = list(range(8))
D_MODEL, N, L_MAX = 8, 16, 100
EPOCHS, LEARNING_RATE = 60000, 1e-3
B_FIXED = 1.0
THRESHOLD = 0.10
C_VALUES = np.logspace(-6, 3, 60)

# (reset_every, x0_range, aprbs_amplitude) per rho, chosen by a scan so
# max|x| ~ 20 for every gain (round 2's version left this uncontrolled).
EXCITATION = {
    0.5: (8, 0.5, 10.0),
    0.9: (3, 2.0, 10.0),
    1.03: (3, 0.5, 10.0),
    1.5: (3, 4.0, 5.0),
    3.0: (3, 0.5, 5.0),
}
LS_TOLERANCE = 1e-9  # data-sanity gate, matching this project's convention elsewhere


def failure_radius(model, rho: float) -> float:
    states = zero_states(model, dtype=jnp.complex128)
    u0 = jnp.zeros((1,))
    for c in C_VALUES:
        jx, _ = _jit_jxju(model, jnp.array([float(c)]), u0, states)
        if abs(float(jx) - rho) / abs(rho) > THRESHOLD:
            return float(c)
    return float(C_VALUES[-1])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== excitation validation (BEFORE any training) ===")
    datasets, val_rows = {}, []
    for rho, (reset, x0r, apr) in EXCITATION.items():
        inputs, targets = open_loop_with_resets(rho, B_FIXED, L_MAX, reset, seed=42,
                                                 aprbs_low=-apr, aprbs_high=apr, x0_range=x0r)
        z, y = np.asarray(inputs, float), np.asarray(targets, float)
        ab, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
        a_err, b_err = abs(ab[0, 0] - rho), abs(ab[1, 0] - B_FIXED)
        cond = regressor_condition_number(inputs)
        max_x = float(np.abs(z[:, 0]).max())
        val_rows.append({"rho": rho, "reset_every": reset, "x0_range": x0r, "aprbs": apr,
                          "max_abs_x": max_x, "cond": cond, "ls_A_err": a_err, "ls_B_err": b_err})
        print(f"  rho={rho}: max|x|={max_x:.2f} cond={cond:.2f} LS_A_err={a_err:.2e} LS_B_err={b_err:.2e}")
        if a_err > LS_TOLERANCE or b_err > LS_TOLERANCE:
            raise SystemExit(f"EXCITATION VALIDATION FAILED at rho={rho} - do not train on this data.")
        datasets[rho] = (jnp.asarray(inputs), jnp.asarray(targets))
    pd.DataFrame(val_rows).to_csv(RESULTS_DIR / "round3_C8revised_excitation_validation.csv", index=False)
    max_xs = [r["max_abs_x"] for r in val_rows]
    print(f"PASS: all LS recoveries < {LS_TOLERANCE:.0e}; max|x| spread {min(max_xs):.1f}-{max(max_xs):.1f} "
          f"(round 2's uncontrolled version: 19.5-311)")

    rows = []
    for rho in EXCITATION:
        inputs_j, targets_j = datasets[rho]
        print(f"\n=== rho={rho} ===")
        for seed in SEEDS:
            param_state, mse = train_arm(
                ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
            )
            model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N,
                                    l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True)
            r_fail = failure_radius(model, rho)
            rows.append({"rho": rho, "seed": seed, "train_mse": mse, "failure_radius": r_fail,
                          "converged": mse < 1e-4})
            print(f"  seed={seed}: train_mse={mse:.3e} failure_radius={r_fail:.4e} "
                  f"{'' if mse < 1e-4 else '<-- POOR CONVERGENCE'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round3_C8revised_results.csv", index=False)

    print("\n=== convergence audit (round 2's silent failure mode) ===")
    conv = df.groupby("rho")["converged"].mean()
    print(conv.to_string())
    print(f"overall: {df['converged'].mean()*100:.0f}% of runs converged below train_mse 1e-4")

    print("\n=== summary: median failure radius per rho (converged runs only) ===")
    ok = df[df["converged"]]
    if len(ok) == 0:
        print("NO converged runs - C8 remains unusable; report as such, do not interpret.")
    else:
        print(ok.groupby("rho")["failure_radius"].agg(["median", "min", "max", "count"]).to_string())
    df.groupby("rho")["failure_radius"].median().to_csv(RESULTS_DIR / "round3_C8revised_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for rho in EXCITATION:
        sub = df[df["rho"] == rho]
        good, bad = sub[sub["converged"]], sub[~sub["converged"]]
        ax.scatter([rho] * len(good), good["failure_radius"], alpha=0.75, s=55, color="tab:blue")
        ax.scatter([rho] * len(bad), bad["failure_radius"], alpha=0.5, s=55, color="tab:red", marker="x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true rho (log)")
    ax.set_ylabel("failure radius (log)")
    ax.set_title("C8-revised: failure radius vs gain (red x = poorly converged)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "round3_C8revised_gain_sweep.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'round3_C8revised_gain_sweep.png'}")


if __name__ == "__main__":
    main()
