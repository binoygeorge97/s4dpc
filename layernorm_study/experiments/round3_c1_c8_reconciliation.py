"""Round 3 audit follow-up: does C1's r* mean "correct" or just "constant"?

C1 reported a bias-determined near-field good region (median measured
r*=0.375). C8's deep-grid rerun found Jx >500% wrong at c=1 and >1000%
wrong down to c=1e-15, for a comparably-scaled plant. Direct hypothesis
under test: C1's r* is the radius where Jx departs from ITS OWN
near-origin plateau value, not where it departs from the true (A,B) -
the two have been conflated. `orthogonality_tests.kink_amplitude_and_r_star`
(quoted in NOTES.md's write-up) never references A_TRUE/B_TRUE anywhere
in its body - it is built entirely from the sweep's own far-field median
and near-origin peak. This script makes that structural fact
quantitative: for each of C1(a)'s own 8 with-bias baseline checkpoints
(same arm/plant/seeds/epochs as round2_partC_c1_bias_ablation.py),
compute BOTH |J(z) - J_plateau| and |J(z) - true| across the same radius
sweep C1 used, and report the plateau value itself against (A_TRUE,
B_TRUE) per seed - the number that actually decides whether the
near-field claim is "the constant is correct, its EXTENT is bias-
determined" (C1 as originally read) or merely "there IS a constant,
and its extent is bias-determined, but the constant is wrong" (a much
narrower claim).

Also answers the training-distribution control: is c=1 (and the radii
C8's deep-grid rerun tested) inside or outside the training data's own
|x| range for that checkpoint, so a large error there isn't quietly
extrapolation being reported as a near-field failure.

Run: python -m layernorm_study.experiments.round3_c1_c8_reconciliation
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

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.conditioning import open_loop_with_resets
from layernorm_study.src.orthogonality_tests import kink_amplitude_and_r_star
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data
from layernorm_study.experiments.round3_c8_revised import EXCITATION, B_FIXED

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
C_VALUES = np.concatenate([-np.logspace(3, -6, 60), np.logspace(-6, 3, 60)])  # C1's exact grid
REPORT_RADII = [1e-6, 0.01, 0.1, 0.375, 1.0, 10.0, 100.0, 1000.0]
CORRECTNESS_THRESHOLD = 0.10  # matches C8's own 10% convention


def sweep_xu(model) -> pd.DataFrame:
    """Full (c, Jx, Ju) sweep - reimplemented inline (not calling
    scalar_diagnostics.jx_vs_c_sweep as a black box) so both components
    are guaranteed captured every point, matching C1's exact sweep
    grid/direction (x-direction, u=0 fixed) but keeping Ju too - C1's own
    sweep already computed Ju at every point via _jit_jxju, it just never
    read that column out."""
    from s4dpc.diagnostics import zero_states
    from layernorm_study.src.scalar_diagnostics import _jit_jxju

    states = zero_states(model, dtype=jnp.complex128)
    u = jnp.zeros((1,))
    rows = []
    for c in C_VALUES:
        x = jnp.array([float(c)])
        jx, ju = _jit_jxju(model, x, u, states)
        rows.append({"c": float(c), "Jx": float(jx), "Ju": float(ju)})
    return pd.DataFrame(rows)


def plateau_and_far(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Reproduces kink_amplitude_and_r_star's own jx_peak/jx_far
    definitions exactly (peak = value at the single point closest to
    c=0; far = median of the 5 largest-|c| points on each side), for
    whichever column (Jx or Ju)."""
    vals = df[col].abs().to_numpy()
    c_abs = df["c"].abs().to_numpy()
    peak_idx = int(np.argmin(c_abs))
    peak = float(vals[peak_idx])
    far = float(np.median(np.concatenate([vals[:5], vals[-5:]])))
    return peak, far, peak_idx


def correctness_radius(df: pd.DataFrame, col: str, true_val: float, threshold: float) -> float | None:
    """Mirrors kink_amplitude_and_r_star's OWN scan direction (from the
    far/positive side inward) but against TRUTH instead of the
    plateau: smallest positive c (scanning inward from large c) at
    which relative error to true_val first drops BELOW threshold and
    STAYS below it all the way to c->0. Returns None if no such radius
    exists anywhere in the tested (positive) range - i.e. the model is
    never within `threshold` of truth at ANY tested scale."""
    pos = df[df["c"] > 0].copy()
    pos = pos.sort_values("c", ascending=False)  # far -> near, matching r*'s own scan
    rel_err = (pos[col] - true_val).abs() / abs(true_val)
    ok = rel_err.to_numpy() <= threshold
    if not ok.any():
        return None
    # must stay within threshold from that point all the way to the nearest c
    first_ok = np.argmax(ok)
    if not ok[first_ok:].all():
        # not a clean single crossing - report the LAST point where it first
        # becomes and remains true, i.e. find the last transition into "ok"
        # that holds to the end
        trans = np.where(~ok)[0]
        first_ok = trans[-1] + 1 if len(trans) else first_ok
        if first_ok >= len(ok):
            # even the single nearest-to-origin point is out of tolerance -
            # no radius exists where it enters tolerance AND stays there
            return None
    return float(pos["c"].to_numpy()[first_ok])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"true (A,B) = ({A_TRUE}, {B_TRUE})\n")

    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    summary_rows, knee_rows = [], []
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS["arm_6"], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS["arm_6"], param_state, d_model=D_MODEL, N=N, l_max=L_MAX,
                                d_input=2, d_output=1, seed=seed, decode=True)
        df = sweep_xu(model)
        df["seed"] = seed
        df.to_csv(RESULTS_DIR / f"round3_c1c8_recon_sweep_seed{seed}.csv", index=False)

        # cross-check against the ORIGINAL C1 function directly, not just my reimplementation
        sweep_records = df[["c", "Jx"]].to_dict("records")
        amp_orig, rstar_orig = kink_amplitude_and_r_star(sweep_records)

        jx_peak, jx_far, peak_idx = plateau_and_far(df, "Jx")
        ju_peak, ju_far, _ = plateau_and_far(df, "Ju")
        c_at_peak = float(df["c"].to_numpy()[peak_idx])

        jx_err_at_peak = abs(jx_peak - A_TRUE) / abs(A_TRUE)
        ju_err_at_peak = abs(ju_peak - B_TRUE) / abs(B_TRUE)

        r_correct_x = correctness_radius(df, "Jx", A_TRUE, CORRECTNESS_THRESHOLD)
        r_correct_u = correctness_radius(df, "Ju", B_TRUE, CORRECTNESS_THRESHOLD)

        print(f"=== seed={seed} (train_mse={mse:.3e}) ===")
        print(f"  cross-check vs kink_amplitude_and_r_star(): amplitude={amp_orig:.4f} r_star={rstar_orig:.4e}")
        print(f"  plateau (c={c_at_peak:.2e}, essentially the origin): Jx_plateau={jx_peak:+.4f} "
              f"(true A={A_TRUE}, rel_err={jx_err_at_peak:.1%})  "
              f"Ju_plateau={ju_peak:+.4f} (true B={B_TRUE}, rel_err={ju_err_at_peak:.1%})")
        print(f"  far-field: Jx_far={jx_far:.4f}  Ju_far={ju_far:.4f}")
        print(f"  'plateau radius' r* (departs from OWN plateau)      = {rstar_orig:.4e}")
        print(f"  'correctness radius' (departs from TRUE (A,B), <=10%): "
              f"Jx: {r_correct_x if r_correct_x is not None else 'NEVER within 10% at any tested c'}  "
              f"Ju: {r_correct_u if r_correct_u is not None else 'NEVER within 10% at any tested c'}")

        for radius in REPORT_RADII:
            row_pos = df.iloc[(df["c"] - radius).abs().argsort()[:1]]
            c_actual = float(row_pos["c"].iloc[0])
            jx_here = float(row_pos["Jx"].iloc[0])
            knee_rows.append({
                "seed": seed, "radius_target": radius, "c_actual": c_actual, "Jx": jx_here,
                "dist_to_plateau": abs(jx_here - jx_peak), "dist_to_truth": abs(jx_here - A_TRUE),
            })

        summary_rows.append({
            "seed": seed, "train_mse": mse,
            "r_star_plateau_radius": rstar_orig, "amplitude": amp_orig,
            "jx_plateau": jx_peak, "jx_plateau_relerr_vs_ATRUE": jx_err_at_peak,
            "ju_plateau": ju_peak, "ju_plateau_relerr_vs_BTRUE": ju_err_at_peak,
            "jx_far": jx_far, "ju_far": ju_far,
            "r_correctness_radius_Jx_10pct": r_correct_x,
            "r_correctness_radius_Ju_10pct": r_correct_u,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "round3_c1c8_recon_summary.csv", index=False)
    pd.DataFrame(knee_rows).to_csv(RESULTS_DIR / "round3_c1c8_recon_knee_table.csv", index=False)

    print("\n=== THE DECISIVE NUMBER: plateau value vs true (A,B), across all 8 seeds ===")
    print(summary_df[["seed", "jx_plateau", "jx_plateau_relerr_vs_ATRUE",
                        "ju_plateau", "ju_plateau_relerr_vs_BTRUE"]].to_string(index=False))
    print(f"\nmedian jx_plateau_relerr_vs_ATRUE = {summary_df['jx_plateau_relerr_vs_ATRUE'].median():.1%}")
    print(f"median ju_plateau_relerr_vs_BTRUE = {summary_df['ju_plateau_relerr_vs_BTRUE'].median():.1%}")
    n_never_x = summary_df["r_correctness_radius_Jx_10pct"].isna().sum()
    print(f"seeds where Jx is NEVER within 10% of true A at any tested c: {n_never_x}/8")

    # --- part (d): training-distribution control for C8's rho=0.5 checkpoint ---
    print("\n=== (d) training-distribution control: is c=1 inside C8's rho=0.5 training range? ===")
    reset, x0r, apr = EXCITATION[0.5]
    c8_inputs, _ = open_loop_with_resets(0.5, B_FIXED, 100, reset, seed=42,
                                          aprbs_low=-apr, aprbs_high=apr, x0_range=x0r)
    x_col = np.abs(np.asarray(c8_inputs)[:, 0])
    pct = np.percentile(x_col, [0, 5, 25, 50, 75, 95, 100])
    frac_le_1 = float((x_col <= 1.0).mean())
    print(f"  |x| percentiles [0,5,25,50,75,95,100]: {np.array2string(pct, precision=3, floatmode='fixed')}")
    print(f"  fraction of the 100 training points with |x| <= 1.0: {frac_le_1:.0%}")
    print(f"  fraction with |x| <= 1e-6 (where the '1000% wrong' figure was read): "
          f"{float((x_col <= 1e-6).mean()):.0%} (expected ~0 - the plateau is a THEORETICAL c->0 "
          f"limit, not literally sampled by training data; what matters is whether small-|x| "
          f"REGIONS are represented, not the exact point)")
    pd.DataFrame({"radius_pctile": [0, 5, 25, 50, 75, 95, 100], "abs_x": pct}).to_csv(
        RESULTS_DIR / "round3_c1c8_recon_c8_training_distribution.csv", index=False)


if __name__ == "__main__":
    main()
