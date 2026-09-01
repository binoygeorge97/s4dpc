"""Independent audit of three round-3 numbers, per direct request: don't
re-read NOTES.md's summary, re-derive from the underlying computation.

No model checkpoint or raw Jacobian array is ever persisted to disk in
this sub-project (checkpoints/, *.msgpack gitignored; arms.train_arm
returns an in-memory param state only) - so "the saved artifacts" for
C3/C4 are the CSVs plus the deterministic training procedure that made
them. Determinism is verified, not assumed: LN+linear seed=0's train_mse
is bit-identical (1.3153311644510672e-06) across round3_C3revised_slopes.csv
(written by round3_c3_c4_revised.py) and round3_C3_radial_check.csv
(written by the separate round3_c3_radial_check.py) - two independent
script runs, same (arm, seed) -> same checkpoint. That determinism is what
makes retraining seed=0 here a genuine re-derivation, not a new sample.

A1 - C3 non-radial slope (-1.005, CI [-1.023,-0.982]): is the "non-radial"
     direction (du) actually non-radial (report angle to zhat vs the
     radial direction dx's angle to zhat)? Is the fit window entirely
     far-field (report r* and pointwise local slope across the window)?
     How sensitive is the fitted slope to the window (sub-windows,
     extended range, just-past-r* range)?

A2 - C4 rank=H-2 100% seeds, angles ~1e-14 deg: was jax_enable_x64
     actually active when the Jacobian was computed (print J.dtype
     directly), not just declared at module level? Report the FULL
     singular value spectrum (all H values) for one representative case,
     not just the two smallest.

A3 - C8 failure radius "pinned at 1e-6" for rho<=1.5: what was the
     sweep's smallest tested value (read directly off C_VALUES, not
     inferred)? How many converged seeds per rho actually hit that exact
     floor vs. show a real measured crossing above it? Bonus: retrain one
     "pinned" case (rho=0.5, seed=0) and extend the grid far below 1e-6
     to check whether this is a true at-origin failure or a hidden
     smaller crossing the original grid never probed.

Run: python -m layernorm_study.experiments.round3_audit
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
from scipy import stats

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.conditioning import open_loop_with_resets
from layernorm_study.src.output_ceiling import which_norm_bounds_output
from layernorm_study.src.plant2 import L_MAX as PLANT2_L_MAX
from layernorm_study.src.plant2 import generate_data as plant2_generate_data
from layernorm_study.src.postnorm_geometry import centering_matrix, predicted_r_star
from layernorm_study.src.postnorm_geometry import _jit_v_and_M
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from layernorm_study.experiments.round3_c3_c4_revised import C3_ARMS, c4_null_space_check, make_arm
from layernorm_study.experiments.round3_c8_revised import EXCITATION, B_FIXED

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
FAR_FIELD_C = np.logspace(1, 3, 15)  # exact original C3 window, for cross-check


def train_checkpoint(arm, seed, inputs, targets, l_max):
    param_state, mse = train_arm(arm, inputs, targets, d_model=D_MODEL, N=N, l_max=l_max,
                                  epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed)
    model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=l_max,
                            d_input=2, d_output=1, seed=seed, decode=True)
    return model, mse


def cosine_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def fit_window(model, states, c_values):
    jx_vals, ju_vals = [], []
    for c in c_values:
        jx, ju = _jit_jxju(model, jnp.array([float(c)]), jnp.zeros((1,)), states)
        jx_vals.append(abs(float(jx)))
        ju_vals.append(abs(float(ju)))
    jx_vals, ju_vals = np.array(jx_vals), np.array(ju_vals)
    out = {}
    for label, vals in [("x", jx_vals), ("u", ju_vals)]:
        good = vals > 0
        if good.sum() < 2:
            out[f"slope_{label}"], out[f"r2_{label}"] = float("nan"), float("nan")
            continue
        s, _, r, _, _ = stats.linregress(np.log10(c_values[good]), np.log10(vals[good]))
        out[f"slope_{label}"], out[f"r2_{label}"] = float(s), float(r ** 2)
    return out, jx_vals, ju_vals


def pointwise_local_slopes(c_values, vals):
    """Local log-log slope between EACH adjacent pair - shows whether the
    window is already in the stable asymptotic regime at its near edge,
    rather than trusting a single window-average fit."""
    lc, lv = np.log10(c_values), np.log10(np.maximum(vals, 1e-300))
    return (lv[1:] - lv[:-1]) / (lc[1:] - lc[:-1])


# ---------------------------------------------------------------------
# A1: C3 non-radial slope - direction check + window sensitivity
# ---------------------------------------------------------------------
def audit_a1():
    print("\n" + "=" * 70)
    print("A1 - C3 non-radial slope: direction + window-sensitivity audit")
    print("=" * 70)
    inputs, targets = plant2_generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    checkpoints = [("LN+linear", 0), ("LN+linear", 3), ("LN+GELU+GLU", 0)]
    direction_rows, window_rows = [], []

    for arm_label, seed in checkpoints:
        arm = make_arm(arm_label, C3_ARMS[arm_label])
        model, mse = train_checkpoint(arm, seed, inputs_j, targets_j, PLANT2_L_MAX)
        states = zero_states(model, dtype=jnp.complex128)
        print(f"\n--- {arm_label} seed={seed} (train_mse={mse:.3e}) ---")

        # cross-check against the recorded CSV row before trusting anything new
        full, jx_full, ju_full = fit_window(model, states, FAR_FIELD_C)
        print(f"  cross-check vs CSV, full window [10,1000], 15 pts: "
              f"Jx slope={full['slope_x']:+.4f} (r2={full['r2_x']:.4f}), "
              f"Ju slope={full['slope_u']:+.4f} (r2={full['r2_u']:.4f})")

        # r* for this checkpoint - is FAR_FIELD_C actually far-field?
        rstar = predicted_r_star(model)["r_star_predicted"]
        print(f"  predicted r* = {rstar:.4g}  (window starts at c=10, "
              f"{'>>' if 10 > 20 * rstar else '>' if 10 > 3 * rstar else 'NOT'} r*)")

        # direction check: angle(dx, zhat) vs angle(du, zhat), at a representative far point
        c_rep = 100.0
        v, M = _jit_v_and_M(model, jnp.array([c_rep, 0.0]), states)
        v_np, M_np = np.asarray(v), np.asarray(M)
        H = v_np.shape[0]
        P = centering_matrix(H)
        Pv = P @ v_np
        zhat = Pv / np.linalg.norm(Pv)
        dir_x, dir_u = M_np[:, 0], M_np[:, 1]
        angle_x = cosine_deg(dir_x, zhat)
        angle_u = cosine_deg(dir_u, zhat)
        print(f"  at c={c_rep:g}: angle(dv/dx, zhat) = {angle_x:.3f} deg (radial - should be ~0)")
        print(f"                angle(dv/du, zhat) = {angle_u:.3f} deg (non-radial - should be far from 0)")
        direction_rows.append({"arm": arm_label, "seed": seed, "c": c_rep, "r_star": rstar,
                                "angle_dx_zhat_deg": angle_x, "angle_du_zhat_deg": angle_u,
                                "full_window_slope_x": full["slope_x"], "full_window_slope_u": full["slope_u"]})

        # pointwise local slope across the original window - confirms asymptote, not average-of-transient
        local_x = pointwise_local_slopes(FAR_FIELD_C, jx_full)
        local_u = pointwise_local_slopes(FAR_FIELD_C, ju_full)
        print(f"  local (adjacent-pair) slope of Jx across window: "
              f"{np.array2string(local_x, precision=2, floatmode='fixed')}")
        print(f"  local (adjacent-pair) slope of Ju across window: "
              f"{np.array2string(local_u, precision=2, floatmode='fixed')}")

        # window-sensitivity: sub-windows, extended range, just-past-r*
        windows = {
            "full [10,1000] n=15": FAR_FIELD_C,
            "lower half [10,100] n=8": FAR_FIELD_C[:8],
            "upper half [100,1000] n=8": FAR_FIELD_C[7:],
            "narrow near-edge [10,19] n=3": FAR_FIELD_C[:3],
            "narrow far-edge [373,1000] n=4": FAR_FIELD_C[-4:],
            "extended far [1e3,1e5] n=15": np.logspace(3, 5, 15),
            "just past r* [2r*,20r*] n=10": np.logspace(np.log10(max(2 * rstar, 1e-6)),
                                                          np.log10(max(20 * rstar, 1e-5)), 10),
        }
        for wlabel, cvals in windows.items():
            res, _, _ = fit_window(model, states, cvals)
            window_rows.append({"arm": arm_label, "seed": seed, "window": wlabel,
                                 "c_min": float(cvals.min()), "c_max": float(cvals.max()), "n": len(cvals),
                                 "slope_x": res["slope_x"], "r2_x": res["r2_x"],
                                 "slope_u": res["slope_u"], "r2_u": res["r2_u"]})
            print(f"    [{wlabel:<28}] Jx={res['slope_x']:+.3f} (r2={res['r2_x']:.3f})  "
                  f"Ju={res['slope_u']:+.3f} (r2={res['r2_u']:.3f})")

    pd.DataFrame(direction_rows).to_csv(RESULTS_DIR / "round3_audit_A1_directions.csv", index=False)
    pd.DataFrame(window_rows).to_csv(RESULTS_DIR / "round3_audit_A1_window_sensitivity.csv", index=False)


# ---------------------------------------------------------------------
# A2: C4 rank/dtype - direct dtype print + full singular spectrum
# ---------------------------------------------------------------------
def audit_a2():
    print("\n" + "=" * 70)
    print("A2 - C4 rank/dtype audit: is x64 actually active, full spectrum")
    print("=" * 70)
    print(f"  jax.config.jax_enable_x64 (read back from live config) = {jax.config.jax_enable_x64}")

    inputs, targets = plant2_generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    arm = make_arm("LN+GELU+GLU", C3_ARMS["LN+GELU+GLU"])
    seed = 0
    model, mse = train_checkpoint(arm, seed, inputs_j, targets_j, PLANT2_L_MAX)
    print(f"  checkpoint: LN+GELU+GLU seed={seed} (train_mse={mse:.3e})")

    rows = []
    for radius in [1.0, 1000.0]:
        # cross-check against the exact function that produced the CSV
        csv_row = c4_null_space_check(model, radius)
        print(f"\n  --- radius={radius:g} ---")
        print(f"  cross-check vs c4_null_space_check(): sigma_max={csv_row['sigma_max']:.6e}, "
              f"sigma_H-1={csv_row['sigma_H_minus_1']:.6e}, sigma_H={csv_row['sigma_H']:.6e}, "
              f"angles=({csv_row['angle1_deg']:.4e}, {csv_row['angle2_deg']:.4e}) deg")

        # now the direct, unabbreviated dtype + full-spectrum check
        from layernorm_study.src.postnorm_geometry import pre_final_norm_activation
        norm = which_norm_bounds_output(model)
        states = zero_states(model, dtype=jnp.complex128)
        v, _ = pre_final_norm_activation(model, jnp.array([radius]), jnp.zeros((1,)), states)
        print(f"  v.dtype (pre-norm activation, JAX array) = {v.dtype}")

        J_jax = jax.jacfwd(norm)(v)
        print(f"  J.dtype (LayerNorm Jacobian, JAX array, straight out of jacfwd) = {J_jax.dtype}")
        J_np = np.asarray(J_jax)
        print(f"  J.dtype after np.asarray() = {J_np.dtype}")

        U, S, Vt = np.linalg.svd(J_np)
        H = J_np.shape[0]
        tol_machine = max(J_np.shape) * np.finfo(float).eps * S[0]
        tol_rel = 1e-4 * S[0]
        rank_machine = int((S > tol_machine).sum())
        rank_rel = int((S > tol_rel).sum())
        print(f"  FULL singular value spectrum (all {H}):")
        for i, s in enumerate(S):
            flag = ""
            if i == rank_rel - 1:
                flag = "  <-- rank cut at rel_tol=1e-4"
            if i == H - 2:
                flag += "  [predicted null start, H-2]"
            print(f"    sigma[{i}] = {s:.6e}  (ratio to sigma_max: {s / S[0]:.3e}){flag}")
        print(f"  rank(rel_tol=1e-4) = {rank_rel}, rank(machine_eps) = {rank_machine}  (H-2 predicted = {H - 2})")

        for i, s in enumerate(S):
            rows.append({"radius": radius, "sigma_index": i, "sigma_value": float(s),
                         "sigma_over_max": float(s / S[0]), "J_dtype": str(J_np.dtype)})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "round3_audit_A2_singular_spectrum.csv", index=False)

    # Independent numerical argument, from the ALREADY-COMMITTED CSV alone,
    # that doesn't require rerunning anything: float32 eps ~ 1.19e-7, so a
    # sigma_H/sigma_max ratio anywhere near 1e-14 to 1e-17 (as already
    # recorded for every one of round3_C4fixed_null_space.csv's 16 rows,
    # e.g. seed=0 r=1: 5.66e-17) is physically impossible to produce via
    # float32 rounding - float32 arithmetic cannot resolve a ratio 9-10
    # orders of magnitude below its own machine epsilon. This alone is
    # near-conclusive that the ORIGINAL run used >=float64, independent of
    # today's rerun.
    existing = pd.read_csv(RESULTS_DIR / "round3_C4fixed_null_space.csv")
    worst_case_ratio = existing["sigma_H_over_max"].max()
    print(f"\n  Indirect check from the ALREADY-COMMITTED CSV (no rerun needed): "
          f"worst-case (largest) sigma_H/sigma_max across all 16 recorded rows = {worst_case_ratio:.3e}. "
          f"float32 machine eps = {np.finfo(np.float32).eps:.3e}. "
          f"{worst_case_ratio:.3e} is {np.finfo(np.float32).eps / worst_case_ratio:.1e}x SMALLER than float32 "
          f"eps - float32 arithmetic cannot produce this; the recorded numbers could only come from >=float64.")


# ---------------------------------------------------------------------
# A3: C8 grid floor - censoring accounting + one extended-grid rerun
# ---------------------------------------------------------------------
def audit_a3():
    print("\n" + "=" * 70)
    print("A3 - C8 'pinned at 1e-6': grid floor + censoring breakdown")
    print("=" * 70)

    C_VALUES_ORIGINAL = np.logspace(-6, 3, 60)
    print(f"  C_VALUES = np.logspace(-6, 3, 60): smallest tested value = {C_VALUES_ORIGINAL[0]:.3e}, "
          f"second-smallest = {C_VALUES_ORIGINAL[1]:.3e} (ratio {C_VALUES_ORIGINAL[1] / C_VALUES_ORIGINAL[0]:.3f}x)")
    print("  failure_radius() scans from the SMALLEST c upward and returns the FIRST c whose "
          "relative error exceeds 10% - so if c[0] already fails, the function returns c[0] "
          "WITHOUT ever probing anything smaller. Any reported 1e-6 is a left-censored value.")

    df = pd.read_csv(RESULTS_DIR / "round3_C8revised_results.csv")
    print("\n  per-rho breakdown, CONVERGED seeds only (train_mse < 1e-4):")
    censor_rows = []
    for rho, sub in df.groupby("rho"):
        ok = sub[sub["converged"]]
        at_floor = (ok["failure_radius"] <= C_VALUES_ORIGINAL[0] * 1.0000001)
        n_ok, n_floor = len(ok), int(at_floor.sum())
        measured = ok[~at_floor]
        meas_str = (", ".join(f"seed{int(s)}={v:.4g}" for s, v in zip(measured["seed"], measured["failure_radius"]))
                    if len(measured) else "none")
        print(f"    rho={rho:<5} converged={n_ok}/8  at_grid_floor={n_floor}/{n_ok} "
              f"({100 * n_floor / n_ok if n_ok else float('nan'):.0f}%)  real_measurements: {meas_str}")
        censor_rows.append({"rho": rho, "n_converged": n_ok, "n_at_floor": n_floor,
                             "pct_at_floor": 100 * n_floor / n_ok if n_ok else float("nan"),
                             "measured_values": meas_str})
    pd.DataFrame(censor_rows).to_csv(RESULTS_DIR / "round3_audit_A3_censoring.csv", index=False)

    # Bonus: retrain rho=0.5 seed=0 (a unanimously-censored case) and push
    # the grid FAR below 1e-6 to see whether this is a true at-origin
    # failure or a hidden smaller crossing the original grid never probed.
    print("\n  BONUS: retrain rho=0.5 seed=0 (reported failure_radius=1e-6, converged=True), "
          "extend grid to 1e-15..1e3 to see what's actually happening below the original floor.")
    reset, x0r, apr = EXCITATION[0.5]
    inputs, targets = open_loop_with_resets(0.5, B_FIXED, PLANT2_L_MAX, reset, seed=42,
                                             aprbs_low=-apr, aprbs_high=apr, x0_range=x0r)
    arm = ARMS["arm_6"]
    model, mse = train_checkpoint(arm, 0, jnp.asarray(inputs), jnp.asarray(targets), PLANT2_L_MAX)
    print(f"  retrained train_mse={mse:.3e} (original CSV: 5.417754700918742e-06) "
          f"{'MATCHES' if abs(mse - 5.417754700918742e-06) < 1e-9 else 'DOES NOT MATCH - investigate'}")

    states = zero_states(model, dtype=jnp.complex128)
    u0 = jnp.zeros((1,))
    C_EXTENDED = np.logspace(-15, 3, 91)
    rows = []
    first_below_10pct = None
    for c in C_EXTENDED:
        jx, _ = _jit_jxju(model, jnp.array([float(c)]), u0, states)
        rel_err = abs(float(jx) - 0.5) / 0.5
        rows.append({"c": float(c), "Jx": float(jx), "rel_err": rel_err})
        if rel_err <= 0.10 and first_below_10pct is None:
            first_below_10pct = float(c)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "round3_audit_A3_extended_grid.csv", index=False)

    print(f"  Jx at c=1e-15: {rows[0]['Jx']:+.4f} (true rho=0.5, rel_err={rows[0]['rel_err']:.2%})")
    print(f"  Jx at c=1e-9:  {[r for r in rows if abs(r['c']-1e-9)/1e-9<0.5][0]['Jx']:+.4f}")
    print(f"  Jx at c=1e-6:  {[r for r in rows if abs(r['c']-1e-6)/1e-6<0.5][0]['Jx']:+.4f} "
          f"(this is what the original 60-pt grid's first probe saw)")
    print(f"  Jx at c=1:     {[r for r in rows if abs(r['c']-1)<0.01][0]['Jx']:+.4f}")
    if first_below_10pct is not None:
        print(f"  first c where rel_err <= 10%: {first_below_10pct:.3e} "
              f"(a real crossing DOES exist, just below where the original grid could see it: "
              f"{'YES, hidden below 1e-6' if first_below_10pct < 1e-6 else 'at/above 1e-6, consistent with original'})")
    else:
        print(f"  NO c in [1e-15, 1e3] ever gets within 10% of true rho=0.5 - Jx is wrong across the "
              f"ENTIRE tested range including 15 orders of magnitude below the original grid's floor. "
              f"This is a genuine at-every-scale failure, not a censored-but-nearby crossing.")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_a1()
    audit_a2()
    audit_a3()
    print("\n" + "=" * 70)
    print("audit complete")
    print("=" * 70)
