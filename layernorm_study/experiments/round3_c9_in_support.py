"""Round 3, C9 - the in-support test.

Direct response to a flagged confound in the C1/C8 reconciliation: 0/100
of plant2's training points have |x| <= 0.375 (C1's own median r*), so
the earlier "plateau is 195%/11353% wrong" finding is evidence about
DATA COVERAGE, not about LayerNorm specifically - at radii the loss
function never scored, no architecture has a reason to be accurate.

C9 asks the question properly: restrict to where training data ACTUALLY
lives, and compare postnorm (arm_6) against prenorm (arm_5) and a
no-norm control (arm_4, this study's standing "no LN, no GELU, no GLU,
real S4 memory" baseline - matches the parent project's M3) on the SAME
plant2 data and the SAME 8 seeds C1 used.

Two in-support measurements, not one, so a synthetic dense sweep isn't
the only evidence:

  1. PRIMARY - dense synthetic sweep, zero-state convention (matching
     every other radius sweep in this study, so it's comparable to
     C1/C3/C8's numbers): |x| swept densely across [x_min, x_max] (the
     empirical training range), u=0, Jx/Ju vs (A_TRUE, B_TRUE).
  2. SECONDARY - the ACTUAL 100 real (x_t, u_t) points, trajectory-
     evolved state (scalar_diagnostics.jx_ju_along_trajectory, already
     existing project code, not reimplemented) - the most literal
     possible "in support" test, and where the real |z|=sqrt(x^2+u^2)
     distribution actually comes from.

Usage (run 3x in parallel, one per arm):
    python -m layernorm_study.experiments.round3_c9_in_support arm_4
    python -m layernorm_study.experiments.round3_c9_in_support arm_5
    python -m layernorm_study.experiments.round3_c9_in_support arm_6
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

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.plant2 import A_TRUE, B_TRUE, L_MAX, generate_data
from layernorm_study.src.scalar_diagnostics import _jit_jxju, jx_ju_along_trajectory

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = list(range(8))
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
N_DENSE = 40


def dense_in_support_sweep(model, x_lo: float, x_hi: float) -> pd.DataFrame:
    states = zero_states(model, dtype=jnp.complex128)
    u = jnp.zeros((1,))
    xs = np.linspace(x_lo, x_hi, N_DENSE)
    rows = []
    for x in xs:
        jx, ju = _jit_jxju(model, jnp.array([float(x)]), u, states)
        jx, ju = float(jx), float(ju)
        rows.append({
            "x": float(x), "Jx": jx, "Ju": ju,
            "jx_relerr": abs(jx - A_TRUE) / abs(A_TRUE),
            "ju_relerr": abs(ju - B_TRUE) / abs(B_TRUE),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: python -m layernorm_study.experiments.round3_c9_in_support <arm_name>, "
                          f"got {sys.argv[1:]!r}, valid: {list(ARMS)}")
    arm_name = sys.argv[1]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    inputs, targets = generate_data(seed=42)
    x_col = np.asarray(inputs)[:, 0]
    u_col = np.asarray(inputs)[:, 1]
    z_norm = np.sqrt(x_col ** 2 + u_col ** 2)
    x_abs = np.abs(x_col)
    x_lo, x_hi = float(x_abs.min()), float(x_abs.max())

    print(f"=== C9, arm={arm_name} ===")
    print(f"true (A,B) = ({A_TRUE}, {B_TRUE})")
    print(f"training data (100 real points): |x| in [{x_lo:.4f}, {x_hi:.4f}], "
          f"|x| quartiles {np.percentile(x_abs, [0,25,50,75,100])}")
    print(f"training data |u| quartiles: {np.percentile(np.abs(u_col), [0,25,50,75,100])}")
    print(f"training data |z|=sqrt(x^2+u^2) quartiles: {np.percentile(z_norm, [0,25,50,75,100])}")
    print(f"IN-SUPPORT INTERVAL (this script's definition): |x| in [{x_lo:.4f}, {x_hi:.4f}]")

    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    dense_rows, traj_summary_rows = [], []
    for seed in SEEDS:
        param_state, mse = train_arm(
            ARMS[arm_name], inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = load_arm_model(ARMS[arm_name], param_state, d_model=D_MODEL, N=N, l_max=L_MAX,
                                d_input=2, d_output=1, seed=seed, decode=True)

        dense = dense_in_support_sweep(model, x_lo, x_hi)
        dense["seed"] = seed
        dense["arm"] = arm_name
        dense_rows.append(dense)

        traj = pd.DataFrame(jx_ju_along_trajectory(model, inputs_j, targets_j))
        traj["jx_relerr"] = (traj["Jx"] - A_TRUE).abs() / abs(A_TRUE)
        traj["ju_relerr"] = (traj["Ju"] - B_TRUE).abs() / abs(B_TRUE)

        print(f"  seed={seed} (train_mse={mse:.3e}):")
        print(f"    PRIMARY (dense synthetic, in-support |x|, u=0, n={N_DENSE}): "
              f"median jx_relerr={dense['jx_relerr'].median():.1%}  median ju_relerr={dense['ju_relerr'].median():.1%}")
        print(f"    SECONDARY (actual 100 real trajectory points, real u, evolved state): "
              f"median jx_relerr={traj['jx_relerr'].median():.1%}  median ju_relerr={traj['ju_relerr'].median():.1%}")

        traj_summary_rows.append({
            "arm": arm_name, "seed": seed, "train_mse": mse,
            "dense_jx_relerr_median": dense["jx_relerr"].median(),
            "dense_ju_relerr_median": dense["ju_relerr"].median(),
            "dense_jx_relerr_max": dense["jx_relerr"].max(),
            "traj_jx_relerr_median": traj["jx_relerr"].median(),
            "traj_ju_relerr_median": traj["ju_relerr"].median(),
            "traj_jx_relerr_max": traj["jx_relerr"].max(),
        })

    dense_df = pd.concat(dense_rows, ignore_index=True)
    dense_df.to_csv(RESULTS_DIR / f"round3_C9_dense_sweep_{arm_name}.csv", index=False)
    summary_df = pd.DataFrame(traj_summary_rows)
    summary_df.to_csv(RESULTS_DIR / f"round3_C9_summary_{arm_name}.csv", index=False)

    print(f"\n=== C9 summary, arm={arm_name}, across all 8 seeds ===")
    print(f"PRIMARY (dense, in-support, u=0):    median jx_relerr = {summary_df['dense_jx_relerr_median'].median():.1%}  "
          f"(range {summary_df['dense_jx_relerr_median'].min():.1%}-{summary_df['dense_jx_relerr_median'].max():.1%})")
    print(f"                                       median ju_relerr = {summary_df['dense_ju_relerr_median'].median():.1%}")
    print(f"SECONDARY (actual real data points):  median jx_relerr = {summary_df['traj_jx_relerr_median'].median():.1%}  "
          f"(range {summary_df['traj_jx_relerr_median'].min():.1%}-{summary_df['traj_jx_relerr_median'].max():.1%})")
    print(f"                                       median ju_relerr = {summary_df['traj_ju_relerr_median'].median():.1%}")


if __name__ == "__main__":
    main()
