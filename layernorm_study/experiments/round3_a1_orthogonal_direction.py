"""Round 3 audit follow-up (item 3, optional half): why is the full
LN+GELU+GLU model's non-radial angle only ~63deg (not ~89deg like
LN+linear), and can a genuinely orthogonal-to-zhat direction be found
for it rather than accepting the far-field exponent as simply
unmeasured?

Two things checked, in order:

1. Does zhat itself rotate substantially across the far-field window
   [10,1000]? If it does, no SINGLE FIXED sweep direction (like a pure
   u-sweep) can stay orthogonal to it throughout - that alone would
   explain why picking a better fixed direction can't fully fix this.

2. Regardless of (1), compute the LOCALLY exact orthogonal direction at
   EVERY sweep point independently (not one fixed global direction):
   at each z=(c,0), let M = dv/d(x,u) (H x 2), zhat = Pv/||Pv||,
   a = M_x . zhat, b = M_u . zhat. d = (b,-a)/||(b,-a)|| is then EXACTLY
   orthogonal to the LOCAL zhat at that point (to first order), by
   construction, regardless of what zhat is doing elsewhere on the
   sweep. The directional derivative of the scalar output F along d at
   that same point is Jx*d0 + Ju*d1 (Jx, Ju already computed by
   _jit_jxju - no new autodiff needed for this part). Fit the log-log
   slope of |Jx*d0+Ju*d1| vs c and compare to the naive Ju-only slope.

Run: python -m layernorm_study.experiments.round3_a1_orthogonal_direction
"""
from __future__ import annotations

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from s4dpc.diagnostics import zero_states
from layernorm_study.src.arms import load_arm_model, train_arm
from layernorm_study.src.plant2 import L_MAX, generate_data
from layernorm_study.src.postnorm_geometry import centering_matrix, _jit_v_and_M
from layernorm_study.src.scalar_diagnostics import _jit_jxju
from layernorm_study.experiments.round3_c3_c4_revised import C3_ARMS, make_arm

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
D_MODEL, N = 8, 16
EPOCHS, LEARNING_RATE = 60000, 1e-3
FAR_FIELD_C = np.logspace(1, 3, 15)  # exact original C3 window
SEED = 0


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_data(seed=42)
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)

    arm = make_arm("LN+GELU+GLU", C3_ARMS["LN+GELU+GLU"])
    param_state, mse = train_arm(arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
                                  epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=SEED)
    model = load_arm_model(arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX,
                            d_input=2, d_output=1, seed=SEED, decode=True)
    print(f"LN+GELU+GLU seed={SEED}: train_mse={mse:.6e} "
          f"({'MATCHES prior audit run (2.685e-05)' if abs(mse - 2.685e-05) < 1e-7 else 'DOES NOT MATCH prior run - investigate'})")

    states = zero_states(model, dtype=jnp.complex128)
    rows = []
    zhats = {}
    for c in FAR_FIELD_C:
        z0 = jnp.array([float(c), 0.0])
        v, M = _jit_v_and_M(model, z0, states)
        v_np, M_np = np.asarray(v), np.asarray(M)
        H = v_np.shape[0]
        P = centering_matrix(H)
        Pv = P @ v_np
        zhat = Pv / np.linalg.norm(Pv)
        zhats[float(c)] = zhat

        jx, ju = _jit_jxju(model, jnp.array([float(c)]), jnp.zeros((1,)), states)
        jx, ju = float(jx), float(ju)

        a = float(M_np[:, 0] @ zhat)  # dv/dx . zhat
        b = float(M_np[:, 1] @ zhat)  # dv/du . zhat
        d = np.array([b, -a])
        dnorm = np.linalg.norm(d)
        if dnorm < 1e-300:
            d0, d1 = 0.0, 0.0
            ortho_deriv = float("nan")
        else:
            d0, d1 = d / dnorm
            ortho_deriv = jx * d0 + ju * d1  # directional derivative of F along the LOCAL orthogonal direction

        rows.append({"c": float(c), "Jx": jx, "Ju": ju, "a_dx_dot_zhat": a, "b_du_dot_zhat": b,
                     "d0": float(d0), "d1": float(d1), "ortho_directional_deriv": ortho_deriv})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "round3_a1_orthogonal_direction.csv", index=False)

    # (1) does zhat itself rotate across the window?
    c_sorted = sorted(zhats)
    z_lo, z_hi = zhats[c_sorted[0]], zhats[c_sorted[-1]]
    end_to_end_angle = np.degrees(np.arccos(np.clip(abs(z_lo @ z_hi), -1, 1)))
    consecutive_angles = []
    for i in range(len(c_sorted) - 1):
        z1, z2 = zhats[c_sorted[i]], zhats[c_sorted[i + 1]]
        consecutive_angles.append(np.degrees(np.arccos(np.clip(abs(z1 @ z2), -1, 1))))
    print(f"\n(1) zhat rotation across the window [c={c_sorted[0]:g}, c={c_sorted[-1]:g}]:")
    print(f"    end-to-end angle(zhat_lo, zhat_hi) = {end_to_end_angle:.2f} deg")
    print(f"    consecutive-point angles: {np.array2string(np.array(consecutive_angles), precision=2, floatmode='fixed')}")
    print(f"    max single-step rotation = {max(consecutive_angles):.2f} deg")

    # (2) locally-orthogonal directional derivative, log-log slope
    good = df["ortho_directional_deriv"].abs() > 0
    slope, _, r, _, se = stats.linregress(np.log10(df["c"][good]), np.log10(df["ortho_directional_deriv"][good].abs()))
    ju_slope, _, ju_r, _, _ = stats.linregress(np.log10(df["c"]), np.log10(df["Ju"].abs()))
    print(f"\n(2) far-field slope comparison, same checkpoint, same 15 points:")
    print(f"    naive Ju-only (fixed u-direction sweep):        slope={ju_slope:+.3f}  r2={ju_r**2:.3f}")
    print(f"    locally-orthogonal directional derivative:      slope={slope:+.3f}  r2={r**2:.3f}")
    print(f"    predicted (LayerNorm's own law, uncancelled): -1")

    with (RESULTS_DIR / "round3_a1_orthogonal_direction_summary.txt").open("w") as f:
        f.write(f"train_mse={mse}\n")
        f.write(f"zhat_end_to_end_angle_deg={end_to_end_angle}\n")
        f.write(f"zhat_max_consecutive_rotation_deg={max(consecutive_angles)}\n")
        f.write(f"naive_ju_slope={ju_slope}, r2={ju_r**2}\n")
        f.write(f"locally_orthogonal_slope={slope}, r2={r**2}\n")


if __name__ == "__main__":
    main()
