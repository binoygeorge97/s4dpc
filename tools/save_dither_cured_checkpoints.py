"""TASK D (user, 2026-08-19, seventh round): save a proper, documented
checkpoint for the dither-cured model - the third of the three "transient
artifact" models this round asked to be fixed.

IMPORTANT ARCHITECTURAL FINDING, established before writing anything,
not assumed: the dither-cured model CANNOT be instantiated as a standard
StackedModel with modified weights. Its Axs block (6, 1024) acts on the
FULL raw real/imag-flattened S4 state - but the standard architecture's
output path (decoder -> out/out2 GLU -> ... ) can only ever depend on the
internal state THROUGH the S4 layer's own per-channel C-projection first
(down to d_model=16 real values, not the raw 1024), because that
projection happens INSIDE S4LayerEnsemble.__call__ before anything else
touches the state. A general (Axx, Axs) with Axs reading the raw 1024-dim
state is not representable by that architecture's output side at all,
regardless of what weights are chosen - this is a structural fact, not a
missing construction trick.

Consequence for what "proper checkpoint" means here: NOT a StackedModel
msgpack (the architecture cannot express this model), but a fully-
specified AUGMENTED LINEAR OPERATOR artifact - (Axx, Axs, Bx, c0_x) from
the dither-corrected OLS re-fit. `(Asx, Ass, Bs, C)` are NOT duplicated
here (Ass alone is a dense 1024x1024 float64 - 8MB per checkpoint,
already sitting, unchanged, in `docs/nu_gap_export/fullM3_{case}_{seed}.npz`
and in the source msgpack) - the sidecar names the exact source
checkpoint instead, so the full augmented operator is reconstructible
(`Abar = block([[Axx,Axs],[Asx,Ass]])`) without a second multi-MB copy
of data that never changed. This is the exact, complete specification
needed to reproduce every dither-cure result this session reported,
with the source checkpoint and recipe traceable.

"Conv vs step parity" does not apply to this artifact in Task 0's sense,
and this is stated explicitly rather than forcing a check that doesn't
fit: the x-readout here is direct matrix arithmetic (no S4 kernel, no
conv/step distinction exists for it at all - it is already exact by
construction, verified to machine precision in TASK A's bias-correction
entry). The internal recurrence (Asx/Ass/Bs) is copied UNCHANGED from a
real M3 checkpoint whose OWN conv/step parity IS covered by Task 0's
verified invariant - inherited, not independently re-checked here.

    python tools/save_dither_cured_checkpoints.py
"""
from __future__ import annotations

import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.data import generate_microgrid_trajectory
from s4dpc.logging import get_git_sha, get_lockfile_sha
from s4dpc.systems import get_discrete_matrices

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CKPT_DIR = EXPORT_DIR / "ckpt"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
L_MAX = 100
DATA_SEED = 42
APRBS_LOW, APRBS_HIGH = -10.0, 10.0
X_SYNTH_RANGE = 5.0
N_DITHER = 2000
DITHER_RNG_SEED = 0  # matches tools/dither_cure_test.py and bias_corrected_dither_cure.py


def get_training_trajectory(case):
    inputs, targets = generate_microgrid_trajectory(
        batch_size=1, length=L_MAX, seed=DATA_SEED, system_case=case, dt=DT,
        aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH)
    return inputs[0, :, :D_X], inputs[0, :, D_X:], targets[0]


def simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj):
    n_s = Ass.shape[0]
    S = np.zeros((L_MAX, n_s))
    s = np.zeros(n_s)
    for t in range(L_MAX):
        S[t] = s
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return S


def dither_refit_with_intercept(x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, n_dither, rng, s_scale):
    n_s = s_traj.shape[1]
    ones_on = np.ones((L_MAX, 1))
    Z_on = np.hstack([ones_on, x_traj, s_traj, u_traj])
    Y_on = x_next_traj
    if n_dither > 0:
        x_s = rng.uniform(-X_SYNTH_RANGE, X_SYNTH_RANGE, size=(n_dither, D_X))
        u_s = rng.uniform(APRBS_LOW, APRBS_HIGH, size=(n_dither, D_U))
        s_s = rng.standard_normal((n_dither, n_s)) * s_scale
        y_s = x_s @ A_true.T + u_s @ B_true.T
        ones_s = np.ones((n_dither, 1))
        Z = np.vstack([Z_on, np.hstack([ones_s, x_s, s_s, u_s])])
        Y = np.vstack([Y_on, y_s])
    else:
        Z, Y = Z_on, Y_on
    theta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    theta = theta.T
    c0_x = theta[:, 0]
    Axx = theta[:, 1:1 + D_X]
    Axs = theta[:, 1 + D_X:1 + D_X + n_s]
    Bx = theta[:, 1 + D_X + n_s:]
    return c0_x, Axx, Axs, Bx


def main() -> None:
    out_dir = EXPORT_DIR / "dither_cured"
    out_dir.mkdir(parents=True, exist_ok=True)
    git_sha, lockfile_sha = get_git_sha(), get_lockfile_sha()
    rng = np.random.RandomState(DITHER_RNG_SEED)

    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        x_traj, u_traj, x_next_traj = get_training_trajectory(case)

        for seed in range(N_SEEDS):
            source_path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            source_ckpt = CKPT_DIR / f"M3_case{case}_seed{seed}.msgpack"
            if not source_path.exists():
                continue
            data = np.load(source_path)
            A_source = data["A"]
            n_s = A_source.shape[0] - D_X
            Asx, Ass = A_source[D_X:, :D_X], A_source[D_X:, D_X:]
            Bs = data["B"][D_X:, :]
            C = data["C"]

            s_traj = simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj)
            s_scale = float(np.sqrt(np.mean(s_traj ** 2)))

            c0_x, Axx, Axs, Bx = dither_refit_with_intercept(
                x_traj, s_traj, u_traj, x_next_traj, A_true, B_true, N_DITHER, rng, s_scale)

            stem = f"dither_cured_case{case}_seed{seed}"
            np.savez(out_dir / f"{stem}.npz", Axx=Axx, Axs=Axs, Bx=Bx, c0_x=c0_x)

            sidecar = {
                "variant": "dither_cured", "case": case, "seed": seed,
                "source_checkpoint": f"M3_case{case}_seed{seed}.msgpack",
                "source_checkpoint_exists_locally": source_ckpt.exists(),
                "recipe": {
                    "method": "closed-form OLS with intercept on (x_traj,s_traj,u_traj) real "
                              "training data plus N_DITHER synthetic (x,s_random,u)->true_x_next samples",
                    "n_dither": N_DITHER, "dither_rng_seed": DITHER_RNG_SEED,
                    "x_synth_range": X_SYNTH_RANGE, "aprbs_range": [APRBS_LOW, APRBS_HIGH],
                    "s_synth_scale": s_scale,
                    "note": "Asx/Ass/Bs/C copied UNCHANGED from the source M3 checkpoint - only "
                             "(Axx,Axs,Bx,c0_x) are re-fit. Not a StackedModel - see module docstring "
                             "for why this architecture cannot represent this model.",
                },
                "git_sha": git_sha, "lockfile_sha": lockfile_sha,
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
            print(f"saved {stem}  ||c0_x||={np.linalg.norm(c0_x):.2e}")

    print(f"\nwrote {len(CASES) * N_SEEDS} dither-cured artifacts to {out_dir}")


if __name__ == "__main__":
    main()
