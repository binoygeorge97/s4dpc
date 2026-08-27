"""EXPERIMENT 2: complexity ladder. Trains every arm in
layernorm_study.src.arms.ARMS on the SAME scalar-plant data/d_model/N/
l_max/optimizer/epochs (varying only the block config and, across the
--seeds list, the init/training seed), then runs the full diagnostics
suite on each (arm, seed) and writes one CSV row + one JSON manifest per
run.

arm_0 is a POSITIVE CONTROL: it MUST recover Jx=3.000, Ju=1.000 at
every point tested, for EVERY seed (it's an exactly affine model by
construction - the only question is whether gradient descent actually
converges). This script halts (raises) if arm_0 fails that check on any
seed, per the task's own instruction, before treating any other arm's
results as meaningful.

Run: python -m layernorm_study.experiments.exp2_train_ladder [--arms arm_0,arm_2,...] [--seeds 0,1,2,...]
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op (CLAUDE.md convention)

import jax.numpy as jnp
import numpy as np
import flax.serialization as serialization

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layernorm_study.src.arms import ARMS, load_arm_model, train_arm
from layernorm_study.src.scalar_diagnostics import (
    directional_asymmetry,
    equilibrium_drift,
    free_run_rmse,
    homogeneity_sweep,
    jx_ju_along_trajectory,
    jx_vs_c_sweep,
    prenorm_sigma,
    teacher_forced_mse,
)
from layernorm_study.src.scalar_system import A_TRUE, B_TRUE, K_STAB, generate_scalar_trajectory

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
CKPT_DIR = RESULTS_DIR / "ckpt"

D_MODEL = 8
N = 16
L_MAX = 100
EPOCHS = 60000  # 20000 was insufficient for some seeds even on arm_0 (the
# simplest possible arm, an exactly affine 2-parameter fit): seed=6 hit
# jx_err_max=1.8e-3 at 20000 epochs but converged to ~3e-9 at 60000 -
# confirmed directly (not assumed) to be a convergence-budget issue, not
# a real failure - see NOTES.md's multi-seed round for the full story.
LEARNING_RATE = 1e-3

C_VALUES = np.concatenate([-np.logspace(3, -6, 40), np.logspace(-6, 3, 40)])
EPS_VALUES = np.logspace(-6, -1, 20)
ARM_0_JX_JU_TOLERANCE = 1e-6  # "must recover Jx=3.000, Ju=1.000 at every point" - hard gate

FIELDNAMES = [
    "arm", "seed", "description", "teacher_mse_train_final", "teacher_mse", "free_run_rmse",
    "jx_err_mean", "jx_err_max", "ju_err_mean", "ju_err_max", "equilibrium_drift",
    "homogeneity_flatness", "jx_far_from_origin", "jx_at_origin", "jx_origin_spike_ratio",
    "directional_asymmetry_at_smallest_eps", "directional_asymmetry_at_largest_eps",
    "sigma_jx_err_correlation",
]


def _stringify_keys(x):
    """Same convention as s4dpc.identify._stringify_keys - msgpack
    requires string dict keys, nnx.State's pure_dict uses ints for list
    indices (e.g. layers[0])."""
    if isinstance(x, dict):
        return {str(k): _stringify_keys(v) for k, v in x.items()}
    return x


def save_arm_checkpoint(arm_name: str, seed: int, param_state) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pure_dict = _stringify_keys(param_state.to_pure_dict())
    (CKPT_DIR / f"{arm_name}_seed{seed}.msgpack").write_bytes(serialization.msgpack_serialize(pure_dict))


def run_one_arm(arm_name: str, seed: int, inputs: np.ndarray, targets: np.ndarray, *, save_plot: bool) -> dict:
    arm = ARMS[arm_name]

    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    param_state, teacher_mse_final_train = train_arm(
        arm, inputs_j, targets_j,
        d_model=D_MODEL, N=N, l_max=L_MAX, epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
    )
    save_arm_checkpoint(arm_name, seed, param_state)
    model = load_arm_model(
        arm, param_state, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, seed=seed, decode=True,
    )

    teacher_mse = teacher_forced_mse(model, inputs_j, targets_j)
    run_rmse = free_run_rmse(model, inputs_j, targets_j, K_STAB)
    traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
    jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
    ju_errs = [abs(r["Ju"] - B_TRUE) / abs(B_TRUE) for r in traj]

    homog = homogeneity_sweep(model, np.array([1.0, 1.0]), C_VALUES)
    jx_sweep = jx_vs_c_sweep(model, np.array([1.0]), C_VALUES)
    asym = directional_asymmetry(model, EPS_VALUES, jax.random.PRNGKey(seed + 1000))
    drift = equilibrium_drift(model)

    sigmas = [prenorm_sigma(model, np.array([r["x"], 0.0])) for r in traj]
    has_prenorm_sigma = sigmas[0] is not None
    sigma_jx_corr = None
    if has_prenorm_sigma:
        sig_arr = np.array(sigmas)
        jx_err_arr = np.array(jx_errs)
        if np.std(sig_arr) > 0 and np.std(jx_err_arr) > 0:
            sigma_jx_corr = float(np.corrcoef(sig_arr, jx_err_arr)[0, 1])

    result = {
        "arm": arm_name,
        "seed": seed,
        "description": arm.description,
        "teacher_mse_train_final": teacher_mse_final_train,
        "teacher_mse": teacher_mse,
        "free_run_rmse": run_rmse,
        "jx_err_mean": float(np.mean(jx_errs)),
        "jx_err_max": float(np.max(jx_errs)),
        "ju_err_mean": float(np.mean(ju_errs)),
        "ju_err_max": float(np.max(ju_errs)),
        "equilibrium_drift": drift,
        "homogeneity_flatness": float(np.std([r["F_over_c"] for r in homog])),
        "jx_far_from_origin": jx_sweep[0]["Jx"],
        "jx_at_origin": jx_sweep[len(jx_sweep) // 2]["Jx"],
        "jx_origin_spike_ratio": abs(jx_sweep[len(jx_sweep) // 2]["Jx"]) / max(abs(jx_sweep[0]["Jx"]), 1e-300),
        "directional_asymmetry_at_smallest_eps": asym[0]["mean_abs_diff"],
        "directional_asymmetry_at_largest_eps": asym[-1]["mean_abs_diff"],
        "sigma_jx_err_correlation": sigma_jx_corr,
    }

    manifest = {
        "arm": arm_name,
        "seed": seed,
        "description": arm.description,
        "block_kwargs": arm.block_kwargs,
        "n_layers": arm.n_layers,
        "config": {"d_model": D_MODEL, "N": N, "l_max": L_MAX, "epochs": EPOCHS,
                   "learning_rate": LEARNING_RATE, "seed": seed},
        "true_plant": {"A": A_TRUE, "B": B_TRUE},
        "result": result,
    }
    (RESULTS_DIR / f"exp2_{arm_name}_seed{seed}_manifest.json").write_text(json.dumps(manifest, indent=2))

    if save_plot:
        plot_arm(arm_name, seed, homog, jx_sweep, traj)

    if arm_name == "arm_0":
        if result["jx_err_max"] > ARM_0_JX_JU_TOLERANCE or result["ju_err_max"] > ARM_0_JX_JU_TOLERANCE:
            raise SystemExit(
                f"ARM_0 POSITIVE CONTROL FAILED at seed={seed}: jx_err_max={result['jx_err_max']:.3e} "
                f"ju_err_max={result['ju_err_max']:.3e} (tolerance {ARM_0_JX_JU_TOLERANCE:.0e}). "
                "STOP - do not trust any other arm's results until this is understood."
            )

    print(
        f"[{arm_name} seed={seed}] teacher_mse={teacher_mse:.3e} free_run_rmse={run_rmse:.3e} "
        f"jx_err_mean={result['jx_err_mean']:.3e} jx_origin_spike_ratio={result['jx_origin_spike_ratio']:.3g} "
        f"equilibrium_drift={drift:.3e}",
        flush=True,
    )
    return result


def plot_arm(arm_name: str, seed: int, homog: list[dict], jx_sweep: list[dict], traj: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    c = [r["c"] for r in homog]
    f_over_c = [r["F_over_c"] for r in homog]
    axes[0].plot(c, f_over_c, color="black", linewidth=1)
    axes[0].set_xscale("symlog", linthresh=1e-6)
    axes[0].set_title(f"{arm_name} seed{seed}: homogeneity sweep (F(c*z0)-F(0,0))/c")
    axes[0].set_xlabel("c (symlog)")
    axes[0].axhline(A_TRUE * 1.0 + B_TRUE * 1.0, color="tab:red", linestyle="--", label="true (flat)")
    axes[0].legend()

    c2 = [r["c"] for r in jx_sweep]
    jx = [abs(r["Jx"]) for r in jx_sweep]
    axes[1].plot(c2, jx, color="black", linewidth=1)
    axes[1].set_xscale("symlog", linthresh=1e-6)
    axes[1].set_yscale("log")
    axes[1].set_title(f"{arm_name} seed{seed}: |Jx(c)| vs c")
    axes[1].axhline(A_TRUE, color="tab:red", linestyle="--", label="true Jx=3")
    axes[1].legend()

    t = [r["t"] for r in traj]
    jx_t = [r["Jx"] for r in traj]
    axes[2].plot(t, jx_t, color="black", linewidth=1)
    axes[2].axhline(A_TRUE, color="tab:red", linestyle="--", label="true Jx=3")
    axes[2].set_title(f"{arm_name} seed{seed}: Jx along real trajectory")
    axes[2].set_xlabel("timestep")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"exp2_{arm_name}_seed{seed}_diagnostics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=str, default=",".join(ARMS.keys()))
    parser.add_argument("--seeds", type=str, default="0")
    args = parser.parse_args()
    arm_names = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_scalar_trajectory(length=L_MAX, seed=42)

    csv_path = RESULTS_DIR / "exp2_ladder.csv"
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a" if not write_header else "w", newline="")
    writer = csv_module.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    for arm_name in arm_names:
        print(f"=== [{arm_name}] {ARMS[arm_name].description} ===", flush=True)
        for i, seed in enumerate(seeds):
            result = run_one_arm(arm_name, seed, inputs, targets, save_plot=(i == 0))
            writer.writerow(result)
            csv_file.flush()

    csv_file.close()
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
