"""Round 1.5, Tasks 1 & 2: is round 1's "arm_2 trains away" finding
actually branch suppression (gamma->0 / W_dec learns to annihilate the
branch subspace), as LayerNorm's degree-0 homogeneity structurally
predicts? And does arm_5's seed sensitivity correlate with the same
branch-on/branch-off basin selection?

Task 1(a-c) reads the EXISTING 8 trained arm_2 checkpoints from round 1's
multi-seed run (results/ckpt/arm_2_seed*.msgpack) - no retraining.
Task 1(d) is the decisive test: retrains arm_2 with gamma/beta FROZEN at
init (branch_suppression.train_arm_frozen_ln) - LN's scaling is still
fully active every forward pass, but the optimizer cannot suppress it by
shrinking gamma or shifting beta. Pinned at 60000 epochs (the same
budget round 1 settled on for arm_0's positive control) per this round's
own methodological note: do not re-tune the epoch budget again here.
Task 2 reuses Task 1(b)'s branch-to-skip ratio machinery on arm_5's 8
EXISTING checkpoints (also no retraining) and correlates it against
arm_5's already-recorded jx_err_mean from results/exp2_ladder.csv.

Run: python -m layernorm_study.experiments.round1_5_branch_suppression
"""
from __future__ import annotations

import csv
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import flax.serialization as serialization
from flax import nnx

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layernorm_study.src.arms import ARMS, build_model, load_arm_model
from layernorm_study.src.branch_suppression import (
    branch_to_skip_ratio_along_trajectory,
    gamma_beta_norms,
    jacobian_decomposition_along_trajectory,
    train_arm_frozen_ln,
)
from layernorm_study.src.scalar_diagnostics import jx_ju_along_trajectory
from layernorm_study.src.scalar_system import A_TRUE, B_TRUE, generate_scalar_trajectory
from layernorm_study.experiments.exp2_train_ladder import D_MODEL, N, L_MAX, EPOCHS, LEARNING_RATE

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
CKPT_DIR = RESULTS_DIR / "ckpt"
SEEDS = list(range(8))
AB_TRUE = np.array([[A_TRUE, B_TRUE]])


def load_trained(arm_name: str, seed: int, decode: bool = True):
    arm = ARMS[arm_name]
    key = jax.random.PRNGKey(seed)
    model = build_model(arm, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, decode=decode, key=key)
    pure_dict = serialization.msgpack_restore((CKPT_DIR / f"{arm_name}_seed{seed}.msgpack").read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)
    return model


def task1_a_gamma_beta(inputs, targets) -> list[dict]:
    rows = []
    for seed in SEEDS:
        init_model = build_model(
            ARMS["arm_2"], d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1,
            decode=False, key=jax.random.PRNGKey(seed),
        )
        gamma_init, beta_init = gamma_beta_norms(init_model)
        final_model = load_trained("arm_2", seed, decode=True)
        gamma_final, beta_final = gamma_beta_norms(final_model)
        rows.append({
            "seed": seed, "gamma_init": gamma_init, "beta_init": beta_init,
            "gamma_final": gamma_final, "beta_final": beta_final,
            "gamma_shrink_factor": gamma_init / gamma_final if gamma_final > 0 else float("inf"),
        })
    return rows


def task1_bc_branch_and_decomposition(arm_name: str, inputs, targets) -> list[dict]:
    rows = []
    for seed in SEEDS:
        model = load_trained(arm_name, seed, decode=True)
        ratio_rows = branch_to_skip_ratio_along_trajectory(model, jnp.asarray(inputs), jnp.asarray(targets))
        decomp_rows = jacobian_decomposition_along_trajectory(model, jnp.asarray(inputs), jnp.asarray(targets), AB_TRUE)
        ratios = np.array([r["ratio"] for r in ratio_rows])
        r_norms = np.array([r["R_norm"] for r in decomp_rows])
        rows.append({
            "arm": arm_name, "seed": seed,
            "branch_skip_ratio_median": float(np.median(ratios)),
            "branch_skip_ratio_mean": float(np.mean(ratios)),
            "C_err_rel": decomp_rows[0]["C_err_rel"],  # constant across t for a given checkpoint
            "R_norm_median": float(np.median(r_norms)),
            "R_norm_max": float(np.max(r_norms)),
        })
    return rows


def task1_d_frozen_ln(inputs, targets) -> list[dict]:
    arm = ARMS["arm_2"]
    inputs_j, targets_j = jnp.asarray(inputs), jnp.asarray(targets)
    rows = []
    for seed in SEEDS:
        param_state, train_mse = train_arm_frozen_ln(
            arm, inputs_j, targets_j, d_model=D_MODEL, N=N, l_max=L_MAX,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        model = build_model(arm, d_model=D_MODEL, N=N, l_max=L_MAX, d_input=2, d_output=1, decode=True, key=jax.random.PRNGKey(seed))
        nnx.update(model, param_state)

        traj = jx_ju_along_trajectory(model, inputs_j, targets_j)
        jx_errs = [abs(r["Jx"] - A_TRUE) / abs(A_TRUE) for r in traj]
        gamma, beta = gamma_beta_norms(model)
        row = {
            "seed": seed, "teacher_mse_train_final": train_mse,
            "jx_err_mean": float(np.mean(jx_errs)), "jx_err_max": float(np.max(jx_errs)),
            "gamma_norm": gamma, "beta_norm": beta,
        }
        rows.append(row)
        print(f"[arm_2_frozen_ln seed={seed}] train_mse={train_mse:.3e} jx_err_mean={row['jx_err_mean']:.3e} "
              f"jx_err_max={row['jx_err_max']:.3e} gamma_norm={gamma:.4f}", flush=True)
    return rows


def task2_arm5_correlation(inputs, targets) -> tuple[float, pd.DataFrame]:
    branch_rows = task1_bc_branch_and_decomposition("arm_5", inputs, targets)
    branch_df = pd.DataFrame(branch_rows)

    ladder_df = pd.read_csv(RESULTS_DIR / "exp2_ladder.csv")
    arm5 = ladder_df[ladder_df["arm"] == "arm_5"][["seed", "jx_err_mean"]]

    merged = branch_df.merge(arm5, on="seed")
    corr = float(np.corrcoef(merged["branch_skip_ratio_median"], merged["jx_err_mean"])[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["branch_skip_ratio_median"], merged["jx_err_mean"])
    for _, r in merged.iterrows():
        ax.annotate(str(int(r["seed"])), (r["branch_skip_ratio_median"], r["jx_err_mean"]))
    ax.set_xlabel("median branch/skip ratio along trajectory")
    ax.set_ylabel("jx_err_mean")
    ax.set_yscale("log")
    ax.set_title(f"arm_5: branch/skip ratio vs Jacobian error (r={corr:.3f})")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "round1_5_arm5_branch_ratio_vs_error.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    return corr, merged


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, targets = generate_scalar_trajectory(length=L_MAX, seed=42)

    print("=== Task 1(a): gamma/beta init vs final, arm_2 ===")
    ga_rows = task1_a_gamma_beta(inputs, targets)
    for r in ga_rows:
        print(f"  seed={r['seed']}: gamma {r['gamma_init']:.4f} -> {r['gamma_final']:.4f} "
              f"(shrink {r['gamma_shrink_factor']:.2f}x)  beta {r['beta_init']:.4f} -> {r['beta_final']:.4f}")
    pd.DataFrame(ga_rows).to_csv(RESULTS_DIR / "round1_5_arm2_gamma_beta.csv", index=False)

    print("\n=== Task 1(b,c): branch/skip ratio + J decomposition, arm_2 ===")
    bc_rows = task1_bc_branch_and_decomposition("arm_2", inputs, targets)
    for r in bc_rows:
        print(f"  seed={r['seed']}: branch/skip median={r['branch_skip_ratio_median']:.3e} "
              f"C_err_rel={r['C_err_rel']:.3e} R_norm_median={r['R_norm_median']:.3e}")
    pd.DataFrame(bc_rows).to_csv(RESULTS_DIR / "round1_5_arm2_branch_decomposition.csv", index=False)

    print("\n=== Task 1(d): arm_2 with gamma/beta FROZEN at init, 60000 epochs, 8 seeds ===")
    frozen_rows = task1_d_frozen_ln(inputs, targets)
    frozen_df = pd.DataFrame(frozen_rows)
    frozen_df.to_csv(RESULTS_DIR / "round1_5_arm2_frozen_ln.csv", index=False)

    unfrozen_df = pd.read_csv(RESULTS_DIR / "exp2_ladder.csv")
    unfrozen_arm2 = unfrozen_df[unfrozen_df["arm"] == "arm_2"][["seed", "jx_err_mean"]].rename(
        columns={"jx_err_mean": "jx_err_mean_unfrozen"}
    )
    compare = frozen_df.merge(unfrozen_arm2, on="seed")
    compare["jx_err_mean_frozen"] = compare["jx_err_mean"]
    compare_path = RESULTS_DIR / "round1_5_arm2_frozen_vs_unfrozen.csv"
    compare[["seed", "jx_err_mean_unfrozen", "jx_err_mean_frozen", "gamma_norm", "beta_norm"]].to_csv(compare_path, index=False)
    print(f"\nwrote {compare_path}")
    print(compare[["seed", "jx_err_mean_unfrozen", "jx_err_mean_frozen"]].to_string(index=False))

    print("\n=== Task 2: arm_5 branch/skip ratio vs Jacobian error correlation ===")
    corr, merged = task2_arm5_correlation(inputs, targets)
    merged.to_csv(RESULTS_DIR / "round1_5_arm5_branch_ratio_vs_error.csv", index=False)
    print(f"Pearson r = {corr:.4f}")
    print(merged[["seed", "branch_skip_ratio_median", "jx_err_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
