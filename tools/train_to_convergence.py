"""Train M3 and M6 on case 3 (float64, wd=0) until the loss curve is
VISIBLY FLAT, not to a fixed step budget - checked every 5000 steps, up
to 200k. Answers "what does 'trained' even mean here" before running any
diagnostic on the result (docs/DECISIONS.md).

Flatness criterion: compare the mean loss over each 5000-step chunk
against the previous chunk, in log10 space (these losses span many
orders of magnitude, so a fixed absolute tolerance is meaningless) - if
FLATNESS_CONSECUTIVE consecutive chunk-to-chunk log10 differences are all
below FLATNESS_LOG10_TOL, declare flat and stop. The full per-step curve
is always logged and plotted regardless of when/whether this triggers,
so the curve is inspectable even if the criterion is wrong or the run
hits the 200k cap.

@nnx.jit on the training step - identify.py's existing loops don't jit
(flagged as a known gap in docs/DECISIONS.md's controller-smoke-test
entry: "Add jit back before any larger controller run"). At up to 200k
steps x 2 variants, eager dispatch would make this run take hours;
jitted, it should not.

Saves trained params to ./ckpt_trained/{variant}_case3.msgpack (relative
to CWD - consumed by tools/diagnose_trained_variants.py in the SAME
Kaggle kernel run, not committed - *.msgpack is gitignored except the
one parity fixture, CLAUDE.md sec 2/5) plus a loss-curve CSV/PNG under
docs/ (committable).

    python tools/train_to_convergence.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
import flax.serialization as serialization

from s4dpc import diagnostics
from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0
CHECK_EVERY = 5000
MAX_STEPS = 200_000
FLATNESS_LOG10_TOL = 0.05
FLATNESS_CONSECUTIVE = 2
FREE_RUN_STEPS = 100
FREE_RUN_EVAL_SEED = 777  # different from DATA_SEED=42 (identify.py) - held-out eval sequence
VARIANTS_TO_TRAIN = ["M3", "M6"]

DOCS_DIR = _REPO_ROOT / "docs"
CKPT_DIR = pathlib.Path("ckpt_trained")


def _cast_all_params(model: StackedModel) -> None:
    def _cast_leaf(x: jax.Array) -> jax.Array:
        target = jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64
        return x.astype(target)

    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(_cast_leaf, state)
    nnx.update(model, state)


def _build(variant: str, decode: bool, key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=decode, rngs=nnx.Rngs(params=key),
    )
    _cast_all_params(model)
    return model


def _save_checkpoint(model: StackedModel, variant: str) -> pathlib.Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{variant}_case{CASE}.msgpack"
    pure_dict = nnx.state(model, nnx.Param).to_pure_dict()
    path.write_bytes(serialization.msgpack_serialize(pure_dict))
    return path


def _free_run_rmse(model_decode_true: StackedModel) -> float:
    """Feeds the model's OWN prediction back as the next step's state
    input (decode=True, stepped - NOT teacher-forced), compared against
    generate_microgrid_trajectory's own recursive TRUE-plant rollout
    (its `targets` ARE the true autonomous trajectory - X_next feeds back
    as X_current inside that function already, s4dpc/data.py - reused
    directly rather than re-deriving a parallel true-plant computation)."""
    inputs, targets = generate_microgrid_trajectory(
        batch_size=1, length=FREE_RUN_STEPS, seed=FREE_RUN_EVAL_SEED, system_case=CASE, dt=0.01,
        aprbs_low=-10.0, aprbs_high=10.0,
    )
    inputs64 = jnp.asarray(inputs[0], dtype=jnp.float64)  # (100, 9)
    true_traj = jnp.asarray(targets[0], dtype=jnp.float64)  # (100, 6) = x_1..x_100

    x0 = inputs64[0, :D_OUTPUT]
    u_seq = inputs64[:, D_OUTPUT:]

    states = diagnostics.zero_states(model_decode_true)
    x = x0
    preds = []
    for k in range(FREE_RUN_STEPS):
        x, states = diagnostics.step(model_decode_true, x, u_seq[k], states)
        preds.append(x)
    preds = jnp.stack(preds)

    return float(jnp.sqrt(jnp.mean((preds - true_traj) ** 2)))


def _train_one_variant(variant: str, inputs: jax.Array, targets: jax.Array) -> dict:
    print(f"\n=== {variant} ===")
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = _build(variant, decode=False, key=key)
    optimizer = nnx.Optimizer(model, optax.adamw(LR, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)
    states0 = model.init_state()

    pred_check, _ = model(inputs, states0)
    if pred_check.dtype != jnp.float64:
        raise RuntimeError(f"[{variant}] forward pass is not float64 (got {pred_check.dtype})")

    @nnx.jit
    def train_step(m, opt, inp, tgt, st):
        def loss_fn(mm):
            pred, _ = mm(inp, st)
            return jnp.mean((pred - tgt) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    all_losses: list[float] = []
    checkpoint_means: list[float] = []
    step = 0
    stopped_reason = "hit MAX_STEPS"
    while step < MAX_STEPS:
        chunk_losses = []
        for _ in range(CHECK_EVERY):
            loss = train_step(model, optimizer, inputs, targets, states0)
            loss_v = float(loss)
            chunk_losses.append(loss_v)
            all_losses.append(loss_v)
            step += 1
            if not np.isfinite(loss_v):
                stopped_reason = f"non-finite loss at step {step}"
                print(f"  [{variant}] {stopped_reason}")
                step = MAX_STEPS  # break outer loop too
                break
        chunk_mean = float(np.mean(chunk_losses))
        checkpoint_means.append(chunk_mean)
        print(f"  [{variant}] step {step:7d}  chunk_mean_mse={chunk_mean:.6e}  "
              f"instantaneous={chunk_losses[-1]:.6e}")
        if len(checkpoint_means) >= FLATNESS_CONSECUTIVE + 1:
            recent = checkpoint_means[-(FLATNESS_CONSECUTIVE + 1):]
            log_diffs = [abs(np.log10(recent[i + 1]) - np.log10(recent[i])) for i in range(len(recent) - 1)]
            if all(d < FLATNESS_LOG10_TOL for d in log_diffs):
                stopped_reason = f"flat (log10 diffs {[f'{d:.4f}' for d in log_diffs]} < {FLATNESS_LOG10_TOL})"
                print(f"  [{variant}] FLAT after {step} steps: {stopped_reason}")
                break

    final_mse = float(np.mean(all_losses[-1000:])) if len(all_losses) >= 1000 else all_losses[-1]
    final_instantaneous = all_losses[-1]

    ckpt_path = _save_checkpoint(model, variant)
    print(f"  [{variant}] saved checkpoint to {ckpt_path}")

    # free-run RMSE needs decode=True with the trained params loaded in
    model_rnn = _build(variant, decode=True, key=key)
    trained_state = nnx.state(model, nnx.Param)
    rnn_state = nnx.state(model_rnn, nnx.Param)
    rnn_state.replace_by_pure_dict(trained_state.to_pure_dict())
    nnx.update(model_rnn, rnn_state)
    free_run_rmse = _free_run_rmse(model_rnn)

    return {
        "variant": variant,
        "steps_trained": step,
        "stopped_reason": stopped_reason,
        "final_mse_windowed": final_mse,
        "final_mse_instantaneous": final_instantaneous,
        "all_losses": all_losses,
        "checkpoint_means": checkpoint_means,
        "free_run_rmse": free_run_rmse,
        "ckpt_path": str(ckpt_path),
    }


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    print(f"case_data dtype: inputs={inputs.dtype}, targets={targets.dtype}")
    mean_target_sq = float(jnp.mean(targets**2))

    ab_hat, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    print(f"LS floor (recomputed fresh, float64): mse={ls_mse:.6e}  nmse={ls_mse / mean_target_sq:.6e}")
    print(f"  (cross-check against the 5.46e-15 figure quoted from the 2026-08-07 entry: "
          f"ratio={ls_mse / 5.46e-15:.4f})")

    results = []
    fig, axes = plt.subplots(len(VARIANTS_TO_TRAIN), 1, figsize=(9, 4 * len(VARIANTS_TO_TRAIN)))
    if len(VARIANTS_TO_TRAIN) == 1:
        axes = [axes]

    for i, variant in enumerate(VARIANTS_TO_TRAIN):
        res = _train_one_variant(variant, inputs, targets)
        nmse = res["final_mse_windowed"] / mean_target_sq
        ratio_to_floor = res["final_mse_windowed"] / ls_mse
        print(f"\n  [{variant}] SUMMARY: steps={res['steps_trained']}  stopped=({res['stopped_reason']})")
        print(f"    teacher_mse (last-1000-step mean) = {res['final_mse_windowed']:.6e}")
        print(f"    teacher_mse (instantaneous, final step) = {res['final_mse_instantaneous']:.6e}")
        print(f"    nmse = {nmse:.6e}")
        print(f"    ratio to LS floor ({ls_mse:.3e}) = {ratio_to_floor:.6e}")
        print(f"    free-run RMSE over {FREE_RUN_STEPS} steps (recursive) = {res['free_run_rmse']:.6e}")

        results.append({
            "variant": variant, "steps_trained": res["steps_trained"],
            "stopped_reason": res["stopped_reason"],
            "teacher_mse_windowed": res["final_mse_windowed"],
            "teacher_mse_instantaneous": res["final_mse_instantaneous"],
            "nmse": nmse, "ratio_to_ls_floor": ratio_to_floor,
            "free_run_rmse": res["free_run_rmse"],
        })

        ax = axes[i]
        steps_arr = np.arange(len(res["all_losses"]))
        finite = np.isfinite(res["all_losses"]) & (np.array(res["all_losses"]) > 0)
        ax.semilogy(steps_arr[finite], np.array(res["all_losses"])[finite], linewidth=0.5)
        ax.axhline(ls_mse, color="red", linestyle="--", label=f"LS floor ({ls_mse:.1e})")
        ax.set_title(f"{variant}, case {CASE}: teacher-forced MSE ({res['steps_trained']} steps, {res['stopped_reason']})")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE (log)")
        ax.legend()

    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "train_to_convergence_curves.png", dpi=110)
    print(f"\nwrote {DOCS_DIR / 'train_to_convergence_curves.png'}")

    header = list(results[0].keys())
    lines = [",".join(header)]
    for row in results:
        lines.append(",".join(str(row[h]) for h in header))
    (DOCS_DIR / "train_to_convergence_summary.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'train_to_convergence_summary.csv'}")

    print("\n=== FINAL SUMMARY ===")
    print(f"{'variant':8s} {'steps':>8s} {'teacher_mse':>13s} {'nmse':>13s} {'ratio_LS':>12s} {'free_run_rmse':>14s}")
    for row in results:
        print(f"{row['variant']:8s} {row['steps_trained']:8d} {row['teacher_mse_windowed']:13.4e} "
              f"{row['nmse']:13.4e} {row['ratio_to_ls_floor']:12.4e} {row['free_run_rmse']:14.4e}")


if __name__ == "__main__":
    main()
