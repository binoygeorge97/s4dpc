"""EXP 4 float64 re-check (docs/DECISIONS.md): EXP 4(b) reported that
D-only initialized EXACTLY at the least-squares optimum (mse 2.86e-14)
DIVERGED ~8 orders of magnitude under 2000 Adam steps, landing at
1.89e-6. That was measured in float32, where 2.86e-14 is below
representable relative precision (~1e-7 of the operand magnitudes
involved) - the "divergence" could be float32 rounding noise around an
exact flat direction rather than a real instability.

Re-runs the SAME D-only/LS-init construction (tools/diagnose_m3_reparam.py,
verbatim: zero D and out-projection, route the LS solution through the
encoder/decoder identity-ish paths so the D-only forward pass computes
inputs @ w_ls EXACTLY at init) with `jax.config.update("jax_enable_x64",
True)` set BEFORE any other JAX import/op - matching
tools/diagnose_m3_rank_x64.py's pattern, INCLUDING the same workaround
that script needed: flax.nnx.Linear/LayerNorm hardcode
`param_dtype=float32` regardless of the global x64 flag, so the model's
OWN parameters do not become float64 just because the flag is set. The
7 D-only-relevant leaves (encoder kernel/bias, D, out kernel/bias,
decoder kernel/bias) are constructed directly as float64 arrays by the
LS-init routine itself (dtype=jnp.float64 hardcoded in the zeros/asarray/
eye calls, not inherited from the pre-existing float32 leaf) - the
`nnx.Optimizer` is built AFTER this, so its momentum/variance state
inherits float64 from the params it is initialized from.

Three variants, D-only, case 3, LS-init, 2000 steps, mse logged EVERY
step (pre-update, so step 0 == the LS floor by construction):
  a. Adam lr=1e-3, wd=0        (the exact EXP4(b) setup, float64 now)
  b. plain SGD, lr=1e-3        (same nominal step size as (a), no
                                 momentum/adaptive scaling - isolates
                                 whether Adam's own dynamics are what
                                 leaves the optimum)
  c. Adam lr=1e-5, wd=0        (100x smaller step - does a slower Adam
                                 stay near the optimum?)

Self-check before trusting anything: step-0 loss for each variant must
match the float64 LS floor (fit_least_squares, already float64
regardless of the JAX flag - pure numpy) to near machine precision.

    python tools/diagnose_m3_exp4_x64.py
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

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
SEED = 0

OUT_CSV = _REPO_ROOT / "docs" / "m3_exp4_x64_recheck.csv"
OUT_PNG = _REPO_ROOT / "docs" / "m3_exp4_x64_recheck.png"

VARIANT_SPECS = {
    "adam_lr1e-3": optax.adamw(1e-3, weight_decay=0.0),
    "sgd_lr1e-3": optax.sgd(1e-3),
    "adam_lr1e-5": optax.adamw(1e-5, weight_decay=0.0),
}


def _d_only_forward(model: StackedModel, inputs: jax.Array) -> jax.Array:
    """Verbatim copy of diagnose_m3_reparam.py's _d_only_forward - the
    exact function EXP4 trained, reused here rather than re-derived."""
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x
    d_vec = block.seq.D.value.squeeze(-1)
    x = x * d_vec[None, :]
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def _build_model(key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    return StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )


def _apply_ls_init_x64(model: StackedModel, w_ls_in_out: np.ndarray) -> None:
    """Same construction as diagnose_m3_reparam.py's _apply_ls_init, but
    every leaf is built directly as float64 (dtype=jnp.float64 hardcoded)
    rather than inherited from the model's pre-existing float32 leaf -
    nnx.Linear's param_dtype defaults to float32 regardless of the global
    jax_enable_x64 flag (see tools/diagnose_m3_rank_x64.py), so inheriting
    dtype from the live module would silently stay float32."""
    block = model.layers[0]
    d_model = D_MODEL
    d_input, d_output = w_ls_in_out.shape

    encoder_kernel = jnp.zeros((d_input, d_model), dtype=jnp.float64)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_ls_in_out, dtype=jnp.float64))
    model.encoder.kernel.value = encoder_kernel
    model.encoder.bias.value = jnp.zeros((d_model,), dtype=jnp.float64)

    block.seq.D.value = jnp.zeros_like(block.seq.D.value, dtype=jnp.float64)
    block.out.kernel.value = jnp.zeros((d_model, d_model), dtype=jnp.float64)
    block.out.bias.value = jnp.zeros((d_model,), dtype=jnp.float64)

    decoder_kernel = jnp.zeros((d_model, d_output), dtype=jnp.float64)
    decoder_kernel = decoder_kernel.at[:d_output, :].set(jnp.eye(d_output, dtype=jnp.float64))
    model.decoder.kernel.value = decoder_kernel
    model.decoder.bias.value = jnp.zeros((d_output,), dtype=jnp.float64)


def _train_logged(model: StackedModel, inputs, targets, tx, epochs: int) -> np.ndarray:
    """Returns losses[0:epochs+1]: losses[t] for t < epochs is the
    PRE-update loss at step t (so losses[0] == init loss by construction);
    losses[epochs] is the loss after the final update."""
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def loss_fn(m):
        pred = _d_only_forward(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    losses = np.full(epochs + 1, np.nan, dtype=np.float64)
    for step in range(epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        loss_v = float(loss)
        losses[step] = loss_v
        if not np.isfinite(loss_v):
            print(f"    step {step}: non-finite loss ({loss_v}), stopping early")
            return losses
        optimizer.update(model, grads)
    losses[epochs] = float(loss_fn(model))
    return losses


def _summarize(name: str, losses: np.ndarray, ls_mse: float) -> None:
    finite = np.isfinite(losses)
    n_finite = int(finite.sum())
    init, final = losses[0], losses[np.where(finite)[0][-1]]
    diffs = np.diff(losses[finite])
    n_up = int(np.sum(diffs > 0))
    n_down = int(np.sum(diffs < 0))
    argmax_step = int(np.nanargmax(losses))
    max_val = float(losses[argmax_step])
    orders = np.log10(final / init) if init > 0 and final > 0 else float("nan")
    monotonic_nondecreasing = bool(np.all(diffs >= 0))
    print(f"  [{name}] init={init:.6e}  final={final:.6e}  ratio_to_LS_floor(final)={final / ls_mse:.6e}")
    print(f"    orders of magnitude rise (log10 final/init): {orders:+.2f}")
    print(f"    steps logged finite: {n_finite}/{len(losses)}  up-steps={n_up}  down-steps={n_down}"
          f"  monotonic_nondecreasing={monotonic_nondecreasing}")
    print(f"    max loss {max_val:.6e} at step {argmax_step}")
    # first step (if any) crossing each order-of-magnitude threshold above init
    for mult, label in [(10, "10x"), (1e3, "1e3x"), (1e6, "1e6x"), (1e8, "1e8x")]:
        idx = np.where(losses[finite] > init * mult)[0]
        step_str = str(int(idx[0])) if idx.size else "never"
        print(f"    first step > {label} init: {step_str}")


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    print(f"case_data dtype: inputs={inputs.dtype}, targets={targets.dtype}")
    mean_target_sq = float(jnp.mean(targets**2))

    ab_hat, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    w_ls_in_out = np.asarray(ab_hat.T, dtype=np.float64)
    print(f"LS floor (float64): mse={ls_mse:.6e}  nmse={ls_mse / mean_target_sq:.6e}")

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)

    results: dict[str, np.ndarray] = {}
    for name, tx in VARIANT_SPECS.items():
        model = _build_model(key)
        _apply_ls_init_x64(model, w_ls_in_out)

        # self-check 1: the forward pass must actually COMPUTE in float64,
        # not just be fed float64 inputs - nnx.Linear stores param_dtype as
        # a plain (non-Variable) attribute fixed at construction (float32,
        # unaffected by our later .value= overwrite or by jax_enable_x64),
        # so this is not guaranteed by construction and must be checked.
        pred_dtype = _d_only_forward(model, inputs).dtype
        print(f"\n[{name}] self-check: forward-pass output dtype = {pred_dtype}")
        if pred_dtype != jnp.float64:
            print(f"[{name}] SELF-CHECK FAILED - forward pass is NOT computing in float64 "
                  f"(got {pred_dtype}). Aborting rather than trusting a mislabeled result.")
            continue

        # self-check 2: init loss must match the float64 LS floor closely
        def loss_fn(m):
            pred = _d_only_forward(m, inputs)
            return jnp.mean((pred - targets) ** 2)

        init_loss = float(loss_fn(model))
        rel_err = abs(init_loss - ls_mse) / max(ls_mse, 1e-300)
        print(f"[{name}] self-check: init_mse={init_loss:.6e} vs LS floor={ls_mse:.6e} rel_err={rel_err:.3e}")
        if rel_err > 1e-6 and abs(init_loss - ls_mse) > 1e-12:
            print(f"[{name}] SELF-CHECK FAILED - construction does not realize the LS solution at float64 precision. Skipping.")
            continue

        losses = _train_logged(model, inputs, targets, tx, EPOCHS)
        results[name] = losses
        _summarize(name, losses, ls_mse)

    # --- persist raw trajectories ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = "step," + ",".join(results.keys())
    rows = [header]
    n_rows = EPOCHS + 1
    for t in range(n_rows):
        vals = [f"{results[name][t]:.10e}" for name in results]
        rows.append(f"{t}," + ",".join(vals))
    OUT_CSV.write_text("\n".join(rows))
    print(f"\nwrote {OUT_CSV}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in results.items():
        steps = np.arange(len(losses))
        finite = np.isfinite(losses) & (losses > 0)
        ax.semilogy(steps[finite], losses[finite], label=name)
    ax.axhline(ls_mse, color="black", linestyle="--", label=f"LS floor ({ls_mse:.1e})")
    ax.set_xlabel("step")
    ax.set_ylabel("teacher-forced MSE (log scale)")
    ax.set_title(f"D-only, case {CASE}, LS-init, float64 - EXP4 re-check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"wrote {OUT_PNG}")

    print("\n[summary]")
    for name, losses in results.items():
        finite = np.isfinite(losses)
        final = losses[np.where(finite)[0][-1]]
        print(f"  {name}: init={losses[0]:.6e}  final={final:.6e}  "
              f"orders_rise={np.log10(final / losses[0]):+.2f}")


if __name__ == "__main__":
    main()
