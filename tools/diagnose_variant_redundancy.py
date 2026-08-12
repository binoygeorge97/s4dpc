"""Task 2 (docs/DECISIONS.md, corollary of the D-only Adam-instability
finding): does LayerNorm make a variant's parameterization MORE
overcomplete than M3's, and does the same mechanism predict it is MORE
susceptible to the Adam null-space random walk? For each of M3, M4, M5,
M6, M6_fix on case 3, float64:
  a. total params, numerical rank of the Gauss-Newton Jacobian, null
     dimension, % redundancy - at a FRESH random init (matching
     tools/diagnose_m3_gauge_rank_scan.py / diagnose_m3_gauge_full_m3.py's
     precedent: the rank bound is generic-parameterization structure, not
     assumed point-dependent, but IS measured, not assumed, for the
     nonlinear variants where it does not have to hold).
  b/c. Adam vs SGD from an LS-init (or best-available) point: does the
     loss rise, how fast, how many steps to lose N orders.
  d. null-space vs orthogonal displacement decomposition, using the
     Gauss-Newton null basis AT the LS-init point specifically (not the
     same point as (a) - LayerNorm/GELU/GLU are genuine input
     nonlinearities, so unlike D-only's provably-everywhere-affine case,
     there is no reason the rank at a random point and at the LS-init
     point need agree for M4/M5/M6/M6_fix).

Generalizes two established, already-validated pieces rather than
re-deriving them:
  - The LS-init construction (diagnose_m3_reparam.py/diagnose_m3_
    divergence_figure.py): D-only zeroes D and out, since the S4 kernel
    path is already architecturally absent from that ablation. Here, for
    the FULL model, the S4Layer's conv-mode output is forced to exactly
    zero for ANY input by zeroing D and C_real_imag together (derivable
    directly from kernel_DPLR: aterm[0]=C.conj(), and both k00 and k01 -
    the only two terms feeding the output - are scaled by aterm[0], so
    C=0 makes them exactly 0 regardless of k10/k11's own values, hence
    K=0 identically); zeroing out/out2 too then makes the whole block's
    contribution exactly 0 for any input, for ANY norm/activation/glu
    combination (GELU(0)=0 exactly; a GLU gate times 0 is 0 regardless of
    the gate value; LayerNorm/StaticNorm only touch the S4Layer's INPUT
    via prenorm, never the skip connection captured before it) - so the
    skip connection alone carries the LS solution through untouched,
    uniformly across all 5 variants.
  - The stacked-multi-trajectory Jacobian (diagnose_m3_gauge_full_m3.py,
    Part B.4): a single L=100 trajectory gives only 600 output samples,
    far fewer than these models' several-thousand params, which
    trivially caps the observed rank - N_TRAJECTORIES=8 independent
    trajectories (4800 samples) comfortably exceeds every variant's
    param count here. Used for BOTH (a)'s random-init rank and (d)'s
    LS-init null basis - Part C's original single-trajectory basis only
    worked because D-only had just 550 params.

Unlike the D-only/full-M3 rank scripts (which hand-rolled a raw-array
reimplementation to differentiate through), this differentiates the
REAL model's own __call__ directly via nnx.split/nnx.merge (the same
idiom control.py's rollout_learned and blocks.py's channel-vmap already
use) - this generalizes to any norm/activation/glu combination for free,
with no per-variant reimplementation or self-check-against-reimplementation
needed, since there is no separate reimplementation to go out of sync.

CAVEAT that matters for reading M6_fix's results: StaticNorm defaults to
mu=0/sigma=1 (identity) until .calibrate() is called explicitly
(s4dpc/blocks.py) - nothing in this script (or identify.py's standard
pipeline) calibrates it, so M6_fix here is an UNCALIBRATED identity norm,
architecturally indistinguishable from M4 (norm=none). Any M6_fix result
that looks identical to M4 is expected, not a finding about static
normalization's effect once actually calibrated.

    python tools/diagnose_variant_redundancy.py
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
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LR = 1e-3
SEED = 0
DISPLACEMENT_EVERY = 20
N_TRAJECTORIES = 8  # output samples (N_TRAJ*L*d_output=4800) must exceed every variant's n_params

VARIANTS_TO_TEST = ["M3", "M4", "M5", "M6", "M6_fix"]

OUT_CSV = _REPO_ROOT / "docs" / "variant_redundancy_summary.csv"
OUT_PNG = _REPO_ROOT / "docs" / "variant_redundancy_figure.png"


# --------------------------------------------------------------------------
# shared dtype cast (complex-aware - see s4dpc/identify.py's _cast_params;
# duplicated here per this repo's standalone-tools-script convention)
# --------------------------------------------------------------------------

def _cast_all_params(model: StackedModel) -> None:
    def _cast_leaf(x: jax.Array) -> jax.Array:
        target = jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64
        return x.astype(target)

    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(_cast_leaf, state)
    nnx.update(model, state)


def _build_model(variant: str, key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    _cast_all_params(model)
    return model


def _apply_ls_init_general(model: StackedModel, w_ls_in_out: np.ndarray, has_glu: bool) -> None:
    block = model.layers[0]
    d_input, d_output = w_ls_in_out.shape

    encoder_kernel = jnp.zeros((d_input, D_MODEL), dtype=jnp.float64)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_ls_in_out, dtype=jnp.float64))
    model.encoder.kernel.value = encoder_kernel
    model.encoder.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    block.seq.D.value = jnp.zeros_like(block.seq.D.value, dtype=jnp.float64)
    block.seq.C_real_imag.value = jnp.zeros_like(block.seq.C_real_imag.value, dtype=jnp.float64)
    block.out.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)
    block.out.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)
    if has_glu:
        block.out2.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)
        block.out2.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    decoder_kernel = jnp.zeros((D_MODEL, d_output), dtype=jnp.float64)
    decoder_kernel = decoder_kernel.at[:d_output, :].set(jnp.eye(d_output, dtype=jnp.float64))
    model.decoder.kernel.value = decoder_kernel
    model.decoder.bias.value = jnp.zeros((d_output,), dtype=jnp.float64)


# --------------------------------------------------------------------------
# real-safe ravel: S4LayerEnsemble's P/B params are genuinely complex
# (unlike Lambda_re/Lambda_im/C_real_imag, which are already split into
# real arrays - see legacy/s4.py). Concatenating them with the real
# leaves via plain ravel_pytree promotes the WHOLE flat vector to
# complex128 (confirmed empirically: this is exactly what happened -
# jax.jacfwd then refuses outright, "requires real-valued inputs...
# got complex128"). This is the SAME hazard tools/diagnose_m3_gauge_
# full_m3.py (Part B.4) hit and got WRONG silently instead of crashing:
# its blanket `jnp.asarray(x, dtype=jnp.float64)` cast DISCARDS P/B's
# imaginary part rather than raising, so B.4's reported "full M3:
# rank=1183, null=2455" was computed on a model with P/B's imaginary
# parts silently zeroed - not the real parameterization (see
# docs/DECISIONS.md's Task 2 entry for the correction). Splitting each
# complex leaf into a real [real,imag] pair before raveling (and
# rejoining after unraveling) keeps the flat vector real-valued for
# jacfwd while losing no information.
# --------------------------------------------------------------------------

def _make_real_ravel(params):
    is_complex = jax.tree_util.tree_map(lambda x: jnp.iscomplexobj(x), params)

    def _split(x):
        return jnp.stack([x.real, x.imag], axis=-1) if jnp.iscomplexobj(x) else x

    real_params = jax.tree_util.tree_map(_split, params)
    flat, unravel_real = ravel_pytree(real_params)

    def unravel(flat_p):
        real_tree = unravel_real(flat_p)

        def _join(x, was_complex):
            return (x[..., 0] + 1j * x[..., 1]) if was_complex else x

        return jax.tree_util.tree_map(_join, real_tree, is_complex)

    return flat, unravel


# --------------------------------------------------------------------------
# stacked-trajectory Gauss-Newton Jacobian, via the model's REAL __call__
# --------------------------------------------------------------------------

def _stacked_jacobian(graphdef, params, rest) -> tuple[np.ndarray, int]:
    flat_params, unravel = _make_real_ravel(params)
    n_params = flat_params.shape[0]

    traj_inputs, _ = generate_microgrid_trajectory(
        batch_size=N_TRAJECTORIES, length=L_MAX, seed=123, system_case=CASE, dt=0.01,
        aprbs_low=-10.0, aprbs_high=10.0,
    )
    j_blocks = []
    for t in range(N_TRAJECTORIES):
        inputs_t = jnp.asarray(traj_inputs[t], dtype=jnp.float64)

        def predict_flat(flat_p, inputs_t=inputs_t):
            # rest (e.g. M6_fix's StaticNorm mu/sigma) is not part of
            # what's being differentiated - held fixed, closed over.
            m = nnx.merge(graphdef, unravel(flat_p), rest)
            states = m.init_state()
            out, _ = m(inputs_t, states)
            return out.reshape(-1)

        j_blocks.append(np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64))
    return np.concatenate(j_blocks, axis=0), n_params


def _rank_and_null(j: np.ndarray, n_params: int) -> tuple[int, int]:
    singular_values = np.linalg.svd(j, compute_uv=False)
    eps = np.finfo(np.float64).eps
    sv_tol = singular_values.max() * max(j.shape) * eps
    rank = int(np.sum(singular_values > sv_tol))
    return rank, n_params - rank


def _null_basis(j: np.ndarray, rank: int) -> np.ndarray:
    _, _, Vt = np.linalg.svd(j, full_matrices=True)
    return Vt[rank:]  # (null_dim, n_params)


# --------------------------------------------------------------------------
# Adam / SGD trajectories from the LS-init point
# --------------------------------------------------------------------------

def _run_trajectory(label, variant, tx, graphdef, ls_init_params, rest, inputs, targets, null_basis):
    # null_basis's columns are in the _make_real_ravel ordering (from
    # _stacked_jacobian) - displacement must be raveled the SAME way for
    # the projection below to be meaningful, not plain ravel_pytree
    # (which would promote to complex128 and silently misalign/break -
    # see _make_real_ravel's docstring).
    flat_init, _ = _make_real_ravel(ls_init_params)
    flat_init_np = np.asarray(flat_init, dtype=np.float64)
    model = nnx.merge(graphdef, ls_init_params, rest)

    pred_dtype_check, _ = model(inputs, model.init_state())
    if pred_dtype_check.dtype != jnp.float64:
        raise RuntimeError(f"[{variant}/{label}] forward pass is not float64 (got {pred_dtype_check.dtype})")

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    def loss_fn(m):
        pred, _ = m(inputs, m.init_state())
        return jnp.mean((pred - targets) ** 2)

    steps, losses, null_disp, orth_disp = [], [], [], []
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        loss_v = float(loss)
        steps.append(step)
        losses.append(loss_v)
        if step % DISPLACEMENT_EVERY == 0 or step == EPOCHS - 1:
            cur_flat = np.asarray(_make_real_ravel(nnx.state(model, nnx.Param))[0], dtype=np.float64)
            disp = cur_flat - flat_init_np
            null_component = null_basis.T @ (null_basis @ disp)
            null_norm = float(np.linalg.norm(null_component))
            orth_norm = float(np.linalg.norm(disp - null_component))
            null_disp.append(null_norm)
            orth_disp.append(orth_norm)
        if not np.isfinite(loss_v):
            print(f"    [{variant}/{label}] step {step}: non-finite loss, stopping early")
            break
        optimizer.update(model, grads)
    return np.array(losses), np.array(null_disp), np.array(orth_disp)


def _orders_report(label: str, losses: np.ndarray, init: float) -> dict:
    final = losses[-1]
    orders = float(np.log10(final / init)) if init > 0 and final > 0 else float("nan")
    first_1e8x = next((s for s, v in enumerate(losses) if v > init * 1e8), None)
    print(f"    [{label}] init={init:.6e}  final={final:.6e}  orders_rise={orders:+.2f}  "
          f"first_step_>1e8x_init={first_1e8x}")
    return {"final": final, "orders_rise": orders, "first_step_1e8x": first_1e8x}


def _self_check_ravel_roundtrip(variant: str, model: StackedModel, inputs: jax.Array) -> None:
    """Every prior script in this repo ravels a hand-built plain dict of
    named leaves, never nnx.split's own State object directly - this
    script's genericity (working across all 5 variants' different
    norm/activation/glu leaf sets without per-variant enumeration) relies
    on ravel_pytree/nnx.merge working correctly on an nnx.State pytree,
    which is unprecedented in this codebase specifically even though it
    should hold (nnx.State is a registered JAX pytree with array leaves -
    the same property jax.vmap/nnx.value_and_grad already rely on
    throughout blocks.py/control.py). Checked directly rather than
    trusted: split -> ravel -> unravel -> merge -> forward, compared
    bit-exactly against the original model's own forward output."""
    # M6_fix's StaticNorm holds mu/sigma as nnx.Variable, not nnx.Param
    # (s4dpc/blocks.py) - nnx.split with a single non-exhaustive filter
    # raises ValueError("Non-exhaustive filters...") once any such
    # leftover state exists; `...` explicitly captures everything else.
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)
    flat, unravel = ravel_pytree(state)
    rebuilt = nnx.merge(graphdef, unravel(flat), rest)
    original_out, _ = model(inputs, model.init_state())
    rebuilt_out, _ = rebuilt(inputs, rebuilt.init_state())
    max_diff = float(jnp.max(jnp.abs(original_out - rebuilt_out)))
    print(f"  [{variant}] ravel/unravel/merge round-trip self-check: max abs diff={max_diff:.3e}")
    if max_diff != 0.0:
        raise RuntimeError(
            f"[{variant}] ravel_pytree/nnx.merge round-trip on an nnx.State does NOT "
            f"reproduce the original forward pass (max abs diff={max_diff:.3e}) - stopping "
            "before trusting any downstream Jacobian/training result."
        )


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    print(f"case_data dtype: inputs={inputs.dtype}, targets={targets.dtype}")

    ab_hat, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    w_ls_in_out = np.asarray(ab_hat.T, dtype=np.float64)
    print(f"LS floor (float64): mse={ls_mse:.6e}\n")

    summary_rows = []
    fig, axes = plt.subplots(len(VARIANTS_TO_TEST), 2, figsize=(11, 4 * len(VARIANTS_TO_TEST)))

    for vi, variant in enumerate(VARIANTS_TO_TEST):
      try:
        print(f"=== {variant} ===")
        has_glu = VARIANTS[variant]["glu"]
        key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)

        # --- (a) rank/redundancy at a FRESH random init ---
        rand_model = _build_model(variant, key)
        _self_check_ravel_roundtrip(variant, rand_model, inputs)
        graphdef, rand_params, rand_rest = nnx.split(rand_model, nnx.Param, ...)
        j_rand, n_params = _stacked_jacobian(graphdef, rand_params, rand_rest)
        rank_rand, null_rand = _rank_and_null(j_rand, n_params)
        pct_redundant = 100 * null_rand / n_params
        print(f"  (a) [random init] n_params={n_params}  rank={rank_rand}  null={null_rand}  "
              f"redundant={pct_redundant:.1f}%")

        # --- LS-init model, self-checked ---
        ls_model = _build_model(variant, key)
        _apply_ls_init_general(ls_model, w_ls_in_out, has_glu)
        init_loss = float(jnp.mean((ls_model(inputs, ls_model.init_state())[0] - targets) ** 2))
        rel_err = abs(init_loss - ls_mse) / max(ls_mse, 1e-300)
        print(f"  LS-init self-check: init_mse={init_loss:.6e} vs LS floor={ls_mse:.6e} rel_err={rel_err:.3e}")
        if rel_err > 1e-6 and abs(init_loss - ls_mse) > 1e-12:
            print(f"  SELF-CHECK FAILED for {variant} - skipping.")
            continue

        # --- (d) null basis AT the LS-init point ---
        graphdef_ls, ls_params, ls_rest = nnx.split(ls_model, nnx.Param, ...)
        j_ls, n_params_ls = _stacked_jacobian(graphdef_ls, ls_params, ls_rest)
        rank_ls, null_ls = _rank_and_null(j_ls, n_params_ls)
        basis_ls = _null_basis(j_ls, rank_ls)
        print(f"  [LS-init point] rank={rank_ls}  null={null_ls}  (may differ from random-init rank - "
              f"this variant has genuine input nonlinearities)" if variant != "M3" else
              f"  [LS-init point] rank={rank_ls}  null={null_ls}")

        # --- (b)/(c)/(d) Adam vs SGD from the LS-init point ---
        print("  --- Adam ---")
        losses_a, null_a, orth_a = _run_trajectory(
            "Adam", variant, optax.adamw(LR, weight_decay=0.0), graphdef_ls, ls_params, ls_rest,
            inputs, targets, basis_ls
        )
        rep_a = _orders_report("Adam", losses_a, init_loss)

        print("  --- SGD ---")
        losses_s, null_s, orth_s = _run_trajectory(
            "SGD", variant, optax.sgd(LR), graphdef_ls, ls_params, ls_rest, inputs, targets, basis_ls
        )
        rep_s = _orders_report("SGD", losses_s, init_loss)

        summary_rows.append({
            "variant": variant, "n_params": n_params, "rank_random_init": rank_rand,
            "null_random_init": null_rand, "pct_redundant_random_init": pct_redundant,
            "rank_ls_init": rank_ls, "null_ls_init": null_ls,
            "adam_final_mse": rep_a["final"], "adam_orders_rise": rep_a["orders_rise"],
            "adam_first_step_1e8x": rep_a["first_step_1e8x"],
            "sgd_final_mse": rep_s["final"], "sgd_orders_rise": rep_s["orders_rise"],
            "sgd_first_step_1e8x": rep_s["first_step_1e8x"],
            "adam_disp_null_final": float(null_a[-1]), "adam_disp_orth_final": float(orth_a[-1]),
            "sgd_disp_null_final": float(null_s[-1]), "sgd_disp_orth_final": float(orth_s[-1]),
        })

        ax1, ax2 = axes[vi]
        steps_arr = np.arange(len(losses_a))
        ax1.semilogy(steps_arr[: len(losses_a)], losses_a + 1e-30, label="Adam", color="C0")
        ax1.semilogy(np.arange(len(losses_s)), losses_s + 1e-30, label="SGD", color="C1")
        ax1.axhline(ls_mse, color="red", linestyle="--", label=f"LS floor")
        ax1.set_title(f"{variant}: loss")
        ax1.set_xlabel("step")
        ax1.legend(fontsize=7)

        disp_steps = np.arange(0, len(losses_a), DISPLACEMENT_EVERY).tolist()
        if (len(losses_a) - 1) not in disp_steps:
            disp_steps.append(len(losses_a) - 1)
        disp_steps = disp_steps[: len(null_a)]
        ax2.semilogy(disp_steps, null_a + 1e-30, label="Adam null", color="C0")
        ax2.semilogy(disp_steps, orth_a + 1e-30, label="Adam orth", color="C0", linestyle="--")
        ax2.semilogy(disp_steps[: len(null_s)], null_s + 1e-30, label="SGD null", color="C1")
        ax2.semilogy(disp_steps[: len(orth_s)], orth_s + 1e-30, label="SGD orth", color="C1", linestyle="--")
        ax2.set_title(f"{variant}: displacement")
        ax2.set_xlabel("step")
        ax2.legend(fontsize=7)
        print()
      except Exception as e:
        import traceback
        print(f"  {variant} FAILED - continuing to the next variant so partial results aren't lost.")
        traceback.print_exc()

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    print(f"wrote {OUT_PNG}")

    if summary_rows:
        header = list(summary_rows[0].keys())
        lines = [",".join(header)]
        for row in summary_rows:
            lines.append(",".join(str(row[h]) for h in header))
        OUT_CSV.write_text("\n".join(lines))
        print(f"wrote {OUT_CSV}")

    print("\n=== SUMMARY ===")
    print(f"{'variant':8s} {'n_params':>9s} {'null%(rand)':>12s} {'null%(ls)':>10s} "
          f"{'adam_orders':>12s} {'sgd_orders':>11s}")
    for row in summary_rows:
        null_pct_ls = 100 * row["null_ls_init"] / row["n_params"]
        print(f"{row['variant']:8s} {row['n_params']:9d} {row['pct_redundant_random_init']:11.1f}% "
              f"{null_pct_ls:9.1f}% {row['adam_orders_rise']:12.2f} {row['sgd_orders_rise']:11.2f}")


if __name__ == "__main__":
    main()
