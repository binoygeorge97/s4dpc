"""Part C (docs/DECISIONS.md) - Figure 1: characterizing the divergence
from the exact optimum, corrected to use the TRUE 490-dimensional null
space (not the earlier float32-artifact 32-dim one - see
diagnose_m3_rank_x64.py). All Gauss-Newton computation forces
jax_enable_x64 throughout.

Re-runs LS-init D-only (same construction as diagnose_m3_reparam.py's
_apply_ls_init, already validated to land within float32 precision of
the LS floor), training with Adam AND separately with plain SGD, logging
MSE every single step. Displacement from the LS-init point is decomposed
into a null-space component (projected onto the 490-dim null basis of
the Gauss-Newton matrix AT the LS-init point) and its orthogonal
complement, tracked periodically (every 20 steps - the projection itself
is cheap, but recomputing it isn't worth doing every single step given
490x550 basis size).

Prediction being tested: parameters travel far along the null space
while function-space error grows from second-order leakage (the null
space is only a FIRST-ORDER/tangent-plane flat direction - moving along
it finitely can still change the function via curvature terms the
linear Gauss-Newton approximation doesn't see). If SGD stays near the
optimum while Adam wanders, that is a specific, mechanistic claim about
Adam's per-coordinate normalization amplifying motion along near-zero-
gradient directions that plain SGD's magnitude-proportional steps would
not.

    python tools/diagnose_m3_divergence_figure.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data, fit_least_squares
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LR = 1e-3
SEED = 0
DISPLACEMENT_EVERY = 20

OUT_PATH = _REPO_ROOT / "docs" / "m3_divergence_figure1.png"


def _extract_params(model: StackedModel) -> dict:
    block = model.layers[0]
    return {
        "encoder_kernel": model.encoder.kernel.value,
        "encoder_bias": model.encoder.bias.value,
        "D": block.seq.D.value,
        "out_kernel": block.out.kernel.value,
        "out_bias": block.out.bias.value,
        "decoder_kernel": model.decoder.kernel.value,
        "decoder_bias": model.decoder.bias.value,
    }


def _apply_ls_init(model: StackedModel, w_ls_in_out: np.ndarray) -> None:
    """Same construction as diagnose_m3_reparam.py's _apply_ls_init, but
    every leaf is built directly as float64 (dtype=jnp.float64 hardcoded)
    rather than inherited from the model's pre-existing float32 leaf -
    nnx.Linear's param_dtype defaults to float32 regardless of the global
    jax_enable_x64 flag (see tools/diagnose_m3_rank_x64.py and
    tools/diagnose_m3_exp4_x64.py), so inheriting dtype from the live
    module would silently stay float32 and undo the whole point of
    forcing x64. Zero D and out (so only the residual/skip path is
    live), route the LS solution straight through via encoder's first
    d_output columns and an identity block in decoder's first d_output
    rows."""
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


def _d_only_forward(model: StackedModel, inputs: jax.Array) -> jax.Array:
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x
    d_vec = block.seq.D.value.squeeze(-1)
    x = x * d_vec[None, :]
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def _predict_raw(params: dict, inputs: jax.Array) -> jax.Array:
    x = inputs @ params["encoder_kernel"] + params["encoder_bias"]
    skip = x
    d_vec = params["D"].squeeze(-1)
    x = x * d_vec[None, :]
    x = x @ params["out_kernel"] + params["out_bias"]
    x = skip + x
    return x @ params["decoder_kernel"] + params["decoder_bias"]


def _null_basis_at(params: dict, inputs64: jax.Array) -> tuple[np.ndarray, int]:
    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]

    def predict_flat(flat_p):
        return _predict_raw(unravel(flat_p), inputs64).reshape(-1)

    j = np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64)
    _, S, Vt = np.linalg.svd(j, full_matrices=True)
    eps = np.finfo(np.float64).eps
    sv_tol = S.max() * max(j.shape) * eps
    rank = int(np.sum(S > sv_tol))
    return Vt[rank:], n_params  # null_basis: (null_dim, n_params)


def _run_trajectory(label: str, tx, ls_init_params: dict, inputs, targets, null_basis: np.ndarray):
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    # apply LS-init params directly (fresh model of the right shape),
    # hardcoded to float64 - same reasoning as _apply_ls_init above
    model.encoder.kernel.value = jnp.asarray(ls_init_params["encoder_kernel"], dtype=jnp.float64)
    model.encoder.bias.value = jnp.asarray(ls_init_params["encoder_bias"], dtype=jnp.float64)
    model.layers[0].seq.D.value = jnp.asarray(ls_init_params["D"], dtype=jnp.float64)
    model.layers[0].out.kernel.value = jnp.asarray(ls_init_params["out_kernel"], dtype=jnp.float64)
    model.layers[0].out.bias.value = jnp.asarray(ls_init_params["out_bias"], dtype=jnp.float64)
    model.decoder.kernel.value = jnp.asarray(ls_init_params["decoder_kernel"], dtype=jnp.float64)
    model.decoder.bias.value = jnp.asarray(ls_init_params["decoder_bias"], dtype=jnp.float64)

    # self-check: forward pass must actually COMPUTE in float64
    pred_dtype = _d_only_forward(model, inputs).dtype
    if pred_dtype != jnp.float64:
        raise RuntimeError(f"[{label}] forward pass is not float64 (got {pred_dtype}) - not trusting this run")

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ls_init_flat, unravel = ravel_pytree(ls_init_params)
    ls_init_flat = np.asarray(ls_init_flat, dtype=np.float64)

    def loss_fn(m):
        pred = _d_only_forward(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    steps, losses, null_disp, orth_disp = [], [], [], []
    for step in range(EPOCHS):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        loss_v = float(loss)
        steps.append(step)
        losses.append(loss_v)
        if step % DISPLACEMENT_EVERY == 0 or step == EPOCHS - 1:
            cur_params = _extract_params(model)
            cur_flat, _ = ravel_pytree(cur_params)
            cur_flat = np.asarray(cur_flat, dtype=np.float64)
            disp = cur_flat - ls_init_flat
            null_component = null_basis.T @ (null_basis @ disp)  # projection onto null space
            null_norm = float(np.linalg.norm(null_component))
            orth_norm = float(np.linalg.norm(disp - null_component))
            null_disp.append((step, null_norm))
            orth_disp.append((step, orth_norm))
            print(f"  [{label}] step {step:4d}  loss={loss_v:.6e}  "
                  f"||disp_null||={null_norm:.4e}  ||disp_orth||={orth_norm:.4e}")
        optimizer.update(model, grads)

    return steps, losses, null_disp, orth_disp


def main() -> None:
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)
    ab_hat, ls_mse = fit_least_squares(CASE, L_MAX, -10.0, 10.0)
    w_ls_in_out = np.asarray(ab_hat.T)
    print(f"LS floor mse={ls_mse:.6e}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    init_model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=False, rngs=nnx.Rngs(params=key),
    )
    _apply_ls_init(init_model, w_ls_in_out)
    ls_init_params = _extract_params(init_model)
    init_loss = float(jnp.mean((_d_only_forward(init_model, inputs) - targets) ** 2))
    print(f"LS-init self-check: init_mse={init_loss:.6e} (should be near the LS floor)")

    inputs64 = jnp.asarray(inputs, dtype=jnp.float64)
    targets64 = jnp.asarray(targets, dtype=jnp.float64)
    print(f"ls_init_params dtype: {jax.tree_util.tree_leaves(ls_init_params)[0].dtype}, "
          f"inputs64 dtype: {inputs64.dtype}")
    null_basis, n_params = _null_basis_at(ls_init_params, inputs64)
    print(f"null_basis shape={null_basis.shape} (n_params={n_params}, expect null_dim~490)")

    print("\n=== Adam ===")
    steps_a, losses_a, null_a, orth_a = _run_trajectory(
        "Adam", optax.adamw(LR, weight_decay=0.0), ls_init_params, inputs64, targets64, null_basis
    )

    print("\n=== SGD ===")
    steps_s, losses_s, null_s, orth_s = _run_trajectory(
        "SGD", optax.sgd(LR), ls_init_params, inputs64, targets64, null_basis
    )

    monotonic_a = all(losses_a[i] <= losses_a[i + 1] * 1.0001 for i in range(len(losses_a) - 1))  # crude check, printed not asserted
    print(f"\nAdam loss trajectory: first 5 = {losses_a[:5]}, last 5 = {losses_a[-5:]}")
    print(f"SGD  loss trajectory: first 5 = {losses_s[:5]}, last 5 = {losses_s[-5:]}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    ax1.semilogy(steps_a, np.array(losses_a) + 1e-20, label="Adam", color="C0")
    ax1.semilogy(steps_s, np.array(losses_s) + 1e-20, label="SGD", color="C1")
    ax1.axhline(ls_mse, color="red", linestyle="--", label=f"LS floor ({ls_mse:.1e})")
    ax1.set_xlabel("step")
    ax1.set_ylabel("teacher-forced MSE (log scale)")
    ax1.set_title("Figure 1a: divergence from LS-init, Adam vs SGD")
    ax1.legend()

    na_steps, na_vals = zip(*null_a)
    oa_steps, oa_vals = zip(*orth_a)
    ns_steps, ns_vals = zip(*null_s)
    os_steps, os_vals = zip(*orth_s)
    ax2.semilogy(na_steps, np.array(na_vals) + 1e-20, label="Adam, null-space displacement", color="C0")
    ax2.semilogy(oa_steps, np.array(oa_vals) + 1e-20, label="Adam, orthogonal displacement", color="C0", linestyle="--")
    ax2.semilogy(ns_steps, np.array(ns_vals) + 1e-20, label="SGD, null-space displacement", color="C1")
    ax2.semilogy(os_steps, np.array(os_vals) + 1e-20, label="SGD, orthogonal displacement", color="C1", linestyle="--")
    ax2.set_xlabel("step")
    ax2.set_ylabel("||parameter displacement|| (log scale)")
    ax2.set_title("Figure 1b: displacement decomposition (null space vs orthogonal)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
