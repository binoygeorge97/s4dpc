"""Diagnostic: Gauss-Newton (J^T J) conditioning of M3's D-only parameters
on case 3, at init and after 2000 training steps. Part of the M3
diagnosis (docs/DECISIONS.md) - not part of the regular pipeline.

J is the Jacobian of the flattened D-only prediction w.r.t. the flattened
D-only parameters (encoder kernel/bias, per-channel D, out-projection
kernel/bias, decoder kernel/bias), via jax.jacfwd on the full case-3
sequence (L=100 - small enough that no further sub-batching is needed).
J^T J is the standard Gauss-Newton approximation to the loss Hessian for
an MSE loss.

Reimplements the D-only forward pass on RAW ARRAYS (not nnx module
calls, since jacfwd needs a plain differentiable function of a flat
parameter vector) - self-checked against the real model's own D-only
output before the Jacobian is trusted.

    python tools/diagnose_m3_conditioning.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.flatten_util import ravel_pytree

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, case_data
from s4dpc.model import StackedModel

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0


def _extract_d_only_params(model: StackedModel) -> dict:
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


def _d_only_forward_nnx(model: StackedModel, inputs: jax.Array) -> jax.Array:
    """Same D-only ablation as tools/diagnose_m3_structure.py, via real
    nnx module calls - the reference this file's raw-array version is
    checked against."""
    block = model.layers[0]
    x = model.encoder(inputs)
    skip = x
    d_vec = block.seq.D.value.squeeze(-1)  # (d_model,)
    x = x * d_vec[None, :]
    x = block.out(x)
    x = skip + x
    return model.decoder(x)


def _predict_raw(params: dict, inputs: jax.Array) -> jax.Array:
    """Same computation as _d_only_forward_nnx, on raw arrays only."""
    x = inputs @ params["encoder_kernel"] + params["encoder_bias"]
    skip = x
    d_vec = params["D"].squeeze(-1)
    x = x * d_vec[None, :]
    x = x @ params["out_kernel"] + params["out_bias"]
    x = skip + x
    return x @ params["decoder_kernel"] + params["decoder_bias"]


def _gauss_newton_report(label: str, model: StackedModel, inputs: jax.Array) -> None:
    params = _extract_d_only_params(model)

    # self-check: raw-array reimplementation must match the real model exactly
    real_out = _d_only_forward_nnx(model, inputs)
    raw_out = _predict_raw(params, inputs)
    max_diff = float(jnp.max(jnp.abs(real_out - raw_out)))
    print(f"[{label}] self-check (raw-array vs real model), max abs diff: {max_diff:.3e}")
    if max_diff > 1e-5:
        print(f"[{label}] SELF-CHECK FAILED - not trusting this Jacobian. Skipping.")
        return

    flat_params, unravel = ravel_pytree(params)
    n_params = flat_params.shape[0]

    def predict_flat(flat_p):
        return _predict_raw(unravel(flat_p), inputs).reshape(-1)

    j = np.asarray(jax.jacfwd(predict_flat)(flat_params), dtype=np.float64)  # (L*d_output, n_params)

    # Eigenvalues of the Gauss-Newton matrix J^T J are singular_values(J)**2.
    # Going through SVD of J directly (rather than eigvalsh on the explicitly
    # formed J^T J) avoids computing near-zero eigenvalues as spurious small
    # negatives - singular values are non-negative by construction, and SVD
    # is the numerically stable way to get them for a matrix this
    # ill-conditioned.
    singular_values = np.linalg.svd(j, compute_uv=False)
    eigvals_sorted = np.sort(singular_values**2)

    # Numerical rank: singular values below max_sv * dim_max * eps are
    # indistinguishable from zero at float64 precision (same convention as
    # np.linalg.matrix_rank).
    eps = np.finfo(np.float64).eps
    sv_tol = singular_values.max() * max(j.shape) * eps
    rank = int(np.sum(singular_values > sv_tol))

    print(f"[{label}] n_params={n_params}, J shape={tuple(j.shape)}")
    print(f"[{label}] bottom 5 eigenvalues (of J^T J, via SVD): {eigvals_sorted[:5]}")
    print(f"[{label}] top 5 eigenvalues (of J^T J, via SVD):    {eigvals_sorted[-5:]}")
    print(f"[{label}] numerical rank: {rank} / {n_params} (tol={sv_tol:.3e})")
    if rank < n_params:
        # Rank-deficient: condition number is formally infinite (there are
        # exact-zero curvature directions). Report it as such rather than
        # dividing by a numerical-noise floor.
        smallest_nonzero = eigvals_sorted[eigvals_sorted > sv_tol**2]
        cond_nonzero = eigvals_sorted[-1] / smallest_nonzero.min() if smallest_nonzero.size else float("nan")
        print(f"[{label}] condition number: inf (rank-deficient by {n_params - rank})")
        print(f"[{label}] condition number restricted to numerically-nonzero eigenvalues: {cond_nonzero:.6e}")
    else:
        cond = eigvals_sorted[-1] / eigvals_sorted[0]
        print(f"[{label}] condition number: {cond:.6e}")


def main() -> None:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    inputs, targets = case_data(CASE, L_MAX, aprbs_low=-10.0, aprbs_high=10.0)

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )

    _gauss_newton_report("init", model, inputs)

    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY), wrt=nnx.Param)

    def loss_fn(m):
        pred = _d_only_forward_nnx(m, inputs)
        return jnp.mean((pred - targets) ** 2)

    for _ in range(EPOCHS):
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

    final_mse = float(loss_fn(model))
    print(f"[step {EPOCHS}] mse after training: {final_mse:.6e}")
    _gauss_newton_report(f"step {EPOCHS}", model, inputs)


if __name__ == "__main__":
    main()
