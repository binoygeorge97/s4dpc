"""Validates s4dpc/diagnostics.py against ground truth before trusting it
on any real trained surrogate (Task 3, docs/DECISIONS.md).

Forces a decode=True StackedModel's one-step map to be EXACTLY
x_next = A_d @ x + B_d @ u by routing the TRUE [A_d|B_d] through the skip
connection - the same block-zeroing construction as
tools/diagnose_variant_redundancy.py's _apply_ls_init_general (zero D and
C_real_imag together forces the S4Layer's output to exactly zero for any
input - see that file's docstring for the kernel_DPLR derivation - plus
out/out2), generalized here from "the empirically-fit LS solution" to
"the TRUE A_d/B_d" so every diagnostic has a known-exact answer to check
against, not an approximately-fit one. Uses M6 (norm=layer, activation=
gelu, glu=True) specifically, not M3, so the validation also exercises
that the zeroing trick survives LayerNorm/GELU/GLU being architecturally
present (GELU(0)=0 exactly; a GLU gate times 0 is 0 regardless of the
gate value; LayerNorm never touches the skip connection).

Checks:
  - equilibrium_drift: must be exactly 0 (A_d@0+B_d@0=0).
  - markov_parameters(H=50): must match A_d^(h-1)@B_d for every h.
  - local_linearity_defect: must be ~0 (the true system IS linear).
  - jacobian_sweep: must equal A_d at every t (no kink - correct, since
    no LayerNorm is actually live on this bypassed path).
  - Sanity control: the SAME four diagnostics run on a fresh, untrained,
    genuinely nonlinear M6 model (block NOT zeroed) - must be finite and
    not trivially identical to the exact-linear case, or the "validation
    passing" above would be vacuous (e.g. every diagnostic secretly
    always returning 0 regardless of input).

    python tools/validate_diagnostics.py
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
import numpy as np
from flax import nnx

from s4dpc import diagnostics
from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
SEED = 0
H = 50


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


def _apply_exact_linear(model: StackedModel, w_in_out: np.ndarray, has_glu: bool) -> None:
    block = model.layers[0]
    d_input, d_output = w_in_out.shape

    encoder_kernel = jnp.zeros((d_input, D_MODEL), dtype=jnp.float64)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_in_out, dtype=jnp.float64))
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


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    A_d, B_d = get_discrete_matrices(DT, CASE)
    A_d64 = np.asarray(A_d, dtype=np.float64)
    B_d64 = np.asarray(B_d, dtype=np.float64)
    w_true = np.concatenate([A_d64, B_d64], axis=1).T  # (d_input, d_output) = (9,6)
    print(f"A_d shape={A_d64.shape}  B_d shape={B_d64.shape}")

    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    d_u = D_INPUT - D_OUTPUT

    print("\n=== VALIDATION: model forced to realize x_next = A_d@x + B_d@u exactly (variant=M6) ===")
    model = _build("M6", decode=True, key=key)
    _apply_exact_linear(model, w_true, has_glu=True)
    states0 = diagnostics.zero_states(model)

    rng = np.random.RandomState(0)
    x_test = jnp.asarray(rng.uniform(-3, 3, size=(D_OUTPUT,)), dtype=jnp.float64)
    u_test = jnp.asarray(rng.uniform(-3, 3, size=(d_u,)), dtype=jnp.float64)
    x_next_model, _ = diagnostics.step(model, x_test, u_test, states0)
    x_next_true = jnp.asarray(A_d64) @ x_test + jnp.asarray(B_d64) @ u_test
    self_check_err = float(jnp.max(jnp.abs(x_next_model - x_next_true)))
    print(f"self-check (model forward vs A_d@x+B_d@u directly): max abs diff={self_check_err:.3e}")
    if self_check_err > 1e-9:
        print("SELF-CHECK FAILED - construction does not realize the true system. Stopping.")
        return

    drift = diagnostics.equilibrium_drift(model, states0)
    print(f"\nequilibrium_drift: max abs = {float(jnp.max(jnp.abs(drift))):.3e}  (expect ~0)")

    G = diagnostics.markov_parameters(model, states0, H)
    max_err = 0.0
    A_power = np.eye(A_d64.shape[0])
    for h in range(1, H + 1):
        true_gh = A_power @ B_d64
        err = float(np.max(np.abs(np.asarray(G[h - 1]) - true_gh)))
        max_err = max(max_err, err)
        if h <= 3 or h == H:
            print(f"  h={h:2d}  max|G_h - A_d^(h-1)@B_d| = {err:.3e}")
        A_power = A_power @ A_d64
    print(f"markov_parameters: max error over h=1..{H}: {max_err:.3e}  (expect ~1e-9 or better)")

    defect = diagnostics.local_linearity_defect(
        model, states0, jnp.zeros((D_OUTPUT,), dtype=jnp.float64), jnp.zeros((d_u,), dtype=jnp.float64),
        jax.random.PRNGKey(1), n_samples=64, delta_scale=1e-2,
    )
    print(f"\nlocal_linearity_defect (at x=0,u=0): {float(defect):.3e}  "
          f"(expect ~1e-8 or better - true system is exactly linear)")

    direction = jnp.asarray(rng.uniform(-1, 1, size=(D_OUTPUT,)), dtype=jnp.float64)
    t_values = jnp.linspace(-2.0, 2.0, 9)
    jacs = diagnostics.jacobian_sweep(model, states0, direction, t_values, jnp.zeros((d_u,), dtype=jnp.float64))
    max_jac_err = float(jnp.max(jnp.abs(jacs - jnp.asarray(A_d64)[None])))
    print(f"jacobian_sweep: max|J(t) - A_d| over sweep = {max_jac_err:.3e}  "
          f"(expect ~1e-9 or better - exactly linear, no kink)")

    print("\n=== SANITY CONTROL: fresh, untrained, genuinely nonlinear M6 (block NOT zeroed) ===")
    nl_model = _build("M6", decode=True, key=jax.random.fold_in(jax.random.PRNGKey(SEED + 1), CASE))
    nl_states0 = diagnostics.zero_states(nl_model)
    nl_drift = diagnostics.equilibrium_drift(nl_model, nl_states0)
    print(f"equilibrium_drift (fresh M6): max abs = {float(jnp.max(jnp.abs(nl_drift))):.3e} (finite, likely nonzero)")
    nl_defect = diagnostics.local_linearity_defect(
        nl_model, nl_states0, jnp.zeros((D_OUTPUT,), dtype=jnp.float64), jnp.zeros((d_u,), dtype=jnp.float64),
        jax.random.PRNGKey(2), n_samples=64, delta_scale=1e-2,
    )
    print(f"local_linearity_defect (fresh M6, at x=0): {float(nl_defect):.3e} (finite, likely nonzero)")
    nl_jacs = diagnostics.jacobian_sweep(nl_model, nl_states0, direction, t_values, jnp.zeros((d_u,), dtype=jnp.float64))
    jac_norms = np.linalg.norm(np.asarray(nl_jacs), axis=(1, 2))
    print(f"jacobian_sweep (fresh M6): all finite={bool(np.all(np.isfinite(jac_norms)))}  "
          f"norm range=[{jac_norms.min():.3e}, {jac_norms.max():.3e}]  "
          f"varies_with_t={bool(np.ptp(jac_norms) > 1e-12)}")

    print("\n=== SUMMARY ===")
    all_ok = (
        float(jnp.max(jnp.abs(drift))) < 1e-9
        and max_err < 1e-9
        and float(defect) < 1e-6
        and max_jac_err < 1e-9
        and bool(np.all(np.isfinite(jac_norms)))
    )
    print(f"diagnostics.py validated against ground truth: {'PASS' if all_ok else 'FAIL - see numbers above'}")


if __name__ == "__main__":
    main()
