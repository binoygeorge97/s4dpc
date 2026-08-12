"""One-off probe (not a standing diagnostic): does S4LayerEnsemble (s4-nnx
v0.2.0, pinned per CLAUDE.md §7) accept a param_dtype kwarg, and what
dtype do nnx.Linear/nnx.LayerNorm/S4LayerEnsemble actually construct
their params as by default? Answers this BEFORE threading a param_dtype
option through s4dpc/model.py + s4dpc/blocks.py for the float64-default
work (docs/DECISIONS.md) - guessing wrong here would silently produce
"float64" identification runs whose weights are still float32-precision,
exactly the bug tools/diagnose_m3_rank_x64.py already found once for
nnx.Linear specifically.

    python tools/probe_s4nnx_dtype.py
"""
from __future__ import annotations

import inspect
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from flax import nnx
from s4_nnx import S4LayerEnsemble


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    print("\n[S4LayerEnsemble.__init__ signature]")
    print(inspect.signature(S4LayerEnsemble.__init__))

    print("\n[does S4LayerEnsemble accept param_dtype?]")
    try:
        layer = S4LayerEnsemble(
            N=8, l_max=10, D_MODEL=4, decode=False, param_dtype=jnp.float64,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
        )
        print("  YES - accepted param_dtype=jnp.float64 without error")
        leaves = jax.tree_util.tree_leaves(nnx.state(layer, nnx.Param))
        dtypes = sorted({str(x.dtype) for x in leaves})
        print(f"  resulting param dtypes: {dtypes}")
    except TypeError as e:
        print(f"  NO - TypeError: {e}")

    print("\n[default dtype with no param_dtype passed, x64 already True]")
    layer_default = S4LayerEnsemble(
        N=8, l_max=10, D_MODEL=4, decode=False,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    )
    leaves = jax.tree_util.tree_leaves(nnx.state(layer_default, nnx.Param))
    dtypes = sorted({str(x.dtype) for x in leaves})
    print(f"  resulting param dtypes: {dtypes}")

    print("\n[for comparison: nnx.Linear default dtype, x64 already True]")
    lin = nnx.Linear(4, 4, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    print(f"  kernel dtype: {lin.kernel.value.dtype}")

    print("\n[nnx.Linear WITH explicit param_dtype=jnp.float64]")
    lin64 = nnx.Linear(4, 4, param_dtype=jnp.float64, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    print(f"  kernel dtype: {lin64.kernel.value.dtype}")

    print("\n[nnx.LayerNorm default vs explicit param_dtype]")
    ln = nnx.LayerNorm(4, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    print(f"  default scale dtype: {ln.scale.value.dtype}")
    ln64 = nnx.LayerNorm(4, param_dtype=jnp.float64, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    print(f"  explicit param_dtype=float64 scale dtype: {ln64.scale.value.dtype}")


if __name__ == "__main__":
    main()
