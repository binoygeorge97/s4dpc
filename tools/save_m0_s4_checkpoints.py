"""TASK D (user, 2026-08-19, seventh round): M0_S4 has been used as a
control throughout this session but was never saved as a loadable
checkpoint - only its EXTRACTED (Abar,Bbar) matrices exist on disk
(docs/nu_gap_export/*.npz). Since it is a deterministic hand-construction
(block-zeroed weights, tools/controller_m0_s4.py's _build_m0_s4 - no
gradient descent, no training), it can and should be saved directly,
with the same git_sha/lockfile_sha stamping convention
s4dpc/identify.py's own save_checkpoint uses for every other checkpoint
in this repo.

    python tools/save_m0_s4_checkpoints.py
"""
from __future__ import annotations

import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from flax import nnx
import flax.serialization as serialization

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.logging import get_git_sha, get_lockfile_sha
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
CKPT_DIR = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt"


def _stringify_keys(x):
    if isinstance(x, dict):
        return {str(k): _stringify_keys(v) for k, v in x.items()}
    return x


def build_m0_s4(case: int, seed: int) -> StackedModel:
    """EXACT copy of tools/controller_m0_s4.py's _build_m0_s4 - same
    construction, not a re-derivation, so this M0_S4 is bit-identical to
    the one every M0_S4 result this session already used."""
    A_d, B_d = get_discrete_matrices(DT, case)
    w_true = np.concatenate([np.asarray(A_d), np.asarray(B_d)], axis=1).T  # (9,6)

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M6"])
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    model = StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT,
                          n_layers=N_LAYERS, decode=True, rngs=nnx.Rngs(params=key))

    def _cast(x):
        return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)

    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(_cast, state)
    nnx.update(model, state)

    block = model.layers[0]
    d_input, d_output = w_true.shape

    encoder_kernel = jnp.zeros((d_input, D_MODEL), dtype=jnp.float64)
    encoder_kernel = encoder_kernel.at[:, :d_output].set(jnp.asarray(w_true, dtype=jnp.float64))
    model.encoder.kernel.value = encoder_kernel
    model.encoder.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    block.seq.D.value = jnp.zeros_like(block.seq.D.value, dtype=jnp.float64)
    block.seq.C_real_imag.value = jnp.zeros_like(block.seq.C_real_imag.value, dtype=jnp.float64)
    block.out.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)
    block.out.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)
    block.out2.kernel.value = jnp.zeros((D_MODEL, D_MODEL), dtype=jnp.float64)
    block.out2.bias.value = jnp.zeros((D_MODEL,), dtype=jnp.float64)

    decoder_kernel = jnp.zeros((D_MODEL, d_output), dtype=jnp.float64)
    decoder_kernel = decoder_kernel.at[:d_output, :].set(jnp.eye(d_output, dtype=jnp.float64))
    model.decoder.kernel.value = decoder_kernel
    model.decoder.bias.value = jnp.zeros((d_output,), dtype=jnp.float64)

    return model


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = get_git_sha()
    lockfile_sha = get_lockfile_sha()

    for case in CASES:
        for seed in range(N_SEEDS):
            model = build_m0_s4(case, seed)
            param_state = nnx.state(model, nnx.Param)
            pure_dict = _stringify_keys(param_state.to_pure_dict())
            msgpack_bytes = serialization.msgpack_serialize(pure_dict)
            stem = f"M0_S4_case{case}_seed{seed}"
            (CKPT_DIR / f"{stem}.msgpack").write_bytes(msgpack_bytes)

            sidecar = {
                "variant": "M0_S4", "case": case, "seed": seed,
                "teacher_mse": 0.0,  # exact by construction, not fit
                "config": {"d_model": D_MODEL, "N": STATE_SIZE, "n_layers": N_LAYERS, "l_max": L_MAX,
                           "construction": "hand-zeroed block, tools/controller_m0_s4.py._build_m0_s4 - "
                                            "deterministic, no training"},
                "git_sha": git_sha, "lockfile_sha": lockfile_sha,
            }
            (CKPT_DIR / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
            print(f"saved {stem}")

    print(f"\nwrote {len(CASES) * N_SEEDS} M0_S4 checkpoints to {CKPT_DIR}")


if __name__ == "__main__":
    main()
