"""Load parent-repo (s4dpc) trained checkpoints as decode=True StackedModel
instances, for read-only analysis. Does not modify or fork s4dpc/s4-nnx -
only imports BlockConfig/VARIANTS/StackedModel/D_INPUT/D_OUTPUT as-is,
same pattern as tools/diagnose_all_cases.py's _load_trained /
tools/characterize_n_unstable.py's load_model_step in the parent repo.

Checkpoints are read from wherever the caller points `ckpt_dir` - this
sub-project does not own or write to any parent-repo checkpoint directory.
"""
from __future__ import annotations

import json
import pathlib

import jax
import flax.serialization as serialization
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.model import StackedModel


def load_sidecar(ckpt_dir: pathlib.Path, variant: str, case: int, seed: int) -> dict:
    path = ckpt_dir / f"{variant}_case{case}_seed{seed}.json"
    return json.loads(path.read_text())


def load_model(
    ckpt_dir: pathlib.Path,
    variant: str,
    case: int,
    seed: int,
    *,
    decode: bool = True,
) -> tuple[StackedModel, dict]:
    """Returns (model, sidecar_config). d_model/N/n_layers/l_max come from
    the checkpoint's own .json sidecar (written by s4dpc.identify.save_checkpoint)
    rather than being hardcoded here, so this loader tracks whatever config
    each checkpoint was actually trained with."""
    sidecar = load_sidecar(ckpt_dir, variant, case, seed)
    cfg = sidecar["config"]

    block_config = BlockConfig(
        d_model=cfg["d_model"], N=cfg["N"], l_max=cfg["l_max"], **VARIANTS[variant]
    )
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    model = StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=cfg["n_layers"],
        decode=decode,
        rngs=nnx.Rngs(params=key),
    )

    msgpack_path = ckpt_dir / f"{variant}_case{case}_seed{seed}.msgpack"
    pure_dict = serialization.msgpack_restore(msgpack_path.read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)

    return model, sidecar


def available_checkpoints(ckpt_dir: pathlib.Path, variant: str) -> list[tuple[int, int]]:
    """(case, seed) pairs with both a .msgpack and .json present for `variant`."""
    pairs = []
    for msgpack_path in sorted(ckpt_dir.glob(f"{variant}_case*_seed*.msgpack")):
        stem = msgpack_path.stem  # e.g. "M5_case3_seed1"
        rest = stem[len(variant) + 1 :]  # "case3_seed1"
        case_str, seed_str = rest.split("_seed")
        case, seed = int(case_str[len("case") :]), int(seed_str)
        if (ckpt_dir / f"{stem}.json").exists():
            pairs.append((case, seed))
    return pairs
