"""CLAUDE.md §5 parity testing.

L1: load tests/fixtures/reference_model.msgpack into a model; same input
    -> identical output. Checked as sha256 digest equality, not rtol/atol -
    any single differing bit produces a completely different hash, which
    is strictly tighter than rtol=0/atol=0 would be.
L2: identical gradients (same digest mechanism).
L3: both decode=False (conv) and decode=True (stepped).

This file currently only exercises the LEGACY model against its own
checkpoint (the checkpoint was generated FROM this exact legacy
model/data/config, so loading it back and re-deriving the same outputs is
a self-consistency check of the fixture, not yet a port comparison). That
is deliberate: this must be green before the s4-nnx port lands, so any
future red is caused by the port, not by a broken or drifted fixture.

Skips loudly (not silently) if the current jax backend differs from the
one the fixture was generated on: float reduction order is not guaranteed
identical across backends (CLAUDE.md §4 rule 4), so a cross-backend
"failure" here would be meaningless noise, not a real parity break.

Same reasoning forces single-threaded CPU execution below, before jax is
imported: XLA:CPU's multi-threaded reduction order isn't pinned across
container instances either (see tools/make_reference_checkpoint.py's
docstring for how this was actually observed - two Kaggle CPU sessions on
identical code+data gave different digests), so without this, this test
could fail on a real machine for a reason that has nothing to do with the
port being tested.
"""
from __future__ import annotations

import os

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_cpu_multi_thread_eigen=false").strip()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
import pathlib

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
import flax.serialization as serialization

from legacy import S4LayerEnsemble, StackedModelRegression
from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.model import StackedModel

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
MSGPACK_PATH = FIXTURE_DIR / "reference_model.msgpack"
SIDECAR_PATH = FIXTURE_DIR / "reference_model.json"


@pytest.fixture(scope="module")
def sidecar() -> dict:
    return json.loads(SIDECAR_PATH.read_text())


def _build_legacy(decode: bool, cfg: dict) -> StackedModelRegression:
    return StackedModelRegression(
        layer_cls=S4LayerEnsemble,
        layer_args={"N": cfg["N"], "l_max": cfg["l_max"]},
        d_input=cfg["d_input"],
        d_output=cfg["d_output"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        decode=decode,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(cfg["seed"])),
    )


@pytest.fixture(scope="module")
def loaded_legacy_model(sidecar):
    current_backend = jax.default_backend()
    if current_backend != sidecar["jax_backend"]:
        pytest.skip(
            "LOUD SKIP: tests/fixtures/reference_model.msgpack was generated "
            f"on jax backend {sidecar['jax_backend']!r}, but this process is "
            f"running on {current_backend!r}. Float reduction order is not "
            "guaranteed identical across backends (CLAUDE.md §4 rule 4), so "
            f"comparing here would be meaningless - re-run on "
            f"{sidecar['jax_backend']!r} to actually exercise parity."
        )

    cfg = sidecar["config"]
    model = _build_legacy(decode=False, cfg=cfg)

    state = nnx.state(model, nnx.Param)
    pure_dict = serialization.msgpack_restore(MSGPACK_PATH.read_bytes())
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)

    return model, cfg


@pytest.fixture(scope="module")
def case_data(loaded_legacy_model):
    _, cfg = loaded_legacy_model
    batch_inputs, batch_targets = generate_microgrid_trajectory(
        batch_size=1,
        length=cfg["l_max"],
        seed=cfg["seed"],
        system_case=cfg["case"],
        dt=cfg["dt"],
    )
    return jnp.asarray(batch_inputs[0]), jnp.asarray(batch_targets[0])


def test_param_tree_sha_matches(loaded_legacy_model, sidecar):
    model, _ = loaded_legacy_model
    state = nnx.state(model, nnx.Param)
    tree_sha = hashlib.sha256(str(jax.tree_util.tree_structure(state)).encode()).hexdigest()
    assert tree_sha == sidecar["param_tree_sha"]


def test_fwd_digest_cnn_matches(loaded_legacy_model, case_data, sidecar):
    """L1 + L3 (decode=False)."""
    model, cfg = loaded_legacy_model
    inputs, _ = case_data
    states = model.init_state(N=cfg["N"])
    out, _ = model(inputs, states=states, training=False)
    digest = hashlib.sha256(jax.device_get(out).tobytes()).hexdigest()
    assert digest == sidecar["fwd_digest_cnn"]


def test_fwd_digest_rnn_matches(loaded_legacy_model, case_data, sidecar):
    """L1 + L3 (decode=True). decode is fixed at construction (legacy and
    s4-nnx both), so the stepped pass needs its own instance; the trained
    params are copied in directly (in-memory, no msgpack round-trip)."""
    model, cfg = loaded_legacy_model
    inputs, _ = case_data

    model_rnn = _build_legacy(decode=True, cfg=cfg)
    nnx.update(model_rnn, nnx.state(model, nnx.Param))

    states = model_rnn.init_state(N=cfg["N"])
    outputs = []
    for t in range(inputs.shape[0]):
        step_out, states = model_rnn(inputs[t], states=states, training=False)
        outputs.append(step_out)
    out = jax.device_get(jnp.stack(outputs, axis=0))
    digest = hashlib.sha256(out.tobytes()).hexdigest()
    assert digest == sidecar["fwd_digest_rnn"]


def test_grad_digest_matches(loaded_legacy_model, case_data, sidecar):
    """L2. Gradients of the same case-3 MSE loss w.r.t. the loaded (trained)
    params, computed but never applied - matches how the sidecar's
    grad_digest was produced (tools/make_reference_checkpoint.py)."""
    model, cfg = loaded_legacy_model
    inputs, targets = case_data
    states = model.init_state(N=cfg["N"])

    def loss_fn(m):
        pred, _ = m(inputs, states=states, training=False)
        return jnp.mean((pred - targets) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(model)
    digest = hashlib.sha256(
        b"".join(jax.device_get(g).tobytes() for g in jax.tree_util.tree_leaves(grads))
    ).hexdigest()
    assert digest == sidecar["grad_digest"]


# ---------------------------------------------------------------------------
# M6 (s4dpc.model, consuming s4-nnx) vs. legacy: the actual port comparison.
# M6's BlockConfig (norm="layer", activation="gelu", glu=True, prenorm=True,
# residual=True) is designed to be architecturally identical to legacy's
# SequenceBlockNNX, with s4dpc.model.StackedModel's RNG key-splitting order
# matching legacy.StackedModelRegression exactly (encoder key, decoder key,
# one fresh nnx.Rngs per layer; within each block, the S4 layer consumes
# rngs.params() first, then norm/out/out2 consume a second split) - so a
# same-seed M6 build should reproduce legacy's params bit-exactly, with no
# key remapping. Checked against the SAME sidecar digests the legacy tests
# above use (already proven equal to legacy's own outputs), which is
# equivalent to and simpler than re-deriving legacy's outputs a second time.
# ---------------------------------------------------------------------------


def _build_m6(decode: bool, cfg: dict) -> StackedModel:
    block_config = BlockConfig(d_model=cfg["d_model"], N=cfg["N"], l_max=cfg["l_max"], **VARIANTS["M6"])
    return StackedModel(
        block_config=block_config,
        d_input=cfg["d_input"],
        d_output=cfg["d_output"],
        n_layers=cfg["n_layers"],
        decode=decode,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(cfg["seed"])),
    )


@pytest.fixture(scope="module")
def loaded_m6_model(sidecar):
    current_backend = jax.default_backend()
    if current_backend != sidecar["jax_backend"]:
        pytest.skip(
            "LOUD SKIP: tests/fixtures/reference_model.msgpack was generated "
            f"on jax backend {sidecar['jax_backend']!r}, but this process is "
            f"running on {current_backend!r}. See loaded_legacy_model's skip "
            "reason above; same logic applies here."
        )

    cfg = sidecar["config"]
    model = _build_m6(decode=False, cfg=cfg)

    state = nnx.state(model, nnx.Param)
    pure_dict = serialization.msgpack_restore(MSGPACK_PATH.read_bytes())
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)

    return model, cfg


def test_m6_param_tree_sha_matches_legacy(loaded_m6_model, sidecar):
    model, _ = loaded_m6_model
    state = nnx.state(model, nnx.Param)
    tree_sha = hashlib.sha256(str(jax.tree_util.tree_structure(state)).encode()).hexdigest()
    assert tree_sha == sidecar["param_tree_sha"]


def test_m6_init_params_match_legacy(sidecar):
    """Same seed, before any checkpoint is loaded: M6 and legacy should
    already agree at initialization, which is the strongest possible
    evidence that the RNG key-splitting order was replicated correctly
    (not just that loading the checkpoint papers over a difference)."""
    cfg = sidecar["config"]
    m6 = _build_m6(decode=False, cfg=cfg)
    legacy = _build_legacy(decode=False, cfg=cfg)

    m6_leaves = jax.tree_util.tree_leaves(nnx.state(m6, nnx.Param))
    legacy_leaves = jax.tree_util.tree_leaves(nnx.state(legacy, nnx.Param))
    assert len(m6_leaves) == len(legacy_leaves)
    for a, b in zip(m6_leaves, legacy_leaves):
        assert a.shape == b.shape
        assert bool(jnp.array_equal(a, b))


def test_m6_fwd_digest_cnn_matches_legacy(loaded_m6_model, case_data, sidecar):
    model, cfg = loaded_m6_model
    inputs, _ = case_data
    states = model.init_state(N=cfg["N"])
    out, _ = model(inputs, states)
    digest = hashlib.sha256(jax.device_get(out).tobytes()).hexdigest()
    assert digest == sidecar["fwd_digest_cnn"]


def test_m6_fwd_digest_rnn_matches_legacy(loaded_m6_model, case_data, sidecar):
    model, cfg = loaded_m6_model
    inputs, _ = case_data

    model_rnn = _build_m6(decode=True, cfg=cfg)
    nnx.update(model_rnn, nnx.state(model, nnx.Param))

    states = model_rnn.init_state(N=cfg["N"])
    outputs = []
    for t in range(inputs.shape[0]):
        step_out, states = model_rnn(inputs[t], states)
        outputs.append(step_out)
    out = jax.device_get(jnp.stack(outputs, axis=0))
    digest = hashlib.sha256(out.tobytes()).hexdigest()
    assert digest == sidecar["fwd_digest_rnn"]


def test_m6_grad_digest_matches_legacy(loaded_m6_model, case_data, sidecar):
    model, cfg = loaded_m6_model
    inputs, targets = case_data
    states = model.init_state(N=cfg["N"])

    def loss_fn(m):
        pred, _ = m(inputs, states)
        return jnp.mean((pred - targets) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(model)
    digest = hashlib.sha256(
        b"".join(jax.device_get(g).tobytes() for g in jax.tree_util.tree_leaves(grads))
    ).hexdigest()
    assert digest == sidecar["grad_digest"]
