"""Generates tests/fixtures/reference_model.msgpack: the parity target for
CLAUDE.md §5 ("the refactored code ... must reproduce the original notebook
code exactly"). Trains the LEGACY model (legacy/s4.py via legacy/_shim.py),
not s4-nnx — this checkpoint is what the s4-nnx port has to reproduce.

Not well-trained, just reproducible: fixed seeds throughout, no wall-clock
or unseeded randomness anywhere, so re-running this on the same backend
reproduces the same msgpack bytes and the same sidecar digests.

    python tools/make_reference_checkpoint.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flax
import jax
import jax.numpy as jnp
import numpy
import optax
from flax import nnx
import flax.serialization as serialization

from legacy import S4LayerEnsemble, StackedModelRegression
from s4dpc.systems import get_discrete_system

SEED = 0
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
D_INPUT, D_OUTPUT = 9, 6
CASE = 3
N_STEPS = 100
LEARNING_RATE = 1e-3

REPO_ROOT = _REPO_ROOT
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "reference_model.msgpack"
SIDECAR_PATH = FIXTURE_PATH.with_suffix(".json")


def _stringify_keys(x):
    """to_pure_dict() renders list-typed submodule containers (self.layers)
    as dicts with int keys; msgpack_restore rejects those (strict_map_key),
    so keys are stringified before serializing. replace_by_pure_dict on load
    matches by structure, not key type, so string keys round-trip fine."""
    if isinstance(x, dict):
        return {str(k): _stringify_keys(v) for k, v in x.items()}
    return x


def _build_model(decode: bool, seed: int = SEED) -> StackedModelRegression:
    return StackedModelRegression(
        layer_cls=S4LayerEnsemble,
        layer_args={"N": STATE_SIZE, "l_max": L_MAX},
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        decode=decode,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(seed)),
    )


def _make_case_data(case: int, seed: int):
    """(state, control) -> next-state data from `case`, generated directly
    via get_discrete_system rather than a ported Datasets registry.
    Uniform-random u: simplest deterministic choice, not APRBS."""
    a_d, b_d, name = get_discrete_system(case)
    a_d, b_d = jnp.asarray(a_d), jnp.asarray(b_d)
    state_dim, input_dim = b_d.shape[0], b_d.shape[1]

    key_x0, key_u = jax.random.split(jax.random.PRNGKey(seed))
    x0 = jax.random.uniform(key_x0, (state_dim,), minval=-1.0, maxval=1.0)
    u = jax.random.uniform(key_u, (L_MAX, input_dim), minval=-1.0, maxval=1.0)

    def step(x, u_t):
        x_next = a_d @ x + b_d @ u_t
        return x_next, (x, x_next)

    _, (states, next_states) = jax.lax.scan(step, x0, u)
    inputs = jnp.concatenate([states, u], axis=-1)  # (L_MAX, state_dim + input_dim)
    targets = next_states  # (L_MAX, state_dim)
    return inputs, targets, name


def _train(model: StackedModelRegression, inputs, targets, n_steps: int) -> list[float]:
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE), wrt=nnx.Param)
    states = model.init_state(N=STATE_SIZE)

    def loss_fn(m):
        pred, _ = m(inputs, states=states, training=False)
        return jnp.mean((pred - targets) ** 2)

    losses = []
    for _ in range(n_steps):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        losses.append(float(loss))
    return losses


def _forward_digest_cnn(model: StackedModelRegression, inputs) -> str:
    states = model.init_state(N=STATE_SIZE)
    out, _ = model(inputs, states=states, training=False)
    return hashlib.sha256(jax.device_get(out).tobytes()).hexdigest()


def _forward_digest_rnn(trained_state, inputs) -> str:
    # decode is fixed at construction (legacy/s4-nnx both), so the stepped
    # pass needs its own instance; same seed => same init params, then the
    # trained values are copied in directly (in-memory, no msgpack needed).
    model_rnn = _build_model(decode=True, seed=SEED)
    nnx.update(model_rnn, trained_state)

    states = model_rnn.init_state(N=STATE_SIZE)
    outputs = []
    for t in range(inputs.shape[0]):
        step_out, states = model_rnn(inputs[t], states=states, training=False)
        outputs.append(step_out)
    out = jax.device_get(jnp.stack(outputs, axis=0))
    return hashlib.sha256(out.tobytes()).hexdigest()


def main() -> None:
    model = _build_model(decode=False, seed=SEED)
    inputs, targets, system_name = _make_case_data(CASE, seed=SEED)

    losses = _train(model, inputs, targets, N_STEPS)

    param_state = nnx.state(model, nnx.Param)
    param_tree_sha = hashlib.sha256(
        str(jax.tree_util.tree_structure(param_state)).encode()
    ).hexdigest()
    param_count = int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(param_state)))

    fwd_digest_cnn = _forward_digest_cnn(model, inputs)
    fwd_digest_rnn = _forward_digest_rnn(param_state, inputs)

    pure_dict = _stringify_keys(param_state.to_pure_dict())
    msgpack_bytes = serialization.msgpack_serialize(pure_dict)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(msgpack_bytes)

    sidecar = {
        "config": {
            "seed": SEED,
            "d_model": D_MODEL,
            "N": STATE_SIZE,
            "n_layers": N_LAYERS,
            "l_max": L_MAX,
            "d_input": D_INPUT,
            "d_output": D_OUTPUT,
            "case": CASE,
            "system_name": system_name,
            "n_steps": N_STEPS,
            "learning_rate": LEARNING_RATE,
        },
        "versions": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": numpy.__version__,
        },
        "param_tree_sha": param_tree_sha,
        "param_count": param_count,
        "fwd_digest_cnn": fwd_digest_cnn,
        "fwd_digest_rnn": fwd_digest_rnn,
        "final_loss": losses[-1],
        "msgpack_sha256": hashlib.sha256(msgpack_bytes).hexdigest(),
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2))

    print(json.dumps(sidecar, indent=2))
    print(f"wrote {FIXTURE_PATH} ({len(msgpack_bytes)} bytes)")


if __name__ == "__main__":
    main()
