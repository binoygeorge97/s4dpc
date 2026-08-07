"""Generates tests/fixtures/reference_model.msgpack: the parity target for
CLAUDE.md §5 ("the refactored code ... must reproduce the original notebook
code exactly"). Trains the LEGACY model (legacy/s4.py via legacy/_shim.py),
not s4-nnx — this checkpoint is what the s4-nnx port has to reproduce.

Real case-3 data (s4dpc.data.generate_microgrid_trajectory, built on the
real s4dpc.systems.get_discrete_matrices - no more placeholder systems),
at the canonical bilinear/Tustin discretization, dt=0.01 (see DT below and
tests/test_systems.py for why that one, not zoh@0.02).

Not well-trained, just reproducible: fixed seeds throughout, no wall-clock
or unseeded randomness anywhere, so re-running this on the same backend
reproduces the same msgpack bytes and the same sidecar digests.

The sidecar's jax_backend records what backend this was generated on
(tests/test_parity.py skips loudly rather than compare cross-backend,
since float reduction order isn't guaranteed identical across backends -
see CLAUDE.md §4 rule 4).

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
from s4dpc.data import generate_microgrid_trajectory

SEED = 0
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
D_INPUT, D_OUTPUT = 9, 6
CASE = 3
N_STEPS = 100
LEARNING_RATE = 1e-3
# Canonical discretization: bilinear/Tustin, not zoh. Settled empirically in
# tests/test_systems.py - only tustin@0.01 reproduces the known rho(A_d) for
# case 3 (~1.02019) and case 6 (~1.034); zoh@0.02 gives 1.04081/1.06912.
DT = 0.01

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
    """Real case data via the ported pipeline (s4dpc.data.
    generate_microgrid_trajectory, itself built on the canonical
    s4dpc.systems.get_discrete_matrices), batch_size=1 then squeezed since
    the legacy model takes one unbatched sequence at a time."""
    batch_inputs, batch_targets = generate_microgrid_trajectory(
        batch_size=1, length=L_MAX, seed=seed, system_case=case, dt=DT,
    )
    inputs = jnp.asarray(batch_inputs[0])
    targets = jnp.asarray(batch_targets[0])
    return inputs, targets


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


def _grad_digest(model: StackedModelRegression, inputs, targets) -> str:
    """Gradients of the same loss/data used for training, w.r.t. the
    TRAINED (post-N_STEPS) params - one more step's worth, computed but
    never applied (no optimizer.update call), so this doesn't perturb
    fwd_digest_cnn/rnn's model state."""
    states = model.init_state(N=STATE_SIZE)

    def loss_fn(m):
        pred, _ = m(inputs, states=states, training=False)
        return jnp.mean((pred - targets) ** 2)

    _, grads = nnx.value_and_grad(loss_fn)(model)
    return hashlib.sha256(
        b"".join(jax.device_get(g).tobytes() for g in jax.tree_util.tree_leaves(grads))
    ).hexdigest()


def main() -> None:
    model = _build_model(decode=False, seed=SEED)
    inputs, targets = _make_case_data(CASE, seed=SEED)

    losses = _train(model, inputs, targets, N_STEPS)

    param_state = nnx.state(model, nnx.Param)
    param_tree_sha = hashlib.sha256(
        str(jax.tree_util.tree_structure(param_state)).encode()
    ).hexdigest()
    param_count = int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(param_state)))

    fwd_digest_cnn = _forward_digest_cnn(model, inputs)
    fwd_digest_rnn = _forward_digest_rnn(param_state, inputs)
    grad_digest = _grad_digest(model, inputs, targets)

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
            "dt": DT,
            "discretization": "bilinear_tustin",
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
        "jax_backend": jax.default_backend(),
        "param_tree_sha": param_tree_sha,
        "param_count": param_count,
        "fwd_digest_cnn": fwd_digest_cnn,
        "fwd_digest_rnn": fwd_digest_rnn,
        "grad_digest": grad_digest,
        "final_loss": losses[-1],
        "msgpack_sha256": hashlib.sha256(msgpack_bytes).hexdigest(),
    }
    SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2))

    print(json.dumps(sidecar, indent=2))
    print(f"wrote {FIXTURE_PATH} ({len(msgpack_bytes)} bytes)")


if __name__ == "__main__":
    main()
