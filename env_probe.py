"""Environment probe: package versions, jax device/backend, a fixed-seed
matmul digest, and an s4-nnx model canary. Run this before trusting any
result from a new machine.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys

_PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "pandas", "s4_nnx")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def _numerics_canary(seed: int = 0, n: int = 512) -> dict:
    """Fixed-seed matmul digest: a cheap fingerprint of this platform's XLA
    numerics. Not asserted against a reference — just recorded per run."""
    import jax
    import jax.numpy as jnp

    key_a, key_b = jax.random.split(jax.random.PRNGKey(seed))
    a = jax.random.normal(key_a, (n, n), dtype=jnp.float32)
    b = jax.random.normal(key_b, (n, n), dtype=jnp.float32)
    c = jax.device_get(a @ b)

    return {
        "seed": seed,
        "shape": [n, n],
        "dtype": "float32",
        "matmul_sha256": hashlib.sha256(c.tobytes()).hexdigest(),
        "matmul_sum": float(c.sum()),
    }


def _model_canary(seed: int = 0) -> dict:
    """s4-nnx checkpoint-portability canary (CLAUDE.md §5 L3): build the real
    S4 layer, hash its param tree, and check that conv-mode (decode=False)
    and step-mode (decode=True) forward passes agree. Uses the real,
    separately-pinned s4-nnx package — never vendored or modified here."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from s4_nnx import S4Config, create_model

    d_model, state_size, n_layers, l_max = 16, 32, 1, 100
    d_input, d_output = 9, 6

    def build(decode: bool):
        config = S4Config(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            state_size=state_size,
            l_max=l_max,
            decode=decode,
        )
        return create_model(config, seed=seed)

    # decode is fixed at construction time, so conv-mode and step-mode need
    # separate instances; same seed => identical params (decode does not
    # affect param initialization, only the __call__ control path).
    model_cnn = build(decode=False)
    model_rnn = build(decode=True)

    param_state = nnx.state(model_cnn, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(param_state)
    param_tree_sha = hashlib.sha256(
        str(jax.tree_util.tree_structure(param_state)).encode()
    ).hexdigest()
    param_count = int(sum(leaf.size for leaf in param_leaves))

    inputs = jax.random.normal(jax.random.PRNGKey(seed + 1000), (l_max, d_input))

    cnn_out, _ = model_cnn(inputs, training=False)
    cnn_out = jax.device_get(cnn_out)

    rnn_states = model_rnn.init_state()
    rnn_out_steps = []
    for t in range(l_max):
        step_out, rnn_states = model_rnn(inputs[t], states=rnn_states, training=False)
        rnn_out_steps.append(step_out)
    rnn_out = jax.device_get(jnp.stack(rnn_out_steps, axis=0))

    cnn_rnn_max_abs_diff = float(jnp.max(jnp.abs(cnn_out - rnn_out)))

    # one optax adamw step on the conv-mode instance, the package's own
    # documented training mode (examples/regression.py)
    targets = jax.random.normal(jax.random.PRNGKey(seed + 2000), (l_max, d_output))

    def loss_fn(m):
        pred, _ = m(inputs, training=False)
        return jnp.mean((pred - targets) ** 2)

    optimizer = nnx.Optimizer(model_cnn, optax.adamw(1e-3), wrt=nnx.Param)
    _, grads = nnx.value_and_grad(loss_fn)(model_cnn)
    grad_digest = hashlib.sha256(
        b"".join(jax.device_get(g).tobytes() for g in jax.tree_util.tree_leaves(grads))
    ).hexdigest()
    optimizer.update(model_cnn, grads)  # proves the training step runs; result unused

    return {
        "seed": seed,
        "config": {
            "d_model": d_model,
            "N": state_size,
            "n_layers": n_layers,
            "l_max": l_max,
            "d_input": d_input,
            "d_output": d_output,
        },
        "param_tree_sha": param_tree_sha,
        "param_count": param_count,
        "fwd_digest_cnn": hashlib.sha256(cnn_out.tobytes()).hexdigest(),
        "fwd_digest_rnn": hashlib.sha256(rnn_out.tobytes()).hexdigest(),
        "cnn_rnn_max_abs_diff": cnn_rnn_max_abs_diff,
        "grad_digest": grad_digest,
    }


def probe() -> dict:
    import jax

    devices = jax.devices()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in devices],
        "jax_device_count": len(devices),
        "numerics_canary": _numerics_canary(),
        "model_canary": _model_canary(),
    }


def main() -> None:
    print(json.dumps(probe(), indent=2))


if __name__ == "__main__":
    main()
