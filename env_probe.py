"""Environment probe: package versions, jax device/backend, and a fixed-seed
matmul digest. Run this before trusting any result from a new machine.

Model param-tree probing is intentionally omitted here — there is no model
yet (CLAUDE.md §1/§7 scope for this week).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys

_PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "pandas")


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
    }


def main() -> None:
    print(json.dumps(probe(), indent=2))


if __name__ == "__main__":
    main()
