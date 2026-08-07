"""Teacher-forced one-step MSE system identification, vmapped over
(case x seed). All cases 1-7 share A:(6,6)/B:(6,3) (s4dpc.systems), so any
subset of them fuses into one nnx.vmap call together with n_seeds. Data
comes from s4dpc.data: canonical bilinear/Tustin discretization at
dt=0.01 (tests/test_systems.py), APRBS range configurable (default
(-10, 10)) rather than a buried constant.

Cases 8/9 (different shapes) are out of scope here - see s4dpc/data.py.

The vmapped ensemble path (nnx.vmap for per-member model construction,
then nnx.split/jax.vmap/nnx.merge for the per-member forward pass, with
one shared nnx.Optimizer over the whole batched ensemble) was verified
against a plain per-member Python loop before being trusted: same
init-time params, same forward pass, same gradients, and - the part that
actually needed checking - the same loss trajectory step-by-step across
multiple training steps (not just after one step, where a scaling bug
could hide). --no-vmap runs literally the same per-member function
(_train_one) in a Python loop instead of through the ensemble machinery,
for readable tracebacks (CLAUDE.md §9); the two paths are checked against
each other in tests/test_identify.py.
"""
from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import flax.serialization as serialization

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.logging import get_git_sha, get_lockfile_sha
from s4dpc.model import StackedModel

D_INPUT, D_OUTPUT = 9, 6  # cases 1-7: state_dim(6) + control_dim(3) -> state_dim(6)
DATA_SEED = 42  # fixed per-case data; only model init varies with seed
DT = 0.01  # canonical bilinear/Tustin - see tests/test_systems.py


def case_data(case: int, l_max: int, aprbs_low: float, aprbs_high: float) -> tuple[jax.Array, jax.Array]:
    batch_inputs, batch_targets = generate_microgrid_trajectory(
        batch_size=1,
        length=l_max,
        seed=DATA_SEED,
        system_case=case,
        dt=DT,
        aprbs_low=aprbs_low,
        aprbs_high=aprbs_high,
    )
    return jnp.asarray(batch_inputs[0]), jnp.asarray(batch_targets[0])


def fit_least_squares(
    case: int, l_max: int = 100, aprbs_low: float = -10.0, aprbs_high: float = 10.0
) -> tuple[np.ndarray, float]:
    """M1: the capacity floor M3 must approach. Closed-form
    [A_hat B_hat] = Y Z^dagger on the SAME (state, control) -> next-state
    data the neural variants train on (case_data), via np.linalg.lstsq
    (numerically preferred over forming Z^dagger explicitly - same
    closed-form solution) in float64.

    float64, not JAX's default float32: the ~1e-14-level floor this is
    meant to establish is inherently a float64-precision number (float32
    machine epsilon alone is ~1e-7), so this is not an apples-to-apples
    precision comparison against the float32-trained neural variants -
    but the conclusion doesn't depend on which: either floor is orders
    of magnitude below where M3 currently sits.

    Deliberately NOT dpc_example's LinearDynamicsModel-with-LayerNorm
    weight extraction: extracting W/V from a LayerNorm'd model does not
    give an equivalent linear system (that file's own comment admits
    this - LayerNorm is a nonlinear op). See docs/DECISIONS.md.

    Returns ([A_hat | B_hat] as a (d_output, d_input) array, mse)."""
    inputs, targets = case_data(case, l_max, aprbs_low, aprbs_high)
    z = np.asarray(inputs, dtype=np.float64)  # (L, d_input)
    y = np.asarray(targets, dtype=np.float64)  # (L, d_output)

    ab_hat_t, _, _, _ = np.linalg.lstsq(z, y, rcond=None)  # (d_input, d_output)
    pred = z @ ab_hat_t
    mse = float(np.mean((pred - y) ** 2))
    return ab_hat_t.T, mse  # (d_output, d_input) = [A_hat | B_hat]


def _build_model(block_config: BlockConfig, n_layers: int, key: jax.Array) -> StackedModel:
    return StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=n_layers,
        decode=False,
        rngs=nnx.Rngs(params=key),
    )


def _train_one(
    block_config: BlockConfig,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    key: jax.Array,
    inputs: jax.Array,
    targets: jax.Array,
    weight_decay: float = 0.0,
) -> tuple[nnx.State, jax.Array]:
    """One (case, seed) member: teacher-forced one-step MSE identification.
    Returns (trained param state, final MSE). The --no-vmap path calls
    this directly per member; the vmapped path's per-member computation
    (inside _train_ensemble) mirrors it exactly.

    weight_decay defaults to 0.0, NOT optax.adamw's own default of 1e-4:
    for an LTI target (case identification), decay actively fights an
    exact fit. See docs/DECISIONS.md."""
    model = _build_model(block_config, n_layers, key)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)
    states = model.init_state(N=block_config.N)

    def loss_fn(m):
        pred, _ = m(inputs, states)
        return jnp.mean((pred - targets) ** 2)

    for _ in range(epochs):
        _, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

    final_mse = loss_fn(model)
    return nnx.state(model, nnx.Param), final_mse


def _train_ensemble(
    block_config: BlockConfig,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    keys: jax.Array,
    inputs_grid: jax.Array,
    targets_grid: jax.Array,
    weight_decay: float = 0.0,
) -> tuple[nnx.State, jax.Array]:
    """keys: (n_ensemble,) PRNGKeys. inputs_grid/targets_grid: (n_ensemble,
    l_max, d_input/d_output). Returns (ensemble param state - every leaf has
    a leading n_ensemble axis - per_member_final_mse: (n_ensemble,))."""

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return _build_model(block_config, n_layers, key)

    ensemble = init_ensemble(keys)
    optimizer = nnx.Optimizer(ensemble, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)

    def loss_fn(model):
        graphdef, params = nnx.split(model, nnx.Param)

        def single_member(p, inp, tgt):
            m = nnx.merge(graphdef, p)
            states = m.init_state(N=block_config.N)
            pred, _ = m(inp, states)
            return jnp.mean((pred - tgt) ** 2)

        losses = jax.vmap(single_member)(params, inputs_grid, targets_grid)
        return jnp.mean(losses), losses

    for _ in range(epochs):
        (_, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ensemble)
        optimizer.update(ensemble, grads)

    # recompute AFTER the loop: per_member from inside the loop reflects the
    # loss BEFORE that iteration's update, which is off by one epoch.
    _, final_per_member = loss_fn(ensemble)
    return nnx.state(ensemble, nnx.Param), final_per_member


def run_identify(
    variant: str,
    cases: list[int],
    n_seeds: int,
    epochs: int,
    d_model: int,
    N: int,
    n_layers: int,
    l_max: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    aprbs_low: float = -10.0,
    aprbs_high: float = 10.0,
    use_vmap: bool = True,
    seed_base: int = 0,
) -> list[dict]:
    """Returns one row per (case, seed): variant, case, seed, teacher_mse,
    and the trained param_state (for checkpoint saving by the caller)."""
    block_config = BlockConfig(d_model=d_model, N=N, l_max=l_max, **VARIANTS[variant])

    data = {c: case_data(c, l_max, aprbs_low, aprbs_high) for c in cases}

    flat_cases: list[int] = []
    flat_seeds: list[int] = []
    for case in cases:
        for seed in range(n_seeds):
            flat_cases.append(case)
            flat_seeds.append(seed)

    rows: list[dict] = []
    if use_vmap:
        inputs_grid = jnp.stack([data[c][0] for c in flat_cases])
        targets_grid = jnp.stack([data[c][1] for c in flat_cases])
        keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.PRNGKey(seed_base + s), c)
                for c, s in zip(flat_cases, flat_seeds)
            ]
        )

        ensemble_state, final_mse = _train_ensemble(
            block_config, n_layers, epochs, learning_rate, keys, inputs_grid, targets_grid,
            weight_decay=weight_decay,
        )
        for i, (case, seed) in enumerate(zip(flat_cases, flat_seeds)):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            rows.append(
                {
                    "variant": variant,
                    "case": case,
                    "seed": seed,
                    "teacher_mse": float(final_mse[i]),
                    "param_state": member_state,
                }
            )
    else:
        for case, seed in zip(flat_cases, flat_seeds):
            key = jax.random.fold_in(jax.random.PRNGKey(seed_base + seed), case)
            inputs, targets = data[case]
            param_state, final_mse = _train_one(
                block_config, n_layers, epochs, learning_rate, key, inputs, targets,
                weight_decay=weight_decay,
            )
            rows.append(
                {
                    "variant": variant,
                    "case": case,
                    "seed": seed,
                    "teacher_mse": float(final_mse),
                    "param_state": param_state,
                }
            )

    return rows


def _stringify_keys(x):
    if isinstance(x, dict):
        return {str(k): _stringify_keys(v) for k, v in x.items()}
    return x


def save_checkpoint(row: dict, config: dict, out_dir: pathlib.Path) -> pathlib.Path:
    """Writes out_dir/ckpt/{variant}_case{case}_seed{seed}.msgpack (+ .json
    sidecar with the variant config and lockfile_sha stamped in)."""
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['variant']}_case{row['case']}_seed{row['seed']}"

    pure_dict = _stringify_keys(row["param_state"].to_pure_dict())
    msgpack_bytes = serialization.msgpack_serialize(pure_dict)
    (ckpt_dir / f"{stem}.msgpack").write_bytes(msgpack_bytes)

    sidecar = {
        "variant": row["variant"],
        "case": row["case"],
        "seed": row["seed"],
        "teacher_mse": row["teacher_mse"],
        "config": config,
        "git_sha": get_git_sha(),
        "lockfile_sha": get_lockfile_sha(),
    }
    (ckpt_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))

    return ckpt_dir / f"{stem}.msgpack"
