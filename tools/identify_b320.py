"""TASK 4 (user, 2026-08-25): rerun M3 identification with 320
trajectories (32,000 transitions) instead of batch_size=1, to check
whether this project's own pipeline reproduces the external
reproduction's B=320 numbers (A_xx ~3.2e-3, drift ~3.9e-4, coupling
unchanged at 5-16, transfer still failing) or whether the discrepancy
is in our own code - per instruction, that outranks everything else if
it doesn't reproduce.

Standalone script, NOT a change to s4dpc/identify.py's canonical B=1
recipe - this is a one-off verification, matching this project's
established pattern for comparison scripts (tools/dimension_sweep.py,
tools/linear_ssm_baseline.py, etc., none of which touched identify.py
either). `train_ensemble_multi_traj` mirrors s4dpc.identify._train_ensemble
exactly, with one addition: an INNER jax.vmap over a trajectory-batch
axis inside the per-ensemble-member loss, so one model instance trains
on N_TRAJ independent trajectories per step instead of one.

    python tools/identify_b320.py            # full run (GPU)
    python tools/identify_b320.py --smoke     # 1 case, 1 seed, 200 epochs (timing/correctness check)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import flax.serialization as serialization

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT, DATA_SEED, DT, _build_model, _stringify_keys
from s4dpc.logging import get_git_sha, get_jax_backend, get_lockfile_sha, get_machine_id
from s4dpc.systems import get_discrete_matrices

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
N_TRAJ = 320
EPOCHS = 40000  # same recipe as every other real checkpoint in this project, for direct comparability
LEARNING_RATE = 1e-3
APRBS_LOW, APRBS_HIGH = -10.0, 10.0
EXPORT_DIR = _REPO_ROOT / "docs" / "b320"


def case_data_multi(case: int, n_traj: int) -> tuple[jax.Array, jax.Array]:
    """n_traj INDEPENDENT trajectories - generate_microgrid_trajectory's
    own RandomState already draws n_traj independent (X_0, APRBS) pairs
    when batch_size=n_traj (s4dpc/data.py) - no new data-generation code
    needed, just batch_size=320 instead of case_data's hardcoded 1."""
    inputs, targets = generate_microgrid_trajectory(
        batch_size=n_traj, length=L_MAX, seed=DATA_SEED, system_case=case,
        dt=DT, aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH,
    )
    return jnp.asarray(inputs), jnp.asarray(targets)  # (n_traj, L, D_input/output)


def train_ensemble_multi_traj(
    block_config: BlockConfig, n_layers: int, epochs: int, learning_rate: float,
    keys: jax.Array, inputs_grid: jax.Array, targets_grid: jax.Array,
) -> tuple[nnx.State, jax.Array]:
    """keys: (n_ensemble,). inputs_grid/targets_grid: (n_ensemble, n_traj,
    l_max, d_input/output). Identical structure to
    s4dpc.identify._train_ensemble, with one addition: single_member's
    loss is now itself a vmap over the n_traj axis (mean MSE across all
    n_traj trajectories for that member), instead of a single (l_max,
    d_input) sequence."""

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return _build_model(block_config, n_layers, key)

    ensemble = init_ensemble(keys)
    optimizer = nnx.Optimizer(ensemble, optax.adamw(learning_rate, weight_decay=0.0), wrt=nnx.Param)

    def loss_fn(model):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def single_member(p, r, inp_multi, tgt_multi):
            m = nnx.merge(graphdef, p, r)
            states = m.init_state(N=block_config.N)

            def single_traj(inp, tgt):
                pred, _ = m(inp, states)
                return jnp.mean((pred - tgt) ** 2)

            per_traj = jax.vmap(single_traj)(inp_multi, tgt_multi)
            return jnp.mean(per_traj)

        losses = jax.vmap(single_member)(params, rest, inputs_grid, targets_grid)
        return jnp.mean(losses), losses

    @nnx.jit
    def train_step(ens, opt):
        (_, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
        opt.update(ens, grads)
        return per_member

    for _ in range(epochs):
        train_step(ensemble, optimizer)

    _, final_per_member = loss_fn(ensemble)
    return nnx.state(ensemble, nnx.Param), final_per_member


def augmented_operator(graphdef, params, D_X: int, d_model: int, N: int):
    """Abar/Bbar/c0 for M3 (no norm/act/glu, so f is exactly affine) via
    jacfwd - same construction as tools/task0_decode_mode_parity.py's
    augmented_operator, reused here for the B=320 checkpoints' Axx/Axs/
    coupling/n_unstable metrics."""
    n_s = 2 * d_model * N

    def _unpack(z):
        x = z[:D_X]
        s_re = z[D_X:D_X + d_model * N].reshape(d_model, N)
        s_im = z[D_X + d_model * N:].reshape(d_model, N)
        return x, [s_re + 1j * s_im]

    def _pack(x, s):
        s0 = s[0]
        return jnp.concatenate([x, s0.real.ravel(), s0.imag.ravel()])

    def f(z, u):
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        model_in = jnp.concatenate([x, u], axis=-1)
        out, new_s = m(model_in, s)
        return _pack(out, new_s)

    D_U = 3
    z0 = jnp.zeros(D_X + n_s)
    u0 = jnp.zeros(D_U)
    Abar = jax.jacfwd(lambda z: f(z, u0))(z0)
    Bbar = jax.jacfwd(lambda u: f(z0, u))(u0)
    c0 = f(z0, u0)
    return np.asarray(Abar), np.asarray(Bbar), np.asarray(c0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override EPOCHS (timing checks)")
    args = parser.parse_args()

    cases = [3] if args.smoke else CASES
    n_seeds = 1 if args.smoke else N_SEEDS
    epochs = (200 if args.smoke else EPOCHS) if args.epochs is None else args.epochs

    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}  smoke={args.smoke}  "
          f"cases={cases}  n_seeds={n_seeds}  epochs={epochs}  N_TRAJ={N_TRAJ}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    D_X = 6

    flat_cases, flat_seeds = [], []
    for c in cases:
        for s in range(n_seeds):
            flat_cases.append(c)
            flat_seeds.append(s)

    data = {c: case_data_multi(c, N_TRAJ) for c in cases}
    inputs_grid = jnp.stack([data[c][0] for c in flat_cases])   # (n_ensemble, N_TRAJ, L, D_INPUT)
    targets_grid = jnp.stack([data[c][1] for c in flat_cases])  # (n_ensemble, N_TRAJ, L, D_OUTPUT)
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), c) for c, s in zip(flat_cases, flat_seeds)])

    t0 = time.time()
    ensemble_state, final_mse = train_ensemble_multi_traj(
        block_config, N_LAYERS, epochs, LEARNING_RATE, keys, inputs_grid, targets_grid,
    )
    wall = time.time() - t0
    print(f"identification wall time: {wall:.1f}s ({wall / len(flat_cases):.2f}s/checkpoint)")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = EXPORT_DIR / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (case, seed) in enumerate(zip(flat_cases, flat_seeds)):
        member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)

        # decode=True (step mode) for the jacfwd extraction below - it
        # calls the model on a SINGLE timestep, which conv mode
        # (_build_model's decode=False) structurally rejects
        # ("Convolution mode currently requires sequence length to equal
        # l_max=100"). Same params, different __call__ path - matching
        # tools/task0_decode_mode_parity.py's model_conv/model_step split.
        from s4dpc.model import StackedModel
        model = StackedModel(
            block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
            decode=True, rngs=nnx.Rngs(params=jax.random.fold_in(jax.random.PRNGKey(seed), case)),
        )
        nnx.update(model, member_state)
        graphdef, params = nnx.split(model, nnx.Param)
        Abar, Bbar, c0 = augmented_operator(graphdef, params, D_X, D_MODEL, STATE_SIZE)

        A_d, B_d = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        Axx = Abar[:D_X, :D_X]
        Axs = Abar[:D_X, D_X:]
        axx_rel_err = float(np.linalg.norm(Axx - A_d, "fro") / np.linalg.norm(A_d, "fro"))
        coupling_ratio = float(np.linalg.norm(Axs, "fro") / np.linalg.norm(Axx, "fro"))
        drift = float(np.linalg.norm(c0[:D_X]))
        eigvals = np.linalg.eigvals(Abar)
        n_unstable = int(np.sum(np.abs(eigvals) > 1.0))

        stem = f"M3_b320_case{case}_seed{seed}"
        pure_dict = _stringify_keys(member_state.to_pure_dict())
        (ckpt_dir / f"{stem}.msgpack").write_bytes(serialization.msgpack_serialize(pure_dict))
        np.savez(ckpt_dir / f"{stem}_operator.npz", Abar=Abar, Bbar=Bbar, c0=c0)

        row = {
            "variant": "M3_b320", "case": case, "seed": seed,
            "teacher_mse": float(final_mse[i]),
            "axx_rel_err": axx_rel_err, "equilibrium_drift": drift,
            "coupling_ratio": coupling_ratio, "n_unstable": n_unstable,
            "n_traj": N_TRAJ, "epochs": epochs,
            "git_sha": get_git_sha(), "lockfile_sha": get_lockfile_sha(),
            "machine": get_machine_id(), "backend": get_jax_backend(),
        }
        rows.append(row)
        (ckpt_dir / f"{stem}.json").write_text(json.dumps(row, indent=2))
        print(f"[{stem}] teacher_mse={row['teacher_mse']:.4e}  axx_rel_err={axx_rel_err:.4e}  "
              f"drift={drift:.4e}  coupling={coupling_ratio:.4f}  n_unstable={n_unstable}")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    if args.smoke:
        out_name = "b320_smoke_summary.csv"
    elif args.epochs is not None:
        out_name = f"b320_timing_e{args.epochs}_summary.csv"
    else:
        out_name = "b320_summary.csv"
    out_path = EXPORT_DIR / out_name
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
