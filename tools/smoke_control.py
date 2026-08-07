"""Controller smoke test (dpc_example-style pipeline, cut down for a fast
end-to-end check): train GRU controllers by unrolling through the LEARNED
M3 and M6 surrogates (s4dpc.identify.run_identify + s4dpc.control's
rollout_learned), evaluate closed-loop cost on the TRUE plant, and
compare against the oracle discrete LQR and a controller trained
directly through the TRUE (A, B) plant. Case 3 only, a short curriculum
(dpc_example's totals ~9000 epochs; this uses ~800) - this establishes
the identify -> control pipeline is wired correctly and produces
directionally sane numbers, NOT a publication-scale run. No @nnx.jit on
the training step (unlike training_code/dpc_example) - deliberately, to
keep this first pass debuggable from a Kaggle log with no local
jax/flax environment to iterate against; add jit back once this is
verified to work end to end.

    python tools/smoke_control.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.control import (
    BoundedGRUController,
    init_batched_state,
    make_controller_optimizer,
    rollout_learned,
    rollout_linear,
    rollout_lqr_true,
    solve_dlqr,
    true_quadratic_cost,
)
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
ID_EPOCHS = 2000
ID_LR = 1e-3

Q_X, R_U, Q_F = 5.0, 0.1, 50.0
HIDDEN_DIM = 64
MAX_ACTION = 10.0  # matches the APRBS training-data range (aprbs default (-10, 10))
CTRL_LR = 1e-3
CTRL_WD = 0.0
BATCH = 256
X0_RANGE = 3.0
EVAL_HORIZON = 50
EVAL_BATCH = 64
EVAL_X0_RANGE = 5.0

CURRICULUM = [
    {"N": 5, "epochs": 200},
    {"N": 10, "epochs": 300},
    {"N": 20, "epochs": 300},
]

SEED = 0


def _identify(variant: str) -> dict:
    rows = run_identify(
        variant=variant,
        cases=[CASE],
        n_seeds=1,
        epochs=ID_EPOCHS,
        d_model=D_MODEL,
        N=STATE_SIZE,
        n_layers=N_LAYERS,
        l_max=L_MAX,
        learning_rate=ID_LR,
        weight_decay=0.0,
        use_vmap=False,
        seed_base=SEED,
    )
    row = rows[0]
    print(f"[{variant}] identify: teacher_mse={row['teacher_mse']:.6e}")
    return row


def _train_controller_on_learned(variant: str, param_state, key: jax.Array) -> BoundedGRUController:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    model = StackedModel(
        block_config=block_config,
        d_input=D_INPUT,
        d_output=D_OUTPUT,
        n_layers=N_LAYERS,
        decode=True,
        rngs=nnx.Rngs(params=jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)),
    )
    nnx.update(model, param_state)
    graphdef, params = nnx.split(model)

    d_x, d_u = D_OUTPUT, D_INPUT - D_OUTPUT
    controller = BoundedGRUController(d_x, HIDDEN_DIM, d_u, MAX_ACTION, rngs=nnx.Rngs(key))
    total_epochs = sum(p["epochs"] for p in CURRICULUM)
    optimizer = make_controller_optimizer(controller, CTRL_LR, CTRL_WD, total_epochs)

    x0_key = jax.random.fold_in(key, 1)
    x0_batch = jax.random.uniform(x0_key, (BATCH, d_x), minval=-X0_RANGE, maxval=X0_RANGE)

    print(f"[{variant}] training GRU controller through the LEARNED model")
    for pi, phase in enumerate(CURRICULUM):
        N = phase["N"]

        def loss_fn(c, N=N):
            loss, _ = rollout_learned(
                c, graphdef, params, x0_batch, init_batched_state(model, x0_batch.shape[0]), Q_X, R_U, Q_F, N
            )
            return loss

        for epoch in range(phase["epochs"]):
            loss, grads = nnx.value_and_grad(loss_fn)(controller)
            optimizer.update(controller, grads)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"  [{variant}] phase {pi + 1} (N={N}) epoch {epoch:4d} | DPC loss: {float(loss):.4f}")

    return controller


def _train_controller_on_true(A: np.ndarray, B: np.ndarray, key: jax.Array) -> BoundedGRUController:
    A_j, B_j = jnp.asarray(A), jnp.asarray(B)
    d_x, d_u = A.shape[0], B.shape[1]
    controller = BoundedGRUController(d_x, HIDDEN_DIM, d_u, MAX_ACTION, rngs=nnx.Rngs(key))
    total_epochs = sum(p["epochs"] for p in CURRICULUM)
    optimizer = make_controller_optimizer(controller, CTRL_LR, CTRL_WD, total_epochs)

    x0_key = jax.random.fold_in(key, 1)
    x0_batch = jax.random.uniform(x0_key, (BATCH, d_x), minval=-X0_RANGE, maxval=X0_RANGE)

    print("[TRUE] training GRU controller through the TRUE plant (reference)")
    for pi, phase in enumerate(CURRICULUM):
        N = phase["N"]

        def loss_fn(c, N=N):
            return rollout_linear(c, x0_batch, A_j, B_j, Q_X, R_U, Q_F, N)

        for epoch in range(phase["epochs"]):
            loss, grads = nnx.value_and_grad(loss_fn)(controller)
            optimizer.update(controller, grads)
            if epoch % max(1, phase["epochs"] // 3) == 0 or epoch == phase["epochs"] - 1:
                print(f"  [TRUE] phase {pi + 1} (N={N}) epoch {epoch:4d} | DPC loss: {float(loss):.4f}")

    return controller


def _eval_on_true(controller: BoundedGRUController, A: np.ndarray, B: np.ndarray, key: jax.Array) -> float:
    A_j, B_j = jnp.asarray(A), jnp.asarray(B)
    d_x = A.shape[0]
    x0 = jax.random.uniform(key, (EVAL_BATCH, d_x), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    h = controller.init_hidden(EVAL_BATCH)
    x = x0
    xs = [x]
    us = []
    for _ in range(EVAL_HORIZON):
        h, u = controller(h, x)
        x = x @ A_j.T + u @ B_j.T
        xs.append(x)
        us.append(u)
    x_hist = np.asarray(jnp.stack(xs))
    u_hist = np.asarray(jnp.stack(us))
    return true_quadratic_cost(x_hist, u_hist, Q_X, R_U, Q_F)


def main() -> None:
    A_true, B_true = get_discrete_matrices(dt=0.01, case=CASE)

    Q = Q_X * np.eye(A_true.shape[0])
    R = R_U * np.eye(B_true.shape[1])
    K = solve_dlqr(A_true, B_true, Q, R)
    eval_key = jax.random.PRNGKey(123)
    x0_eval = np.asarray(
        jax.random.uniform(eval_key, (EVAL_BATCH, A_true.shape[0]), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    )
    x_hist_lqr, u_hist_lqr = rollout_lqr_true(A_true, B_true, K, x0_eval, EVAL_HORIZON)
    cost_lqr = true_quadratic_cost(x_hist_lqr, u_hist_lqr, Q_X, R_U, Q_F)
    print(f"\n[oracle LQR] TRUE-plant cost: {cost_lqr:.4f}\n")

    ctrl_true = _train_controller_on_true(A_true, B_true, jax.random.PRNGKey(SEED + 100))
    cost_true_ref = _eval_on_true(ctrl_true, A_true, B_true, eval_key)
    print(f"\n[GRU on TRUE plant] TRUE-plant cost: {cost_true_ref:.4f}\n")

    results = {}
    for variant in ["M3", "M6"]:
        row = _identify(variant)
        ctrl = _train_controller_on_learned(variant, row["param_state"], jax.random.PRNGKey(SEED + 200))
        cost = _eval_on_true(ctrl, A_true, B_true, eval_key)
        print(f"\n[GRU on {variant} surrogate] TRUE-plant cost: {cost:.4f}\n")
        results[variant] = {"teacher_mse": row["teacher_mse"], "true_plant_cost": cost}

    print("\n================= SUMMARY (case 3) =================")
    print(f"{'controller':28s} | {'true-plant cost':>16s}")
    print("-" * 48)
    print(f"{'oracle LQR':28s} | {cost_lqr:16.4f}")
    print(f"{'GRU on TRUE plant':28s} | {cost_true_ref:16.4f}")
    for variant, r in results.items():
        label = f"GRU on {variant} surrogate"
        print(f"{label:28s} | {r['true_plant_cost']:16.4f}   (id teacher_mse={r['teacher_mse']:.3e})")


if __name__ == "__main__":
    main()
