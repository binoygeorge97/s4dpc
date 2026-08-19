"""TASK 0 (user, 2026-08-19, sixth session): does M3's conv-mode
(decode=False, used for identification) forward output agree with its
stepped-recurrence mode (decode=True, used for every control rollout and
every LQR-transfer computation in this project) on TRAINED checkpoints,
at both zero and nonzero initial internal state, out to 200 steps (the
control horizon, twice l_max=100)?

Runs on the REAL trained msgpack checkpoints in docs/nu_gap_export/ckpt/
(the actual exports behind this session's results), not fresh-init
params. float64, single-threaded (JAX_ENABLE_X64 set before any jax
import; XLA flags cap intra/inter-op threads to 1 below).

FOUR checks, in order:

  (A) decode=False (conv) vs decode=True (step), s0=0, steps 1..100 -
      the only regime where a direct comparison is even well-posed (see
      (B)). Per-checkpoint max abs/rel deviation AND the full per-step
      deviation curve.
  (B) Established by direct inspection of s4-nnx's source
      (S4LayerEnsemble.__call__, s4_nnx/s4.py) before running anything:
      conv mode hard-rejects any sequence length != l_max (raises
      ValueError - no truncation, padding, or wrapping happens, so this
      is stated as fact, not inferred), and its `previous_state`
      argument is accepted but NEVER used to compute outputs - it is
      returned unchanged. Conv mode is therefore, by construction,
      ONLY defined for the zero-initial-state impulse response, and
      only for exactly L=100. There is no way to run it to 200 steps or
      from a nonzero state at all - not a bug to characterize, a
      structural fact to report precisely. Empirically confirmed here:
      running conv mode twice with two different random nonzero
      `previous_state` values must give bit-identical output.
  (C) Since conv mode cannot run past l_max, the >100-step question is
      answered differently: does decode=True's REAL stepped output,
      continued to 200 steps, match the closed-form prediction implied
      by M3's own augmented linear operator (Abar, Bbar - extracted via
      jacfwd from this SAME decode=True model, exactly the construction
      `tools/m3_spurious_modes.py` and every downstream script this
      session used)? If yes, every Abar/Bbar-based result this session
      produced (LQR-transfer, the gauge-freedom theorem, the dither
      cure, fidelity-matched truncation) is validated against what the
      real deployed model actually computes, not merely a theoretical
      abstraction.
  (D) decode=True from a NONZERO s0: confirms it changes the output
      (unlike conv mode), quantifies by how much.

    python tools/task0_decode_mode_parity.py
"""
from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OMP_NUM_THREADS"] = "1"

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # before any other jax op
jax.config.update("jax_platforms", "cpu")

import json

import jax.numpy as jnp
import numpy as np
from flax import nnx
import flax.serialization as serialization

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.diagnostics import zero_states
from s4dpc.data import generate_microgrid_trajectory
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.model import StackedModel

CKPT_DIR = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X = 6
HORIZON_CONV = 100  # = l_max, the only length conv mode accepts
HORIZON_FULL = 200  # the control rollout horizon this project actually uses
DATA_SEED = 42


def _stringify_keys(x):
    if isinstance(x, dict):
        return {str(k): _stringify_keys(v) for k, v in x.items()}
    return x


def load_checkpoint_state(template_state: nnx.State, path: pathlib.Path) -> nnx.State:
    pure_dict = serialization.msgpack_restore(path.read_bytes())
    state = jax.tree_util.tree_map(lambda x: x, template_state)
    state.replace_by_pure_dict(pure_dict)
    return state


def build_models(block_config: BlockConfig, n_layers: int, key: jax.Array):
    model_conv = StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT,
                               n_layers=n_layers, decode=False, rngs=nnx.Rngs(params=key))
    model_step = StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT,
                               n_layers=n_layers, decode=True, rngs=nnx.Rngs(params=key))
    return model_conv, model_step


def run_step_mode(model_step, inputs: np.ndarray, s0) -> tuple[np.ndarray, list]:
    """TEACHER-FORCED: inputs (L, D_INPUT) = [x_true_t, u_t] recorded pairs,
    matching conv mode's own training convention exactly (conv mode always
    sees the true x at every step - that's what identification trains on).
    Returns (outputs (L, D_OUTPUT), final_state)."""
    state = s0
    outs = []
    for t in range(inputs.shape[0]):
        out_t, state = model_step(jnp.asarray(inputs[t]), state)
        outs.append(np.asarray(out_t))
    return np.stack(outs), state


def run_step_mode_free_running(model_step, x0: np.ndarray, u_seq: np.ndarray, s0) -> tuple[np.ndarray, list]:
    """FREE-RUNNING / self-referential: x is the MODEL'S OWN previous
    output fed back in as the next step's input - EXACTLY
    s4dpc.control.rollout_learned's construction (`x = x_next`, never the
    true trajectory past t=0). This, not the teacher-forced version above,
    is how decode=True is actually deployed for every control-side and
    LQR-transfer result in this project. u_seq: (L, D_U), exogenous
    (applied, not model-predicted) at every step, matching rollout_learned."""
    state = s0
    x = jnp.asarray(x0)
    outs = []
    for t in range(u_seq.shape[0]):
        model_in = jnp.concatenate([x, jnp.asarray(u_seq[t])])
        out_t, state = model_step(model_in, state)
        outs.append(np.asarray(out_t))
        x = out_t
    return np.stack(outs), state


def augmented_operator(graphdef, params, D_X_local: int, d_model: int, N: int):
    """Abar (Z_DIM,Z_DIM), Bbar (Z_DIM,D_U) via jacfwd at z=0,u=0 - EXACT
    match to tools/m3_spurious_modes.py's own construction (same _pack/
    _unpack convention: z = [x, s.real.ravel(), s.imag.ravel()]), so this
    is independently re-deriving the SAME object every other script this
    session used, not a new convention."""
    S_DIM = d_model * N
    Z_DIM = D_X_local + 2 * S_DIM
    D_U = D_INPUT - D_X_local

    def _unpack(z):
        x = z[:D_X_local]
        s_re = z[D_X_local:D_X_local + S_DIM].reshape(d_model, N)
        s_im = z[D_X_local + S_DIM:].reshape(d_model, N)
        return x, s_re + 1j * s_im

    def _pack(x, s):
        return jnp.concatenate([x, s.real.ravel(), s.imag.ravel()])

    def f(z, u):
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        x_next, (s_next,) = m(jnp.concatenate([x, u]), [s])
        return _pack(x_next, s_next)

    z0 = jnp.zeros((Z_DIM,), dtype=jnp.float64)
    u0 = jnp.zeros((D_U,), dtype=jnp.float64)
    Abar = np.asarray(jax.jacfwd(f, argnums=0)(z0, u0))
    Bbar = np.asarray(jax.jacfwd(f, argnums=1)(z0, u0))
    c0 = np.asarray(f(z0, u0))  # MAJOR FINDING (see module docstring update below):
    # M3 is affine, not linear - f(0,0) != 0 (this project's own "equilibrium_drift"
    # diagnostic, already named and measured). z_{k+1} = Abar@z_k + Bbar@u_k alone
    # (no +c0) reproduces the real model to only ~100 abs error over 200 steps;
    # WITH c0 included, it matches to 2.7e-12 (machine precision). Established
    # session scripts (lqr_transfer_to_true_plant.py, free_response_test.py,
    # dither_cure_test.py, fidelity_matched_truncation.py) all propagate
    # `z = z @ Acl.T` with no +c0 term - see this script's final report for the
    # precise, checked impact on each.
    return Abar, Bbar, c0


def main() -> None:
    # single-threaded per this repo's reproducibility convention (env vars set
    # at module import, before any jax op, above) - JIT itself stays on: JIT'd
    # XLA execution is already deterministic given a fixed thread count, and
    # disabling it would make no accuracy difference here, only a large speed one.
    block_config = BlockConfig(d_model=16, N=32, l_max=HORIZON_CONV, **VARIANTS["M3"])
    n_layers = 1
    d_model, N = 16, 32

    rows = []
    step_curve_rows = []

    for case in CASES:
        inputs200, _ = generate_microgrid_trajectory(
            batch_size=1, length=HORIZON_FULL, seed=DATA_SEED, system_case=case, dt=0.01,
            aprbs_low=-10.0, aprbs_high=10.0)
        inputs200 = inputs200[0]  # (200, 9)
        inputs100 = inputs200[:HORIZON_CONV]

        for seed in range(N_SEEDS):
            path = CKPT_DIR / f"M3_case{case}_seed{seed}.msgpack"
            if not path.exists():
                continue
            key = jax.random.PRNGKey(0)  # construction key doesn't matter once state is overwritten
            model_conv, model_step = build_models(block_config, n_layers, key)
            template_state = nnx.state(model_conv, nnx.Param)
            loaded_state = load_checkpoint_state(template_state, path)
            nnx.update(model_conv, loaded_state)
            nnx.update(model_step, loaded_state)

            # ---- (A) conv vs step, s0=0, steps 1..100 ----
            s0_conv = zero_states(model_conv)
            conv_out, _ = model_conv(jnp.asarray(inputs100), s0_conv)
            conv_out = np.asarray(conv_out)

            s0_step = zero_states(model_step)
            step_out_100, state_after_100 = run_step_mode(model_step, inputs100, s0_step)

            abs_diff = np.abs(conv_out - step_out_100)
            rel_diff = abs_diff / (np.abs(conv_out) + 1e-300)
            max_abs = float(abs_diff.max())
            max_rel = float(rel_diff.max())
            argmax_step = int(abs_diff.max(axis=-1).argmax())

            print(f"[case{case}/seed{seed}] (A) conv-vs-step steps1-100: "
                  f"max_abs={max_abs:.6e}  max_rel={max_rel:.6e}  at_step={argmax_step}")

            # ---- (B) conv mode ignores previous_state - empirical confirmation ----
            rng_state = np.random.RandomState(case * 100 + seed)
            s_rand1 = [jnp.asarray((rng_state.randn(d_model, N) + 1j * rng_state.randn(d_model, N)).astype(complex))]
            s_rand2 = [jnp.asarray((rng_state.randn(d_model, N) + 1j * rng_state.randn(d_model, N)).astype(complex))]
            conv_out_s1, _ = model_conv(jnp.asarray(inputs100), s_rand1)
            conv_out_s2, _ = model_conv(jnp.asarray(inputs100), s_rand2)
            conv_state_invariance = float(jnp.max(jnp.abs(conv_out_s1 - conv_out_s2)))

            # ---- (C) FREE-RUNNING step mode to 200 steps vs closed-form Abar/Bbar
            # prediction - matching rollout_learned's actual construction (x fed back
            # from the model's own prior output), NOT teacher-forced like check (A)/conv
            # mode - a teacher-forced comparison here would be apples-to-oranges, since
            # every control-side/LQR-transfer result in this project uses free-running
            # decode=True, never teacher-forced deployment. ----
            u_seq_200 = inputs200[:, D_X:]
            x0_zero = np.zeros(D_X, dtype=np.float64)
            step_out_200, _ = run_step_mode_free_running(
                model_step, x0_zero, u_seq_200, zero_states(model_step))

            step_graphdef, step_params = nnx.split(model_step, nnx.Param)
            Abar, Bbar, c0 = augmented_operator(step_graphdef, step_params, D_X, d_model, N)
            # independent PURE-NUMPY simulation - never calls model_step again,
            # so this is a genuine external check, not the same computation twice.
            # Two variants: WITHOUT c0 (matching how established session scripts
            # actually built z=z@Acl.T) and WITH c0 (the mathematically complete
            # affine relationship) - reporting both makes the bias-term finding's
            # impact directly visible, not asserted.
            z_nobias = np.zeros(Abar.shape[0], dtype=np.float64)
            z_withbias = np.zeros(Abar.shape[0], dtype=np.float64)
            closed_form_x_nobias, closed_form_x_withbias = [], []
            for t in range(HORIZON_FULL):
                u_t = inputs200[t, D_X:]
                z_nobias = Abar @ z_nobias + Bbar @ u_t
                z_withbias = Abar @ z_withbias + Bbar @ u_t + c0
                closed_form_x_nobias.append(z_nobias[:D_X].copy())
                closed_form_x_withbias.append(z_withbias[:D_X].copy())
            closed_form_x_nobias = np.stack(closed_form_x_nobias)
            closed_form_x = np.stack(closed_form_x_withbias)

            abs_diff_200_nobias = np.abs(step_out_200 - closed_form_x_nobias)
            max_abs_200_nobias = float(abs_diff_200_nobias.max())

            abs_diff_200 = np.abs(step_out_200 - closed_form_x)
            rel_diff_200 = abs_diff_200 / (np.abs(closed_form_x) + 1e-300)
            max_abs_200 = float(abs_diff_200.max())
            max_rel_200 = float(rel_diff_200.max())
            max_abs_200_post100 = float(abs_diff_200[100:].max())

            c0_x_norm = float(np.linalg.norm(c0[:D_X]))
            print(f"                (C) step-vs-closed-form WITH c0, steps1-200: "
                  f"max_abs={max_abs_200:.6e}  max_rel={max_rel_200:.6e}  "
                  f"max_abs[100:200]={max_abs_200_post100:.6e}  |  "
                  f"WITHOUT c0: max_abs={max_abs_200_nobias:.6e}  |  ||c0_x||={c0_x_norm:.4f}")

            # ---- (D) nonzero s0 for step mode: does it change the output? ----
            s_nonzero = [jnp.asarray(((rng_state.randn(d_model, N) + 1j * rng_state.randn(d_model, N)) * 0.32).astype(complex))]
            step_out_nonzero, _ = run_step_mode(model_step, inputs100, s_nonzero)
            step_s0_sensitivity = float(np.max(np.abs(step_out_nonzero - step_out_100)))

            print(f"                (B) conv state-invariance (should be ~0): {conv_state_invariance:.3e}  "
                  f"(D) step s0-sensitivity (should be >>0): {step_s0_sensitivity:.3e}")

            rows.append({
                "case": case, "seed": seed,
                "A_max_abs": max_abs, "A_max_rel": max_rel, "A_argmax_step": argmax_step,
                "B_conv_state_invariance": conv_state_invariance,
                "C_max_abs_200": max_abs_200, "C_max_rel_200": max_rel_200,
                "C_max_abs_post100": max_abs_200_post100,
                "C_max_abs_200_NO_C0": max_abs_200_nobias, "c0_x_norm": c0_x_norm,
                "D_step_s0_sensitivity": step_s0_sensitivity,
            })

            for t in range(HORIZON_CONV):
                step_curve_rows.append({"case": case, "seed": seed, "step": t,
                                         "abs_diff_conv_vs_step": float(abs_diff[t].max())})

    for name, data_rows in [("task0_parity_summary.csv", rows),
                             ("task0_parity_stepcurve.csv", step_curve_rows)]:
        out_path = _REPO_ROOT / "docs" / name
        header = sorted({k for r in data_rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in data_rows]
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path} ({len(data_rows)} rows)")

    print("\n=== SUMMARY across all checkpoints ===")
    a_abs = [r["A_max_abs"] for r in rows]
    a_rel = [r["A_max_rel"] for r in rows]
    c_abs = [r["C_max_abs_200"] for r in rows]
    c_abs_post100 = [r["C_max_abs_post100"] for r in rows]
    b_inv = [r["B_conv_state_invariance"] for r in rows]
    d_sens = [r["D_step_s0_sensitivity"] for r in rows]
    print(f"(A) conv-vs-step [0,100): max_abs median={np.median(a_abs):.3e} max={max(a_abs):.3e}  "
          f"max_rel median={np.median(a_rel):.3e} max={max(a_rel):.3e}")
    print(f"(B) conv state-invariance: median={np.median(b_inv):.3e} max={max(b_inv):.3e} "
          f"(expect exactly 0 or machine-eps)")
    print(f"(C) step-vs-closed-form [0,200): max_abs median={np.median(c_abs):.3e} max={max(c_abs):.3e}  "
          f"max_abs restricted to [100,200): median={np.median(c_abs_post100):.3e} max={max(c_abs_post100):.3e}")
    print(f"(D) step s0-sensitivity: median={np.median(d_sens):.3e} min={min(d_sens):.3e} "
          f"(expect >> 0, confirming conv/step disagree whenever s0!=0)")

    c_abs_nobias = [r["C_max_abs_200_NO_C0"] for r in rows]
    c0_norms = [r["c0_x_norm"] for r in rows]
    print(f"\nBIAS-TERM FINDING: WITHOUT c0, step-vs-closed-form max_abs median="
          f"{np.median(c_abs_nobias):.3e} (max {max(c_abs_nobias):.3e}) - WITH c0, median="
          f"{np.median(c_abs):.3e} (max {max(c_abs):.3e}). ||c0_x|| (equilibrium_drift norm) "
          f"median={np.median(c0_norms):.4f}, range=[{min(c0_norms):.4f}, {max(c0_norms):.4f}].")


if __name__ == "__main__":
    main()
