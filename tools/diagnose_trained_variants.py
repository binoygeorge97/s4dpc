"""Task 2 (docs/DECISIONS.md): first real diagnostics, on ACTUALLY
TRAINED (not LS-init-constructed) M3/M6 checkpoints from
tools/train_to_convergence.py (SAME Kaggle kernel run, loaded from
./ckpt_trained/ - not committed, *.msgpack is gitignored except the one
parity fixture).

The single most informative comparison available right now: M3 has no
norm/activation/glu (VARIANTS["M3"]), so it is an affine function of the
input for ANY fixed params - its Jacobian d F/d x MUST be exactly
t-independent (a real constant matrix, whatever the model actually
converged to), not merely "close to A_d". Two different questions, both
reported: dev_from_A_d(t) = ||J(t)-A_d|| (how well the model converged
to the true dynamics - a convergence question) vs dev_from_J0(t) =
||J(t)-J(0)|| (how much the Jacobian ITSELF changes moving away from the
origin - the actual kink/flatness question, independent of convergence
quality). M3's dev_from_J0 should be ~0 (machine precision) regardless
of how well-converged it is. If it isn't, the kink is not caused by
LayerNorm/GELU/GLU and the original hypothesis needs revision. If M6's
dev_from_J0 is nontrivial and M3's isn't, the hypothesis survives on
this case.

    python tools/diagnose_trained_variants.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
import flax.serialization as serialization

from s4dpc import diagnostics
from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

CASE = 3
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
SEED = 0
H = 50
VARIANTS_TO_DIAGNOSE = ["M3", "M6"]
SWEEP_T = np.linspace(-2.0, 2.0, 41)
LOCAL_LINEARITY_SAMPLES = 128
LOCAL_LINEARITY_DELTA = 1e-2

CKPT_DIR = pathlib.Path("ckpt_trained")
DOCS_DIR = _REPO_ROOT / "docs"


def _build(variant: str, decode: bool, key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    return StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=decode, rngs=nnx.Rngs(params=key),
    )


def _load_trained(variant: str) -> StackedModel:
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), CASE)
    model = _build(variant, decode=True, key=key)
    path = CKPT_DIR / f"{variant}_case{CASE}.msgpack"
    pure_dict = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)

    pred_dtype = diagnostics.step(model, jnp.zeros((D_OUTPUT,), dtype=jnp.float64),
                                   jnp.zeros((D_INPUT - D_OUTPUT,), dtype=jnp.float64),
                                   diagnostics.zero_states(model))[0].dtype
    if pred_dtype != jnp.float64:
        raise RuntimeError(f"[{variant}] loaded checkpoint does not compute in float64 (got {pred_dtype})")
    return model


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    A_d, B_d = get_discrete_matrices(DT, CASE)
    A_d64 = jnp.asarray(A_d, dtype=jnp.float64)
    B_d64 = jnp.asarray(B_d, dtype=jnp.float64)
    d_u = D_INPUT - D_OUTPUT
    u0 = jnp.zeros((d_u,), dtype=jnp.float64)

    results = {}
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n=== {variant} ===")
        model = _load_trained(variant)
        states0 = diagnostics.zero_states(model)

        # 1. Markov parameters vs A_d^(h-1) @ B_d
        G = diagnostics.markov_parameters(model, states0, H)
        A_power = jnp.eye(D_OUTPUT, dtype=jnp.float64)
        rel_errs = []
        for h in range(1, H + 1):
            true_gh = A_power @ B_d64
            rel_err = float(jnp.linalg.norm(G[h - 1] - true_gh) / (jnp.linalg.norm(true_gh) + 1e-300))
            rel_errs.append(rel_err)
            A_power = A_power @ A_d64
        print(f"  Markov rel err: h=1:{rel_errs[0]:.4e}  h=10:{rel_errs[9]:.4e}  h=50:{rel_errs[49]:.4e}  "
              f"mean={np.mean(rel_errs):.4e}  max={np.max(rel_errs):.4e}")

        # 2. equilibrium drift
        drift = diagnostics.equilibrium_drift(model, states0)
        drift_norm = float(jnp.linalg.norm(drift))
        print(f"  equilibrium_drift |F(0,0,s)| = {drift_norm:.6e}")

        # 3. local-linearity defect at x=0, u=0
        defect = diagnostics.local_linearity_defect(
            model, states0, jnp.zeros((D_OUTPUT,), dtype=jnp.float64), u0,
            jax.random.PRNGKey(1), n_samples=LOCAL_LINEARITY_SAMPLES, delta_scale=LOCAL_LINEARITY_DELTA,
        )
        print(f"  local_linearity_defect (at x=0,u=0) = {float(defect):.6e}")

        # 4. Jacobian sweep through x=0, along each of the 6 state-basis directions.
        # Two different deviations, answering two different questions:
        #   dev_from_A_d(t) = ||J(t) - A_d||   -> convergence (did it learn the right dynamics)
        #   dev_from_J0(t)  = ||J(t) - J(0)||  -> kink/flatness (does J change moving off x=0),
        #                                         independent of whether it converged to the right answer
        sweep_summary = {}
        all_sweeps_A_d = {}
        all_sweeps_J0 = {}
        for dim in range(D_OUTPUT):
            direction = jnp.zeros((D_OUTPUT,), dtype=jnp.float64).at[dim].set(1.0)
            jacs = diagnostics.jacobian_sweep(model, states0, direction, jnp.asarray(SWEEP_T), u0)
            origin_idx = int(np.argmin(np.abs(SWEEP_T)))
            j0 = jacs[origin_idx]

            dev_A_d = np.asarray(jnp.linalg.norm(jacs - A_d64[None], axis=(1, 2)))
            dev_J0 = np.asarray(jnp.linalg.norm(jacs - j0[None], axis=(1, 2)))
            all_sweeps_A_d[dim] = dev_A_d
            all_sweeps_J0[dim] = dev_J0

            sweep_summary[dim] = {
                "dev_A_d_at_origin": float(dev_A_d[origin_idx]),
                "dev_A_d_max": float(dev_A_d.max()),
                "kink_strength_max_dev_from_J0": float(dev_J0.max()),
            }
            print(f"  dim={dim}: ||J(0)-A_d||={sweep_summary[dim]['dev_A_d_at_origin']:.4e}  "
                  f"max||J(t)-A_d||={sweep_summary[dim]['dev_A_d_max']:.4e}  "
                  f"KINK max||J(t)-J(0)||={sweep_summary[dim]['kink_strength_max_dev_from_J0']:.4e}")

        results[variant] = {
            "rel_errs": rel_errs, "drift_norm": drift_norm, "defect": float(defect),
            "sweep_summary": sweep_summary, "all_sweeps_A_d": all_sweeps_A_d, "all_sweeps_J0": all_sweeps_J0,
        }

    # --- side-by-side report ---
    print("\n=== SIDE BY SIDE: M3 vs M6 ===")
    print(f"{'metric':38s} {'M3':>16s} {'M6':>16s}")
    rows = [
        ("markov rel err (h=1)", lambda r: r["rel_errs"][0]),
        ("markov rel err (h=10)", lambda r: r["rel_errs"][9]),
        ("markov rel err (h=50)", lambda r: r["rel_errs"][49]),
        ("markov rel err (mean over h=1..50)", lambda r: float(np.mean(r["rel_errs"]))),
        ("equilibrium_drift |F(0,0,s)|", lambda r: r["drift_norm"]),
        ("local_linearity_defect (x=0)", lambda r: r["defect"]),
        ("KINK: max||J(t)-J(0)||, dim 0", lambda r: r["sweep_summary"][0]["kink_strength_max_dev_from_J0"]),
        ("KINK: max||J(t)-J(0)||, max over dims", lambda r: max(s["kink_strength_max_dev_from_J0"] for s in r["sweep_summary"].values())),
    ]
    for label, fn in rows:
        vals = []
        for variant in ["M3", "M6"]:
            vals.append(fn(results[variant]) if variant in results else float("nan"))
        print(f"{label:38s} {vals[0]:16.4e} {vals[1]:16.4e}")

    if "M3" in results:
        m3_kink = max(s["kink_strength_max_dev_from_J0"] for s in results["M3"]["sweep_summary"].values())
        print(f"\nM3 kink strength (should be ~1e-9 or better - linear by construction): {m3_kink:.4e}")
        if m3_kink > 1e-6:
            print("  M3 shows a NONTRIVIAL kink despite having no norm/activation/glu - "
                  "the kink is NOT caused by LayerNorm/GELU/GLU nonlinearities. Hypothesis needs revision.")
        else:
            print("  M3 is flat, as expected from its architecture.")
        if "M6" in results:
            m6_kink = max(s["kink_strength_max_dev_from_J0"] for s in results["M6"]["sweep_summary"].values())
            if m3_kink < 1e-6 and m6_kink > 1e-3:
                print(f"  M6 kink strength ({m6_kink:.4e}) IS nontrivial while M3's is not - "
                      "the hypothesis SURVIVES on this case.")

    # --- plot: both deviations vs t, dim 0, both variants ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    for variant, style in [("M3", "-"), ("M6", "--")]:
        if variant in results:
            ax1.plot(SWEEP_T, results[variant]["all_sweeps_A_d"][0], style, label=variant)
            ax2.plot(SWEEP_T, results[variant]["all_sweeps_J0"][0], style, label=variant)
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("t (state = t * e_0)")
        ax.legend()
    ax1.set_ylabel("||J(t) - A_d||_F")
    ax1.set_title(f"Convergence: Jacobian deviation from true A_d, case {CASE}")
    ax2.set_ylabel("||J(t) - J(0)||_F")
    ax2.set_title(f"Kink figure: Jacobian deviation from J(0) through x=0, case {CASE}")
    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "kink_figure_trained.png", dpi=120)
    print(f"\nwrote {DOCS_DIR / 'kink_figure_trained.png'}")

    # --- CSVs ---
    sweep_lines = ["variant,dim,t,dev_from_A_d,dev_from_J0"]
    for variant in results:
        for dim in range(D_OUTPUT):
            for t, dev_a, dev_j in zip(SWEEP_T, results[variant]["all_sweeps_A_d"][dim], results[variant]["all_sweeps_J0"][dim]):
                sweep_lines.append(f"{variant},{dim},{t},{dev_a},{dev_j}")
    (DOCS_DIR / "kink_figure_trained_sweep.csv").write_text("\n".join(sweep_lines))
    print(f"wrote {DOCS_DIR / 'kink_figure_trained_sweep.csv'}")

    summary_lines = ["variant,markov_rel_err_mean,markov_rel_err_h50,equilibrium_drift,"
                      "local_linearity_defect,kink_strength_max_dim0,kink_strength_max_over_dims"]
    for variant in results:
        r = results[variant]
        summary_lines.append(",".join(str(x) for x in [
            variant, np.mean(r["rel_errs"]), r["rel_errs"][49], r["drift_norm"], r["defect"],
            r["sweep_summary"][0]["kink_strength_max_dev_from_J0"],
            max(s["kink_strength_max_dev_from_J0"] for s in r["sweep_summary"].values()),
        ]))
    (DOCS_DIR / "diagnose_trained_variants_summary.csv").write_text("\n".join(summary_lines))
    print(f"wrote {DOCS_DIR / 'diagnose_trained_variants_summary.csv'}")


if __name__ == "__main__":
    main()
