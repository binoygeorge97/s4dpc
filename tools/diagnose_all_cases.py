"""Task 3 (docs/DECISIONS.md): extend Task 1/2 to all 7 cases, M3 and
M6, 3 seeds - correlate Markov-parameter error against Kreiss-like
transient amplification (tests/test_systems.py's kreiss_like =
max_k ||A_d^k||_2/rho^k), not spectral radius. rho(A_d) is flat
(~1.02-1.04) across every case here, so it structurally cannot explain
any per-case variation the way Kreiss amplification - which spans
roughly 1 to 330 across these 7 cases - could.

Loads the 7 cases x 3 seeds x 2 variants = 42 checkpoints
identify.save_checkpoint wrote under results/all_cases/ckpt/ (from
`python -m s4dpc.sweep --variant {M3,M6} --cases 1,2,3,4,5,6,7
--n_seeds 3 --epochs 40000 --out results/all_cases/{variant}.csv` - see
docs/LOG.md), builds a decode=True model per checkpoint, and runs
diagnostics.markov_parameters + equilibrium_drift against THAT CASE's
own (A_d, B_d) - not case 3's, unlike Task 1/2 above.

    python tools/diagnose_all_cases.py
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

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
H = 50
CASES = list(range(1, 8))
SEEDS = [0, 1, 2]
VARIANTS_TO_DIAGNOSE = ["M3", "M6"]
K_VALUES = (1, 5, 10, 25, 50)  # matches tests/test_systems.py's K_VALUES

CKPT_DIR = _REPO_ROOT / "results" / "all_cases" / "ckpt"
DOCS_DIR = _REPO_ROOT / "docs"


def _kreiss_like(A_d: np.ndarray) -> tuple[float, float]:
    """Duplicated from tests/test_systems.py's _stats (per this repo's
    standalone-tools-script convention) - returns (rho, kreiss_like)."""
    rho = float(np.max(np.abs(np.linalg.eigvals(A_d))))
    power_norms = {k: float(np.linalg.norm(np.linalg.matrix_power(A_d, k), ord=2)) for k in K_VALUES}
    kreiss_like = max(power_norms[k] / rho**k for k in K_VALUES)
    return rho, kreiss_like


def _build(variant: str, decode: bool, key: jax.Array) -> StackedModel:
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[variant])
    return StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=decode, rngs=nnx.Rngs(params=key),
    )


def _load_trained(variant: str, case: int, seed: int) -> StackedModel:
    path = CKPT_DIR / f"{variant}_case{case}_seed{seed}.msgpack"
    key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
    model = _build(variant, decode=True, key=key)
    pure_dict = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)
    return model


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    case_stats = {}
    for case in CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        rho, kreiss = _kreiss_like(A_d)
        case_stats[case] = {"A_d": A_d, "B_d": B_d, "rho": rho, "kreiss_like": kreiss}
        print(f"case {case}: rho={rho:.4f}  kreiss_like={kreiss:.4f}")

    rows: list[dict] = []
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n=== {variant} ===")
        for case in CASES:
            A_d64 = jnp.asarray(case_stats[case]["A_d"], dtype=jnp.float64)
            B_d64 = jnp.asarray(case_stats[case]["B_d"], dtype=jnp.float64)
            case_markov, case_drift = [], []
            for seed in SEEDS:
                path = CKPT_DIR / f"{variant}_case{case}_seed{seed}.msgpack"
                if not path.exists():
                    print(f"  [{variant}/case{case}/seed{seed}] MISSING checkpoint at {path} - skipping")
                    continue
                model = _load_trained(variant, case, seed)
                states0 = diagnostics.zero_states(model)

                G = diagnostics.markov_parameters(model, states0, H)
                A_power = jnp.eye(D_OUTPUT, dtype=jnp.float64)
                rel_errs = []
                for h in range(1, H + 1):
                    true_gh = A_power @ B_d64
                    rel_errs.append(float(jnp.linalg.norm(G[h - 1] - true_gh) / (jnp.linalg.norm(true_gh) + 1e-300)))
                    A_power = A_power @ A_d64
                mean_rel_err = float(np.mean(rel_errs))

                drift_norm = float(jnp.linalg.norm(diagnostics.equilibrium_drift(model, states0)))

                case_markov.append(mean_rel_err)
                case_drift.append(drift_norm)
                rows.append({"variant": variant, "case": case, "seed": seed,
                             "markov_rel_err_mean": mean_rel_err, "equilibrium_drift": drift_norm})
                print(f"  [case{case}/seed{seed}] markov_rel_err_mean={mean_rel_err:.4e}  "
                      f"equilibrium_drift={drift_norm:.4e}")

            if case_markov:
                print(f"  [case{case}] over {len(case_markov)} seeds: "
                      f"markov mean={np.mean(case_markov):.4e} std={np.std(case_markov):.4e}  |  "
                      f"drift mean={np.mean(case_drift):.4e} std={np.std(case_drift):.4e}  |  "
                      f"rho={case_stats[case]['rho']:.4f}  kreiss={case_stats[case]['kreiss_like']:.4f}")

    if not rows:
        print("\nNO CHECKPOINTS FOUND AT ALL - nothing to correlate. Stopping.")
        return

    # --- correlation analysis ---
    print("\n=== CORRELATION: per-case mean Markov error vs rho vs Kreiss-like amplification ===")
    correlations = {}
    for variant in VARIANTS_TO_DIAGNOSE:
        xs_rho, xs_kreiss, ys = [], [], []
        for case in CASES:
            vals = [r["markov_rel_err_mean"] for r in rows if r["variant"] == variant and r["case"] == case]
            if vals:
                ys.append(float(np.mean(vals)))
                xs_rho.append(case_stats[case]["rho"])
                xs_kreiss.append(case_stats[case]["kreiss_like"])
        if len(ys) >= 3:
            corr_rho = float(np.corrcoef(ys, xs_rho)[0, 1])
            corr_kreiss = float(np.corrcoef(ys, xs_kreiss)[0, 1])
            log_ys = np.log10(np.array(ys) + 1e-300)
            log_kreiss = np.log10(np.array(xs_kreiss) + 1e-300)
            corr_kreiss_log = float(np.corrcoef(log_ys, log_kreiss)[0, 1])
            correlations[variant] = {"corr_rho": corr_rho, "corr_kreiss": corr_kreiss, "corr_kreiss_log": corr_kreiss_log}
            print(f"  [{variant}] n_cases={len(ys)}  corr(markov_err, rho)={corr_rho:+.4f}  "
                  f"corr(markov_err, kreiss_like)={corr_kreiss:+.4f}  "
                  f"corr(log markov_err, log kreiss_like)={corr_kreiss_log:+.4f}")
        else:
            print(f"  [{variant}] only {len(ys)} cases have data - skipping correlation")

    # --- plot ---
    fig, axes = plt.subplots(1, len(VARIANTS_TO_DIAGNOSE), figsize=(6 * len(VARIANTS_TO_DIAGNOSE), 5))
    if len(VARIANTS_TO_DIAGNOSE) == 1:
        axes = [axes]
    for ax, variant in zip(axes, VARIANTS_TO_DIAGNOSE):
        for case in CASES:
            vals = [r["markov_rel_err_mean"] for r in rows if r["variant"] == variant and r["case"] == case]
            if vals:
                x, y = case_stats[case]["kreiss_like"], float(np.mean(vals))
                ax.loglog([x], [y], "o", color="C0")
                ax.annotate(str(case), (x, y))
        ax.set_xlabel("Kreiss-like amplification")
        ax.set_ylabel("mean Markov relative error")
        ax.set_title(variant)
    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "markov_err_vs_kreiss.png", dpi=120)
    print(f"\nwrote {DOCS_DIR / 'markov_err_vs_kreiss.png'}")

    # --- CSVs ---
    header = list(rows[0].keys())
    lines = [",".join(header)] + [",".join(str(row[h]) for h in header) for row in rows]
    (DOCS_DIR / "diagnose_all_cases_summary.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'diagnose_all_cases_summary.csv'}")

    case_lines = ["case,rho,kreiss_like"] + [
        f"{case},{case_stats[case]['rho']},{case_stats[case]['kreiss_like']}" for case in CASES
    ]
    (DOCS_DIR / "case_kreiss_stats.csv").write_text("\n".join(case_lines))
    print(f"wrote {DOCS_DIR / 'case_kreiss_stats.csv'}")


if __name__ == "__main__":
    main()
