"""Task 1/2/3 (docs/DECISIONS.md), following up on the 3-seed all-cases
sweep whose raw correlation (+1.0000) turned out to be an outlier
artifact: 10 seeds instead of 3, the FULL diagnostics.py suite (not
just markov+drift) on every non-diverged checkpoint, an explicit
divergence-rate analysis against 4 case-level predictors, and Spearman
alongside Pearson (n=7 cases - Spearman is the more honest statistic
here, robust to the single-outlier problem the first correlation
attempt had; Pearson kept for direct comparison, both with p-values).

Reuses results/all_cases/ckpt/{variant}_case{case}_seed{seed}.msgpack
from `python -m s4dpc.sweep --variant {M3,M6} --cases 1..7 --n_seeds 10
--epochs 40000 ...` (same command as the 3-seed run, n_seeds bumped -
see docs/LOG.md).

Divergence: markov_err_mean > DIVERGED_THRESHOLD (1e3, per instruction).
Diverged checkpoints are EXCLUDED from the per-case/variant median
diagnostics (Task 3) but COUNTED (Task 1b/2) - and their expensive
diagnostics (local_linearity_defect, jacobian_sweep) are skipped
entirely: meaningless on a blown-up model, and jacfwd through likely
NaN/Inf weights is wasted compute at best.

    python tools/diagnose_all_cases_10seeds.py
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
from scipy.stats import pearsonr, spearmanr
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
H_REPORT = [1, 10, 50]
CASES = list(range(1, 8))
SEEDS = list(range(10))
VARIANTS_TO_DIAGNOSE = ["M3", "M6"]
K_VALUES = (1, 5, 10, 25, 50)
DIVERGED_THRESHOLD = 1e3
SWEEP_T = np.linspace(-2.0, 2.0, 15)  # reduced from the single-case script's 41 - 140-checkpoint scale
LOCAL_LINEARITY_SAMPLES = 64  # reduced from 128, same reason
LOCAL_LINEARITY_DELTA = 1e-2

CKPT_DIR = _REPO_ROOT / "results" / "all_cases" / "ckpt"
DOCS_DIR = _REPO_ROOT / "docs"


def _case_predictors(A_d: np.ndarray) -> dict:
    """Duplicated from tests/test_systems.py's _stats (per this repo's
    standalone-tools-script convention), extended to expose max_k
    ||A_d^k||_2 as its own predictor rather than only folded into
    kreiss_like."""
    rho = float(np.max(np.abs(np.linalg.eigvals(A_d))))
    power_norms = {k: float(np.linalg.norm(np.linalg.matrix_power(A_d, k), ord=2)) for k in K_VALUES}
    kreiss_like = max(power_norms[k] / rho**k for k in K_VALUES)
    max_power_norm = max(power_norms.values())
    non_normality = float(np.linalg.norm(A_d @ A_d.T - A_d.T @ A_d, ord="fro"))
    return {"rho": rho, "kreiss_like": kreiss_like, "max_power_norm": max_power_norm, "non_normality": non_normality}


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


def _markov_and_drift(model, states0, A_d64, B_d64):
    G = diagnostics.markov_parameters(model, states0, H)
    A_power = jnp.eye(D_OUTPUT, dtype=jnp.float64)
    rel_errs = []
    for h in range(1, H + 1):
        true_gh = A_power @ B_d64
        rel_errs.append(float(jnp.linalg.norm(G[h - 1] - true_gh) / (jnp.linalg.norm(true_gh) + 1e-300)))
        A_power = A_power @ A_d64
    drift_norm = float(jnp.linalg.norm(diagnostics.equilibrium_drift(model, states0)))
    return rel_errs, drift_norm


def _kink_and_defect(model, states0, u0):
    defect = float(diagnostics.local_linearity_defect(
        model, states0, jnp.zeros((D_OUTPUT,), dtype=jnp.float64), u0,
        jax.random.PRNGKey(1), n_samples=LOCAL_LINEARITY_SAMPLES, delta_scale=LOCAL_LINEARITY_DELTA,
    ))
    kink = 0.0
    for dim in range(D_OUTPUT):
        direction = jnp.zeros((D_OUTPUT,), dtype=jnp.float64).at[dim].set(1.0)
        jacs = diagnostics.jacobian_sweep(model, states0, direction, jnp.asarray(SWEEP_T), u0)
        origin_idx = int(np.argmin(np.abs(SWEEP_T)))
        dev_j0 = float(jnp.max(jnp.linalg.norm(jacs - jacs[origin_idx][None], axis=(1, 2))))
        kink = max(kink, dev_j0)
    return defect, kink


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")

    case_stats = {}
    for case in CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        case_stats[case] = {"A_d": A_d, "B_d": B_d, **_case_predictors(A_d)}
        print(f"case {case}: rho={case_stats[case]['rho']:.4f}  kreiss_like={case_stats[case]['kreiss_like']:.4f}  "
              f"max_power_norm={case_stats[case]['max_power_norm']:.4f}  "
              f"non_normality={case_stats[case]['non_normality']:.4f}")

    d_u = D_INPUT - D_OUTPUT
    u0 = jnp.zeros((d_u,), dtype=jnp.float64)

    rows: list[dict] = []
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n=== {variant} ===")
        for case in CASES:
            A_d64 = jnp.asarray(case_stats[case]["A_d"], dtype=jnp.float64)
            B_d64 = jnp.asarray(case_stats[case]["B_d"], dtype=jnp.float64)
            for seed in SEEDS:
                path = CKPT_DIR / f"{variant}_case{case}_seed{seed}.msgpack"
                row = {"variant": variant, "case": case, "seed": seed}
                if not path.exists():
                    print(f"  [case{case}/seed{seed}] MISSING checkpoint - skipping")
                    row.update(failed=True, diverged=None)
                    rows.append(row)
                    continue
                try:
                    model = _load_trained(variant, case, seed)
                    states0 = diagnostics.zero_states(model)
                    rel_errs, drift_norm = _markov_and_drift(model, states0, A_d64, B_d64)
                    markov_mean = float(np.mean(rel_errs))
                    diverged = markov_mean > DIVERGED_THRESHOLD
                    row.update(failed=False, diverged=diverged, equilibrium_drift=drift_norm,
                               markov_err_mean=markov_mean,
                               **{f"markov_err_h{h}": rel_errs[h - 1] for h in H_REPORT})
                    if diverged:
                        print(f"  [case{case}/seed{seed}] DIVERGED  markov_err_mean={markov_mean:.4e}  "
                              f"(skipping local_linearity/kink)")
                        row.update(local_linearity_defect=float("nan"), kink_magnitude=float("nan"))
                    else:
                        defect, kink = _kink_and_defect(model, states0, u0)
                        row.update(local_linearity_defect=defect, kink_magnitude=kink)
                        print(f"  [case{case}/seed{seed}] markov_err_mean={markov_mean:.4e}  "
                              f"drift={drift_norm:.4e}  defect={defect:.4e}  kink={kink:.4e}")
                except Exception as e:
                    import traceback
                    print(f"  [case{case}/seed{seed}] FAILED: {e}")
                    traceback.print_exc()
                    row.update(failed=True, diverged=None)
                rows.append(row)

    ok_rows = [r for r in rows if not r.get("failed", False)]
    if not ok_rows:
        print("\nNOTHING SUCCEEDED - stopping.")
        return

    # ============================================================
    # TASK 1: per-case median/IQR of markov_err, divergence count
    # ============================================================
    print("\n\n=== TASK 1: per-case median/IQR of markov_err, divergence count (threshold=1e3) ===")
    case_variant_stats = {}
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n  [{variant}]")
        for case in CASES:
            vals = [r["markov_err_mean"] for r in ok_rows if r["variant"] == variant and r["case"] == case]
            n_diverged = sum(1 for r in ok_rows if r["variant"] == variant and r["case"] == case and r["diverged"])
            n_total = len(vals)
            if not vals:
                continue
            median = float(np.median(vals))
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            case_variant_stats[(variant, case)] = {
                "median": median, "q1": q1, "q3": q3, "iqr": q3 - q1,
                "n_diverged": n_diverged, "n_total": n_total,
            }
            print(f"    case{case}: median={median:.4e}  IQR=[{q1:.4e}, {q3:.4e}] (width={q3 - q1:.4e})  "
                  f"diverged={n_diverged}/{n_total}")

    # ============================================================
    # TASK 1a: Pearson + Spearman, markov_err (median) vs kreiss_like
    # ============================================================
    print("\n=== TASK 1a: correlation, per-case MEDIAN markov_err vs kreiss_like ===")
    for variant in VARIANTS_TO_DIAGNOSE:
        for excl6, label in [(False, "all 7 cases"), (True, "excluding case 6")]:
            cases_here = [c for c in CASES if not (excl6 and c == 6)]
            xs = [case_stats[c]["kreiss_like"] for c in cases_here if (variant, c) in case_variant_stats]
            ys = [case_variant_stats[(variant, c)]["median"] for c in cases_here if (variant, c) in case_variant_stats]
            if len(xs) < 3:
                continue
            pr, pp = pearsonr(xs, ys)
            sr, sp = spearmanr(xs, ys)
            print(f"  [{variant}, {label}] n={len(xs)}  Pearson r={pr:+.4f} (p={pp:.4f})  "
                  f"Spearman rho={sr:+.4f} (p={sp:.4f})")

    # ============================================================
    # TASK 1b: case-6 divergence rate
    # ============================================================
    print("\n=== TASK 1b: case-6 divergence rate ===")
    for variant in VARIANTS_TO_DIAGNOSE:
        if (variant, 6) in case_variant_stats:
            s = case_variant_stats[(variant, 6)]
            print(f"  [{variant}] case6: {s['n_diverged']}/{s['n_total']} seeds diverged "
                  f"({100 * s['n_diverged'] / s['n_total']:.0f}%)")

    # ============================================================
    # TASK 2: divergence rate vs 4 case-level predictors
    # ============================================================
    print("\n=== TASK 2: divergence rate per case vs 4 predictors ===")
    predictor_names = ["kreiss_like", "rho", "non_normality", "max_power_norm"]
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n  [{variant}]")
        div_rates, preds = {p: [] for p in predictor_names}, {p: [] for p in predictor_names}
        for case in CASES:
            if (variant, case) not in case_variant_stats:
                continue
            s = case_variant_stats[(variant, case)]
            rate = s["n_diverged"] / s["n_total"]
            print(f"    case{case}: div_rate={rate:.2f}  kreiss={case_stats[case]['kreiss_like']:.3f}  "
                  f"rho={case_stats[case]['rho']:.4f}  non_normality={case_stats[case]['non_normality']:.3f}  "
                  f"max_power_norm={case_stats[case]['max_power_norm']:.3f}")
            for p in predictor_names:
                div_rates[p].append(rate)
                preds[p].append(case_stats[case][p])
        print(f"    correlation(divergence_rate, predictor):")
        for p in predictor_names:
            if len(set(div_rates[p])) < 2 or len(set(preds[p])) < 2:
                print(f"      {p}: degenerate (no variation) - skipping")
                continue
            pr, pp = pearsonr(preds[p], div_rates[p])
            sr, sp = spearmanr(preds[p], div_rates[p])
            print(f"      {p}: Pearson r={pr:+.4f} (p={pp:.4f})  Spearman rho={sr:+.4f} (p={sp:.4f})")

    print("\n=== case 4 vs case 6 tension check ===")
    for variant in VARIANTS_TO_DIAGNOSE:
        if all((variant, c) in case_variant_stats for c in (3, 4, 6)):
            m3_, m4_, m6_ = (case_variant_stats[(variant, c)]["median"] for c in (3, 4, 6))
            d3_, d4_, d6_ = (
                case_variant_stats[(variant, c)]["n_diverged"] / case_variant_stats[(variant, c)]["n_total"]
                for c in (3, 4, 6)
            )
            print(f"  [{variant}] case3(kreiss=1.00): median={m3_:.4e} div_rate={d3_:.2f}  |  "
                  f"case4(kreiss=3.30): median={m4_:.4e} div_rate={d4_:.2f}  |  "
                  f"case6(kreiss=330.3): median={m6_:.4e} div_rate={d6_:.2f}")
            between = m3_ < m4_ < m6_
            print(f"    case4 clearly between case3 and case6 (by median markov_err)? {'YES' if between else 'NO'}")

    # ============================================================
    # TASK 3: per case/variant median diagnostics (excluding diverged)
    # ============================================================
    print("\n=== TASK 3: per case/variant median diagnostics (excluding diverged) ===")
    task3_stats = {}
    for variant in VARIANTS_TO_DIAGNOSE:
        print(f"\n  [{variant}]")
        for case in CASES:
            clean = [r for r in ok_rows if r["variant"] == variant and r["case"] == case and not r["diverged"]]
            if not clean:
                print(f"    case{case}: NO non-diverged seeds - skipping")
                continue
            stats = {
                "n_clean": len(clean),
                **{f"markov_err_h{h}_median": float(np.median([r[f"markov_err_h{h}"] for r in clean])) for h in H_REPORT},
                "equilibrium_drift_median": float(np.median([r["equilibrium_drift"] for r in clean])),
                "local_linearity_defect_median": float(np.median([r["local_linearity_defect"] for r in clean])),
                "kink_magnitude_median": float(np.median([r["kink_magnitude"] for r in clean])),
            }
            task3_stats[(variant, case)] = stats
            print(f"    case{case} (n={stats['n_clean']}): "
                  f"markov[h1,h10,h50]=[{stats['markov_err_h1_median']:.3e}, "
                  f"{stats['markov_err_h10_median']:.3e}, {stats['markov_err_h50_median']:.3e}]  "
                  f"drift={stats['equilibrium_drift_median']:.3e}  "
                  f"defect={stats['local_linearity_defect_median']:.3e}  "
                  f"kink={stats['kink_magnitude_median']:.3e}")

    # ============================================================
    # Final plot: kink & markov_err vs kreiss_like, M3 vs M6 overlaid
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = {"M3": "o", "M6": "s"}
    for variant in VARIANTS_TO_DIAGNOSE:
        xs, ys_kink, ys_markov, labels = [], [], [], []
        for case in CASES:
            if (variant, case) not in task3_stats:
                continue
            xs.append(case_stats[case]["kreiss_like"])
            ys_kink.append(max(task3_stats[(variant, case)]["kink_magnitude_median"], 1e-16))
            ys_markov.append(task3_stats[(variant, case)]["markov_err_h50_median"])
            labels.append(case)
        ax1.loglog(xs, ys_kink, markers[variant], label=variant, markersize=8)
        ax2.loglog(xs, ys_markov, markers[variant], label=variant, markersize=8)
        for x, y, case in zip(xs, ys_kink, labels):
            ax1.annotate(str(case), (x, y), fontsize=7)
        for x, y, case in zip(xs, ys_markov, labels):
            ax2.annotate(str(case), (x, y), fontsize=7)
    ax1.set_xlabel("Kreiss-like amplification")
    ax1.set_ylabel("kink magnitude (median, non-diverged seeds)")
    ax1.set_title("Kink magnitude vs Kreiss (M3 is the built-in zero-kink control)")
    ax1.legend()
    ax2.set_xlabel("Kreiss-like amplification")
    ax2.set_ylabel("Markov error, h=50 (median, non-diverged seeds)")
    ax2.set_title("Markov error vs Kreiss")
    ax2.legend()
    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "kink_and_markov_vs_kreiss_10seeds.png", dpi=120)
    print(f"\nwrote {DOCS_DIR / 'kink_and_markov_vs_kreiss_10seeds.png'}")

    # ============================================================
    # CSVs
    # ============================================================
    header = sorted({k for r in ok_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in ok_rows]
    (DOCS_DIR / "diagnose_all_cases_10seeds_raw.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'diagnose_all_cases_10seeds_raw.csv'}")

    case_lines = ["case,rho,kreiss_like,max_power_norm,non_normality"] + [
        f"{c},{case_stats[c]['rho']},{case_stats[c]['kreiss_like']},"
        f"{case_stats[c]['max_power_norm']},{case_stats[c]['non_normality']}"
        for c in CASES
    ]
    (DOCS_DIR / "case_predictors_full.csv").write_text("\n".join(case_lines))
    print(f"wrote {DOCS_DIR / 'case_predictors_full.csv'}")


if __name__ == "__main__":
    main()
