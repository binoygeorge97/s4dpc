"""TASK A (user, 2026-08-18, third round): for the first time this session
there is real outcome variance to contrast - 1/30 stability-penalized M3,
6/30 generic linear SSM, several dimension-sweep S4 checkpoints, all
transferring successfully alongside a much larger failing population.
Every prior spurious-mode correlation came back null partly because almost
everything failed (no variance to correlate against). This pools every
checkpoint across every variant identified this session (all of which
already have Abar/Bbar/K_lqr/teacher_mse/outcome sitting on disk from
earlier scripts - no new GPU work, pure CPU, seconds) and asks which,
if any, of the quantities already used this session actually separates
successful transfers from failures.

Quantities compared, all computed from each checkpoint's own extracted
Abar (M3's/the generic model's own open-loop augmented operator - NOT the
closed-loop transfer-construction matrix eigenmode_decomposition.py used,
which by definition has zero unstable eigenvalues on every success and so
can't be compared across the success/failure boundary):
  - teacher_mse
  - n_unstable at |lambda|>0.99 and the strict |lambda|>1.0 (RECONCILED entry's
    two thresholds)
  - ||K_s||/||K_x|| (LQR gain norm on the internal block vs the physical block)
  - median internal-block energy fraction of Abar's own unstable eigenvectors
    (NaN when n_unstable=0 - no unstable eigenvalues to measure)
  - Hankel singular value spread (sigma_6/sigma_1, since the true systems are
    exactly 6-dimensional - how much of a rank-6 truncation would have to
    discard) via the same markov-parameter/block-Hankel/SVD construction as
    tools/balanced_truncation.py
  - internal dimension

Two pools, reported separately because pooling them is confounded: ALL
(includes M0_S4, a hand construction that always succeeds by design and
always has n_unstable=0 by design - any quantity that merely distinguishes
"was this hand-built or learned" would look like a discovery here without
being one) and LEARNED-ONLY (every checkpoint that came from an actual
identification/training process: fullM3, stability-constrained M3, the
generic linear SSM, dimension-sweep d64/d256 S4 - this is the pool where
"does property X, measured on what gradient descent produced, predict
whether transfer succeeds" is actually a fair question). M1 is included in
neither internal-state comparison (it has no augmented state) but does
contribute to the outcome-only summary.

    python tools/success_vs_failure_contrast.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.stats import mannwhitneyu

DOCS = _REPO_ROOT / "docs"
D_X = 6
SUCCESS_RATIO_THRESHOLD = 100.0
N_HANKEL = 20


def markov_from_augmented(Abar: np.ndarray, Bbar: np.ndarray, Cout: np.ndarray, H: int) -> list[np.ndarray]:
    G = []
    Apow = np.eye(Abar.shape[0])
    for _ in range(H):
        G.append(Cout @ Apow @ Bbar)
        Apow = Abar @ Apow
    return G


def hankel_sv_spread(Abar: np.ndarray, Bbar: np.ndarray, D_X_local: int) -> tuple[float, int]:
    Cout = np.eye(D_X_local, Abar.shape[0])
    H = 2 * N_HANKEL
    G = markov_from_augmented(Abar, Bbar, Cout, H)
    H0 = np.block([[G[i + j] for j in range(N_HANKEL)] for i in range(N_HANKEL)])
    sv = np.linalg.svd(H0, compute_uv=False)
    sigma6_over_sigma1 = float(sv[5] / sv[0]) if len(sv) > 5 and sv[0] > 0 else float("nan")
    eff_rank = int(np.sum(sv > 1e-8 * sv[0]))
    return sigma6_over_sigma1, eff_rank


def spectral_quantities(Abar: np.ndarray) -> dict:
    eigvals, eigvecs = np.linalg.eig(Abar)
    abs_eig = np.abs(eigvals)
    n_unstable_099 = int(np.sum(abs_eig > 0.99))
    n_unstable_10 = int(np.sum(abs_eig > 1.0))
    unstable_idx = np.where(abs_eig > 1.0)[0]
    if len(unstable_idx) > 0:
        fracs_s = []
        for i in unstable_idx:
            v = eigvecs[:, i]
            frac_s = float(np.sum(np.abs(v[D_X:]) ** 2) / np.sum(np.abs(v) ** 2))
            fracs_s.append(frac_s)
        median_frac_s_unstable = float(np.median(fracs_s))
    else:
        median_frac_s_unstable = float("nan")
    return {
        "n_unstable_099": n_unstable_099,
        "n_unstable_10": n_unstable_10,
        "median_frac_s_unstable": median_frac_s_unstable,
    }


def ks_over_kx(K_lqr: np.ndarray) -> float:
    K_x, K_s = K_lqr[:, :D_X], K_lqr[:, D_X:]
    kx_norm = np.linalg.norm(K_x)
    if kx_norm == 0 or K_s.shape[1] == 0:
        return float("nan")
    return float(np.linalg.norm(K_s) / kx_norm)


def load_transfer_outcomes() -> dict:
    """variant,case,seed -> (cost_ratio, finite) from lqr_transfer_to_true_plant.csv."""
    out = {}
    with open(DOCS / "lqr_transfer_to_true_plant.csv") as f:
        for r in csv.DictReader(f):
            out[(r["variant"], int(r["case"]), int(r["seed"]))] = (
                float(r["cost_ratio"]), r["finite"] == "True")
    return out


def build_records() -> list[dict]:
    records = []
    outcomes = load_transfer_outcomes()
    export_dir = DOCS / "nu_gap_export"
    lqr_cache = DOCS / "lqr_cache"

    # fullM3, M0_S4: Abar/Bbar/C from nu_gap_export, K_lqr from lqr_cache, outcome from CSV
    for variant, label in [("fullM3", "fullM3"), ("M0_S4", "M0_S4")]:
        for path in sorted(export_dir.glob(f"{variant}_*.npz")):
            _, case_s, seed_s = path.stem.rsplit("_", 2)
            case, seed = int(case_s), int(seed_s)
            key = (variant, case, seed)
            cache_path = lqr_cache / f"{variant}_{case}_{seed}.npz"
            if key not in outcomes or not cache_path.exists():
                continue
            data = np.load(path)
            Abar, Bbar = data["A"], data["B"]
            K_lqr = np.load(cache_path)["K_lqr"]
            cost_ratio, finite = outcomes[key]
            rec = {"variant": label, "case": case, "seed": seed,
                   "teacher_mse": float(data["teacher_mse"]), "cost_ratio": cost_ratio, "finite": finite,
                   "internal_dim": Abar.shape[0] - D_X, "ks_over_kx": ks_over_kx(K_lqr)}
            rec.update(spectral_quantities(Abar))
            sv_spread, eff_rank = hankel_sv_spread(Abar, Bbar, D_X)
            rec["hankel_sigma6_over_sigma1"] = sv_spread
            rec["hankel_eff_rank"] = eff_rank
            records.append(rec)

    # M1: pure 6-dim, no augmented state - outcome + teacher_mse only
    for path in sorted(export_dir.glob("M1_*.npz")):
        _, case_s, seed_s = path.stem.rsplit("_", 2)
        case, seed = int(case_s), int(seed_s)
        key = ("M1", case, seed)
        if key not in outcomes:
            continue
        data = np.load(path)
        cost_ratio, finite = outcomes[key]
        records.append({"variant": "M1", "case": case, "seed": seed,
                         "teacher_mse": float(data["teacher_mse"]), "cost_ratio": cost_ratio, "finite": finite,
                         "internal_dim": 0, "ks_over_kx": float("nan"),
                         "n_unstable_099": np.nan, "n_unstable_10": np.nan,
                         "median_frac_s_unstable": np.nan,
                         "hankel_sigma6_over_sigma1": np.nan, "hankel_eff_rank": np.nan})

    # self-contained npz dirs: Abar/Bbar/K_lqr/teacher_mse/ratio_transfer/finite all present
    self_contained = [
        ("stability_penalized_M3", DOCS / "stability_constrained", "case{case}_seed{seed}.npz"),
        ("generic_linear_SSM", DOCS / "linear_ssm_baseline", "case{case}_seed{seed}.npz"),
    ]
    for label, dirpath, _pattern in self_contained:
        for path in sorted(dirpath.glob("case*_seed*.npz")):
            data = np.load(path)
            stem = path.stem  # "case{case}_seed{seed}"
            case = int(stem.split("_")[0].replace("case", ""))
            seed = int(stem.split("_")[1].replace("seed", ""))
            Abar, Bbar, K_lqr = data["Abar"], data["Bbar"], data["K_lqr"]
            rec = {"variant": label, "case": case, "seed": seed,
                   "teacher_mse": float(data["teacher_mse"]), "cost_ratio": float(data["ratio_transfer"]),
                   "finite": bool(data["finite"]), "internal_dim": Abar.shape[0] - D_X,
                   "ks_over_kx": ks_over_kx(K_lqr)}
            rec.update(spectral_quantities(Abar))
            sv_spread, eff_rank = hankel_sv_spread(Abar, Bbar, D_X)
            rec["hankel_sigma6_over_sigma1"] = sv_spread
            rec["hankel_eff_rank"] = eff_rank
            records.append(rec)

    # dimension_sweep: d64, d256 S4 (multiple labels in one dir, filename prefix = label)
    for path in sorted((DOCS / "dimension_sweep").glob("*.npz")):
        label_s, case_s, seed_s = path.stem.rsplit("_", 2)
        data = np.load(path)
        Abar, Bbar, K_lqr = data["Abar"], data["Bbar"], data["K_lqr"]
        rec = {"variant": f"S4_{label_s}", "case": int(case_s), "seed": int(seed_s),
               "teacher_mse": float(data["teacher_mse"]), "cost_ratio": float(data["ratio_transfer"]),
               "finite": bool(data["finite"]), "internal_dim": Abar.shape[0] - D_X,
               "ks_over_kx": ks_over_kx(K_lqr)}
        rec.update(spectral_quantities(Abar))
        sv_spread, eff_rank = hankel_sv_spread(Abar, Bbar, D_X)
        rec["hankel_sigma6_over_sigma1"] = sv_spread
        rec["hankel_eff_rank"] = eff_rank
        records.append(rec)

    for r in records:
        r["success"] = bool(r["finite"] and r["cost_ratio"] < SUCCESS_RATIO_THRESHOLD)
    return records


QUANTITIES = ["teacher_mse", "n_unstable_099", "n_unstable_10", "ks_over_kx",
              "median_frac_s_unstable", "hankel_sigma6_over_sigma1", "hankel_eff_rank", "internal_dim"]


def contrast(records: list[dict], pool_name: str) -> None:
    succ = [r for r in records if r["success"]]
    fail = [r for r in records if not r["success"]]
    print(f"\n=== {pool_name}: {len(succ)} success (ratio<{SUCCESS_RATIO_THRESHOLD:.0f}x) "
          f"vs {len(fail)} failure, n={len(records)} total ===")
    from collections import Counter
    print("  success by variant:", dict(Counter(r["variant"] for r in succ)))
    print("  failure by variant:", dict(Counter(r["variant"] for r in fail)))
    if not succ or not fail:
        print("  (one class empty - no contrast possible)")
        return
    for q in QUANTITIES:
        s_vals = np.array([r[q] for r in succ if r[q] == r[q]], dtype=float)  # drop NaN
        f_vals = np.array([r[q] for r in fail if r[q] == r[q]], dtype=float)
        if len(s_vals) < 2 or len(f_vals) < 2:
            print(f"  {q:28s}  too few non-NaN values (n_succ={len(s_vals)}, n_fail={len(f_vals)})")
            continue
        stat, p = mannwhitneyu(s_vals, f_vals, alternative="two-sided")
        print(f"  {q:28s}  median_succ={np.median(s_vals):>12.4g}  median_fail={np.median(f_vals):>12.4g}"
              f"  Mann-Whitney p={p:.4f}  (n_succ={len(s_vals)}, n_fail={len(f_vals)})")


def main() -> None:
    records = build_records()
    out_path = DOCS / "success_vs_failure_contrast.csv"
    header = sorted({k for r in records for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in records]
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(records)} checkpoints)")

    contrast(records, "ALL (includes hand-built M0_S4)")
    learned_only = [r for r in records if r["variant"] not in ("M0_S4", "M1")]
    contrast(learned_only, "LEARNED-ONLY (excludes M0_S4 hand-construction and M1's no-internal-state case)")


if __name__ == "__main__":
    main()
