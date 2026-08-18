"""TASK B (user, 2026-08-18, fifth round): the nu-gap winding-condition
guard conflates two different events, and the current code cannot tell
them apart.

`tools/nu_gap_analysis.py`'s `nu_gap()` computes `cond = wno + eta_Phat -
eta_P` and calls the comparison invalid (forcing `delta_nu=1.0`) whenever
`abs(cond) >= 0.4`. But among the FAILED rows, there are two structurally
different situations:
  (i)  cond is close to a nonzero INTEGER (e.g. cond~1.0, cond~-2.0) - a
       genuine winding-condition violation. The nu-gap is a TOTAL metric;
       delta_nu=1 here is not a missing value, it is the metric's own
       correct, maximal verdict - a positive diagnosis.
  (ii) cond is NOT close to any integer (e.g. cond=0.6, cond=1.3) - the
       numerical winding-number computation itself (phase unwrapping over
       a finite frequency grid) broke down. Reporting delta_nu=1 here is
       reporting a number the theory never predicted; the honest report
       is INDETERMINATE.

Given the ALREADY-measured pole-count mismatch this project has
established (M3's augmented operator has 2-19 unstable eigenvalues vs the
true plant's 0-6), theory predicts genuinely nonzero winding numbers ARE
common here - but predicting integers is not the same as verifying them,
and this project has already found three bugs disguised as results this
session (float32-vs-float64 rank, the "cleanly falsified" balanced-
truncation overstatement, the EXP1 optimizer-speed false alarm). Treat
this the same way: bin it, don't assume it.

Reruns `nu_gap_analysis.py`'s EXACT comparison (`nu_gap((A_true,B_true,I,0),
(A,B,C,0))` per (variant,case,seed)) but captures the raw `wno`, `eta_P`,
`eta_Phat`, `cond` for every row - the existing `docs/nu_gap_analysis.csv`
only stored the boolean `dnu_valid`, not these underlying diagnostics.

    python tools/nugap_integer_check.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.systems import get_discrete_matrices  # noqa: E402


def _sqrtm_psd(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def _eig_outside_unit_disk(A, tol: float = 1e-6):
    return int(np.sum(np.abs(np.linalg.eigvals(A)) > 1.0 + tol))


def _fast_freq_response_grid(A, B, C, D, z_grid):
    """C(zI-A)^-1 B + D over the whole z_grid via ONE eigendecomposition
    instead of one direct solve per frequency point - same speedup as
    tools/free_response_test.py's transfer_function, needed here because
    the original tools/nu_gap_analysis.py's _freq_response does a fresh
    np.linalg.solve per point, which at d1030 x 2000 points x ~90
    checkpoints does not finish in practical time locally (killed after
    11 CPU-minutes with zero checkpoints completed - confirmed too slow,
    not just slow)."""
    eigvals, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    CV = C @ V
    VinvB = Vinv @ B
    inv_diag = 1.0 / (z_grid[:, None] - eigvals[None, :])
    H = np.einsum("pi,fi,ij->fpj", CV, inv_diag, VinvB)
    if np.any(D):
        H = H + D[None, :, :]
    return H


def nu_gap(P_ss, Phat_ss, n_freq: int = 2000) -> tuple[float, dict]:
    """Same math as tools/nu_gap_analysis.py's nu_gap - reimplemented here
    only to swap the per-frequency direct solve for a per-checkpoint
    eigendecomposition (see _fast_freq_response_grid)."""
    A, B, C, D = P_ss
    Ah, Bh, Ch, Dh = Phat_ss
    thetas = np.linspace(0, 2 * np.pi, n_freq, endpoint=False) + (np.pi / n_freq)
    z_grid = np.exp(1j * thetas)

    Pz_all = _fast_freq_response_grid(A, B, C, D, z_grid)
    Pzh_all = _fast_freq_response_grid(Ah, Bh, Ch, Dh, z_grid)

    gaps, arg_vals = [], []
    for i in range(n_freq):
        Pz, Pzh = Pz_all[i], Pzh_all[i]
        p, m = Pz.shape
        M1 = np.linalg.inv(_sqrtm_psd(np.eye(p) + Pzh @ Pzh.conj().T))
        M2 = np.linalg.inv(_sqrtm_psd(np.eye(m) + Pz.conj().T @ Pz))
        s = np.linalg.svd(M1 @ (Pzh - Pz) @ M2, compute_uv=False)
        gaps.append(s[0])
        arg_vals.append(np.linalg.det(np.eye(m) + Pzh.conj().T @ Pz))
    gaps, arg_vals = np.array(gaps), np.array(arg_vals)
    phases = np.unwrap(np.angle(arg_vals))
    wno = (phases[-1] + (phases[-1] - phases[-2]) - phases[0]) / (2 * np.pi)
    eta_P, eta_Phat = _eig_outside_unit_disk(A), _eig_outside_unit_disk(Ah)
    cond = wno + eta_Phat - eta_P
    sup_gap = float(np.max(gaps))
    min_det = float(np.min(np.abs(arg_vals)))
    is_valid = abs(cond) < 0.4 and sup_gap < 1.0 and min_det > 1e-6
    info = {"wno": float(wno), "eta_P": eta_P, "eta_Phat": eta_Phat,
            "cond": float(cond), "valid": is_valid, "min_det": min_det}
    return (sup_gap if is_valid else 1.0), info

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
VARIANTS_ORDER = ["M1", "fullM3", "M0_S4"]
NEAR_INTEGER_TOL = 0.1  # how close to a whole number counts as "clean" for this bin


def main() -> None:
    rows = []
    true_AB = {c: tuple(np.asarray(m) for m in get_discrete_matrices(DT, c)) for c in CASES}

    for variant in VARIANTS_ORDER:
        for case in CASES:
            A_true, B_true = true_AB[case]
            for seed in range(N_SEEDS):
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A, B, C = data["A"], data["B"], data["C"]
                dnu, info = nu_gap(
                    (A_true, B_true, np.eye(D_X), np.zeros((D_X, D_U))),
                    (A, B, C, np.zeros((C.shape[0], D_U))),
                )
                cond = info["cond"]
                nearest_int = round(cond)
                dist_to_int = abs(cond - nearest_int)
                rows.append({
                    "variant": variant, "case": case, "seed": seed,
                    "wno": info["wno"], "eta_P": info["eta_P"], "eta_Phat": info["eta_Phat"],
                    "cond": cond, "nearest_int": nearest_int, "dist_to_nearest_int": dist_to_int,
                    "valid": info["valid"], "delta_nu": dnu,
                })
                print(f"  [{variant}/case{case}/seed{seed}] wno={info['wno']:.4f}  "
                      f"eta_P={info['eta_P']}  eta_Phat={info['eta_Phat']}  cond={cond:.4f}  "
                      f"nearest_int={nearest_int}  dist={dist_to_int:.4f}  valid={info['valid']}")

    out_path = DOCS / "nugap_integer_check.csv"
    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    failed = [r for r in rows if not r["valid"]]
    print(f"\n=== {len(failed)}/{len(rows)} rows FAILED the current abs(cond)<0.4 guard "
          f"(delta_nu forced to 1.0) ===")
    clean_integer = [r for r in failed if r["dist_to_nearest_int"] < NEAR_INTEGER_TOL]
    genuinely_zero = [r for r in failed if r["nearest_int"] == 0 and r["dist_to_nearest_int"] < NEAR_INTEGER_TOL]
    nonzero_clean_integer = [r for r in clean_integer if r["nearest_int"] != 0]
    non_integer = [r for r in failed if r["dist_to_nearest_int"] >= NEAR_INTEGER_TOL]

    print(f"  (a) clean NONZERO integer (genuine winding violation, delta_nu=1 is CORRECT): "
          f"{len(nonzero_clean_integer)}/{len(failed)}")
    print(f"  (b) clean ZERO-ish but still failed the 0.4 guard (guard threshold too tight, "
          f"not a real violation - the metric itself would call this fine): "
          f"{len(genuinely_zero)}/{len(failed)}")
    print(f"  (c) NOT close to any integer (numerical breakdown, indeterminate not 1.0): "
          f"{len(non_integer)}/{len(failed)}")

    if non_integer:
        print("\n  Non-integer (bin c, numerical breakdown) rows:")
        for r in non_integer:
            print(f"    {r['variant']}/case{r['case']}/seed{r['seed']}: cond={r['cond']:.4f} "
                  f"(nearest int {r['nearest_int']}, dist {r['dist_to_nearest_int']:.4f})")

    print("\n  Distribution of nearest_int among ALL failed rows:")
    from collections import Counter
    print(" ", dict(sorted(Counter(r["nearest_int"] for r in failed).items())))


if __name__ == "__main__":
    main()
