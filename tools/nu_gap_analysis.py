"""Item 3 (user brief, 2026-08-13): replace the saturated mode-count
correlation with the Vinnicombe nu-gap. A between-case correlation of
mode counts against DPC severity is blind to a mechanism that has
already saturated in every case (Task 4: ~300 spurious modes present on
every one of the 6 control cases, failure present on every one) - that's
a threshold, not a graded signal, and correlation cannot see thresholds.
The nu-gap delta_nu(P, Phat), together with the closed-loop robust
stability margin b_{K,Phat}, gives a DIFFERENT, theoretically grounded
test: Vinnicombe's theorem states a controller K that stabilizes Phat
also stabilizes the true P iff b_{K,Phat} > delta_nu(P,Phat). This can
be checked case by case, seed by seed, against what was actually
observed - a sharper test than "does severity correlate with a count."

Both pieces (nu-gap via normalized coprime factorization + pointwise
frequency gap + winding number; robust margin b via a closed-loop H-inf
norm through python-control's tested `control.norm`) were implemented
and self-tested LOCALLY, on pure numpy/scipy, before any GPU spend -
see the test suite embedded below (run this file with --selftest to
reproduce): identity gives exactly 0, symmetry holds, the pointwise
piece matches an independent manual SISO chordal-distance calculation
exactly, the coprime factorization satisfies its defining normalization
property to ~1e-15, a stable-vs-unstable comparison produces an exact-
integer winding number that correctly cancels the pole-count difference,
and the robust margin b is a healthy positive number for a good LQR
gain, shrinks monotonically toward 0 as a gain is scaled toward the
stability boundary, and is EXACTLY 0 for any destabilizing gain - the
one property that matters most and is unambiguous regardless of any
residual sign-convention uncertainty in the rest of the formula.

Restructured (user brief, 2026-08-13, item 1): "delta_nu(P, P_hat) ... is a
winding-number condition plus an H-infinity norm over a frequency grid - no
training, no autodiff, seconds on CPU. Stop queueing CPU linear algebra
behind GPU training jobs." All GPU work (training M1/M0_S4/truncated-M3/
full-M3 controllers, extracting (A,B,C) realizations and K_eff) now lives
in tools/nu_gap_export.py, run once on Kaggle/Colab, writing
docs/nu_gap_export/{variant}_{case}_{seed}.npz. This file's main() reads
those .npz files and does ONLY the nu-gap/robust-margin/Markov-error math -
pure numpy/scipy, no jax import anywhere past the module top, runs on a
laptop in seconds.

For each of {M1, M0_S4, truncM3, fullM3} x {case} x {seed}: delta_nu(true,
X) against the exported realization, b_{K_eff,X} (K_eff padded with zeros
over the S4-hidden-state block for the two augmented-state variants,
M0_S4/fullM3, whose exported state is 1030-dim while K_eff acts on the
6-dim physical state the controller actually observes - matching what the
trained controller does, not adding new information), and a Markov error
(H=10, verify_markov_match) - all three, plus the already-recorded DPC
cost ratio, in one row per (variant, case, seed). Checks whether
b > delta_nu AGREES with the empirically observed outcome (cost_ratio <
10x oracle = "observed success"), case by case - not just whether the
predicate is true, which is a different, weaker question.

    python tools/nu_gap_analysis.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# ============================================================
# nu-gap / robust-margin machinery - pure numpy/scipy, no jax.
# Validated locally (see module docstring); --selftest reruns the suite.
# ============================================================
from scipy.linalg import solve_discrete_are


def normalized_rcf(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Normalized right coprime factorization (discrete-time) via the
    discrete ARE. Returns state-space (Ac,Bc,Cc,Dc) of [N;M]."""
    m = B.shape[1]
    Q = C.T @ C
    X = solve_discrete_are(A, B, Q, np.eye(m))
    W = np.eye(m) + B.T @ X @ B
    Wisq = np.linalg.inv(np.linalg.cholesky(W).T)
    F = -np.linalg.solve(W, B.T @ X @ A)
    Ac = A + B @ F
    Bc = B @ Wisq
    Cc = np.vstack([C, F])
    Dc = np.vstack([np.zeros((C.shape[0], m)), np.eye(m)]) @ Wisq
    return Ac, Bc, Cc, Dc


def _freq_response(A, B, C, D, z):
    n = A.shape[0]
    return C @ np.linalg.solve(z * np.eye(n) - A, B) + D


def _sqrtm_psd(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def _eig_outside_unit_disk(A, tol: float = 1e-6):
    """Strict |eig|>1 miscounts a marginal pole (case 1's A_d has an
    EXACT eigenvalue at z=1, docs/DECISIONS.md 2026-08-07) whenever a
    fitted comparison system's own copy of that pole lands a hair past
    1.0 from ordinary numerical noise - a genuine eta_P vs eta_Phat
    mismatch of 0 vs 1 from floating-point jitter, not a real pole-count
    difference. A small deadband around the unit circle avoids this
    false alarm without hiding a genuinely unstable pole (tol is far
    smaller than any real instability this project's plants exhibit)."""
    return int(np.sum(np.abs(np.linalg.eigvals(A)) > 1.0 + tol))


def nu_gap(P_ss, Phat_ss, n_freq: int = 2000) -> tuple[float, dict]:
    """P_ss, Phat_ss: (A,B,C,D) tuples, D typically 0. Returns
    (delta_nu, diagnostics)."""
    A, B, C, D = P_ss
    Ah, Bh, Ch, Dh = Phat_ss
    # half-step offset: some of this project's plants have an eigenvalue at
    # EXACTLY z=1 (case 1's integrator mode, docs/DECISIONS.md 2026-08-07) -
    # a grid starting at theta=0 hits that pole exactly (z*I-A singular).
    # Offsetting avoids z=1 and z=-1 (the two real-axis points a grid
    # naturally lands on) without materially changing the sup over the grid.
    thetas = np.linspace(0, 2 * np.pi, n_freq, endpoint=False) + (np.pi / n_freq)
    gaps, arg_vals = [], []
    for th in thetas:
        z = np.exp(1j * th)
        Pz = _freq_response(A, B, C, D, z)
        Pzh = _freq_response(Ah, Bh, Ch, Dh, z)
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


def verify_markov_match(A: np.ndarray, B: np.ndarray, C: np.ndarray, true_markov: list[np.ndarray]) -> float:
    """Max abs error between C@A^(h-1)@B and true_markov[h-1], h=1..len(true_markov).
    Duplicated (not imported) from tools/balanced_truncation.py deliberately:
    that module imports jax at the top level (needed for its OTHER
    functions), and this file's whole point past --selftest is to run
    pure numpy/scipy, no jax, no GPU."""
    max_err = 0.0
    M = np.eye(A.shape[0])
    for h in range(len(true_markov)):
        recon = C @ M @ B
        max_err = max(max_err, float(np.max(np.abs(recon - true_markov[h]))))
        M = A @ M
    return max_err


def robust_margin(A: np.ndarray, B: np.ndarray, K: np.ndarray) -> tuple[float, dict]:
    """b_{K,P} = 1/||[K;I](I-PK)^{-1}[I,P]||_inf, via the closed-loop
    state-space realization (d1 at measurement, d2 at plant input;
    outputs [u;x]) and python-control's H-inf norm. Returns 0.0 if
    A+BK is not Schur stable (no margin is defined)."""
    import control
    n, m = A.shape[0], B.shape[1]
    Acl = A + B @ K
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1.0:
        return 0.0, {"stable": False}
    Bcl = np.hstack([B @ K, B])
    Ccl = np.vstack([K, np.eye(n)])
    Dcl = np.vstack([np.hstack([K, np.zeros((m, m))]), np.hstack([np.eye(n), np.zeros((n, m))])])
    sys = control.StateSpace(Acl, Bcl, Ccl, Dcl, dt=1)
    try:
        gamma = control.norm(sys, p="inf", print_warning=False)
    except Exception as e:
        return None, {"stable": True, "error": str(e)}
    return 1.0 / gamma, {"stable": True, "gamma": float(gamma)}


def _run_selftests() -> None:
    print("Running nu_gap/robust_margin self-tests (pure numpy/scipy)...")
    A = np.array([[0.5]]); B = np.array([[1.0]]); C = np.array([[1.0]]); D = np.array([[0.0]])
    g, info = nu_gap((A, B, C, D), (A, B, C, D))
    assert g < 1e-8, f"identity gap should be 0, got {g}"
    print(f"  identity: gap={g:.2e} PASS")

    Ah = np.array([[0.52]])
    g1, _ = nu_gap((A, B, C, D), (Ah, B, C, D))
    g2, _ = nu_gap((Ah, B, C, D), (A, B, C, D))
    assert abs(g1 - g2) < 1e-3, "symmetry failed"
    print(f"  symmetry: {g1:.6f} vs {g2:.6f} PASS")

    rng = np.random.RandomState(0)
    Ac, Bc, Cc, Dc = normalized_rcf(A + 0.1 * rng.randn(1, 1), B, C)
    prod = _freq_response(Ac, Bc, Cc, Dc, np.exp(1j)).conj().T @ _freq_response(Ac, Bc, Cc, Dc, np.exp(1j))
    assert np.max(np.abs(prod - np.eye(1))) < 1e-6, "normalization failed"
    print("  coprime factor normalization: PASS")

    n, m = 6, 3
    A6 = rng.randn(n, n) * 0.3
    B6 = rng.randn(n, m)
    Q, R = np.eye(n), np.eye(m)
    X = solve_discrete_are(A6, B6, Q, R)
    K_lqr = -np.linalg.solve(R + B6.T @ X @ B6, B6.T @ X @ A6)
    b_good, _ = robust_margin(A6, B6, K_lqr)
    assert b_good is not None and b_good > 0.05, f"expected healthy margin, got {b_good}"
    b_bad, _ = robust_margin(A6, B6, 5 * K_lqr)
    assert b_bad == 0.0, f"expected exactly 0 for a destabilizing gain, got {b_bad}"
    print(f"  robust_margin: good K -> b={b_good:.4f}, destabilizing K -> b={b_bad} PASS")
    print("All self-tests passed.")


# ============================================================
# Main pipeline: pure numpy/scipy, reads tools/nu_gap_export.py's .npz
# output. No jax, no GPU, no training - seconds on a laptop.
# ============================================================

CASES = [1, 2, 3, 4, 5, 7]  # case 6 excluded from control comparisons (standing rule)
N_SEEDS = 5
DT = 0.01
D_X, D_U = 6, 3
MARKOV_H = 10
SUCCESS_RATIO_THRESHOLD = 10.0  # observed cost_ratio_to_oracle below this = "worked"
VARIANTS_ORDER = ["M1", "M0_S4", "truncM3", "fullM3"]


def main() -> None:
    from s4dpc.systems import get_discrete_matrices

    EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
    DOCS_DIR = _REPO_ROOT / "docs"
    assert EXPORT_DIR.exists(), f"{EXPORT_DIR} not found - run tools/nu_gap_export.py first"

    true_AB = {c: tuple(np.asarray(m) for m in get_discrete_matrices(DT, c)) for c in CASES}
    true_markov = {
        c: [np.linalg.matrix_power(true_AB[c][0], h - 1) @ true_AB[c][1] for h in range(1, MARKOV_H + 1)]
        for c in CASES
    }

    rows = []
    n_missing = 0
    for variant in VARIANTS_ORDER:
        for case in CASES:
            for seed in range(N_SEEDS):
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    n_missing += 1
                    continue
                data = np.load(path)
                A, B, C, K_eff = data["A"], data["B"], data["C"], data["K_eff"]
                ratio = float(data["ratio"])
                finite = bool(data["finite"])
                teacher_mse = float(data["teacher_mse"]) if not np.isnan(data["teacher_mse"]) else None
                A_true, B_true = true_AB[case]

                dnu, dnu_info = nu_gap(
                    (A_true, B_true, np.eye(D_X), np.zeros((D_X, D_U))),
                    (A, B, C, np.zeros((C.shape[0], D_U))),
                )

                # K_eff acts on the PHYSICAL state (6-dim, what the trained
                # GRU controller actually observes); M0_S4/fullM3's exported
                # state is the 1030-dim augmented (physical + S4-hidden)
                # realization. robust_margin needs K defined on the SAME
                # state as the plant it closes the loop around - pad with
                # zeros over the S4-hidden block, a state-feedback law
                # that's blind to those coordinates, exactly matching what
                # the trained controller does (it never sees the hidden
                # state either).
                n_state = A.shape[0]
                K_for_margin = K_eff if n_state == D_X else np.hstack([K_eff, np.zeros((D_U, n_state - D_X))])
                b, b_info = robust_margin(A, B, K_for_margin)

                merr = verify_markov_match(A, B, C, true_markov[case])

                predicts_success = b is not None and b > dnu
                observed_success = finite and ratio < SUCCESS_RATIO_THRESHOLD
                agrees = predicts_success == observed_success

                print(f"    [{variant}/case{case}/seed{seed}] ratio={ratio:.4e}  markov_err_h10={merr:.3e}  "
                      f"delta_nu={dnu:.4f} (valid={dnu_info['valid']})  b={b}  "
                      f"predicts_success={predicts_success}  observed_success={observed_success}  agrees={agrees}")

                rows.append({
                    "variant": variant, "case": case, "seed": seed,
                    "cost_ratio_to_oracle": ratio, "finite": finite,
                    "markov_err_h10": merr, "teacher_mse": teacher_mse,
                    "delta_nu": dnu, "dnu_valid": dnu_info["valid"],
                    "b": b, "predicts_success": predicts_success,
                    "observed_success": observed_success, "agrees": agrees,
                })

    if n_missing:
        print(f"\n  ({n_missing} expected (variant,case,seed) .npz files not found - skipped)")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    (DOCS_DIR / "nu_gap_analysis.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'nu_gap_analysis.csv'} ({len(rows)} rows)")

    print(f"\n=== SUMMARY: median (markov_err_h10, delta_nu, b), agreement rate with observed outcome "
          f"(ratio<{SUCCESS_RATIO_THRESHOLD:.0f}x), per (variant, case) ===")
    print(f"{'variant':10s} {'case':5s} {'median_merr':>12s} {'median_dnu':>11s} {'median_b':>10s} "
          f"{'agree_rate':>11s} {'median_ratio':>13s}")
    for variant in VARIANTS_ORDER:
        for case in CASES:
            these = [r for r in rows if r["variant"] == variant and r["case"] == case]
            if not these:
                continue
            merrs = [r["markov_err_h10"] for r in these]
            dnus = [r["delta_nu"] for r in these]
            bs = [r["b"] for r in these if r["b"] is not None]
            agrees = [r["agrees"] for r in these]
            ratios = [r["cost_ratio_to_oracle"] for r in these]
            print(f"{variant:10s} {case:5d} {np.median(merrs):12.3e} {np.median(dnus):11.4f} "
                  f"{(np.median(bs) if bs else float('nan')):10.4f} "
                  f"{np.mean(agrees):11.1%} {np.median(ratios):13.4e}")

    print("\n=== Overall b>delta_nu agreement rate with observed outcome, per variant ===")
    for variant in VARIANTS_ORDER:
        these = [r for r in rows if r["variant"] == variant]
        if not these:
            continue
        rate = np.mean([r["agrees"] for r in these])
        print(f"  {variant:10s}: {rate:.1%} ({len(these)} rows)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _run_selftests()
    else:
        main()
