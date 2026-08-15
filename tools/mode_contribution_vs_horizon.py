"""TASK C (user, 2026-08-15): is the horizon-blindness mechanism real?
If a slow/spurious mode's contribution to the DPC cost is negligible at the
N=200 training cap and grows past it, that's a measurable, falsifiable claim,
not just a story - and it predicts what would fix it (a cost that doesn't
vanish on slow modes: terminal penalty, non-decaying discount).

Pure numpy - decomposes M3 case3's closed-loop operator (tools/
verify_closed_loop_instability.py already showed 32/1030 genuinely unstable
eigenvalues, spectral radius 1.05, and 54-orders-of-magnitude divergence by
step 2500) into eigenmodes, groups them into real conjugate pairs, and
computes EACH unstable mode's own contribution to the same normalized
quadratic cost tools/controller_oracles.py trains against
(Q_x=5.0, R_u=0.1, Q_f=50.0), in isolation, as a function of horizon N.

    python tools/mode_contribution_vs_horizon.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
D_X, D_U = 6, 3
Q_X, R_U, Q_F = 5.0, 0.1, 50.0
TRAIN_X0_RANGE = 3.0
N_GRID = [5, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 1500, 2000]
TRAIN_N = 200


def closed_loop(A, B, K_eff):
    n = A.shape[0]
    K_pad = K_eff if n == D_X else np.hstack([K_eff, np.zeros((D_U, n - D_X))])
    return A + B @ K_pad, K_pad


def group_conjugate_pairs(eigvals: np.ndarray) -> list[list[int]]:
    n = len(eigvals)
    used = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if used[i]:
            continue
        if abs(eigvals[i].imag) < 1e-8:
            groups.append([i])
            used[i] = True
            continue
        target = np.conj(eigvals[i])
        cand = [j for j in range(n) if (not used[j]) and j != i
                and abs(eigvals[j] - target) < 1e-6 * max(1.0, abs(target))]
        if cand:
            j = cand[0]
            groups.append([i, j])
            used[i] = used[j] = True
        else:
            groups.append([i])  # shouldn't happen for a real matrix, defensive fallback
            used[i] = True
    return groups


def mode_cost_curve(V, w0, K_pad, C, idx: list[int], eigvals: np.ndarray, n_max: int) -> np.ndarray:
    """Isolated cost contribution of ONE eigengroup (real singlet or a
    conjugate pair reconstructed to a real trajectory), as raw (unnormalized)
    running stage cost at each step 0..n_max-1, plus we add the terminal term
    separately per N when reporting. Returns per-step stage-cost array
    (length n_max), so the caller can sum/normalize for any horizon <= n_max."""
    Vg = V[:, idx]
    wg = w0[idx]
    stage = np.zeros(n_max)
    lam_pow = np.ones(len(idx), dtype=complex)
    for k in range(n_max):
        z_k = np.real(Vg @ (lam_pow * wg))
        x_k = C @ z_k
        u_k = K_pad @ z_k
        stage[k] = Q_X * np.sum(x_k ** 2) + R_U * np.sum(u_k ** 2)
        lam_pow = lam_pow * eigvals[idx]
    return stage, Vg, wg  # stage[k] = contribution to the k-th stage cost term


def terminal_cost_at(V, w0, eigvals, idx, C, N) -> float:
    lam_N = eigvals[idx] ** N
    z_N = np.real(V[:, idx] @ (lam_N * w0[idx]))
    x_N = C @ z_N
    return Q_F * float(np.sum(x_N ** 2))


def main() -> None:
    case, seed = 3, 0
    data = np.load(EXPORT_DIR / f"fullM3_{case}_{seed}.npz")
    A, B, C, K_eff = data["A"], data["B"], data["C"], data["K_eff"]
    Acl, K_pad = closed_loop(A, B, K_eff)
    n_state = Acl.shape[0]

    print(f"Eigendecomposing M3 case{case} seed{seed}'s closed-loop operator ({n_state}x{n_state})...")
    eigvals, V = np.linalg.eig(Acl)
    Vinv_z0_solver = V  # solve per-x0 below

    rng = np.random.RandomState(0)
    x0_phys = rng.uniform(-TRAIN_X0_RANGE, TRAIN_X0_RANGE, size=D_X)
    z0 = np.zeros(n_state)
    z0[:D_X] = x0_phys
    w0 = np.linalg.solve(V, z0.astype(complex))

    groups = group_conjugate_pairs(eigvals)
    unstable_groups = [g for g in groups if abs(eigvals[g[0]]) > 1.0]
    print(f"{len(groups)} eigengroups total ({sum(len(g) for g in groups)} eigenvalues), "
          f"{len(unstable_groups)} unstable groups (|lambda|>1)")

    n_max = max(N_GRID) + 1
    curves = []
    for g in unstable_groups:
        stage, Vg, wg = mode_cost_curve(V, w0, K_pad, C, g, eigvals, n_max)
        curves.append((g, stage))

    # full-system cost curve for reference (same formula, full trajectory)
    z = z0.copy()
    full_stage = np.zeros(n_max)
    for k in range(n_max):
        x_k = C @ z
        u_k = K_pad @ z
        full_stage[k] = Q_X * np.sum(x_k ** 2) + R_U * np.sum(u_k ** 2)
        z = Acl @ z

    print(f"\nInitial condition: ||x0||={np.linalg.norm(x0_phys):.4f} (uniform in [-{TRAIN_X0_RANGE},{TRAIN_X0_RANGE}]^6)")
    print(f"\n{'lambda':>22s} {'|lambda|':>9s} {'excitation|w0|':>15s}  " +
          "  ".join(f"N={n:<6d}" for n in N_GRID))
    rows_for_report = []
    for g, stage in sorted(curves, key=lambda gs: -np.sum(gs[1][:max(N_GRID)])):
        lam = eigvals[g[0]]
        excitation = float(np.abs(w0[g[0]]))
        cost_at_N = []
        for N in N_GRID:
            stage_sum = float(np.sum(stage[:N]))
            term = terminal_cost_at(V, w0, eigvals, g, C, N)
            cost_at_N.append((stage_sum + term) / N)
        full_at_N = []
        for N in N_GRID:
            stage_sum = float(np.sum(full_stage[:N]))
            z_N = z0.astype(complex)
            zz = z0.copy()
            for _ in range(N):
                zz = Acl @ zz
            term = Q_F * float(np.sum((C @ zz) ** 2))
            full_at_N.append((stage_sum + term) / N)
        lam_str = f"{lam.real:+.4f}{lam.imag:+.4f}j" if abs(lam.imag) > 1e-8 else f"{lam.real:+.6f}"
        print(f"{lam_str:>22s} {abs(lam):>9.5f} {excitation:>15.4f}  " +
              "  ".join(f"{c:<9.2e}" for c in cost_at_N))
        rows_for_report.append((lam, cost_at_N))

    print(f"\n{'FULL SYSTEM (all modes)':>22s} {'':>9s} {'':>15s}  " +
          "  ".join(f"{c:<9.2e}" for c in full_at_N))

    dominant = rows_for_report[0]
    print(f"\nDominant unstable mode: lambda={dominant[0]:.6f}")
    ratio_200_to_2000 = dominant[1][N_GRID.index(200)] / max(dominant[1][N_GRID.index(2000)], 1e-300)
    print(f"  cost contribution at N=200: {dominant[1][N_GRID.index(200)]:.4e}")
    print(f"  cost contribution at N=2000: {dominant[1][N_GRID.index(2000)]:.4e}")
    print(f"  ratio N=200 / N=2000: {ratio_200_to_2000:.4e}")
    print(f"  fraction of FULL system cost at N=200: "
          f"{dominant[1][N_GRID.index(200)] / full_at_N[N_GRID.index(200)]:.4%}")
    print(f"  fraction of FULL system cost at N=2000: "
          f"{dominant[1][N_GRID.index(2000)] / full_at_N[N_GRID.index(2000)]:.4%}")


if __name__ == "__main__":
    main()
