"""Direct verification that b=0.0 (nu_gap_analysis.py, docs/nu_gap_analysis.csv)
on M3-based rows is a real closed-loop instability, not a computation artifact -
and the pole-count question behind delta_nu's saturated/invalid winding-number
check. Pure numpy, reads tools/nu_gap_export.py's .npz exports directly - no
jax, no GPU, no retraining.

TASK A: for one M3 case-3 controller (b=0.0 reported), report the closed-loop
eigenvalues against M3 itself AND against the true plant (same K_eff, different
plant), then simulate the M3 closed loop for 2000+ steps (well past the
N=200 training cap) and check whether the state diverges. Same for one M1
controller as the control (expected: stable, bounded).

TASK B (part 1): count eigenvalues of the FULL M3 augmented operator with
|lambda|>1, per case and seed, against the true plant's own count (computed
directly, not via the deadbanded _eig_outside_unit_disk used for the nu-gap
winding-number check elsewhere - this is a literal, undeadbanded count).

    python tools/verify_closed_loop_instability.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.systems import get_discrete_matrices

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
DT = 0.01
D_X, D_U = 6, 3
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
TRAIN_X0_RANGE = 3.0  # tools/controller_oracles.py


def load(variant: str, case: int, seed: int):
    return np.load(EXPORT_DIR / f"{variant}_{case}_{seed}.npz")


def closed_loop(A: np.ndarray, B: np.ndarray, K_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    K_pad = K_eff if n == D_X else np.hstack([K_eff, np.zeros((D_U, n - D_X))])
    return A + B @ K_pad, K_pad


def task_a() -> None:
    print("=" * 20 + " TASK A: is b=0.0 real? " + "=" * 20)
    case, seed = 3, 0
    A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))

    m3 = load("fullM3", case, seed)
    A3, B3, C3, K3 = m3["A"], m3["B"], m3["C"], m3["K_eff"]
    print(f"\n-- M3 case{case} seed{seed} (b={float(m3['ratio']):.3e}x oracle, reported b=0.0) --")

    Acl_m3, _ = closed_loop(A3, B3, K3)
    eig_m3 = np.linalg.eigvals(Acl_m3)
    n_unstable = int(np.sum(np.abs(eig_m3) > 1.0))
    worst = eig_m3[np.argmax(np.abs(eig_m3))]
    print(f"  closed loop vs M3 itself: spectral radius = {np.max(np.abs(eig_m3)):.6f}, "
          f"{n_unstable}/{len(eig_m3)} eigenvalues with |lambda|>1")
    print(f"  worst eigenvalue: {worst:.6f}  (|.|={abs(worst):.6f})")
    top5 = eig_m3[np.argsort(-np.abs(eig_m3))[:5]]
    print(f"  top-5 |eig|: {sorted(np.abs(eig_m3), reverse=True)[:5]}")

    # SAME K_eff, closed around the TRUE plant instead (6-dim, no padding needed)
    Acl_true, _ = closed_loop(A_true, B_true, K3)
    eig_true = np.linalg.eigvals(Acl_true)
    print(f"\n  SAME controller (K_eff), closed around the TRUE plant instead:")
    print(f"    spectral radius = {np.max(np.abs(eig_true)):.6f}, all eig = {np.round(eig_true, 4)}")

    # 2000+ step simulation on the M3 closed loop
    rng = np.random.RandomState(0)
    x0_phys = rng.uniform(-TRAIN_X0_RANGE, TRAIN_X0_RANGE, size=D_X)
    z0 = np.zeros(A3.shape[0])
    z0[:D_X] = x0_phys
    N_SIM = 2500
    z = z0.copy()
    norms = np.zeros(N_SIM + 1)
    norms[0] = np.linalg.norm(C3 @ z)
    for k in range(1, N_SIM + 1):
        z = Acl_m3 @ z
        norms[k] = np.linalg.norm(C3 @ z)
        if not np.isfinite(norms[k]):
            norms[k:] = np.inf
            break
    print(f"\n  M3 closed-loop simulation, {N_SIM} steps from a training-range x0:")
    for k in [0, 50, 100, 200, 500, 1000, 1500, 2000, 2500]:
        if k <= N_SIM:
            print(f"    ||x_{k}|| = {norms[k]:.6e}")
    diverged = not np.isfinite(norms[-1]) or norms[-1] > 1e6
    print(f"  DIVERGES past training horizon (N=200): {diverged}")

    # ---- M1 control, same case/seed ----
    m1 = load("M1", case, seed)
    A1, B1, C1, K1 = m1["A"], m1["B"], m1["C"], m1["K_eff"]
    print(f"\n-- M1 case{case} seed{seed} (control, ratio={float(m1['ratio']):.4e}x oracle) --")
    Acl_m1, _ = closed_loop(A1, B1, K1)
    eig_m1 = np.linalg.eigvals(Acl_m1)
    print(f"  closed loop vs M1: spectral radius = {np.max(np.abs(eig_m1)):.6f}")
    z = z0[:D_X].copy()
    norms1 = np.zeros(N_SIM + 1)
    norms1[0] = np.linalg.norm(z)
    for k in range(1, N_SIM + 1):
        z = Acl_m1 @ z
        norms1[k] = np.linalg.norm(z)
    print(f"  M1 closed-loop simulation, {N_SIM} steps: ||x_0||={norms1[0]:.4f} -> ||x_{N_SIM}||={norms1[-1]:.6e}")
    print(f"  DIVERGES: {norms1[-1] > 1e6 or not np.isfinite(norms1[-1])}")


def task_b_pole_count() -> None:
    print("\n\n" + "=" * 20 + " TASK B (part 1): unstable pole counts, fullM3 vs true plant " + "=" * 20)
    print(f"{'case':5s} {'seed':5s} {'true_unstable':>14s} {'true_rho':>10s} {'M3_unstable':>12s} {'M3_rho':>10s} {'M3_n_state':>11s}")
    rows = []
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        eig_true = np.linalg.eigvals(A_true)
        n_true_unstable = int(np.sum(np.abs(eig_true) > 1.0))
        rho_true = float(np.max(np.abs(eig_true)))
        for seed in range(N_SEEDS):
            path = EXPORT_DIR / f"fullM3_{case}_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            A3 = data["A"]
            eig3 = np.linalg.eigvals(A3)
            n_unstable = int(np.sum(np.abs(eig3) > 1.0))
            rho3 = float(np.max(np.abs(eig3)))
            print(f"{case:<5d} {seed:<5d} {n_true_unstable:>14d} {rho_true:>10.6f} "
                  f"{n_unstable:>12d} {rho3:>10.6f} {A3.shape[0]:>11d}")
            rows.append({"case": case, "seed": seed, "n_true_unstable": n_true_unstable, "rho_true": rho_true,
                         "n_m3_unstable": n_unstable, "rho_m3": rho3})
    return rows


if __name__ == "__main__":
    task_a()
    task_b_pole_count()
