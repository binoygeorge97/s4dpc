"""TASK A (user, 2026-08-18): verify the observer construction (tools/
lqr_transfer_to_true_plant.py) does what it claims, and produce the
transient/state-norm-vs-time figure.

(1) Compares s_hat (M3's internal recursion driven by the TRUE, measured
    x/u - what the transfer construction actually uses) against s_free
    (the SAME recursion driven by M3's OWN self-referential x-prediction -
    i.e. M3 running completely on its own, no true plant involved at all),
    from the SAME initial condition and the SAME control sequence, to check
    whether they track each other for a while before diverging, or diverge
    immediately - qualifies "exact linear predictor" to "exact GIVEN
    ground-truth driving," which is the claim actually needed and the more
    defensible one anyway.

(2) Logs ||x|| and ||s|| vs timestep for: M1 (control, true plant, stable),
    M0_S4 (control, true plant via the observer construction, stable), M3
    (test, true plant via the observer construction, unstable), and M3
    free-running (reference - M3 on its own turf, no true plant) - one
    representative case (case 3, seed 0, this session's workhorse case
    throughout). Writes a CSV (all four trajectories) and a PNG plot.

Reuses docs/lqr_cache/*.npz (K_lqr already solved by tools/
lqr_transfer_to_true_plant.py) - no new DARE solves, pure CPU, seconds.

    python tools/trajectory_comparison.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s4dpc.systems import get_discrete_matrices

EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
CACHE_DIR = _REPO_ROOT / "docs" / "lqr_cache"
DOCS_DIR = _REPO_ROOT / "docs"
CASE, SEED = 3, 0
D_X, D_U = 6, 3
DT = 0.01
N_STEPS = 300  # past the 200-step eval horizon, to show the pattern clearly
EVAL_X0_RANGE = 5.0


def get_x0(case: int) -> np.ndarray:
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (100, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0[0], dtype=np.float64)  # first sample - one clean trajectory to plot


def load_klqr(variant: str, case: int, seed: int) -> np.ndarray:
    return np.load(CACHE_DIR / f"{variant}_{case}_{seed}.npz")["K_lqr"]


def main() -> None:
    A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, CASE))
    x0 = get_x0(CASE)

    m3 = np.load(EXPORT_DIR / f"fullM3_{CASE}_{SEED}.npz")
    A3, B3 = m3["A"], m3["B"]
    K3 = load_klqr("fullM3", CASE, SEED)
    K3x, K3s = K3[:, :D_X], K3[:, D_X:]
    n_s = A3.shape[0] - D_X
    Axx, Axs = A3[:D_X, :D_X], A3[:D_X, D_X:]
    Asx, Ass = A3[D_X:, :D_X], A3[D_X:, D_X:]
    Bx, Bs = B3[:D_X, :], B3[D_X:, :]

    # ---- (1) M3 driven by TRUTH (observer, s_hat) vs M3 free-running (s_free) ----
    # SAME control law (u = -K3x@x - K3s@s), SAME x0 - but the "truth" arm's x
    # comes from the TRUE plant, while the "free-running" arm's x comes from
    # M3's OWN prediction. Comparing s_hat vs s_free isolates exactly what
    # substituting the true x for M3's self-referential x changes.
    x_true, s_hat = x0.copy(), np.zeros(n_s)
    x_free, s_free = x0.copy(), np.zeros(n_s)
    rows = []
    for k in range(N_STEPS):
        u_truth = -K3x @ x_true - K3s @ s_hat
        u_free = -K3x @ x_free - K3s @ s_free

        rows.append({
            "step": k, "norm_x_true": float(np.linalg.norm(x_true)), "norm_s_hat": float(np.linalg.norm(s_hat)),
            "norm_x_free": float(np.linalg.norm(x_free)), "norm_s_free": float(np.linalg.norm(s_free)),
            "s_hat_vs_s_free_diff": float(np.linalg.norm(s_hat - s_free)),
        })

        s_hat_next = Asx @ x_true + Ass @ s_hat + Bs @ u_truth
        x_true_next = A_true @ x_true + B_true @ u_truth

        x_free_next = Axx @ x_free + Axs @ s_free + Bx @ u_free
        s_free_next = Asx @ x_free + Ass @ s_free + Bs @ u_free

        x_true, s_hat = x_true_next, s_hat_next
        x_free, s_free = x_free_next, s_free_next

        if not (np.all(np.isfinite(x_true)) and np.all(np.isfinite(s_hat))
                and np.all(np.isfinite(x_free)) and np.all(np.isfinite(s_free))):
            print(f"  overflowed to inf/nan at step {k}, stopping early")
            break

    (DOCS_DIR / "trajectory_observer_vs_freerun.csv").write_text(
        ",".join(rows[0].keys()) + "\n" + "\n".join(",".join(str(r[k]) for k in r) for r in rows)
    )
    print(f"wrote {DOCS_DIR / 'trajectory_observer_vs_freerun.csv'}")

    # when does s_hat vs s_free diverge relative to when x_true blows up?
    diffs = [r["s_hat_vs_s_free_diff"] for r in rows]
    norm_x_true = [r["norm_x_true"] for r in rows]
    first_1pct = next((i for i, d in enumerate(diffs) if d > 0.01 * max(diffs[0], 1e-12)), None)
    first_x_10x = next((i for i, v in enumerate(norm_x_true) if v > 10 * norm_x_true[0]), None)
    print(f"  s_hat vs s_free: step 0 diff={diffs[0]:.6e}, diverges past 1% of final scale at step "
          f"{first_1pct}; x_true first exceeds 10x its initial norm at step {first_x_10x}")

    # ---- (2) trajectory comparison plot: M1, M0_S4, M3(transfer), M3(free-running) ----
    def simulate_transfer(variant: str) -> list[float]:
        data = np.load(EXPORT_DIR / f"{variant}_{CASE}_{SEED}.npz")
        A, B = data["A"], data["B"]
        K = load_klqr(variant, CASE, SEED)
        n = A.shape[0]
        if n == D_X:
            Kx, Ks, n_s_ = K, np.zeros((D_U, 0)), 0
        else:
            Kx, Ks, n_s_ = K[:, :D_X], K[:, D_X:], n - D_X
        Asx_, Ass_, Bs_ = (A[D_X:, :D_X], A[D_X:, D_X:], B[D_X:, :]) if n_s_ else (None, None, None)

        x, s = x0.copy(), (np.zeros(n_s_) if n_s_ else None)
        norms = []
        for _ in range(N_STEPS):
            u = -Kx @ x - (Ks @ s if n_s_ else 0.0)
            norms.append(float(np.linalg.norm(x)))
            x_next = A_true @ x + B_true @ u
            if n_s_:
                s = Asx_ @ x + Ass_ @ s + Bs_ @ u
            x = x_next
            if not np.all(np.isfinite(x)):
                norms.extend([np.inf] * (N_STEPS - len(norms)))
                break
        return norms

    norms_m1 = simulate_transfer("M1")
    norms_m0s4 = simulate_transfer("M0_S4")
    norms_m3_transfer = [r["norm_x_true"] for r in rows]  # already computed above
    norms_m3_transfer += [norms_m3_transfer[-1] * float("inf")] * (N_STEPS - len(norms_m3_transfer)) \
        if len(norms_m3_transfer) < N_STEPS else []
    norms_m3_free = [r["norm_x_free"] for r in rows]
    norms_m3_free += [norms_m3_free[-1] * float("inf")] * (N_STEPS - len(norms_m3_free)) \
        if len(norms_m3_free) < N_STEPS else []

    steps = list(range(len(norms_m1)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(steps, norms_m1, label="M1 (true plant, control)", color="#2c7fb8", linewidth=2)
    ax.plot(range(len(norms_m0s4)), norms_m0s4, label="M0_S4 (observer, control)", color="#31a354", linewidth=2)
    ax.plot(range(len(norms_m3_transfer)), norms_m3_transfer,
             label="M3 (observer -> true plant, test)", color="#e34a33", linewidth=2.5)
    ax.plot(range(len(norms_m3_free)), norms_m3_free,
             label="M3 (free-running, its own turf)", color="#e34a33", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(200, color="gray", linestyle=":", linewidth=1, label="200-step eval horizon")
    ax.set_yscale("log")
    ax.set_xlabel("timestep")
    ax.set_ylabel(r"$\|x_k\|$ (physical state norm, log scale)")
    ax.set_title(f"Case {CASE}, seed {SEED}: LQR-optimal controller for M3, transferred to the true plant")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig_path = DOCS_DIR / "figures" / "lqr_transfer_trajectory.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
