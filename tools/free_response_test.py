"""TASK B (user, 2026-08-18, third round): the actual hypothesis under test.

Every fidelity metric used this project so far - Markov parameters, teacher-
forced MSE, impulse response - measures the FORCED response from rest (s=0,
x=0 or x=APRBS-excited-from-near-zero). Every control-relevant rollout starts
from x0 != 0. A surrogate can match its response to excitation from rest to
1e-6 and still carry internal modes that a rest-excited input never reveals,
if those modes only become visible from a genuinely nonzero initial
condition. This tests that directly, on the already-extracted linear
operators (M3 is exactly LTI, no new GPU work):

  1. FREE RESPONSE: drive the true plant and M3 from matched initial
     conditions with u=0 for 200 steps. "Matched" needs a choice for the
     surrogate's internal state s0, since x0 alone doesn't determine it:
       (a) OBSERVER-DERIVED s0 - burn in the SAME observer recursion
           tools/lqr_transfer_to_true_plant.py uses (s_hat_{k+1} = Asx@x_true
           + Ass@s_hat, u=0) for T_BURN=50 steps on the true free-decay
           trajectory starting from a random seed state, THEN take x0 :=
           x_true(T_BURN) and s0 := s_hat(T_BURN) - a "genuinely arrived at"
           internal state, not an arbitrary one.
       (b) s0 = 0, paired with the SAME x0 (only s0 differs, isolating it as
           the one free variable).
     From z0=[x0,s0], M3 runs its OWN full free dynamics (z_{k+1}=Abar@z_k,
     using Axx/Axs too, not driven by true x anymore - this is what M3 would
     actually do if deployed open-loop) for 200 steps; compared against the
     true plant continuing its own free decay from the same x0.

  2. FREQUENCY RESPONSE: H_M3(z) = C(zI-Abar)^-1 Bbar vs H_true(z) =
     (zI-A_true)^-1 B_true, evaluated over a grid z=e^{jw}, w in (0, pi] -
     unlike the finite-horizon Markov-parameter comparison, this can show
     disagreement concentrated near the unit circle even when the truncated
     impulse response looks fine.

M1 and M0_S4 run as controls (both should track near-exactly - M0_S4 by
construction, M1 because it has no extra internal state for a mismatched s0
to live in). If free responses diverge while impulse responses already agree
to 1e-6, that's the discriminating quantity this project was missing. If
free responses also agree, the hypothesis is dead and gets reported as such.

    python tools/free_response_test.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.systems import get_discrete_matrices

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
T_BURN = 50
HORIZON = 200
EVAL_BATCH, EVAL_X0_RANGE = 50, 5.0
FREQ_GRID = np.linspace(1e-3, np.pi, 400)


def get_seed_x0_batch(case: int) -> np.ndarray:
    """Same PRNG convention as lqr_transfer_to_true_plant.get_x0_batch, so this
    is reproducible against the rest of the project - smaller batch (50 vs
    100) since this is CPU-only free-response simulation, not a bottleneck
    either way, kept modest for fast iteration."""
    import jax
    eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
    x0 = jax.random.uniform(eval_key, (EVAL_BATCH, D_X), minval=-EVAL_X0_RANGE, maxval=EVAL_X0_RANGE)
    return np.asarray(x0, dtype=np.float64)


def free_response_true(A_true: np.ndarray, x_seed_batch: np.ndarray, n_steps: int) -> np.ndarray:
    """Returns (n_steps+1, batch, D_X): x_true(t) = A_true^t @ x_seed, t=0..n_steps."""
    xs = [x_seed_batch]
    x = x_seed_batch
    for _ in range(n_steps):
        x = x @ A_true.T
        xs.append(x)
    return np.stack(xs)


def observer_burn_in(Asx: np.ndarray, Ass: np.ndarray, x_true_hist: np.ndarray) -> np.ndarray:
    """x_true_hist: (T_BURN+1, batch, D_X), s_hat(0)=0, s_hat_{k+1}=Asx@x_true_k+Ass@s_hat_k
    (u=0). Returns s_hat at t=T_BURN, shape (batch, n_s)."""
    batch = x_true_hist.shape[1]
    n_s = Ass.shape[0]
    s_hat = np.zeros((batch, n_s))
    for k in range(x_true_hist.shape[0] - 1):
        s_hat = x_true_hist[k] @ Asx.T + s_hat @ Ass.T
    return s_hat


def free_run_surrogate(Abar: np.ndarray, z0: np.ndarray, n_steps: int) -> np.ndarray:
    """z0: (batch, n_total). Returns (n_steps+1, batch, D_X) - the x-block only."""
    zs = [z0[:, :D_X]]
    z = z0
    for _ in range(n_steps):
        z = z @ Abar.T
        zs.append(z[:, :D_X])
    return np.stack(zs)


def rel_error_over_time(x_pred: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """Per-step relative L2 error, averaged over batch. x_pred/x_true: (T+1, batch, D_X)."""
    num = np.linalg.norm(x_pred - x_true, axis=-1)
    den = np.linalg.norm(x_true, axis=-1)
    den = np.where(den < 1e-12, 1e-12, den)
    return np.mean(num / den, axis=-1)


def transfer_function(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """Returns (len(omega_grid), C.shape[0], B.shape[1]) complex H(e^{jw}) = C(zI-A)^-1 B.
    One eigendecomposition (A = V diag(lambda) V^-1) turns the O(n^3)-per-frequency
    direct solve into an O(n^2)-per-frequency diagonal scaling - at n=1030 with a
    400-point grid, the direct-solve version is minutes-to-hours slower and was not
    used. (zI-A)^-1 = V diag(1/(z-lambda)) V^-1, so H(z) = (CV) diag(1/(z-lambda)) (V^-1 B)."""
    eigvals, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    CV = C @ V
    VinvB = Vinv @ B
    z = np.exp(1j * omega_grid)  # (n_freq,)
    inv_diag = 1.0 / (z[:, None] - eigvals[None, :])  # (n_freq, n)
    H = np.einsum("pi,fi,ij->fpj", CV, inv_diag, VinvB)
    return H


def main() -> None:
    free_rows = []
    freq_rows = []
    for case in CASES:
        A_true, B_true = (np.asarray(m) for m in get_discrete_matrices(DT, case))
        x_seed = get_seed_x0_batch(case)
        x_true_burn = free_response_true(A_true, x_seed, T_BURN)  # (T_BURN+1, batch, D_X)
        x0 = x_true_burn[-1]  # (batch, D_X) - the shared initial condition for both s0 choices
        x_true_test = free_response_true(A_true, x0, HORIZON)  # (HORIZON+1, batch, D_X) - ground truth to compare against

        H_true = transfer_function(A_true, B_true, np.eye(D_X), FREQ_GRID)

        for seed in range(N_SEEDS):
            for variant in ["fullM3", "M0_S4", "M1"]:
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A, B, C = data["A"], data["B"], data["C"]
                n_total = A.shape[0]
                n_s = n_total - D_X

                if n_s > 0:
                    Asx, Ass = A[D_X:, :D_X], A[D_X:, D_X:]
                    s0_observer = observer_burn_in(Asx, Ass, x_true_burn)
                    s0_zero = np.zeros((x0.shape[0], n_s))
                    z0_observer = np.hstack([x0, s0_observer])
                    z0_zero = np.hstack([x0, s0_zero])
                    s0_choices = {"observer": z0_observer, "zero": z0_zero}
                else:
                    s0_choices = {"observer": x0, "zero": x0}  # M1: no internal state, both identical

                for s0_label, z0 in s0_choices.items():
                    x_pred = free_run_surrogate(A, z0, HORIZON)
                    err = rel_error_over_time(x_pred, x_true_test)
                    finite = bool(np.all(np.isfinite(x_pred)))
                    free_rows.append({
                        "variant": variant, "case": case, "seed": seed, "s0_choice": s0_label,
                        "err_t1": float(err[1]), "err_t10": float(err[min(10, len(err) - 1)]),
                        "err_t50": float(err[min(50, len(err) - 1)]),
                        "err_t200": float(err[-1]) if finite else float("inf"),
                        "finite": finite,
                    })

                H_m3 = transfer_function(A, B, C, FREQ_GRID)
                diff = np.linalg.norm((H_m3 - H_true).reshape(len(FREQ_GRID), -1), axis=-1)
                true_norm = np.linalg.norm(H_true.reshape(len(FREQ_GRID), -1), axis=-1)
                rel_diff = diff / np.where(true_norm < 1e-12, 1e-12, true_norm)
                worst_idx = int(np.argmax(rel_diff))
                freq_rows.append({
                    "variant": variant, "case": case, "seed": seed,
                    "max_rel_diff": float(rel_diff[worst_idx]), "worst_omega": float(FREQ_GRID[worst_idx]),
                    "median_rel_diff": float(np.median(rel_diff)), "rel_diff_at_omega0": float(rel_diff[0]),
                })

        print(f"case {case} done")

    for name, rows in [("free_response_test.csv", free_rows), ("frequency_response_test.csv", freq_rows)]:
        out_path = DOCS / name
        header = sorted({k for r in rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path} ({len(rows)} rows)")

    print("\n=== FREE RESPONSE summary (median relative error, by variant x s0_choice) ===")
    for variant in ["fullM3", "M0_S4", "M1"]:
        for s0_label in ["observer", "zero"]:
            these = [r for r in free_rows if r["variant"] == variant and r["s0_choice"] == s0_label]
            if not these:
                continue
            finite_these = [r for r in these if r["finite"]]
            n_diverged = len(these) - len(finite_these)
            if finite_these:
                med_t1 = np.median([r["err_t1"] for r in finite_these])
                med_t200 = np.median([r["err_t200"] for r in finite_these])
            else:
                med_t1 = med_t200 = float("nan")
            print(f"  {variant:8s} s0={s0_label:9s}  median err@t=1: {med_t1:.4e}  "
                  f"median err@t=200: {med_t200:.4e}  diverged: {n_diverged}/{len(these)}")

    print("\n=== FREQUENCY RESPONSE summary (median over checkpoints) ===")
    for variant in ["fullM3", "M0_S4", "M1"]:
        these = [r for r in freq_rows if r["variant"] == variant]
        if not these:
            continue
        print(f"  {variant:8s}  median max_rel_diff: {np.median([r['max_rel_diff'] for r in these]):.4e}"
              f"  median rel_diff_at_omega~0: {np.median([r['rel_diff_at_omega0'] for r in these]):.4e}")


if __name__ == "__main__":
    main()
