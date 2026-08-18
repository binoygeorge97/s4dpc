"""TASK A (user, 2026-08-18, fifth round): make the (Axx, Axs) non-
identifiability a theorem, not a hypothesis.

The user's reframing: the 91.7% Axx-vs-A_d error is not "the model got the
dynamics wrong" - it's gauge freedom. Teacher-forced training only
constrains x_next = Axx@x + Axs@s + Bx@u along the manifold traced out by
the ACTUAL training trajectory, where s is a fixed deterministic function
of that same trajectory's (x, u) history (s evolves via a SEPARATE
recursion, s_{k+1}=Asx@x_k+Ass@s_k+Bs@u_k, that does not involve Axx/Axs
at all). Any (Axx, Axs) that agrees with the true dynamics on that
manifold fits the training data equally well - the loss cannot see the
difference.

The user asked for two things, and this script shows they are the SAME
computation, not two separate ones:

  (1) stack the teacher-forced (x, s) pairs from the real training
      trajectory (regenerated deterministically - same DATA_SEED,
      same case_data() as actual identification) and report the
      singular-value spectrum of the joint matrix Z = [X | S].

  (2) the Fisher/Gauss-Newton null space restricted to (Axx, Axs)
      coordinates, holding every other parameter (Bx, and the s-recursion
      itself) fixed. Since x_hat_{t+1} = Axx@x_t + Axs@s_t + Bx@u_t is
      EXACTLY LINEAR in (Axx, Axs) with s_t held fixed (s_t doesn't depend
      on Axx/Axs at all - it's produced by separate parameters), the
      Gauss-Newton Hessian w.r.t. (Axx, Axs), per output row, is EXACTLY
      Z^T Z (same for every row, since the design vector z_t=[x_t;s_t]
      doesn't depend on which output row is being predicted). So:

          null space of the (Axx,Axs) Gauss-Newton Hessian == null space of Z

      and a null vector v=[v_x; v_s] of Z (z_t . v = 0 for every t in the
      training data) is EXACTLY a flat direction: theta -> theta + alpha*v
      changes NEITHER Axx@x_t+Axs@s_t at any training t, NOR the loss, for
      any alpha - literally "move dynamics from Axx into Axs" as a
      zero-eigenvalue direction of the real, local Gauss-Newton Hessian,
      not a heuristic. No raw model parameters or GPU needed for this -
      only the already-extracted Asx/Ass/Bs (nu_gap_export) and the
      already-deterministic training data generator (s4dpc.data).

CAVEAT stated up front, not buried: training used ONE trajectory of length
L=100 per case (batch_size=1, s4dpc.data.generate_microgrid_trajectory).
Z is (100, 6+n_s) - with n_s=1024 (fullM3, d1030) or 0 (M1), Z is TRIVIALLY
rank <= 100 just from having fewer samples than columns, regardless of any
non-identifiability story. The informative question is NOT "is rank(Z) <
6+n_s" (guaranteed) but "is rank(Z) << 100" (the sample-count ceiling) -
i.e. is there a cliff well short of full row rank. A random 100x(6+n_s)
Gaussian matrix is full row rank (100) almost surely, so this script also
reports that null control explicitly, not just the true-trajectory result.

Runs for fullM3 and M0_S4 (control - if M0_S4 ALSO shows the same data-side
collinearity despite its own Axs being architecturally zero, that
confirms the user's framing precisely: the gauge orbit is a property of
the DATA/architecture, M0_S4 just happens to sit at the transfer-safe
point of it by hand construction, not because the orbit doesn't exist for
it too).

    python tools/nonidentifiability_test.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from s4dpc.data import generate_microgrid_trajectory

DOCS = _REPO_ROOT / "docs"
EXPORT_DIR = DOCS / "nu_gap_export"
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
D_X, D_U = 6, 3
DT = 0.01
L_MAX = 100
DATA_SEED = 42
APRBS_LOW, APRBS_HIGH = -10.0, 10.0


def get_training_trajectory(case: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact match to s4dpc.identify.case_data (DATA_SEED=42, batch_size=1,
    l_max=100, aprbs (-10,10)) - reimplemented via generate_microgrid_trajectory
    directly rather than importing s4dpc.identify, to avoid that module's
    flax/s4-nnx import chain (not needed for pure data regeneration, and
    not reliably importable in this local CPU environment - DECISIONS.md)."""
    inputs, _targets = generate_microgrid_trajectory(
        batch_size=1, length=L_MAX, seed=DATA_SEED, system_case=case, dt=DT,
        aprbs_low=APRBS_LOW, aprbs_high=APRBS_HIGH)
    x = inputs[0, :, :D_X]  # (L, D_X)
    u = inputs[0, :, D_X:]  # (L, D_U)
    return x, u


def simulate_s_trajectory(Asx: np.ndarray, Ass: np.ndarray, Bs: np.ndarray,
                           x_traj: np.ndarray, u_traj: np.ndarray) -> np.ndarray:
    """s_0=0 (matching training convention everywhere in this project),
    s_{t+1} = Asx@x_t + Ass@s_t + Bs@u_t. Returns S (L, n_s), S[t] = s_t."""
    n_s = Ass.shape[0]
    L = x_traj.shape[0]
    S = np.zeros((L, n_s))
    s = np.zeros(n_s)
    for t in range(L):
        S[t] = s
        s = Asx @ x_traj[t] + Ass @ s + Bs @ u_traj[t]
    return S


def spectrum_summary(Z: np.ndarray, label: str) -> dict:
    sv = np.linalg.svd(Z, compute_uv=False)
    sigma_max = sv[0] if len(sv) else 0.0
    eff_rank_1e6 = int(np.sum(sv > 1e-6 * sigma_max))
    eff_rank_1e3 = int(np.sum(sv > 1e-3 * sigma_max))
    return {"label": label, "n_rows": Z.shape[0], "n_cols": Z.shape[1],
            "sigma_max": float(sigma_max), "sigma_min": float(sv[-1]) if len(sv) else 0.0,
            "eff_rank_1e-6": eff_rank_1e6, "eff_rank_1e-3": eff_rank_1e3,
            "n_singular_values": len(sv)}


def near_null_direction_energy(Z: np.ndarray, n_null: int = 10) -> list[dict]:
    """For the n_null smallest-singular-value right-singular vectors, report
    the x-block vs s-block energy split - genuine mixing (both nonzero, not
    one block ~0) is what "move dynamics from Axx into Axs" looks like."""
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    rows = []
    for i in range(1, n_null + 1):
        v = Vt[-i]
        frac_x = float(np.sum(v[:D_X] ** 2))
        rows.append({"rank_from_bottom": i, "sigma": float(sv[-i]), "frac_x_energy": frac_x,
                      "frac_s_energy": 1.0 - frac_x})
    return rows


def main() -> None:
    rng = np.random.RandomState(0)
    all_rows = []
    null_dir_rows = []

    for case in CASES:
        x_traj, u_traj = get_training_trajectory(case)
        x_only_summary = spectrum_summary(x_traj, f"case{case}_X_ONLY")
        print(f"case {case}: X alone (floor check) - eff_rank(1e-6)={x_only_summary['eff_rank_1e-6']} "
              f"of {D_X} (should be full rank if the true trajectory itself isn't degenerate)")
        all_rows.append(x_only_summary)

        random_Z = rng.standard_normal((L_MAX, D_X + 1024))  # matches d1030 shape for the null control
        random_summary = spectrum_summary(random_Z, f"case{case}_RANDOM_CONTROL")
        all_rows.append(random_summary)

        for variant in ["fullM3", "M0_S4"]:
            for seed in range(N_SEEDS):
                path = EXPORT_DIR / f"{variant}_{case}_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                A = data["A"]
                n_s = A.shape[0] - D_X
                Asx, Ass = A[D_X:, :D_X], A[D_X:, D_X:]
                Bs = data["B"][D_X:, :]

                S_traj = simulate_s_trajectory(Asx, Ass, Bs, x_traj, u_traj)
                Z = np.hstack([x_traj, S_traj])  # (L, D_X+n_s)
                summ = spectrum_summary(Z, f"{variant}_case{case}_seed{seed}")
                summ["variant"], summ["case"], summ["seed"] = variant, case, seed
                all_rows.append(summ)
                print(f"  {variant} case{case} seed{seed}: Z shape {Z.shape}  "
                      f"eff_rank(1e-6)={summ['eff_rank_1e-6']}  eff_rank(1e-3)={summ['eff_rank_1e-3']}  "
                      f"sigma_max={summ['sigma_max']:.3e}  sigma_min={summ['sigma_min']:.3e}")

                if variant == "fullM3" and seed == 0:
                    for r in near_null_direction_energy(Z):
                        r.update({"variant": variant, "case": case, "seed": seed})
                        null_dir_rows.append(r)

    for name, rows in [("nonidentifiability_spectrum.csv", all_rows),
                        ("nonidentifiability_null_directions.csv", null_dir_rows)]:
        out_path = DOCS / name
        header = sorted({k for r in rows for k in r.keys()})
        lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path} ({len(rows)} rows)")

    print("\n=== SUMMARY ===")
    fullm3_ranks = [r["eff_rank_1e-6"] for r in all_rows if r.get("variant") == "fullM3"]
    m0s4_ranks = [r["eff_rank_1e-6"] for r in all_rows if r.get("variant") == "M0_S4"]
    random_ranks = [r["eff_rank_1e-6"] for r in all_rows if "RANDOM_CONTROL" in r["label"]]
    x_only_ranks = [r["eff_rank_1e-6"] for r in all_rows if "X_ONLY" in r["label"]]
    print(f"fullM3 Z=[x,s] eff_rank(1e-6): median={np.median(fullm3_ranks):.1f}  "
          f"range=[{min(fullm3_ranks)},{max(fullm3_ranks)}]  (ceiling=100 samples, floor=6 if x alone explained it)")
    print(f"M0_S4  Z=[x,s] eff_rank(1e-6): median={np.median(m0s4_ranks):.1f}  "
          f"range=[{min(m0s4_ranks)},{max(m0s4_ranks)}]")
    print(f"RANDOM control (null hypothesis - no collinearity expected): "
          f"median={np.median(random_ranks):.1f}  range=[{min(random_ranks)},{max(random_ranks)}]")
    print(f"X alone (floor - true trajectory's own richness): "
          f"median={np.median(x_only_ranks):.1f}  range=[{min(x_only_ranks)},{max(x_only_ranks)}]")


if __name__ == "__main__":
    main()
