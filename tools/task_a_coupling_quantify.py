"""TASK A (user, 2026-08-25, verbatim spec): quantify the coupling-
induced instability gap between A_ss (0 unstable eigenvalues, all 60
checkpoints) and the full Abar (median 8.5 at B=1, 40 at B=320).

For M3 at B=1 and B=320, with M1 and M0_S4 as controls:
  (i)   n_unstable(A_xx) - the physical-block Jacobian alone
  (ii)  eigenvalue condition numbers kappa(lambda) = 1/|y^H x| (left/right
        eigenvectors, normalized) for A_ss modes within 1e-2 of the unit
        circle - median, max, distribution
  (iii) departure from normality of A_ss:
        ||A_ss^H A_ss - A_ss A_ss^H||_F / ||A_ss||_F^2
  (iv)  ||A_xs||_F and ||A_sx||_F
  (v)   kappa_max * ||A_xs||_F vs the observed n_unstable(Abar)

Also: learned per-channel step (Delta) statistics from the raw S4
params, and the closed-form discrete-time bound on |lambda| implied by
Lambda_re<=-1e-4 combined with Delta's own clip range [0.001, 1.0].

Pure linear algebra on saved checkpoints/params - no training, no GPU.

    python tools/task_a_coupling_quantify.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import flax.serialization as serialization

D_X = 6
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5


def load_pure_dict(path: pathlib.Path) -> dict:
    return serialization.msgpack_restore(path.read_bytes())


def get_step_and_lambda_re(pure_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    layer = pure_dict["layers"]["0"]["seq"]
    log_step = np.asarray(layer["log_step"])  # (d_model, 1)
    lambda_re = np.asarray(layer["Lambda_re"])  # (d_model, N)
    step = np.clip(np.exp(log_step), 0.001, 1.0)
    lambda_re_clipped = np.clip(lambda_re, None, -1e-4)
    return step, lambda_re_clipped


def eig_condition_numbers(A: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """kappa(lambda_i) = 1/|y_i^H x_i| for right eigvec x_i, left eigvec y_i,
    both normalized to unit 2-norm - the standard Bauer-Fike eigenvalue
    condition number. `indices` selects which of A's eigenpairs to report
    (the near-unit-circle ones - computing all 1024 is wasteful when only
    a few hundred are of interest, but eig() itself must still factor the
    whole matrix once)."""
    w, VR = np.linalg.eig(A)
    _, VL = np.linalg.eig(A.conj().T)  # left eigenvectors of A = right eigenvectors of A^H, per eigenvalue conj(w)
    # match VL's eigenvalues (conj(w) up to reordering) back to VR's w by nearest value
    wl = np.linalg.eigvals(A.conj().T)
    kappas = np.full(len(indices), np.nan)
    for out_i, idx in enumerate(indices):
        lam = w[idx]
        x = VR[:, idx]
        x = x / np.linalg.norm(x)
        # find the left eigenvector whose eigenvalue (of A^H) is conj(lam)
        j = int(np.argmin(np.abs(wl - np.conj(lam))))
        y = VL[:, j]
        y = y / np.linalg.norm(y)
        denom = np.abs(np.vdot(y, x))
        kappas[out_i] = 1.0 / denom if denom > 1e-300 else np.inf
    return kappas, w


def analyze_checkpoint(A: np.ndarray, variant: str) -> dict:
    has_latent = A.shape[0] > D_X
    Axx = A[:D_X, :D_X]
    eig_axx = np.linalg.eigvals(Axx)
    n_unstable_axx = int(np.sum(np.abs(eig_axx) > 1.0))

    result = {"n_unstable_axx": n_unstable_axx, "has_latent": has_latent}
    if not has_latent:
        return result

    Axs = A[:D_X, D_X:]
    Asx = A[D_X:, :D_X]
    Ass = A[D_X:, D_X:]

    axs_norm = float(np.linalg.norm(Axs, "fro"))
    asx_norm = float(np.linalg.norm(Asx, "fro"))
    ass_norm = float(np.linalg.norm(Ass, "fro"))
    non_normality = float(np.linalg.norm(Ass.conj().T @ Ass - Ass @ Ass.conj().T, "fro") / (ass_norm ** 2))

    eig_ass = np.linalg.eigvals(Ass)
    abs_ass = np.abs(eig_ass)
    near_circle_idx = np.where((abs_ass > 1 - 1e-2) & (abs_ass <= 1.0))[0]

    kappas = np.array([])
    if len(near_circle_idx) > 0:
        kappas, _ = eig_condition_numbers(Ass, near_circle_idx)

    result.update({
        "axs_norm": axs_norm, "asx_norm": asx_norm, "ass_norm": ass_norm,
        "non_normality_ass": non_normality,
        "n_near_circle_ass": int(len(near_circle_idx)),
        "kappa_median": float(np.median(kappas)) if len(kappas) else None,
        "kappa_max": float(np.max(kappas)) if len(kappas) else None,
        "kappas_all": kappas,
    })
    return result


def process_population(npz_dir: pathlib.Path, npz_prefix: str, ckpt_dir: pathlib.Path | None,
                        msgpack_prefix: str, label: str) -> list[dict]:
    rows = []
    for case in CASES:
        for seed in range(N_SEEDS):
            npz_path = npz_dir / f"{npz_prefix}_{case}_{seed}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            A = data["A"]
            res = analyze_checkpoint(A, label)
            res.update({"label": label, "case": case, "seed": seed})

            if ckpt_dir is not None:
                msgpack_path = ckpt_dir / f"{msgpack_prefix}_case{case}_seed{seed}.msgpack"
                if msgpack_path.exists():
                    pure_dict = load_pure_dict(msgpack_path)
                    step, lambda_re = get_step_and_lambda_re(pure_dict)
                    res["step_median"] = float(np.median(step))
                    res["step_min"] = float(np.min(step))
                    res["step_max"] = float(np.max(step))
                    res["lambda_re_median"] = float(np.median(lambda_re))
                    res["lambda_re_max"] = float(np.max(lambda_re))  # closest to 0 (least negative)

            rows.append(res)
            print(f"[{label}/case{case}/seed{seed}] n_unstable_axx={res['n_unstable_axx']}  "
                  + (f"axs_norm={res.get('axs_norm', 'n/a'):.4f}  asx_norm={res.get('asx_norm', 'n/a'):.4f}  "
                     f"non_normality_ass={res.get('non_normality_ass', 'n/a'):.4e}  "
                     f"n_near_circle={res.get('n_near_circle_ass', 'n/a')}  "
                     f"kappa_median={res.get('kappa_median')}  kappa_max={res.get('kappa_max')}"
                     if res["has_latent"] else "(no latent state)"))
    return rows


def main() -> None:
    EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"
    CKPT_DIR = EXPORT_DIR / "ckpt"
    B320_DIR = _REPO_ROOT / "docs" / "b320"
    B320_CKPT_DIR = B320_DIR / "ckpt"

    all_rows = []
    all_rows += process_population(EXPORT_DIR, "fullM3", CKPT_DIR, "M3", "M3_B1")
    all_rows += process_population(EXPORT_DIR, "M1", None, "M1", "M1")
    all_rows += process_population(EXPORT_DIR, "M0_S4", CKPT_DIR, "M0_S4", "M0_S4")
    all_rows += process_population(B320_DIR, "M3_b320", B320_CKPT_DIR, "M3_b320", "M3_B320")

    # closed-form bound: step in [0.001, 1.0], Lambda_re <= -1e-4 (so |Re| >= 1e-4)
    # first-order Tustin: |lambda_d| ~= 1 + step*Re for small step*Re (Re<0)
    # so 1-|lambda_d| ~= step*|Re| >= 0.001 * 1e-4 = 1e-7 at the extreme allowed values
    step_min, re_min_mag = 0.001, 1e-4
    bound = step_min * re_min_mag
    print(f"\n=== closed-form bound ===")
    print(f"step (Delta) allowed range: [0.001, 1.0]  (jnp.clip(exp(log_step), 0.001, 1.0))")
    print(f"Lambda_re allowed range: (-inf, -1e-4]  (jnp.clip(Lambda_re, None, -1e-4))")
    print(f"minimum achievable 1-|lambda_discrete| (first-order Tustin, step*|Re|): "
          f"{step_min} * {re_min_mag} = {bound:.2e}")
    print(f"=> the parameterization DOES permit discrete poles within ~{bound:.0e} of the unit circle - "
          f"tighter than the 1e-6 threshold asked about.")

    import csv
    out_path = _REPO_ROOT / "docs" / "task_a_coupling_quantify.csv"
    flat_rows = []
    for r in all_rows:
        r2 = {k: v for k, v in r.items() if k != "kappas_all"}
        flat_rows.append(r2)
    header = sorted({k for r in flat_rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(flat_rows)
    print(f"\nwrote {out_path}")

    import statistics as st
    print("\n=== summary by population ===")
    for label in ["M3_B1", "M1", "M0_S4", "M3_B320"]:
        these = [r for r in all_rows if r["label"] == label]
        if not these:
            continue
        nua = [r["n_unstable_axx"] for r in these]
        print(f"\n{label} (n={len(these)}): n_unstable_axx median={st.median(nua)} "
              f"min={min(nua)} max={max(nua)}")
        if these[0]["has_latent"]:
            axs = [r["axs_norm"] for r in these]
            asx = [r["asx_norm"] for r in these]
            nn = [r["non_normality_ass"] for r in these]
            kmed = [r["kappa_median"] for r in these if r["kappa_median"] is not None]
            kmax_list = [r["kappa_max"] for r in these if r["kappa_max"] is not None]
            print(f"  axs_norm: median={st.median(axs):.4f} max={max(axs):.4f}")
            print(f"  asx_norm: median={st.median(asx):.4f} max={max(asx):.4f}")
            print(f"  non_normality_ass: median={st.median(nn):.4e} max={max(nn):.4e}")
            if kmed:
                print(f"  kappa_median (of per-checkpoint medians): {st.median(kmed):.4e}")
            if kmax_list:
                print(f"  kappa_max (of per-checkpoint maxes): {max(kmax_list):.4e}  median-of-maxes={st.median(kmax_list):.4e}")
            if "step_median" in these[0]:
                sm = [r["step_median"] for r in these]
                lrm = [r["lambda_re_max"] for r in these]
                print(f"  step(Delta) median-of-medians={st.median(sm):.4e}")
                print(f"  Lambda_re max (closest to 0) median={st.median(lrm):.4e}")


if __name__ == "__main__":
    main()
