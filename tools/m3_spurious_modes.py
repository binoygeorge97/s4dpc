"""Task 4 (session brief, 2026-08-13): characterize M3's spurious internal
modes. M3 is exactly LTI (no norm/activation/glu) - CLAUDE.md's kink
finding already established that jacfwd of an affine map is a constant
independent of the evaluation point, so the AUGMENTED state-transition
operator (physical state x PLUS the S4 layer's own hidden state s,
stacked into one vector z) can be extracted EXACTLY, at any point (z=0,
u=0 is used, arbitrarily), via forward-mode autodiff - "use linear
tools, not numerical probing" per the brief.

s is complex (S4LayerEnsemble's hidden state, shape (d_model, N) per
layer); this script differentiates a REAL-valued wrapper (state packed
as concatenated real/imag parts) rather than calling jax.jacfwd directly
on a complex leaf - avoids the exact ravel_pytree/complex-promotion
hazard docs/DECISIONS.md already hit twice (Part B.4's silent
imaginary-part-discarding cast; tools/diagnose_variant_redundancy.py's
"jacfwd requires real-valued inputs" crash) by never handing autodiff a
mixed real/complex flat vector in the first place.

Abar: (1030, 1030) real = [[d_x_next/dx, d_x_next/ds]; [d_s_next/dx,
d_s_next/ds]] (6 physical + 2*16*32=1024 real-flattened S4 state dims).
Reports, per (case, seed), all 7 cases x 10 seeds, M3 only (M6 is not
exactly LTI, so a single global Jacobian doesn't characterize it the
same way - out of scope here per the brief):

  - eigenvalues of Abar vs the case's 6 true eigenvalues of A_d: how
    many of Abar's 1030 modes sit near the unit circle (slow modes) vs
    how many of A_d's 6 do.
  - observability proxy: ||Abar[:6, 6:]||_2 (SVD 2-norm of the block
    mapping S4 state -> next PHYSICAL state in one step) - how strongly
    the internal state can move x_next, without needing an eigenvector
    solve (banned on defective/near-defective matrices,
    docs/DECISIONS.md's 2026-08-07 entries).
  - ||Abar^k||_2 vs ||A_d^k||_2 for k up to 200 (SVD-based, same
    convention as tests/test_systems.py) - same I/O map does not imply
    same transient conditioning; this is realization-dependent in a way
    markov_parameters deliberately is not.

Then correlates (Pearson AND Spearman, per-seed AND per-case-median,
matching this project's established practice of reporting both rather
than leaning on whichever looks cleaner) these quantities against the
per-(case,seed) M3 DPC cost ratio already on record in
docs/controller_surrogates_summary.csv - joined by (case, seed), which
lines up because run_identify's key derivation
(fold_in(PRNGKey(seed_base+seed), case), seed_base=0 default) is
deterministic and this script uses the same config (40k epochs,
d_model=16, N=32, n_layers=1) as the sweep that produced those
checkpoints.

    python tools/m3_spurious_modes.py
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
from flax import nnx

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify
from s4dpc.model import StackedModel
from s4dpc.systems import get_discrete_matrices

CASES = list(range(1, 8))
N_SEEDS = 10
EPOCHS = 40000
D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
DT = 0.01
VARIANT = "M3"
D_X, D_U = 6, 3
S_DIM = D_MODEL * STATE_SIZE  # complex dims per layer
Z_DIM = D_X + 2 * S_DIM  # augmented real dim: x (6) + s as (re, im) flattened

K_GRID = [1, 5, 10, 25, 50, 100, 150, 200]
NEAR_UNIT_THRESHOLD = 0.99

DOCS_DIR = _REPO_ROOT / "docs"
SURROGATES_CSV = DOCS_DIR / "controller_surrogates_summary.csv"


def _pack(x: jax.Array, s: jax.Array) -> jax.Array:
    return jnp.concatenate([x, s.real.ravel(), s.imag.ravel()])


def _unpack(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = z[:D_X]
    s_re = z[D_X:D_X + S_DIM].reshape(D_MODEL, STATE_SIZE)
    s_im = z[D_X + S_DIM:].reshape(D_MODEL, STATE_SIZE)
    return x, s_re + 1j * s_im


def augmented_operator(graphdef, params) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Abar: (Z_DIM, Z_DIM), Bbar: (Z_DIM, d_u)), both real,
    evaluated at z=0, u=0 - valid EVERYWHERE since M3 is exactly affine."""

    def f(z: jax.Array, u: jax.Array) -> jax.Array:
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        x_next, (s_next,) = m(jnp.concatenate([x, u]), [s])
        return _pack(x_next, s_next)

    z0 = jnp.zeros((Z_DIM,), dtype=jnp.float64)
    u0 = jnp.zeros((D_U,), dtype=jnp.float64)
    Abar = jax.jacfwd(f, argnums=0)(z0, u0)
    Bbar = jax.jacfwd(f, argnums=1)(z0, u0)
    return np.asarray(Abar), np.asarray(Bbar)


def spectral_norm_powers(Abar_j: jax.Array, k_grid: list[int]) -> list[float]:
    """||Abar^k||_2 for each k in k_grid, via repeated squaring-free
    iterative multiply (k_grid is small and sparse, not every k)."""
    norms = []
    M = jnp.eye(Abar_j.shape[0], dtype=Abar_j.dtype)
    k_done = 0
    for k in k_grid:
        while k_done < k:
            M = M @ Abar_j
            k_done += 1
        norms.append(float(jnp.linalg.norm(M, ord=2)))
    return norms


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CASES={CASES}  N_SEEDS={N_SEEDS}  Z_DIM={Z_DIM}  K_GRID={K_GRID}")

    print(f"\n{'=' * 20} identifying {VARIANT}, all 7 cases x {N_SEEDS} seeds, {EPOCHS} epochs {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant=VARIANT, cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS[VARIANT])
    graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )

    # true A_d eigenvalues/powers per case, computed once
    true_by_case = {}
    for case in CASES:
        A_d, _ = get_discrete_matrices(DT, case)
        eig_true = np.linalg.eigvals(A_d)
        norms_true = []
        M = np.eye(A_d.shape[0])
        k_done = 0
        for k in K_GRID:
            while k_done < k:
                M = M @ A_d
                k_done += 1
            norms_true.append(float(np.linalg.norm(M, ord=2)))
        true_by_case[case] = {"eig": eig_true, "norms": norms_true,
                               "n_near_unit": int(np.sum(np.abs(eig_true) > NEAR_UNIT_THRESHOLD))}

    rows = []
    t_start_all = time.time()
    for r in id_rows:
        case, seed, teacher_mse, param_state = r["case"], r["seed"], r["teacher_mse"], r["param_state"]
        t0 = time.time()
        Abar, Bbar = augmented_operator(graphdef, param_state)
        Abar_j = jnp.asarray(Abar)

        eig_abar = np.linalg.eigvals(Abar)
        rho_abar = float(np.max(np.abs(eig_abar)))
        n_near_unit = int(np.sum(np.abs(eig_abar) > NEAR_UNIT_THRESHOLD))

        obs_norm = float(np.linalg.norm(Abar[:D_X, D_X:], ord=2))  # s -> x_next block

        norms_abar = spectral_norm_powers(Abar_j, K_GRID)
        norms_true = true_by_case[case]["norms"]
        growth_ratio_k200 = norms_abar[-1] / norms_true[-1] if norms_true[-1] > 0 else float("nan")

        elapsed = time.time() - t0
        print(f"  [{VARIANT}/case{case}/seed{seed}] wall={elapsed:.1f}s  teacher_mse={teacher_mse:.3e}  "
              f"rho(Abar)={rho_abar:.4f}  n_near_unit={n_near_unit}/{Z_DIM} (true: "
              f"{true_by_case[case]['n_near_unit']}/6)  obs_norm={obs_norm:.3e}  "
              f"||Abar^200||={norms_abar[-1]:.3e}  ||A_d^200||={norms_true[-1]:.3e}  "
              f"ratio={growth_ratio_k200:.3e}")

        rows.append({
            "case": case, "seed": seed, "teacher_mse": teacher_mse,
            "rho_abar": rho_abar, "n_near_unit_abar": n_near_unit, "n_near_unit_true": true_by_case[case]["n_near_unit"],
            "obs_norm": obs_norm, "growth_ratio_k200": growth_ratio_k200,
            **{f"norm_abar_k{k}": v for k, v in zip(K_GRID, norms_abar)},
            **{f"norm_true_k{k}": v for k, v in zip(K_GRID, norms_true)},
        })
    print(f"\ntotal wall time for {len(id_rows)} members: {time.time() - t_start_all:.1f}s")

    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "m3_spurious_modes.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'm3_spurious_modes.csv'}")

    # ---- correlate against the existing M3 DPC ratio table ----
    if not SURROGATES_CSV.exists():
        print(f"\n{SURROGATES_CSV} not found - skipping DPC-ratio correlation.")
        return

    with open(SURROGATES_CSV, newline="") as f:
        dpc_rows = [r for r in csv.DictReader(f) if r["oracle"] == "M3"]
    dpc_by_case_seed = {(int(r["case"]), int(r["seed"])): float(r["cost_ratio_to_oracle"]) for r in dpc_rows}

    joined = [(r, dpc_by_case_seed[(r["case"], r["seed"])]) for r in rows if (r["case"], r["seed"]) in dpc_by_case_seed]
    print(f"\n=== DPC-ratio correlation: {len(joined)} (case,seed) pairs matched against {SURROGATES_CSV.name} ===")
    if len(joined) < 3:
        print("  too few matched pairs to correlate - SEED_SELECTION's chosen seeds may not overlap with this run's.")
        return

    from scipy.stats import pearsonr, spearmanr

    predictors = ["rho_abar", "n_near_unit_abar", "obs_norm", "growth_ratio_k200"]
    ratios = np.array([j[1] for j in joined])
    print(f"{'predictor':20s} {'pearson r':>12s} {'p':>10s} {'spearman rho':>14s} {'p':>10s}")
    for pred in predictors:
        vals = np.array([j[0][pred] for j in joined])
        finite = np.isfinite(vals) & np.isfinite(ratios)
        if finite.sum() < 3:
            continue
        pr, pp = pearsonr(vals[finite], ratios[finite])
        sr, sp = spearmanr(vals[finite], ratios[finite])
        print(f"{pred:20s} {pr:12.4f} {pp:10.4f} {sr:14.4f} {sp:10.4f}")

    # per-case median version, matching this project's established habit
    # of checking both (raw correlation can be leverage-point-dominated)
    print("\n  per-case-median version (n=6 cases, weaker but robust to seed noise):")
    case_medians = {}
    for case in set(r["case"] for r, _ in joined):
        case_pairs = [(r, ratio) for r, ratio in joined if r["case"] == case]
        case_medians[case] = {
            "ratio": float(np.median([p[1] for p in case_pairs])),
            **{pred: float(np.median([p[0][pred] for p in case_pairs])) for pred in predictors},
        }
    case_list = sorted(case_medians.keys())
    ratios_median = np.array([case_medians[c]["ratio"] for c in case_list])
    for pred in predictors:
        vals = np.array([case_medians[c][pred] for c in case_list])
        finite = np.isfinite(vals) & np.isfinite(ratios_median)
        if finite.sum() < 3:
            continue
        pr, pp = pearsonr(vals[finite], ratios_median[finite])
        sr, sp = spearmanr(vals[finite], ratios_median[finite])
        print(f"{pred:20s} {pr:12.4f} {pp:10.4f} {sr:14.4f} {sp:10.4f}")


if __name__ == "__main__":
    main()
