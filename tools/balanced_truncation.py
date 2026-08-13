"""The deciding experiment (user brief, 2026-08-13, item 2): does
truncating M3's augmented realization down to 6 states - discarding
everything Task 4 identified as spurious - fix DPC, while preserving
M3's own near-exact Markov-parameter fidelity?

M3 is exactly LTI, so its augmented state-transition operator (Abar,
Bbar - physical state + S4 hidden state, 1030 dims total) is available
exactly via tools/m3_spurious_modes.py's augmented_operator. Standard
GRAMIAN-based balanced truncation (solve discrete Lyapunov equations for
the controllability/observability Gramians) requires Schur stability;
Task 4 found rho(Abar) sits at 1.02-1.03 for 6/7 cases (higher for case
6) - the augmented system is marginally UNSTABLE, so those Gramians do
not converge. This uses Hankel-SVD / Kung's method (the Eigensystem
Realization Algorithm, ERA) instead: it operates directly on a finite
window of Markov parameters and needs no stability assumption at all -
the standard, numerically robust way to do exactly this when the
system isn't known to be stable.

Pipeline, in order, each step gating the next:
  1. Self-check: run the SAME Hankel-SVD/ERA pipeline on the TRUE
     (A_d, B_d) (already exactly minimal, 6-state) - truncating an
     already-minimal system to its own order should recover it (up to
     a similarity transform) near machine precision. If this fails,
     the pipeline itself is broken and nothing below is trustworthy.
  2. Per (case, seed) M3 checkpoint: extract Markov parameters from the
     augmented operator, build the block Hankel matrix, report the
     FULL singular value spectrum (not just top-6 pass/fail), truncate
     to r=6, reconstruct (Ar, Br, Cr) via ERA, transform to output-
     normal form (state = physical output) when Cr is square and
     well-conditioned, verify the reconstruction still matches the
     ORIGINAL Markov parameters to ~1e-6 (M3's own fidelity) before
     trusting it.
  3. Train the STANDARD BoundedGRUController through the truncated
     system via rollout_linear (exactly like M0/M1 - it's now a plain
     linear system, no S4 machinery involved at all), all 6 control
     cases, >=5 seeds, standard curriculum. Evaluate on the true plant,
     exactly as every other oracle/surrogate in this project.

Also (cheap, piggybacking on the same augmented_operator machinery):
  - M0_S4's obs_norm: verify Abar[:6,6:] is EXACTLY 0 (not merely
    small) - the S4 state is present but permanently unobservable by
    construction, not absent.
  - Lambda_re-clip check: does a FRESH, UNTRAINED M3 already show
    Task 4's ~300-near-unit-mode signature (an artifact of S4's own
    Lambda_re<=-1e-4 clipping, present regardless of training), or is
    it something training introduces? Compare fresh vs Task 4's
    already-recorded trained numbers.

    python tools/balanced_truncation.py
"""
from __future__ import annotations

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

sys.path.insert(0, str(_REPO_ROOT / "tools"))
import controller_oracles as co  # noqa: E402
from m3_spurious_modes import (  # noqa: E402
    D_MODEL, STATE_SIZE, N_LAYERS, L_MAX, D_X, D_U, Z_DIM, S_DIM,
    augmented_operator,
)

from s4dpc.blocks import BlockConfig, VARIANTS  # noqa: E402
from s4dpc.control import BoundedGRUController, make_controller_optimizer, rollout_linear  # noqa: E402
from s4dpc.identify import D_INPUT, D_OUTPUT, run_identify  # noqa: E402
from s4dpc.model import StackedModel  # noqa: E402
from s4dpc.systems import get_discrete_matrices  # noqa: E402

CONTROL_CASES = [c for c in co.CASES if c != 6]
N_SEEDS = 5
EPOCHS_ID = 40000  # matches the established identification budget
DT = 0.01
R_TRUNC = 6  # target truncated order = the true plant's own order
N_HANKEL = 20  # Hankel block dimension (N1=N2=20) - needs G_1..G_{2N-1}=G_39
DOCS_DIR = _REPO_ROOT / "docs"

COUT = np.concatenate([np.eye(D_X), np.zeros((D_X, Z_DIM - D_X))], axis=1)  # (6, 1030)


def markov_from_augmented(Abar: np.ndarray, Bbar: np.ndarray, Cout: np.ndarray, H: int) -> list[np.ndarray]:
    """G_h = Cout @ Abar^(h-1) @ Bbar, h=1..H."""
    Gs = []
    M = np.eye(Abar.shape[0])
    for h in range(1, H + 1):
        Gs.append(Cout @ M @ Bbar)
        M = Abar @ M
    return Gs


def era(markov_params: list[np.ndarray], r: int, n_hankel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eigensystem Realization Algorithm (Kung's method / Hankel-SVD).
    markov_params: [G_1, ..., G_{2*n_hankel}], each (p, m).
    Returns (Ar, Br, Cr, hankel_singular_values) - a MINIMAL r-state
    realization matching the given Markov parameters, no stability
    assumption required (operates on a finite Markov-parameter window,
    not on Gramians/Lyapunov equations)."""
    p, m = markov_params[0].shape
    assert len(markov_params) >= 2 * n_hankel, "need G_1..G_{2*n_hankel}"

    H0 = np.block([[markov_params[i + j] for j in range(n_hankel)] for i in range(n_hankel)])
    H1 = np.block([[markov_params[i + j + 1] for j in range(n_hankel)] for i in range(n_hankel)])

    U, S, Vt = np.linalg.svd(H0, full_matrices=False)
    hankel_singular_values = S.copy()

    Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
    Sr_sqrt = np.sqrt(Sr)
    Sr_inv_sqrt = 1.0 / Sr_sqrt

    Ar = np.diag(Sr_inv_sqrt) @ Ur.T @ H1 @ Vtr.T @ np.diag(Sr_inv_sqrt)
    Br = np.diag(Sr_sqrt) @ Vtr[:, :m]
    Cr = Ur[:p, :] @ np.diag(Sr_sqrt)

    return Ar, Br, Cr, hankel_singular_values


def to_output_normal_form(Ar: np.ndarray, Br: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """If Cr is square and well-conditioned, transform so the reduced
    state IS the physical output (matching M0/M1's own (A_hat,B_hat)
    convention) - state z' = Cr @ z, A' = Cr Ar Cr^-1, B' = Cr Br.
    Returns (A_final, B_final, cond(Cr)) - caller checks cond before
    trusting the transform."""
    cond = np.linalg.cond(Cr)
    Cr_inv = np.linalg.inv(Cr)
    A_final = Cr @ Ar @ Cr_inv
    B_final = Cr @ Br
    return A_final, B_final, cond


def verify_markov_match(A: np.ndarray, B: np.ndarray, C: np.ndarray, true_markov: list[np.ndarray]) -> float:
    """Max abs error between C@A^(h-1)@B and true_markov[h-1], h=1..len(true_markov)."""
    max_err = 0.0
    M = np.eye(A.shape[0])
    for h in range(len(true_markov)):
        recon = C @ M @ B
        max_err = max(max_err, float(np.max(np.abs(recon - true_markov[h]))))
        M = A @ M
    return max_err


def self_check_true_system() -> None:
    print("\n" + "=" * 20 + " SELF-CHECK: ERA on the TRUE (A_d, B_d), all 6 control cases " + "=" * 20)
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        A_d, B_d = np.asarray(A_d), np.asarray(B_d)
        C_true = np.eye(D_X)
        H = 2 * N_HANKEL
        true_markov = [np.linalg.matrix_power(A_d, h - 1) @ B_d for h in range(1, H + 1)]

        Ar, Br, Cr, hsv = era(true_markov, R_TRUNC, N_HANKEL)
        cond = np.linalg.cond(Cr)
        if cond > 1e8:
            print(f"  case{case}: Cr ill-conditioned (cond={cond:.3e}) - self-check FAILED, cannot proceed")
            continue
        A_final, B_final, _ = to_output_normal_form(Ar, Br, Cr)
        err = verify_markov_match(A_final, B_final, np.eye(D_X), true_markov)
        print(f"  case{case}: reconstruction max Markov error = {err:.3e}  (expect ~1e-9 or better)  "
              f"top-8 HSV={hsv[:8].round(3)}  HSV[6:10]={hsv[6:10].round(3)}")


def build_truncated_m3(case: int, seed: int, param_state, graphdef) -> dict:
    Abar, Bbar = augmented_operator(graphdef, param_state)
    H = 2 * N_HANKEL
    m3_markov = markov_from_augmented(Abar, Bbar, COUT, H)

    Ar, Br, Cr, hsv = era(m3_markov, R_TRUNC, N_HANKEL)
    cond = np.linalg.cond(Cr)
    result = {"case": case, "seed": seed, "hsv": hsv, "cond_Cr": cond}
    if cond > 1e8:
        result["ok"] = False
        return result

    A_final, B_final, _ = to_output_normal_form(Ar, Br, Cr)
    err_vs_m3 = verify_markov_match(A_final, B_final, np.eye(D_X), m3_markov)

    A_d, B_d = get_discrete_matrices(DT, case)
    true_markov = [np.linalg.matrix_power(np.asarray(A_d), h - 1) @ np.asarray(B_d) for h in range(1, H + 1)]
    err_vs_true = verify_markov_match(A_final, B_final, np.eye(D_X), true_markov)

    result.update({"ok": True, "A": A_final, "B": B_final,
                    "err_vs_m3_markov": err_vs_m3, "err_vs_true_markov": err_vs_true})
    return result


def check_m0_s4_observability() -> None:
    print("\n" + "=" * 20 + " M0_S4 observability check (Abar[:6,6:] should be EXACTLY 0) " + "=" * 20)
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from controller_m0_s4 import _build_m0_s4  # noqa: E402

    for case in [1, 3]:  # a couple of cases is enough - this is a construction property, not per-case
        model = _build_m0_s4(case, seed=0)
        graphdef, params = nnx.split(model, nnx.Param)
        Abar, _ = augmented_operator(graphdef, params)
        obs_block = Abar[:D_X, D_X:]
        print(f"  case{case}: max|Abar[:6,6:]| = {np.max(np.abs(obs_block)):.3e}  (expect exactly 0, "
              f"up to float64 roundoff ~1e-15)")


def check_lambda_re_clip_artifact() -> None:
    print("\n" + "=" * 20 + " Lambda_re-clip check: fresh (untrained) M3 vs Task 4's trained numbers " + "=" * 20)
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )
    for case in [1, 3]:
        for seed in range(3):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
            model = StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                                  decode=True, rngs=nnx.Rngs(params=key))

            def _cast(x):
                return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)

            state = nnx.state(model, nnx.Param)
            state = jax.tree_util.tree_map(_cast, state)
            nnx.update(model, state)
            params = nnx.state(model, nnx.Param)

            Abar, _ = augmented_operator(graphdef, params)
            eig = np.linalg.eigvals(Abar)
            rho = float(np.max(np.abs(eig)))
            n_near_unit = int(np.sum(np.abs(eig) > 0.99))
            print(f"  case{case}/seed{seed} FRESH (untrained): rho(Abar)={rho:.4f}  "
                  f"n_near_unit={n_near_unit}/{Z_DIM}")
    print("  (compare against Task 4's TRAINED numbers, docs/DECISIONS.md: "
          "rho~1.02-1.03, n_near_unit median 264-369/1030 per case)")


def _train_ensemble_truncated(members: list[tuple[int, int]], AB_by_case_seed: dict, max_action: float) -> nnx.State:
    A_list, B_list, x0_list, key_list = [], [], [], []
    for case, seed in members:
        A, B = AB_by_case_seed[(case, seed)]
        A_list.append(jnp.asarray(A, dtype=jnp.float64))
        B_list.append(jnp.asarray(B, dtype=jnp.float64))
        init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
        x0_key = jax.random.fold_in(init_key, 999)
        x0 = jax.random.uniform(
            x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE, dtype=jnp.float64
        )
        x0_list.append(x0)
        key_list.append(init_key)
    A_batch, B_batch = jnp.stack(A_list), jnp.stack(B_list)
    x0_batch, keys = jnp.stack(x0_list), jnp.stack(key_list)

    @nnx.vmap(in_axes=0, out_axes=0)
    def init_ensemble(key):
        return BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

    ensemble = init_ensemble(keys)
    optimizer = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)

    for pi, phase in enumerate(co.CURRICULUM):
        N = phase["N"]

        @nnx.jit
        def train_step(ens, opt, N=N):
            def loss_fn(e):
                cg, cp = nnx.split(e, nnx.Param)

                def single_member(p, A, B, x0):
                    c = nnx.merge(cg, p)
                    return rollout_linear(c, x0, A, B, co.Q_X, co.R_U, co.Q_F, N)

                losses = jax.vmap(single_member)(cp, A_batch, B_batch, x0_batch)
                return jnp.mean(losses), losses

            (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
            opt.update(ens, grads)
            return loss, per_member

        for epoch in range(phase["epochs"]):
            loss, per_member = train_step(ensemble, optimizer)
        print(f"    phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")

    return nnx.state(ensemble, nnx.Param)


def main() -> None:
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}  R_TRUNC={R_TRUNC}  N_HANKEL={N_HANKEL}")

    self_check_true_system()
    check_m0_s4_observability()
    check_lambda_re_clip_artifact()

    print(f"\n{'=' * 20} identifying M3, cases {CONTROL_CASES} x {N_SEEDS} seeds, {EPOCHS_ID} epochs {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant="M3", cases=CONTROL_CASES, n_seeds=N_SEEDS, epochs=EPOCHS_ID,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")
    diverged = {(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0}
    print(f"  diverged (teacher_mse > 10): {sorted(diverged)}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )

    print(f"\n{'=' * 20} balanced truncation (Hankel-SVD/ERA), per (case, seed) {'=' * 20}")
    trunc_rows = []
    AB_by_case_seed = {}
    for r in id_rows:
        case, seed = r["case"], r["seed"]
        if (case, seed) in diverged:
            print(f"  case{case}/seed{seed}: SKIPPED (diverged identification)")
            continue
        res = build_truncated_m3(case, seed, r["param_state"], graphdef)
        if not res.get("ok"):
            print(f"  case{case}/seed{seed}: Cr ill-conditioned (cond={res['cond_Cr']:.3e}) - SKIPPED")
            trunc_rows.append({"case": case, "seed": seed, "ok": False, "cond_Cr": res["cond_Cr"]})
            continue
        AB_by_case_seed[(case, seed)] = (res["A"], res["B"])
        hsv = res["hsv"]
        print(f"  case{case}/seed{seed}: err_vs_m3_markov={res['err_vs_m3_markov']:.3e}  "
              f"err_vs_true_markov={res['err_vs_true_markov']:.3e}  cond(Cr)={res['cond_Cr']:.3e}  "
              f"top-8 HSV={hsv[:8].round(3)}  HSV[6:10]={hsv[6:10].round(4)}")
        trunc_rows.append({
            "case": case, "seed": seed, "ok": True, "cond_Cr": res["cond_Cr"],
            "err_vs_m3_markov": res["err_vs_m3_markov"], "err_vs_true_markov": res["err_vs_true_markov"],
            **{f"hsv_{i}": float(hsv[i]) for i in range(min(15, len(hsv)))},
        })

    header = sorted({k for r in trunc_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in trunc_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "balanced_truncation_diagnostics.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'balanced_truncation_diagnostics.csv'}")

    print(f"\n{'=' * 20} DPC control through truncated M3, {len(AB_by_case_seed)} members {'=' * 20}")
    oracle_costs, eval_keys, true_AB = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(co.DT, case)
        true_AB[case] = (A_d, B_d)
        eval_key = jax.random.fold_in(jax.random.PRNGKey(123), case)
        eval_keys[case] = eval_key
        Q = co.Q_X * np.eye(A_d.shape[0])
        R = co.R_U * np.eye(B_d.shape[1])
        K = co.solve_dlqr(A_d, B_d, Q, R)
        x0_eval_np = np.asarray(
            jax.random.uniform(eval_key, (co.EVAL_BATCH, A_d.shape[0]), minval=-co.EVAL_X0_RANGE, maxval=co.EVAL_X0_RANGE)
        )
        x_hist_lqr, u_hist_lqr = co.rollout_lqr_true(A_d, B_d, K, x0_eval_np, co.EVAL_HORIZON)
        oracle_costs[case] = co.true_quadratic_cost(x_hist_lqr, u_hist_lqr, co.Q_X, co.R_U, co.Q_F)

    by_bound: dict[float, list[tuple[int, int]]] = {}
    for (case, seed) in AB_by_case_seed:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append((case, seed))

    control_rows = []
    for max_action, members in by_bound.items():
        print(f"\n  --- max_action={max_action}, {len(members)} members: {members} ---")
        t0 = time.time()
        ensemble_state = _train_ensemble_truncated(members, AB_by_case_seed, max_action)
        print(f"  wall time: {time.time() - t0:.1f}s")

        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ensemble_state)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            A_d, B_d = true_AB[case]
            result = co._evaluate(controller, A_d, B_d, eval_keys[case])
            ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
            print(f"    [truncM3/case{case}/seed{seed}] ratio_to_oracle={ratio:.4e}  finite={result['finite']}")
            control_rows.append({"case": case, "seed": seed, "cost": result["cost"],
                                  "cost_ratio_to_oracle": ratio, "finite": result["finite"]})

    header = sorted({k for r in control_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in control_rows]
    (DOCS_DIR / "balanced_truncation_control.csv").write_text("\n".join(lines))
    print(f"\nwrote {DOCS_DIR / 'balanced_truncation_control.csv'}")

    print("\n=== SUMMARY: truncated-M3 median cost_ratio_to_oracle, per case (vs full M3 for reference) ===")
    print(f"{'case':5s} {'trunc_M3_median':>16s} {'n_finite':>9s}")
    for case in CONTROL_CASES:
        these = [r for r in control_rows if r["case"] == case]
        if not these:
            continue
        n_finite = sum(1 for r in these if r["finite"])
        med = float(np.median([r["cost_ratio_to_oracle"] for r in these]))
        print(f"{case:5d} {med:16.4e} {n_finite:9d}/{len(these)}")


if __name__ == "__main__":
    main()
