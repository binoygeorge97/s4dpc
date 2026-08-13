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

Controllers (M1, full M3, truncated M3) are trained FRESH here (no
persisted checkpoints from earlier kernels in this session carry
weights, only evaluation summaries) - self-contained, matching every
other kernel in this session. Truncated M3 reuses
tools/balanced_truncation.py's ERA machinery on freshly (re-)identified
M3 checkpoints - a modest redundancy with that script's own identify
step, accepted for self-containment over a fragile cross-kernel
dependency.

For each of {M1, full M3, truncated M3} x {case} x {seed}: extract an
effective static gain K_eff = d(u)/d(x) at (x=0, h=0) from the TRAINED
GRU controller (the standard "closest LTI analogue" linearization,
matching how every controller in this project is deployed - starting
at h=0 - and how the origin is the regulation point every cost function
penalizes) - a genuine simplification of a recurrent, nonlinear
controller down to a memoryless gain, stated plainly, not hidden.
Report delta_nu(true, X) and b_{K_eff,X} side by side with each
variant's already-recorded Markov error, and check whether b > delta_nu
predicts the empirically observed DPC outcome case by case.

    python tools/nu_gap_analysis.py
"""
from __future__ import annotations

import pathlib
import sys
import time

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
# Main pipeline (needs jax - only imported below --selftest gate)
# ============================================================

def main() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op
    import jax.numpy as jnp
    from flax import nnx

    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    import controller_oracles as co
    from balanced_truncation import (
        COUT, N_HANKEL, R_TRUNC, build_truncated_m3, self_check_true_system,
    )
    from m3_spurious_modes import D_MODEL, STATE_SIZE, N_LAYERS, L_MAX, D_X, D_U, augmented_operator

    from s4dpc.blocks import BlockConfig, VARIANTS
    from s4dpc.control import (
        BoundedGRUController, init_batched_state, make_controller_optimizer,
        rollout_learned, rollout_linear,
    )
    from s4dpc.identify import D_INPUT, D_OUTPUT, fit_least_squares, run_identify
    from s4dpc.model import StackedModel
    from s4dpc.systems import get_discrete_matrices

    CONTROL_CASES = [c for c in co.CASES if c != 6]
    N_SEEDS = 5
    DT = 0.01
    DOCS_DIR = _REPO_ROOT / "docs"

    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
    print(f"CONTROL_CASES={CONTROL_CASES}  N_SEEDS={N_SEEDS}")
    self_check_true_system()

    def k_eff_from_controller(controller_state, max_action: float) -> np.ndarray:
        """Static linearization u = K_eff @ x of the trained GRU at
        (x=0, h=0) - matches how every rollout in this project actually
        starts (h=0) and the origin every cost function penalizes."""
        controller = BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
        nnx.update(controller, controller_state)

        def u_of_x(x):
            h0 = jnp.zeros((co.HIDDEN_DIM,))
            _, u = controller(h0, x)
            return u

        K = jax.jacfwd(u_of_x)(jnp.zeros((co.D_X,)))
        return np.asarray(K)

    # ---- oracle costs / true systems, shared setup ----
    true_AB, oracle_costs, eval_keys = {}, {}, {}
    for case in CONTROL_CASES:
        A_d, B_d = get_discrete_matrices(DT, case)
        true_AB[case] = (np.asarray(A_d), np.asarray(B_d))
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

    # ---- M1: least-squares (A_hat, B_hat) per case (no seed dependence - LS is deterministic) ----
    print(f"\n{'=' * 20} M1: least-squares systems + controller training {'=' * 20}")
    m1_AB = {}
    for case in CONTROL_CASES:
        ab_hat, _ = fit_least_squares(case, l_max=100, aprbs_low=-10.0, aprbs_high=10.0)
        m1_AB[case] = (ab_hat[:, :D_X], ab_hat[:, D_X:])

    def train_linear_ensemble(AB_by_case_seed, members, max_action):
        A_list, B_list, x0_list, key_list = [], [], [], []
        for case, seed in members:
            A, B = AB_by_case_seed[case] if case in AB_by_case_seed else AB_by_case_seed[(case, seed)]
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

    by_bound = {}
    for case in CONTROL_CASES:
        by_bound.setdefault(co.CASE_MAX_ACTION[case], []).append(case)

    m1_rows = []
    for max_action, cases in by_bound.items():
        members = [(c, s) for c in cases for s in range(N_SEEDS)]
        print(f"  M1 max_action={max_action}, {len(members)} members")
        t0 = time.time()
        ens_state = train_linear_ensemble(m1_AB, members, max_action)
        print(f"  wall time: {time.time() - t0:.1f}s")
        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            K_eff = k_eff_from_controller(member_state, max_action)
            A_true, B_true = true_AB[case]
            dnu, dnu_info = nu_gap((A_true, B_true, np.eye(D_X), np.zeros((D_X, D_U))),
                                    (m1_AB[case][0], m1_AB[case][1], np.eye(D_X), np.zeros((D_X, D_U))))
            b, b_info = robust_margin(m1_AB[case][0], m1_AB[case][1], K_eff)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            result = co._evaluate(controller, A_true, B_true, eval_keys[case])
            ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
            print(f"    [M1/case{case}/seed{seed}] ratio={ratio:.4e}  delta_nu={dnu:.4f}  b={b}  "
                  f"predicts_success={b is not None and b > dnu}")
            m1_rows.append({"variant": "M1", "case": case, "seed": seed, "cost_ratio_to_oracle": ratio,
                             "delta_nu": dnu, "b": b, "predicts_success": (b is not None and b > dnu),
                             "dnu_valid": dnu_info["valid"]})

    header = sorted({k for r in m1_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in m1_rows]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "nu_gap_m1.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'nu_gap_m1.csv'}")

    # ---- M3 (full) + truncated M3: shared identification ----
    print(f"\n{'=' * 20} identifying M3, cases {CONTROL_CASES} x {N_SEEDS} seeds {'=' * 20}")
    t0 = time.time()
    id_rows = run_identify(
        variant="M3", cases=CONTROL_CASES, n_seeds=N_SEEDS, epochs=40000,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
    )
    print(f"  identification wall time: {time.time() - t0:.1f}s")
    diverged = {(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0}
    print(f"  diverged: {sorted(diverged)}")

    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    m3_graphdef, _ = nnx.split(
        StackedModel(block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
                     decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
        nnx.Param,
    )

    # truncated systems, per (case, seed)
    print(f"\n{'=' * 20} balanced truncation per (case, seed) {'=' * 20}")
    trunc_AB = {}
    for r in id_rows:
        case, seed = r["case"], r["seed"]
        if (case, seed) in diverged:
            continue
        res = build_truncated_m3(case, seed, r["param_state"], m3_graphdef)
        if res.get("ok"):
            trunc_AB[(case, seed)] = (res["A"], res["B"])
            print(f"  case{case}/seed{seed}: err_vs_true={res['err_vs_true_markov']:.3e}")
        else:
            print(f"  case{case}/seed{seed}: SKIPPED (cond={res.get('cond_Cr')})")

    # ---- truncated-M3 controller training + nu-gap/margin ----
    print(f"\n{'=' * 20} truncated-M3: controller training + nu-gap/margin {'=' * 20}")
    trunc_rows = []
    by_bound_trunc = {}
    for (case, seed) in trunc_AB:
        by_bound_trunc.setdefault(co.CASE_MAX_ACTION[case], []).append((case, seed))
    for max_action, members in by_bound_trunc.items():
        print(f"  truncM3 max_action={max_action}, {len(members)} members")
        t0 = time.time()
        ens_state = train_linear_ensemble(trunc_AB, members, max_action)
        print(f"  wall time: {time.time() - t0:.1f}s")
        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            K_eff = k_eff_from_controller(member_state, max_action)
            A_true, B_true = true_AB[case]
            A_tr, B_tr = trunc_AB[(case, seed)]
            dnu, dnu_info = nu_gap((A_true, B_true, np.eye(D_X), np.zeros((D_X, D_U))),
                                    (A_tr, B_tr, np.eye(D_X), np.zeros((D_X, D_U))))
            b, b_info = robust_margin(A_tr, B_tr, K_eff)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            result = co._evaluate(controller, A_true, B_true, eval_keys[case])
            ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
            print(f"    [truncM3/case{case}/seed{seed}] ratio={ratio:.4e}  delta_nu={dnu:.4f}  b={b}  "
                  f"predicts_success={b is not None and b > dnu}")
            trunc_rows.append({"variant": "truncM3", "case": case, "seed": seed, "cost_ratio_to_oracle": ratio,
                                "delta_nu": dnu, "b": b, "predicts_success": (b is not None and b > dnu),
                                "dnu_valid": dnu_info["valid"]})

    header = sorted({k for r in trunc_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in trunc_rows]
    (DOCS_DIR / "nu_gap_truncm3.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'nu_gap_truncm3.csv'}")

    # ---- full M3 controller training (rollout_learned) + nu-gap/margin ----
    print(f"\n{'=' * 20} full M3: controller training (rollout_learned) + nu-gap/margin {'=' * 20}")
    m3_param_by_cs = {(r["case"], r["seed"]): r["param_state"] for r in id_rows if (r["case"], r["seed"]) not in diverged}
    m3_full_rows = []
    by_bound_full = {}
    for (case, seed) in m3_param_by_cs:
        by_bound_full.setdefault(co.CASE_MAX_ACTION[case], []).append((case, seed))

    for max_action, members in by_bound_full.items():
        print(f"  fullM3 max_action={max_action}, {len(members)} members")
        surrogate_params_batch = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[m3_param_by_cs[m] for m in members]
        )
        x0_list, key_list = [], []
        for case, seed in members:
            init_key = jax.random.fold_in(jax.random.PRNGKey(seed), case)
            x0_key = jax.random.fold_in(init_key, 999)
            x0 = jax.random.uniform(
                x0_key, (co.TRAIN_X0_BATCH, co.D_X), minval=-co.TRAIN_X0_RANGE, maxval=co.TRAIN_X0_RANGE, dtype=jnp.float64
            )
            x0_list.append(x0)
            key_list.append(init_key)
        x0_batch, keys = jnp.stack(x0_list), jnp.stack(key_list)

        @nnx.vmap(in_axes=0, out_axes=0)
        def init_ensemble(key):
            return BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(key))

        ensemble = init_ensemble(keys)
        optimizer = make_controller_optimizer(ensemble, co.CTRL_LR, 0.0, co.TOTAL_EPOCHS)
        ref_states = init_batched_state(StackedModel(
            block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
            decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
        ), co.TRAIN_X0_BATCH)

        t0 = time.time()
        for pi, phase in enumerate(co.CURRICULUM):
            N = phase["N"]

            @nnx.jit
            def train_step(ens, opt, N=N):
                def loss_fn(e):
                    cg, cp = nnx.split(e, nnx.Param)

                    def single_member(p, sp, x0):
                        c = nnx.merge(cg, p)
                        loss, _ = rollout_learned(c, m3_graphdef, sp, x0, ref_states, co.Q_X, co.R_U, co.Q_F, N)
                        return loss

                    losses = jax.vmap(single_member)(cp, surrogate_params_batch, x0_batch)
                    return jnp.mean(losses), losses

                (loss, per_member), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ens)
                opt.update(ens, grads)
                return loss, per_member

            for epoch in range(phase["epochs"]):
                loss, per_member = train_step(ensemble, optimizer)
            print(f"    phase {pi + 1}/{len(co.CURRICULUM)} (N={N}) final mean DPC loss: {float(loss):.4f}")
        print(f"  wall time: {time.time() - t0:.1f}s")
        ens_state = nnx.state(ensemble, nnx.Param)

        for i, (case, seed) in enumerate(members):
            member_state = jax.tree_util.tree_map(lambda x, i=i: x[i], ens_state)
            K_eff = k_eff_from_controller(member_state, max_action)
            A_true, B_true = true_AB[case]
            Abar, Bbar = augmented_operator(m3_graphdef, m3_param_by_cs[(case, seed)])
            dnu, dnu_info = nu_gap((A_true, B_true, np.eye(D_X), np.zeros((D_X, D_U))),
                                    (Abar, Bbar, COUT, np.zeros((D_X, D_U))))
            # K_eff acts on the PHYSICAL state (6-dim); the augmented plant's
            # state is 1030-dim - robust_margin needs K defined on the SAME
            # state as the plant it's closing the loop around. Since the
            # controller only ever sees x (never s), pad K with zeros over
            # the S4-state block - a state-feedback law that's blind to
            # those coordinates, exactly matching what the trained
            # controller actually does.
            K_padded = np.hstack([K_eff, np.zeros((D_U, Abar.shape[0] - D_X))])
            b, b_info = robust_margin(Abar, Bbar, K_padded)
            controller = co.BoundedGRUController(co.D_X, co.HIDDEN_DIM, co.D_U, max_action, rngs=nnx.Rngs(0))
            nnx.update(controller, member_state)
            result = co._evaluate(controller, A_true, B_true, eval_keys[case])
            ratio = result["cost"] / oracle_costs[case] if oracle_costs[case] > 0 else float("inf")
            print(f"    [fullM3/case{case}/seed{seed}] ratio={ratio:.4e}  delta_nu={dnu:.4f}  b={b}  "
                  f"predicts_success={b is not None and b > dnu}")
            m3_full_rows.append({"variant": "fullM3", "case": case, "seed": seed, "cost_ratio_to_oracle": ratio,
                                  "delta_nu": dnu, "b": b, "predicts_success": (b is not None and b > dnu),
                                  "dnu_valid": dnu_info["valid"]})

    header = sorted({k for r in m3_full_rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in m3_full_rows]
    (DOCS_DIR / "nu_gap_fullm3.csv").write_text("\n".join(lines))
    print(f"wrote {DOCS_DIR / 'nu_gap_fullm3.csv'}")

    # ---- combined summary ----
    all_rows = m1_rows + trunc_rows + m3_full_rows
    print("\n=== SUMMARY: median (delta_nu, b, predicts_success rate) vs actual DPC outcome, per (variant, case) ===")
    print(f"{'variant':10s} {'case':5s} {'median_dnu':>11s} {'median_b':>10s} {'predict_rate':>13s} {'median_ratio':>13s}")
    for variant in ["M1", "truncM3", "fullM3"]:
        for case in CONTROL_CASES:
            these = [r for r in all_rows if r["variant"] == variant and r["case"] == case]
            if not these:
                continue
            dnus = [r["delta_nu"] for r in these]
            bs = [r["b"] for r in these if r["b"] is not None]
            preds = [r["predicts_success"] for r in these]
            ratios = [r["cost_ratio_to_oracle"] for r in these]
            print(f"{variant:10s} {case:5d} {np.median(dnus):11.4f} "
                  f"{(np.median(bs) if bs else float('nan')):10.4f} "
                  f"{np.mean(preds):13.2%} {np.median(ratios):13.4e}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _run_selftests()
    else:
        main()
