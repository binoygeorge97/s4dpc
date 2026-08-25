"""Regression guard (user, 2026-08-25, TASK 2): a "transfer to the true
plant" stability check must close the loop on the TRUE plant's dynamics,
not the model's own learned dynamics - using the model's own (A, B)
for both DARE synthesis AND the transfer simulation is tautological (a
model's own LQR gain trivially stabilizes the model's own dynamics for
any stabilizable pair, regardless of whether the model matches reality)
and can only manufacture false SUCCESSES, never false failures
(docs/DECISIONS.md's TASK 4 entry, 2026-08-25 - this exact bug was
caught in tools/lqr_transfer_b320.py before being reported).

This test constructs a synthetic (A_true, A_model) pair where a
model-designed gain is real, provably UNSTABLE on the true plant, and
asserts robust_margin_and_rho - the actual production function every
Family-B script in tools/ calls - reports it unstable when given the
TRUE plant's matrices (the correct construction). It also verifies the
scenario is a genuine test, not a vacuous one: the SAME gain
tautologically LOOKS stable if evaluated against the model's own
(wrong) dynamics instead - so a future regression that swaps A_true
for the model's own A in a transfer check would flip this test's first
assertion from pass to fail.

Deliberately stays in the rho>=1 (unstable) branch of
robust_margin_and_rho for the real-transfer check, which short-circuits
before `import control` (see that function's source) - python-control
is an undocumented, unpinned dependency of this project's LQR-transfer
scripts (not in requirements.lock, discovered while writing this test),
so a test in the tracked suite must not require it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

from lqr_transfer_to_true_plant import robust_margin_and_rho  # noqa: E402


def test_transfer_check_catches_a_model_that_only_stabilizes_itself():
    # True plant: one genuinely unstable mode (eigenvalue 2.0), one
    # already-stable, uncontrolled mode.
    A_true = np.diag([2.0, 0.5])
    B_true = np.array([[1.0], [0.0]])

    # The "model": wrongly believes BOTH modes are already stable at
    # 0.5 - exactly the shape of this project's own established Axx
    # failure mode (a learned Axx that understates the true plant's
    # instability).
    A_model = np.diag([0.5, 0.5])
    B_model = B_true.copy()

    Q = 5.0 * np.eye(2)
    R = 0.1 * np.eye(1)
    P = solve_discrete_are(A_model, B_model, Q, R)
    K = np.linalg.solve(R + B_model.T @ P @ B_model, B_model.T @ P @ A_model)

    # Sanity check on the scenario itself: the SAME gain, evaluated
    # tautologically against the MODEL's own (wrong) dynamics, must
    # look stable - otherwise this isn't testing the bug this guards
    # against, it's just testing "is K a bad gain." Computed via a
    # direct eigenvalue check, NOT robust_margin_and_rho: this branch
    # is the rho<1 one, which (unlike the real-transfer check below)
    # does NOT short-circuit before `import control` in that function -
    # see this file's module docstring on why the suite must not
    # require python-control.
    rho_own = float(np.max(np.abs(np.linalg.eigvals(A_model - B_model @ K))))
    assert rho_own < 1.0, (
        f"scenario setup is broken: the model-designed gain should trivially stabilize the "
        f"model's OWN dynamics (got rho_own={rho_own:.4f}) - fix the synthetic A_model/K before "
        f"trusting the real-transfer assertion below"
    )

    # The actual regression guard: the CORRECT construction (true
    # plant's dynamics, not the model's) must report this unstable.
    b_transfer, rho_transfer = robust_margin_and_rho(A_true, B_true, -K)
    assert rho_transfer >= 1.0, (
        f"tautological-stability regression: a gain designed on a model that understates the "
        f"true plant's instability was reported STABLE (rho_transfer={rho_transfer:.4f}) when "
        f"transferred to the true plant - this is the exact bug docs/DECISIONS.md's TASK 4 entry "
        f"(2026-08-25) found and fixed in tools/lqr_transfer_b320.py. Check whether the transfer "
        f"construction under test is using the model's own (A, B) instead of the true plant's."
    )
    assert b_transfer == 0.0  # robust_margin_and_rho's own convention: b=0.0 whenever rho>=1
