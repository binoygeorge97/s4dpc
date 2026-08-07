"""Shape checks for cases 1-7, and the empirical comparison (Tustin@0.01 vs
ZOH@0.02) that settles which discretization is canonical: whichever one
reproduces the user's prior rho(A_d) values (case 3 ~= 1.02019, case 6
~= 1.034) is the one identify.py and control.py must standardize on.
"""
from __future__ import annotations

import numpy as np
import pytest

from s4dpc.systems import get_discrete_matrices, get_discrete_matrices_zoh

CASES = range(1, 8)
DISCRETIZATIONS = (
    ("tustin@0.01", lambda case: get_discrete_matrices(dt=0.01, case=case)),
    ("zoh@0.02", lambda case: get_discrete_matrices_zoh(dt=0.02, case=case)),
)


@pytest.mark.parametrize("case", CASES)
def test_case_shapes(case):
    Ad, Bd = get_discrete_matrices(dt=0.01, case=case)
    assert Ad.shape == (6, 6)
    assert Bd.shape == (6, 3)


def _stats(Ad: np.ndarray) -> tuple[float, float, float, float]:
    eigvals, eigvecs = np.linalg.eig(Ad)
    rho = float(np.max(np.abs(eigvals)))
    norm2 = float(np.linalg.norm(Ad, ord=2))
    cond_eigvecs = float(np.linalg.cond(eigvecs))
    non_normality = float(np.linalg.norm(Ad @ Ad.T - Ad.T @ Ad, ord="fro"))
    return rho, norm2, cond_eigvecs, non_normality


def build_comparison_table() -> dict[tuple[int, str], tuple[float, float, float, float]]:
    rows: dict[tuple[int, str], tuple[float, float, float, float]] = {}
    for case in CASES:
        for name, discretize in DISCRETIZATIONS:
            Ad, _ = discretize(case)
            rows[(case, name)] = _stats(Ad)
    return rows


def print_comparison_table(rows: dict[tuple[int, str], tuple[float, float, float, float]]) -> None:
    header = f"{'case':<5}{'method':<14}{'rho(A_d)':>12}{'||A_d||_2':>12}{'cond(V)':>16}{'||AAT-ATA||_F':>16}"
    print(header)
    print("-" * len(header))
    for case in CASES:
        for name, _ in DISCRETIZATIONS:
            rho, norm2, cond_v, nn = rows[(case, name)]
            print(f"{case:<5}{name:<14}{rho:>12.5f}{norm2:>12.5f}{cond_v:>16.5f}{nn:>16.5f}")


def test_discretization_comparison_table(capsys):
    rows = build_comparison_table()
    with capsys.disabled():
        print()
        print_comparison_table(rows)

    for (case, name), (rho, norm2, cond_v, nn) in rows.items():
        assert np.isfinite(rho) and np.isfinite(norm2) and np.isfinite(cond_v) and np.isfinite(nn)

    # The two discretizations should generally disagree (different dt AND
    # method) - except case 1, whose continuous A has exact zero eigenvalues
    # (block-triangular with [[a,b],[0,0]] diagonal blocks). A zero
    # eigenvalue maps to exactly 1 under any discretization method, so both
    # legitimately agree there; that's not a bug, so it's excluded below
    # rather than asserted away.
    disagreements = [
        case
        for case in CASES
        if rows[(case, "tustin@0.01")][0] != pytest.approx(rows[(case, "zoh@0.02")][0], rel=1e-6)
    ]
    assert disagreements == [c for c in CASES if c != 1], (
        "expected tustin@0.01 and zoh@0.02 to disagree on every case except "
        f"case 1 (zero-eigenvalue coincidence); got disagreements={disagreements}"
    )


def test_canonical_matches_prior_analysis():
    """get_discrete_matrices (bilinear/Tustin @ dt=0.01) is canonical because
    it, not the zoh@0.02 variant, reproduces the user's known-good rho(A_d)
    values. See the printed table from test_discretization_comparison_table
    (run with `-s`) for the full comparison that established this."""
    rho_case3, _, _, _ = _stats(get_discrete_matrices(dt=0.01, case=3)[0])
    rho_case6, _, _, _ = _stats(get_discrete_matrices(dt=0.01, case=6)[0])
    assert rho_case3 == pytest.approx(1.02019, abs=5e-5)
    assert rho_case6 == pytest.approx(1.034, abs=5e-4)


if __name__ == "__main__":
    print_comparison_table(build_comparison_table())
