"""Shape checks for cases 1-7, and the empirical comparison (Tustin@0.01 vs
ZOH@0.02) that settles which discretization is canonical: whichever one
reproduces the user's prior rho(A_d) values (case 3 ~= 1.02019, case 6
~= 1.034) is the one identify.py and control.py must standardize on.

Never uses np.linalg.eig: its eigenvector output is unreliable for
defective/near-defective matrices (case 4, by design, is close to a
non-diagonalizable Jordan block) and a condition number computed from a
numerically-broken eigenvector matrix can silently read as small/finite
when the true answer is "this matrix doesn't have a reliable eigenbasis."
rho(A_d) uses np.linalg.eigvals (eigenvalues only - no eigenvector solve
to go unstable); everything else here is SVD-based
(np.linalg.norm(..., ord=2) and np.linalg.matrix_power), which stays
well-behaved regardless of how defective A_d is.
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
# transient-growth horizons for ||A_d^k||_2 and the Kreiss-like ratio
K_VALUES = (1, 5, 10, 25, 50)


@pytest.mark.parametrize("case", CASES)
def test_case_shapes(case):
    Ad, Bd = get_discrete_matrices(dt=0.01, case=case)
    assert Ad.shape == (6, 6)
    assert Bd.shape == (6, 3)


def _stats(Ad: np.ndarray) -> dict:
    rho = float(np.max(np.abs(np.linalg.eigvals(Ad))))

    power_norms = {}
    for k in K_VALUES:
        power_norms[k] = float(np.linalg.norm(np.linalg.matrix_power(Ad, k), ord=2))

    # Kreiss-like amplification: max_k ||A^k||_2 / rho^k, over the same
    # k-grid as power_norms (a practical proxy for the true Kreiss constant
    # sup_{k>=1}, not an exact sup over all k).
    kreiss_like = max(power_norms[k] / rho**k for k in K_VALUES)

    non_normality = float(np.linalg.norm(Ad @ Ad.T - Ad.T @ Ad, ord="fro"))

    return {
        "rho": rho,
        "power_norms": power_norms,
        "kreiss_like": kreiss_like,
        "non_normality": non_normality,
    }


def build_comparison_table() -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    for case in CASES:
        for name, discretize in DISCRETIZATIONS:
            Ad, _ = discretize(case)
            rows[(case, name)] = _stats(Ad)
    return rows


def print_comparison_table(rows: dict[tuple[int, str], dict]) -> None:
    power_headers = "".join(f"{'||A^' + str(k) + '||_2':>13}" for k in K_VALUES)
    header = f"{'case':<5}{'method':<14}{'rho(A_d)':>10}{power_headers}{'kreiss-like':>13}{'||AAT-ATA||_F':>15}"
    print(header)
    print("-" * len(header))
    for case in CASES:
        for name, _ in DISCRETIZATIONS:
            s = rows[(case, name)]
            power_cells = "".join(f"{s['power_norms'][k]:>13.3e}" for k in K_VALUES)
            print(
                f"{case:<5}{name:<14}{s['rho']:>10.5f}{power_cells}"
                f"{s['kreiss_like']:>13.3e}{s['non_normality']:>15.3e}"
            )


def test_discretization_comparison_table(capsys):
    rows = build_comparison_table()
    with capsys.disabled():
        print()
        print_comparison_table(rows)

    for stats in rows.values():
        assert np.isfinite(stats["rho"])
        assert all(np.isfinite(v) for v in stats["power_norms"].values())
        assert np.isfinite(stats["kreiss_like"])
        assert np.isfinite(stats["non_normality"])

    # The two discretizations should generally disagree (different dt AND
    # method) - except case 1, whose continuous A has exact zero eigenvalues
    # (block-triangular with [[a,b],[0,0]] diagonal blocks). A zero
    # eigenvalue maps to exactly 1 under any discretization method, so both
    # legitimately agree there; that's not a bug, so it's excluded below
    # rather than asserted away.
    disagreements = [
        case
        for case in CASES
        if rows[(case, "tustin@0.01")]["rho"] != pytest.approx(rows[(case, "zoh@0.02")]["rho"], rel=1e-6)
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
    rho_case3 = _stats(get_discrete_matrices(dt=0.01, case=3)[0])["rho"]
    rho_case6 = _stats(get_discrete_matrices(dt=0.01, case=6)[0])["rho"]
    assert rho_case3 == pytest.approx(1.02019, abs=5e-5)
    assert rho_case6 == pytest.approx(1.034, abs=5e-4)


if __name__ == "__main__":
    print_comparison_table(build_comparison_table())
