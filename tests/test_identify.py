"""s4dpc.identify: vmap and --no-vmap must agree (CLAUDE.md §9, "Same
function, same outputs"). Small-scale (not the real d_model=16/N=32) so
this runs fast as a regular test, not a smoke test.
"""
from __future__ import annotations

import pytest

from s4dpc.identify import run_identify


@pytest.mark.parametrize("variant", ["M3", "M6"])
def test_vmap_matches_no_vmap(variant):
    kwargs = dict(
        variant=variant,
        cases=[3, 6],
        n_seeds=2,
        epochs=5,
        d_model=8,
        N=8,
        n_layers=1,
        l_max=10,
    )
    rows_vmap = run_identify(use_vmap=True, **kwargs)
    rows_novmap = run_identify(use_vmap=False, **kwargs)

    assert len(rows_vmap) == len(rows_novmap) == 2 * 2
    for rv, rn in zip(rows_vmap, rows_novmap):
        assert rv["case"] == rn["case"]
        assert rv["seed"] == rn["seed"]
        assert rv["teacher_mse"] == pytest.approx(rn["teacher_mse"], rel=1e-4)
