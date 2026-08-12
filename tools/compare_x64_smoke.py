"""Task 1 (docs/DECISIONS.md): re-run the M3/M6 smoke test paired
float32-vs-float64 (same seed, same config - CLAUDE.md's own quick-start
command: case 3, 1 seed, 2000 epochs, d_model=16, N=32, n_layers=1) and
report whether the numbers moved. Reads the 4 CSVs written by
launch/kaggle-smoke-x64/runner.ipynb's four `python -m s4dpc.sweep`
invocations (M3/M6 x float64/float32 - four separate process
invocations, not four in-process runs, since jax_enable_x64 is a
process-global flag decided once at s4dpc.sweep's module import).

    python tools/compare_x64_smoke.py
"""
from __future__ import annotations

import csv
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "out_smoke_x64"


def _read_row(path: pathlib.Path) -> dict:
    with path.open() as f:
        return next(csv.DictReader(f))


def main() -> None:
    paths = {
        ("M3", True): OUT_DIR / "m3_x64.csv",
        ("M3", False): OUT_DIR / "m3_f32.csv",
        ("M6", True): OUT_DIR / "m6_x64.csv",
        ("M6", False): OUT_DIR / "m6_f32.csv",
    }

    print(f"{'variant':8s} {'x64':6s} {'teacher_mse':>14s}")
    results: dict[tuple[str, bool], float] = {}
    for (variant, x64), path in paths.items():
        row = _read_row(path)
        mse = float(row["teacher_mse"])
        x64_in_csv = row["x64"] == "True"
        assert x64_in_csv == x64, f"{path}: expected x64={x64}, csv row says x64={row['x64']!r}"
        results[(variant, x64)] = mse
        print(f"{variant:8s} {str(x64):6s} {mse:14.6e}")

    print("\n[did the numbers move?]")
    for variant in ("M3", "M6"):
        f32 = results[(variant, False)]
        f64 = results[(variant, True)]
        ratio = f64 / f32 if f32 != 0 else float("nan")
        print(f"  {variant}: float32={f32:.6e}  float64={f64:.6e}  ratio(f64/f32)={ratio:.6e}  "
              f"moved={'YES' if abs(ratio - 1.0) > 0.01 else 'no (within 1%)'}")


if __name__ == "__main__":
    main()
