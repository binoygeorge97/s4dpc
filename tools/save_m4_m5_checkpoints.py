"""TASK A (user, 2026-08-19, eighth round): isolate whether M6's conv/step
parity break (docs/DECISIONS.md's TASK 0 EXTENSION entry) is a numerical-
sensitivity nuisance or a genuine non-equivalent nonlinear composition
between conv mode and step mode. Decisive test: run the same parity check
on M4 (activation=gelu, glu=True, norm=none) and M5 (norm=layer,
activation=none, glu=False) - the two single-component slices of M6's
(norm, activation, glu) triple. Neither has a saved checkpoint yet
(only M3/M0_S4/M6 do), so both need real GPU training first - same
recipe as tools/save_m6_checkpoints.py (which is itself the same recipe
every other checkpoint in this project uses).

    python tools/save_m4_m5_checkpoints.py
"""
from __future__ import annotations

import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from s4dpc.identify import run_identify, save_checkpoint

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
EPOCHS = 40000
EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"


def run_variant(variant: str) -> None:
    ckpt_dir = EXPORT_DIR / "ckpt"
    all_done = all(
        (ckpt_dir / f"{variant}_case{c}_seed{s}.msgpack").exists() for c in CASES for s in range(N_SEEDS)
    )
    if all_done:
        print(f"all {variant} checkpoints already exist - nothing to do")
        return

    t0 = time.time()
    id_rows = run_identify(
        variant=variant, cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
        checkpoint_dir=EXPORT_DIR, checkpoint_every=2000,
    )
    print(f"[{variant}] identification wall time: {time.time() - t0:.1f}s")

    ident_config = {"epochs": EPOCHS, "d_model": D_MODEL, "N": STATE_SIZE, "n_layers": N_LAYERS, "l_max": L_MAX}
    for row in id_rows:
        path = save_checkpoint(row, ident_config, EXPORT_DIR)
        print(f"saved {path.name}  teacher_mse={row['teacher_mse']:.4e}")

    diverged = [(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0]
    print(f"\n[{variant}] wrote {len(id_rows)} checkpoints to {ckpt_dir}")
    if diverged:
        print(f"[{variant}] diverged (teacher_mse>10.0), tagged not deleted per CLAUDE.md sec 3 rule 6: {diverged}")


def main() -> None:
    for variant in ["M4", "M5"]:
        run_variant(variant)


if __name__ == "__main__":
    main()
