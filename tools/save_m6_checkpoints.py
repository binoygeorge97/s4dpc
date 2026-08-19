"""TASK D (user, 2026-08-19, seventh round): M6 has been used as a
reference point throughout this project (the kink hypothesis, the
original variant-ladder work, Task 3/6's DPC comparisons) but was never
checkpointed by any script that still holds its weights - every M6
number in this repo traces back to a training run whose params were
never saved. Real GPU training needed (unlike M0_S4's deterministic
construction) - same identification recipe every other checkpoint in
this project uses (`tools/nu_gap_export.py`'s own M3 identification:
d_model=16, N=32, n_layers=1, l_max=100, 40000 epochs, cases {1,2,3,4,5,7}
x 5 seeds), variant swapped to M6, with `save_checkpoint` called for
every member - the same mechanism that produced the 30 fullM3 msgpack
checkpoints this whole session's Task 0 work verified.

    python tools/save_m6_checkpoints.py
"""
from __future__ import annotations

import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from s4dpc.identify import run_identify, save_checkpoint
from s4dpc.systems import get_discrete_matrices  # noqa: F401  (import parity with other scripts, unused directly)

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5
EPOCHS = 40000
EXPORT_DIR = _REPO_ROOT / "docs" / "nu_gap_export"


def main() -> None:
    ckpt_dir = EXPORT_DIR / "ckpt"
    all_done = all(
        (ckpt_dir / f"M6_case{c}_seed{s}.msgpack").exists() for c in CASES for s in range(N_SEEDS)
    )
    if all_done:
        print("all M6 checkpoints already exist - nothing to do")
        return

    t0 = time.time()
    id_rows = run_identify(
        variant="M6", cases=CASES, n_seeds=N_SEEDS, epochs=EPOCHS,
        d_model=D_MODEL, N=STATE_SIZE, n_layers=N_LAYERS, l_max=L_MAX,
        checkpoint_dir=EXPORT_DIR, checkpoint_every=2000,
    )
    print(f"identification wall time: {time.time() - t0:.1f}s")

    ident_config = {"epochs": EPOCHS, "d_model": D_MODEL, "N": STATE_SIZE, "n_layers": N_LAYERS, "l_max": L_MAX}
    for row in id_rows:
        path = save_checkpoint(row, ident_config, EXPORT_DIR)
        print(f"saved {path.name}  teacher_mse={row['teacher_mse']:.4e}")

    diverged = [(r["case"], r["seed"]) for r in id_rows if r["teacher_mse"] > 10.0]
    print(f"\nwrote {len(id_rows)} M6 checkpoints to {ckpt_dir}")
    if diverged:
        print(f"diverged (teacher_mse>10.0), tagged not deleted per CLAUDE.md sec 3 rule 6: {diverged}")


if __name__ == "__main__":
    main()
