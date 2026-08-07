"""Sweep entry point — the only entry point (CLAUDE.md §3 rule 1).

Notebooks, Kaggle kernels, and SLURM jobs all call this with different
flags; there is no platform-specific training code.

Teacher-forced one-step MSE system identification (s4dpc.identify),
vmapped over (case x seed) by default; --no-vmap loops in Python instead,
for readable tracebacks, using the exact same per-member function
(CLAUDE.md §9).
"""
from __future__ import annotations

import argparse
import pathlib
import time

from s4dpc.blocks import VARIANTS
from s4dpc.identify import run_identify, save_checkpoint
from s4dpc.logging import Logger, write_csv


def _parse_cases(raw: str) -> list[int]:
    return [int(c) for c in raw.split(",") if c.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="s4dpc sweep entry point")
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument(
        "--cases",
        required=True,
        type=_parse_cases,
        help="comma-separated case numbers, e.g. '3' or '1,2,3,4,5,6,7' (cases 1-7 only; see s4dpc.data)",
    )
    p.add_argument("--n_seeds", required=True, type=int)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--d_model", required=True, type=int)
    p.add_argument("--N", required=True, type=int, dest="N", help="S4 state size")
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--wandb", default="off", choices=("online", "offline", "off"))
    p.add_argument("--out", required=True, type=str, help="CSV output path; checkpoints go to its parent dir / ckpt/")
    p.add_argument(
        "--no-vmap",
        action="store_true",
        help="loop in Python instead of jax.vmap, for readable tracebacks",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    t0 = time.time()
    rows = run_identify(
        variant=args.variant,
        cases=args.cases,
        n_seeds=args.n_seeds,
        epochs=args.epochs,
        d_model=args.d_model,
        N=args.N,
        n_layers=args.n_layers,
        use_vmap=not args.no_vmap,
    )
    elapsed = time.time() - t0

    out_path = pathlib.Path(args.out)
    config = {
        "variant": args.variant,
        "d_model": args.d_model,
        "N": args.N,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
    }

    csv_rows = []
    with Logger(mode=args.wandb) as logger:
        for row in rows:
            save_checkpoint(row, config, out_path.parent)
            csv_row = {
                "variant": row["variant"],
                "case": row["case"],
                "seed": row["seed"],
                "epochs": args.epochs,
                "d_model": args.d_model,
                "N": args.N,
                "n_layers": args.n_layers,
                "teacher_mse": row["teacher_mse"],
            }
            logger.log(csv_row)
            csv_rows.append(csv_row)

    write_csv(args.out, csv_rows)
    print(f"wrote {len(csv_rows)} rows to {args.out} in {elapsed:.2f}s (vmap={not args.no_vmap})")


if __name__ == "__main__":
    main()
