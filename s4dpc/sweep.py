"""Sweep entry point — the only entry point (CLAUDE.md §3 rule 1).

Notebooks, Kaggle kernels, and SLURM jobs all call this with different
flags; there is no platform-specific training code.

Currently a placeholder: the "experiment" fits a synthetic 2x2 linear system
by least squares. No S4, no controller — this exists only to prove the CLI
-> compute -> CSV path end to end before any model/control code lands.
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from s4dpc.logging import Logger, write_csv

VARIANTS = ("M0", "M1", "M3", "M4", "M5", "M6", "M6_fix")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="s4dpc sweep entry point")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--cases", required=True, type=int, help="number of systems to sweep")
    p.add_argument("--n_seeds", required=True, type=int)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--wandb", default="off", choices=("online", "offline", "off"))
    p.add_argument("--out", required=True, type=str, help="CSV output path")
    p.add_argument(
        "--no-vmap",
        action="store_true",
        help="loop in Python instead of jax.vmap, for readable tracebacks",
    )
    return p.parse_args(argv)


def _fit_2x2_core(case: int, seed: jax.Array, epochs: int) -> tuple[jax.Array, jax.Array]:
    # epochs is accepted for CLI-surface parity; unused until a real training loop exists
    del epochs
    key = jax.random.fold_in(jax.random.PRNGKey(case), seed)
    key_a, key_x, key_noise = jax.random.split(key, 3)

    q, _ = jnp.linalg.qr(jax.random.normal(key_a, (2, 2)))
    a_true = 0.9 * q

    n_samples = 64
    x0 = jax.random.normal(key_x, (n_samples, 2))
    x1 = x0 @ a_true.T + 1e-3 * jax.random.normal(key_noise, (n_samples, 2))

    # normal equations: a_hat_t solves x0 @ a_hat_t ~= x1
    a_hat_t = jnp.linalg.solve(x0.T @ x0, x0.T @ x1)

    fit_err = jnp.linalg.norm(a_hat_t.T - a_true)
    mse = jnp.mean((x0 @ a_hat_t - x1) ** 2)
    return fit_err, mse


def run_sweep(variant: str, cases: int, n_seeds: int, epochs: int, use_vmap: bool) -> list[dict]:
    if n_seeds == 0:
        return []

    rows: list[dict] = []
    for case in range(cases):
        seeds = jnp.arange(n_seeds)
        if use_vmap:
            fit_errs, mses = jax.vmap(lambda s: _fit_2x2_core(case, s, epochs))(seeds)
        else:
            outs = [_fit_2x2_core(case, s, epochs) for s in seeds]
            fit_errs = jnp.stack([o[0] for o in outs])
            mses = jnp.stack([o[1] for o in outs])
        for seed, fit_err, mse in zip(range(n_seeds), fit_errs, mses):
            rows.append(
                {
                    "variant": variant,
                    "case": case,
                    "seed": seed,
                    "epochs": epochs,
                    "a_fit_err_fro": float(fit_err),
                    "mse": float(mse),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    t0 = time.time()
    rows = run_sweep(
        variant=args.variant,
        cases=args.cases,
        n_seeds=args.n_seeds,
        epochs=args.epochs,
        use_vmap=not args.no_vmap,
    )
    elapsed = time.time() - t0

    with Logger(mode=args.wandb) as logger:
        for row in rows:
            logger.log(row)

    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out} in {elapsed:.2f}s (vmap={not args.no_vmap})")


if __name__ == "__main__":
    main()
