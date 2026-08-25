"""TASK 3 (user, 2026-08-25): disambiguate what n_unstable counts, and
characterize the latent block A_ss separately, for both B=1 and B=320.

Existing convention (tools/identify_stability_constrained.py:298-300,
tools/identify_b320.py's own n_unstable) counts eigenvalues of the FULL
augmented Abar (D_X physical rows + n_s S4-latent rows, 1030 total for
d_model=16/N=32), threshold |lambda|>1.0 strictly - confirmed by direct
code comparison, not assumed. This script additionally isolates the
latent block A_ss = Abar[D_X:, D_X:] on its own.

Pure linear algebra on saved checkpoints - no training, no GPU.

    python tools/characterize_n_unstable.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)  # BEFORE any other jax import/op

import jax.numpy as jnp
import numpy as np
from flax import nnx
import flax.serialization as serialization

from s4dpc.blocks import BlockConfig, VARIANTS
from s4dpc.identify import D_INPUT, D_OUTPUT
from s4dpc.model import StackedModel

D_MODEL, STATE_SIZE, N_LAYERS, L_MAX = 16, 32, 1, 100
D_X = 6
CASES = [1, 2, 3, 4, 5, 7]
N_SEEDS = 5


def augmented_operator(graphdef, params, D_X: int, d_model: int, N: int):
    n_s = 2 * d_model * N

    def _unpack(z):
        x = z[:D_X]
        s_re = z[D_X:D_X + d_model * N].reshape(d_model, N)
        s_im = z[D_X + d_model * N:].reshape(d_model, N)
        return x, [s_re + 1j * s_im]

    def _pack(x, s):
        s0 = s[0]
        return jnp.concatenate([x, s0.real.ravel(), s0.imag.ravel()])

    def f(z, u):
        x, s = _unpack(z)
        m = nnx.merge(graphdef, params)
        model_in = jnp.concatenate([x, u], axis=-1)
        out, new_s = m(model_in, s)
        return _pack(out, new_s)

    D_U = 3
    z0 = jnp.zeros(D_X + n_s)
    u0 = jnp.zeros(D_U)
    Abar = jax.jacfwd(lambda z: f(z, u0))(z0)
    return np.asarray(Abar)


def load_model_step(path: pathlib.Path):
    block_config = BlockConfig(d_model=D_MODEL, N=STATE_SIZE, l_max=L_MAX, **VARIANTS["M3"])
    model = StackedModel(
        block_config=block_config, d_input=D_INPUT, d_output=D_OUTPUT, n_layers=N_LAYERS,
        decode=True, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
    )
    pure_dict = serialization.msgpack_restore(path.read_bytes())
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)
    return model


def analyze(ckpt_dir: pathlib.Path, prefix: str, label: str) -> tuple[list[dict], list[np.ndarray]]:
    rows = []
    all_abs_ass = []
    for case in CASES:
        for seed in range(N_SEEDS):
            path = ckpt_dir / f"{prefix}_case{case}_seed{seed}.msgpack"
            if not path.exists():
                continue
            model = load_model_step(path)
            graphdef, params = nnx.split(model, nnx.Param)
            Abar = augmented_operator(graphdef, params, D_X, D_MODEL, STATE_SIZE)
            Ass = Abar[D_X:, D_X:]

            eig_full = np.linalg.eigvals(Abar)
            eig_ass = np.linalg.eigvals(Ass)
            abs_full = np.abs(eig_full)
            abs_ass = np.abs(eig_ass)
            all_abs_ass.append(abs_ass)

            row = {
                "label": label, "case": case, "seed": seed,
                "n_unstable_full": int(np.sum(abs_full > 1.0)),
                "max_abs_full": float(abs_full.max()),
                "n_unstable_ass": int(np.sum(abs_ass > 1.0)),
                "n_near_circle_ass": int(np.sum((abs_ass > 1 - 1e-2) & (abs_ass <= 1.0))),
                "max_abs_ass": float(abs_ass.max()),
                "ass_dim": int(Ass.shape[0]),
            }
            rows.append(row)
            print(f"[{label}/case{case}/seed{seed}] n_unstable_full={row['n_unstable_full']}  "
                  f"n_unstable_ass={row['n_unstable_ass']}  n_near_circle_ass={row['n_near_circle_ass']}  "
                  f"max_abs_ass={row['max_abs_ass']:.6f}")
    return rows, all_abs_ass


def print_histogram(label: str, all_abs_ass: list[np.ndarray]) -> None:
    pooled = np.concatenate(all_abs_ass)
    bins = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0, np.inf]
    counts, _ = np.histogram(pooled, bins=bins)
    print(f"\n{label} |lambda(A_ss)| histogram, pooled over {len(all_abs_ass)} checkpoints, "
          f"{pooled.size} total eigenvalues:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        hi_s = "inf" if np.isinf(hi) else f"{hi}"
        print(f"  [{lo}, {hi_s}): {counts[i]}")


def main() -> None:
    b1_dir = _REPO_ROOT / "docs" / "nu_gap_export" / "ckpt"
    b320_dir = _REPO_ROOT / "docs" / "b320" / "ckpt"

    rows1, abs_ass_1 = analyze(b1_dir, "M3", "B1")
    rows2, abs_ass_320 = analyze(b320_dir, "M3_b320", "B320")
    rows = rows1 + rows2
    print_histogram("B1", abs_ass_1)
    print_histogram("B320", abs_ass_320)

    out_path = _REPO_ROOT / "docs" / "n_unstable_characterization.csv"
    header = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(header)] + [",".join(str(r.get(h, "")) for h in header) for r in rows]
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")

    for label in ["B1", "B320"]:
        these = [r for r in rows if r["label"] == label]
        if not these:
            continue
        import statistics as st
        nuf = [r["n_unstable_full"] for r in these]
        nua = [r["n_unstable_ass"] for r in these]
        nca = [r["n_near_circle_ass"] for r in these]
        maf = [r["max_abs_full"] for r in these]
        maa = [r["max_abs_ass"] for r in these]
        print(f"\n=== {label} (n={len(these)}) ===")
        print(f"n_unstable_full: median={st.median(nuf)} min={min(nuf)} max={max(nuf)}")
        print(f"max_abs_full: median={st.median(maf):.6f} max={max(maf):.6f}")
        print(f"n_unstable_ass (latent block only): median={st.median(nua)} min={min(nua)} max={max(nua)}")
        print(f"n_near_circle_ass (1-1e-2, 1]: median={st.median(nca)} min={min(nca)} max={max(nca)}")
        print(f"max_abs_ass: median={st.median(maa):.6f} max={max(maa):.6f}")


if __name__ == "__main__":
    main()
