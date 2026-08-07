# s4dpc

Research code for a conference paper on why differentiable predictive
control (DPC) fails when the plant model is a learned S4 surrogate, even
when one-step prediction MSE is low.

Core claim under test: **control-relevant derivative fidelity**, not
prediction error, determines DPC success. Suspected mechanism: LayerNorm
(degree-0 homogeneous, so it cannot represent a linear map) distorts the
learned Jacobian near the regulation setpoint.

The full operating guide — golden rules, Kaggle/TACC workflow, W&B
conventions, the variant ladder — is [CLAUDE.md](CLAUDE.md). This file is
just a status snapshot.

## Status: infrastructure and parity scaffolding, no model/controller yet

**Done:**

- End-to-end CLI/logging/Kaggle harness — `env_probe.py`, `s4dpc/sweep.py`,
  `s4dpc/logging.py`, `launch/kaggle/` — proven on both Kaggle T4 and CPU.
  `env_probe.py` also runs a model canary against the real `s4-nnx`
  package: param-tree hash, conv-mode vs. step-mode forward-pass parity,
  one training-step gradient digest.
- `legacy/` — a verbatim port of the original notebook's S4 implementation
  (`legacy/s4.py`, byte-identical to the source). This is the bit-exact
  parity target the `s4-nnx`-based refactor has to reproduce. A missing
  import in the original is patched externally, after import
  (`legacy/_shim.py`), without editing the original file.
- `s4dpc/systems.py` — the real plant matrices: 7 discrete-time LTI systems
  (all `A:(6,6)`, `B:(6,3)`), plus 2 extra dimension-agnostic cases for
  testing. Canonical discretization is bilinear/Tustin at `dt=0.01`,
  settled empirically against a competing ZOH@0.02 codepath by reproducing
  known `rho(A_d)` values — see `docs/DECISIONS.md` for that and other
  recorded decisions (case 4's rescaled matrix entries, case 1's actual
  stability character).
- `s4dpc/data.py` — APRBS-driven trajectory data generation
  (`generate_microgrid_trajectory`, `create_microgrid_dataset`).
- `tools/make_reference_checkpoint.py` — trains the legacy model on real
  case-3 data and writes `tests/fixtures/reference_model.msgpack` plus a
  JSON sidecar (config, versions, param-tree hash, forward-pass digests).
  Verified bit-reproducible across independent Kaggle CPU runs.
- `tests/` — decode-construction parity for the `s4-nnx` port, and a
  systems-table sanity suite (shapes; spectral radius; SVD-based
  transient-growth and Kreiss-like amplification measures — deliberately
  not `np.linalg.eig`, which is unreliable on the defective/near-defective
  matrices some of these cases use by design).
- `gpu_ledger.csv` — running log of Kaggle GPU minutes against the weekly
  quota.

**Not started yet:** `s4dpc/model.py`, `s4dpc/blocks.py`,
`s4dpc/identify.py`, `s4dpc/control.py`, `s4dpc/diagnostics.py`, and
`tests/test_parity.py` (the bit-exact refactor-vs-`legacy` comparison
against the reference checkpoint above).

## Quick start

```bash
python env_probe.py                          # env/device/model canary
pytest tests/                                 # parity + systems tests
python tools/make_reference_checkpoint.py     # regenerate the fixture
```

Kaggle kernels live under `launch/kaggle/` (smoke test: env probe + sweep
placeholder) and `launch/kaggle-checkpoint/` (reference-checkpoint
generation, CPU-only). See CLAUDE.md §4 for the push/poll/pull workflow.
