# layernorm_study

A focused sub-project isolating the role of Layer Normalization — and
specifically prenorm-vs-postnorm placement — inside the S4 architecture used
by the parent `s4dpc` repository.

## Scope

This sub-project does not train models, run sweeps, or modify any code under
`s4dpc/`. It exists to hold the research notes, open questions, and (later)
small, self-contained diagnostic scripts that examine LayerNorm's effect on
the local Jacobian (`dF/dx`, `dF/du`) of a `SequenceBlockNNX` layer, separate
from the full identification/DPC pipeline.

The motivating observation from the parent project's ongoing analysis: the
learned S4 surrogate's local linearization is state- and history-dependent
even though the true plant is a fixed linear system, including a "kink" in
the local gain near the origin. LayerNorm — a state-dependent nonlinear
rescaling by construction — is a prime suspect for that kink. See
`NOTES.md` for the full argument and open questions.

Note: the parent repo's `CLAUDE.md` records that this kink hypothesis was
**tested directly on control-side DPC data and refuted** as the primary
driver of DPC failure (M3, which has zero kink by construction, still fails
DPC by orders of magnitude). This sub-project is retained because the
Jacobian-fidelity question is narrower and still open on its own terms
(does LayerNorm placement measurably distort `dF/dx`/`dF/du` locally,
independent of whether that distortion turns out to explain the larger DPC
failure) — see `NOTES.md`'s Motivation section for how the two are related
but distinct.

## Relationship to the parent repo

- Reuses the parent repo's conceptual framing (Jacobian fidelity, the
  M0–M6 variant ladder, the `SequenceBlockNNX` prenorm/postnorm switch in
  `s4dpc/blocks.py`) but does not import or depend on `s4dpc/` code yet.
- Any experiment code eventually added here should stay import-only against
  `s4dpc/` (e.g. reusing `s4dpc/systems.py` ground-truth plants or
  `s4dpc/blocks.py`'s block config), never fork or duplicate it.
- Findings that end up informing the parent paper belong in the parent
  repo's `docs/DECISIONS.md`, dated, per the parent `CLAUDE.md`'s
  documentation rules — this sub-project's `NOTES.md` is a scratchpad, not
  the paper's record of truth.

## Status

**Scaffolding only — no experiments run yet.** This folder currently
contains only this README and research notes. No model code, training
scripts, or numerical results exist here.
