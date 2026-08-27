# NOTES

## Motivation

The parent repo's `SequenceBlockNNX` wraps each S4 layer as either

```
prenorm:  x -> LayerNorm -> S4 -> GELU -> GLU -> (+ skip)
postnorm: x -> S4 -> GELU -> GLU -> (+ skip) -> LayerNorm
```

controlled by a `prenorm` boolean. The parent project's control-side
analysis found that the learned surrogate's local Jacobians (`dF/dx`,
`dF/du`) are state- and history-dependent even though the true plant being
identified is a fixed linear system `(A_d, B_d)`. In particular there is a
"kink" in the local gain as the state passes through the origin.

LayerNorm is a natural suspect for this, independent of any specific
downstream DPC result, for a structural reason:

- LayerNorm normalizes by the input's own statistics
  (`(x - mean(x)) / sqrt(var(x) + eps)`), which makes the map from layer
  input to layer output a *non-homogeneous, state-dependent* function even
  when everything else in the block is linear. A true LTI plant has a
  single constant Jacobian everywhere in state space; a LayerNorm'd block
  does not, by construction — its local Jacobian depends on the norm and
  direction of the current input, not just a fixed weight matrix.
- This directly threatens the premise that a learned block can represent a
  linear plant with a *constant* Jacobian. If the true system is
  `x_{t+1} = A_d x_t + B_d u_t`, a block containing LayerNorm cannot
  represent this exactly for all `x_t` — at best it approximates a
  constant Jacobian locally, with the approximation error growing as
  `||x||` shrinks (relatively) or as the input direction changes.
- **Equilibrium concern:** at `x = 0` (or more precisely, whenever the
  block's pre-norm input is the zero vector), `var(x) = 0`, so the
  normalization is `0 / sqrt(0 + eps) = 0`, entirely dominated by the
  `eps` floor rather than by any meaningful signal. This is exactly the
  operating point a regulation-task DPC controller drives the system
  toward (the origin is the setpoint), so if LayerNorm's behavior is
  singular or numerically degenerate near `x = 0`, it is singular exactly
  where the controller spends most of its time and needs Jacobian
  fidelity the most. This also connects to the parent repo's separately
  documented finding that the learned block is *affine*, not linear
  (`F(0, 0, s) != 0`, nonzero `equilibrium_drift`) — LayerNorm is one
  plausible structural source of a nonzero map-of-the-origin, since even a
  literal zero input does not guarantee a zero pre-activation once GELU,
  GLU, and skip connections are composed around it, and the norm itself
  behaves differently in the immediate neighborhood of zero than it does
  away from it.
- Prenorm and postnorm place this non-homogeneous operation at different
  points in the residual computation graph, so they are not guaranteed to
  distort the Jacobian in the same way or by the same amount. Prenorm
  normalizes the S4 layer's *input* before every layer; postnorm
  normalizes the *output* of the full sub-block (S4 + GELU + GLU + skip).
  Since the skip connection in prenorm bypasses LayerNorm entirely while
  in postnorm it does not, the two placements could plausibly have very
  different equilibrium and near-origin behavior even holding all other
  weights fixed.

**Relationship to the parent repo's kink-refutation finding.** The parent
repo's `CLAUDE.md` documents that the kink hypothesis was tested directly
on control-side DPC data and refuted as an explanation for *DPC failure*
(M3 — zero kink by construction, no norm/activation/glu — still fails DPC
by 2-6 orders of magnitude, and kink magnitude anti-correlates with DPC
cost ratio across cases). That refutation is about whether the kink
explains the *specific, large* M3-vs-M6 DPC cost gap. It does **not**
establish that LayerNorm has no effect on Jacobian fidelity at all — only
that some other mechanism (the augmented-state realization/gauge story,
see parent `CLAUDE.md` §1's later entries) dominates the DPC failure
magnitude. Whether LayerNorm placement measurably perturbs `dF/dx`/`dF/du`
locally, and whether that perturbation compounds with or is dwarfed by the
realization-level mechanism, remains open and is the actual scope of this
sub-project. This sub-project is not resurrecting the refuted
"kink causes the DPC failure" claim — it is asking a narrower, still-open
question about LayerNorm's local Jacobian behavior on its own terms.

## LayerNorm background

**Definition.** For an input vector `x` of dimension `d` (normalized over
the feature axis), LayerNorm computes

```
mu    = mean(x)                         # scalar
var   = mean((x - mu)^2)                # scalar, biased (population) variance
x_hat = (x - mu) / sqrt(var + eps)
y     = gamma * x_hat + beta
```

where `gamma, beta` are learnable per-feature scale and shift parameters,
and `eps` is a small constant (commonly `1e-5` or `1e-6`) added purely for
numerical stability — it is not a modeling choice, but it does mean
`x_hat` is not exactly scale-invariant at very small `||x - mu||`.

**Analytical Jacobian.** Writing `x_hat = (x - mu * 1) / sigma` with
`sigma = sqrt(var + eps)` and `1` the all-ones vector, the Jacobian of
`x_hat` with respect to `x` (before applying `gamma`/`beta`) is

```
d(x_hat)/dx = (1/sigma) * ( I - (1/d) * 1 @ 1^T - x_hat @ x_hat^T / d )
```

(This is the standard LayerNorm backward-pass Jacobian; see e.g. the
derivation used in most transformer LayerNorm backward implementations.)
Structurally, this is `(1/sigma)` times a projection-like operator: the
`I - (1/d) 1 1^T` term projects out the mean direction (removing the
all-ones component, since mean-subtraction is invariant to shifts along
`1`), and the `x_hat x_hat^T / d` term further projects out the current
normalized-direction component (removing the scale/variance component,
since scaling `x` by any positive constant leaves `x_hat` unchanged). The
composition is an oblique projection onto the subspace orthogonal to both
`1` and the current `x_hat` direction, scaled by `1/sigma`. Two
consequences matter for this sub-project:

1. **The `1/sigma` scaling makes the Jacobian's magnitude depend on the
   input's own norm** (through `var`), not just its direction — this is
   the direct mechanism for state-dependent gain.
2. **The projection means directions aligned with the current `x_hat`
   (or with `1`) are annihilated** in the LayerNorm Jacobian — a term
   downstream (GELU, GLU, S4) sees a different effective input
   sensitivity depending on the current state's direction, not a fixed
   linear map.

After the learnable affine step, the full LayerNorm Jacobian is
`diag(gamma) @ d(x_hat)/dx` — `gamma` rescales each output row but does
not change the state-dependence structure above.

**RMSNorm comparison.** RMSNorm drops the mean-centering step entirely:

```
rms   = sqrt(mean(x^2) + eps)
y     = gamma * (x / rms)
```

RMSNorm's Jacobian is `(1/rms) * (I - x @ x^T / (d * rms^2))` scaled by
`gamma` — no `1 1^T` mean-projection term, and no separate `beta` shift
(most RMSNorm formulations omit the additive bias). It is still
state-dependent through `rms` and still contains a projection along the
current input direction, so it does **not** trivially solve the
non-homogeneity problem motivating this study — but it removes one source
of it (mean-centering) and is a natural ablation arm alongside "no norm at
all" (M3) and full LayerNorm (M5/M6).

## Prenorm vs postnorm

Standard framing from the Transformer literature (**claims below need
citations — see Reading List**):

- **Postnorm** (original Transformer, Vaswani et al.) applies LayerNorm
  after the residual addition. This is reported to require careful
  learning-rate warmup to train stably at depth — without warmup, gradient
  magnitudes through the un-normalized residual stream are claimed to
  grow uncontrollably in early training. *[NEEDS CITATION]*
- **Prenorm** (used in GPT-2 and most modern large-scale transformers)
  applies LayerNorm to the sub-block's input, before the sub-block, with
  the residual/skip path bypassing normalization entirely. This is
  reported to give a cleaner identity path through the residual stream
  (gradients can flow through the skip connection without passing through
  any normalization), improving training stability at depth and often
  removing the need for warmup. *[NEEDS CITATION]*
- The general claimed trade-off in the literature: prenorm trains more
  stably/robustly but can have worse final performance or "representation
  collapse" at very large depth compared to a well-tuned postnorm model,
  because the identity path can let the effective network act shallower
  than its layer count. *[NEEDS CITATION]*

**Why this matters here, distinct from the general transformer story:**
this project is not concerned with training-depth stability per se (the
S4 stack here is shallow), but the *identity-path* framing maps directly
onto the Jacobian-fidelity question above. Prenorm's skip connection
bypassing LayerNorm means part of the block's output-to-input map is
exactly linear (the skip term) regardless of state; postnorm's
LayerNorm-after-everything means the *entire* block output, including the
skip contribution, is subject to the state-dependent rescaling. This
suggests (untested — see Open Research Questions) that prenorm may
preserve a larger constant-Jacobian component than postnorm, which would
be a novel, pipeline-specific point not obviously covered by the existing
transformer-training literature (which is about gradient flow during
optimization, not the trained model's own input-output Jacobian at
inference/rollout time).

## Open research questions

Numbered, falsifiable, ordered roughly from cheapest/most-local to
most-expensive/most-global:

1. **Does prenorm vs postnorm change one-step identification MSE?**
   Falsifiable via: train matched M5-prenorm / M5-postnorm (or M6
   variants) checkpoints on the same case/seed grid used elsewhere in the
   parent repo, compare one-step teacher-forced MSE. A null result (no
   difference) would suggest placement doesn't matter for raw fit quality.

2. **Does prenorm vs postnorm change `dF/dx`/`dF/du` fidelity against the
   true `(A_d, B_d)`?** Falsifiable via: compute the local Jacobian
   (autodiff, as already done elsewhere in the parent repo's diagnostics)
   at a grid of states/inputs for both placements, compare relative
   Frobenius error against `A_d`/`B_d`. This is the direct test of the
   Motivation section's structural argument.

3. **Does prenorm vs postnorm change the magnitude of the origin kink?**
   Falsifiable via: measure the Jacobian's discontinuity/curvature as
   state crosses `x = 0` (e.g. compare `dF/dx` evaluated at `x = +delta`
   vs `x = -delta` for small `delta`, sweeping `delta -> 0`) for both
   placements. A meaningfully smaller kink under one placement would be a
   clean, actionable result even without touching the larger DPC-failure
   question.

4. **Does removing LayerNorm entirely (M3-style) or swapping to RMSNorm
   recover a constant Jacobian?** Falsifiable via: same Jacobian-grid
   measurement as (2)/(3), applied to a no-norm block and an RMSNorm
   block. Prediction from the Motivation section's math: no-norm should
   recover an exactly constant Jacobian (already established for M3 in
   the parent repo — zero kink by construction); RMSNorm is predicted to
   still show *some* state-dependence (it retains a projection/scaling
   structure) but this has not been measured directly.

5. **Does the prenorm/postnorm choice change downstream DPC closed-loop
   stability or cost**, independent of whatever the dominant
   realization-level mechanism turns out to be? Falsifiable via: run the
   parent repo's existing DPC harness with matched prenorm/postnorm
   checkpoints, compare closed-loop cost ratios. Given the parent repo's
   finding that kink magnitude does *not* correlate with DPC cost ratio
   (Spearman = -0.54, wrong sign, n=6), the prior going into this question
   should be that placement differences here are a second-order effect at
   best — this question is included for completeness and because a null
   result here is itself informative (it would further support the
   realization/gauge-freedom explanation over any norm-placement-based
   one).

## Reading list

*(empty — fill in with prenorm/postnorm and LayerNorm-analysis papers as
they're identified; the "NEEDS CITATION" flags above in particular need
sourcing before any claim here goes into the parent paper)*
