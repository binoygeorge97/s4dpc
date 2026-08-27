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

## Results, round 1 (2026-08-27)

**Experiment 1 (existing checkpoints M3/M4/M5/M6, 120 checkpoints, all 7
cases x 5 seeds — `results/exp1_jacobian_decomposition.csv`, gitignored,
regenerate via `python -m layernorm_study.experiments.exp1_jacobian_decomposition`):**
the skip/branch decomposition `F(z) = W_dec@W_enc@z + W_dec@branch(z)`
is exact by construction (decomposition residual ~1e-16 everywhere — a
correctness check on the tooling, not a finding). The origin sweep of
`dF/dx` (case1/seed0, representative) is the clean result: M3 (no norm)
is EXACTLY constant from `t=-100` to `t=+100` including `t=1e-6` — zero
kink, as expected by construction. M4 (GELU+GLU, no LN) varies only
mildly (1.63→1.73, ~6%) with no feature localized at the origin — this
matters because the trajectory-averaged "contamination ratio" alone is
a poor discriminator (M4's average is *higher* than M5's despite having
no LayerNorm — it's the origin-LOCALIZED spike specifically, not
general nonlinear state-dependence, that is LayerNorm's signature). M5
(LN only) and M6 (LN+GELU+GLU) both show a sharp ~4-4.7x Jacobian-norm
spike within `1e-5` of the origin, entirely carried by the branch term,
surviving on top of GELU/GLU. Two caveats worth carrying forward: (1)
the skip term alone does NOT recover `[A_d|B_d]` for ANY variant
including M3 (relative error 0.5-1.3x) — training has no reason to
privilege that split, so "branch is a small correction to skip's
correct linear map" is not literally what happens, even though the
kink itself is real; (2) M5 has a real ~0.1-0.3% conv/step numerics gap
(recomputed vs. recorded teacher MSE) that M3 doesn't (M3 matches to
~1e-12) — small, doesn't affect the kink conclusion, but is a genuine
finding not previously documented for M5 specifically (parent repo's
CLAUDE.md documents it for M6 only).

**Experiment 2 (scalar plant `x_next=3x+u`, complexity ladder, single
seed=0 per arm so far — `results/exp2_ladder.csv` + per-arm manifests +
figures, gitignored, regenerate via
`python -m layernorm_study.experiments.exp2_train_ladder`):** data
sanity check passes at machine precision (least squares recovers
`(3.0, 1.0)` to `~1e-16`, far under the `1e-9` gate). arm_0 (skip-only,
`n_layers=0`) passes its positive-control check exactly: `Jx`/`Ju`
match `(3, 1)` to `~2e-8` at every trajectory point, Jacobian exactly
flat (spike ratio `=1`). arm_1 (linear branch, memoryless, no LN) is
likewise exactly flat, as expected for a composition of affine maps.
arm_4 (real S4 memory, no LN, no GELU/GLU) is ALSO exactly flat to
machine precision — confirms memory alone, without LayerNorm, does not
produce state-dependence, matching M3's finding in Experiment 1 at a
completely different (scalar, real-memory) architecture.

arm_2 (linear + LayerNorm, memoryless — the KEY ARM) DOES show real,
autodiff-confirmed Jacobian distortion (a ~3x-swing dip-then-spike in
`|Jx(c)|` vs. sweep parameter `c`) — but, surprisingly, it is NOT
located at the physical origin the way Exp1's M5/M6 spikes were. This
was caught and verified directly, not assumed: a plain sweep of
`Jx` vs. `c` (autodiff, drift-invariant) shows the distortion sitting
at a moderate, off-center `c` value, while `Jx` right at the physical
origin is close to the true value. This is a real refinement, not a
contradiction, of Experiment 1's finding — LayerNorm's singular
direction is set by whether the network's `encoder(0,0)` bias happens
to land near LayerNorm's degenerate (equal-component) subspace, which
is empirically true for Exp1's 6D closed-loop-trained M5/M6 checkpoints
but was NOT true for this particular scalar-plant training run. arm_3
(GELU+GLU, memoryless, no LN) shows a similarly-shaped but far MILDER
dip/spike (~10-20% swing, not ~3x+) — consistent with Experiment 1's
finding that generic smooth nonlinearity produces mild, bounded
curvature, while LayerNorm's is categorically larger.

**A real bug was caught and fixed in this round, not just noted**: the
first version of the "homogeneity sweep" `||F(c*z0)||/c` plot showed a
dip/spike near `c=0` for every arm except arm_0/1 — including arm_4,
which is PROVABLY exactly linear (its own `Jx(c)` is flat to machine
precision). The shape was entirely the model's nonzero equilibrium
drift `F(0,0)` divided by a vanishing `c` (`F(c*z0)/c = J@z0 +
drift/c`, which diverges as `c→0` for any `drift != 0`), not a
curvature effect at all — the same class of drift-vs-derivative
confound the parent repo's CLAUDE.md documents under its bias-term-round
corrections. Fixed by subtracting `F(0,0)` before dividing
(`scalar_diagnostics.homogeneity_sweep`); re-verified arm_4 is now flat
in this plot too, matching its Jacobian.

**arm_5 vs. arm_6 vs. arm_7 — the actual prenorm/postnorm comparison,
and the most striking single result of this round**: arm_5 (S4+LN
prenorm, full model) mostly tracks the true `Jx=3` closely along the
real trajectory (mostly within ±5%) with a few sharp, NARROW, isolated
spikes (up to `Jx≈3.4`) at specific timesteps — a milder, more
localized version of arm_2's kink, since the real trajectory only
occasionally passes near the degenerate direction. arm_6 (S4+LN
POSTNORM, otherwise identical) is qualitatively different and far
worse: `|Jx(c)|` swings over **6 orders of magnitude** (`~1e-5` to
`~10`) across almost the ENTIRE tested range, not a narrow kink but
pervasive derivative corruption, and the real-trajectory `Jx` swings
wildly between `-1` and `8` (mean relative error `48%`, vs. arm_5's
`0.76%`) — despite arm_6's teacher-forced MSE (`4.3e-5`) being only
~4x worse than arm_5's (`1.0e-5`), NOT orders of magnitude worse. This
is the starkest illustration in this project so far of "low prediction
error does not guarantee Jacobian fidelity" — postnorm's placement
(forcing the ENTIRE skip+branch sum through LayerNorm, rather than
leaving an LN-free identity path via the skip connection) looks
categorically worse than prenorm here, not just "also kinked."
arm_7 (prenorm AND postnorm combined, the user-requested arm) is
similarly catastrophic to arm_6 (`jx_err_mean=35%`, spike ratio `1.4e6`)
— adding a second LN on top of prenorm does not rescue postnorm's
damage, and the combined arm's teacher MSE (`1.0e-3`) is actually the
single worst of the whole ladder, suggesting the two LNs interfere with
training rather than each contributing independently.

**Standing caveat, load-bearing**: every Experiment 2 number above is
from a SINGLE seed (seed=0) per arm. Given how much the exact kink
LOCATION already varied between Experiment 1's 6D checkpoints and this
round's scalar arm_2 (same underlying mechanism, different manifestation),
none of arm_2/5/6/7's specific numbers (spike ratio, error magnitude,
kink location) should be treated as a general property of the
architecture until replicated across multiple seeds — only the
qualitative ranking (arm_0/1/4 flat < arm_3 mild < arm_2/5 kinked <
arm_6/7 severely broken) is well-supported by the mechanism-level
argument (degree-0 vs. degree-1 homogeneity) and cross-checked against
Experiment 1's independent, 30-checkpoint-per-variant, 6D population.
