# NOTES

**Provenance (added 2026-08-28):** every number in this document, across
rounds 1, 1.5, and 2, was produced on LOCAL CPU on the development
laptop, not TACC - see `layernorm_study/CLAUDE.md` for the compute
policy this sub-project actually uses (a scoped, deliberate override of
the parent repo's TACC-by-default policy, not an oversight) and
`layernorm_study/requirements.txt` for the exact pinned environment.

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

**Standing caveat from round 1 (SUPERSEDED below, kept for the record
per this project's culture of dated corrections rather than silent
rewrites)**: round 1 above was single-seed (seed=0) per arm and treated
arm_2 as showing a real, if off-origin, kink. The multi-seed round below
shows this needs correcting, not just re-confirming with more data.

## Results, round 2: multi-seed (8 seeds x 8 arms), and a correction to arm_2 (2026-08-27)

Round 1's arm_0 positive control, re-run across 8 seeds at the SAME
20000 epochs, **failed at seed=6** (`jx_err_max=1.8e-3`, ju similarly)
— the script halted exactly as designed, per the task's own instruction.
Diagnosed directly, not assumed: re-training seed=6 alone at 60000
epochs converged to `jx_err_max=3.4e-9` (and 120000 epochs to
`1.4e-9`, no further improvement) — a pure convergence-budget issue
(some random inits need more Adam steps for this well-posed, effectively
convex 2-parameter fit), not a real failure of the arm_0 concept. Fixed
by raising `EPOCHS` to 60000 for the whole ladder and re-running all 64
(arm x seed) combinations clean — zero errors, all arm_0 seeds pass.

**The correction**: at 60000 epochs, arm_2's (LN + memoryless, round 1's
"KEY ARM") median `jx_err_mean` across 8 seeds is `2.7e-9` — in the SAME
near-machine-precision tier as arm_0 (`2.4e-9`) and arm_1 (`3.7e-12`),
not the `~10-20%` distortion round 1 reported at 20000 epochs. Verified
directly by re-plotting arm_2/seed0's origin sweep at 60000 epochs: the
dip-then-spike SHAPE is still visibly present at the exact same location
in `|Jx(c)|` vs. `c` (confirming the mechanism itself didn't vanish) —
but its AMPLITUDE collapsed from a ~3x swing to a ~1e-9 RELATIVE swing,
three orders of magnitude below anything practically meaningful. **The
corrected reading: for LayerNorm WITHOUT real S4 memory, on this
problem, the kink is real but "trainable away" — more optimization
steps let the network converge to a solution where the branch's
practical contribution (and therefore LayerNorm's influence) shrinks
toward negligible, even though the branch is never literally forced to
zero.** Round 1's characterization of arm_2 as "the key arm confirming a
persistent kink" is retracted; what round 1 actually measured was an
UNDER-TRAINED arm_2, not a fundamental property of LN-without-memory.

**What DOES replicate robustly across seeds (`results/exp2_ladder.csv`,
`results/exp2_ladder_seed_summary.csv`, `figures/exp2_seed_variance_summary.png`
— a box plot of `jx_err_mean` and `teacher_mse` per arm across all 8
seeds is the clearest single figure from this round):**

- arm_0, arm_1, arm_2 cluster tightly in a near-machine-precision tier
  (`jx_err_mean` medians `1e-9` to `1e-12`) — LayerNorm ALONE, without
  real S4 memory, is not a robust failure mode at this problem's scale
  once given enough training.
- arm_3 (GELU+GLU, memoryless) and arm_4 (real S4 memory, no LN) sit in
  a middle tier (`jx_err_mean` medians `2.1e-2` and `2.9e-3`) — neither
  converges to the machine-precision tier even at 60000 epochs, a
  genuine optimization-difficulty finding distinct from any LN-specific
  claim (their `teacher_mse` medians, `1.9e-5` and `3.1e-6`, are also
  4-5 orders of magnitude worse than arm_0/1/2's floor).
- arm_5 (S4 + LN prenorm, full model) is the round's most seed-SENSITIVE
  result: `jx_err_mean` ranges from `4.7e-3` to `4.0e-1` across 8 seeds
  (median `6.8e-2`) — sometimes nearly as good as arm_3/4, sometimes far
  worse. This is a real property of the architecture (LN interacting
  with genuine S4 memory), not noise to average away — it means a
  single training run cannot be trusted to characterize this arm's
  behavior, exactly the caveat round 1 flagged in advance.
- arm_6 (S4 + LN POSTNORM) and arm_7 (combined) are the round's most
  ROBUST finding: consistently catastrophic across every seed
  (`jx_err_mean` between `25%` and `77%` for arm_6, `32%` and `77%` for
  arm_7 — tight boxes at the top of the plot, not scattered), regardless
  of the 3x increase in training budget that rescued arm_2. Postnorm's
  failure is not a training-budget artifact and not seed-dependent — it
  looks like a structural property of forcing the ENTIRE skip+branch
  sum through LayerNorm.

**Revised bottom line for this sub-project's core question**: at this
problem's scale, LayerNorm's Jacobian-corrupting effect is not really
about LayerNorm in isolation (arm_2 trains away) — it is about
LayerNorm's INTERACTION with S4's real recurrent dynamics (arm_5,
seed-sensitive but real) and, most severely and robustly, about
POSTNORM PLACEMENT specifically (arm_6/7, robust and severe regardless
of seed or training budget). This sharpens rather than overturns the
motivating hypothesis (LN's degree-0 homogeneity), but relocates where
the real, hard-to-train-away damage lives: not "LN exists" but "LN
placed after the residual, or LN composed with genuine memory."

## Results, round 1.5: is "trainable away" actually branch suppression? (2026-08-28)

Motivated by a direct objection to round 2's arm_2 correction: LayerNorm
is homogeneous of degree ZERO by construction (no gamma/beta makes
`LN(c*z) = c*LN(z)`) - a prenorm block containing a real LayerNorm
CANNOT represent an exactly homogeneous map unless the branch's
contribution is driven toward zero. So "trainable away" (round 2)
should mean the optimizer found an escape route (branch suppression),
not that LN's nonlinearity itself became benign. Tested directly rather
than assumed.

**Task 1(a) - gamma/beta, init vs final, all 8 arm_2 seeds**: `gamma`
does NOT collapse toward zero. It stays close to its init norm
(`2.83`) throughout - final values range `2.84` to `3.29` (shrink
factors `0.86x`-`0.99x`, several actually GROW slightly). `beta` grows
from `0` to a modest `0.15`-`0.73`. **The specific "gamma -> 0" escape
route is refuted by this alone**, before even running the frozen-LN
test.

**Task 1(b,c) - branch/skip ratio and J = C + R decomposition, arm_2**:
the branch's OUTPUT magnitude relative to skip is genuinely small
(median ratio `0.7%`-`9.4%` across seeds), and the constant term `C`
(skip alone) already matches `[A_true, B_true]` almost exactly
(`C_err_rel` `0` to `8.4e-8`) - consistent with suppression, but not
yet distinguishing WHICH parameter is doing the suppressing.

**Task 1(d) - the decisive test: retrain arm_2 with gamma FROZEN at 1,
beta FROZEN at 0 (LN's normalization is still fully active every
forward pass - mean-subtraction, `1/sigma` scaling - the optimizer just
cannot shrink gamma or shift beta), same pinned 60000 epochs, same 8
seeds.** Result: **error does NOT jump back to percent-level.** 7 of 8
seeds stay in the `1e-10` to `1e-8` range (statistically indistinguishable
from the unfrozen run); the one exception (seed 6, `1.7e-3`) is still
three orders of magnitude below "percent-level," not a restoration of
round 1's original finding. **This refutes the specific "gamma->0"
hypothesis as stated.** Freezing LN's own affine parameters does not
stop whatever is suppressing the branch.

**Follow-up (not in the original task list, run because Task 1(d)'s
result demanded it): which knob IS doing the suppression, if not
gamma/beta?** Checked the S4 gain itself (arm_2 is memoryless, so its
zero-lag gain `h_0 = C_bar@B_bar` is the only other multiplicative
factor on the branch path) - it does NOT shrink either (norm `2.7`-`2.9`
at init, `2.7`-`3.5` at final, several seeds growing). Checked the
actual geometry instead: at a representative trajectory point (seed 0),
the PRE-decoder branch vector has substantial norm (`3.07`), `W_dec`
has a normal norm (`1.63`), but `cos(angle between them) = 0.15` - **the
two are nearly orthogonal**. Confirmed on a second seed (`5`): branch
norm `3.82`, `W_dec` norm `1.09`, `cos = 0.16`. **This directly confirms
the task's alternative hypothesis (e): `W_dec` learned a DIRECTION
that geometrically nearly-annihilates the branch's contribution, not
that any individual scalar (gamma, beta, or the S4 gain) collapsed.**
Precise statement of what round 2 actually found: LayerNorm's degree-0
term cannot become linear, so it becomes irrelevant via a *geometric*
route (the decoder projects it out), not a *magnitude* route on any
single learnable scale factor.

**Task 2 - does arm_5's seed sensitivity correlate with the same
phenomenon?** Computed the same branch/skip ratio (Task 1b's machinery,
generic, applied unchanged to arm_5's 8 existing checkpoints - no
retraining) and correlated it against arm_5's already-recorded
`jx_err_mean` per seed. **Pearson r = 0.973** (`results/
round1_5_arm5_branch_ratio_vs_error.csv`, `figures/
round1_5_arm5_branch_ratio_vs_error.png`) - a very strong, clean,
monotonic relationship. **Arm_5's seed sensitivity is not mysterious:
it is basin selection between "branch mostly suppressed" (low ratio,
low error, e.g. seed 0 at ratio `0.04`/error `0.5%`) and "branch stays
live" (high ratio, high error, e.g. seed 5 at ratio `1.04`/error
`40%`).** Exactly the mechanism Task 1 characterizes for arm_2, playing
out as literal seed-to-seed variance for arm_5 rather than being
uniformly resolved.

**Task 3 - is arm_0's slow convergence (round 2: 60000 epochs needed,
failed at 20000 on one seed for an exactly-affine 2-parameter fit) a
data-conditioning artifact?** `cond(E[z z^T])` for round 1/2's data
(`k_stab=-2.7`) is `156` - `corr(x, u) = -0.977`, confirming the
predicted near-rank-deficiency from proportional feedback
(`u = k_stab*x + a`, closed form `corr = k_stab/sqrt(k_stab^2 + 1 -
pole^2)`). A scan over the stabilizing range found APRBS amplitude
barely moves `cond` (the closed-form correlation doesn't depend on it)
but `k_stab` does: `k_stab=-3.5` (pole `-0.5`) empirically minimizes it
at `cond=82` - a real `47.6%` reduction, but not dramatic; there is an
apparent FLOOR to how decorrelated a purely-proportional stabilizing
loop can make `(x, u)` for this specific unstable plant without
changing the excitation paradigm entirely (out of scope this round).
Re-ran arm_0 and arm_5 on the decorrelated data at the SAME PINNED
60000 epochs (per this round's own methodological note - the budget was
not re-tuned again): **arm_0's median error barely moves** (`2.4e-9` ->
`1.1e-9`, both already at the machine-precision floor - conditioning
doesn't matter once the epoch budget is adequate). **arm_5's median
error does not improve** (`6.8e-2` -> `1.1e-1`, if anything slightly
worse) **and which SEEDS are good/bad completely reshuffles** (seed 1:
`2.6%` -> `23%`; seed 5: `40%` -> `1.9%`; seed 6: `9.2%` -> `0.6%`).
**The conclusions do not move: data conditioning is real (and now
documented/parameterized - `scalar_system.generate_scalar_trajectory`'s
new `k_stab` argument, `regressor_condition_number` helper) but is NOT
the driver of arm_5's seed sensitivity.** That remains best explained
by Task 2's basin-selection finding, which is architecture/optimization-
landscape-inherent, not data-inherent.

**Standing status (superseded below)**: Tasks 4 (harden the postnorm
ceiling claim) and 5 (epsilon sweep on arm_5) were queued but not run,
per instruction to stop and report Task 1(d)/Task 3 first. Round 2
below re-examines the round 1.5 orthogonality claim before proceeding
to Tasks 4/5.

## Round 2, Part A: null tests for the orthogonality claim (2026-08-28)

Direct objection to round 1.5's "W_dec learns a direction nearly
orthogonal to the branch (cos~0.15)": for two INDEPENDENT RANDOM
vectors in R^H, typical |cosine| ~ 1/sqrt(H) - at this study's
`d_model=8`, that baseline is `0.354`, well ABOVE the observed
`cos~0.15`. Near-orthogonality in high dimensions is the default, so a
below-baseline cosine needed an explicit null test, not a bare number.
Round 1.5's `cos=0.15` was also a single-point spot-check (one arbitrary
`t`), not a trajectory-level statistic - both issues are addressed
directly below (`results/round2_partA_arm2_orthogonality.csv`,
`layernorm_study/src/orthogonality_tests.py`).

**A1/A2 - cosine at init vs convergence, against an explicit null,
all 8 arm_2 seeds:**

| seed | cos init | pctl init | cos final | pctl final | proj. ratio init | proj. ratio final |
|---|---|---|---|---|---|---|
| 0 | -0.096 | 20.0 | -0.008 | **1.8** | 0.064 | 0.007 |
| 1 | 0.324 | 59.6 | 0.023 | **5.0** | 0.105 | 0.036 |
| 2 | -0.464 | 78.7 | 0.029 | **6.0** | 0.530 | 0.017 |
| 3 | 0.136 | 26.2 | 0.066 | 12.5 | 0.043 | 0.075 |
| 4 | -0.464 | 80.5 | -0.129 | 25.7 | 0.170 | 0.058 |
| 5 | -0.193 | 37.8 | 0.106 | 21.5 | 1.665 | 0.092 |
| 6 | 0.066 | 12.8 | 0.058 | 11.2 | 0.022 | 0.055 |
| 7 | 0.134 | 27.0 | 0.040 | 8.1 | 0.023 | 0.021 |

(`pctl` = percentile of `|cos(w, branch)|` in a 2000-sample null of
random unit directions against the SAME branch vector - a LOW
percentile means the trained/init decoder is MORE orthogonal to the
branch than random chance.)

At **init**, percentiles scatter across the full `12.8`-`80.5` range
with no consistent direction - exactly what pure random initialization
should produce, and a sanity check that the null-test methodology
itself is not biased. At **convergence**, all 8 seeds land at or below
the 26th percentile, 6 of 8 at or below the 13th, and 3 of 8 below the
7th. **This is not noise and not a generic property of high-dimensional
geometry** - training consistently and reliably moves `W_dec` toward a
direction MORE orthogonal to the branch than chance would predict,
across every seed. Round 1.5's `cos=0.15` number is superseded (a
single-point estimate, not wrong in kind but noisier and less
extreme than the trajectory-median values here, which run as low as
`-0.008`).

**A3 - projected contribution `|<w,branch>| / |<w,skip>|` (scale-
sensitive, unlike cosine) at init vs final**: does NOT shrink
monotonically in every seed (seeds 3, 6, 7 are flat or even grow
slightly from init) - this is flagged, not smoothed over, since it
complicates a simple "training monotonically suppresses the branch"
story. But the CONVERGED value is small in absolute terms for every
seed regardless of its own trajectory (`0.007`-`0.092`, all under
`10%`) - so what's robust across seeds is the FINAL state (small
projected contribution, low percentile), not necessarily a monotonic
shrinking path to get there.

**Note on gamma/beta, restated precisely per the task's own caution**:
freezing gamma/beta (round 1.5 Task 1(d)) could only ever have tested
whether LN's OWN affine parameters were the suppression mechanism. It
was never a test of the projection/W_dec hypothesis - if `W_dec` does
the annihilating, gamma is irrelevant to it by construction, so that
result is consistent with the projection story but was never evidence
FOR it on its own. A1-A3 above are the actual test of the projection
hypothesis, and they support it: `W_dec` reliably lands in the low tail
of the null distribution after training, and its converged projected
contribution is uniformly small.

**A4 - Task 2 robustness: Spearman alongside Pearson.** Pearson
`r=0.973` (unchanged from round 1.5); **Spearman rho=0.738 (p=0.037)** -
still positive and still significant at `n=8`, but MATERIALLY weaker
than the Pearson number suggested. The labeled scatter
(`figures/round2_partA_arm5_correlation_labeled.png`) shows why: seeds
3 and 5 (branch/skip ratio `0.65` and `1.04`) sit well outside the
cluster of the other 6 seeds (ratio `0.04`-`0.17`) and are doing much of
the work in the Pearson number - among the clustered 6, the relationship
is directionally consistent but not strictly monotonic (e.g. seed 1
has LOWER error than seed 4 despite a HIGHER ratio). **Revised
statement: the branch-ratio/Jacobian-error relationship for arm_5 is
real and significant, not an artifact of two leverage points - but it
is noisier than the Pearson r=0.973 alone implied, and should be
described as "strong and significant" rather than "near-deterministic."**

**A5 - Task 3, is conditioning ruled out?** `corr(x,u)` for the
original data is `-0.977`; for the `k_stab=-3.5` decorrelated data,
`-0.945` - barely moved, confirming round 1.5's own characterization
("factor of two [in cond], not an elimination"). Tried a genuinely
different decorrelation strategy per instruction: OPEN-LOOP excitation
over a short horizon (x and u independent BY CONSTRUCTION in open
loop, since u is pure APRBS). **Result: catastrophically worse, not
better** - because the plant is open-loop unstable (`rho=3`), even a
short horizon lets `|x|` grow enormously relative to `u`'s fixed
APRBS scale (length 8: `max|x|=717`, `cond=4.8e5`; length 20:
`max|x|=3.7e8`, `cond=7.8e16`), despite LOWER raw correlation
(`0.44`-`0.62` vs. the closed-loop's `0.94`-`0.98`) - condition number
depends on the whole covariance spectrum, not correlation alone, and
an unstable open-loop system creates a variance mismatch between `x`
and `u` that dominates. **Explicit answer to "is conditioning ruled
out": no, not rigorously - a `cond~1` dataset was never achieved by
either method tried, so conditioning-as-a-contributing-factor cannot
be logically excluded. But it is not demonstrated to be the ACTIONABLE
driver either: round 1.5's direct empirical test (retrain arm_0/arm_5
on the ~50%-better-conditioned k_stab=-3.5 data) showed no improvement
and a full reshuffling of which seeds succeed - the available evidence
says conditioning is not the lever that explains arm_5's seed
sensitivity, even though "conditioning matters zero" was never proven
in the strong mathematical sense.**

**Summary of Part A's effect on round 1.5's claims**: the branch-
suppression/orthogonality mechanism (Task 1) is CONFIRMED more rigorously
than before, not weakened - the null test was the right skeptical check
to run, and it came back supporting the original claim more strongly
(low percentiles are a cleaner signal than a bare cosine number could
ever be). Task 2's correlation survives but should be described more
modestly (Spearman, not just Pearson). Task 3's "conditioning doesn't
explain it" conclusion survives a genuine second attempt, with the
caveat that "ruled out" overstates what was actually shown.

## Round 2, Part A follow-ups: A6-A8 (2026-08-28)

**A6 - direction ambiguity: does W_dec rotate toward the branch, or
does the branch reorganize into W_dec's null space?** Measured angular
displacement of each from its OWN initialization
(`results/round2_partA_a6_a7_direction_and_signed_vs_abs.csv`). W_dec
moved more in 6 of 8 seeds (often by a wide margin - e.g. seed 5:
`75.8` deg vs `30.2` deg); the branch direction moved more in 2 of 8
(seeds 4 and 7, and seed 7's margin is a near-tie, `21.5` vs `20.7`
deg). **Not unanimous, but the dominant pattern (6/8, larger margins)
supports the original framing ("W_dec learns to project the branch
out") over the alternative ("the branch reorganizes into W_dec's null
space") - stated as a majority finding, not a universal one.**

**A7 - signed vs. unsigned cosine.** `abs_cos_median` equals
`signed_cos_median` to displayed precision for every one of the 8
seeds. This means the concern behind A7 (a sign-flipping cosine along
the trajectory hiding a larger true magnitude under a near-zero signed
median) did NOT occur here empirically - the branch's projection onto
`W_dec` keeps a consistent sign throughout the real trajectory for
every seed tested. Worth checking, and checked - the earlier signed
numbers were not an artifact.

**A8, part 1 - one more conditioning attempt (randomized feedback
gain).** Scanned `segment_length` (how often the closed-loop gain `k`
is resampled within the 100-step trajectory) from 2 to 100
(`results/round2_partA_a8_gain_randomization_scan.csv`). **Contradicts
the stated prediction**: gain randomization does NOT move conditioning
"much further than -0.977 -> -0.945" - every segment length tried lands
at `cond~76-95` (well-randomized, short segments) or WORSE (`cond` up
to `383` at long segments, which approach single-fixed-k behavior).
There is an apparent hard floor around `cond~76-80` for this specific
plant that neither fixed-`k` selection (round 1.5) nor randomization
(this round) beats - because `B_true=1` gives strong control authority,
keeping `u` tightly tied to `x` regardless of which stabilizing `k` is
used. A bonus attempt applying the same method to the user's example
plant (`A=1.03, B=0.01`, the system Part C will use) hit a clear
numerical pathology (`cond=1.6e14`, `corr=1.0000` exactly) - flagged as
NOT a meaningful measurement, not silently reported as a result; that
plant's weak control authority needs its own dedicated excitation
tuning, done properly when Part C sets it up rather than reusing this
plant's defaults opportunistically.

**A8, part 2 - does arm_5's variance track conditioning level? THE
RESULT IS REAL BUT CONFOUNDED, AND THAT CONFOUND IS REPORTED RATHER
THAN HIDDEN.** Since the originally-requested targets (`cond~5, 20, 80,
156`) were not achievable (part 1's floor), used the actual achieved
spread instead: `segment_length` in `{6, 10, 20, 50}`, giving `cond`
approximately `{76, 82, 130, 318}`. Retrained arm_5 (8 seeds) at each
level (`results/round2_partA_a8_arm5_multilevel_results.csv`,
`figures/round2_partA_a8_arm5_variance_vs_cond.png`):

| cond | median jx_err_mean | std | min | max |
|---|---|---|---|---|
| 76.4 | 0.0065 | 0.082 | 0.0007 | 0.197 |
| 82.1 | 0.0076 | 0.060 | 0.0004 | 0.175 |
| 130.3 | 0.0964 | 0.312 | 0.0004 | 0.913 |
| 317.7 | 0.1844 | 0.169 | 0.0004 | 0.424 |

This IS a clear, visible, monotonic-in-median trend - the two
low-`cond` levels have median error `<1%`; the two high-`cond` levels
have median error `10-18%`, both wider spread AND worse minimums are
NOT what moved (the min stays `~0.0004`-`0.0007` at every level - what
changed is the ceiling). **This looks, at first read, like exactly the
"conditioning drives arm_5's variance" result the prediction hoped for
- but it directly CONTRADICTS round 1.5's own direct test (fixed
`k=-2.7` vs. fixed `k=-3.5`, which found NO improvement and full seed
reshuffling), and the reason for the contradiction is a confound in
THIS experiment's own design, not a real reversal of round 1.5's
finding.** `segment_length` controls two things at once: it changes
`cond` (as intended), but it ALSO changes how often the effective
closed-loop relationship switches within one trajectory - a SHORT
segment_length means the model sees many different local (x,u)
relationships packed into one 100-step trajectory, which is a
"persistence of excitation" effect (a classical, distinct concept in
system identification: does the trajectory excite enough different
regimes for identification to be well-posed), not the same thing as the
marginal correlation/condition-number of the pooled (x,u) pairs. This
round's 4-level sweep varies `segment_length`, and `cond` merely
happens to move WITH it - the experiment as designed cannot attribute
the observed trend to conditioning specifically rather than excitation
richness. **Honest statement: there is a real, strong relationship
between `segment_length` (short vs. long gain-switching) and arm_5's
error ceiling, but this round's design conflates that with raw
regressor conditioning, and cannot separate the two. Round 1.5's
direct, unconfounded test (same segment structure - a single fixed `k`
throughout - varying only which `k`) remains the cleaner evidence, and
it found no improvement. A properly separated follow-up (hold
`segment_length` fixed, vary `cond` some OTHER way, or vice versa)
would be needed to settle which variable is doing the work - not run
this round.**

## Round 2, A9/A10 (2026-08-28)

**A9 - the cond floor is structural, decoupling `u` from `x` fixes it.**
The closed-form `cond ~ (1+k^2) var(x)/var(a)` predicts the correct
TREND across `k` (correlates with which `k` gives lower/higher `cond`
in round 2's own A8 scan) but is off by a roughly constant factor
(`~7`-`14x`, not exactly constant) - the qualitative structural claim
holds, the exact formula is an approximation
(`results/round2_A9_our_plant_reset_scan.csv`).

Short-horizon open-loop-with-resets (draw `x_0` fresh from a wide
range every `reset_every` steps, inject pure APRBS `u` - independent
of `x` by construction) works dramatically better than any feedback-
gain approach: for this study's plant (`rho=3`), `reset_every=2` gives
`cond=3.1`, `reset_every=3` gives `cond=10.1` (`max|x|=54`, a
reasonable scale) - both far below the `~76`-`80` floor round 2's A8
found for every closed-loop scheme tried. Growth is `rho^reset_every`,
so this only works because the window is short (`reset_every=5` already
gives `cond=467`, `max|x|=490`; `reset_every=8` is unusable,
`cond=2.2e5`).

For the `A=1.03, B=0.01` plant Part C will use: the earlier `cond=1.6e14`
blowup is confirmed to be exactly the predicted artifact, not real
ill-conditioning. Under short-horizon resets (`reset_every=20`, per the
task's own `1.03^20~1.8` growth-bound argument) with the SAME raw
excitation range reused from the other plant, `cond_raw=11.2` already -
much better than the long-horizon closed-loop attempt. Reporting the
SCALE-INVARIANT (standardized/correlation-matrix) condition number
alongside it, per the task's own instruction that a bare `cond` number
is meaningless without stating the unit convention:
`cond_standardized=1.38` - close to the `1.0` floor, confirming the
plant itself is nearly perfectly identifiable once `B`'s tiny raw scale
stops dominating the Gram matrix. **Both plants: the fix works, and the
mechanism (structural floor from closed-loop `u in span(x)`, not a
tuning failure) is confirmed.**

**A10 - the unconfounded conditioning test, and a genuine structural
limit found along the way.** Attempted exactly as specified: ONE fixed
dataset (round 1's `k_stab=-2.7`, `cond=156`, the SAME 100 real points
in their ORIGINAL temporal order/PE structure throughout), reweighted
via a per-timestep loss weight (`arms.train_arm_weighted`, new) to hit
different empirical `cond` targets. **Before the retraining sweep, an
unconstrained numerical search for minimum-cond weights found a real,
structural limit, not a search failure**: the lowest achievable `cond`
via reweighting this specific dataset is `~6.8`, but ONLY by collapsing
effective sample size (`1/sum(w^2)`) to `~1.1` - i.e. by putting nearly
all loss-weight on a single timestep. This is itself a confound (a
dataset that is EFFECTIVELY 1-2 points carries far less identification
signal than one that is effectively 100, independent of its raw
conditioning), so it was not used for the retraining sweep. Instead,
traced the full achievable `(cond, eff_n)` trade-off curve under an
`eff_n` floor constraint (`results/round2_A10_conditioning_levels.csv`):

| level | cond | eff_n |
|---|---|---|
| eff_n>=15 | 41.0 | 15.0 |
| eff_n>=30 | 65.6 | 30.0 |
| eff_n>=50 | 85.3 | 50.0 |
| eff_n>=80 | 107.8 | 80.0 |
| native (uniform) | 156.0 | 100.0 |

This is a narrower achievable range (`41`-`156`, `~3.8x`) than the
originally-hoped `5`-`300` (`60x`), and `cond` and `eff_n` remain
coupled throughout it - a genuinely cleaner test than A8 (no
excitation-scheme/persistence-of-excitation confound: every level uses
the identical 100 real data points in the identical order, differing
ONLY in loss-weighting) but not a perfectly isolated one, since `eff_n`
could not be held exactly fixed while cond varies for this dataset.
Retrained arm_5 (8 seeds) at each of the 5 levels
(`results/round2_A10_arm5_reweighted_results.csv`,
`figures/round2_A10_arm5_cond_vs_effn.png`):

| level | cond | eff_n | median err | std err | min | max |
|---|---|---|---|---|---|---|
| eff_n>=15 | 41.0 | 15.0 | 0.0187 | 0.080 | 0.0042 | 0.228 |
| eff_n>=30 | 65.6 | 30.0 | 0.0587 | 0.150 | 0.0013 | 0.413 |
| eff_n>=50 | 85.3 | 50.0 | 0.0417 | 0.042 | 0.0024 | 0.124 |
| eff_n>=80 | 107.8 | 80.0 | 0.0566 | 0.102 | 0.0013 | 0.294 |
| native | 156.0 | 100.0 | 0.0681 | 0.149 | 0.0047 | 0.403 |

**This directly contradicts A8's apparent clean monotonic trend, and
is reported as such rather than reconciled away.** The median is NOT
monotonic in `cond` (`65.6->85.3` DECREASES, `0.059->0.042`, before
rising again) and the figure (both panels - vs. `cond` and vs. `eff_n`
separately) shows no visible trend at all: every level's 8 seeds
scatter across roughly the SAME `0.001`-`0.4` range regardless of
which level they belong to. Within-level standard deviation is
frequently LARGER than the between-level median differences (e.g.
`eff_n>=30`'s own std, `0.150`, exceeds every other level's median).
**Conclusion: with the segment_length confound removed and the
achievable range honestly narrower than hoped, arm_5's seed-to-seed
error spread is FLAT within noise across the `cond` range this method
could reach (`41`-`156`) - conditioning is not the driver of arm_5's
variance. This is now supported by evidence within a properly
(if imperfectly) controlled design, not by elimination or by round
1.5's single unconfounded data point alone.** The dominant driver
remains round 2's Task 2 finding: basin selection between branch-
suppressed and branch-live solutions (Pearson `r=0.973`, Spearman
`rho=0.738`), independent of data conditioning.

**A6 restated per instruction, majority not universal**: W_dec's own
angular displacement from init exceeded the branch direction's in 6 of
8 arm_2 seeds (not all 8) - the dominant pattern, not a unanimous one.

## Round 2, A11/A12 (2026-08-28)

**A12 - tightening the closed form.** The stated formula, `cond ~
(1+k^2) var(x)/var(a)`, was off by a "roughly constant" `7`-`14x` factor
- re-derived properly rather than accepting that as final. The exact
2x2 eigenvalue product/sum (`det=lambda_max*lambda_min`,
`trace=lambda_max+lambda_min`) gives, in the small-`var(a)` asymptotic
regime, `lambda_max ~ Mxx(1+k^2)` and `lambda_min ~ Maa/(1+k^2)` (not
`Maa` alone) - an EXTRA factor of `(1+k^2)` was missing. **The
corrected formula is `cond ~ (1+k^2)^2 * var(x)/var(a)`,** confirmed
directly (`results/round2_A12_k_scan.csv`,
`results/round2_A12_seed_scan.csv`, `results/round2_A12_summary.txt`):
the residual ratio (actual/predicted) collapses from `5.15`-`16.49x`
(original formula, systematic and large, mean `10.50`) to
`0.80`-`1.32x` (corrected formula, mean `1.05`) across a `k` scan from
`-3.99` to `-2.01`. Per the task's own instruction to stop calling any
residual "constant" without checking: the corrected formula's residual
DOES still correlate with `k` (Pearson `r=0.759`, `p=1.4e-8`) - a real,
if much smaller, remaining `k`-dependence, most likely a next-order
term in the asymptotic expansion (not derived further - diminishing
returns for this study's purposes). Separately, and importantly for
interpreting that residual: holding `k` FIXED at `-2.7` and varying
only the random seed gives residual values ranging `0.91`-`2.50`
(`std=0.47`) - LARGER than the across-`k` scan's own spread
(`std=0.18`) - meaning a substantial part of what looked like
"unexplained k-dependence" is actually ordinary finite-sample noise
(only 100 timesteps per trajectory), not a further systematic
correction waiting to be found. **Both effects are now separated and
quantified: a real but small residual k-dependence (`r=0.759`) plus a
comparable-or-larger finite-sample noise floor - neither hidden inside
a vague "roughly constant" label.**

**A11 - closing the conditioning range, and it changes the
conclusion.** A10's sweep only reached `cond=41` (via reweighting, at
the cost of `eff_n=15`), so its "flat variance" finding spanned `~4x`
in `cond`, not the `~30x` the test needed. Fix, exactly as specified:
use A9's short-horizon open-loop-with-resets data (`reset_every=3`)
DIRECTLY as arm_5's training set - `cond=10.1` at FULL `eff_n=100`,
persistence-of-excitation held fixed by construction (all 100 real
points used, no reweighting at all), not by argument. Retrained arm_5
(8 seeds) on it and added it to A10's combined sweep
(`results/round2_A11_combined_results.csv`,
`figures/round2_A11_arm5_full_range.png`):

| level | cond | eff_n | median err | std err | min | max |
|---|---|---|---|---|---|---|
| reset_based (A11) | 10.1 | 100.0 | **0.00047** | 0.0085 | 0.00005 | 0.024 |
| eff_n>=15 (A10) | 41.0 | 15.0 | 0.0187 | 0.080 | 0.0042 | 0.228 |
| eff_n>=30 (A10) | 65.6 | 30.0 | 0.0587 | 0.150 | 0.0013 | 0.413 |
| eff_n>=50 (A10) | 85.3 | 50.0 | 0.0417 | 0.042 | 0.0024 | 0.124 |
| eff_n>=80 (A10) | 107.8 | 80.0 | 0.0566 | 0.102 | 0.0013 | 0.294 |
| native (A10) | 156.0 | 100.0 | 0.0681 | 0.149 | 0.0047 | 0.403 |

**The spread DOES open up at `cond~10`, exactly the alternative
outcome flagged in advance.** Median error at `cond=10` is `~40x`
better than at `cond=41` and `~145x` better than native; the figure
shows the `cond=10` cluster sitting entirely below every other level,
not overlapping the noise band the way A10's five closed-loop-derived
levels overlapped each other. **This is a genuine change of
conclusion from A10/round 1.5, and is retracted rather than
reconciled: "conditioning is not the driver of arm_5's variance,
closed" (A10's stated conclusion) does NOT survive extending the range
to true low conditioning.**

**However - and this must be reported with the same rigor as the
result itself - the `cond=10` anchor is NOT produced by the same
mechanism as A10's other five levels, and this reintroduces a version
of A8's original confound rather than eliminating it.** A10's five
closed-loop-derived levels are all reweightings of ONE fixed dataset
(same 100 real closed-loop-plus-dither points, same temporal order,
differing only in per-timestep loss weight). A11's `cond=10` anchor is
a COMPLETELY DIFFERENT dataset, generated by a different excitation
SCHEME (open-loop, `x_0` redrawn independently every 3 steps - roughly
33 independent state-space draws packed into 100 steps, versus one
single continuous closed-loop trajectory for every other level). This
is, structurally, the same category of confound as `segment_length` in
A8: reaching low `cond` via any method found so far in this study
requires changing HOW the excitation is generated, and every such
change also changes how much of the state space gets visited (a
persistence-of-excitation effect), not conditioning in isolation.
**Honest statement, not resolved further this round: extending the
range reveals a large, real effect that A10 alone could not see
(a genuine result, not an artifact of A10's narrower range) - but
whether that effect is attributable to LOW CONDITIONING specifically,
to the RICHER STATE-SPACE COVERAGE that came bundled with the only
method found to reach it, or to both, remains open.** A clean
follow-up would need low `cond` achieved WITHOUT a scheme change (not
found this round - A10's own numerical search shows this dataset's
reweighting floor is `cond~41` at usable `eff_n`) or richer coverage
achieved WITHOUT lower `cond` (not attempted this round). The
practical bottom line for the rest of this study: **conditioning (or
whatever is bundled with reaching it) is a REAL, first-order effect on
arm_5's error, at least at the low end - it should not be treated as a
solved non-factor going into Parts B/C, even though within any single
FIXED excitation scheme (A10's reweighting-only comparisons, or A8's
segment-length-only comparisons among short segments) its effect looks
much smaller or flat.**

## Part B: Task 4 / Task 5 (2026-08-28)

**Task 4 - postnorm output ceiling, original plant (rho=3), existing
60000-epoch arm_6/arm_7 checkpoints, all 8 seeds each, no retraining.**
`Y_max = ||W_dec||(||gamma||_inf sqrt(H) + ||beta||) + ||b_dec||`
computed from trained weights; swept `||F(c*z0)||` for `c` up to `1e4`
along a fixed direction. **PASS (bound never exceeded)**: every one of
16 checkpoints stays below its own `Y_max` (ratio range `0.05`-`0.66`,
median `~0.37`-`0.40`). Not tight for a single fixed direction - a
broader search (5000 random directions x 5 large scales, one
representative checkpoint) reaches `0.74`, confirming the gap is "a
single ray isn't the worst-case direction," not a loose bound.

**Task 5 - epsilon sweep on arm_5 (prenorm, original plant), eps in
{1e-8,...,1e-1}, 8 seeds each, 60000 epochs, new
`BlockConfig.layer_norm_eps` flag.** **FAIL.** Predicted: kink
amplitude/width scale as `eps^0.5`. Measured (log-log fit across the 5
eps levels): amplitude `~ eps^-0.066` (`r^2=0.58`), width
`~ eps^0.030` (`r^2=0.19`) - both essentially FLAT, nowhere near the
predicted `+-0.5` exponent, and the width fit is barely better than
noise (`r^2=0.19`). **Not fitted to look better than it is**: this is
a real, clean failure of the simple `sqrt(eps)` picture for arm_5 as
tested. Plausible (untested this round) explanation: arm_5 has REAL S4
memory plus GELU+GLU on top of LayerNorm, and the origin-sweep shape
this diagnostic reads off likely reflects those nonlinearities/memory
at least as much as LayerNorm's own epsilon floor - a cleaner test
would isolate LN the way arm_2 (memoryless, LN-only) does, rather than
running it on the full arm_5 architecture where multiple mechanisms
are superimposed. Flagged as the natural follow-up, not run this
round.

## Part C: Postnorm boundedness (2026-08-28)

### Derivation sketch (for interpreting the numbers below)

`P = I - (1/H) 1 1^T` (centering matrix). LayerNorm:
`zhat = Pv / sqrt(||Pv||^2/H + eps)`, so `||zhat|| <= sqrt(H)` always
(equality at `eps=0`) - LayerNorm maps all of `R^H` into a compact
ball. A POSTNORM block - LN the LAST operation before the decoder, no
skip bypassing it - therefore has uniformly bounded output:
`||F(z)|| <= ||W_dec||(||gamma||_inf sqrt(H) + ||beta||) + ||b_dec|| =: Y_max`
(the bound uses `||gamma||_inf`, not `||gamma||_2` - the worst case
over the unit ball puts all of `zhat`'s norm on the single largest-
`|gamma_i|` component).

LayerNorm's own Jacobian: `dLN/dv = (1/sigma) diag(gamma)[P - zhat zhat^T/H]`,
so `||J|| ~ C/||z||` far from the origin, and the bracket structurally
annihilates `zhat` (and the all-ones direction), making `J` exactly
rank-deficient by at least 2, at every point, not just far away.

Writing the pre-norm activation `v(z) = b + Mz` (b = v(0), M = dv/dz|_0,
both from the ACTUAL nonlinear branch, autodiff'd - not assumed
literally affine): near field (`||PMz|| << ||Pb||`) gives `sigma~const`,
`J~const`; far field (`||PMz|| >> ||Pb||`) gives `sigma~||z||`,
`J` decays. Crossover: `r* ~ ||Pb|| / sigma_max(PM)`, floored at
`sqrt(eps*H)/sigma_max(PM)`.

**Architectural caveat, verified directly against `s4dpc/blocks.py`
before relying on any of this** (not assumed): `skip = x` is set from
the block's raw input at the top of `ConfigurableBlock.__call__`; for
arm_6 (`prenorm=False`), the function's return value is
`self.norm(skip + branch)` with nothing else in between; for arm_7
(`postnorm_also=True`), it is `self.norm_post(skip + branch)`. In both
cases `StackedModel.__call__` feeds that return value directly into
`self.decoder` - no path from `skip` (or from `x` at any earlier point)
to the decoder bypasses the final norm call. **The theorem's
precondition holds exactly for both arms tested, confirmed by reading
the code.** This does NOT generalize to postnorm variants elsewhere
that place the norm inside the residual branch rather than around the
whole residual sum - that architecture would not satisfy this
precondition, and the boundedness argument would not apply to it
without separate verification.

All of C1-C8 below use `plant2` (`x_next = 1.03 x + 0.01 u`), excited
via A9's short-horizon open-loop-with-resets scheme (`reset_every=20`
unless noted), NOT the closed-loop scheme - per this round's own
instruction not to carry that scheme's conditioning artifact into
these tests.

### C1 - bias ablation (decisive). PASS.

**Framing correction per direct instruction**: the `r* ~ ||Pb||^0.93`
scaling law is directionally right but `r^2=0.31` means it explains
under a third of the variance - `p=0.0002` only says the slope is
non-zero, not that the fit is tight. **The decisive result is the
475x collapse under bias ablation, not the scaling exponent - reporting
it that way, not as a clean power law.**

- **C1(a) with-bias baseline** (`results/round2_C1a_withbias_baseline.csv`):
  median measured `r* = 0.375` - a real, if narrow, near-field region.
- **C1(b) no-bias ablation** (all biases AND beta frozen at exactly
  zero before training - `arms.train_arm_no_bias`,
  `results/round2_C1b_nobias_ablation.csv`): median measured
  `r* = 0.00079` - **a 475x collapse**. The near-field region does not
  shrink, it effectively vanishes: the amplitude simultaneously EXPLODES
  (median `2.80 -> 1652.6`), exactly as predicted for a map whose
  denominator now saturates at `sqrt(eps)` immediately at the origin
  with nothing to hold it away.
- **C1(c) bias-scale converse** (post-hoc scaling of the TRAINED
  encoder bias by `k in {0.25,0.5,1,2,4}`, no retraining,
  `results/round2_C1c_bias_scale_converse.csv`): `r* ~ ||Pb||^0.93`,
  noisy (`r^2=0.31`) but the right sign and roughly the right order.

### C2 - output ceiling (plant2, free-run rollout). PASS.

`results/round2_C2_output_ceiling.csv`, `figures/round2_C2_output_ceiling.png`.
Median `plateau/Y_max = 0.316` across 8 seeds - the free-run rollout's
own plateau never exceeds the predicted ceiling (consistent with
Task 4), though (same pattern as Task 4) a single rollout trajectory
doesn't reach the theoretical worst-case ceiling exactly.

### C3 - decay slope. FAIL as stated, with a plausible explanation not yet tested.

`results/round2_C3_decay_slope.csv`. Predicted: log-log slope of `||J||`
vs `||z||` in the far field is `-1`. **Measured median slope: `-2.33`,
95% CI (across 8 seeds) `[-2.79, -1.76]` - excludes `-1` entirely. This
is a real discrepancy, not noise, and is reported as a failure of the
exact prediction, not smoothed into "roughly -1."** Plausible,
untested explanation: the `-1` derivation considers LayerNorm's own
`1/sigma` scaling in isolation, but arm_6's branch ALSO has GELU and a
GLU gate, both of which have their own saturating behavior for large
inputs and would compound with LN's decay to give something steeper.
The natural test - rerun C3 on a GELU/GLU-free postnorm arm (e.g.
`norm="layer", activation="none", glu=False`, postnorm) to see if
`-1` holds when LN is the only nonlinearity present - was not run
this round.

### C4 - rank deficiency / Euler identity. PASS on Euler; AMBIGUOUS on null-alignment framing.

`results/round2_C4_rank_deficiency.csv`. Computed on the block's OWN
`H x H` LayerNorm Jacobian (`dLN/dv`, via `postnorm_geometry.py`) - NOT
the overall 1x2 input-output Jacobian, which has only one singular
value and no rank-deficiency structure to speak of.

- **Euler identity**: `||J(v)v|| / (||J(v)|| ||v||)` is `~3e-7` at
  `c=1` and `~4e-13` at `c=1000` - essentially exactly zero at BOTH
  radii. **PASS**, cleanly.
- **Null-direction alignment**: measured `~0.28`-`0.31` (median across
  seeds, both radii) - close to OR BELOW the random baseline for
  `R^8` (`1/sqrt(8)=0.354`), i.e. this specific check does NOT show
  the predicted alignment. **Flagged as a measurement-definition
  problem, not a clean fail of the underlying claim**: LayerNorm's
  Jacobian has TWO theoretically-exact zero directions (the all-ones
  direction from centering, and the `zhat` direction from the
  self-projection term), and `sigma_min` is already at `1e-17` to
  `1e-20` - numerically indistinguishable from a second near-zero
  singular value, so which one SVD returns as "the" smallest is not
  reliably resolved. The correct test compares `vhat` against the 2D
  SPAN of both near-null directions, not one arbitrarily-chosen one;
  not re-run this round.
- **"sigma_min collapses before sigma_max"**: does not quite apply as
  framed. `sigma_min` is ALREADY at machine-zero at `c=1` (near field),
  not something that progressively collapses as radius grows - this
  is a structural property of LayerNorm's Jacobian at EVERY point
  (exact rank deficiency by construction), not a distinctively
  far-field signature. `sigma_max` itself does decay with radius
  (`~1.2-1.7` at `c=1` down to `~0.001` at `c=1000`, consistent with
  C3), but the "before" framing in the prediction doesn't match what
  was measured.

### C5 - two-shell test. Genuinely surprising: neither arm reaches the predicted floor.

`results/round2_C5_two_shell_results.csv`, `figures/round2_C5_two_shell.png`.
Theoretical worst-case floor (splitting the difference between targets
at `r1=1` and `r2=50`): `25.235`. **Neither arm comes close**: arm_5
(prenorm control) RMSE `~0.0002`-`0.04` at both shells; arm_6
(postnorm) RMSE `~0.004`-`0.08` - `2`-`3` orders of magnitude BELOW
the predicted floor for both. Postnorm is consistently worse than
prenorm here (roughly `10`-`40x` on `rmse_r2`), so the qualitative
prenorm-vs-postnorm ranking survives, but **the specific "network is
provably at its best and its best is provably bad" claim does NOT
land** - postnorm was never forced anywhere near the predicted
compromise. Most likely explanation, not directly verified this round:
the floor formula assumes the model is FORCED to use a single constant
gain across `[r1, r2]`, which is only true if the model's own `r*`
(near-field radius) is SMALLER than `r2=50` - if these particular
two-shell-trained checkpoints happen to have `r* > 50`, both shells sit
inside the same near-field ball and there is no forced compromise to
hit the floor on. `predicted_r_star` was not computed for these
specific checkpoints this round - the natural, missing diagnostic that
would confirm or refute this explanation, flagged as an open follow-up.

### C6 - WITHDRAWN by the author of the hypothesis, not counted against the theory.

The original C6 prediction ("postnorm relocates its correct region to
the training center by adapting the bias") was wrong on the theory's
OWN terms, not merely inconvenient. The near-field region
`||PMz| << ||Pb||` is a ball CENTERED AT `z=0`, always - increasing the
bias enlarges that ball, it never moves it to an annulus elsewhere.
Getting good behavior at `||z||=50` needs `||Pb|| >> 50 sigma_max(PM)`,
which gives a ball that ALSO contains the origin. **This entry is
recorded as WITHDRAWN, not as a failed prediction of the boundedness
theory** - the theory was never actually tested by this prediction,
because the prediction did not follow from it. (The measured numbers -
`Jx=4.56` at the origin, `Jx=0.035` at `x=50`, no plateau anywhere in
the swept range - stand as raw data and are picked up by C6-revised
below, not discarded.)

### C6-revised. Parts (a-d) AMBIGUOUS-leaning-FAIL for simple saturation; part (e) PASS on scaling, FAIL on the origin-containment companion claim.

`results/round2_C6revised_ceiling_check.csv`,
`figures/round2_C6revised_ceiling.png`.

**(a-d) Is the original C6 finding just output saturation?** Required
output at `x_ref=50`: `~51.5`. Measured `Y_max` (trained weights,
8 seeds): `115.9`-`146.0`, median ratio `Y_max/required = 2.34` -
**consistently, across all 8 seeds, more than DOUBLE what's needed.**
`max|pred|/Y_max` median `0.84` (predictions get moderately close to
but do not fully saturate the ceiling). **Conclusion: simple output
saturation does NOT explain the original C6 numbers either** - the
model has ample headroom to represent the required magnitude, so
`Jx=0.035` at `x=50` isn't "the model literally can't reach 51.5," it's
something more specific: DERIVATIVE fidelity (needing the RIGHT LOCAL
SLOPE) fails independently of whether the OUTPUT AMPLITUDE itself is
capped - postnorm can hit approximately the right VALUE at a distant
point via curve-fitting flexibility (a non-locally-linear function
shape) without that implying anything about the SLOPE being right
there. This is a sharper, third reading of the original finding -
neither "relocation" (withdrawn) nor simple "saturation" (this section),
but a genuine value/derivative dissociation.

**(e) Enlargement test** (`results/round2_C6revised_enlargement_test.csv`,
`figures/round2_C6revised_enlargement.png`) - origin-centered data,
`b_enc` scaled by `k in {1,10,50,200}` post-hoc on trained arm_6
checkpoints:
- `r*` scaling: median log-log slope `0.913` vs `k` - **close to the
  predicted `1.0`, PASS.**
- Companion claim ("the good region always contains the origin,
  Jx-at-origin should stay near true as `k` grows"): **FAILS as
  tested.** Median `Jx_at_origin` across all `(seed, k)` combinations
  is `0.121` (true `1.03`), and it gets WORSE, not better, as `k`
  increases (e.g. seed 7: `Jx_at_origin` goes `-1.00 -> 0.047 ->
  0.005 -> -0.002` as `k` goes `1 -> 10 -> 50 -> 200`). **Flagged as
  AMBIGUOUS rather than a clean refutation**: scaling `b_enc` alone by
  up to `200x`, post-hoc and without retraining, pushes `v(0)` far
  outside the S4+GELU+GLU branch's TRAINED operating range - the
  branch was never exposed to an input of that magnitude during
  training, so nothing guarantees its response there is well-behaved,
  independent of whether the theory's own `r*` formula (which DOES
  recompute `M` and `b` correctly at each new bias level) is right.
  This complicates POST-HOC bias scaling as a clean test of the
  theory specifically - a genuine retraining at each bias scale
  (not attempted this round) would be the clean version of this test.

### C7 - which part of LayerNorm does the damage. PASS, cleanly.

`results/round2_C7_which_part_results.csv`, `figures/round2_C7_which_part.png`.
Median `jx_err_mean` across 8 seeds each:

| arm | jx_err_mean |
|---|---|
| real LayerNorm (arm_6 baseline, computed separately for this comparison) | 0.492 |
| rmsnorm (drops centering, keeps 1/sigma) | 0.530 |
| frozen_sigma (keeps centering, drops 1/sigma) | 0.057 |
| centering_only (identical construction to frozen_sigma, separately labeled) | 0.057 |

**rmsnorm fails within `8%` of real LayerNorm's own error** (`0.530`
vs `0.492`) - "fails IDENTICALLY" is confirmed closely, not just in
kind. **frozen_sigma and centering_only both recover to `~1/9` of that
error** (`0.057`), an order of magnitude improvement, from breaking
degree-0 homogeneity alone (a fixed denominator) with NOTHING else
changed. **The 1/sigma prefactor is the culprit; mean-centering is
exonerated** - both predictions land, and land together (they were
designed to be redundant checks from opposite directions: killing
degree-0 fixes it, and a degree-0-preserving variant with centering
removed fails just as badly).

### C8 - gain sweep. AMBIGUOUS - likely confounded by inconsistent training convergence, not a clean test.

`results/round2_C8_gain_sweep_results.csv`, `figures/round2_C8_gain_sweep.png`.
**Every one of the 5 gains tested (`rho in {0.5, 0.9, 1.03, 1.5, 3.0}`)
shows median failure radius `= 1e-6`** - the smallest value tested, an
IMMEDIATE failure with zero discrimination between stable and unstable
systems. This directly contradicts C1's OWN finding of a real,
non-trivial near-field region (`r*~0.375`) for a similarly-configured
arm_6 checkpoint, which is the internal-consistency flag that matters
here: **this looks like a setup/convergence problem, not a real
"failure radius shrinks monotonically" or "flat" finding.** Supporting
evidence: `train_mse` values are inconsistent and often large (many in
the `1e-2` to `0.8` range, versus the `1e-6`-`1e-8` this project
otherwise treats as "well converged") - the per-`rho` excitation
scheme (`B_true` fixed at `1`, `reset_every` chosen ad hoc per `rho`
rather than validated the way plant2's own `(A=1.03, B=0.01,
reset_every=20)` configuration was validated in A9) was not checked
for training stability before being used to draw conclusions. **Not
reported as a PASS or FAIL of the gain-sweep prediction - reported as
AMBIGUOUS, needing a redo with per-`rho` excitation validated for
convergence quality (matching A9's own rigor) before it says anything
trustworthy.**

### Bottom line

The user's own stated bar for the strongest, most general claim was
"if C1 and C5 both land." **C1 lands cleanly (475x collapse under
bias ablation, confirmed from two independent directions via C7's
frozen-sigma/centering-only recovery). C5 does not land** - neither
arm reached the predicted floor, most likely because these particular
checkpoints' near-field radius already covers both test shells, so
postnorm was never actually forced into the predicted compromise.
**The generalized claim ("postnorm residual blocks are uniformly
bounded-output maps that can only fit a linear system on a bounded
neighborhood of a bias-determined operating point, and can never
reproduce an unstable one") is therefore NOT written up as established
past S4 here** - the mechanism (bounded output, near-field-only
Jacobian fidelity, degree-0 homogeneity traceable specifically to
LayerNorm's `1/sigma` term) is well-supported by C1, C2, C4's Euler
identity, and C7; but C3's exact decay-rate prediction fails as
stated, C5's clean impossibility demonstration doesn't materialize as
designed, C6's original form is withdrawn, C6-revised's enlargement
test is only half-confirmed, and C8 is confounded rather than
decisive. **What IS established**: postnorm's failure away from the
origin is a real, structural consequence of LayerNorm's `1/sigma`
term specifically (not mean-centering, not a generic "postnorm is bad"
statement, and not a training/relocation artifact) - a narrower,
better-evidenced claim than the full generalization, with several of
this round's own sub-tests (C3, C5, C8) identifying exactly where the
simple picture needs more care before it would support that broader
statement.

## Round 3: C5-revised, C3-revised, C4-fixed, C8-revised (2026-08-31)

Three corrections to how round 2 should be read, plus the C8 fix/drop
decision (chose fix). Same execution provenance as every other round:
local CPU, `layernorm_study/CLAUDE.md`.

### Results table (rounds 2-3 combined, current status)

| Test | Status | Key number |
|---|---|---|
| Task 4 (output bound) | PASS | bound never exceeded, 16/16 ckpts (max ratio 0.74) |
| Task 5 (sqrt(eps) kink scaling) | FAIL | amplitude `~eps^-0.066`, width `~eps^0.030` (predicted `+-0.5`) |
| C1 (bias ablation) | **PASS, decisive** | r\* collapses `0.375 -> 0.00079` = **475x** |
| C2 (ceiling, free-run) | PASS | median plateau/Y_max `0.316` |
| C3 (decay slope) | **PASS (corrected prediction)** | non-radial `Ju` slope `-1.005`, CI `[-1.023, -0.982]` |
| C4 (rank / Euler) | **PASS, clean** | rank `= H-2` 100% of seeds; principal angles `~1e-14` deg |
| C5 (two-shell floor) | **does not apply** (3rd reason) | postnorm beats floor by `530x` |
| C6 (relocation) | WITHDRAWN | ill-posed prediction, retracted by hypothesis author |
| C6-revised (ceiling / enlargement) | mixed | `Y_max/required = 2.34`; r\* scaling slope `0.913` |
| C7 (which part of LN) | **PASS, clean** | RMSNorm `0.530` vs real LN `0.492`; frozen-sigma `0.057` |
| C8 (gain sweep) | FAIL (prediction), no longer ambiguous | failure radius pinned at `1e-6` for `rho<=1.5` |

### C3-revised + radial check: PASS, with the prediction corrected

Peeling the block's nonlinearities off (all postnorm, all else equal,
8 seeds each, `results/round3_C3revised_slopes.csv`):

| arm | median far-field slope | 95% CI across seeds |
|---|---|---|
| LN+GELU+GLU | `-2.332` | `[-2.787, -1.759]` |
| LN+GELU | `-1.984` | `[-2.691, -1.787]` |
| LN+linear | `-1.925` | `[-2.498, -1.592]` |

The GLU-compounding hypothesis is **partially confirmed**: removing GLU
moves the slope `-2.33 -> -1.98`, which is most of the available
movement, and removing GELU adds almost nothing (`-1.98 -> -1.93`). But
it does **not** walk to `-1` - a purely linear branch still measures
`-1.9`. Rather than log that as a second bare FAIL, the mechanism was
worked out and tested (`round3_c3_radial_check.py`).

**The `-1` law was being measured along the one direction where it
cancels.** LayerNorm's Jacobian annihilates `zhat` (exactly C4's Euler
identity). Sweeping the SCALAR state `x` outward moves the pre-norm
activation `v = b + Mz` essentially radially, so the perturbation
direction is asymptotically parallel to `zhat` itself - the direction
the bracket kills. The leading `1/r` term cancels and the subleading
`1/r^2` sets the slope. The discriminating test is the **non-radial**
derivative on the SAME checkpoints
(`results/round3_C3_radial_check.csv`):

| arm | `Jx` (radial) | `Ju` (non-radial) |
|---|---|---|
| LN+linear | `-1.925` CI `[-2.498, -1.592]` | **`-1.005`** CI `[-1.023, -0.982]` |
| LN+GELU+GLU | `-2.332` CI `[-2.787, -1.759]` | `-1.027` CI `[-1.671, -0.881]` |

`Ju` for the linear branch lands at `-1.005` with a CI of
`[-1.023, -0.982]` - tight, and containing `-1` while excluding
everything else. **C3 becomes a PASS with a corrected prediction: the
`-1` law is LayerNorm's own and holds in every direction EXCEPT the
radial one, where the Euler identity cancels it to `-2`.** Round 2's
`-2.33` was never evidence against the theory; it was the theory's own
C4 structure showing up in C3's measurement geometry. C3 and C4 are the
same fact seen twice.

### C4-fixed: PASS, cleanly

`results/round3_C4fixed_null_space.csv`. Corrected per the round-2
catch (LN's Jacobian has TWO exact null directions - `1` via `P`, and
`zhat` via the outer-product term - so a single-vector alignment check
was the wrong test):

- **rank `= H-2 = 6`** at relative tolerance `1e-4`, for **100% of
  seeds at both radii** tested.
- **Principal angles** between the measured 2D null space and
  `span{1, zhat}`: median `(2.2e-14, 1.3e-5)` deg at `r=1`, and
  `(2.0e-14, 1.1e-11)` deg at `r=1000`. Essentially exact.
- Bonus confirmation not predicted in advance but consistent: the
  second null direction gets *more* exact with radius
  (`sigma_{H-1}/sigma_max`: `6.4e-7` at `r=1` -> `9.1e-13` at
  `r=1000`), which is what the `eps->0` limit requires, since
  `sigma = sqrt(||Pv||^2/H + eps)` makes `eps` relatively negligible as
  `||v||` grows. Reported at two tolerances rather than one precisely
  because a machine-eps rank alone says `H-1` and hides this.

### C5-revised: still does not apply - and the third reason is the important one

`results/round3_C5revised_results.csv`. Both shells were placed above
the C1-measured `r*` (`r1 = 10r* = 5`, `r2 = 500r* = 250`) and the
far-field condition was verified empirically as instructed: median
local log-log Jacobian slope `-1.794` at `r1` and `-1.772` at `r2` -
**both clearly in the decay regime, not the plateau. Round 2's
mis-specification is genuinely fixed.**

**And postnorm still misses the floor - by 530x in the safe direction.**
Predicted floor `126.175`; achieved postnorm `rmse_r2` median `0.236`
(ratio `0.0019`). Prenorm control `0.00069`.

**Flagging a confound in my own verification before interpreting this**:
the check performed (local Jacobian decay slope at each shell, which is
what was specified) is NECESSARY but NOT SUFFICIENT for the floor to
apply. The floor's actual precondition is on `F`, not `J`: it assumes
`F(r1 w) ~ F(r2 w)`, i.e. the map is degree-0 BETWEEN the shells. A
decaying Jacobian does not establish that - `J ~ r^-1.8` still
integrates to a large output change over `5 -> 250`. Measured directly
from the achieved errors: `F(r1) ~ 5.15` and `F(r2) ~ 257.5`, both
correct, **differing by ~50x**. The map is emphatically not degree-0
across that range, so the floor does not bind. My verification passed
while the condition the floor actually needs failed.

**Why the model can do this, and why it matters more than the C5
result itself:** `Y_max` is not a fixed structural barrier. It is
`||W_dec||(||gamma||_inf sqrt(H) + ||beta||) + ||b_dec||` - every
factor trainable. Measured across three separate experiments, it
tracks the training-data scale at a near-constant ratio:

| experiment | max abs target | measured `Y_max` | ratio |
|---|---|---|---|
| round 2 C2 (origin-centered) | `3.5` | `6.7` | `1.91` |
| round 2 C6-revised (`x_ref=50`) | `100.9` | `120.3` | `1.19` |
| round 3 C5-revised (`r2=250`) | `336.0` | `407.5` | `1.21` |

**The ceiling is a LEARNED quantity that expands to cover whatever
range the training data occupies.** This substantially corrects the
impossibility framing at its root: `||F(z)|| <= Y_max` is true for any
FIXED checkpoint, but nothing stops training from scaling `Y_max` up to
whatever the data demands. So there is no impossibility for data in a
bounded range - and the two-shell test appears to be **unfalsifiable by
construction in this setup**, because both `r*` and `Y_max` adapt to
wherever the shells are put. Making it bite would require CONSTRAINING
`Y_max` (freezing the `gamma`/`W_dec` scales) while pushing the data
out - not attempted here, and flagged as the design that would be
needed rather than claimed as a result.

**What survives:** boundedness is real per-checkpoint, so the
DPC-relevant statement is about EXTRAPOLATION, not representability -
a trained postnorm surrogate is bounded by its own `Y_max`, and a
free-running rollout of an unstable plant leaves the training range and
hits that ceiling (round 2's C2, which measured exactly this). That is
a narrower claim than the impossibility theorem, and it is the one the
evidence supports.

### C8-revised: FAIL for the stated prediction, no longer ambiguous

Chose fix over drop, and the round-2 confound is confirmed as the
diagnosis. Per-rho excitation is now scale-matched and validated BEFORE
training (`results/round3_C8revised_excitation_validation.csv`):
`max|x|` spread `19.1-22.8` across all five gains (round 2's
uncontrolled version: `19.5-311`), condition numbers `4.95-9.33`
(round 2: `10-165`), and least-squares `(A,B)` recovery to `~1e-16`
everywhere - so the data is exactly identifiable at every gain, and
round 2's rho=1.5 dataset (which required outputs of `~467` against a
`Y_max` of `~120-150`, i.e. unrepresentable) is fixed.

Result (`results/round3_C8revised_results.csv`), converged runs only:

| rho | median failure radius | converged |
|---|---|---|
| 0.5 | `1e-6` | 5/8 |
| 0.9 | `1e-6` | 6/8 |
| 1.03 | `1e-6` | 6/8 |
| 1.5 | `1e-6` | 7/8 |
| 3.0 | `0.902` (min `1e-6`, max `29.8`) | 4/8 |

**The prediction ("failure radius shrinks monotonically as gain rises,
with stable systems on the same curve") is NOT supported.** For
`rho <= 1.5` the failure radius is pinned at the smallest radius
tested - postnorm's Jacobian is already >10% wrong AT THE ORIGIN - and
`rho=3.0` moves in the OPPOSITE direction (larger, not smaller) with a
huge spread and the worst convergence rate (4/8).

Two honest observations rather than one tidy conclusion. First, this is
now a real negative on validated data, not a setup artifact - the
`1e-6` floor is not an artifact of unrepresentable targets any more.
Second, and more informative: postnorm's Jacobian error is present at
every radius **including the origin, for STABLE plants (`rho=0.5`,
`0.9`) exactly as much as unstable ones**. That cuts against framing
this failure as being about instability at all. Standing caveat: 30% of
runs still did not converge below `1e-4` (and only 4/8 at `rho=3`), so
the `rho=3` row in particular should not be over-read.

### Round 3 bottom line

**The bar was C1 and C5 both landing. C1 lands; C5 does not** - and
round 3 establishes precisely *why* it does not, which is a more useful
outcome than a third inconclusive attempt: the impossibility argument
implicitly treats `Y_max` as fixed, and it is not. **The generalized
claim is therefore kept explicitly UNESTABLISHED**, as instructed.

What IS supported, correctly scoped:

1. **Postnorm's failure away from the origin traces specifically to
   LayerNorm's `1/sigma` term** - C7: RMSNorm (drops centering, keeps
   `1/sigma`) fails within 8% of real LayerNorm (`0.530` vs `0.492`);
   frozen-sigma (keeps centering, drops `1/sigma`) recovers ~9x to
   `0.057`. Mean-centering is exonerated.
2. **The good region is a bias-determined operating point** - C1: 475x
   collapse in `r*` under bias ablation, with `r* ~ ||Pb||^0.93`
   (directionally right, `r^2=0.31`, not a clean power law).
3. **The output ceiling is real and measurable per-checkpoint** - Task
   4 and C2 - **but it is learned, not fixed** (round 3's `Y_max`
   scaling table above). The defensible form is about extrapolation
   beyond the training range, not about representability within it.
4. **LayerNorm's Jacobian is exactly rank-deficient by 2, with the
   `-1/r` law holding off the radial direction** - C4 (principal
   angles `~1e-14` deg, rank `H-2` on 100% of seeds) and C3's radial
   check (`Ju` slope `-1.005`, CI `[-1.023, -0.982]`) are the same
   structural fact measured two ways.

**Open thread, named rather than resolved (the link back to the parent
project):** C6-revised found `Y_max/required = 2.34` - the model had
ample headroom to represent the required magnitude yet still got
`Jx = 0.035` where the truth is `1.03`. Combined with round 3's C5
result (postnorm fits VALUES at both shells to `~0.2` RMSE while its
Jacobian decays as `r^-1.8` through that same range), this is a
**value/derivative dissociation**: the model fits outputs acceptably
while getting derivatives badly wrong, and the two failures are not
the same failure. That is precisely the phenomenon the parent repo's
DPC analysis is about - DPC backpropagates through the surrogate's
DERIVATIVES, so a surrogate can look well-fit by one-step MSE and
still be useless for control. This sub-project's LayerNorm mechanism
is one concrete, fully-traced instance of that general failure mode.

## Round 3 correction, by the hypothesis author: Theorem 2 is vacuous, not merely blocked by a confound (2026-09-01)

Two corrections to how round 3's own findings should be understood
theoretically, recorded by the person who wrote the original boundedness
theory, after reviewing round 3's results.

**Theorem 2 (the postnorm output bound) is TRUE and VACUOUS.** The
original claim was that postnorm output is bounded by
`Y_max = ||W_dec||(||gamma||_inf sqrt(H) + ||beta||) + ||b_dec||`, stated
as if this were an ARCHITECTURAL constraint - a structural limit that
holds of the postnorm architecture as such, the way "LayerNorm's output
lies in a ball of radius `sqrt(H)`" holds regardless of what any
particular checkpoint's weights are. That framing was wrong. `gamma`,
`W_dec`, and `b_dec` are TRAINABLE weights, not architectural constants -
the bound is correctly derived, and is exactly true for any one FIXED
`theta`, but nothing in the postnorm architecture stops `gamma` or
`W_dec` from growing during training to whatever scale the data demands.
Round 3's own measurement is the direct evidence this isn't a
theoretical nitpick but what actually happens: `Y_max` tracked the
training-data scale at a near-constant ratio across three separate,
independently-trained experiments (`1.19`-`1.91x` the required output
magnitude - round 3's `Y_max` scaling table, above). **A bound that is
true for every fixed `theta` but where `theta` is itself free to grow
without limit does not restrict the function class the architecture can
express - "bounded for any given trained model" is a fundamentally
weaker statement than "bounded, therefore cannot represent an unstable
system," and only the first one is what the theorem actually proves.**
This is the precise sense in which the theorem is vacuous: not incorrect
in its derivation, but applied to a claim (representability) it was
never positioned to settle, because its own free parameter absorbs
exactly the growth the impossibility argument needed to be impossible.

**Consequence for the generalized claim.** "Postnorm residual blocks can
never reproduce an unstable system" stays exactly where round 3 left it -
explicitly UNESTABLISHED - but now for a STATED structural reason rather
than an unresolved negative result. What Theorem 2 actually licenses is
narrower and remains true: postnorm IS bounded for any given trained
checkpoint, and that bound explains failure OUTSIDE that checkpoint's
training range (round 2's C2 free-run ceiling, Task 4) - an
extrapolation claim, not a representability claim. It does not, and was
never going to, prove postnorm cannot fit an unstable system on some
bounded operating range, because "bounded on a fixed range with
free-to-grow weights" was never in tension with representing instability
on that same range. Nothing about this weakens C1, C4, or C7 - those are
claims about the LOCAL JACOBIAN's structure (rank deficiency, the
`1/sigma` term, the bias-determined near field), which do not route
through `Y_max` and are unaffected by this correction.

**C3's original GLU-compounding hypothesis was WRONG - the real
mechanism is the same fact C4 already proved, seen from a different
angle.** The prediction going into C3-revised was that GLU's product
structure (`a(v)*sigmoid(b(v))`, each factor independently decaying like
`1/r`) would compound to `1/r^2`, and that peeling GELU/GLU off would
walk the slope back to `-1`. That prediction is falsified by round 3's
own data: `LN+linear` - no GELU, no GLU, nothing left to compound -
still measures a radial slope of `-1.9`, not `-1` (`round3_C3revised_slopes.csv`).
The actual mechanism, found by round3_c3_radial_check.py, has nothing to
do with nonlinearity composition: LayerNorm's Jacobian bracket
`[P - zhat zhat^T/H]` ANNIHILATES the `zhat` direction exactly - this is
C4's Euler identity, `||J(v)v|| ~ 1e-14` relative, confirmed to
principal angles of `~1e-14` degrees. Sweeping the scalar state `x`
outward moves the pre-norm activation `v = b + Mz` essentially radially,
so the swept perturbation direction is asymptotically PARALLEL to
`zhat` itself - exactly the direction the bracket kills. The leading
`1/r` term cancels along that one specific direction, and the
subleading `1/r^2` term is what's left to set the slope - giving `-2`,
not because two nonlinear stages compounded, but because the radial
sweep direction happens to coincide with LN's own null direction. The
non-radial derivative (`Ju`, perturbing along `u` instead of `x`, which
is NOT parallel to `zhat`) retains the uncancelled leading term and
measures `-1.005` (CI `[-1.023, -0.982]`) for the same checkpoints where
the radial derivative measures `-1.9` to `-2.3`.

**This is the sharpest methodological finding in the study.** C3 (a
decay-RATE measurement, log-log slope vs. radius) and C4 (a
rank-deficiency measurement, SVD of the Jacobian at a point) look like
two unrelated diagnostics testing two unrelated predictions. They are
the SAME structural fact - LayerNorm's Jacobian exactly annihilates the
`zhat` direction - measured two different ways. C4 finds the null
direction directly, by decomposing the Jacobian. C3 found it
indirectly and by accident, by choosing a sweep direction that happened
to align with it, which silently removed the leading-order term from
what was being measured and reported the next order down as if it were
the whole story. The original round-2 `-2.33` result was never evidence
against the `-1` law - it was the theory's own null space showing up
inside an underspecified measurement, and it would have looked exactly
as "wrong" for ANY architecture whose Jacobian has an exact null
direction aligned with how someone chose to sweep it. The general
caution: when measuring "the" decay slope of a rank-deficient linear
map, the direction of the sweep is not a free implementation detail -
it determines whether the leading-order term is even present in what
gets measured, and a single scalar "slope" number silently encodes that
choice without flagging it.

## Round 3 audit: independent re-derivation of A1/A2/A3 (2026-09-01)

Requested standard: re-derive from the underlying computation, not from
this document's own prose. `layernorm_study/experiments/round3_audit.py`
retrains the specific (arm, seed) checkpoints the audited numbers came
from and interrogates them directly - full method and all three results
below, `results/round3_audit_A*.csv`.

**Environment caveat, disclosed up front because it affects how to read
every number below.** This machine has no Python 3.11 (the version
pinned in `requirements.txt`) - reran on 3.12, identical package pins
otherwise (`jax==0.7.2`, `flax==0.11.2`, `optax==0.2.8`, confirmed
installed). The retrained checkpoints are **not bit-identical** to the
original round-3 runs despite identical seeds: e.g. `LN+linear` seed=0's
`train_mse` is `2.403e-06` here vs. the committed CSV's
`1.315e-06`; the `rho=0.5` seed=0 bonus retrain gives `1.940e-05` vs.
`5.418e-06`. This is NOT the determinism failing in general - two
INDEPENDENT scripts in the ORIGINAL environment reproduced each other to
every printed digit (verified before any of this, see the audit script's
own docstring) - it is specifically the Python 3.11->3.12 (and whatever
wheel-build/BLAS difference rides along with it) that breaks bit-exactness,
consistent with the parent repo's own documented worry that float
differences compound over many nonlinear optimization/recurrence steps.
Each fresh checkpoint IS internally self-consistent (independent code
paths computing the same quantity on the same checkpoint agree to full
precision - e.g. A2's direct dtype-audit and its cross-check against
`c4_null_space_check()` agree on `sigma_max` to all 6 printed digits).
**Read everything below as an independent replication on freshly-trained
checkpoints of the same architecture/data/seed - not as a bit-exact
verification of the existing CSV rows.** Where the two disagree
quantitatively but agree qualitatively, that is itself informative (the
phenomenon survives a real environmental perturbation); it is flagged
explicitly wherever it isn't just qualitative agreement.

### A1 - C3 non-radial slope: CONFIRMED for LN+linear, NOT robust for the full model

**Direction check (is "non-radial" actually non-radial?).** Measured
directly, not assumed: at a representative far-field point, `LN+linear`
seed=0's radial direction (`dv/dx`) sits `7.5deg` from `zhat`; the
non-radial direction (`dv/du`) sits `89.3deg` from it - i.e. essentially
exactly orthogonal, not merely "different." Seed=3: `14.7deg` / `89.2deg`,
same pattern. **This is a clean, decisive confirmation of the mechanism**:
the radial sweep really is (nearly) the direction LN's bracket
annihilates, and the non-radial direction really is (nearly) untouched
by it - not a coincidence of labeling.

**Window sensitivity - the CI's tightness IS robust, but only for
`LN+linear`.** Fit on 7 different windows per checkpoint (halves,
narrow near/far edges, an extended range to `1e5`, and a window placed
just past the checkpoint's own measured `r*`): `Ju` (non-radial) stays
in `[-0.96, -1.00]` across EVERY window tried, both seeds - the tight CI
survives fit-range interrogation cleanly. Pointwise (adjacent-pair)
local slopes confirm this isn't an average masking a transient: `Ju` is
`-1.00` at every single point across the entire `c=10` to `c=1000`
sweep for both seeds. `Jx` (radial) does drift within the window
(`-1.79` at `c=10` climbing to `-2.00` by `c~300+`) - the `-2` law is a
genuine far-field asymptote, not instantaneous, and the window
`[10,1000]` sits mostly-but-not-entirely inside full convergence (`r*`
for these checkpoints is `0.9`-`1.3`, so the window starts `~8`-`11x`
past `r*` - sufficient here, but see below, not universally sufficient).

**It is NOT robust for `LN+GELU+GLU` (the full model) - flagged, not
smoothed over, per the standing instruction.** The same window sweep on
`LN+GELU+GLU` seed=0 (`train_mse=2.7e-5`, one of the noisier fits in the
original 8-seed spread too - its own recorded `r2=0.91` was already the
second-lowest of the eight) swings wildly: full-window `Jx=-2.02`
(`r2=0.74`), lower-half `Jx=+0.20` (`r2=0.05`, wrong SIGN), upper-half
`Jx=-4.20`, extended-far `Jx=-0.45`. `Ju` is similarly unstable
(`-0.71` full window, `-2.49` near-edge, `-0.33` upper-half). The
direction check also weakens for this arm: radial-to-`zhat` angle is
`24.3deg` (vs `LN+linear`'s `7.5-14.7deg`), non-radial is only `63.3deg`
from `zhat` (vs `~89deg`) - the clean orthogonality is specific to the
no-nonlinearity case. **Honest reading: the `-1.005, CI[-1.023,-0.982]`
result is a real, robust, decisive confirmation of LayerNorm's own law -
but specifically for the isolated-LayerNorm (`LN+linear`) case. The full
model's non-radial slope is directionally consistent with `-1` (its own
originally-recorded CI, `[-1.671,-0.881]`, does contain it) but is not
independently window-robust on the one checkpoint checked here - GELU/GLU's
curvature genuinely interferes with getting a clean asymptotic read, it
doesn't just add noise around an otherwise-stable number.** Only one
`LN+GELU+GLU` seed was checked this way (seed=0); whether this
non-robustness holds for all 8 seeds is not established either way.

### A2 - C4 rank/dtype: CONFIRMED directly, not inferred

`jax.config.jax_enable_x64` reads back `True` from the live config (not
just "we called `update()` so it must be on"). Recomputing the LayerNorm
Jacobian fresh (`LN+GELU+GLU` seed=0) and printing dtype at each stage,
unabbreviated: `v.dtype = float64`, `J.dtype` straight out of
`jax.jacfwd` `= float64`, still `float64` after `np.asarray()`. The full
8-value singular spectrum makes the rank cut visible rather than
asserted: at `r=1`, six values cluster within `1.5x` of each other
(`1.55` down to `1.07`), then drop SIX orders of magnitude to `1.56e-6`,
then a further ELEVEN orders to `5.48e-17` - an unambiguous rank-6 cut,
at either tolerance convention. At `r=1000` the same structure holds and
the second null direction is cleaner still (`4.90e-15`, dropping to
`1.90e-19`), matching the theory's own prediction that it sharpens as
`eps` becomes relatively more negligible.

A second, independent argument that needs no rerun at all, from the
ALREADY-COMMITTED CSV alone: the worst-case (largest) `sigma_H/sigma_max`
across all 16 originally-recorded rows is `1.24e-16`. float32 machine
epsilon is `1.19e-7`. `1.24e-16` is roughly `10^9` times smaller than
float32 can resolve - float32 arithmetic is physically incapable of
producing a ratio that small; the original numbers could only have come
from `>=float64`, independent of anything reproduced today. **Both
lines of evidence agree: x64 was genuinely active, not just declared.**

### A3 - C8 "pinned at 1e-6": reworded, and independently confirmed as a real failure, not a grid artifact

`C_VALUES = np.logspace(-6, 3, 60)`: smallest tested value is exactly
`1.000e-6` (second-smallest `1.421e-6`). `failure_radius()` scans from
the smallest `c` upward and returns the FIRST value whose relative error
exceeds 10% - if `c[0]` itself already fails, the function returns
`c[0]` without ever probing anything smaller. **Per-rho breakdown of
converged seeds, computed directly from the existing results CSV:**

| rho | converged | at grid floor | real measured crossings |
|---|---|---|---|
| 0.5 | 5/8 | 5/5 (100%) | none |
| 0.9 | 6/8 | 6/6 (100%) | none |
| 1.03 | 6/8 | 6/6 (100%) | none |
| 1.5 | 7/8 | 5/7 (71%) | seed4=0.441, seed5=0.310 |
| 3.0 | 4/8 | 1/4 (25%) | seed0=29.8, seed1=1.80, seed4=0.0092 |

**For `rho<=1.03`, "failure radius = 1e-6" is unanimous censoring - not
one of 17 converged seeds across those three gains ever showed a real
measured crossing.** The correct statement is "postnorm's Jacobian is
already wrong by >10% at the smallest scale tested," not "the failure
radius is 1e-6" - the latter wording implies a measured crossing point
that these rows never actually contain. `rho=1.5` and especially
`rho=3.0` are different: a real minority of seeds show genuine measured
crossings well above the floor, so "pinned" over-states uniformity there
specifically (median is still `1e-6` because it's the majority value,
not because every seed lands there).

**Bonus, to settle whether this is a hidden nearby crossing or a
genuine at-every-scale failure**: retrained `rho=0.5` seed=0 (a
unanimously-censored case) and extended the grid to `[1e-15, 1e3]`, 91
points. Despite the environment-driven non-reproduction (`train_mse`
doesn't match, see caveat above, though it's still converged by the
project's own `<1e-4` threshold), the result is unambiguous: `Jx=+6.25`
at `c=1e-15`, IDENTICAL at `c=1e-9` and `c=1e-6` (i.e. flat - the
near-field plateau C1 predicts really is flat here), still `Jx=+3.31`
even at `c=1` - against a true `rho=0.5`, that is a relative error of
`1150%` at the smallest scale tested, decaying only to `562%` by `c=1`.
**No point anywhere in 15 orders of magnitude ever comes within 10% of
the true value.** This is not a censored-but-nearby crossing the
original 60-point grid merely failed to resolve - it is a genuine,
uniform, order-of-magnitude failure across the model's entire near-field
plateau, for one of the mildest (stable, `rho=0.5`) gains in the sweep.
If anything this independently STRENGTHENS C8-revised's own reading
("postnorm's Jacobian error is present at every radius including the
origin... cuts against framing this failure as being about instability
at all") rather than calling it into question.
