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

**Standing status**: Tasks 4 (harden the postnorm ceiling claim) and 5
(epsilon sweep on arm_5) are queued but not yet run, per instruction
to stop and report Task 1(d)/Task 3 first.
