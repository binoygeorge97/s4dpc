# DECISIONS.md — s4dpc

Append-only, dated. Decisions that change what enters the paper or the
canonical code paths, with the reasoning, so reviewers (and future us) don't
have to reconstruct it from commit messages.

---

## 2026-08-07 — Case 4's off-diagonals rescaled (100/80/120 -> 5/4/6)

`s4dpc/systems.py`'s `get_discrete_matrices`, case 4 ("Non-normal Jordan
Block"), originally had off-diagonal entries 100, 80, 120 in its three 2x2
blocks. Computing `tests/test_systems.py`'s comparison table with
`np.linalg.eig` on the resulting `A_d` gave `cond(eigenvectors) = 0.00000`
and `||A_d||_2 = 1.8e24`. A condition number below 1 is mathematically
impossible (cond(V) = sigma_max(V)/sigma_min(V) >= 1 always) — `eig`'s
eigenvector solve had broken down on a near-defective matrix, silently
returning a numerically meaningless `V`, not a genuine small value.

Off-diagonals rescaled to 5, 4, 6 (same diagonal entries, same block
structure, same "Non-normal Jordan Block" intent). This keeps genuine
non-normality and defective structure — case 4's Kreiss-like amplification
is still ~3.3x, and its Frobenius-normality departure is still nonzero —
while keeping every quantity in the comparison table finite and in a
numerically trustworthy range. rho(A_d) is essentially unchanged (1.02020
both before and after, since it only depends on the unchanged diagonal
entries).

**How to apply:** don't revert this to the original 100/80/120 scale without
re-deriving the comparison table and re-checking every downstream quantity
(condition numbers, transient-growth norms) stays finite. See the entry
below for why `np.linalg.eig` itself is banned from this comparison,
independent of this rescaling.

---

## 2026-08-07 — np.linalg.eig banned from tests/test_systems.py

`np.linalg.eig`'s eigenvector output is unreliable for defective or
near-defective matrices (exactly the kind case 4 and case 6 are designed to
be) — a matrix without a full eigenbasis doesn't have a numerically stable
`V` to compute, and `eig` can return something that looks small/finite
while being meaningless (see the case-4 entry above: `cond(V) = 0` is not a
valid condition number).

`tests/test_systems.py` now computes `rho(A_d)` via `np.linalg.eigvals`
(eigenvalues only, no eigenvector solve to go unstable) and replaces the
eigenvector-based `cond(V)` diagnostic with SVD-based measures instead,
which stay well-behaved regardless of how defective `A_d` is:
`||A_d^k||_2` for k in (1, 5, 10, 25, 50) (transient growth), and
`max_k ||A_d^k||_2 / rho^k` over that same k-grid (a Kreiss-like
amplification proxy — not an exact sup over all k, just the max over the
sampled grid). `||A A^T - A^T A||_F` (departure from normality) is
unchanged.

**How to apply:** any future diagnostic touching these plants' `A_d`
matrices should default to SVD-based measures (`np.linalg.norm(..., ord=2)`,
`np.linalg.svd`, `np.linalg.cond` on a matrix directly rather than on an
eigenvector matrix) rather than `np.linalg.eig`'s eigenvectors.

---

## 2026-08-07 — Case 1 is an integrator, not a typical stable baseline

Case 1's continuous-time `A` is block upper-triangular with each 2x2
diagonal block of the form `[[a, b], [0, 0]]`. Such a block's eigenvalues
are trivially `{a, 0}` (already upper triangular). So case 1's continuous
system has three exact-zero eigenvalues (from the three diagonal blocks)
alongside `{-3.5, -3.5, -5.2}` — marginally stable / integrator modes, not
asymptotically stable ones, despite the case comment reading "Base Original
(Stable with some coupling)".

A continuous eigenvalue of exactly 0 maps to a discrete eigenvalue of
exactly 1 under *any* discretization method (bilinear and ZOH both agree
here — see `tests/test_systems.py`'s `rho(A_d)` column: case 1 is 1.00000
for both `tustin@0.01` and `zoh@0.02`, the only case where the two methods
coincide). That's why case 1 has `rho(A_d) = 1.0` exactly, not because of
any bug in the discretization — it is mathematically forced by the zero
continuous eigenvalues.

**How to apply:** don't use case 1 as "the" representative stable baseline
when the distinction between marginal and asymptotic stability matters
(e.g. any analysis of decay rates, or claims like "the model should learn a
contracting map"). Checked all seven continuous spectra directly
(`np.linalg.eigvals`) rather than trust the case-comment names: case 1 is
in fact the *most* benign of the seven in open loop. Every other case has
zero or more eigenvalues with strictly positive real part — cases 2, 3, 5
are fully unstable (every eigenvalue positive, matching their "Unstable"
comments), case 4 is also fully unstable (its comment doesn't claim
otherwise), case 6 has one exact-zero eigenvalue plus five positive, and
case 7 is genuinely mixed (3 negative, 3 positive, matching its "Mixed
Stable/Unstable" comment). If a non-integrator baseline with any stable
modes is needed, case 7 is the only candidate, and even it is only
half-stable. Case 1's comment ("Stable with some coupling") is the only
one of the seven that's actually misleading about stability character —
every other case's name matches what its spectrum shows.

---

## 2026-08-07 — M3 identification underfits case 3 by ~9-11 orders of magnitude; root cause is optimization, not capacity or structure

The initial M3-vs-M6 smoke test (case 3, d_model=16, 50 epochs) gave
teacher_mse=3.46 for M3, an LTI S4 model on a normal, well-conditioned
plant. Diagnosed in four steps (each run and reported before moving on):

**1. Scale.** `nmse = mse / mean(target**2)` = 0.51-0.58 for M3 (0.82-0.85
for M6) against case 3's `|x_k+1|` (mean 1.81, std 1.76, max 6.73). Not a
units/scale illusion - the raw numbers reflect a genuinely bad fit.

**2. Optimization.** `identify.py` called `optax.adamw(learning_rate)`
with no `weight_decay` override; optax's own default is 1e-4, not 0 (now
fixed - `weight_decay` is an explicit parameter, default 0.0, threaded
through `_train_one`/`_train_ensemble`/`run_identify`). No LR schedule
existed (flat lr, still true - not changed, per the instruction to test
with lr=1e-3 flat). Re-ran M3/case 3/1 seed/2000 epochs/lr=1e-3/wd=0
(`tools/diagnose_m3_convergence.py`, plot at `docs/m3_case3_loss_curve.png`):
mse dropped from 3.46 to 0.0048 (nmse 7.6e-4) - a ~650x improvement, but
the curve is still descending, not plateaued, at step 2000, and still
~9 orders above the expected floor. Weight decay's own arithmetic effect
(`lr * wd = 1e-3 * 1e-4 = 1e-7`/step) is too small to have caused the
original gap alone - the epoch count (50, vastly too few) is almost
certainly the dominant factor, though this run changed both at once so
that isn't isolated.

**3. Capacity.** Added `identify.fit_least_squares` (M1): closed-form
`[A_hat|B_hat] = Y Z^dagger` (via `np.linalg.lstsq`, float64) on the same
case_data. Case 3: mse=5.46e-15 (nmse=8.6e-16) - matches the expected
~1e-14 floor almost exactly. M3 after step 2 (nmse=7.6e-4) is still
~9 orders of magnitude above this. Deliberately not
dpc_example's `LinearDynamicsModel` LayerNorm-weight-extraction approach:
that file's own comment (around line 165) admits extracting W/V from a
LayerNorm'd model does not give an equivalent linear system, since
LayerNorm is a nonlinear operation - not a substitute for a real
closed-form solve.

**4. Structure.** Case 3, 2000 epochs, lr=1e-3, wd=0 throughout
(`tools/diagnose_m3_structure.py`, self-check against the real model's
output passed at max_abs_diff=0.0 before trusting any ablated result):

| ablation | nmse |
|---|---|
| full M3 (d_model=16) | 7.6e-4 |
| d_model=64 (else identical) | 2.3e-4 |
| D-only (S4 kernel zeroed, feedthrough alone) | 8.9e-4 |
| conv-only (D zeroed, kernel alone) | 6.4e-4 |

D-only and conv-only both land in the same range as the full model -
the S4 recurrent/convolution path isn't contributing much beyond what a
pure instantaneous (memoryless) map already gets. This is *expected*
given teacher forcing: since the true state `x_k` is handed to the model
explicitly in every input, the target map `(x_k, u_k) -> x_k+1` is
inherently instantaneous (literally `[A|B] @ [x_k; u_k]`) - no recurrence
is needed to solve *this* task, regardless of whether the underlying
plant has real dynamics. The encoder/per-block out-projection/decoder
are all full linear layers, so D-only (encoder -> per-channel-scalar-D ->
out-projection -> decoder, all linear) is a composition of linear maps
and is theoretically capable of representing the exact optimal linear
map - the same function class the least-squares floor achieves - yet it
still sits ~11 orders of magnitude away from that floor. Widening to
d_model=64 helped only ~3x, not orders of magnitude.

**Conclusion so far:** capacity is not the bottleneck (D-only's function
class contains the exact solution; case 3 is a small, well-conditioned,
normal system) and the S4 recurrence isn't adding value on this
teacher-forced task either way. The remaining gap looks like an
optimization problem: fitting a linear map through a deep, multiplicatively
-factored parameterization (encoder / per-channel D or S4 kernel /
out-projection / decoder) via Adam converges far more slowly than the
one-shot closed-form solve, similar to known slow-convergence behavior in
deep linear network training. Not yet root-caused further (e.g., whether
more steps alone would eventually close most of the gap, or whether
initialization/conditioning of the specific factorization is the limiter)
- reported per-step as instructed; controller work is on hold pending
this.

---

## 2026-08-07 — M3's D-only gap is a genuine parameter non-identifiability (exact rank-deficient Gauss-Newton matrix), not slow conditioning

Follow-up to the entry above. Four experiments (case 3, D-only ablation
throughout unless noted), each self-checked before trusting the result and
reported before the next:

**1. Asymptotic-or-stuck** (`tools/diagnose_m3_asymptotic.py`, 50k steps,
lr=1e-3, wd=0, Kaggle T4, 64.70 GPU-minutes - logged in `gpu_ledger.csv`):
neither pattern predicted. A clean power-law descent for the first ~2000
steps (mse 29.5 -> ~1e-2), then a steep drop to ~1e-6 by step ~10k, then
**non-monotonic noisy oscillation** between ~1e-4 and ~1e-8 for the
remaining 40k steps - never getting closer to the LS floor (5.46e-15).
`final mse (50k) = 1.575e-7`, ratio to floor = 2.88e7. See
`docs/m3_d_only_50k_loglog.png`.

**2. Conditioning** (`tools/diagnose_m3_conditioning.py`, Gauss-Newton
`J^T J` over D-only params via `jax.jacfwd`, at init and step 2000. First
pass used `eigvalsh` directly on `J^T J` and produced small spurious
negative eigenvalues from float roundoff on near-zero true eigenvalues,
which a naive `cond = max/max(min, floor)` turned into a meaningless
`1e+305` - not reported; fixed by computing eigenvalues as
`svd(J)**2` (non-negative by construction) before trusting or reporting
anything):

| | init | step 2000 |
|---|---|---|
| numerical rank | 518 / 550 | 518 / 550 |
| rank deficiency | **32 = 2 x d_model (16)** | **32 = 2 x d_model (16)** |
| top eigenvalue | 1.04e5 | 1.03e5 |
| cond. restricted to nonzero eigenvalues | 4.86e19 | 6.02e19 |

`J^T J` is exactly rank-deficient by 32 dimensions, unchanged from init to
step 2000. This is the signature of an exact scale/gauge symmetry in the
factorization (e.g. `D_i -> D_i * t`, `out_kernel[i,:] -> out_kernel[i,:] / t`
per channel leaves the D-only forward pass invariant) - a genuine flat
manifold in parameter space, not merely a badly-scaled but full-rank
optimization landscape.

**3. Does reparameterization fix it** (`tools/diagnose_m3_reparam.py`,
2000 steps each, Kaggle CPU):

| variant | init_mse | final_mse (2000 steps) | nmse |
|---|---|---|---|
| a. as-is (random init, adamw lr=1e-3) | 29.5 | 5.67e-3 | 8.90e-4 |
| b. ls_init (hand-set to the exact LS solution at init) | 2.86e-14 | **1.89e-6** | 2.97e-7 |
| c. clipped_adam (random init, lr=1e-2 + grad-norm clip) | 29.5 | 1.05e-3 | 1.64e-4 |

(b)'s self-check confirms the construction lands within float32 precision
of the true LS floor at step 0 (2.86e-14 vs 5.46e-15). Neither (b) nor (c)
closes the gap. (c) helps only ~5x over (a). (b) is the sharper result:
**starting exactly at the optimum, 2000 steps of ordinary Adam training
moves the model 8 orders of magnitude away from it**, landing at
`1.89e-6` - squarely inside the same noisy attractor band (~1e-4 to 1e-8)
that experiment 1's 50k-step run oscillates in regardless of starting
point. The optimum is not merely hard to *reach* under this
parameterization; it is unstable under continued gradient-based training.

**Revised conclusion:** this is not a conditioning-severity/speed problem
that more steps, a better init, or a cruder optimizer would fix (all three
were tried; none closes the gap). It is a genuine non-identifiability
in the S4 block's multiplicatively-factored parameterization: an exact
32-dimensional (= 2 x d_model) null space in the loss's local curvature,
present at init and persisting through training, that a per-channel
feedthrough/gain factorization (`D`, `out_kernel`) exposes to gradient
noise regardless of where training starts. Combined with experiment 4
above (a single unfactored `nnx.Linear` also fails, ruling out
factorization *alone* as sufficient to explain the failure), the
likeliest complete picture is: the factorization creates the flat
directions, and Adam's own dynamics (momentum + per-parameter adaptive
step size) actively wander along them rather than converging - a specific,
checkable mechanism, not a vague "optimization is hard" statement.
Controller work remains on hold pending explicit go-ahead.

---

## 2026-08-08 — s4dpc/control.py built and smoke-tested; controller work gated on the M3 investigation above, not on general readiness

Per CLAUDE.md's planned repo layout: bounded GRU controller (`u =
max_action * tanh(head(h))`) + DPC loss + rollout through either the TRUE
linear plant or a trained S4 surrogate (decode=True, stepped) + an oracle
discrete-LQR solve via `scipy.linalg.solve_discrete_are`. Adapted from a
user-provided reference pipeline, not ported verbatim - the one
deliberate architectural change is that the action is actually bounded
(the reference computed `max_action` but never applied it).

**Verification before trusting any controller result:** `identify.py`
trains in conv mode (`decode=False`), but closed-loop control must step
the model one input at a time (`decode=True`) since future inputs aren't
known in advance. `tests/test_decode_construction_parity.py` already
checked the two modes' *params* match at init; that says nothing about
whether their *forward output* agrees. Added
`tests/test_control_decode_parity.py`: train a model in conv mode
(50 epochs, enough to move params off init), load those params into a
fresh decode=True model, step through the same case-3 trajectory one
input at a time, compare against the conv-mode output. Passed for both
M3 (max abs diff 2.73e-4, relative ~4e-5 of target scale) and M6
(1.31e-5, relative ~2e-6) - `rollout_learned` deploys the same function
`identify.py` trained, not a silently different one.

**Smoke test** (`tools/smoke_control.py`, case 3, a curriculum ~1/10th
the length of the reference pipeline's - this was about proving the
wiring works, not a real result): GRU controllers trained by unrolling
through the M3 and M6 surrogates, evaluated on the TRUE plant, compared
against an oracle LQR baseline and a GRU trained directly on the TRUE
plant (same short budget):

| controller | true-plant cost |
|---|---|
| oracle LQR | 235.90 |
| GRU on TRUE plant (reference) | 415.88 |
| GRU on M3 surrogate | 895.81 (id teacher_mse=3.62e-3) |
| GRU on M6 surrogate | 878.69 (id teacher_mse=3.34e-3) |

The ordering (oracle < true-plant GRU < surrogate GRUs) is sane and
confirms the identify -> control pipeline is wired correctly end to end.
Cost: 192.70 T4-minutes (logged in `gpu_ledger.csv`) - unexpectedly high,
because `@nnx.jit` was deliberately left off the training step for this
first pass (no local jax/flax environment to debug a jit-related failure
against, so correctness was prioritized over speed). Add jit back before
any larger controller run.

**Do not read the M3-vs-M6 numbers above as evidence for or against the
LayerNorm/derivative-fidelity hypothesis.** Both identification runs
feeding this smoke test were stopped at teacher_mse ~3-4e-3 - by the
investigation in the entries above and below, neither variant is
anywhere near its own achievable floor at that budget, and M3
specifically carries its own unresolved confound. M6 scoring (mildly)
better than M3 here is exactly the kind of noise that confound predicts,
not a finding. Controller work stays paused on the M3 investigation, not
on the controller code's own readiness - the pipeline itself is verified
and ready to use as soon as identification is trustworthy.

---

## 2026-08-09 — Mechanism chain, run as branch-gated experiments: both a generic Adam/data-scale problem AND an S4-specific exact non-identifiability are real, and compound

Working hypothesis going in: the S4 parameterization is multiplicatively
factored (`decoder x out x C x kernel(Lambda,P,B,step) x encoder`) and
badly conditioned for representing a near-instantaneous linear map,
because `Lambda_re <= -1e-4` and `log_step` in `[log 1e-3, log 1e-1]` are
both tuned for long-range sequence modelling, not for identifying an
unstable plant. Run as a branch-gated chain; result and branch decision
recorded after each step.

**EXP 1 - the clean control** (`tools/diagnose_m3_linear_control.py`,
new script - no committed script/log survived from the earlier ad-hoc
run of this same control before this session's context was compacted,
so it was rebuilt and re-run rather than trusting a recollected number
for the experiment that gates the whole chain). Single `nnx.Linear(9,6)`
on case-3 data, same Adam budget as the M3 diagnosis (lr=1e-3, wd=0,
2000 epochs) - same function class as the least-squares floor, zero S4
factorization:

```
LS floor:              mse=5.464e-15  nmse=8.582e-16
nnx.Linear(9,6):  init_mse=1.721e+01  final_mse=3.272e-02
                   nmse=5.140e-03     ratio_to_LS_floor=5.989e+12
```

**Implication:** does NOT reach ~1e-12; stalls at nmse~5e-3, ~12.8 orders
of magnitude above the floor - in the same ballpark as D-only's own
final nmse (8.9e-4, section above), not qualitatively better.

**Branch taken:** per the stated rule this result triggers "skip EXP
2-3, go straight to EXP 4" - a plain, already-fully-identifiable linear
regression struggling by itself means the S4 factorization is not
*required* to reproduce a failure of this magnitude, which is a real
complication for the working hypothesis. **Deviation, flagged explicitly:**
EXP 2 and EXP 3 are included anyway, because both were already run (same
exact specs the chain calls for) in the immediately preceding
investigation in this session, and their results don't just confirm what
EXP 1 already showed - they add information EXP 1 alone cannot produce
(see EXP 3's implication below). Nothing was re-run to pad the chain;
this is existing, already-verified data being read into the new branch
structure, not new compute.

**EXP 2 - asymptotic or stuck?** (`tools/diagnose_m3_asymptotic.py`,
D-only, case 3, 50k steps, lr=1e-3, wd=0 - identical to this section's
spec; result reused verbatim from the entry above): non-monotonic noisy
oscillation between ~1e-4 and ~1e-8 after an initial fast descent, never
converging closer to the floor. Neither a clean straight line nor a
clean bend/plateau.

**Implication:** doesn't cleanly match either predicted pattern, so
strictly the chain's branch rule for this step doesn't resolve cleanly
either - continued to EXP 3 regardless, since a noisy non-convergent
floor is at least consistent with "there's a mechanism actively
preventing convergence," worth explaining.

**EXP 3 - confirm the mechanism** (`tools/diagnose_m3_conditioning.py`,
Gauss-Newton `J^T J` over D-only params, reused verbatim from the entry
above): exactly rank-deficient by 32 = 2 x d_model at both init and step
2000; condition number restricted to the numerically-nonzero eigenvalues
is 4.86e19-6.02e19, vastly past the stated 1e8 threshold.

**Implication - this is the piece EXP 1 alone cannot produce:** a plain
`nnx.Linear(9,6)`'s Gauss-Newton matrix (`Z^T Z` on the augmented input,
L=100 samples >> 10 parameters) is generically full-rank - it has no
multiplicative factorization to create a scale/gauge symmetry in the
first place. D-only's *exact*, dimension-matched (2 x d_model) rank
deficiency is a signature that only a factorized parameterization like
`D_i -> D_i * t`, `out_kernel[i,:] -> out_kernel[i,:] / t` can produce.
So EXP 1 and EXP 3 are not in conflict - they show two different,
additive things: a generic Adam/data-scale sensitivity that affects even
the minimal, fully-identifiable linear regression (EXP 1), plus an
*extra*, S4-specific exact non-identifiability that only the factored
parameterization has (EXP 3). **Branch:** condition number confirms the
mechanism per the stated threshold - continue to EXP 4.

**EXP 4 - what actually fixes it** (`tools/diagnose_m3_reparam.py`,
D-only, case 3, 2000 epochs each, reused verbatim from the entry above).
(c) used gradient-clipped high-lr Adam (the explicitly offered
alternative to `optax.lbfgs`):

```
a. as-is (baseline):        final_mse=5.67e-3   nmse=8.90e-4
b. informed (LS) init:      final_mse=1.89e-6   nmse=2.97e-7  (init was 2.86e-14, at the floor)
c. clipped high-lr Adam:    final_mse=1.05e-3   nmse=1.64e-4
```

**Implication:** neither closes the gap. (c) helps only ~5x. (b) is the
sharp result - initialized within float32 precision of the exact
optimum, 2000 ordinary Adam steps move it 8 orders of magnitude away,
landing inside the same noisy attractor band EXP 2's 50k-step run
oscillates in regardless of starting point. **Branch:** "nothing closes
the gap" - the harder, more interesting finding. Stated plainly, as
instructed.

**Full synthesis:** the working hypothesis as originally stated (badly-
conditioned multiplicative S4 factorization, full stop) is not quite
right, and EXP 1 is why: a plain, single, already-full-rank linear layer
fails by a comparable order of magnitude under the same optimizer, so
factorization is not *necessary* to produce a large gap. But EXP 3 shows
the factorization adds something a plain linear layer structurally
cannot have - an exact, dimension-matched null space - and EXP 4's (b)
shows that null space is not benign: it actively destabilizes an exact
optimum under continued training, which a full-rank quadratic bowl
(the plain-linear-layer case) would not do. The most defensible complete
picture is two compounding effects, not one:
1. A generic Adam/data-scale sensitivity on this case-3 regression task
   that affects even the minimal, fully-identifiable linear map (EXP 1).
2. An S4-specific, exact multiplicative-gauge non-identifiability that
   only the factored parameterization has, which turns "slow" into
   "actively unstable at the optimum" (EXP 3 + EXP 4b).
Neither alone explains all four results; both together do.

**M3-as-control confound, flagged per instruction:** M3 was intended as
the "capacity is fine" control for the M3/M4/M5/M6 variant ladder - the
LTI cell the other variants' norm/activation/glu ablations are compared
against, on the assumption that M3 itself sits near its achievable
floor. It does not: at the identification budget used everywhere else in
this project (2000 epochs, lr=1e-3, wd=0), M3 sits ~9-13 orders of
magnitude above the least-squares floor it can provably represent. Every
M3-vs-M4/M5/M6 comparison run so far (including the controller smoke
test) is implicitly comparing "M4/M5/M6 against an M3 that is itself
badly under-converged," not against M3's true capacity ceiling. **Whether
EXP 4's fix should be applied to all variants before the ladder is
run:** no fix found here closes the gap (EXP 4's conclusion above), so
there is not yet a fix to propagate - applying (b) or (c) to M4/M5/M6
would not resolve this confound, only relocate it. What the ladder needs
before it's trustworthy is either (i) a training budget/optimizer change
that closes M3's gap for real (not yet found), or (ii) explicitly
reporting every variant's nmse *relative to its own variant-specific
least-squares-reachable floor* rather than assuming M3 defines that
floor for the whole ladder - (ii) is the low-risk option available today
and doesn't require a another open-ended optimization investigation.

---

**Recommendation:** don't chase a fix for M3's gap this week - EXP 4
already tried the two most standard fixes (better init, cruder/higher-lr
optimizer) and neither worked, and further reparameterization search is
exactly the kind of open-ended detour CLAUDE.md sec 11 warns against on
a 7-day timeline. Instead: (1) keep this as a reported, documented
finding rather than a solved bug - "even the LTI control cell is far
from its achievable floor under the standard optimizer, and that failure
has two identified, compounding causes" is a legitimate, checkable paper
result in its own right, not a loose end; (2) when the M3-M6 ladder is
run for the paper, report every variant's nmse normalized against its
own fit_least_squares floor (M0/M1), not against an assumed-converged
M3, so the norm/activation/glu comparisons stay valid regardless of how
far any one variant is from its own ceiling; (3) the controller smoke
test already run (M3 vs M6 through the learned surrogate, true-plant
cost 895.8 vs 878.7) should not be over-read as a real M3-vs-M6 signal
yet, precisely because of this confound - both identification runs
feeding it were similarly under-converged (teacher_mse 3.6e-3 and
3.3e-3), so the controller comparison inherits the same normalization
problem before it says anything about the LayerNorm/derivative-fidelity
hypothesis.
