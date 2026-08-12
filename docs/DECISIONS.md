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

**EXP 1 was a false alarm, and its investigation surfaced a real bug in
EXP 3's number - both resolved here.** `nnx.Linear(9,6)` under MSE loss
IS least squares: convex, unique optimum, nothing to tune. Two closed
-form-adjacent checks (`tools/diagnose_m3_exp1_resolve.py`, standardized,
Kaggle CPU):

1. `||W_adam - W_LS||_F / ||W_LS||_F`, tracked every 400 steps: `1.322 ->
   0.994 -> 0.910 -> 0.875 -> 0.851 -> 0.831` at step 2000 - monotonically
   shrinking, i.e. moving toward the correct unique optimum the whole
   time, just slowly.
2. The same convex objective refit with a quasi-Newton method: `optax.
   lbfgs` triggered a reproducible XLA/LLVM out-of-memory compilation
   failure on the Kaggle CPU kernel (5 duplicate "Cannot allocate memory"
   errors from parallel compile workers; not a catchable Python
   exception, so a try/except around it doesn't help - switched to
   scipy's L-BFGS-B, same method family, no optax compilation path).
   Converged in 46 iterations to `loss=2.93e-8` - about 5 orders of
   magnitude closer to the `2.12e-15` standardized-LS floor than 2000
   Adam steps (`7.03e-3`), using 43x fewer iterations.

**Verdict: EXP 1 was a false alarm.** The branch rule that treated 2000
Adam steps as decisive for a convex objective was wrong to do so - a
convex problem under a slow first-order optimizer at too small a budget
is not evidence of anything except optimizer choice/speed. No fix
needed, no harness bug, nothing to learn from tuning Adam further here.

**While investigating EXP 1, a real bug was found in EXP 3's reported
rank.** D-only's forward pass (encoder -> elementwise D-scale -> out ->
residual add -> decoder) is a composition of exclusively affine
operations, so its output is *exactly* `inputs @ W_eff + b_eff` for some
`W_eff` (9,6) and `b_eff` (6,) - **60 effective degrees of freedom,
globally, regardless of how the 550 raw parameters vary.** Verified two
ways (`tools/diagnose_m3_rank_sanity.py`): analytically from the code
structure, and empirically (`|y(a*u+(1-a)*v) - (a*y(u)+(1-a)*y(v))| =
4.8e-7`, zero up to float32 noise). Since predictions over any batch are
a fixed linear embedding of `(W_eff, b_eff)`, the Gauss-Newton Jacobian's
rank is bounded by 60 everywhere - a hard linear-algebra fact, not an
approximation. EXP 3 reported rank 518 (null 32). Printing the *full*
550-value singular spectrum immediately showed why: 60 "genuine" values
(322.6 down to 0.49) followed by an abrupt ~5-order-of-magnitude drop to
~8e-6 at index 60 - exactly the predicted cliff - but that band (8e-6
down to 4.6e-8, indices 60-517) is NOT machine-precision zero, it's
sitting right where float32 rounding noise would land relative to a
~322 top singular value (~1e-7 relative precision). EXP 3's Jacobian was
computed in JAX's default float32 and only cast to float64 *after* the
fact - the differentiation itself never ran in float64.

**Confirmed by forcing true float64 throughout**
(`tools/diagnose_m3_rank_x64.py`, `jax.config.update("jax_enable_x64",
True)` before any JAX op, not just an output cast): index 59 = 0.488,
index 60 = **1.73e-13** - a genuine machine-epsilon cliff, not a noise
band. Numerical rank = **60 exactly**. Null dimension = **490**, not 32.

**"32 = 2 x d_model" was a coincidence of where float32 noise happened
to sit relative to the rank tolerance, not a real mathematical
signature.** The qualitative finding survives and is if anything
cleaner: D-only's 550 raw parameters compute a function with only 60
true degrees of freedom - *exactly* the same effective dimensionality as
EXP 1's plain `nnx.Linear(9,6)`. D-only and the plain linear layer
compute the identical function class; D-only just reaches it through 550
redundant coordinates instead of 60 exact ones. The `D_i -> c*D_i,
out_kernel[i,:] -> out_kernel[i,:]/c` symmetry derived and verified
earlier is still exactly correct algebraically (it doesn't depend on
floating point at all) and is a genuine subset of the true null space -
it just accounts for 16 of the real 490 dimensions, not 32 of a false
518. The remaining ~474 dimensions are not yet characterized; the most
likely source (not yet verified) is generic deep-linear-network
redundancy from composing several affine layers to represent one small
affine map (e.g. `out_kernel`/`decoder_kernel` only ever appearing as a
product in the D-only forward pass, discussed but not yet confirmed
below), which would make this a generic overparameterization phenomenon
rather than something specific to S4's `Lambda_re`/`log_step`
parameterization choices - weakening, not strengthening, the
S4-specific framing of the original working hypothesis. Every downstream
Part B/C check below uses the corrected 490-dimensional null space, not
the earlier 32-dimensional one.

---

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

---

## 2026-08-12 — Part B (gauge symmetry verified directly) and a float64 recheck of EXP 4(b): both survive, the divergence is if anything sharper

Direct verification of the 490-dim null space (corrected above), plus a
float64 recheck of whether EXP 4(b)'s "diverges 8 orders of magnitude
from the exact optimum" finding was itself a float32 artifact, the same
way EXP 3's rank number was.

**B.1 - transform invariance** (`tools/diagnose_m3_gauge_verify.py`,
trained D-only model, c=2.3):

| candidate transform | max abs output diff | holds? |
|---|---|---|
| `D_i -> c*D_i`, encoder column `i -> /c` (the working hypothesis's own proposal) | 2.52 (relative 0.38) | NO |
| `D_i -> c*D_i`, `out_kernel[i,:] -> /c` (derived in the entry above) | 2.67e-15 (relative 4.0e-16) | YES, to machine precision |

The proposed encoder-based pairing does not hold - it breaks the skip
connection (`skip = encoder(x)` feeds the decoder directly, untouched by
`D` or `out_kernel`, so scaling the encoder column alone changes `skip`
with nothing to compensate). Worth stating plainly since it was offered
as the working hypothesis: the analytically-derived pairing is the real
one, not this one.

**B.2 - null-space alignment**: for all 16 channels, the analytic
generator of the confirmed `(D_i, out_kernel[i,:])` symmetry has cosine
similarity **1.000000** (exactly, to printed precision) with its own
projection onto the empirically-measured 490-dim null space. This
symmetry family is a real, verified subset of the true null space - not
merely algebraically plausible, but empirically confirmed to be exactly
tangent to the loss surface's flat directions.

**B.3 - rank scan across d_model** (`tools/diagnose_m3_gauge_rank_scan.py`,
freshly-initialized models, x64 throughout):

| d_model | n_params | rank | null_dim | predicted (n_params - 60) | match |
|---|---|---|---|---|---|
| 8 | 214 | 60 | 154 | 154 | YES |
| 16 | 550 | 60 | 490 | 490 | YES |
| 32 | 1606 | 60 | 1546 | 1546 | YES |
| 64 | 5254 | 60 | 5194 | 5194 | YES |

Rank is exactly 60 at every d_model tested - as clean a structural
confirmation as this kind of check produces. D-only's redundancy is
generic overparameterization (a fixed 60-dof affine map, however many
raw parameters implement it), not a d_model-scaled symmetry.

**B.4 - does the full S4 conv path add or remove redundancy?**
(`tools/diagnose_m3_gauge_full_m3.py`). Unlike D-only, full M3 has no
clean closed-form DOF bound - the S4 kernel makes each channel a causal
linear filter over the L=100 sequence via N=32 exponential modes, so
this was measured, not predicted. First attempt crashed: a single
L=100 trajectory gives only 600 output samples against 3638 raw
parameters, so the Jacobian's rank was trivially capped at 600
regardless of any real structure (caught by an IndexError when the
rank-cliff printer assumed more headroom than the singular-value array
actually had - not a subtle bug, a batch-too-small one). Fixed by
stacking 8 independent trajectories (4800 output samples > 3638
params):

```
D-only:  rank=60,   null=490   (of 550 params, 89.1% redundant)
full M3: rank=1183, null=2455  (of 3638 params, 67.5% redundant)
```

**Caveat that matters:** D-only's 60/490 split is a sharp, provable
cliff (idx 59 = 0.49, idx 60 = 1.7e-13, one index step, exact machine
zero). Full M3's spectrum has no such cliff - it decays *smoothly*
around the rank boundary (idx 1173: 4.9e-9 -> idx 1192: 2.8e-10 over 19
indices, with true machine-zero only reached around idx ~3600 at
~1e-20). So "rank=1183" is a reasonable, tolerance-based estimate of
where the S4 kernel's genuinely-informative directions give way to
increasingly-negligible ones (consistent with a bank of exponential
modes at a continuum of decay rates - some fast, most too slow or too
fast to matter over L=100 steps), not a hard structural fact the way
D-only's 60 is. Qualitative answer to B.4's question: the conv path
adds real expressive capacity (1183 effective directions vs D-only's
60, roughly 19x) while *also* remaining substantially redundant in
relative terms (67.5% vs 89.1% - better, not solved).

**Float64 recheck of EXP 4(b)** (`tools/diagnose_m3_exp4_x64.py` - this
file and its Kaggle kernel existed on disk, written but never run or
pushed; found while auditing repo state for this write-up, run now
rather than left stale). Same D-only/LS-init construction, x64
throughout (including the same `nnx.Linear` `param_dtype=float32`-by-
default gotcha `diagnose_m3_rank_x64.py` needed - the LS-init leaves
must be constructed as float64 directly, not inherited from the model's
pre-existing float32 leaf, or the "float64" run silently trains in
float32 anyway). Three variants, MSE logged every step:

```
LS floor (float64): mse=4.746674e-30
init (LS-init, self-checked): mse=4.775484e-30

adam_lr1e-3 (the original EXP4(b) setup):
  final=1.053126e-07   orders of magnitude rise: +22.34
  first step exceeding 1e8x init: step 1

sgd_lr1e-3 (same nominal step size, no momentum/adaptive scaling):
  final=6.162454e-31   orders of magnitude rise: -0.89 (went DOWN)
  first step exceeding 10x init: never (over all 2000 steps)

adam_lr1e-5 (100x smaller step):
  final=2.422722e-11   orders of magnitude rise: +18.71
  first step exceeding 1e8x init: step 1
```

**This is not a float32 artifact - it is real, and float64 makes it
look worse, not better.** Starting from a point 16 orders of magnitude
closer to the true optimum than the float32 measurement could even
represent (1e-30 vs float32's representable floor of ~1e-14), Adam
still explodes past 1e8x the initial loss in a single step, landing at a
final MSE of a similar order to the original float32 finding (1.05e-7
here vs 1.89e-6 there). Plain SGD at the identical nominal learning rate
does not move - it stays at or below the initial loss for the entire
run. Cutting Adam's learning rate 100x (`adam_lr1e-5`) does not fix it
either - still an 18.71-order-of-magnitude, single-step blowup. That
last result is the sharp one: this rules out "Adam's step size is too
large" as the mechanism. Adam's update is *not* proportional to the
nominal learning rate here - it is dominated by its own per-coordinate
normalization (dividing by a second-moment estimate built from the same
near-zero, noise-level gradient), which produces a roughly
`lr`-scaled step regardless of how small the true gradient is, as long
as it is not exactly zero. SGD's step, being directly proportional to
the raw gradient magnitude, stays correspondingly tiny. This is now a
specific, mechanistic, float64-verified claim about *why* Adam
destabilizes the exact optimum: its adaptive normalization amplifies
whatever machine-precision noise exists along the null space's flat
directions into a full-sized step, independent of the nominal learning
rate.

**Part C - the displacement decomposition, and it is exactly the
predicted shape** (`tools/diagnose_m3_divergence_figure.py`, corrected
to use the true 490-dim null space and to actually run in float64 - the
first version of this script hardcoded `dtype=jnp.float32` when
constructing the LS-init model despite `jax_enable_x64=True`, the same
bug the exp4x64 script above had already found and fixed; caught before
running, not after). `docs/m3_divergence_figure1.png`:

```
Adam, step 1999: loss=1.008e-07  ||disp_null||=2.965e-3  ||disp_orth||=1.241e-4
SGD,  step 1999: loss=5.946e-31  ||disp_null||=2.840e-16 ||disp_orth||=1.246e-15
```

(Adam's final loss, 1.008e-7, matches the independently-run exp4x64
script's 1.053e-7 to within 5% - good agreement between two separate
implementations of the same experiment.)

**Is the rise monotonic?** No, and the shape is informative. Loss jumps
immediately (step 0 to step ~20: floor to ~1e-6), dips back down toward
~1e-11 by step ~250, then settles into a **noisy, non-decaying
oscillation** between roughly 1e-9 and 1e-5 for the remaining 1750
steps - visually identical in character to Experiment 1's 50k-step
D-only run from the earlier investigation (`docs/m3_d_only_50k_loglog.png`),
just reached in far fewer steps because this run starts exactly at the
optimum instead of a random init.

**The displacement decomposition is the clean part.** `||disp_null||`
grows **steadily and close to monotonically** the entire run (1.8e-3 at
step 20 -> 2.97e-3 at step 1999 - keeps climbing, no sign of leveling
off within 2000 steps). `||disp_orth||` does the opposite: it
**oscillates without a clear trend**, repeatedly returning to small
values (e.g. 3.4e-5 at step 780, 3.4e-5 at step 1300) and back up again
(e.g. 5.2e-4 at step 700), tracking the loss's own oscillation almost
exactly (small `disp_orth` <-> small loss, at matching steps). This is
*precisely* the predicted mechanism: motion along the null space is (to
first order) exactly flat, so Adam's normalized per-coordinate updates
push the parameters along it with no restoring force - an unbounded
random walk, visible directly in the steadily-climbing solid blue line.
But the null space is only flat *locally* (it is the tangent space of
the true, curved level set at the LS-init point) - a large enough
excursion along it leaks into the orthogonal, function-relevant
directions via second-order curvature, which is exactly what shows up
as `disp_orth`'s noisy but bounded fluctuation and the loss's matching
oscillation. Displacement magnitude, not first-order gradient, is what
governs how much the function actually changes.

**SGD is the control that makes this a mechanistic claim rather than a
correlation.** Both displacement components stay 13-15 orders of
magnitude smaller than Adam's for the entire 2000 steps (`disp_null`:
2.8e-16 vs 3.0e-3; `disp_orth`: 1.2e-15 vs 1.2e-4), and the loss never
leaves the ~1e-30 floor. Same nominal step size, same gradients, same
loss landscape - the only difference is Adam's per-coordinate
normalization, and removing it removes the entire effect. This is the
clean, specific claim about Adam requested at the top of this chain.

---

## Full synthesis and read on the paper's central claim

**The chain, in order:** EXP 1 (single linear layer, "should reach
~1e-12") turned out to be a false alarm - a convex problem under too
short an Adam budget, resolved by a shrinking-error trajectory and a
scipy L-BFGS-B refit that closed 5 of the remaining orders of magnitude
in 46 iterations. Investigating it surfaced a real bug: EXP 3's
headline "rank 518, null 32 = 2 x d_model" was float32 rounding noise,
not a mathematical fact - the true, float64-verified split is rank=60,
null=490, and "60" is exactly D-only's effective degrees of freedom as
an affine map (matching a plain linear layer's function class exactly).
Part B then verified this directly rather than trusting the algebra:
the specific `(D_i, out_kernel row i)` symmetry holds to machine
precision and its generators span the null space exactly (cos_sim =
1.000000, all 16 channels); the rank is exactly 60 at every d_model
tested (8/16/32/64); the full S4 conv path is less redundant in
relative terms (67.5% vs D-only's 89.1%) but still substantially so,
with a smoothly-decaying spectrum rather than D-only's sharp cliff.
Separately, a float64 recheck of EXP 4(b)'s "diverges 8 orders from the
exact optimum" finding - the most surprising number in the whole
investigation, and therefore the most important one to re-verify before
trusting - showed the effect is not a float32 artifact: it is *sharper*
under float64 (22 orders of magnitude in a single step, from a point 16
orders closer to the true optimum than float32 could even represent),
survives a 100x smaller learning rate, and vanishes almost completely
under plain SGD. Part C's displacement decomposition explains why:
Adam's per-coordinate normalization drives an unbounded random walk
along the exact 490-dimensional null space, and large enough excursions
along that walk leak into real function-space error through second-order
curvature - visible directly as the correlation between `disp_orth` and
the loss's own noisy oscillation.

**Does this reframe the paper's central claim?** Partially, and in a
way that should be reported as a strength, not walked back.

The original working hypothesis (S4's specific parameterization -
`Lambda_re` clipping, `log_step` range - creates pathological
conditioning for a near-instantaneous linear map) is **not quite what
the evidence supports**. EXP 1 showed a plain, unfactored linear layer
also converges slowly under the same optimizer/budget - so *some*
generic Adam/data-scale sensitivity is present independent of S4's
factorization. But the S4-specific part is real and now precisely
characterized: D-only's parameterization has an exact, verified
non-identifiability (not merely bad conditioning) that a plain linear
layer structurally cannot have, and that non-identifiability is what
turns "slow" into "actively unstable at the exact optimum" - a
qualitatively different and more interesting failure mode than either
"Adam is slow" or "S4 is badly conditioned" alone.

**The mechanism, stated as precisely as the evidence now supports:**
D-only's parameterization is overcomplete (60 true degrees of freedom
realized through 550 raw parameters, most channel-pairs of which admit
an exact continuous rescaling symmetry). Adam's per-coordinate adaptive
normalization does not merely tolerate this redundancy - it actively
random-walks through it, because normalization by a near-zero gradient's
own magnitude produces a step of roughly fixed size regardless of how
small the true (first-order) signal is. That walk is invisible to the
loss at first order (the null space is exactly flat there) but not at
second order, once the accumulated displacement is large enough for the
parameterization's genuine nonlinearity (products like `D_i *
out_kernel[i,j]`) to leak into the represented function. This is a
specific, falsifiable, now twice-independently-confirmed (exp4x64 and
Part C, different scripts, matching final-loss numbers) claim about an
interaction between adaptive optimizers and overparameterized
multiplicative factorizations - not a vague "optimization is hard"
statement, and not specific to S4's long-range-sequence-modelling
tuning choices the original hypothesis blamed. Full M3 (Part B.4) is
somewhat less redundant than D-only in relative terms, which is
consistent with - though does not by itself prove - the S4 recurrence
providing some genuine escape from this trap as more of the
parameterization becomes load-bearing; that is a good next question,
not yet answered.

**Recommendation, updated:** this is a stronger paper result than the
version reported two entries above, precisely because the chain now
distinguishes a generic Adam-on-overparameterized-linear-maps effect
(plausibly known in the deep-linear-network literature, worth a
citation check before claiming novelty) from what appears genuinely
tied to this specific class of factorization. Two concrete next steps
worth prioritizing over further open-ended investigation: (1) check
whether this exact mechanism (adaptive-optimizer random walk through an
exact null space, second-order leakage into loss) has prior art in the
deep linear network / overparameterization literature, since if so the
contribution is "this happens in S4 specifically and here is why it
matters for control," not "this happens" per se; (2) the M3-vs-M6
ladder should still be run with the per-variant floor normalization
recommended two entries above, but now with an added expectation worth
testing directly: if M6's LayerNorm makes its own effective
parameterization *more* overcomplete (plausible, given LayerNorm adds
scale-invariance on top of whatever redundancy the linear layers
already have), the same Adam-random-walk mechanism found here for
D-only predicts M6 should be *more* susceptible to this failure mode
than M3, not less - a testable, specific corollary of this entry's
finding, and a much sharper version of the original LayerNorm hypothesis
than "LayerNorm distorts the Jacobian" alone. Controller work remains
on hold pending explicit go-ahead, per standing instruction.

---

## 2026-08-12 — float64 is now the default for identification; ALL prior teacher_mse numbers are superseded

Per the finding above (EXP4(b)'s divergence survives, and is sharper, in
float64; EXP3's rank number was a float32 artifact), float32 is not
trustworthy for this project's relevant loss scales. `s4dpc/sweep.py` now
sets `jax.config.update("jax_enable_x64", True)` before any other JAX
import or op (peeked from `sys.argv` at module import time, since the
flag must precede any JAX op and therefore can't wait for argparse to
run) - default on, `--no-x64` to opt out. Every CSV row now stamps an
`x64` column.

**Setting the flag alone is not sufficient.** `nnx.Linear`/`nnx.LayerNorm`
construct their own params at float32 by default *regardless of the
global flag* (first found in `tools/diagnose_m3_rank_x64.py` - see
2026-08-07 entries above) - `s4dpc/identify.py._build_model` now casts
the model's whole trainable param tree to match the global flag
immediately after construction (`_cast_params`, a no-op when x64 is
off, so float32 runs are byte-for-byte unaffected).

**That cast has to be complex-aware.** `tools/probe_s4nnx_dtype.py`
(Kaggle CPU) found S4LayerEnsemble does NOT accept a `param_dtype` kwarg,
but its own params already follow the global flag correctly on their
own (unlike Linear/LayerNorm) - *except* `P` and `B`, which are
genuinely complex (`complex128` under x64, confirmed empirically). A
blanket `.astype(float64)` over the whole tree would silently discard
their imaginary part instead of widening precision - `_cast_params`
casts complex and real leaves to their respective same-kind dtype.

**A second, independent pipeline bug found and fixed while wiring this
up:** M6_fix's `StaticNorm` holds `mu`/`sigma` as `nnx.Variable`, not
`nnx.Param` (deliberate - `nnx.Optimizer(..., wrt=nnx.Param)` must never
touch them, see `s4dpc/blocks.py`). `nnx.split(model, nnx.Param)` with a
single, non-exhaustive filter raises `ValueError: Non-exhaustive
filters...` the instant any such leftover state exists - `nnx.state(x,
nnx.Param)` alone tolerates it silently, which is why this went
unnoticed until the variant-redundancy work below became the first
script in this repo to call `nnx.split` (not just `nnx.state`) on a
model that can be M6_fix. Fixed in `identify.py`'s `_train_ensemble`
(`nnx.split(model, nnx.Param, ...)`, threading the resulting `rest`
state back through `nnx.merge`) - this would have broken the real
M6_fix identification run the first time anyone tried it, independent
of anything else in this entry. **M6_fix has never successfully
completed an identification run before this fix landed** - treat any
number for it from before this date as untrustworthy/nonexistent, not
merely float32.

**M3/M6 smoke test, re-run paired float32-vs-float64** (`launch/kaggle-
smoke-x64`, CLAUDE.md's own quick-start config: case 3, 1 seed, 2000
epochs, d_model=16, N=32, n_layers=1, wd=0 - four separate `python -m
s4dpc.sweep` process invocations, since `jax_enable_x64` is decided once
per process and can't be toggled mid-run):

```
variant  x64    teacher_mse
M3       False  4.822968e-03
M3       True   4.385848e-03   (ratio 0.909)
M6       False  3.144979e-03
M6       True   2.927023e-03   (ratio 0.931)
```

**The numbers moved (~7-9%), but don't read this as "float64 identifies
more accurately" at this budget.** Every entry above establishes that
2000 Adam steps at lr=1e-3 leaves M3 (and, by the corollary below, every
variant) nowhere near its own least-squares floor - EXP2's 50k-step
D-only run oscillated non-monotonically in a noisy attractor band for
40k of those steps, never converging closer to the floor regardless of
budget. A 2000-step run in either precision is sampling a point inside
that same kind of noisy, non-convergent regime, not approaching a
floor - float32 vs float64 arithmetic produces a different rounding
trajectory through that regime, which plausibly explains a ~10% shift
with no implication that either number is "more correct" in an
absolute sense. The float64 value that DOES matter is the one already
established above: EXP4(b)'s instability is real, not float32 noise,
and is *sharper* in float64. Every teacher_mse number reported anywhere
above this line (all of it float32) is superseded by this flag flip and
should not be compared against any post-2026-08-12 number without
re-running it.

**How to apply:** any new identification run is float64 unless
`--no-x64` is passed explicitly. Do not compare a pre-2026-08-12
teacher_mse (or any number derived from one, e.g. the controller smoke
test's 895.8/878.7) against a post-flip number without re-running the
older side in float64 first.

---

## 2026-08-12 — Task 3: `s4dpc/diagnostics.py` built and validated against ground truth

Four functions, operating on an already-constructed `decode=True`
`StackedModel` with trained params loaded in (identify.py trains
`decode=False`; deploying/analyzing one input at a time needs
`decode=True`, same params - `tests/test_control_decode_parity.py`
already verifies the two modes agree on trained, not just initialized,
params):

- `markov_parameters`: `G_h = d x_{k+h} / d u_k`, h=1..H, of the
  linearized AUGMENTED (physical state + S4 hidden state) system.
  Computed by autodiff (`jacfwd`, cheap - cost scales with `d_u`=3, not H
  or model size) through an H-step free-running unroll w.r.t. `u_0` (x
  feeds back as the next step's input, like `control.py`'s
  `rollout_learned` - NOT the teacher-forced one-step map identify.py
  trains on). This is *realization-invariant* - the chain rule through
  the unroll composes each step's local linearization exactly the way
  hand-linearizing the augmented system would, so it reflects the
  learned input-output map regardless of the surrogate's own (generally
  non-`A_d`-shaped) internal realization. A raw one-step `d x_{k+1}/d
  x_k` Jacobian would NOT have this property - it only sees one step and
  ignores whatever the hidden state carries forward.
- `equilibrium_drift`: `|F(0,0,s)|`.
- `local_linearity_defect`: `E|F(z+d)-F(z)-J(z)d|/|d|`, Monte Carlo over
  random small `d` - the diagnostic the "kink" hypothesis predicts should
  be sharply elevated for LayerNorm'd variants near `x=0`.
- `jacobian_sweep`: `J(t*direction)` for a sweep of `t` through 0 - the
  diagnostic behind the original kink figure.

**Validated against ground truth before trusting on any real surrogate**
(`tools/validate_diagnostics.py`, Kaggle CPU): forced an M6 model's
one-step map to be EXACTLY `x_next = A_d@x + B_d@u` by routing the TRUE
`[A_d|B_d]` through the skip connection (same block-zeroing construction
as the variant-redundancy work below - zero `D` and `C_real_imag`
together, plus `out`/`out2` - generalized from "the fitted LS solution"
to "the true system" so every diagnostic has a known-exact answer, not
an approximately-fit one). Chose M6 specifically (not M3) so the
validation also confirms the zeroing trick survives LayerNorm/GELU/GLU
being architecturally present, not just architecturally absent as in M3.

```
self-check (model forward vs A_d@x+B_d@u directly): max abs diff=5.6e-17
equilibrium_drift: max abs = 0.0e+00                        (expect ~0)
markov_parameters: max error over h=1..50: 2.1e-17           (expect ~1e-9 or better)
local_linearity_defect (at x=0,u=0): 0.0e+00                 (expect ~1e-8 or better)
jacobian_sweep: max|J(t) - A_d| over sweep = 0.0e+00          (expect ~1e-9 or better)
```

All four land at or near machine precision - `markov_parameters`
reproduces `A_d^(h-1)@B_d` for every h from 1 to 50, not just small h,
which is the real test (errors from an autodiff/indexing bug would
typically compound or drift with h; they don't).

**Sanity control, so the PASS above isn't vacuous:** the same four
diagnostics run on a fresh, untrained, genuinely nonlinear M6 model
(block NOT zeroed) give `equilibrium_drift=0` (expected and correct,
not a bug - zero-initialized biases plus origin-preserving nonlinearities
[GELU(0)=0, LayerNorm(0)=0 with its epsilon, a zero-signal convolution
is 0] make x=0 a fixed point of a freshly-initialized network generically,
independent of the random kernel values), but `local_linearity_defect=
168.5` and a `jacobian_sweep` that varies with t (norm range 1.8 to
509) and stays finite throughout - confirming the functions detect real
curvature when it's actually there, rather than always returning ~0
regardless of input.

**Not yet done:** running these on an actually-trained surrogate and
correlating against per-case behavior - blocked on the variant-ladder
identification run (Task 2/the original variant-ladder work) landing
first, so there are trained checkpoints worth diagnosing.

---

## 2026-08-12 — Task 2: the redundancy ordering does NOT support the LayerNorm corollary, and the instability test as designed cannot speak to norm at all (construction artifact, caught and explained, not swept under)

Tested the corollary: if LayerNorm makes a variant's parameterization
more overcomplete than M3's, the same Adam-null-space mechanism found
for D-only predicts M6 should be more unstable, not just M3.
`tools/diagnose_variant_redundancy.py`, case 3, float64, all 5 variants
(`launch/kaggle-variant-redundancy`, Kaggle T4, 60.93 min - see below for
a first attempt that failed on two real bugs before this).

```
variant  n_params  rank(rand)  null%(rand)  rank(LS-init)  null%(LS-init)  adam_orders  sgd_orders
M3         4662        929        80.1%          60            98.7%         +21.98       +18.74
M4         4934       1451        70.6%          60            98.8%         +22.12       +18.67
M5         4694        964        79.5%          60            98.7%         +21.98       +18.74
M6         4966       1476        70.3%          60            98.8%         +22.12       +18.67
M6_fix     4934       1451        70.6%          60            98.8%         +22.12       +18.67
```

**Part (a), redundancy at a random init - the corollary's first half
fails outright.** M5 (79.5%) is very slightly LESS redundant than M3
(80.1%); M6 (70.3%) is very slightly LESS redundant than M4 (70.6%) -
the opposite direction of "LayerNorm makes it more overcomplete."
LayerNorm adds exactly 2×d_model=32 new params (scale, bias) that are
generically full-rank contributors at a random point, which mechanically
nudges the overall % redundant down a little, not up. M6_fix (uncalibrated
StaticNorm ≡ identity, `s4dpc/blocks.py` - never calibrated anywhere in
this pipeline) is bit-for-bit identical to M4 on every column, exactly
as the architectural equivalence predicts - not a new finding, a
consistency check passing.

**Part (b)/(c)/(d) - a real problem with the experiment design, not a
result:** M3 and M5's `adam_orders`/`sgd_orders`/`adam_final_mse`/
`sgd_final_mse`/both displacement columns are bit-for-bit IDENTICAL
(not just close - identical to every printed digit in the raw CSV). So
are M4, M6, and M6_fix, all three, with each other. This is not noise or
a coincidence; it is a structural consequence of the LS-init construction
itself (the same one used for part (a)'s rank at the LS-init point and
for `tools/validate_diagnostics.py` above): zeroing `out.kernel` (and
`out2.kernel` for GLU variants) - needed to force the block's output to
exactly zero for the LS solution to route through the skip connection
cleanly - has a side effect nothing in the derivation above flagged:
**it makes the Gauss-Newton gradient of every parameter upstream of `out`
(`D`, `C_real_imag`, `Lambda_re`, `Lambda_im`, `P`, `B`, `log_step`, AND
LayerNorm's scale/bias) exactly zero, and this stays true for the entire
training run, not just at step 0.**

Why it persists, not just holds at init: `seq_output` (the S4Layer's raw
output, pre-activation) is `conv(norm(skip), kernel-derived-from-C) +
D*norm(skip)`, and with `C=0, D=0` this is exactly `0` for *any* value
`norm(skip)` takes - so `d(out.kernel)/d(loss)` is proportional to
`activation(seq_output) = activation(0) = 0` (GELU(0)=0, or just 0
directly for M3/M5's no-activation case) at every step, not only step 0,
*as long as `out.kernel` itself never moves* - which Adam/SGD guarantee
for an exactly-zero-gradient parameter (zero grad -> zero moment
estimates -> zero update, every step, by induction). `out.kernel=0`
therefore isn't just the init condition, it's an exact fixed point that
gates the entire S4-kernel-and-norm sub-path off from training
permanently. What's left doing all the work is `out.bias`, `out2.bias`
(GLU only), `encoder`, `decoder` - a small, LayerNorm-independent
sub-problem, which is why norm choice cannot possibly show up in
columns computed from this trajectory. The mechanism itself (Adam's
per-coordinate normalization random-walking an exact null space, ~22
orders in one step, SGD staying ~4 orders smaller) clearly generalizes
beyond D-only's specific factorization to this GLU-gated residual
sub-network too - a mildly interesting extension - but this experiment,
as built, cannot and does not test whether LayerNorm's presence changes
that answer.

**Verdict: the corollary is not confirmed by this experiment, and part
of it (redundancy) is actively contradicted.** Don't read "M3 ≡ M5 and
M4 ≡ M6 ≡ M6_fix" as "norm doesn't matter for stability" - it means
*this specific test* structurally cannot see norm at all, which is a
different and weaker claim. A valid dynamical test would need an
LS-init-style starting point where `out.kernel`/`out2.kernel` (and
therefore the norm/S4-kernel path) are NOT held at a permanent zero-
gradient fixed point - e.g. genuinely training each variant to a good
point first (L-BFGS, per the EXP1 resolution above, converges far faster
than Adam on these small problems and wouldn't inherit this specific
artifact) and testing Adam-vs-SGD stability from *that* point instead.
Not attempted here - a real piece of new infrastructure (nonlinear
optimization to a good point per variant, not a closed-form route
through zeroed params), and exactly the kind of open-ended detour
CLAUDE.md §11 warns against building without explicit go-ahead, flagged
for the same reason the D-only reparameterization search was capped
after two standard attempts rather than continued indefinitely.

**First attempt failed on two real, independent bugs, both fixed before
the run above (not swept aside - see the commit and the code comments):**
(1) `ravel_pytree` concatenates S4LayerEnsemble's genuinely-complex `P`/
`B` params with the real leaves, promoting the whole flat vector to
`complex128` - `jax.jacfwd` then refuses outright ("requires real-valued
inputs"). This is the SAME hazard `tools/diagnose_m3_gauge_full_m3.py`
(Part B.4, 2026-08-12 entry above) hit and got wrong *silently* instead
of crashing - its blanket `.astype(float64)` cast discards `P`/`B`'s
imaginary part rather than raising, so **B.4's reported "full M3:
rank=1183, null=2455" was computed with `P`/`B`'s imaginary parts
zeroed, not the real parameterization.** Not re-run/edited retroactively
(CLAUDE.md: diverged/wrong runs are results, tag don't delete) - superseded
by this entry's M3 row instead (`rank=929`, `null=3733`, `n_params=4662`
- note `n_params` is also higher than B.4's 3638 once P/B's imaginary
parts are counted correctly as real DOF, which B.4's silent cast also
undercounted). (2) M6_fix's `StaticNorm` holds `mu`/`sigma` as
`nnx.Variable`, not `nnx.Param` - `nnx.split(model, nnx.Param)` with a
single non-exhaustive filter raises `ValueError` the instant such
leftover state exists; `nnx.state(x, nnx.Param)` alone tolerates it
silently, which is why this went unnoticed until this script became the
first in the repo to call `nnx.split` (not just `nnx.state`) on a model
that can be M6_fix - also fixed in `identify.py`'s `_train_ensemble`
(2026-08-12 float64-default entry above), independent of this
investigation, since it would have broken the first real M6_fix
identification run regardless.

GPU cost: first (failed) attempt 1.04 T4-min, corrected re-run 60.93
T4-min - both logged in `gpu_ledger.csv`.

---

## 2026-08-12 — Corollary closed: LayerNorm/overparameterization thread ends here, dead

Reviewed the entry above with the actual numbers in front of us rather
than the direction predicted: M5 vs M3 is 79.5% vs 80.1% redundant at a
random init; M6 vs M4 is 70.3% vs 70.6%. LayerNorm does not meaningfully
change overparameterization in either pair - both differences are under
1 percentage point, and both go slightly the wrong way (LayerNorm's own
params are generically full-rank, so adding them can only ever nudge %
redundant down, never up). The premise the corollary needed - "LayerNorm
makes the parameterization more overcomplete" - is measured and flat.
There is no need to chase a corrected dynamical (Adam-vs-SGD) test to
settle the second half of the corollary, because the first half already
fails on its own terms.

**Verdict: the Adam/overparameterization finding (D-only, generalizes to
the GLU-gated residual sub-network - both entries above) and the
LayerNorm/kink hypothesis (CLAUDE.md sec 1's original claim) are two
separate mechanisms, not one chain.** Nothing links them. The
Adam-instability work stands on its own as a real, float64-confirmed,
mechanistically-explained result about optimizer choice under
overparameterized multiplicative factorizations - worth reporting as
exactly that, not as a cause of or explanation for whatever LayerNorm
turns out to be doing to the Jacobian. This thread is closed; the next
entries return to the original kink hypothesis directly, on trained
(not LS-init-constructed) checkpoints.

**For the record:** M6_fix's `StaticNorm` never completed a real
identification run before this session's `nnx.split` fix (entry above)
- every M6_fix number anywhere before 2026-08-12 is void, not merely
float32-imprecise. There is no valid pre-fix M6_fix baseline to compare
anything against.

---

## 2026-08-12 — First real result: M3 trained on real data shows EXACTLY zero kink; M6 shows a real one. The hypothesis survives on case 3.

Every prior number in this document came from either an under-converged
smoke test or an LS-init-constructed model built to sit exactly at (or
displaced from) a known optimum. Neither answers the actual question:
train M3 and M6 for real, on case 3, and look at the Jacobian.

**`tools/train_to_convergence.py`** (case 3, float64, wd=0, `@nnx.jit`
per-step - eager dispatch was fine at previous epoch counts but not at
this scale): trained until the loss curve is visibly flat (two
consecutive 5000-step chunks with <0.05 log10-unit change), not to a
fixed budget, capped at 200k. Both variants flattened at the SAME step
count:

```
variant  steps   teacher_mse   nmse       ratio_to_LS_floor   free_run_rmse(100 steps)
M3       35000   2.487e-05     3.906e-06  3.245e+24           5.245
M6       35000   7.639e-06     1.200e-06  9.966e+23           5.123
```

LS floor recomputed fresh: `7.665364e-30` (differs from an earlier
CPU-kernel recompute of the same deterministic quantity, `4.746674e-30`,
from the EXP4-x64 entry above - both are consistent with "float64
noise-floor territory" for a system this exactly-representable; read as
LAPACK-backend-dependent rounding at the ~1e-30 level, not a real
discrepancy, since `fit_least_squares` is pure numpy and its inputs are
bit-fixed).

**"Flat" is not "converged."** Both variants look good by nmse alone
(~1e-6) but sit **24 orders of magnitude** above the true LS floor -
direct, quantified confirmation of the whole chain above (Adam settles
into a noisy attractor band, not the optimum, regardless of how flat
the curve looks by a step-to-step criterion). Free-run RMSE (~5.2,
recursive rollout feeding the model's own prediction back, NOT
teacher-forced) is large relative to case 3's state scale (mean 1.8,
std 1.8) for BOTH variants roughly equally - caveat noted before reading
too much into this: case 3 is open-loop unstable, so some growth over
100 free-running steps is expected of any imperfect model, not
necessarily a discriminating signal between M3 and M6 on its own (their
free-run numbers are close: 5.245 vs 5.123).

**`tools/diagnose_trained_variants.py`** loaded both checkpoints and ran
all four `s4dpc/diagnostics.py` functions against case 3's true `(A_d,
B_d)`. (First attempt crashed on load - `nnx`'s pure-dict form encodes
list indices like `self.layers[0]` as literal int keys, which
`msgpack_serialize` writes fine but `msgpack_restore`'s
`strict_map_key` rejects on the way back in; `identify.py`'s own
`save_checkpoint` already guards against this with `_stringify_keys`,
`train_to_convergence.py`'s didn't. Fixed, re-ran - training itself had
already succeeded and was cheap enough that nothing was lost.)

```
metric                                   M3                M6
markov rel err (h=1)                  1.7114e-01        3.9700e+02
markov rel err (h=10)                 1.5114e+00        2.0533e+02
markov rel err (h=50)                 2.7687e+00        2.3116e+02
markov rel err (mean, h=1..50)        2.0897e+00        2.3255e+02
equilibrium_drift |F(0,0,s)|          1.0799e+00        2.9725e+00
local_linearity_defect (x=0)          1.3517e-14        3.6279e-01
KINK max||J(t)-J(0)||, dim 0          0.0000e+00        1.2136e+01
KINK max||J(t)-J(0)||, max over dims  0.0000e+00        1.3055e+01
```

**M3's kink is not "small" - it is exactly `0.0000e+00`, bit-for-bit,
across all 6 state dimensions.** This is not a coincidence or a
threshold call: `jacfwd` of a truly affine map (M3 has no
norm/activation/glu anywhere) is a mathematical constant independent of
the evaluation point, so JAX's forward-mode AD must return bit-identical
matrices regardless of `t` - the same computation graph, evaluated at
different primal points, for a linear operation whose Jacobian doesn't
depend on the primal at all. `local_linearity_defect` (1.35e-14, machine
precision) confirms the same thing a different way. This is as clean a
validation of `diagnostics.py`'s own correctness as the ground-truth
check in the entry above, from an entirely independent angle (a
genuinely-trained model rather than a hand-constructed one).

**M6's kink is real and substantial: ~12-13 in magnitude**, alongside a
local-linearity defect of 0.36 (vs M3's ~0) and a Markov relative error
two orders of magnitude worse than M3's (232x vs 2x mean relative
error). **The hypothesis survives on this case:** M3, which cannot have
a kink by construction, doesn't have one; M6, which has LayerNorm/GELU/
GLU, has a clear one.

**Honest caveat on the Markov/equilibrium numbers specifically (the
kink-strength number does not have this problem):** M6's Markov error
being worse than M3's is not yet cleanly attributable to nonlinearity
alone - both variants are, per the ratio-to-LS-floor finding above,
badly under-converged relative to the true optimum, and M6 has more
parameters and a harder optimization landscape, so some of its worse
Markov-parameter fit could reflect convergence quality rather than the
kink mechanism specifically. The kink-strength metric (`||J(t)-J(0)||`)
is immune to this confound by construction - it measures how much the
Jacobian moves away from its OWN value at the origin, independent of
whether that origin value is close to the true `A_d` - which is exactly
why it, not the Markov/equilibrium numbers, is the number to lead with.

**Scope, stated plainly:** one case, one seed, one architecture pair.
Proceeding to Task 3 (all 7 cases, M3/M6, 3 seeds) per instruction, now
that this has landed cleanly.

---

## 2026-08-12 — Task 3: the "+1.0000" Markov-error/Kreiss correlation is real for M6, an outlier artifact for M3 - reported both ways, not the headline number alone

All 7 cases, M3 and M6, 3 seeds, real training via the actual
`sweep.py`/`identify.py` path (not a bespoke script) for the first time
in this whole investigation: `python -m s4dpc.sweep --variant {M3,M6}
--cases 1,2,3,4,5,6,7 --n_seeds 3 --epochs 40000 --wandb off --out
results/all_cases/{variant}.csv`, then `tools/diagnose_all_cases.py`
against each case's own `(A_d, B_d)`.

**Prerequisite: `@nnx.jit` added to `_train_one`/`_train_ensemble`'s
per-step update** (eager dispatch was fine at every epoch count used
before this week; not at 40k epochs x a 21-way vmapped ensemble).
Changing shared, already-tested code warranted a check before committing
to an expensive run: `tools/pilot_jit_check.py` (small config, Kaggle
T4, 1.77 min) compared vmap vs `--no-vmap` final teacher_mse (mirrors
`tests/test_identify.py`'s own correctness check, which needs a local
jax install this environment doesn't have) - agreement at 1e-6 to 1e-12
relative precision for every case/seed tested, on both variants. (The
pilot's own naive "max_mse should be <1.0" sanity check failed for case
6 specifically - not a bug: case 6's Kreiss-like amplification is ~330x
everything else, so its raw, un-normalized state magnitudes over a
100-step trajectory are naturally huge regardless of fit quality. The
vmap/no-vmap agreement is the decisive check and it passed cleanly.)
Training itself: 110.98s (M3) + 125.56s (M6) for 40k epochs x 21
members each - confirms the jit fix works as intended, not just
correctly.

**The raw correlation, as first computed, looked like a clean paper
result:**

```
        corr(markov_err, rho)   corr(markov_err, kreiss_like)   corr(log,log)
M3            +0.267                    +1.0000                   +0.975
M6            +0.267                    +1.0000                   +0.988
```

**It is not that clean, and reporting only the number above would be
misleading.** Case 6 (kreiss_like=330.3, ~100x every other case) also
contains what is unmistakably a diverged training run, not a poorly-fit
one: M3 case6/seed2's Markov error is `1.178e+11` - eleven orders of
magnitude beyond every other (case, seed) point in the entire 42-run
sweep (the next-highest is M6 case6/seed2 at `3.775e+11`, same case,
different variant - both seed2, both case 6). A single point that is a
simultaneous extreme outlier on BOTH axes will drive a Pearson
correlation to ~1.0 almost regardless of what the other 6 points do -
this is a basic leverage-point problem, not evidence the relationship
is real across the range that actually varies smoothly.

**Recomputed four ways** (mean vs median across the 3 seeds per case -
median is far more robust to one diverged seed; all 7 cases vs excluding
case 6 - to see whether the relationship holds among the cases that
aren't off-scale outliers):

```
                                    M3                      M6
mean,  all 7 cases              +1.0000                  +1.0000
mean,  excluding case 6          -0.233                   +0.848
median, all 7 cases             +0.989                    +0.499
median, excluding case 6         -0.388                   +0.964
```

**M3: no relationship survives removing the leverage point** (-0.23 to
-0.39 excluding case 6, by either statistic) - case 4, this ladder's
second-highest Kreiss case (3.30), has one of M3's LOWEST median Markov
errors (1.19); case 2, near-lowest Kreiss (1.01), has a comparably-sized
error (1.74 median, 6.50 mean - itself seed-noisy, case2/seed2=16.8 vs
seed0/seed1 both <2). M3's fit quality looks dominated by per-run
optimization noise (consistent with everything established earlier in
this document about Adam not reliably reaching anywhere near the
achievable floor) rather than by the plant's own transient-amplification
structure, except at the true extreme (case 6) where it shows up as
outright divergence rather than a graded effect.

**M6: a real relationship survives removing the leverage point**
(median, excluding case 6: **+0.964**) - case 4 (kreiss=3.30, highest of
the remaining 6) has M6's highest median Markov error among them (461.9);
case 2 (kreiss=1.01, lowest) has the lowest (40.7). This is not the
outlier-driven artifact the raw all-cases number is - it holds among
cases whose Kreiss values differ by only a factor of ~3, not the ~100x
case 6 introduces.

**Read together with Task 1/2's kink finding, a coherent (not yet
proven, but internally consistent) picture:** M3's errors look
optimization-noise-dominated with an outright-divergence tail at the
extreme; M6's errors track the plant's own transient-amplification
structure more smoothly, on top of the fact (Task 1/2 above) that M6 has
a real, substantial kink near the origin that M3 provably cannot have.
Both threads point the same direction - M6's failures look structural
(tied to plant dynamics and to the norm/activation-driven Jacobian
distortion), M3's look like optimization noise - but this entry's
correlation, on its own, is weaker evidence than the raw number
suggested, and should not be cited as "+1.0000" without this context.

**Divergence, tagged not deleted (CLAUDE.md sec 3 rule 6):** M3
case6/seed2 and M6 case6/seed2 are flagged as diverged runs, not
low-quality fits - both land 8-11 orders of magnitude beyond their own
case's other two seeds, matching the exact signature (catastrophic,
single-run, order-of-magnitude blowup) of the Adam/overparameterization
mechanism characterized earlier in this document, now observed for the
first time in a real full-scale training sweep rather than a contrived
LS-init experiment. Milder elevation also visible and worth a second
look before trusting individual (case, seed) cells at face value: M6
case4/seed0 (3050 vs case median 462, ~6.6x) and case7/seed2 (2068 vs
case median 69, ~30x).

**Caveats this entry does not resolve:** 3 seeds is not enough to
distinguish "M6's correlation is real" from "M6 also got unlucky in a
way that happens to correlate with Kreiss" with high confidence: more
seeds, or explicitly re-running the two case-6 outlier seeds fresh to
see if the divergence is reproducible with a different key, would
strengthen this either way. Not done here - flagging as the natural next
step rather than open-ended follow-up.

GPU cost this batch: jit pilot 1.77 min, all-cases sweep+diagnostics
18.23 min (both logged in `gpu_ledger.csv`).

---

## 2026-08-12 — 10-seed rerun: the kink is now a clean, universal result across all 7 cases; the Kreiss correlation is confirmed for M6, killed for M3; case 6 is a reproducible ~50-60% divergence case, not a fluke

`tools/diagnose_all_cases_10seeds.py`, same checkpoints structure as the
3-seed run (`sweep.py --n_seeds 10`, same 40k budget), full
`diagnostics.py` suite on every non-diverged checkpoint this time, not
just Markov+drift. 52.27 T4-min (`gpu_ledger.csv`).

**Lead result - the cleanest thing this investigation has produced:**
median kink magnitude and local-linearity defect, non-diverged seeds,
every one of the 7 cases:

```
        M3 kink        M3 defect       M6 kink       M6 defect
case1   0.000e+00      6.14e-15        7.90e+00      2.16e-01
case2   0.000e+00      8.69e-15        1.12e+01      3.37e-01
case3   0.000e+00      6.81e-15        9.56e+00      2.67e-01
case4   0.000e+00      7.17e-15        1.58e+01      4.70e-01
case5   0.000e+00      5.87e-15        1.14e+01      3.09e-01
case6   0.000e+00      9.78e-14        1.45e+02      2.88e+00
case7   0.000e+00      6.91e-15        6.62e+00      1.83e-01
```

M3's kink is **exactly** `0.000e+00`, bit-for-bit, on every single case
- not "small," zero, for the same reason the case-3-only result was
exactly zero (`jacfwd` of a truly affine map is a mathematical constant
independent of the evaluation point). Its local-linearity defect never
exceeds ~1e-13 anywhere. M6's kink and defect are substantial in every
one of the 7 cases with zero overlap against M3's values anywhere in
the table - not one case where M6 looks linear or M3 looks kinked. This
is the built-in control working exactly as designed, now validated
across the full case ladder rather than one case: **the kink is real,
architectural, and present regardless of which of these 7 plants is
being identified.**

**Case 6's divergence is reproducible, not a 3-seed fluke: 5/10 (M3)
and 6/10 (M6) seeds diverged** (`markov_err_mean > 1e3`). Every other
case's divergence rate is 0-10%. This is a measurable ~50-60% failure
probability specific to case 6, matching this document's established
Adam/overparameterization mechanism (D-only, then the redundancy study,
now a real full-scale sweep) closely enough that it reads as the same
phenomenon recurring, not a coincidence.

**Correlation, redone properly (Pearson AND Spearman, with p-values -
Spearman is the more honest statistic at n=7, robust to the
outlier-magnitude problem the 3-seed Pearson-only version had):**

```
                              Pearson r (p)         Spearman rho (p)
M3, all 7 cases              +1.0000 (0.0000)        +0.179 (0.702)
M3, excluding case 6         -0.093  (0.862)          -0.314 (0.544)
M6, all 7 cases               +0.999 (0.0000)          +0.786 (0.036)
M6, excluding case 6          +0.861 (0.028)            +0.657 (0.156)
```

**M3: dead on arrival, by the statistic that actually matters here.**
Pearson still reads +1.0000 on the full 7-case set (case 6's absolute
*magnitude* - a median of 1.67e4, driven by half its seeds landing near
1e11 - still dominates a magnitude-sensitive statistic), but Spearman -
which only cares about rank order, not magnitude - is 0.18 (p=0.70): the
case ranking by M3's Markov error does not track the case ranking by
Kreiss amplification at all. Both statistics agree once case 6 is
excluded (-0.09 to -0.31, neither significant). M3's fit quality looks
like pure optimization noise, exactly as the single-case entry above
already suggested, now confirmed with 7x the data.

**M6: real by Spearman on the full set, real by Pearson with case 6
excluded - genuinely mixed evidence, not a clean confirmation, stated
plainly rather than rounded up.** Spearman rho=0.786 (p=0.036) on all 7
cases clears significance. Pearson r=0.861 (p=0.028) with case 6
excluded also clears it. But Spearman with case 6 excluded (rho=0.657,
p=0.156) does not - n=6 leaves rank correlation underpowered, and this
is the reading the user flagged in advance as "cannot be load-bearing at
3 seeds," now showing the same fragility persists at 10. The honest
summary: two of four reasonable statistics clear p<0.05, none of them
by a wide margin, and they do not all agree. This is evidence for a real
relationship, not proof of one.

**Divergence rate vs 4 case-level predictors** (kreiss_like, rho,
non-normality `||AA^T-A^TA||_F`, `max_k||A_d^k||_2`):

```
                    Pearson r (p)              Spearman rho (p)
            M3                M6           M3              M6
kreiss      +0.980 (0.0001)   +0.973 (0.0002)   +0.134 (0.775)   +0.579 (0.174)
rho         +0.240 (0.605)    +0.313 (0.494)    -0.091 (0.847)   +0.373 (0.410)
non-norm.   +0.980 (0.0001)   +0.972 (0.0002)   +0.134 (0.775)   +0.694 (0.083)
max||A^k||  +0.980 (0.0001)   +0.973 (0.0002)   +0.267 (0.562)   **+0.810 (0.027)**
```

**Spectral radius (`rho`) never predicts divergence - Pearson or
Spearman, either variant.** It is flat (1.00-1.04) across every case in
this ladder, so it structurally can't. The three transient-amplification
-flavored predictors (kreiss_like, non-normality, max transient growth)
all show the same story: Pearson near-perfect for both variants
(dominated by case 6's simultaneous extremity on every one of these
correlated predictors - they are not independent signals here, so this
mirrors the earlier magnitude-vs-rank problem), Spearman weaker and only
clearing significance once (`max_k||A_d^k||_2` for M6, rho=0.810,
p=0.027). Divergence rate is 0 for 5-6 of 7 cases for either variant,
which makes rank correlation on it inherently underpowered at n=7 (a
near-degenerate, mostly-tied variable) - this is a sample-size
limitation on the statistic, not evidence against the predictor.
Qualitatively unambiguous either way: rho never shows even a hint of a
relationship: every other correlation attempted, including the weak
ones, is at minimum in the same direction; rho isn't just weaker, it's
uncorrelated.

**The case 4-vs-6 tension, and why it matters more than the correlation
numbers above:** the user's own DPC failures were observed on cases 4
AND 6, despite case 4's Kreiss (3.30) sitting far below case 6's (330.3)
- barely above the "easy" cases. If transient amplification is really
what drives this, case 4 should read as meaningfully harder than the
easy cases on the identification side too, not just on the control side.

```
                case3 (kreiss=1.00)   case4 (kreiss=3.30)   case6 (kreiss=330.3)   case4 between?
M3 median          2.168e+00             1.403e+00              1.671e+04            NO
M6 median          1.902e+02             3.395e+02              4.554e+03             YES
```

**M3: case 4 sits BELOW case 3 - not intermediate, not even elevated.**
**M6: case 3 < case 4 < case 6, cleanly monotonic - case 4 genuinely
reads as intermediate-difficulty, consistent with it being a DPC
failure case alongside case 6.** This is the single most direct piece
of evidence in this document connecting the identification-side story
to the control-side (DPC) observation that motivated the whole
project: the ordering that matters for control (case 4 and case 6 both
hard, case 3 easy) shows up in M6 - the architecture actually used for
DPC - and does NOT show up in M3, the artificial linear control. That
asymmetry is more informative than either variant's raw correlation
coefficient, because it is exactly the pattern you'd want if the
mechanism runs through M6's specific architecture rather than through
generic identification difficulty that any variant would share.

**Sample-size caveat on case 6 specifically:** with 50-60% divergence,
the Task 3 diagnostics table above is computed from only 4-5 of 10
seeds for case 6 (vs 9-10 for every other case) - a survivors-only
sample. This does not affect the kink/defect finding (which is already
unambiguous on the 6 cases with full samples) but should be kept in
mind before leaning on case 6's specific magnitude numbers.

**Not yet done, flagged rather than attempted:** re-running the case-6
outlier seeds specifically with fresh keys to check whether the exact
same seeds reproduce divergence (would distinguish "divergence is a
property of the case" from "divergence is a property of specific
unlucky initializations") - the 5/10 and 6/10 rates already answer the
weaker "how often" question the way more seeds would, so this is a
refinement, not a gap that blocks any claim made above.
