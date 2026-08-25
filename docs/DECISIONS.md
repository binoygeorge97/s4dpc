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

---

## 2026-08-13 — Controller Task 2 (M0/M1 oracles): KILL CRITERION TRIGGERED on case 6, not on case 4. Stopped before Task 3 per instruction.

`tools/controller_oracles.py`, sha 1cb6ada: `BoundedGRUController`
(bounded action, `u = max_action * tanh(head(h))`, `max_action=50`)
trained via the full dpc_example curriculum (9000 epochs,
5->10->20->50->100->200) through M0 (true `(A_d, B_d)`) and M1
(least-squares `(A_hat, B_hat)`), all 7 cases x 3 seeds, evaluated by
closed-loop LQR cost on the TRUE plant. Full table
(`docs/controller_oracles_summary.csv`), median cost ratio to oracle
LQR:

```
case   oracle_lqr    M0 ratio      M1 ratio
1        18.5034    1.0047e+00    1.0047e+00
2       739.3073    4.1171e+01    4.1171e+01
3        63.1015    1.0215e+00    1.0215e+00
4        50.5102    1.0633e+00    1.0633e+00
5       343.6217    4.1740e+01    4.1740e+01
6       158.3335    1.3030e+07    1.3030e+07
7        28.8276    1.0041e+00    1.0041e+00
```

**Case 6 fails catastrophically (~7 orders of magnitude over oracle)
through BOTH M0 and M1 - i.e. through the exact TRUE dynamics, no
surrogate involved at all.** Per the kill criterion as specified
("if the GRU fails to stabilize cases 4 and 6 through the TRUE (A_d,
B_d) with bounded actions, then the failure is BPTT through unstable
dynamics, not the S4 surrogate"): case 6 alone satisfies exactly the
mechanism that criterion was checking for. `controller_oracles.py`'s
kill-criterion check ORs case 4 and case 6 together, so it prints
"TRIGGERED" - noted here explicitly because the literal wording
("cases 4 AND 6") is ambiguous between "the specific pattern of both
failing together" and "either one failing." Reported both readings
rather than picking one: the conjunctive pattern (both fail) did NOT
reproduce; the disjunctive check (either fails) did.

**Case 4 is NOT a kill-criterion case: it stabilizes at essentially
oracle-optimal cost (ratio 1.06) through the TRUE plant, under both
M0 and M1.** This directly contradicts the previous entry's read of
the identification-side evidence ("case 4 genuinely reads as
intermediate-difficulty, consistent with it being a DPC failure case
alongside case 6") - that entry is not being edited (append-only), but
this result supersedes its speculation about case 4's role. The
straightforward explanation: the user's ORIGINAL DPC failures on case
4 (pre-Task-1) were produced by dpc_example's UNBOUNDED action bug
(controls reaching ~1e7) - Task 1's entire reason for existing - not by
any inherent difficulty of case 4's dynamics under BPTT. With the
action correctly bounded, case 4 is easy. Case 6 is not, and was never
explained by the action bug (its failure reproduces even with bounded
actions, through the exact true dynamics).

**M0 and M1 agree to ~10 significant figures on every case, including
case 6** (e.g. case 6/seed 0: cost 2063014778.68 vs 2063014778.05).
This is not a bug (`_build_member_grid`'s M1 branch does call
`fit_least_squares`, verified by reading the code, not assumed) - it is
the expected consequence of `identify.py.fit_least_squares`'s own
docstring claim that the LS floor is ~1e-14-level for these noiseless,
persistently-excited linear systems. Consequence for THIS entry's
argument: M0 and M1 are not two independent confirmations of the case-6
failure, they are one confirmation measured twice against
near-identical targets - stated plainly so the table above isn't
mistaken for double evidence.

**Secondary observation, not part of the kill criterion, reported at
exactly the strength shown and no further:** cases 2 and 5 both show a
~41x cost ratio (41.17 / 41.74 - suspiciously close to each other) that
clears neither "trivial" (~1.0x, cases 1/3/4/7) nor "catastrophic"
(1.3e7, case 6). Cross-referenced against
`docs/case_predictors_full.csv` (unchanged from the 10-seed
identification entry): case 2 and case 5 have IDENTICAL `rho`
(1.0408163265306123) and both have LOW `kreiss_like` (1.007 and 1.049 -
among the lowest of all 7 cases, far below case 4's 3.30). So on the
control side, `kreiss_like` does not order difficulty the way it (partly)
did on the identification side for M6: case 4 has the second-highest
`kreiss_like` of all 7 cases yet controls trivially; cases 2/5 have
near-lowest `kreiss_like` yet show real (40x) inflation; only case 6 is
both a `kreiss_like` extreme AND a control catastrophe. Not investigated
further this entry (no re-running, no new correlation pass - the
standing instruction against pushing statistics past what the sample
supports applies here too); flagged for whoever decides what Task 3
looks like next, since it suggests transient amplification is at best a
partial explanation for control-side difficulty, not the same clean
single predictor it was (with caveats) for M6's identification error.

**Decision, per explicit instruction ("Stop and tell me immediately if
that happens"): Task 3 (M3/M6 surrogate controllers) is NOT launched.**
It was fully prepared in parallel while this job ran (checkpoint
selection, Kaggle dataset upload, `tools/controller_surrogates.py`,
pilot script - all committed) but stays gated. Training a controller
through the M3/M6 case-6 checkpoint would be uninterpretable right now:
any case-6 failure there could not be distinguished from "case 6 is
just hard for BPTT regardless of what's being controlled through,"
which this entry now shows is independently true of the TRUE plant
itself. Case 4 is unambiguously clear to proceed on. What to do about
case 6 specifically (exclude it from the Task 3 table with this entry
as the reason; run it anyway and report the confound explicitly
alongside the numbers; something else) is left to the user.

---

## 2026-08-13 — User-confirmed corrections from the Task 2 kill criterion: two claims superseded, one distinction stated as the paper's central caveat

Three things to put on record before any further running, per instruction.

**1. Case 4's DPC failure was the unbounded-action bug, not its
dynamics - SUPERSEDED.** A prior analysis (external to this document -
not something recorded in DECISIONS.md, so nothing here is being
retroactively edited) attributed case 4's original DPC failure to
Jordan-block non-normality and bad input Jacobians. That analysis was
diagnosing a controller emitting ~1e7 against training inputs in
[-10,10] (dpc_example's unbounded `StandaloneGRUController`) - i.e. it
was characterizing the FAILURE MODE of an already-broken controller,
not a property of case 4's dynamics. With the action bound fixed
(Task 1, `s4dpc.control.BoundedGRUController`), case 4 stabilizes at
1.06x oracle cost through the true plant. The Jordan-block/input-
Jacobian analysis is superseded as an explanation for case 4's DPC
failure specifically; it may still be a valid description of case 4's
linear-algebraic structure, just not the cause of the DPC result it was
invoked to explain.

**2. Case 6's 1.3e7x failure carries no information about surrogate
quality.** It fails at the same magnitude through the TRUE `(A_d, B_d)`
as through any surrogate. What it measures is BPTT instability through
a plant with `kreiss_like=330.3` (two orders of magnitude above every
other case) - a property of gradient-based optimization through that
specific transient-amplification profile, not of how well any model
(learned or exact) approximates the dynamics. Any future case-6 result,
in this document or the paper, needs this framing attached or it will
misread as a surrogate-fidelity finding.

**3. Identification difficulty and control difficulty are different
phenomena - the paper must not conflate them.** The 10-seed
identification entry found `kreiss_like` orders M6's Markov-parameter
error (case3 < case4 < case6: 190 / 340 / 4554). The controller run
just showed `kreiss_like` does NOT order GRU/DPC control difficulty
under bounded actions: cases 2 and 5 sit at ~41x oracle cost with
`kreiss_like` 1.01/1.05 (near the lowest of all 7 cases), while case 4
at `kreiss_like=3.30` (second-highest) is trivial (1.06x). A single
"transient amplification explains everything" story cannot be true of
both axes at once with these numbers. Whatever explains cases 2/5 is
task 2026-08-13's open question (saturation check, in progress) - but
regardless of that answer, the ordering mismatch itself is real and
already-established: identification error and control cost respond to
different properties of the case, and the paper's framing needs to keep
those two claims separably supported rather than treating one as a
proxy for the other.

**Next, per instruction:** the saturation check on cases 2/5 (does
`max_action=50` bind for those two plants specifically?) runs before
Task 3 is launched, since a bound artifact there would confound the
M3/M6 surrogate comparison on those cases the same way the missing
bound confounded case 4 originally. Task 3 itself now excludes case 6
(the oracle control fails there, so no surrogate result on case 6 would
be interpretable) while keeping case 6 in all identification-side
results, where `docs/DECISIONS.md`'s prior entries already show it is
informative.

---

## 2026-08-13 — Saturation check result: case 5's ~41x IS a bound artifact, case 2's is NOT. Split verdict, no single max_action fix for both.

`tools/controller_saturation.py` (all 7 cases, `max_action=50`, same
PRNGKeys as `controller_oracles.py`'s original M0 run - reproduces
rather than re-experiments) then `tools/controller_saturation_phase2.py`
(cases 2/5 re-trained at `max_action=200`).

**Phase 1** - median saturation_frac (fraction of timesteps with
`|u| >= 0.95*max_action`) and median max|u|, all 7 cases:

```
case   median_max|u|   median_sat_frac   median_ratio
1        3.444e+01          0.0000         1.0047e+00
2        5.000e+01          0.0698         4.1171e+01
3        4.854e+01          0.0002         1.0215e+00
4        4.992e+01          0.0030         1.0633e+00
5        5.000e+01          0.0448         4.1740e+01
6        5.000e+01          0.1632         1.3030e+07
7        3.610e+01          0.0000         1.0041e+00
```

Cases 2 and 5 both sit clearly apart from cases 1/3/4/7 (near-zero
saturation, max|u| meaningfully below 50) - both hit `|u|~50.000`
(pegged) and saturate 4.5-7.0% of timesteps. The script's own automated
trigger (both cases above a flat 5% threshold) technically did not fire
- case 5 measured 4.48%, just under the cutoff - but that cutoff was an
arbitrary round number, not a feature of the data, and the qualitative
pattern (2 and 5 both meaningfully saturating, 1/3/4/7 essentially
never) clearly holds. Ran phase 2 directly rather than trusting the
literal threshold miss. (Case 6, excluded from Task 3, saturates
hardest of all - 16-18% - consistent with a controller fighting a
plant its own BPTT already can't stabilize, not a separate finding.)

**Phase 2** - re-trained at `max_action=200`:

```
case      ratio@50    ratio@200   sat_frac@50  sat_frac@200   max|u|@50  max|u|@200
2      4.1171e+01   3.0268e+01        0.0698        0.0000   5.000e+01   1.841e+02
5      4.1740e+01   2.2800e+00        0.0448        0.0000   5.000e+01   1.437e+02
```

**Case 5: BOUND ARTIFACT CONFIRMED.** 41.74x -> 2.28x (18.3x reduction),
saturation drops to 0.0000, `max|u|` settles at 143.7 - well under the
new 200 bound, i.e. it stopped needing the bound at all once given room.
`max_action=50` was a genuine controller-capacity limit for case 5, not
a dynamics property.

**Case 2: NOT explained by the bound.** 41.17x -> 30.27x (only a 1.36x
reduction) despite saturation ALSO dropping to 0.0000 and `max|u|`
reaching up to 188.4 (still under 200, so it isn't hitting the new bound
either). Whatever makes case 2 hard for a bounded GRU, it is not
controller capacity - the cost stays ~30x oracle even with 4x more
authority and zero saturation. Also notable, flagged rather than
explained: case 2's per-seed spread exploded at the higher bound (ratios
1.92 / 30.27 / 128.05 - a ~67x range across 3 seeds, vs phase 1's tight
36.8-52.2 range). A larger action range plausibly makes this
particular optimization landscape harder to hit consistently, but this
is a hypothesis, not tested here - flagged for whoever looks at case 2
next, not investigated further this entry.

**Consequence for Task 1: no single max_action serves both cases
cleanly**, and this is being handed to the user rather than decided
unilaterally - which bound(s) to use for the M0/M1/M3/M6 comparison
table changes what the table means, not just its numbers. Task 3
(`tools/controller_surrogates.py`) is built, committed, mount-path-fixed
and pilot-validated, but not yet launched pending that call.

---

## 2026-08-13 — max_action rule for Task 1, fixed in advance (methods-section wording)

Per-case max_action for Task 1's M0/M1/M3/M6 comparison, decided by the
user, stated here as the rule the paper's methods section should use:

> max_action is the smallest value in {50, 200} at which the ORACLE
> controller (M0, true A_d/B_d) does not saturate. The rule references
> only the true plant and the oracle, never M3 or M6, so it cannot
> favour either surrogate.

Applied to `docs/controller_saturation_summary.csv`'s phase-1 numbers
(fraction of timesteps at `|u| >= 0.95*max_action`), this criterion
alone says cases 2 and 5 both saturate at 50 (6.98% and 4.48%) while
1/3/4/7 don't (0-0.3%) - i.e. by saturation alone it points at 200 for
two cases, 50 for four.

**Documented exception, still oracle-only evidence:** case 2 stays at
50. Phase 2 (previous entry) showed that moving case 2 to 200 does not
serve the rule's underlying purpose - M0's own cost ratio barely
improves (41.17x -> 30.27x, an 18x weaker effect than case 5's
41.74x -> 2.28x) and per-seed variance explodes (1.92x/30.27x/128.05x
at 200 vs a tight 36.8-52.2x at 50). Saturation alone said "raise it";
cost-and-stability together, measured on the oracle alone, said case
2's ~30-41x difficulty isn't the bound - raising it just trades one
confound (saturation) for another (seed instability) without buying
interpretability. This keeps the exception grounded entirely in M0
evidence, never in how M3 or M6 happen to perform, so it still cannot
favour either surrogate.

**Final assignment, `controller_oracles.CASE_MAX_ACTION`:** 50 for
cases {1, 2, 3, 4, 7}, 200 for case {5} only.

**Why not uniform 200 for every case (rejected):** a looser bound gives
every controller more room to emit actions further outside the
identification range - and M6 (the LayerNorm/kink variant) is the one
whose failure mode is exactly emitting distorted, potentially
out-of-range actions. A bound that lets M6 range further than M3 would
inflate M6's cost for a reason unrelated to the kink hypothesis, i.e.
it would be a confound pointing toward the paper's own hypothesis -
the worst kind to have sitting unexamined in a headline result.

**Built into `tools/controller_surrogates.py` (not left as an
after-the-fact caveat):** the combined table carries a `max_action`
column and, for every (oracle, case) row including M3 and M6, a
`median_saturation_frac` / `median_max_abs_u` pair. Two explicit checks
run after the table: (1) does any surrogate saturate meaningfully more
than the oracle did at that case's same bound, and (2) does M6 saturate
where M3 doesn't. Either prints a flagged confound, not a silent
number - see the script's own printed "CONFOUND CHECK" sections when
Task 1 runs.

**Oracle-side rebuild:** `tools/controller_oracles_final.py` re-trains
M0 and M1 for all 6 control cases at these final per-case bounds (a
single, complete, consistent source - `docs/controller_oracles_summary.
csv` predates the saturation_frac field this table needs for every row,
so it is not reused for Task 1's table; it remains the record of Task
2's kill-criterion check, which is a separate, already-settled
question). `tools/controller_saturation._run_m0` was generalized (an
`oracle_name` parameter, default "M0" preserving every existing caller)
so this script can route it through `fit_least_squares` for M1 instead
of only ever training through the true plant.

---

## 2026-08-13 — Task 3 results: M3 AND M6 both catastrophically fail control transfer on EVERY case - confound checks clean, kink magnitude does NOT track the M6-vs-M3 gap, and two DIFFERENT failure modes are visible in the training loss itself

`tools/controller_surrogates.py`, sha 97585f7 (oracle side) + a9194c7
(surrogate code) - M3 and M6, cases {1,2,3,4,5,7} at their final
per-case `max_action` (`controller_oracles.CASE_MAX_ACTION`), 3 seeds,
full curriculum. Wall time 11327.9s (188.80 T4-min).

**Combined table, median cost ratio to oracle LQR:**

```
oracle  case  max_action   M0/M1 ratio     M3 ratio      M6 ratio
M0/M1     1       50         1.00x       323.28x       155.50x
M0/M1     2       50        41.17x     55903.40x     68293.11x
M0/M1     3       50         1.02x       310.28x       622.74x
M0/M1     4       50         1.06x    121941.94x     53978.45x
M0/M1     5      200         2.28x    466718.19x    361699.72x
M0/M1     7       50         1.00x       494.42x       734.71x
```

**Both confound checks are clean.** No surrogate saturates meaningfully
more than the oracle did at the same case's bound (check 1: every
case/variant "clean" against a 0.05 threshold); M6 never saturates more
than M3 anywhere (check 2: every case "clean"). Whatever is driving
these numbers, it is not the action bound - the full printed checks are
in the kernel log, `docs/controller_comparison_summary.csv` carries
`median_saturation_frac`/`median_max_abs_u` for every row.

**The headline, unavoidable observation: both M3 and M6 are 2-6 orders
of magnitude worse than the oracle on EVERY case, including the ones
that are trivial for M0/M1** (cases 1, 3, 7 - near-1x for the oracle,
100x-750x for the surrogates). This holds even for M3, which the
10-seed identification entry showed recovers Markov parameters to
~1e-6 and has EXACTLY ZERO kink by construction. Teacher-forced
identification fidelity, even M3's near-perfect fidelity, does not
imply the resulting surrogate is safe to optimize a controller through.

**Direct answer to the kink-magnitude question, computed exactly as
asked:**

```
case   kink   M3 ratio      M6 ratio     M6/M3   M6 vs M3
 7     6.62   4.944e+02    7.347e+02    1.486   WORSE
 1     7.90   3.233e+02    1.555e+02    0.481   better
 3     9.56   3.103e+02    6.227e+02    2.007   WORSE
 2    11.20   5.590e+04    6.829e+04    1.222   WORSE
 5    11.40   4.667e+05    3.617e+05    0.775   better
 4    15.80   1.219e+05    5.398e+04    0.443   better
```

**No.** Case 4 has the highest kink magnitude (15.8) of all six cases
and is the case where M6 does BEST relative to M3 (0.443x - M6 costs
less than half of M3). Case 7 has the lowest kink (6.62) and is where
M6 does WORST relative to M3 (1.486x). Spearman(kink, M6/M3 ratio) =
-0.54 (n=6) - weakly NEGATIVE, i.e. pointing away from the hypothesized
direction, and nowhere near the ~0.81 threshold this document has
already established as needed for significance at this sample size.
Reported at exactly this strength: not evidence for the kink mechanism
driving the control-side cost gap, and not strong enough (n=6) to be
evidence against it either - just not the clean tracking relationship
that would have made this table read as confirmation.

**Not asked for, found while checking the training-loss curves for
sanity before trusting the table above - two DIFFERENT failure modes,
not one:**

- **M3 shows training-time BPTT instability.** For the {1,2,3,4,7}@50
  ensemble, the moment the curriculum reaches N=200 (the final phase),
  mean DPC loss (measured THROUGH the M3 surrogate, i.e. the actual
  training objective) jumps to 6.9e13 with a per-member max of 1.04e15,
  and is still at 1.06e13 (max 1.59e14) after 2000 epochs of that
  phase - down from where it started, but nowhere near converged, and
  astronomically larger than the ~1-2 range that "trivial" cases showed
  under M0/M1. This is the same signature this document has already
  named for identification (BPTT/Adam through a numerically demanding
  rollout diverging in some members while others stay small) - here
  showing up in the CONTROL loop, for cases that were completely clean
  when the same GRU trained through the TRUE plant.
- **M6 shows a clean reality gap instead - training loss stays bounded
  (~10-560 across the same phase, same case group) while the TRUE-plant
  transfer is just as catastrophic (67x-187,000x).** The M6 controller
  looks like it converged fine, by the only signal available during
  training; it just didn't learn to control the real system. This is
  closer to a sim-to-real / reward-hacking failure than an optimization
  failure - the controller found a way to look good to the M6 surrogate
  specifically without that generalizing to `(A_d, B_d)`.

**What this does and doesn't mean, stated carefully:** M3's failure
mode (training instability) is not obviously kink-related - M3 has no
norm layer at all. M6's failure mode (bounded training loss, terrible
transfer) is at least CONSISTENT with a kink-flavored story (a
distorted local Jacobian near the origin could make it easy for BPTT to
find a policy that exploits M6's specific behavior there without that
policy generalizing) but the same evidence is equally consistent with
generic model-based-RL reality-gap - nothing here isolates the kink
specifically as the cause versus "any imperfect surrogate gets
exploited by a long-horizon BPTT controller." Not resolved this entry;
flagged for the user's read before drawing a paper-level conclusion.

**Not investigated further this entry** (per the standing instruction
against pushing past what's been checked): per-seed variance within
each case, whether shortening the curriculum's final horizon changes
either failure mode, and whether M3's training-loss blowup is
reproducible with fresh seeds. All three are natural next steps, none
attempted here.

GPU: 188.80 T4-min for this run. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — REFUTATION: the kink does not cause DPC failure

Stated plainly, per instruction, not softened: the central hypothesis
this project was built to test - that LayerNorm's degree-0-homogeneous
kink near the regulation setpoint is why DPC fails through a learned S4
surrogate - is refuted by this project's own control-side data.

M3 has EXACTLY ZERO kink by construction (0.000e+00 bit-for-bit across
all 7 cases, the 10-seed identification entry) and recovers Markov
parameters to ~1e-6. If the kink were the mechanism, M3 should control
close to the oracle. It does not: 310x-466,000x oracle cost across
every case tested, the same order of magnitude of failure as M6 (which
has substantial kink everywhere). Spearman(kink magnitude, M6/M3 cost
ratio) = -0.54 at n=6 - not just non-significant, INVERTED from the
predicted direction: case 4, the highest-kink case (15.8), is the one
case where M6 controls BETTER than M3 (0.443x), and case 7, the
lowest-kink case (6.62), is where M6 is worst relative to M3 (1.486x).
A mechanism whose signature is most absent exactly where the mechanism
is strongest is not a surviving hypothesis.

This does not erase the identification-side results - M3's zero kink
vs M6's universal nonzero kink, and the Kreiss-like correlation with
M6's Markov error, are still real, still correctly measured, and still
in this document. What's refuted is specifically the causal claim that
this identification-side property (the kink) is what breaks DPC. It
isolates cleanly on the identification side and does not transfer to
the control side - the two are different phenomena, matching the
2026-08-13 entry that first named this distinction from the saturation
work (kreiss_like ordering M6's identification error but not control
difficulty).

**What survives, and is the bigger finding:** BOTH surrogates fail
catastrophically on cases where the oracle controls near-perfectly
(1.00-1.02x) - including M3, whose one-step and multistep-Markov
fidelity are essentially exact. Near-exact teacher-forced prediction is
not sufficient for a surrogate to be safe to optimize a DPC controller
through. The hypothesis document's requirements for DPC success
(J_x ~ A_d, J_u ~ B_d, F(0,0,s) ~ 0, correct multistep composition) are
essentially satisfied by M3 and it still fails - that list is
incomplete. Finding what property is missing (candidates raised by the
same entry: BPTT-through-the-surrogate numerical stability at long
horizon for M3; a reality-gap/exploitable-imperfection mechanism for
M6) is the open question the project now turns on. Verification of
these numbers (docs/DECISIONS.md, next entry) is in progress before
either candidate is investigated further.

---

## 2026-08-13 — Task 1 verification: the 310x M3/case3 number is real, not a bug. M3's own free-running dynamics diverge from the true plant, even open-loop with no controller involved.

`tools/verify_m3_case3.py`, one controller (M3/case3/seed0, an easy
case: oracle 1.02x, M3 measured at 310x in the Task 3 sweep), one
shared x0. Checkpoint, raw trajectories, and both plots saved
(`docs/verify_m3_case3/`).

**Closed-loop, same controller, true plant vs the M3 surrogate it
trained through:**

```
                cost      ||x(0)||   ||x(200)||   max||x||   max|u|
true plant    1.749e+04     6.635      138.6        138.6     46.5
M3 surrogate  7.734e+01     6.635        1.58         7.5     16.5
```

The plot (`closed_loop.png`) makes this unambiguous: `||x(t)||` on the
M3 surrogate oscillates in a bounded 1-8 band, looking like ordinary,
if imperfect, regulation. On the true plant, `||x(t)||` climbs smoothly
and monotonically from 7 to 139 over 200 steps, with `||u(t)||` growing
in step - a controller correctly fighting a divergence it cannot win
against, not noise or an artifact.

**Open-loop, no controller at all - the cleaner check, run because the
closed-loop result alone can't separate "bad surrogate" from "bug in
the rollout":** the oracle LQR's own closed-loop control sequence on
case 3 (which drives the TRUE plant from `||x||=6.6` to `||x||=5.0e-4`
by step 200 - LQR doing exactly what it should) was replayed blind
through M3, no feedback, nothing controller-related in the loop at all.
M3 predicts `||x(200)||=96.7` under the EXACT sequence that in reality
converges the system almost to machine zero. Tracking error
(`open_loop.png`, bottom panel) grows to ~12 by step 15, briefly
recovers to ~4 near step 30, then grows again to a peak of 205 at step
176. M3 gets the short-horizon behavior approximately right and then
diverges - consistent with (though not proof of) the identification-
side finding that M3 matches Markov parameters (a LOCAL linearization
at u=0) to ~1e-6: that check tests infinitesimal, near-origin behavior,
not what happens under a sustained, realistically-large control
sequence over 200 steps.

**Verdict, per the decision tree specified: genuine reality gap, not a
bug.** Reasoning: (1) the M3-surrogate trajectory is smooth and bounded,
not NaN/exploding/frozen - not the signature decode=True or
state-threading bugs produce; (2) the open-loop check has NO controller
or BPTT in it at all, isolating the surrogate's own free-running
dynamics from anything about how it was optimized against; (3) the
rollout mechanics themselves (`model_in = concat([x,u])`, same step
function as `rollout_learned`) are the same code already exercised
across every Task 3 pilot without incident, and structurally match
`diagnostics.markov_parameters`, independently validated to machine
precision in `tools/validate_diagnostics.py`. Nothing here points at
the harness; everything points at M3's learned realization having
long-horizon/large-input dynamics that diverge from the true system's,
despite matching its short-horizon linearization almost exactly.

**Results stand. Proceeding to Task 2** (characterizing the M3
training-instability and M6 reality-gap mechanisms) on this basis.

GPU: 14.90 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 2b: M6's reality gap does NOT primarily come from exiting the identification support - disagreeing with the predicted mechanism

`tools/task2b_m6_reality_gap.py`, one M6/case3/seed0 controller (freshly
trained - Task 3 never saved controller checkpoints, so this is a new
instance, not a re-load; see the note on non-determinism below), rolled
out on both the M6 surrogate and the true plant from 100 x0s.

**Cost: 68.8 on the M6 surrogate (looks converged) vs 1.681e4 on the
true plant** - same qualitative pattern Task 3 found for this case
group (bounded training loss, catastrophic transfer), reproduced
independently.

**The predicted mechanism - "it exits the identification support" -
is only partially right, and the plot (`support_comparison.png`) points
at something more specific:**

```
                          state outside envelope   control outside envelope
M6-surrogate-driven              42.05%                    6.86%
true-plant (same controller)     99.80%                    96.92%
```

Raw numbers alone would read as confirmation. The plot doesn't. The
identification envelope (gray band) sits at `||x||`~3-10 and `||u||`~5-14.
**The M6-surrogate-driven trajectory's mean stays inside or right at the
edge of that band for essentially the entire 200 steps** - it looks
like an ordinary, in-distribution trajectory. **The true-plant
trajectory, same controller, same x0s, leaves the band almost
immediately and climbs smoothly away from it, reaching ~100+ by step
200.** The controller was not driving M6 into some exotic corner of its
input space to exploit a blind spot there - by M6's own accounting, it
mostly stayed home. The state-support-violation number (42%) is
inflated by a baseline effect that applies to every variant equally:
`EVAL_X0_RANGE=5.0` is already wider than the identification
trajectory's own natural excursion (several state dimensions have
identification ranges narrower than 5 units), so many x0s start outside
the envelope before any dynamics run at all - M0/M1/M3-driven
trajectories share this same starting condition and still return
toward the origin; only the true-plant-under-M6 trajectory diverges
away from it.

**Disagreeing with the predicted mechanism, stated plainly as asked:**
this looks less like "the controller found and exploited an
out-of-support region of M6" and more like **M6 is simply wrong about
how the true system responds to sustained control, even within the
region it was trained on** - the same character as Task 1's open-loop
finding for M3 (accurate near the origin / on short horizons, wrong
over a sustained 200-step rollout), not a distinct "found a blind spot"
mechanism. BPTT did not need to push M6 out of distribution to find a
policy M6 rates as good; M6's own in-distribution long-horizon
predictions were already wrong enough. This is a more parsimonious
explanation than the identification-support-exit hypothesis, and it
unifies with M3's failure mode rather than requiring two unrelated
mechanisms - both surrogates get local/short-horizon behavior right and
global/long-horizon free-running behavior wrong, and DPC's cost
function only ever looks at the surrogate's own (wrong) long-horizon
prediction during training.

**Caveat on the cost number specifically:** this controller was trained
as a SINGLE-member ensemble (batch size 1), not as part of Task 3's
original 15-member vmapped batch. The result (1.681e4) is the same
order of magnitude and qualitative pattern as Task 3's original
M6/case3/seed0 (1.058e4) but not bit-identical, despite identical
PRNGKeys - plausibly the same class of effect CLAUDE.md already warns
about for cross-platform runs (BPTT through sensitive dynamics
amplifies small floating-point differences), here triggered by batch
size changing XLA's fusion/reduction order rather than by hardware.
Flagged, not investigated further - doesn't change the qualitative
finding, but single-member Task 2 reruns should not be read as
bit-exact reproductions of Task 3's ensemble-batched numbers.

GPU: 17.98 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 2a: M3 does NOT break at N=200 - it never worked, from N=5 onward. This forces a correction to the "two failure modes" framing.

`tools/task2a_m3_horizon_ablation.py`, case3/seed0, curriculum capped at
N=5/20/50/100/200, true plant vs M3, all evaluated on the true plant
(`docs/task2a_m3_horizon_ablation.csv`):

```
cap    true ratio    M3 ratio
  5       4.16x       357.75x
 20       2.55x       303.83x
 50       1.45x       281.40x
100       1.04x       156.19x
200       1.02x       282.12x
```

**True plant: smooth, monotonic improvement with more horizon** -
exactly what a normal learning curve looks like. **M3: never close to
usable at ANY cap, including the shortest (N=5, 1000 epochs) - and NOT
monotonic.** The premise in the task instruction ("if M3 breaks at
N=50 while true is fine at N=200") does not hold - M3 does not break
at a threshold, because it never worked. A controller that only ever
saw 5-step BPTT through M3 already transfers at 358x oracle cost on
the true plant. More training horizon does not make it worse, but it
also does not fix it (156x at cap=100, back up to 282x at cap=200 -
single seed, not a reliable trend, but certainly not monotonic decay
either). The bottleneck is not how well the controller is optimized
against M3 - by every indication (the SMOOTH, bounded training loss at
N=200 shown below) the controller optimizes against M3 just fine. The
bottleneck is that M3 itself does not match the true system, and no
amount of additional controller training against a fixed, wrong
surrogate can close that gap - explaining directly why longer horizon
doesn't help: BPTT only ever updates the controller, never M3.

**Correction to the Task 3 "two failure modes" framing, found by
checking the training-loss curve for this run's cap=200 phase before
trusting the table:** cap=200's M3 training loss is completely calm -
262.9 -> 38.8 -> 37.5 -> 37.3, smooth and convergent, nothing like the
6.9e13-to-1e15 blowup Task 3 recorded for the M3 {1,2,3,4,7}@50
ensemble at the same N=200 phase. Since this run trains case3/seed0
ALONE and reproduces neither the blowup nor (per the Task 3 per-member
table) case3's own eval numbers were never the extreme ones in that
batch (case3's worst seed there was 1981.7x; case2 and case4 reached
131,980x and 486,120x) - **the training-instability signature belongs
to specific OTHER cases in that ensemble (most likely case 2 and/or
case 4, going by their eval-ratio outliers), not to case 3, and
probably not to "M3" as a variant uniformly.** Case 3's M3 failure is
the SAME bounded-training-loss/bad-transfer signature this document
attributed to M6 specifically in the Task 3 entry. That entry's clean
M3-trains-unstably/M6-has-a-reality-gap dichotomy is not supported by
this case - it was read off aggregate ensemble loss statistics (a
15-member mean and range), which can't distinguish "one member
exploded" from "the variant is uniformly unstable." **Reality-gap
(bounded loss, bad transfer) looks like it may be the dominant pattern
across both variants and most cases; training-time instability may be
a separate, case-specific (not variant-specific) phenomenon layered on
top for a minority of cases.** This is an inference from the evidence
assembled so far (the per-case eval-ratio outliers in Task 3's table),
not independently re-verified by looking at case 2 or case 4's own
per-member training-loss curves directly - flagged as the natural next
check, not attempted here.

GPU: 33.00 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 3 (new session): the s0=0/x0!=0 inconsistency is NOT the (or a) cause of the divergence - warm-starting the S4 state does not help, and for M3 it makes things monotonically WORSE

`tools/openloop_warmstart.py`, sha a638f7d: fresh M3+M6 identification
(all 7 cases, 10 seeds, 40k epochs - identification wall time 162.2s/
154.7s, diverged: M3 case6 seeds {1,4,5}, M6 none), then for 3 restart
points (t0=50/150/250, giving naturally-growing ||x(t0)|| per case's
own open-loop instability) x 4 burn-in lengths B in {0,5,20,50},
teacher-forced burn-in then a 150-step free run, no controller anywhere
- open-loop PREDICTION only. 1680 rows, `docs/openloop_warmstart.csv`.
No confound with control/BPTT: this isolates the initial-condition
question from anything about training.

**Median err_h150, cold (B=0) vs warmest (B=50), the 4 "moderate"
cases (1/3/4/7 - cases 2/5/6 grow so explosively open-loop that x0 at
t0=150/250 reaches 10^2-10^6 in magnitude, past the point where state
initialization is the dominant factor):**

```
       case1        case3        case4        case7
     B=0->B=50    B=0->B=50    B=0->B=50    B=0->B=50
M3   2.15-5.49x    1.24-4.15x   1.26-1.98x   1.09-1.21x   (all WORSE with warm start)
M6   0.84-1.49x    1.00-1.25x   1.00-1.01x   1.00-1.05x   (roughly flat)
```

**Full B sweep (0/5/20/50), case-median err_h150, t0=50, same 4
cases - the monotonic-with-B shape is the sharper result than the
single B=0-vs-B=50 endpoint:**

```
        B=0        B=5        B=20        B=50
M3-c1  62.97      77.42      124.04      164.96   (steadily worse)
M3-c3  400.13     404.75     502.71      497.38   (steadily worse)
M3-c4  655.26     662.99     727.35      824.29   (steadily worse)
M3-c7  102.94     111.01     109.13      124.08   (worse, less clean)
M6-c1  12.75      13.29      11.07       12.32    (flat/noisy)
M6-c3  225.37     218.57     175.56      273.67   (flat/noisy)
M6-c4  649.04     650.33     651.92      656.28   (flat)
M6-c7  66.07      67.24      67.99       69.23    (flat)
```

**Verdict, per the decision tree stated in advance: mechanism (a) is
dead.** Warm-starting does not substantially reduce free-run error for
either variant - and for M3, on every one of the 4 informative cases,
error rises *monotonically* with more burn-in, the opposite of what an
IC-consistency fix should do. This isn't noise around 1.0x; it's a
clean, one-directional trend across 4 independent cases.

**Why MORE history makes M3 worse, read together with the mechanism
this session opened with:** if the S4 hidden state carried genuinely
useful information about the plant's true state, teacher-forced burn-in
(feeding it more real history before evaluation) should only help or be
neutral. That it instead hurts, and does so more as burn-in lengthens,
is consistent with mechanism (b) from this session's brief (spurious
internal modes) and inconsistent with mechanism (a): there is no
"correct" state for the S4 recurrence to warm up *to*, because its
internal modes don't correspond to anything physically meaningful in
the first place - feeding more history just accumulates more of
whatever ungrounded dynamics those modes have learned, rather than
converging toward a state that helps prediction. M6's near-flat
response (vs. M3's clean monotonic degradation) is itself informative:
M6's LayerNorm re-normalizes the block's input every step, which
plausibly bounds how much a longer burn-in can let any accumulated
internal drift compound - a testable follow-up, not established here.

**Consequence for the session's mechanism ranking:** (a) is dead by
its own stated test. Task 4 (spurious internal modes, M3's augmented
state-transition operator) carries the investigation forward, and this
result is a positive prior for what it will find - M3 punishing more
genuine history is exactly the signature an ungrounded/spurious
internal-mode story predicts, not a neutral coincidence.

GPU: 10.33 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 4: M3's augmented state-transition operator has ~300 spurious near-unit-circle modes (of 1030) where the true system has 3-6, and often wildly excessive transient growth - but neither predicts the per-case DPC ratio

`tools/m3_spurious_modes.py`, sha 3c3563f (script committed at this sha;
run at f3a9623): M3 has no norm/activation/glu, so its ENTIRE one-step
map - physical state (6) plus the S4 layer's own hidden state (2x16x32
= 1024 real-flattened) stacked into one 1030-dim vector - is exactly
affine, and its Jacobian (`jax.jacfwd` on a real-valued wrapper, evaluated
at z=0/u=0 - valid everywhere) is the exact augmented linear operator
`Abar`, not merely a local approximation. All 7 cases, 10 seeds, 70
members, identification wall time 163.6s.

**Every case, all 10 seeds: `rho(Abar)` sits at 1.02-1.03** (barely past
1, cases 1/2/3/4/5/7 - only case 6 goes much higher, median 1.25, up to
2.58 in the worst seed) - a naive spectral-radius read would call these
systems mildly, comparably unstable to the true plant (`rho(A_d)`
1.00-1.04 across the same 7 cases per the earlier identification
entries). **That reading is wrong, and the rest of the numbers show why:**

```
case   n_near_unit_true(/6)  n_near_unit_abar(/1030,med)  obs_norm(med)  growth_ratio_k200(med)
1              3                      325.0                  14.6              4.13e+03
2              6                      325.5                  15.9              3.00e+00
3              6                      368.5                  15.4              1.01e+03
4              6                      274.0                  15.8              2.53e+01
5              6                      291.5                  12.4              1.26e+00
6              6                      328.5                  19.2              2.77e+23
7              4                      264.0                  12.5              1.37e+02
```

**M3 has learned 264-369 (median per case; overall range 121-619) modes
within 0.01 of the unit circle, out of 1030 augmented state dimensions -
40-60x more near-marginal modes than the true system has (3-6), on
EVERY case, not just the hard ones.** `obs_norm` (`||Abar[:6,6:]||_2`,
the SVD 2-norm of the block mapping S4 state to next PHYSICAL state in
one step - an eigenvector-free observability proxy, per
`docs/DECISIONS.md`'s standing ban on eigenvector solves for
defective/near-defective matrices) is O(10-20) for every case: these
spurious modes are not inert bystanders sitting in a null space the
output never sees - they have real, comparable-magnitude one-step
leverage on the physical state.

**`||Abar^200||_2` vs `||A_d^200||_2` is where "same rho, same I/O
fidelity" breaks down completely.** Case-median ratios range from 1.26x
(case 5, barely more than the true system's own growth) to 4128x (case
1) among the six non-case-6 cases, and the OVERALL range across all 70
members is 0.06x to 6.4e77x. Two "easy," near-identical-`rho`, similar-
`kreiss_like` cases (1 and 5, from the identification-side tables
earlier in this document) sit 3 orders of magnitude apart in transient
growth here - this is realization-dependent in exactly the way
`markov_parameters`/Kreiss-on-`A_d` alone cannot see, confirming the
task's premise: same input-output map (M3's Markov error is ~1e-6
everywhere) does not imply same transient conditioning.

**Correlation against the recorded per-(case,seed) M3 DPC ratio
(`docs/controller_surrogates_summary.csv`, 18 matched pairs across 6
control cases x up to 3 seeds each): null, honestly.** None of
`rho_abar`, `n_near_unit_abar`, `obs_norm`, `growth_ratio_k200` clears
significance, Pearson or Spearman, seed-level (n=18) or case-median
(n=6) - every p-value is >=0.13, most >0.4. Reported at exactly this
strength, not rounded up: this is a real, structural, universally-
present abnormality (every one of the 7 cases has 40-60x too many
near-unit modes and O(10) observability), but it does not explain WHY
case 1 fails worse than case 5 in the DPC table the way `kreiss_like`
partially explained M6's identification error two entries ago. Read
together with the kink finding (also real, also uncorrelated with the
M6-vs-M3 cost gap): this project now has TWO independently-confirmed,
architecturally-clean abnormalities in these surrogates - LayerNorm's
kink for M6, ~300 spurious near-unit modes with wild transient growth
for M3 - that plausibly explain why backpropagating a DPC controller
through either surrogate is categorically dangerous, without either one
explaining the specific case-by-case gradient of how badly it fails.
That gradient remains open.

**Consistent with, not just adjacent to, Task 3's finding:** Task 3
found that MORE genuine burned-in history makes M3's free-running
prediction monotonically worse, which only makes sense if the S4
state's own dynamics are actively unhelpful once excited - exactly what
~300 near-unit modes with real one-step leverage on the output (`obs_norm`
O(10-20), not ~0) and, on several cases, many-orders-of-magnitude
transient amplification would produce: feeding more history gives those
modes more room to move away from a useful trajectory, not less.

GPU: 6.37 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 2: THE key result of this session. M0_S4 (the true plant, realized exactly by an S4 and deployed through the identical BPTT/decode=True machinery M3/M6 use) controls IDENTICALLY to training directly through the true (A_d,B_d) matrices, at every horizon. The S4/BPTT graph is innocent - the failure is entirely about what identification learns.

`tools/controller_m0_s4.py`, sha 637abae (script; run at f3a9623):
builds an M6-architecture `StackedModel` per (case, seed) with the block
zeroed exactly as in `tools/validate_diagnostics.py` (`out.kernel=0`,
`C_real_imag=0`, etc.) so its one-step map is `x_next = A_d@x + B_d@u`
to ~1e-17, for every case - then trains the STANDARD
`BoundedGRUController` through it via `rollout_learned`, the exact same
decode=True/stepped/BPTT code path `controller_surrogates.py` uses for
M3/M6. Three parts: (A) construction diagnostics, (B) a case-3 horizon
ablation directly comparable to `docs/task2a_m3_horizon_ablation.csv`,
(C) the full 6-case x 5-seed ensemble.

**Part A confirmed the analytical prediction before any GPU spend:**
reading `s4dpc/blocks.py`'s `ConfigurableBlock.__call__` shows
`out.kernel=0, out.bias=0` forces the block's post-GLU contribution to
exactly 0 regardless of the S4 layer's own output, so with
`residual=True` the block output is exactly `skip = encoder(x,u)`,
INDEPENDENT of the S4 hidden state - meaning cold-start (s0=0) vs a
burned-in warm start must give bit-identical rollouts for M0_S4 by
construction, unless the (causally dead) state itself overflows to
inf/nan. Checked directly per case, not merely asserted: self-check
error ~1e-17, cold-vs-warm output diff and state-finiteness confirmed
clean for every case. (This also means Task 3's warm-start question is
answered trivially/vacuously for M0_S4 by construction - Task 3's real
answer, on the actually-learned M3/M6, is the entry two above this one.)

**Part B is the sharpest number in this entire project.** Case 3, one
seed, curriculum capped at N in {5,20,50,100,200}, evaluated on the
true plant exactly as everywhere else:

```
cap    true-plant ratio    M0_S4 ratio
  5       4.156256067932085x   4.156256067932085x
 20       2.5517450039151535x  2.5517450039151535x
 50       1.449500203629581x   1.4495002036295808x
100       1.0380708325516528x  1.0380708325516528x
200       1.0214865143264333x  1.0214865143264333x
```

Identical to the printed precision at caps 5/20/100/200, and identical
to 14 significant figures at cap 50 (the only cap with any visible
difference at all, and that difference is at the level of ordinary
float64 accumulation noise across two independently-JIT-compiled
training runs, not a real gap). **A controller trained via BPTT through
an S4 - stepped one input at a time through the full decode=True/
jax.lax.scan machinery - that happens to compute the exact true
dynamics is INDISTINGUISHABLE from a controller trained directly
through the true `(A_d, B_d)` matrices.** This is not "close to
oracle" (the brief's stated success threshold was ~1.0-1.1x) - it is
exact agreement with the true-plant training baseline itself, at every
horizon tested, refuting even the possibility of a horizon-dependent
BPTT-through-S4 numerical-conditioning effect this document speculated
about earlier (Task 2b's entry, and this session's own opening
mechanism list).

**Verdict, unambiguous: the S4 realization and its backprop graph are
completely innocent.** None of the following are the cause of M3/M6's
300x-700,000x DPC failure: the S4 architecture itself, the decode=True
stepped/`jax.lax.scan` machinery, BPTT through that machinery at any
horizon from 5 to 200, or anything about how `rollout_learned` deploys
a surrogate. The entire failure is about what identification actually
LEARNS - given a surrogate that is merely CLOSE to the true dynamics
(M3's ~1e-6 Markov error, not exact), something about that residual
imperfection is catastrophic for BPTT-trained control, even though the
identical training/eval machinery is provably safe when fed the exact
answer. This sharpens every other finding in this document: Task 4's
~300 spurious near-unit modes and Task 3's warm-start-makes-it-worse
result are not properties of "being an S4 surrogate" in the abstract -
Part A above shows M0_S4's own internal S4 state is completely inert
by construction, with none of Task 4's pathology, because it was never
asked to LEARN a realization, only handed one. The spurious modes are
something Adam's optimization introduces when fitting the factored S4
parameterization to data (consistent with, and now given fresh teeth
by, this document's much earlier finding of an exact ~490-dim gauge
non-identifiability in this same parameterization) - not something
inherent to the architecture's capacity to represent the true system.

**Part C (the full 6-case x 5-seed ensemble) did not complete cleanly
this run - see the next entry.** Given Part B's result, Part C is
expected to be confirmatory, not decisive on its own; still being run
for the statistical power the session brief asked for.

**Part C's OOM, fixed (sha 637abae):** a 25-member batch (cases
{1,2,3,4,7} sharing `max_action=50`, x 5 seeds) hit
`RESOURCE_EXHAUSTED` at the N=200 phase - BPTT through 200 float64
steps for 25 members at once needed >15GB, past a T4's 16GB.
`controller_surrogates.py`'s own batches never exceeded ~21 members at
only 3 seeds; fixed by batching one case (5 members) per ensemble call
instead of grouping cases by shared `max_action` bound - same total
member count, smaller peak memory, no scope reduction. Re-running.

GPU: 90.01 T4-min (Parts A+B complete, Part C OOM'd partway through).
Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — CORRECTION to the Task 2 entries above: M0_S4 exonerates the S4/BPTT machinery, NOT realization - the entries above conflated the two

Flagged by the user, correct on inspection - the two entries above (and
this document's Part C entry) say things like "given a surrogate that
is merely CLOSE to the true dynamics... something about that residual
imperfection is catastrophic" and describe the result as being "about
what identification learns" without distinguishing two different
things that changed at once. Not silently rewritten (this project's
own convention) - correction recorded here, original text stays.

**The actual logic.** M0_S4 differs from a real M3 checkpoint in TWO
ways simultaneously: (1) exact I/O map (`~1e-17` vs M3's own
`~1e-6` Markov error) and (2) zero OBSERVABLE internal state (Task 4's
`obs_norm` exactly 0 for M0_S4 by construction, vs `~10-20`, real and
substantial, for trained M3). Part B/C's result - M0_S4 controls
identically to the true plant - cannot by itself distinguish which of
these two changes did the work, because both changed together. What
resolves the ambiguity is evidence already in this document from
BEFORE this session: M3's Markov error is already `~1e-6`, and this
project established early on (the 2026-08-12 "kink" investigation) that
errors this small, propagated through a system with `rho~1.02`, would
compound to only `~5e-5` over 200 steps under ordinary error
propagation - nowhere near enough to explain a 300x-700,000x cost
blowup on its own. **Wrong I/O content was already a poor explanation
before Part B ran.** That leaves realization (specifically, the
observable spurious modes Task 4 measured) as the surviving, not yet
refuted, candidate - Part B/C confirm the S4/BPTT graph is safe to
train through GIVEN a correct realization; they say nothing about
whether M3's specific, incorrect realization is what's causing the
failure. That question is what the balanced-truncation experiment
(below) is designed to answer directly, by changing ONLY the
observability-of-spurious-modes variable while holding M3's own
(already-established-near-exact) I/O content fixed.

**Precise statement of the M0_S4/M3 structural difference, verified
directly, not just inferred:** the `~1024` real-dimensional S4 hidden
state is NOT absent in M0_S4 - `S4LayerEnsemble`'s own recurrence
(`Lambda`, `B`, `P`, none of which `tools/controller_m0_s4.py` zeroes)
still runs and still evolves the state every step, driven by
`norm(skip)`. What's zeroed is every path FROM that state back to the
output: `out.kernel=0`, `out.bias=0`, `C_real_imag=0` together mean
`Abar[:6, 6:]` (Task 4's `obs_norm` block exactly) is identically zero
for M0_S4, not merely small. **The M3-vs-M0_S4 difference is
observability of a fixed-size (1030-dim augmented) mode set, not the
presence or absence of that mode set.** M0_S4's internal state is
present but permanently disconnected from anything that matters;
M3's is present and, per Task 4, has real (`~10-20`) one-step leverage
on the physical output.

**How to apply:** do not cite Task 2/Part C as having "refuted" or
"ruled out" the realization/spurious-modes mechanism - it refutes only
the S4-architecture-in-the-abstract and BPTT-through-`decode=True`
mechanisms. CLAUDE.md §1 updated to match.

---

## 2026-08-13 — Housekeeping resolved: cases 2/4's training-instability is a minority-exploding-member phenomenon (1/5 seeds each), confirming Task 2a's hypothesis directly

`tools/case24_instability_check.py`, sha 3c3563f: fresh M3 identification
for cases 2 and 4 alone, 5 seeds each, full standard curriculum,
PER-SEED (not aggregate) DPC loss recorded at every epoch of the N=200
phase - the phase where Task 3's original 15-member ensemble saw mean
loss spike to 6.9e13 (max 1.04e15).

```
case  seed  first_epoch  max_over_phase  final_epoch  exploded(>1e6)
 2     0        66.44          69.95          7.96        False
 2     1        29.59          29.59         14.86        False
 2     2       384831207802065.5   9214038371972132.0   82608989546421.9   TRUE
 2     3       195.46         195.46          9.48        False
 2     4        32.97          32.97         11.31        False
 4     0     3746785415.08    45457869298.9   29547797.8   TRUE
 4     1       315.04         315.04         49.21        False
 4     2       296.77        2456.03          28.63        False
 4     3      4438.57        4438.57         253.22        False
 4     4      5617.76       30117.63          91.26        False
```

**Confirms Task 2a's hypothesis (sha cf86e51) exactly: 1/5 seeds
exploded for EACH case (case2/seed2, case4/seed0) - a minority-
exploding-member phenomenon, not a reproducible property of cases 2/4
as a whole.** The other 8 of 10 seeds across both cases stay in the
tens-to-thousands range for the entire N=200 phase, the same order of
magnitude as the "trivial" cases' DPC loss - completely unremarkable.
The two that do explode reach truly extreme values (9.2e15, 4.5e10),
matching this document's long-established Adam/overparameterization
random-walk signature (D-only null-space work, the 10-seed
identification sweep's case6 divergences) rather than looking like a
different mechanism specific to these two cases' dynamics.

**Resolved, per the housekeeping item's own question ("reproducible
property or one exploding member"): one exploding member, not a
reproducible property.** Task 3's original aggregate-ensemble read
("M3 shows training-time BPTT instability... case2/case4 outliers")
should be understood as: most M3 seeds train stably through control
BPTT on every case tested so far, including 2 and 4; a small,
non-zero fraction (here 2/10 = 20%, consistent with though not
identical in rate to case6's ~50-60% identification-side divergence
rate) explode catastrophically regardless of which case they're
trained on. This is now the third independent context (D-only,
10-seed identification, control BPTT) this exact signature has shown
up in - it looks like a general property of training this S4
parameterization with Adam, not something tied to control, DPC, or any
particular case's dynamics.

GPU: 51.46 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 5: k-step free-running identification does NOT fix DPC at any k tested - but this is confounded by k-step training itself getting harder to converge as k grows at the (necessarily reduced) budget used, so it is not the clean "fidelity improves, DPC still fails" negative it could have been

`tools/identify_kstep.py`, sha 3c3563f (bugfixed after an x64/complex64
dtype crash - see the commit two entries back): M3, all 7 cases, 5
seeds, k in {1,5,10,20} (k=1 = teacher-forced anchor every step,
equivalent in spirit to today's baseline but a fresh implementation via
this script's own decode=True chunked loss, not identify.py's conv-mode
path - not bit-comparable to the standard 40k-epoch numbers, see
below), 2000 epochs (fewer than the standard 40k - a chunked
free-running loss needs a genuinely sequential 100-step-per-epoch
Python loop, unlike conv mode's one parallel call, so this budget was
reduced up front to keep 4 k-values' worth of training tractable -
flagged in the script's own docstring before this run, not discovered
after). Total wall time 316.4 min.

**DPC control (3 seeds/case, halved curriculum - also reduced in
advance for the same tractability reason): cost ratio to oracle stays
in the 1e2-1e6x range at every k, no case shows a clear trend toward
the oracle as k increases:**

```
case    k=1        k=5        k=10       k=20
 1     117.4x      69.7x     397.5x     563.5x
 2   112,600x   127,410x   114,270x   100,710x
 3     542.8x     581.1x   2,030.5x   1,677.7x
 4   184,890x   160,910x   391,600x   372,620x
 5   502,530x 1,080,700x 1,433,900x 1,083,900x
 7   3,388.4x   6,519.6x   4,171.9x  11,267.0x
```

Every k=1 number lands within the same order of magnitude as the
established M3 baseline for that case (`docs/controller_surrogates_
summary.csv`: 323x/55903x/310x/121942x/466718x/494x for cases
1/2/3/4/5/7) - a reasonable, consistent starting point, even though
this is a different identification run (this script's own chunked
k=1 loss, not identify.py's standard path) so exact agreement isn't
expected. **No k value gets meaningfully closer to the oracle than k=1
on any case - if anything, k=5/10/20 are flat-to-worse more often than
better** (case 1 dips to 69.7x at k=5 then rises again; every other
case either stays flat or climbs with k).

**Important caveat, checked before trusting the DPC result at face
value - and it changes how strongly this negative should be read:**
`teacher_mse` (the SAME chunked loss being optimized) systematically
WORSENS as k grows, at this budget, on most cases:

```
case    k=1        k=5        k=10       k=20
 1    4.2e-03     4.9e-02     3.3e+00     7.1e+00
 2    8.4e-03     3.7e-02     3.9e-01     8.0e+01
 3    5.7e-03     6.7e-02     1.7e+02     9.9e+04
 4    1.7e-02     1.2e-01     1.3e+00     1.5e+02
 5    1.2e-02     2.4e-02     3.4e-02     8.9e-02
 7    5.0e-03     3.7e-02     1.1e-01     2.2e-01
```

`openloop_rmse` (free-running, cold-start, same trajectory) shows the
same pattern - WORSE at k=10/20 than at k=1 or k=5 on cases 1/3/4,
case 3 catastrophically so (rmse 4.8 -> 8690 -> 19187 at k=1/10/20).
**This is not "free-running loss converges fine and DPC still fails
regardless" - it is "free-running loss, at a budget necessarily
reduced for tractability, does not even reliably converge as k grows,
and DPC fails regardless."** The longer within-chunk free-running
horizon at higher k is itself a harder BPTT-through-imperfect-dynamics
problem (exactly this project's own repeatedly-demonstrated fragility,
now showing up in identification training, not just control training),
and 2000 epochs - already 20x less than the standard 40k - was not
enough to converge it, especially past k=10.

**Honest verdict, not overclaimed:** this rules out the WEAKEST version
of the free-running-loss fix ("as easily achievable as teacher-forced
training, drop-in", at this budget) but does NOT cleanly rule out the
STRONGER version the brief was actually asking about (free-running
loss, properly converged, still fails at DPC) - that would need a
budget large enough to converge k=10/20 as well as k=1 does, which
this run's own numbers show it did not reach. Given every k value
still fails DPC by 2-6 orders of magnitude regardless, and given this
session's Task 2 result already answers the higher-order question
this fix candidate was aimed at (the S4/BPTT machinery is innocent;
Task 4 already shows WHY teacher-forced M3 has hundreds of spurious
modes) - re-running k-step at the full 40k-epoch budget is not
prioritized further this session; flagged as the natural next step if
this fix candidate needs a truly decisive answer.

GPU: 316.38 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 2 Part C: the full 6-case x 5-seed confirmation lands - M0_S4 matches the M0/M1 oracle baseline almost exactly on every single case, not just case 3

`tools/controller_m0_s4.py` rerun on a fresh kernel slug (sha 5ebb37e;
Kaggle 409'd on re-pushing the same slug immediately after the Part A/B
run completed - worked around, not investigated further) after the
per-case-batching OOM fix. All 30 members (6 cases x 5 seeds) completed
with no errors. `docs/controller_m0_s4_summary.csv`.

```
case   M0/M1 median   M0_S4 median   M0_S4 range
  1       1.0047         1.0047      [1.0042, 1.0052]
  2      41.1714        36.9993      [32.10, 52.23]
  3       1.0215         1.0215      [1.0215, 1.0225]
  4       1.0633         1.0638      [1.0628, 1.0648]
  5       2.2800         2.2800      [1.0802, 13.8001]
  7       1.0041         1.0038      [1.0036, 1.0044]
```

**Identical to 4 decimal places on cases 1, 3, 5; identical to 3 on
case 4; case 7 differs only in the 4th decimal; case 2 is the sole case
with a visible gap (37.0x vs 41.2x), and it is well inside case 2's own
already-documented seed-to-seed variance** (this document's 2026-08-13
saturation entries: case 2's per-seed spread was already flagged as
unusually wide even before this session, "1.92x/30.27x/128.05x - a
~67x range" at a wider action bound; M0_S4's own 5-seed range here,
32.10-52.23x, is the same qualitative pattern at a milder scale, not a
new discrepancy this run introduced). No confound: saturation stayed
at or near zero for every case except 2 (max 7.3%, matching M0/M1's
own known saturation profile there, not something M0_S4 introduces).

**This is no longer a single-case result - it holds across every one
of the 6 control cases, each at >= 5 seeds, matching the M0/M1 oracle
essentially exactly. Combined with Part B's cap-by-cap agreement on
case 3, the verdict from two entries ago is now fully confirmed at the
statistical power the session brief asked for, not just suggested by
one case:** training a BoundedGRUController via BPTT through an S4
realizing the exact true dynamics, deployed through the identical
decode=True/`jax.lax.scan`/`rollout_learned` machinery M3 and M6 use,
is indistinguishable from training directly through `(A_d, B_d)`. The
300x-700,000x M3/M6 failure is not a property of "being an S4
surrogate," of BPTT through S4, or of the control/training machinery in
any form tested. It is entirely a property of what identification
LEARNS when it has to fit the dynamics from data rather than being
handed them exactly - and Task 4's ~300 spurious near-unit modes
(present in every real M3 checkpoint, absent by construction in M0_S4)
is this document's best current candidate for what that learned,
non-exact difference actually is.

GPU: 243.84 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 6, oracle half: M0/M1 horizon sweep is clean and monotonic on every case at proper (5-seed) power, with a striking new case-4-specific short-horizon story

`tools/horizon_sweep_oracle.py`, sha 9ecbeab: `rollout_linear` (M0=true
`(A_d,B_d)`, M1=least-squares - identical to machine precision as
established throughout this document), all 6 control cases, 5 seeds,
caps {5,20,50,100,200}. 300 rows, `docs/horizon_sweep_oracle.csv`. No
errors. M0 and M1 agree to the same ~10 significant figures established
elsewhere in this document (`fit_least_squares`'s ~1e-14 floor) - listed
once below, not twice.

```
case    cap=5      cap=20     cap=50     cap=100    cap=200
 1      1.210x     1.021x     1.007x     1.005x     1.005x
 2     17.515x     6.024x     4.733x     3.078x    37.00x    (!)
 3      4.156x     2.650x     1.479x     1.040x     1.022x
 4   1120.4x      14.088x     1.110x     1.071x     1.064x   (!!)
 5     23.625x    12.882x     9.523x     5.530x     1.341x
 7      1.216x     1.024x     1.009x     1.005x     1.004x
```

**Cases 1, 3, 5, 7: clean, monotonic decrease with horizon - exactly
what a normal learning curve looks like**, and case 3's numbers here
(4.156/2.650/1.479/1.040/1.022) match Part B's independent case-3-only
run (4.156/2.552/1.4495/1.038/1.0215) to 2-3 significant figures -
good agreement between two independently-run sweeps of the same
quantity.

**Case 4 is the new, striking finding this proper sweep surfaces: at
cap=5 the median ratio is 1120x (worst seed: 12,920x), then it
collapses to 14.1x at cap=20 and is within 10% of oracle by cap=50.**
Case 4 is this ladder's second-highest-Kreiss case (3.30, per the
identification-side tables) - at a 5-step optimization horizon, an
LQR-style short-horizon objective apparently cannot see enough of a
non-normal, Kreiss-amplifying plant's transient behavior to avoid a
catastrophically bad policy, but the true plant (unlike any surrogate
in this document) resolves this almost entirely once given horizon
>=50. This is a genuinely new result, not visible in the single-seed
case-3-only sweep this replaces, and it means "short-horizon training
transfers badly" is not unique to the surrogates - it can happen
through the TRUE plant too, on a case with the right transient-
amplification structure, and self-corrects with more horizon there in
a way M3 (next entry) does not.

**Case 2's cap=200 uptick (3.08x at cap=100 -> 37.0x at cap=200) is
real, not noise** - it matches this document's own already-established
finding that case 2 has unusually wide seed-to-seed variance (the
2026-08-13 saturation-phase-2 entry, and Task 2 Part C two entries
above) independent of horizon; not investigated further here per that
entry's own flag.

Awaiting the M3/M6 half (`tools/horizon_sweep_surrogate.py`, running)
before drawing the full comparison the brief asked for (does the
single-seed M3 non-monotonicity survive proper seed counts).

**EXCLUSION RULE, flagged before the surrogate half lands so it's
applied consistently, not decided post-hoc:** case 4 at cap=5 fails
through the TRUE plant (1120x median, worst seed 12,920x) by the exact
same logic that excludes case 6 from control comparisons entirely
(CLAUDE.md/this document, 2026-08-13 Task 2 kill-criterion entries) -
if the ORACLE cannot stabilize a case at a given horizon, a surrogate's
number at that same (case, horizon) carries no information about the
surrogate, only about BPTT-through-that-horizon-through-any-model.
Unlike case 6 (excluded at every horizon), case 4 resolves to
near-oracle by cap>=50, so the exclusion is scoped to cap=5 only for
case 4, not the whole case. Any M3/M6-vs-oracle ratio at case4/cap5
must be marked as uninterpretable, not averaged into a headline
number, when the surrogate half's results are written up.

GPU: 139.56 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — Task 6 surrogate half: first attempt OOM'd exactly as predicted, fixed, rerunning

`tools/horizon_sweep_surrogate.py` (pre-fix version, sha 9ecbeab) was
built before Task 2 Part C's OOM fix existed and carried the same risk
(grouping cases sharing a `max_action` bound into one batch, up to 5
cases x 5 seeds = 25 members). It ran cleanly through M3's caps
5/20/50/100 (all 6 control cases, 5 seeds), then hit the identical
`RESOURCE_EXHAUSTED` (16.6GB requested) at M3's cap=200 N=200 phase -
M6 never started. 132.71 T4-min spent before the crash. Fixed (sha
9ecbeab, same per-case-batching pattern as Task 2's fix) and relaunched
on a fresh kernel slug (`s4dpc-horizon-sweep-surrogate-v2` - re-pushing
the same slug 409'd, same as Task 2 Part C's rerun; worked around the
same way). Results in the next Task 6 entry once that run completes.

GPU: 132.71 T4-min (crashed run). Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — The deciding experiment: truncated M3 is WORSE than full M3 on every case, not better - but the truncation itself introduces a real ~1e4x fidelity loss M3's own realization apparently cannot avoid, so this does NOT cleanly settle the mode-based story either way

`tools/balanced_truncation.py`, sha b9ca367: self-check on the TRUE
`(A_d,B_d)` (all 6 control cases) reconstructs it via Hankel-SVD/ERA at
machine precision (1.9e-16 to 8.2e-15) with an EXACT rank-6 cliff in
the Hankel singular values (7th/8th singular values exactly 0) -
already confirmed locally before this run, reproduced here on GPU.
Fresh M3 identification (6 control cases, 5 seeds, 40k epochs, 85.9s,
zero divergence). No errors, 43.5 min total.

**M0_S4 observability, confirmed exactly as stated in the correction
two entries above:** `max|Abar[:6,6:]|` = exactly 0.0 for M0_S4 on
every case checked - not merely small.

**Lambda_re-clip check: partially an initialization artifact, but
training roughly doubles it - not purely either story.** Fresh,
UNTRAINED M3 (3 seeds, cases 1/3) already shows 122-182 near-unit
modes (of 1030) before a single gradient step - so the S4
parameterization's own init (via `Lambda_re`'s `<=-1e-4` clip) already
produces a substantial population of near-marginal modes, not
something training invents from nothing. But Task 4's TRAINED numbers
(264-369 median per case) are consistently ~1.5-2x higher than these
untrained figures, and untrained `rho(Abar)` is wildly noisy across
just 3 seeds (1.01 to 1.91) - training also clearly adds to and
reshapes the population, it does not merely inherit it unchanged. Both
things are true at once; neither the "pure artifact" nor "purely
learned" framing survives contact with the numbers.

**The truncation itself does NOT recover M3's own ~1e-6 fidelity - it
lands at ~1e-2, four orders of magnitude worse, and the Hankel
singular values explain exactly why:**

```
        M3's median HSV, indices 0-14 (all cases/seeds pooled):
idx:     0      1      2      3      4      5      6      7      8      9     10     11     12     13     14
HSV:  0.561  0.263  0.156  0.121  0.088  0.061  0.049  0.033  0.026  0.020  0.015  0.012  0.009  0.008  0.006
```

Unlike the true system's EXACT cliff to 0 at index 6, **M3's spectrum
decays smoothly with no cliff anywhere** - index 6 (0.049) is barely
smaller than index 5 (0.061), and meaningful energy (hsv[14]=0.006, ~1%
of hsv[0]) remains far past index 6. Per-case median reconstruction
error at r=6: `err_vs_m3_markov` 1.3e-2 to 3.3e-2 (vs M3's own
teacher-forced Markov error, ~1e-6, established in the 10-seed
identification entries), `err_vs_true_markov` 2.6e-2 to 5.6e-2 -
consistently worse than `err_vs_m3_markov`, as expected (ERA
reconstructs from M3's own data, not the true plant's). `cond(Cr)`
(the output-normal-form transform) ranges 13-377, usable but not
uniformly clean.

**DPC through the truncated system: catastrophic, and WORSE than full
M3 on every one of the 6 control cases, not better:**

```
case   full M3 (established)   truncated M3 (median, 5 seeds)   ratio (trunc/full)
  1          323x                     2340x                          7.2x
  2       55,903x                   173,360x                         3.1x
  3          310x                     3,107x                        10.0x
  4      121,942x                 1,563,200x                        12.8x
  5      466,718x                 5,903,800x                        12.7x
  7          494x                    20,214x                        40.9x
```

**Verdict, stated as precisely as the confound allows: the optimistic
branch is cleanly falsified - truncated M3 does NOT land near oracle,
not on any case, by a wide margin. But this experiment cannot cleanly
distinguish "spurious modes are innocent" from "the truncation's own
~1e4x fidelity loss is itself sufficient to explain even-worse DPC
failure," because both changed at once.** M1's fidelity (~1e-14) gives
~1x; M3's own teacher-forced fidelity (~1e-6) gives 300x-700,000x;
truncated-M3's fidelity (~1e-2, an unavoidable side effect of the
truncation itself, not a choice) gives 2,340x-5,900,000x - worse
fidelity tracking worse DPC outcome across this three-point comparison
is at least as consistent with "fidelity still matters, non-linearly"
as with "spurious modes are what's really driving this." The mode-
count story is NOT rescued by this result, but it is not cleanly killed
either - what IS killed, unambiguously, is the simple "remove the junk,
get the good realization back" picture, because **M3's own realization
does not have a clean 6-good-dimensions-plus-junk decomposition to
recover in the first place.** The smooth (uncliffed) Hankel spectrum is
itself the finding: M3's learned dynamics are genuinely spread across
far more than 6 directions, not concentrated in 6 good ones sitting
next to ~300 separable spurious ones. Task 4's spurious-mode count and
this entry's smooth-HSV-spectrum finding are both real, both
architecturally clean measurements, and both point at the same
underlying picture - M3's realization is diffuse, not neatly
decomposable - without either one, on its own, causally explaining the
DPC failure.

**What would cleanly separate the two explanations, not attempted
here:** a truncation-quality-matched control - e.g., truncate to a
LARGER r (the smallest r at which `err_vs_m3_markov` reaches ~1e-6,
whatever that turns out to be) and check whether DPC improves as r
grows toward full M3's own fidelity floor, or fails just as badly even
once fidelity is matched. If it fails just as badly at matched
fidelity, that would be a much cleaner argument for realization/
structure over raw Markov error than anything in this document so far.
Flagged as the natural next step, not run this session.

GPU: 43.46 T4-min. Logged in `gpu_ledger.csv`.

---

## 2026-08-13 — CORRECTION: "the optimistic branch is cleanly falsified" overstated what this experiment can support - it decided nothing about the spurious-mode hypothesis, and should not be read as having tested it

Flagged by the user, correct on inspection. Not silently rewritten
(this project's convention) - the entry above stands as written; this
is the correction.

**The problem, stated plainly:** the confound isn't a minor caveat on
an otherwise-decisive result - it removes the result's evidentiary
value for the question the experiment was built to answer. Truncated
M3 has ~1e-2 Markov fidelity, four orders of magnitude worse than
M3's own ~1e-6. This project already established, before this
experiment ever ran, that fidelity and DPC outcome move together
somehow: M1 (~1e-14) gives ~1x, M3 (~1e-6) gives 300x-700,000x. A
system at ~1e-2 - worse than M3 by as much as M3 is worse than M1 -
failing DPC even more badly is close to the DEFAULT PREDICTION from
data already on record, not new evidence about whether spurious modes
specifically are causal. **Calling the optimistic branch "cleanly
falsified" - even hedged in the same paragraph - overstates what a
confounded measurement can support, because "truncated M3 fails" was
already the likely outcome for a reason that has nothing to do with
spurious modes.** The word "falsified" implies the experiment
discriminated between hypotheses; it did not, because it never
isolated the variable it was designed to isolate.

**What survives this correction, unconfounded, because it is a
measurement of M3's OWN Hankel spectrum, not of the truncated system's
downstream DPC performance:** M3's Hankel singular values decay
smoothly with no cliff at (or near) rank 6, unlike the true system's
exact cliff. This is real, was measured directly, and does not depend
on how the truncated system subsequently performed. It says M3 has no
clean 6-good-dimensions-plus-junk decomposition available to recover -
a fact about M3's realization, standing on its own.

**What does NOT survive:** any claim that this experiment tested,
weighed in favor of, or weighed against "spurious modes are causal."
It did neither. The DPC numbers in the entry above are accurately
reported and worth keeping on record, but they answer "does an
uncontrolled-fidelity-loss truncation control well" (answer: no,
unsurprisingly), not "are the spurious modes themselves the cause of
M3's DPC failure" (answer: still open).

**The only experiment that would actually answer the intended
question is the fidelity-matched truncation already flagged at the end
of the entry above** - find the smallest r where `err_vs_m3_markov`
reaches M3's own ~1e-6 floor, and see whether DPC improves as r grows
while most spurious modes stay excluded. Until that runs, this
document has no experiment that speaks to the spurious-mode causal
question one way or the other, and should not be cited as if it does.

---

## 2026-08-13 — k-step at matched budget: even 5x the epochs does NOT rescue convergence - 46% of seeds diverge outright, and DPC improves only marginally (still catastrophic). This makes Task 5's original negative MORE robust, not less.

`tools/identify_kstep_matched.py`, sha 46fae0c: k=10, all 7 cases, 5
seeds, 10,000 epochs (5x Task 5's original 2000). No errors, 142.05
T4-min (2.37hr - faster than the ~6.6hr estimate).

**Divergence rate at k=10@10k epochs: 16/35 members (46%) have
`teacher_mse > 1.0`, several catastrophically** (`teacher_mse` up to
2.8e9, `openloop_rmse` up to 7.4e20, `rho(Abar)` up to 1.66 - a
genuinely unstable augmented operator, not just a slow one):

```
case   median_teacher_mse   k=1 baseline (40k epochs)   n_diverged(>1.0)
  1        5.44e-02              2.77e-06                    2/5
  2        7.18e-03              8.11e-06                    2/5
  3        4.59e+01              2.76e-06                    3/5
  4        2.73e-02              1.12e-05                    2/5
  5        6.28e-04              3.36e-06                    1/5
  7        9.51e-04              3.11e-06                    1/5
```

**Even the non-diverged majority never gets close to the k=1
baseline's floor** - best case-level medians land at ~1e-3 to ~1e-4,
still 2-3 orders of magnitude above the standard teacher-forced
path's ~1e-6, despite 5x the gradient steps. This is not "slow
convergence that more steps would eventually fix" - a near-half
divergence rate combined with a floor that doesn't move much even
where it doesn't diverge is the same qualitative signature this
document has repeatedly identified for Adam-on-overparameterized-S4
(the D-only gauge-symmetry work, the case-6 identification
divergence, the cases-2/4 control-training explosions) - chaining k
free-running steps before re-anchoring appears to make this MORE
likely to trigger, not merely slower to escape.

**DPC control (halved curriculum, 3 seeds, unchanged from Task 5 -
this rerun targeted the identification budget specifically): ratios
improve modestly but stay catastrophic:**

```
case   k=10@2000epochs (Task 5)   k=10@10000epochs   improvement
  1          397.5x                    103.5x             3.8x
  2       114,270x                  65,903x               1.7x
  3        2,030.5x                  1,323.2x              1.5x
  4      391,600x                 232,620x                 1.7x
  5     1,433,900x                473,540x                 3.0x
  7        4,171.9x                 3,211.4x                1.3x
```

**Verdict: this makes Task 5's original negative result MORE robust,
not less.** A 5x budget increase produced a 1.3-3.8x improvement in an
already-catastrophic DPC ratio while identification itself became, if
anything, a more visibly unstable optimization problem (46% outright
divergence). Extrapolating the trend, closing the remaining 2-6 orders
of magnitude to oracle would need a budget increase far beyond what
any further single-session Kaggle run could supply - and there is no
evidence in this data that more budget would even get there, given
divergence rate did not visibly improve with more steps. The honest
reading: k-step free-running identification, as implemented, is not
merely under-provisioned at low budgets - it appears to be a
genuinely harder, more divergence-prone optimization problem than
teacher-forced training, and DPC fails by a similar order of
magnitude regardless of how much of that harder problem gets solved.
Combined with Task 2's result (the S4/BPTT machinery is innocent when
fed exact dynamics) and Task 4's finding (spurious modes are real and
substantial even at M3's OWN near-exact teacher-forced fidelity), this
fix candidate does not look like it is chasing the right variable.

GPU: 142.05 T4-min. Logged in `gpu_ledger.csv`.

## 2026-08-14 — Task 6: proper horizon sweep, M3/M6, 5 caps x 6 cases x 5 seeds

sha: (pending commit) | kaggle: s4dpc-horizon-sweep-surrogate-v2 | `docs/horizon_sweep_surrogate.csv`

Full factorial re-run of the curriculum-horizon-cap sweep this task originally
asked for: caps `{5, 20, 50, 100, 200}` x cases `{1,2,3,4,5,7}` (case 6 excluded
per the standing rule) x 5 seeds, both M3 and M6, evaluated against the same
LQR-oracle cost baseline as every other control-side result this session. 300/300
runs finite - no divergence, no OOM (the per-case-not-per-bound batching fix from
Task 2 Part C held).

**Median `cost_ratio_to_oracle`, per (variant, cap, case):**

```
        cap=5        cap=20       cap=50       cap=100      cap=200
case1 M3  83x    M3  74x     M3  89x     M3  86x     M3  81x
      M6  76x    M6  40x     M6  37x     M6  37x     M6  18x
case2 M3 5.4e4   M3 5.2e4    M3 3.7e4    M3 3.1e4    M3 5.1e4
      M6 3.9e4   M6 4.3e4    M6 4.8e4    M6 4.7e4    M6 3.8e4
case3 M3  344x   M3  539x    M3  553x    M3  407x    M3  330x
      M6  904x   M6 1180x    M6 1374x    M6 1138x    M6 1636x
case4 M3 8.6e4   M3 8.1e4    M3 7.5e4    M3 1.0e5    M3 1.1e5
      M6 2.7e5   M6 1.8e5    M6 1.6e5    M6 2.3e5    M6 3.3e5
case5 M3 1.1e6   M3 3.2e5    M3 3.4e5    M3 2.4e5    M3 2.4e5
      M6 8.2e5   M6 5.7e5    M6 5.6e5    M6 2.6e5    M6 8.6e5
case7 M3  513x   M3  420x    M3  825x    M3  748x    M3  578x
      M6  539x   M6  463x    M6  636x    M6  595x    M6  780x
```

**No cap in this range rescues either variant on any case.** Ratios stay within
roughly a factor of 2-3 of each other across the full cap range per (variant,
case) cell - there is no monotonic trend with cap, in either direction, on any
case. This directly answers Task 6's original question: the catastrophic M3/M6
DPC failure documented earlier this session is **not** a curriculum-truncation
artifact that a longer/shorter horizon cap would fix - it is stable across a
40x range of caps. M3 and M6 remain within the same order of magnitude of each
other on every case (neither is systematically much better), consistent with
this session's central finding that the kink (M6-only) is not what's driving
the failure, since M3 (zero kink by construction) fails just as hard.

GPU: 621.51 T4-min (~10.4h - the single largest job this session; already
running before the weekly quota was hit, so exempt from the block per CLAUDE.md
§4's "already-running kernels are not blocked" reading). Logged in
`gpu_ledger.csv`.

## 2026-08-15 — nu-gap / robust-margin: full table, M1/M0_S4/truncM3/fullM3, 6 cases x 5 seeds

sha: (pending commit) | kaggle: s4dpc-nu-gap-export | `docs/nu_gap_export_summary.csv`,
`docs/nu_gap_analysis.csv`

Wave 2/3's central ask, finally completed after three lost Colab attempts (see
CLAUDE.md's `launch/colab/orchestrate.py` section): `tools/nu_gap_export.py` (GPU,
Kaggle, 389.7 T4-min) trained/evaluated all four variants and exported
`(A,B,C,K_eff)` per (variant, case, seed); `tools/nu_gap_analysis.py` (CPU-only,
pure numpy/scipy, no jax, seconds) computed `delta_nu`, `b`, and a 10-step Markov
error against the true plant for all 120 rows.

**Controls behave exactly as expected.** M1 and M0_S4 both give `delta_nu~0`
(machine-precision-level Markov error, 1e-16 to 1e-19) on every case, and
`b > delta_nu` agrees with the observed DPC outcome on 29/30 rows each (96.7%).
The one disagreement in each (case 5, seed 3, ratio=13.8x) is a borderline call
against an admittedly arbitrary `ratio<10x = success` threshold, not a real
theory failure - 13.8x sits far closer to the 1-4x range every other case-5 seed
lands in than to any catastrophic M3-scale failure.

**b > delta_nu agrees with the observed outcome on 100% of truncM3/fullM3 rows
(60/60) - but this number is weaker than it looks, and I want to say why rather
than just report it.** Two things undercut treating this as a clean
confirmation:

1. **It only ever predicts failure, because failure is the only outcome M3-based
   DPC ever produces in this entire session.** There is no successful M3/truncM3
   case anywhere to test whether the criterion could ALSO correctly predict a
   success - "100% agreement" here means "100% agreement with the majority
   (only) class," not genuine two-way discrimination.
2. **In the large majority of these 60 rows `b` is exactly 0.0** - the trained
   controller's own closed-loop linearization (`A + B@K_eff`) isn't even
   locally Schur-stable against the plant it was trained on. That's a much
   cruder signal than the coprime-factorization nu-gap comparison; you don't
   need `delta_nu` at all to predict failure when `b=0` trivially loses to any
   `delta_nu>=0`. In the minority of rows where `b` IS nonzero (truncM3 only,
   9/30 - fullM3 is `b=0.0` in all 30), it's still 2+ orders of magnitude below
   `delta_nu`, so the criterion still correctly predicts failure, just via a
   less degenerate route.

**A genuine data-quality caveat, not swept under the rug: `delta_nu`'s own
validity check fails on most truncM3/fullM3 rows** (`dnu_valid=False`,
reported as the 1.0 saturation fallback rather than a computed value - see
`nu_gap()`'s winding-number consistency guard, `|wno + eta_Phat - eta_P| < 0.4`).
Working hypothesis: M3's augmented realization carries ~300 near-unit-circle
modes (Task 4, this session) against the true plant's 3-6, and that large,
asymmetric unstable/marginal-pole-count mismatch is exactly the kind of thing
that can break a winding-number-based consistency check - M0_S4 is also
1030-dim but has ZERO such spurious modes by construction, and its `delta_nu`
validates cleanly on every row. Where truncM3/fullM3's `delta_nu` DOES validate
(9/60 rows), it still lands close to the cap (0.58-0.9995), which is reassuring
that the fallback isn't hiding something qualitatively different - but I can't
prove that for the invalid rows specifically, and the 1.0000 figure printed for
most of them is a saturation flag, not a precise computed gap.

**Overall verdict: the b > delta_nu theory is never contradicted by this
dataset, and cleanly explains the M1/M0_S4-vs-M3 split - but it does not give
the graded, case-by-case severity signal the original ask (Wave 2, item 3) was
hoping to get by moving past the saturated mode-count correlation.** Both `b`
and `delta_nu` saturate to boundary values (0 and ~1 respectively) for every
M3-based row, so the test degenerates to a binary confirmation of something
already established many times this session (M3-based DPC always fails, M1/M0_S4
never do), not a new discriminator of WHY some cases fail 300x and others fail
700,000x. That's the same structural problem the mode-count correlation had,
reached by a more theoretically principled route this time, not a different
outcome.

GPU: 389.70 T4-min (`s4dpc-nu-gap-export`). CPU analysis: seconds, no GPU spend.
Logged in `gpu_ledger.csv`.

## 2026-08-15 — CORRECTION + mechanism: b=0 is the finding, not a saturation artifact; traced to source

**Correction to the entry above.** The previous entry treated `b=0.0` on
60/60 M3-based rows as a degenerate nuisance that made the `b > delta_nu`
agreement number look stronger than it was. That was wrong. `b` is the
robust-stability margin of the closed loop the trained controller forms with
the model it was TRAINED on. `b=0` on 60/60 M3-based rows, against `b>0` on
29/30 M1 rows and 29/30 M0_S4 rows, means: **DPC training, after full
convergence on a calm/decreasing loss curve, produced controllers that do not
stabilize the very model their cost was computed through - every single time,
on every case and seed. M1/M0_S4 training essentially never has this problem.**
That IS the two-way discrimination the previous entry said was missing -
`b`'s sign alone predicts the M1/M0_S4-vs-M3 split perfectly, on data neither
the mode-count correlation nor the raw DPC ratio directly encodes. Restating
this as a positive result, not a saturation artifact.

**Verified directly, not asserted (`tools/verify_closed_loop_instability.py`):**
for M3 case 3 seed 0 (ratio=368x oracle), the closed-loop operator
`Abar + Bbar@K_eff` has spectral radius **1.0515**, with 32/1030 eigenvalues
genuinely outside the unit circle (worst: `1.0204+0.2541j`) - not a boundary
artifact at 1.0000001, a real 5% instability. Simulated 2500 steps from a
training-range initial condition: `||x||` goes from 1.78 to **1.4e+54** - 54
orders of magnitude of unbounded growth, well past the N=200 training horizon.
The M1 control (same case/seed): spectral radius 0.982, simulated state decays
to 1.2e-20 over the same 2500 steps. `b=0.0` is real.

**One honest wrinkle, not swept under it:** the SAME M3-trained K_eff, closed
around the TRUE plant instead of M3, has spectral radius **1.0055** - barely
over 1, one marginal complex pair. By my `robust_margin`'s exact `>=1.0` test
this would also register `b=0`, yet the REAL (nonlinear, tanh-bounded,
recurrent) controller performs at ratio~1.02x against the true plant in
practice - excellent. The static-K_eff-linearization proxy is a good, sharp
signal where the margin is large (M3: not close, 32 unstable directions), but
it is not infinitely precise right at the boundary, where the actual
nonlinear/bounded controller can still work fine despite a marginally-unstable
local linearization. Worth flagging since the whole `b`-based reading rests on
this proxy; it doesn't change the M3 verdict (nowhere close to marginal).

**Pole-count question (`tools/verify_closed_loop_instability.py`, all 6
cases x 5 seeds, fullM3): M3 has genuinely spurious UNSTABLE modes, not just
near-boundary ones - `delta_nu=1.0000` (the invalid/fallback value in
`nu_gap()`) is a correct reflection of a real structural mismatch, not a
numerical artifact.** Strict `|lambda|>1` count (no deadband): true plant has
0 (case 1, whose one non-stable pole sits exactly at z=1), 3 (case 7), or 6
(cases 2/3/4/5) unstable poles. M3's augmented operator has 2-19 per
(case,seed) - generally MORE than the true plant, including 4-12 in case 1
where the true plant has ZERO. This is exactly the kind of large, seed-variable
unstable-pole-count mismatch that breaks the nu-gap winding-number consistency
check (`|wno + eta_Phat - eta_P| < 0.4`) - not a bug in the check, a real
signal that these systems are topologically dissimilar in a way the metric is
supposed to catch.

**Where do the spurious unstable modes come from, given `Lambda_re` is
clipped? Traced to source (`tools/check_lambda_re_clip.py`, calling s4-nnx's
own `discrete_dplr` directly on all 30 checkpoints - no re-derivation).** The
clip (`jnp.clip(Lambda_re, None, -1e-4)` in s4-nnx's `S4LayerEnsemble.__call__`)
does exactly what it was designed to do: **every one of the 512 per-checkpoint
S4 channels, in every one of the 30 checkpoints tested, is stable in BOTH
continuous time and discrete time** - zero exceptions. Max per-channel discrete
spectral radius across the whole set: 0.9998 (near-boundary, matching the
"~300 near-unit-circle modes" figure from Task 4, but never over). The clip
only bounds the DIAGONAL part (`Lambda_re`); the actual state matrix is
diagonal-plus-low-rank (`diag(lambd) - P@P.conj().T`, P a free/unclipped
parameter) - checked this specific escape route directly (a non-Hermitian
diag doesn't get Weyl's-inequality protection from a Hermitian rank-1
correction) and it does NOT happen: 0/30 checkpoints have an unstable
continuous-time DPLR matrix either. (Raw `Lambda_re` values DO drift past the
-1e-4 boundary during training - 29-55 of 512 per checkpoint, up to +0.022 -
confirming the clip is actively binding and gradients vanish once a parameter
crosses it, but this drift is fully absorbed; it causes no instability.)

**The instability lives entirely in the composition, not any single
component.** `s4dpc/blocks.py`'s `ConfigurableBlock` wraps the (individually,
provably stable) S4 recurrence in an encoder (`Linear`, 9->16), a per-block
`out` projection (`Linear`, unconditional regardless of variant), and a
decoder (`Linear`, 16->6) - none of which carry ANY stability constraint.
Composing stable subsystems through unconstrained linear feedback does not
generally preserve stability (standard systems theory - the clip was never
positioned to prevent this, and structurally couldn't be from where it sits).
The augmented operator IS this whole composed loop, which is exactly why
`augmented_operator`/`robust_margin` see instability that no individual S4
channel exhibits.

**Horizon-blindness mechanism (`tools/mode_contribution_vs_horizon.py`, M3
case 3 seed 0): real, but more nuanced than "invisible until N=200."** The
single dominant unstable mode (lambda=1.0204+0.2541j, the same one identified
above) accounts for ~100% of the full closed-loop cost at EVERY horizon
tested, from N=5 to N=2000 - this is not a diffuse effect of ~300 modes, one
mode explains essentially all of it. Its (normalized) cost contribution:
N=5: 3.6, N=50: 26, N=100: 1890, N=150: 223,000, **N=200 (training cap):
3.2e7**, N=500: 1.1e20, N=2000: 1.5e85. It is NOT literally invisible even at
N=5 (already ~14% of a modest total cost there) - but in ABSOLUTE terms its
contribution is tiny relative to what it becomes, and the curriculum
(`tools/controller_oracles.py`'s `CURRICULUM`) gives the N=200 phase a full
2000 epochs, the same as the largest allocation - so this isn't simple
epoch-starvation either. The more accurate reading: the mode's growth is
exponential (`|lambda|^(2N)` for the quadratic cost), so its absolute
gradient PRESSURE stays negligible relative to the well-behaved loss terms
across most of the curriculum and only becomes overwhelming once N is large
enough - by which point training may already be in a basin 2000 further
epochs at N=200 can't escape. Falsifiable prediction this makes: a cost with
a non-vanishing terminal penalty or discount would surface this mode much
earlier in the curriculum. **Not yet tested**: an actual N=1000 retraining
run on case 3 (GPU) - proposed as the direct test, not yet run pending
sign-off given the standing budget-reporting convention.

GPU: 0 (all three checks pure CPU/numpy, reading existing exports + a
read-only local clone of s4-nnx's source for the discretization function -
no retraining, no jax execution on GPU).

## 2026-08-16 — N=1000 test on case 3: the simple fix does NOT work

sha: (pending commit) | kaggle: s4dpc-n1000-case3 (v2, after an OOM on v1) |
`docs/n1000_case3_summary.csv`

The falsifiable prediction from the horizon-blindness entry above, tested
directly. `tools/test_n1000_case3.py`: standard curriculum first (baseline,
reproduces the established result exactly - all 5 seeds b=0, ratios
37-2000x), then the SAME ensemble continued for one more phase at N=1000,
2000 epochs, a FRESH cosine LR schedule (so the extension isn't crippled by
the baseline schedule's already-decayed-to-0 tail), smaller x0 batch (100
vs the standard 1000 - `rollout_learned` unrolls its horizon via a plain
Python loop, not `jax.lax.scan`, and BPTT through 1000 steps at the full
batch hit `RESOURCE_EXHAUSTED` on a T4 on the first attempt before a single
epoch ran; the batch reduction fixed it, `git log` has both attempts, first
one's GPU time logged and kept per CLAUDE.md's "diverged runs are results"
rule).

**Result: extending to N=1000 does not fix it. Per-seed, baseline -> extended:**

```
seed   rho (N=200) -> rho (+N=1000)   n_unstable (N=200) -> (+N=1000)   b   ratio (N=200) -> (+N=1000)
  0        1.0528    ->   1.0486            31   ->   30              0.0     373x  ->    78.5x
  1        1.0179    ->   1.0166            29   ->   31              0.0    1999x  ->   1800x
  2        1.0149    ->   1.0155            10   ->   10              0.0     269x  ->    292x
  3        1.0313    ->   1.0313            18   ->   18              0.0      37x  ->     35x
  4        1.0584    ->   1.0560            17   ->   17              0.0     452x  ->    438x
```

Spectral radius and unstable-eigenvalue count are essentially UNCHANGED
across all 5 seeds - none crosses back inside the unit circle, `b` stays
exactly 0.0 in every seed both before and after. The raw DPC ratio moved a
little (seed 0 improved 4.7x, seed 1 improved slightly, seeds 2-4 flat to
marginally worse) - a small, seed-inconsistent wobble, not a fix.

**This was not a weak test - the optimizer clearly got the predicted signal.**
The extended-phase loss (dominated by the same mode `tools/
mode_contribution_vs_horizon.py` identified) started at 7.8e21 and dropped
~100x to 7.9e19 by epoch 2000 - the mode's cost WAS visible and WAS being
pushed on, hard, for the entire dedicated phase (no competing short-horizon
objective in this phase - N=1000 was the only term). Loss was still
decreasing at 2000 epochs (though flattening, and the fresh cosine schedule
was also intentionally decaying toward 0 by design at that point, so this
run alone can't distinguish "genuinely stuck" from "just needs more epochs
under a longer schedule").

**Reading: the horizon-blindness mechanism correctly describes why STANDARD
(N<=200) training never sees enough signal to fix this - but simply forcing
a longer horizon is not, by itself, a working fix, at least not within 2000
epochs of gradient descent under overwhelming, uncontested pressure.** That
argues against a pure "no signal" story and toward the encoder/decoder/out
composition (docs/DECISIONS.md's clip-tracing entry above) being a genuinely
hard region of the loss landscape for Adam to navigate out of from this
starting point - possibly a capacity question (can THIS architecture
represent a stable closed loop at all, holding the already-learned S4
channels fixed), possibly an optimization-difficulty question (the needed
change is a coordinated shift across many interacting unconstrained
parameters, and local gradient steps do not find it easily) - not yet
distinguished by this experiment. Not run: more epochs at N=1000, a longer
schedule, or freezing the (already individually-stable) S4 layer and
retraining only encoder/decoder/out - any of these could help pull the two
apart, but that's the next question, not answered here.

GPU: 313.39 T4-min total (157.05 min OOM'd first attempt + 156.34 min the
successful rerun). Logged in `gpu_ledger.csv`.

## 2026-08-16 — CORRECTION on b, confirmed M3 was frozen in the N=1000 test, and PBH refutes the uncontrollability hypothesis

**Correction: `b` is a binary stability LABEL, not a severity measure - both
that and the M1/M0_S4-vs-M3 class discrimination are true at once.** M3
case 3 seed 0's DPC ratio moved from 373x to 78.5x (a real, nearly-5x
improvement from the N=1000 extension below) while `b` stayed pinned at
exactly 0.0 throughout - `b`'s sign cannot and does not track that kind of
within-class improvement. That does not undercut the earlier finding that
`b`'s sign alone cleanly discriminates M1/M0_S4 (stable, b>0 on 29/30 rows
each) from every M3-based row (unstable, b=0 on 60/60) - both statements
hold simultaneously, at different resolutions.

**Confirmed, empirically not just by reading the code: M3 WAS frozen during
the earlier N=1000 controller test - it was the controller-side horizon
extension, not a re-run of Task 5's k-step identification fix.** A user
question raised real doubt about this (`test_n1000_case3.py`'s own docstring
opened with the genuinely ambiguous "Trains M3 case 3," which reads either
way). Verified two ways: (1) static reading of the actual gradient call -
`nnx.value_and_grad(loss_fn, has_aux=True)(ens)` differentiates only the
controller ensemble `ens`; M3's params are a plain closed-over JAX array
(`surrogate_params_batch`), never part of the differentiated pytree; both
`evaluate_and_report` calls (baseline and extended) are passed the exact
same unchanged `m3_params_by_seed` dict. (2) A minimal local repro of the
identical pattern (a "frozen" nnx module closed into a loss function via
`nnx.merge` on a fixed snapshot, only a separate "controller" module
differentiated) run for 50 real gradient steps: the frozen module's params
showed max abs diff of **exactly 0.0** after training, while the controller's
own params changed as expected. The rho/n_unstable movement reported
earlier is fully explained by the closed-loop quantity `Acl = Abar + Bbar@K`
depending on the CONTROLLER's `K_eff` (which legitimately changed), not on
`Abar`/`Bbar` (M3, which did not). **So the controller-side N=1000 test (M3
frozen, DPC horizon extended) has already been run, and it already failed
to fix the instability** - see the entry above. It does not need to be
re-run.

**NEW HYPOTHESIS TESTED AND REFUTED: M3's spurious unstable modes are NOT
uncontrollable or unobservable.** `tools/pbh_controllability_check.py`: PBH
test (`rank[lambda*I - A, B]` for controllability, `rank[[lambda*I - A]; C]`
for observability, smallest singular value reported directly rather than a
silently-thresholded rank) on every eigenvalue with `|lambda|>1`, for every
M3 checkpoint (all 6 cases x 5 seeds, 30 checkpoints, 2-19 unstable modes
each), against M1 and M0_S4 as controls - all read directly from the
existing `.npz` exports, zero new GPU work.

M0_S4 is 1030-dimensional like M3 but KNOWN stabilizable (near-oracle DPC
performance on every case, established many times this session) - it is the
right control for "is a small absolute singular value actually meaningful
at this dimension." Its raw PBH singular values ARE small (controllability:
6e-6 to 1.2e-4; observability: 1.0e-3 to 1.5e-3) - but M0_S4's `A`/`B` are
also much larger in norm than M3's (confirmed: `||A||_F` ~370 vs ~35, `||B||_F`
~2.98 vs ~0.34 for case 3 seed 0), so an absolute comparison between them is
apples-to-oranges. Normalizing each singular value by its own PBH matrix's
Frobenius norm (a scale-fair comparison) gives the real answer: **M3's
relative controllability margin (2e-7 to 1.6e-5 across all cases/seeds) is
consistently 10-100x LARGER (better-conditioned) than M0_S4's own (1.7e-8 to
4.0e-7) in every single case tested - never once smaller.** Same pattern for
observability (M3: 1.9e-5 to 2.4e-4; M0_S4: 1.8e-6 to 6.6e-6). If M0_S4 -
provably stabilizable - has this margin and works, M3 having a consistently
BETTER margin cannot be uncontrollable or unobservable in any meaningful
PBH sense. No case, no seed, no unstable eigenvalue shows a PBH singular
value trending toward zero (the actual signature of a genuine
uncontrollable/unobservable mode) - the smallest value found anywhere
(4.49e-7 relative, case 4 seed 3) is still a real, bounded-away-from-zero
number, not degenerate.

**Verdict: the uncontrollability hypothesis is refuted by direct computation,
comprehensively, not just on the one checkpoint spot-checked first.** M3 is
stabilizable and detectable by this test. Per the user's own stated logic,
this means the optimization-difficulty questions (more epochs at a longer
N=1000 schedule, or freezing the individually-stable S4 layer and retraining
only encoder/decoder/out) are back in play, NOT moot - but the specific
"controller-side horizon test" step already ran (see above) and already
came back negative, so whichever of those is chosen next is a genuinely new
experiment, not a repeat.

GPU: 0 (PBH check and the frozen-params repro are both pure CPU - the
repro used a local jax/flax install, no GPU, no project checkpoints
touched).

## 2026-08-16 — TASK A resolved, but not into the predicted dichotomy: a THIRD reading

sha: (pending commit) | `docs/lqr_on_m3.csv` | `tools/lqr_on_m3.py`

Full-state discrete LQR synthesized directly for M3's augmented (A, B, C) - all
6 cases x 5 seeds, 30/30 checkpoints, pure CPU (`scipy.linalg.solve_discrete_are`,
~100s per checkpoint given the 1030-dim state). Q/R match the DPC cost's own
weights (`Q = C^T (Q_x=5.0 * I_6) C`, `R = R_u=0.1 * I_3`).

**STOP-CHECK cleared: 0/30 checkpoints have b=0 for the full-state-optimal LQR.**
Every single case/seed gives a strictly positive robust margin (0.0001-0.0044) -
small in absolute H-infinity terms, but categorically different from the
b=0.0 DPC always produces. A stabilizing controller for M3 provably exists,
in every case and seed, with zero exceptions. Confirms Task A's PBH result
from a completely independent angle (direct synthesis, not just a rank test).

**But the planned next step - regress a GRU onto this gain, check width -
turned out not to be well-posed, for a reason the branching logic didn't
anticipate.** The full LQR gain acts on the ENTIRE 1030-dim augmented state
(x, the 6-dim physical state, AND s, M3's own 1024-dim internal S4 state) -
but the GRU controller (like every controller in this project) only ever
observes x. It cannot see s at all, not even at t=0. So "fit a GRU to
reproduce K_lqr" is asking a memoryless function of x alone to reproduce a
law that needs s - not fair to any controller, regardless of width.

**Checked the fair version instead: zero out K_lqr's s-columns (K_xonly -
the only linear form ANY x-only controller, of any width, could ever
represent) and re-check the closed loop. Result, 30/30 checkpoints, zero
exceptions: b_xonly = 0.0000 and rho_xonly > 1 in every single case.** A
memoryless, x-only linear gain NEVER stabilizes M3 - not once, across every
case and seed tested. This is not a "not wide enough" finding (no width
fixes a function class problem when the OPTIMAL member of that class
already fails) - it is an information problem.

**Where the gain actually goes explains why:** `||K_lqr[:, s-columns]||_op`
dwarfs `||K_lqr[:, x-columns]||_op` by roughly 20x to 90x+ in every
checkpoint (e.g. case 7 seed 3: 4397 vs 174; smallest ratio found, case 5
seed 4: 241 vs 12, still ~20x). The overwhelming majority of the control
authority LQR needs is directed at the S4 hidden state, not the physical
state - a controller that cannot observe or estimate s is missing almost
all of what stabilization actually requires.

**A second, independent obstruction stacks on top: even the (already
insufficient) x-only gain's own required control magnitude routinely
exceeds `max_action` anyway.** At the edge of the training x0 range
(`||x||=TRAIN_X0_RANGE*sqrt(6)`), `||K_xonly@x||` exceeds `max_action` in
most checkpoints tested - up to 25x over budget (case 7 seed 3: 1279 vs 50),
though not in all (case 1 seed 4: 21.5 vs 50, under budget). This is the
"third reading" flagged as a possible outranking finding - it's real, but
it turns out to be secondary to the deeper information problem above: even
with UNLIMITED action magnitude (no tanh/max_action bound at all), the
x-only gain still wouldn't stabilize the loop, because it's missing access
to s, not just running out of control authority.

**Verdict: neither reading (1) [GRU too narrow] nor reading (2) [objective
doesn't penalize instability] as originally framed. The actual obstruction:
stabilizing M3 requires a controller that can observe or reconstruct its
1024-dim internal state, which is structurally a state-ESTIMATION problem
(build an implicit observer from the history of x through the GRU's own
recurrent hidden state h, since h is the only place that history can live),
not a capacity-at-a-single-timestep problem or a missing-terminal-cost
problem in isolation.** Task B's Riccati terminal cost (which - importantly -
CAN legitimately use the model's true internal state z_N even though the
controller can't, since the trainer sees the full rollout) is not moot by
this finding - it's now a test of whether DPC's gradient signal, over
~200 steps, can push the GRU's hidden state into implicitly performing that
estimation, not a test of whether a wider network would help. Worth being
explicit before running it: this reframes what a positive OR negative
result would mean - success would show the recurrence given the right
incentive; failure would leave open whether it's an optimization-difficulty
problem (can't find the estimator) or a more fundamental one (the estimator
isn't learnable via gradient descent through this rollout at all, or 64
hidden units aren't enough to REPRESENT a working estimator regardless of
training - a genuinely different width question than the one originally
posed, about estimator capacity, not gain capacity). Not yet distinguished.
Paused here for input before spending GPU on Task B, given how much the
plan changed based on this result.

GPU: 0 (full-state discrete LQR synthesis is pure CPU/scipy - ~30 x 100s ~=
50 minutes total wall-clock, no jax, no GPU).

## 2026-08-16 — CORRECTION: reframe "state estimation problem" - it's about what the surrogate IS, not what the controller can see

**Correction to the entry above's verdict, per user challenge.** "Stabilizing
M3 requires a controller that can observe s" is not the sharpest reading,
and here is the control that shows it: **M0_S4 is also 1030-dimensional,
its GRU controller is also x-only (never observes its own S4 hidden state
either), and it controls the true plant at ~1.0x oracle cost.** If "the
controller can't see the internal state" were the obstruction on its own,
M0_S4 should fail exactly the same way M3 does - it doesn't. What actually
differs: M0_S4's internal state is either dead (block-zeroed by
construction, `Abar[:6,6:]==0` - the M0_S4 observability check earlier this
session) or, where it isn't dead, stable and inert. M3's is LIVE and
UNSTABLE - `tools/verify_closed_loop_instability.py`'s finding that real M3
checkpoints carry 2-19 genuinely unstable internal modes stands unchanged.

**The correct statement: identification from I/O data produced a surrogate
whose stabilization requires acting on internal state that has no
counterpart in the true plant.** That is a claim about what M3 IS (an
artifact of fitting a 1030-dimensional model to identify a 6-dimensional
system, given no incentive during identification to keep the unreachable
majority of that capacity inert), not a claim about controller information
access. The "controller can't see s" framing was necessary but not
sufficient - M0_S4 already had that same property and it cost nothing.

**Consequence: Task B's originally-planned Riccati-terminal-cost DPC retrain
is NOT being run.** Even its positive branch would be uninformative by this
new framing: if DPC succeeds at building an implicit estimator and
stabilizes M3, the resulting policy is dominated (median 89x, see below) by
a term reacting to internal state that doesn't exist in the true plant. That
outcome - a controller that works on the surrogate and is dominated by
machinery irrelevant to reality - IS the reality gap, manufactured by
construction rather than discovered. Not worth GPU to go find out.

**Quantified (Task B, user's second numbering - the ||Ks||/||Kx|| ratio
distribution, free from the existing `docs/lqr_on_m3.csv`, all 30
checkpoints): median 88.6x, range 19.9x-275.4x, every single case's median
above 58x.** Per-case medians: case1 91.6x, case2 156.3x, case3 85.7x,
case4 65.2x, case5 97.6x, case7 58.8x. No case comes close to a ratio near
1 (which would indicate the physical and internal-state control demands are
comparable) - the internal-state term dominates everywhere, by roughly two
orders of magnitude, without exception.

**The max_action finding is NOT secondary - restating it as a co-equal,
independent failure, and its structural echo of this project's founding
bug.** Even the (already-insufficient, s-blind) x-only gain's own required
control magnitude exceeds `max_action` in most checkpoints - two independent
ways to fail stacked on top of each other, not one dominant explanation with
a footnote. And this is the SAME underlying question - "is the control
magnitude this policy needs physically realistic?" - that motivated the
project's very first fix: `dpc_example`'s original `StandaloneGRUController`
was UNBOUNDED, letting early DPC runs emit controls reaching ~1e7, which
misdiagnosed at least one case's failure (case 4) as an inherent dynamics
problem when it was actually the bug (2026-08-13 entries above). That bug
HID an unreasonable action requirement by letting the controller cheat past
it. What's found here is the same question from the opposite direction:
with `max_action` now correctly enforced (as it has been since Task 1), the
LQR-optimal x-only gain for M3 needs actions the bound correctly refuses to
allow - not a bug this time, a genuine, load-bearing fact about what M3
demands.

**In progress, not yet complete: the sharper test this correction points
to.** `tools/lqr_transfer_to_true_plant.py` - drive M3's own s-dynamics
with the TRUE (measured) x and u at every step (an exact, ideal linear
observer built from M3's own identified dynamics, not a learned
approximation - the most generous possible construction), apply the SAME
full-state LQR gain to the true plant, check whether the combined
closed loop is stable. All cases, all seeds, M0_S4 and M1 as controls.
Pure linear algebra throughout - no neural net, no BPTT, no optimizer
anywhere in the construction, so a negative result here would be about as
clean as this investigation can produce. Launched; result pending in a
follow-up entry.

GPU: 0 (Task B is arithmetic on an existing CSV; the DECISIONS reframing
itself needed no computation).

## 2026-08-17 — DECISIVE RESULT: M3's provably-optimal controller, with an ideal observer, does not stabilize the true plant. 0/30, no exceptions. Pure linear algebra.

sha: (pending commit) | `docs/lqr_transfer_to_true_plant.csv` | `tools/lqr_transfer_to_true_plant.py`

The direct test of the reframed claim (previous entry): drive M3's own
internal (S4-hidden-state) dynamics with the TRUE, measured trajectory
(an exact, non-learned observer built from M3's own identified matrices -
not a GRU trying to approximate one), apply the SAME full-state LQR gain
already proven to stabilize M3's own closed loop (`b_own > 0` on 30/30
checkpoints, previous LQR entry), and check whether the combined system
stabilizes the TRUE plant. All 6 cases x 5 seeds, plus M0_S4 and M1 as
controls, identical construction. No neural net, no BPTT, no optimizer, no
capacity limit anywhere in this computation - closed-loop eigenvalues of a
fixed linear system, full stop.

**Result: 0/30 for M3. 30/30 for M0_S4. 30/30 for M1. Zero exceptions in
either direction.**

```
variant   n_stable_on_true_plant   median cost_ratio (unstable ones: astronomical)
M1        30/30                    1.005x  (every case, every seed)
M0_S4     30/30                    1.005x  (every case, every seed)
fullM3     0/30                    25,300x  (min 11.1x, max 9.67e12x - case 5 seed 0)
```

`rho` (transferred closed-loop spectral radius) for M3 ranges 1.0028 to
1.0851 across all 30 - never once back inside the unit circle, not even
close, not even the best case (case 1 seed 4, ratio "only" 11.1x is still
unambiguously unstable, `rho`=1.0028). M0_S4 and M1 both land at essentially
exactly the true-plant LQR's own achievable cost (1.005x) in every single
case - not approximately right, correct to 3 decimal places, every time.

**This is the central finding of the investigation.** A controller that is
PROVABLY OPTIMAL for the surrogate (not a trained approximation - the exact
LQR solution) - equipped with the most generous possible reconstruction of
the surrogate's own internal state (not a learned estimator with any
approximation error - the exact linear predictor implied by the surrogate's
own identified dynamics, driven by ground truth) - fails to control the true
plant, unstably, in literally every case and seed tested. Nothing about the
GRU, BPTT, the DPC objective's finite horizon, the optimizer, warm-starting,
k-step identification, or curriculum design is implicated anywhere in this
argument - all of those were this session's earlier hypotheses, and all of
them are now ruled out simultaneously, because this result doesn't route
through any of them. The failure is fully explained by two established
facts about what M3 IS as an identified object: (1) it has genuinely
unstable internal modes with no counterpart in the true plant (this
session's closed-loop-instability and PBH entries), and (2) stabilizing
those modes requires control action that, even under the most favorable
possible construction, does not transfer - the surrogate's own optimal
fix for its own problem actively destabilizes reality.

**Why M0_S4 and M1 are the right controls, restated with this result in
hand:** M0_S4 shares M3's exact architectural blind spot (x-only GRU,
1030-dim augmented realization, controller never observes its own S4 state)
and controls perfectly - ruling out "the controller can't see internal
state" as sufficient on its own (previous entry). M1 has no internal state
at all - ruling out any argument that depends on the augmented realization's
existence per se. Between them they isolate the one thing M3 has that
neither control does: LIVE, UNSTABLE internal dynamics with no physical
counterpart, io-identified into existence by fitting a 1030-dimensional
model to a 6-dimensional system with nothing constraining the unreachable
majority of that capacity to stay inert.

GPU: 0. ~100 minutes total CPU wall-clock (60 fresh 1030-dim DARE solves,
M3 and M0_S4, ~95-120s each, cached to `docs/lqr_cache/` for reuse; M1's 30
6-dim solves are near-instant).

## 2026-08-18 — Bulletproofing the decisive result: construction verified, spectrum decomposed, framing corrected

Three checks against the previous entry's central result, per user request,
plus a reframing of how it gets stated. `docs/figures/lqr_transfer_trajectory.png`
is the paper figure.

**TASK A(1) - the "exact linear predictor" claim needed a qualification, and
the user was right to ask for it.** `tools/trajectory_comparison.py`, case 3
seed 0: `s_hat` (M3's internal recursion driven by the TRUE trajectory) and
`s_free` (the SAME recursion driven by M3's OWN self-referential
free-running x) start identical (both 0) but diverge almost immediately -
by step 2 their difference is already a meaningful fraction of where they
eventually end up, simply because `x_true` (growing) and `x_free` (M3's own
closed loop, which IS stable on its own turf - `lqr_on_m3.py`'s `b_full>0`
result) are different inputs from step 1 onward. **Restated claim: "exact
GIVEN ground-truth driving," not "exact, verified independently against
M3's own behavior."** The two were never going to match, because M3 itself
free-running under its own optimal gain does something totally different
(converges) from what it's asked to track here (a growing true-plant
trajectory). This doesn't weaken the result - it's the more defensible
version of the claim, and worth stating precisely rather than the looser
original phrasing.

**TASK A(2) - the transient, and the figure.** No sudden transition: `‖x_true‖`
grows smoothly and monotonically from essentially step 0 (case 3 seed 0:
6.52 -> 6.83 by step 5 -> 7.16 by step 9 -> ...), consistent with a linear
system whose dominant closed-loop eigenvalue has `|lambda|` just over 1 -
slow-looking at first purely because exponential growth at a small rate
looks flat over a few steps, not because anything is being tracked and then
lost. By step 175 it exceeds 10x its initial norm; by step 300 it's at
323.7, still climbing. M1 and M0_S4 (both ~1.005x oracle) are numerically
indistinguishable on the plot - confirmed directly (agree to 6 decimal
places over the first 10 steps), consistent with this session's earlier
"M1 and M0_S4 agree to ~10 significant figures" finding. M3 free-running
(dashed, its own turf) stays bounded and decays with oscillation, exactly
as `lqr_on_m3.py` already established. The qualitative picture: two
overlapping stable trajectories, one bounded/decaying-but-oscillating
trajectory, and one relentlessly, smoothly diverging trajectory - visually
unambiguous.

**TASK B - decomposed the unstable closed-loop spectrum, all 30 checkpoints,
292 unstable eigenvalues total (`tools/eigenmode_decomposition.py`).** For
each unstable eigenvector, computed its energy fraction in the physical
(x, 6-dim) block vs the internal (s, ~1024-dim) block. **Median frac_x =
0.180 (82% of eigenvector energy in the s-block); mean 0.249; 240/292 (82%)
of unstable modes are majority-s-block.** Precise statement, not
overclaimed: no mode is EXCLUSIVELY one or the other (max frac_x found
anywhere: 0.897, never 1.0; modes are genuinely x/s-coupled, not cleanly
separable) - but the large majority of the closed-loop instability's energy
concentrates in the block with no physical counterpart, consistent with
(and now a second, independent line of evidence for) the ||Ks||>>||Kx||
gain-magnitude finding from the previous entry. Real per-checkpoint spread
worth being honest about: case 2 seed 0's unstable modes are mostly
x-dominated (median frac_x=0.587) while case 1 seed 1's are almost entirely
s-dominated (median frac_x=0.016) - the split is real and directionally
consistent, not uniform across every single checkpoint.

**FRAMING correction for how this whole result gets stated, per instruction:**
- **The headline is the controlled comparison, not the failure in
  isolation.** The same LQR-synthesis-and-transfer procedure applied to
  three surrogates with statistically indistinguishable prediction fidelity
  (M1, M0_S4, M3 all recover Markov parameters to 1e-6-to-1e-14) gives
  1.005x / 1.005x / 25,300x (median). M0_S4 is the arm that forecloses "the
  gap is just 1e-6 vs 1e-16 precision" as an explanation - it is equally
  1030-dimensional, has an equally unobserved internal state, and works
  perfectly. The difference is entirely what identification put in those
  dimensions, not their existence.
- **Nothing in the failing computation is DPC-specific.** The transfer
  experiment is a single LQR solve plus a closed-loop eigenvalue check -
  no BPTT, no GRU, no gradient step, no training curriculum anywhere in it.
  ANY controller synthesis method through this surrogate would hit the same
  wall; DPC is simply the method this project happened to be using when the
  problem was found. Stating this explicitly so the result doesn't read as
  narrower than it is.
- **Scope limits, recorded before a reviewer asks:** this demonstration
  covers linear plants (n=6 cases), one surrogate architecture family (S4,
  d_model=16/N=32, 1030 augmented dims, teacher-forced one-step MSE
  identification), and one controller synthesis method demonstrated to fail
  (full-state LQR) alongside one to fail during actual training (DPC/GRU).
  The honest claim is about THIS class of surrogate under THIS
  identification objective, not "learned surrogates" or "neural state-space
  models" unqualified.
- **M6 has been silent throughout this entire linear-algebra line of
  investigation, and that needs to be said rather than left implicit.**
  Every tool since the nu-gap analysis (`nu_gap_export.py` onward) is
  LTI-only - M6's LayerNorm/GELU/GLU make it not exactly affine, so
  `augmented_operator`'s jacfwd linearization is a LOCAL approximation for
  M6, not an exact global realization the way it is for M3. Nothing here
  has been run on M6. Task 6's horizon-sweep result (M6 fails at the same
  order of magnitude as M3 on every case) is the only evidence connecting
  the two, and it's DPC-outcome evidence, not a mechanism demonstration -
  the LQR-transfer argument in this entry has not been extended to M6 and
  is not being claimed to apply to it directly. Two honest options, neither
  taken yet: extend via trajectory-linearized LQR at a specific operating
  point (an approximation, not exact), or state plainly in the writeup that
  M6's inclusion in the mechanism claim is by analogy to the DPC-outcome
  parallel, not by direct demonstration.

GPU: 0 (all three checks are pure CPU, reusing `docs/lqr_cache/`'s already-
solved K_lqr matrices - no new DARE solves).

## 2026-08-18 — TASK C: does the reality gap scale with over-parameterization? Real trend, not a mitigation

sha: (pending commit) | kaggle: s4dpc-dimension-sweep (v3, after two setup bugs)
| `docs/dimension_sweep_summary.csv`

Re-identified M3 at ~64 and ~256 real internal dims (`d_model=4,N=8` and
`d_model=8,N=16`), all 6 cases x 5 seeds, and ran the identical PBH-unstable-
count + LQR-synthesis + observer-transfer pipeline as the decisive result,
against the existing ~1024-dim (`d_model=16,N=32`) data. Two setup bugs on
the way (`python-control` missing from the Kaggle environment - every prior
use of it this session ran locally; then a `None`-vs-`NaN` formatting crash
when `control.norm`'s H-infinity solve legitimately failed to converge on
one checkpoint) - both fixed, both logged as diverged attempts, not deleted
(2.49 + 3.01 T4-min before the working v3 run).

```
scale   internal_dims   median_teacher_mse   median_n_unstable   n_stable/30   median_ratio
d64     64              3.39e-03             5.0                 2/30         3.34e+02
d256    256             1.04e-05             7.5                 1/30         1.22e+04
d1030   1024            7.03e-06             8.5                 0/30         2.53e+04
```

**Does not vanish at minimal dimension - the failure dominates at every
scale tested, including the smallest (28/30 still fail even at d64).**
That rules out "just shrink your S4 identifier" as a reliable fix. But
there IS a real, directionally consistent trend across all three scales,
not nothing: unstable-mode count, transfer-success rate, and median
failure severity all move the same direction as dimension shrinks
(5.0/2 stable/334x -> 7.5/1 stable/12,223x -> 8.5/0 stable/25,300x).

**The honest complication: teacher_mse (prediction fidelity) is NOT held
constant across the sweep, and this matters for how much weight the trend
can bear.** d64 fits ~500x worse than d1030 (3.4e-3 vs 7.0e-6) - its
improvement is entangled with being a markedly worse-fitting model, exactly
the confound this script's own docstring flagged as a risk before running
it. d256, by contrast, fits comparably to d1030 (1.0e-5 vs 7.0e-6, within
1.5x - genuinely the fairer comparison) and STILL shows a real gap from
d1030: transfer success 1/30 vs 0/30, median ratio roughly 2x milder
(12,223x vs 25,300x). That fidelity-matched pair is the cleanest evidence
in this entry, and it shows a modest, real effect - not a vanishing one.

**Verdict: over-parameterization is a real contributing factor (more
internal capacity gives identification more room to place unstable,
uncontrolled-for internal dynamics), but reducing it is not, on this
evidence, an actionable mitigation on its own.** Even the fairest
comparison available (d256 vs d1030) moves the failure rate from "always"
to "almost always" and the severity from "catastrophic" to "still
catastrophic, somewhat less so." A practitioner reading this table should
not conclude "use a smaller S4 and the problem goes away" - at best,
"smaller may help somewhat, and does not solve it." Restating this
precisely because the hoped-for outcome (a vanishing effect at minimal
dimension, turning the finding into a mitigation) is not what the data
shows, and the previous entries' pattern (every hoped-for fix this session
- warm-starting, k-step identification, horizon extension - failing to
resolve the core problem) extends to this one too, more informative than
the alternative would have been.

GPU: 14.77 T4-min total (2.49 + 3.01 min on the two setup-bug attempts,
9.29 min on the successful run - identification itself is fast at these
smaller dimensions, ~95-100s per scale for all 6 cases x 5 seeds; the CPU-
side PBH/DARE/transfer analysis runs inline in the same kernel, seconds per
checkpoint at these dimensions).

## 2026-08-18 — RECONCILED: "~300 spurious modes" vs "single-digit unstable modes" are two different thresholds, not a contradiction. Plus: this is an objective property, not an architecture-size property.

**Reconciliation, computed directly on the same 30 d1030 checkpoints, not
inferred:** Task 4's original figure used `NEAR_UNIT_THRESHOLD = 0.99`
(`tools/m3_spurious_modes.py`) - counting everything with `|lambda| > 0.99`,
STABLE and unstable alike. This session's more recent PBH/LQR-transfer work
counts strictly `|lambda| > 1.0` - genuinely unstable only. Same
checkpoints, both thresholds, computed together:

```
|lambda| > 0.99 (original convention):  median 302.5  (mean 307.8, range 175-619)
|lambda| > 1.0  (strict, recent work):  median 8.5    (mean 9.1,   range 2-19)
in [0.99, 1.0] (near boundary, STABLE): median 292.0  (mean 298.7)
```

**~97% of the original "~300 near-unit-circle" count is stable modes sitting
close to, but on the correct side of, the boundary. Only ~8.5 of them
(median) are actually on the wrong side.** This is not a correction that
weakens the mechanism - per the reading that predicted it: "identification
places a handful of modes on the wrong side of the unit circle, and that
handful is unfixable downstream" is the tighter, more surprising, and (per
every result since the LQR-transfer entry) the more accurate story. The
~300 figure remains true and useful as a description of how much of M3's
capacity is spent near the stability boundary generally (worth keeping,
since it's what makes the small unstable minority hard to find/fix without
exactly this kind of eigenvalue-level analysis) - but the number that
explains the transfer failure is single digits to low teens, not hundreds.

**Second question, answered with the dimension-sweep data already on hand:
does the unstable-mode count scale with available capacity? No - checked,
not assumed, count AND severity.** 16x more internal dimension (64 -> 1024)
moved the median unstable count from 5.0 to 8.5 (not 16x, nowhere close) -
and 4x more dimension (256 -> 1024) moved it from 7.5 to 8.5, barely at
all - while teacher_mse improved ~500x/~1.5x respectively over the same
steps. Severity is flatter still: median |lambda| among unstable modes
(d1030: 1.023) and median closed-loop `rho_transfer` (d64: 1.018, d256:
1.030) sit within a narrow band regardless of scale - no trend toward
either worse or better as dimension grows. **Agree with the reading: this
is not a capacity phenomenon where a bigger network has proportionally more
room for garbage. The count and severity of unstable modes look like a
near-fixed, small "leftover" - consistent with teacher-forced one-step MSE
structurally being unable to penalize a mode's stability when that mode's
one-step contribution to the loss (its direct Markov-parameter contribution)
is small enough that gradient descent has no signal either way.** A mode
weakly excited within a single step costs nothing in this objective whether
`|lambda|` is 0.999 or 1.001 - stability of such a mode is unconstrained,
not selected against, and whether it lands inside or outside the boundary
is left to whatever the rest of the optimization (initialization, the fit
to the STRONGLY-excited modes, ordinary SGD noise) happens to do to it.
Larger models apparently do not create proportionally more such
weakly-constrained directions - if anything the FRACTION of dimensions
that are unstable falls sharply with scale (7.1% at d64, 2.9% at d256,
0.8% at d1030), even as the absolute count stays roughly flat.

**Disagreement/caveat, stated rather than smoothed over:** the near-flat
trend rests on exactly 3 points per quantity, not a proper scaling curve -
enough to reject "scales with capacity" as the explanation (the effect size
would need to be far larger than what's observed), not enough to claim the
count is a universal constant independent of dimension. Proceeding on the
reading as the working hypothesis; Task A below is the test that matters
more than this observational one.

GPU: 0 (both reconciliations are arithmetic/eigenvalue counts on already-
exported matrices - no new identification, no new DARE solves).

## 2026-08-18 — TASK B: this is not S4-specific. Generalizes, with a real severity difference not yet explained.

sha: (pending commit) | kaggle: s4dpc-linear-ssm-baseline | `docs/linear_ssm_baseline_summary.csv`

A fully unconstrained linear state-space model - dense `A` (64x64), `B`, `C`,
`D`, no diagonal/DPLR structure, no HiPPO initialization, no Lambda_re clip,
no complex parameterization anywhere - trained by plain gradient descent on
the IDENTICAL teacher-forced one-step MSE loss and APRBS data as every M3
run this session (`tools/linear_ssm_baseline.py`). 64 hidden states, 70
augmented total - matching `tools/dimension_sweep.py`'s d64 S4 config
exactly, for a direct, dimension-matched comparison. Same 40000-epoch
budget, all 6 cases, 5 seeds. Ran clean on the full budget after a smoke
test at 2000 epochs (5% of budget) showed slow but genuine convergence
(mean MSE 220 -> 85), confirming this needed the full epoch count before
judging it, not a sign of anything wrong with the setup.

**It also produces spurious unstable modes and also mostly fails the
LQR-transfer test - the phenomenon is not S4-specific. But there's a real,
unexplained severity difference, not just "also fails":**

```
                          median teacher_mse   median n_unstable   n_stable/30   median ratio
linear SSM (70 dims)      5.30e-05             6.0                 6/30          13,735x
S4 d64 (70 dims)          3.39e-03             5.0                 2/30          334x
S4 d1030 (1030 dims)      7.03e-06             8.5                 0/30          25,300x
```

At matched dimension (70 total), the generic linear model achieves BETTER
fidelity than S4 (5.3e-5 vs 3.4e-3 - S4's structure is not obviously helping
optimization here, arguably hurting it) and a HIGHER transfer success rate
(6/30 = 20% vs 2/30 = 6.7%). The 6 successes are genuinely good, not
marginal: ratios 1.4x-97.6x, most near-oracle. **But the 24 that still fail,
fail WORSE than any S4 configuration tested this session** - ratios up to
1.36e147 (vs S4's worst on record, ~1e15 at d1030) - which is why the
MEDIAN ratio (13,735x) looks worse than d64 S4's despite the higher success
rate: with only 30 samples split roughly 20%/80%, the median still falls
inside the failing majority, and that majority's tail is far heavier here.

**Reading: the underlying mechanism (teacher-forced one-step MSE cannot
constrain a weakly-excited mode's stability) generalizes past S4 to a
completely generic linear recurrence - this widens the scope of the paper's
claim substantially, per the user's own framing.** But it does NOT mean S4
is unremarkable in this story - removing S4's structure changed the outcome
in both directions at once (more successes, worse failures), which is
itself informative and not yet explained by anything established so far.
Candidate factors, none tested here: S4's positive-real-part-forcing clip
concentrating initial spectral mass near (not away from) the boundary
(`docs/DECISIONS.md`'s clip-tracing entry already found raw `Lambda_re`
drifts TOWARD the clip during training, gradients vanishing once caught);
the complex/DPLR parameterization's own optimization landscape; or simply
that a dense 64x64 real matrix and a 16-channel x 32-state complex DPLR
structure aren't actually "the same total capacity" in any way that makes
the comparison as matched as the shared "70 augmented dims" framing
suggests. Flagged, not resolved.

GPU: 4.22 T4-min (1.08 min smoke test + 3.14 min full run - fast because,
unlike Task A, there is no power-iteration penalty in this script; plain
teacher-forced training only).

## 2026-08-18 — TASK A: stability-hinge-penalized identification fits comparably and still fails - the strongest-negative branch, with a direct falsification of "just eliminate the unstable modes"

sha: (pending commit) | kaggle: s4dpc-stability-constrained | `docs/stability_constrained_summary.csv`

Re-identified M3 at the standard d1030 internal dimension, all 6 cases, 5
seeds, identical 40000-epoch budget and data to every other M3 run this
session, with a hinge penalty `PENALTY_WEIGHT * relu(rho_estimate - 1.0)^2`
added to teacher-forced MSE (`PENALTY_WEIGHT=1.0`). `rho_estimate` comes
from 20 power iterations (`jax.jvp`) run against a separate `decode=True`
step-graphdef built alongside the `decode=False` training graphdef, since
the training model can't take single-step inputs (`tools/identify_stability_constrained.py`).
Validated locally before spending GPU: power iteration matches exact
eigendecomposition to ~1.5-2% at 20 iterations on a real M3 checkpoint
(oscillation from the dominant complex-conjugate pair, not divergence).
**Every `n_unstable` below is from the EXACT eigendecomposition of the
extracted Abar, never the training-time power-iteration proxy** - same
strict `|lambda| > 1.0` convention as the RECONCILED entry above, so this
is directly comparable to the standard-M3 d1030 numbers there.

**Fairness check first, per instruction - does the penalty degrade
teacher_mse badly enough to confound the test? No. ~2x, not an order of
magnitude:**

```
                          median teacher_mse   median n_unstable   n_stable/30   median ratio
standard M3 (d1030)       7.03e-06             8.5                 0/30          25,300x
stability-penalized M3    1.36e-05             3.0                 1/30          4,313x
```

teacher_mse: 1.94x higher than standard M3 - comparable, not confounded.
The penalty is doing real, comparably-priced work: median unstable count
drops 2.83x (8.5 -> 3.0), median transfer ratio improves 5.87x (25,300x ->
4,313x) - genuine, measurable effects, same "real but not decisive" shape
as the dimension-sweep entry.

**Transfer success: 0/30 -> 1/30 - in any practical sense, unchanged.** The
one success (case1/seed3) is marginal, not robust: `rho_transfer=0.999812`
(barely under 1), ratio 6.5x - the best of a bad set, not a clean win.

**The decisive result: two checkpoints hit `n_unstable=0` EXACTLY - the
penalty's own stated goal, fully achieved, confirmed by exact
eigendecomposition. Both still failed transfer:**

```
case1/seed2:  n_unstable=0/1030  teacher_mse=1.12e-06 (excellent)  rho_transfer=1.002441  ratio=10.3x
case7/seed2:  n_unstable=0/1030  teacher_mse=3.28e-05              rho_transfer=1.021277  ratio=1,397x
```

A fully, provably internally-stable augmented realization - zero augmented
eigenvalues outside the unit circle, both by construction target and by
direct post-hoc verification - still produces a controller that
destabilizes the true plant. This falsifies "just eliminate the unstable
modes and transfer will work" by direct existence proof, not just by a
population-level trend.

**Confirmed at the population level too, not only by these two points:**
`Spearman(n_unstable, rho_transfer) = 0.171` (p=0.366, n=30) and
`Spearman(n_unstable, log ratio_transfer) = 0.227` (p=0.228) - both weak
and not statistically significant. Unlike the earlier kink-magnitude
correlation (wrong sign, -0.54), this one isn't even wrong-signed - there
is simply no reliable relationship, in either direction, between how many
augmented eigenvalues sit outside the unit circle and how badly the
transferred controller performs. Within case1 alone: seed2 (`n_unstable=0`)
FAILS while seed3 (`n_unstable=3`, MORE unstable modes) is the one
checkpoint across all 30 that SUCCEEDS. `n_unstable` does not predict
transfer outcome, even locally, even in the direction naively expected.

**Verdict: the strongest-negative branch. Even a surrogate whose augmented
operator is exactly, provably Schur-stable does not transfer.** This closes
the "eliminate the spurious unstable modes" mitigation path directly - not
by inference from a correlation, but by two checkpoints that hit the target
exactly and still failed. Whatever "spurious" ultimately means in this
project's standing realization-mismatch candidate (CLAUDE.md §1's "current
working picture"), it is not captured by eigenvalue-stability of the
augmented operator alone. Reading: internal stability is a property of
M3's recursion in isolation; transfer success depends on whether M3's
LINEARIZATION (`Abar, Bbar`) is close enough to the true plant's
(`A_d, B_d`) for a controller synthesized from one to stabilize the other -
a stable-but-wrong linearization fails the same way a Markov-accurate-but-
wrong one already does (§1: M3's ~1e-6 Markov error was already too small
to explain the blowup under ordinary error propagation). Internal stability
and linearization fidelity are different axes; this experiment pushed one
axis to its limit (exactly zero unstable modes, verified) while leaving the
other unconstrained, and the failure persisted essentially unchanged.
CLAUDE.md §1 updated to match.

**How to apply:** do not propose "penalize/clip internal eigenvalues" as a
fix again without a new argument for why it would touch the linearization-
fidelity axis rather than only the stability axis - this experiment tested
that specific lever directly, at the specific dimension used throughout
this project, and it did not work, including in its best case.

GPU: 116.25 T4-min (single vmapped ensemble across all 30 case x seed
members trained together in one kernel - `training wall time: 2298.3s`
[~38.3 min] for the GPU-side fit; the remainder is CPU-side PBH/DARE/
transfer analysis for 30 checkpoints plus environment setup).

## 2026-08-18 — Correction: "mismatch between M3's (Abar,Bbar) and the true plant's (A_d,B_d)" was retracted from CLAUDE.md sec 1 as ill-posed

The dimensions don't match (1030 vs 6) - there is no direct matrix
comparison to make, so the wording was a category error, not a hedged
finding. Retracted, not softened; replaced in CLAUDE.md sec 1 with an
explicit open question (a control-relevant quantity this project hadn't
yet measured) rather than a named mechanism. TASK B below answers it.

## 2026-08-18 — TASK A (third round): success-vs-failure contrast - a real but modest signal that does not replicate in the flagship M3 population, and is not actionable there

sha: (pending commit) | `docs/success_vs_failure_contrast.csv`

For the first time this session there was real outcome variance to
contrast: 1/30 stability-penalized M3, 6/30 generic linear SSM, 2/30 S4
d64, several more from the dimension sweep - alongside a much larger
failing population. `tools/success_vs_failure_contrast.py` pools every
checkpoint from every variant identified this session (fullM3, M0_S4, M1,
stability-penalized M3, the generic linear SSM, S4 d64/d256 - 210
checkpoints total) using only already-extracted Abar/Bbar/K_lqr/teacher_mse
sitting on disk from earlier scripts - no new GPU work, pure CPU,
~5 minutes total (eigendecomposition of ~90 1030-dim matrices is the
bottleneck). Compares success (LQR-transfer `cost_ratio<100x`) against
failure on: teacher_mse, `n_unstable` at both RECONCILED-entry thresholds,
`||K_s||/||K_x||` (LQR gain norm on the internal vs physical block), the
median internal-block energy fraction of Abar's OWN unstable eigenvectors
(not the closed-loop transfer matrix's, unlike `eigenmode_decomposition.py`
- that matrix has zero unstable eigenvalues on every success by
definition, so it can't be compared across the boundary), Hankel
singular-value spread (`sigma_6/sigma_1`, since the true systems are
exactly 6-dimensional - same `markov_from_augmented`/block-Hankel/SVD
construction as `tools/balanced_truncation.py`), and internal dimension.

**Pooling everything together showed several "significant" separators
(n_unstable, `||K_s||/||K_x||`, internal_dim, all p<0.01) - but internal_dim
itself was one of them, which is a confound warning, not a discovery:**
smaller-dimension variants (S4 d64, the generic SSM) simply have higher
baseline success rates (established already, Task B/dimension-sweep
entries above), so they're overrepresented in the success bucket, and any
quantity that merely tracks "which architecture class is this" will look
like a separator without being a real within-architecture predictor.

**Stratifying by internal dimension to control for that confound gives a
much more honest, much weaker picture:**

```
d1030 (fullM3 + stability-penalized M3, 9 succ / 51 fail):
  n_unstable (|l|>1.0):       p=0.24   (NOT significant)
  n_unstable (|l|>0.99):      p=0.06   (marginal)
  ||K_s||/||K_x||:            p=0.08   (marginal)
  frac_s of unstable eigvecs: p=0.70   (NOT significant)
  Hankel sigma6/sigma1:       p=0.74   (NOT significant)

d64-equiv (S4 d64 + generic SSM, 23 succ / 37 fail):
  n_unstable (|l|>1.0):       p=0.0025  succ median 4 vs fail median 6 (real)
  ||K_s||/||K_x||:            p=0.54, succ median 10.5 vs fail median 1.6
                               - REVERSED sign from the d1030 stratum (there
                               succ=47 < fail=114) - not a consistent
                               predictor, variant-confounded
  frac_s of unstable eigvecs: p=0.90   (NOT significant)
```

**Per-variant-class descriptive check (n too small per class for its own
significance test, but the direction is informative): every variant class
EXCEPT fullM3 itself shows lower `n_unstable` in successes** (stability-
penalized M3: 2.0 vs 3.0; generic SSM: 4.5 vs 6.5; S4 d64: 4.0 vs 6.0; S4
d256: 6.0 vs 8.0) - **but fullM3, the flagship variant this whole project
is about, shows no such pattern at all (succ median 9.0 vs fail median
8.5, n=4 vs 26 - if anything the wrong direction).** Cross-checked directly
against the free-response error from TASK B below, within fullM3 only
(n=4 success is too small to power this cleanly): no clean separation
(p=0.06-0.76 across error horizons and s0-choices), and the one metric
close to significance (`err_t1`, p=0.06) is confounded by the same n=4
problem.

**Verdict, stated as plainly as the "say so" instruction asked for: a
real, weak, population-level association between `n_unstable` and transfer
outcome exists when pooling across a wide range of architectures and
dimensions, but it does NOT replicate in the d1030 population this
project's central claims are actually about, and `||K_s||/||K_x||`
flips sign across dimension strata - not a real universal predictor,
variant-confounded.** teacher_mse, unstable-eigenvector energy fraction,
and Hankel-SV-spread show no signal anywhere, at any dimension. This is
consistent with, not in tension with, TASK A's second-round direct
intervention (`docs/DECISIONS.md`'s stability-hinge entry): even where a
population-level correlation exists elsewhere, it was not, on direct
manipulation of the flagship variant, an actionable lever.

GPU: 0 (every input already extracted by earlier GPU runs; this is pure
CPU linear algebra on stored checkpoints).

## 2026-08-18 — TASK B (third round): the free-response test - CONFIRMED, sharper than hypothesized, and resolves the linearization-mismatch paradox

sha: (pending commit) | `docs/free_response_test.csv`, `docs/frequency_response_test.csv`

Every fidelity metric used this project so far - Markov parameters,
teacher-forced MSE, impulse response - measures the FORCED response from
rest (`s=0`, driven by input `u` through `Bbar`). Every control rollout
starts from a genuinely nonzero `x0`. `tools/free_response_test.py` tests
directly whether M3's response to a nonzero INITIAL CONDITION with `u=0`
(no forcing at all) matches the true plant, on the already-extracted
linear operators - M3 is exactly LTI, so this needed no new GPU work.

**Method:** for each case, sample a random seed state (same PRNG
convention as `lqr_transfer_to_true_plant.py`'s `get_x0_batch`, `[-5,5]^6`,
batch 50), free-decay it under the TRUE plant (`u=0`) for `T_BURN=50`
steps to get `x0 := x_true(50)` - the true plants are themselves mildly
open-loop unstable (`rho(A_true)` 1.00-1.02, matching the project's known
`rho~1.02` figure), so this `x0` is a healthy, non-degenerate magnitude
(median norm 6.6-15 across cases), not a numerical-artifact near-zero
state. `x0` alone doesn't fix the surrogate's internal state, so two
choices for `s0`, both paired with the SAME `x0` so only `s0` varies:
(a) **observer-derived** - burn in the SAME recursion
`lqr_transfer_to_true_plant.py` uses (`s_hat_{k+1}=Asx@x_true+Ass@s_hat`,
`u=0`) on the true free-decay trajectory for the same 50 steps; (b)
**s0=0**. From `z0=[x0,s0]`, the surrogate runs its OWN full free dynamics
(`z_{k+1}=Abar@z_k`, using `Axx`/`Axs` too - this is exactly the "feeds its
own compounding-error x-prediction back into itself" free-running mode
already named in `lqr_transfer_to_true_plant.py`'s own docstring, not a new
mode of using Abar) for 200 steps, compared against the true plant
continuing its free decay from the same `x0`. Separately, `H_M3(z) =
C(zI-Abar)^-1 Bbar` vs `H_true(z) = (zI-A_true)^-1 B_true` compared over a
400-point frequency grid `z=e^{jw}` (one eigendecomposition per checkpoint,
not 400 direct solves - the direct-solve version was minutes-to-hours
slower at d1030 and abandoned before running).

**Controls validate the pipeline to machine precision - a bug in the
harness would have broken these too:**

```
                s0=observer, err@t=1   err@t=200   |  freq max_rel_diff
M0_S4 (exact)   5.1e-19                6.1e-18     |  3.2e-16
M1 (6-dim only) 6.6e-15                9.5e-13     |  1.3e-12
fullM3          0.77                   1.10        |  27.4 (median), 3.2 near DC
```

**fullM3 is already ~77% relative error at the VERY FIRST FREE-RUNNING
STEP, for BOTH s0 choices (0.769 observer, 0.773 zero - the choice barely
matters, so this isn't an artifact of picking the "wrong" s0)** - not a
slow leak that compounds over the 200-step horizon, a near-immediate
failure. The frequency response confirms this isn't confined to a narrow
band near specific spurious eigenvalues: disagreement is broad, present
even near DC (median 3.2x, i.e. 320% relative error in the near-steady-
state gain), peaking at 27x at each checkpoint's own worst frequency.

**This is not a contradiction with the established `~1e-6` Markov/teacher-
forced fidelity - it's the mechanism that explains how both can be true at
once, and it is the answer to the open question CLAUDE.md sec 1 was left
with after this session's earlier ill-posed "linearization mismatch"
wording was retracted.** Teacher-forced/Markov evaluation always drives
the model through `Bbar` from a fixed (usually zero or APRBS-near-zero)
starting state - a mode that couples weakly to `Bbar` contributes
negligibly to that forced trajectory regardless of its own stability
(the RECONCILED entry's "objective blind spot" reading, restated here in
input-space terms rather than eigenvalue terms). A raw nonzero physical
`x0`, injected directly into the state rather than reached by driving
`Bbar`, has no reason to respect that same weak coupling - and once such a
mode is excited, near-unit-circle dynamics amplify it fast, which is
exactly why the frequency response (which integrates the SAME operator's
behavior at every horizon at once, not just the truncated Markov window)
shows the mismatch so starkly even near DC.

**Does NOT explain the per-case severity gradient - checked, not
assumed:** Spearman between free-response error (`err_t1`, either s0
choice) or frequency-response `max_rel_diff` and `log10(cost_ratio)`
across all 30 fullM3 checkpoints is weak and non-significant throughout
(`err_t1`/observer: rho=0.325, p=0.079 - closest to significant and still
short; `err_t1`/zero: rho=0.183, p=0.333; `err_t200`: rho~0, p>0.88 both
s0 choices; frequency `max_rel_diff`: rho=-0.239, p=0.203). This extends
CLAUDE.md sec 1's existing "no tested spectral quantity correlates with
the per-case DPC ratio" to free-response and frequency-response error too
- the mechanism explains the QUALITATIVE paradox (why forced-response
fidelity and free-running failure coexist), not WHY some cases fail worse
than others, which remains open.

**Verdict: hypothesis confirmed, more sharply than posed.** Free response
from a genuinely nonzero physical initial condition - exactly what any
controller must contend with at every rollout's first step - is where
M3's spurious near-unit-circle modes actually show up; the finite-horizon,
input-driven fidelity metrics used everywhere else in this project were
structurally blind to them, not by bad luck but because those metrics
never excite the modes that matter. CLAUDE.md sec 1 updated to state this
directly.

GPU: 0 (CPU-only; the 400-point frequency grid uses one eigendecomposition
per checkpoint instead of 400 direct solves, which would not have fit in
the platform's compute budget at d1030).

## 2026-08-18 — TASK A (fourth round): M3's raw one-step x-to-x Jacobian (Axx) vs A_d, directly - decisively reading (1) by the user's own stated criterion

sha: (pending commit)

The 77%-at-step-1 free-response result invited a sharper question: is
this "M3's state-transition dynamics are genuinely wrong" (reading 1) or
"x enters M3 as an encoder INPUT, not a state, so 'initialize at x0' isn't
well-defined and the two representations are merely inconsistent"
(reading 2, which would be a narrower, interface-specific claim)? Since
M3 is exactly affine, its one-step Jacobian is a single constant matrix
regardless of evaluation point - `Axx := Abar[:6,:6]` (the x-role block of
the SAME already-extracted, already-validated augmented operator every
other script this session uses, `docs/nu_gap_export/fullM3_*.npz`) IS
this Jacobian, with no burn-in, no s0 choice, and no simulation required.
Compared entrywise and in relative Frobenius norm to `A_d`, all 6 cases x
5 seeds:

```
median ||Axx - A_d||_F / ||A_d||_F = 0.917   (range 0.569 - 1.402, n=30)
```

**Not close to A_d at any checkpoint - worse than the 77% reference point
in the question, not marginally different from it.** By the user's own
stated criterion ("~77% off -> reading (1)... close to A_d, divergence
only emerging over several steps -> reading (2)"), this is unambiguously
reading (1): stated plainly, per instruction - M3's own one-step
state-transition map is simply wrong, despite reproducing the impulse
response to `~1e-6`. There is no "divergence emerging over several
steps" pattern to find, because the map is already wrong in its most
local, single-step form.

**One additional fact, not requested but load-bearing for interpreting
this correctly: `||Axs||_F` (the x-prediction's sensitivity to the
internal S4 state) is 5-15x LARGER than `||Axx||_F` in every single
checkpoint** (e.g. case1/seed0: `||Axx||=2.01`, `||Axs||=19.1`). M3's
one-step x-prediction is dominated by the internal-state channel, not the
x-input channel, by a wide margin. This is why the two s0 choices in the
free-response test agreed so closely (0.769 vs 0.773) even though they
differ enormously in the internal state actually being fed in - `Axx@x0`
alone is already catastrophically wrong (consistent with this entry's
direct measurement), so which `s0` gets added on top barely changes the
qualitative outcome.

**On the mechanism, offered as the best-supported hypothesis, not a
verified fact:** teacher-forced training never presents x_k and s_k as
independently varying - s_k is ALWAYS built from the same true trajectory
that produced x_k, so the two are correlated throughout training by
construction. The training objective therefore has no way to constrain
Axx and Axs SEPARATELY - only their joint action on the (x_k, s_k) pairs
actually seen. Gradient descent has no incentive to prefer the
"physically disentangled" solution (Axx=A_d, Axs=0) over any other
(Axx, Axs) pair that reconstructs the training trajectories equally well,
and nothing in the loss ever isolates Axx alone the way this entry's
direct comparison does. This is closer in spirit to the user's reading
(2) as a CAUSE (a training-procedure/identifiability defect, not a
representational failure of learned dynamics models in general) even
though the MEASURED SYMPTOM matches reading (1)'s criterion exactly. Not
independently verified here (would need access to real per-step (x,s)
training trajectories to check the collinearity claim directly - not
attempted, flagged as the natural next step if this mechanism matters to
the paper's framing).

GPU: 0 (reuses already-extracted `nu_gap_export` matrices).

## 2026-08-18 — TASK B (fourth round): BLOCKED - cannot locate "the first hypothesis doc" or its J_x=0.003 figure in this repository

Searched exhaustively before reporting this as blocked, not guessing:
`grep -rn "J_x"` and `grep -rn "0\.003"` across every `.py`/`.md` file in
the repo (only one unrelated `J_x` hit, `docs/DECISIONS.md`'s 2026-08-13
REFUTATION entry, which NAMES `J_x ~ A_d` as one of "the hypothesis
document's requirements" but does not reproduce its number); `git log
--all -p` across the full history for the same strings, including
deleted files; the repo layout's own referenced `docs/00_protocol.md`
(frozen tables/kill-criteria) does not exist in this checkout. Whatever
produced the `0.003` figure is not in this git history - it lives in a
document outside this repository that I don't have access to.

**What IS established, from this repo alone, and is relevant context for
whoever does have that document:** `s4dpc/diagnostics.py`'s own docstring
(written before this session, predating today's work) already
distinguishes `markov_parameters` ("realization-invariant") from "a raw
one-step `d x_{k+1}/d x_k` Jacobian (which only sees one step and ignores
however the surrogate's own, generally non-A_d-shaped, internal
realization carries information forward through the S4 hidden state)" -
language that reads as already anticipating something like this entry's
Axx result, for a quantity the module explicitly avoids using as its
primary diagnostic. If the original `J_x` measurement used
`markov_parameters`, `jacobian_sweep`, or `local_linearity_defect` rather
than a raw x-block extraction, that would be a genuinely different
quantity from this entry's `Axx` (different: which derivative, w.r.t.
what, at what point, s held fixed or not, and what normalization) - but
I can't confirm which without the source. **Asking the user directly for
the original script or document rather than fabricating a reconciliation
I can't verify.**

GPU: 0.

## 2026-08-18 — TASK C: the free-response-vs-severity correlation was against the LQR-transfer construction's outcome specifically, not the original BPTT/GRU-DPC controller's - stated explicitly since these are different constructions

The correlation reported in the previous entry (`Spearman(err_t1, log10
cost_ratio)` etc.) used `cost_ratio` from `docs/lqr_transfer_to_true_plant.csv`,
`variant=fullM3` rows only - the full-state LQR-plus-observer construction
introduced this session (`tools/lqr_transfer_to_true_plant.py`: "pure
linear algebra, no neural net, no BPTT, no optimizer"), NOT the original
BPTT-trained GRU/DPC controller's `cost_ratio_to_oracle` from
`tools/controller_surrogates.py`/`controller_oracles.py` (the "310x-
466,000x oracle cost" headline number from the REFUTATION entry). These
are related but distinct outcome measures - the LQR-transfer construction
was built specifically to isolate whether M3's own dynamics are sufficient
to explain the failure independent of any BPTT/training-mechanics
confound, and the whole of this session's TASK A/B work (stability
penalty, generic-SSM generalization, success/failure contrast, free
response) has been run against IT, not against the original DPC number.
If "the outcome the paper is built on" means the BPTT/GRU/DPC cost ratio
specifically, that correlation has not been checked this session - the
per-checkpoint DPC cost ratios exist in `docs/controller_surrogates_summary.csv`
(case-level, from the original Task 2/3 sweep) and could be joined against
this session's per-checkpoint free-response error if useful, but the two
datasets come from different identification runs (different checkpoints
entirely - the original DPC sweep's M3 checkpoints were never exported to
`docs/nu_gap_export/`), so a direct per-checkpoint join is not currently
possible without re-identifying or re-exporting. Flagging rather than
silently assuming which outcome was wanted.

GPU: 0.

## 2026-08-18 — BOOKKEEPING: superseding the J_x=0.003 figure

The `J_x ~ A_d, error ~0.003` figure referenced this round predates this
repository (the "first hypothesis document," not locatable in this git
history or its full log - previous entry). Recorded here so the project
has one number for this quantity, not two silently contradictory ones:
**the 91.7% Axx-vs-A_d relative Frobenius error (this session, computed
directly from the already-validated augmented-operator extraction)
supersedes the 0.003 figure.** Whatever the original measurement did
differently (evaluation point, s handling, normalization - all
unconfirmed, previous entry), this session's number is measured against
the SAME `A_d` used everywhere else in this project's control-side work,
on the SAME 30 real, trained fullM3 checkpoints the rest of this
session's findings are built from - not a fresh, unverifiable number from
outside the repository.

## 2026-08-18 — TASK A (fifth round): the (Axx, Axs) split is exactly non-identifiable from teacher-forced training, for every timestep after the first - the gauge-freedom reframing holds, precisely characterized

sha: (pending commit) | `docs/nonidentifiability_spectrum.csv`, `docs/nonidentifiability_null_directions.csv`

The user's reframing: the 91.7% Axx error isn't "the model got the
dynamics wrong," it's gauge freedom - training only constrains
`x_next = Axx@x + Axs@s + Bx@u` along the manifold traced out by the real
training trajectory (where `s` is a FIXED, deterministic function of that
SAME trajectory's history, evolving via separate parameters `Asx, Ass, Bs`
that don't involve `Axx`/`Axs` at all). Any `(Axx, Axs)` agreeing with the
true dynamics on that manifold fits equally well - the loss can't see the
difference. Asked to make this a theorem, not a hypothesis - reported
honestly below, including where the first, more literal attempt at "stack
(x,s) and take the SVD" did NOT cleanly show what the theory predicts, and
what a more targeted test found instead.

**Mathematical reduction verified first, since the user's two asks turned
out to be the same computation:** because `x_next` is exactly linear in
`(Axx, Axs)` holding `s_t` fixed (a genuine fact of this project's
augmented-state formalism, not an approximation), the Gauss-Newton
Hessian of the training loss restricted to `(Axx, Axs)` coordinates, per
output row, is exactly `Z^T Z` where `Z = [X | S]` is the stacked
training-trajectory data matrix - identical for every output row, since
the design vector `z_t = [x_t; s_t]` doesn't depend on which row is being
predicted. A null vector of `Z` is therefore an EXACT flat direction of
this real, local Gauss-Newton Hessian, not a heuristic - "stack (x,s) and
find the null space" (part 1) and "compute the Gauss-Newton null space
restricted to (Axx,Axs)" (part 2) are the same calculation, and this
script does it once.

**First attempt - raw SVD of `Z = [X | S]` (real fullM3/M0_S4 training
data, `L=100` per case, `s4dpc.data.generate_microgrid_trajectory`,
DATA_SEED=42, matching real identification exactly) - honestly reported
as inconclusive, not force-fit into the desired result:**

```
fullM3 Z=[x,s] eff_rank(1e-6): median 100 of 100 possible (full row rank -
  same as a RANDOM 100x1030 Gaussian control matrix, also rank 100)
bottom-10 near-null right-singular vectors: median x-block energy fraction
  0.0009 (essentially ALL s, negligible x-mixing)
```

At the raw-SVD level, `Z` looks close to full rank, and its weakest
directions are almost pure-`S`, not genuine x-s mixing - on its face, this
does NOT support the gauge-freedom reading. Cause identified before
concluding anything from it: `S` alone (1024 columns, only 100 training
samples) is trivially capacity-rich relative to the sample count, so most
of its "weak" directions are just unexplored `S`-capacity unrelated to
`X` at all - a generic-overparameterization artifact (the SAME phenomenon
this project already found and named in the D-only 490-dim null-space
work), not evidence about `(Axx,Axs)` collinearity specifically.

**Targeted test, precise and decisive: for each physical-state
perturbation direction `e_i` (i=1..6), does SOME combination of `S`'s
columns reproduce `-x_traj[:,i]` on the real training data?** Solved via
ordinary least squares, `S @ v_s ~= -x_traj[:, i]`, all 6 cases x 5 seeds,
both fullM3 and M0_S4:

```
INCLUDING t=0:  median relative residual = 0.0310  (S mostly, not fully, compensates)
EXCLUDING t=0:  median relative residual = 0.0000  (EXACT compensation, to ~1e-16-2e-5)
```

**The entire residual is concentrated at the single t=0 training sample,
where `s_0=0` by construction (the "cold start" convention used
everywhere in this project) structurally prevents ANY compensation -
that one sample is the ONLY thing in the entire 100-step training
trajectory that constrains `Axx` at all. For every t>=1 (99 of the 100
training samples), `S`'s column space contains an EXACT combination
(residual at the level of floating-point noise) that reproduces any
specific physical-state perturbation's trajectory-level signature.** This
is the theorem-shaped statement: on 99% of the real training signal, the
`(Axx, Axs)` split is exactly non-identifiable, not approximately.
Confirmed identical (median 0.0310 including t=0, ~0.0000/max 2.1e-5
excluding it) for M0_S4 - **the SAME gauge orbit is present for the hand-
constructed model too. M0_S4 doesn't avoid this ambiguity; it happens, by
construction, to sit at the one point of the orbit (Axs=0 exactly) that
is transfer-safe. Identification has no gradient toward that point,
because the objective is exactly constant across the whole orbit (proven
above, not merely observed) - this is the user's framing, verified, not
just restated.**

**Unifying four measurements as one fact, seen four ways, per
instruction:**

1. M3's Hankel singular-value spectrum has no cliff at rank 6 (balanced-
   truncation entry, 2026-08-13) - a system whose excess ~1024 dimensions
   were inert would show one; M3's don't, because they aren't inert.
2. `Axx` is 91.7% off from `A_d` (this session, earlier entry) - the model
   landed on a point of the orbit far from the transfer-safe one.
3. `||K_s||/||K_x||` is ~91x median for fullM3 (this session's success/
   failure contrast, `docs/success_vs_failure_contrast.csv`) - the LQR
   gain built on top of this realization inherits the same imbalance,
   leaning overwhelmingly on the (uncontrolled, mismatched-to-truth)
   internal-state channel.
4. (This entry) `S` exactly spans whatever's needed to reproduce any
   physical-state perturbation, for 99 of 100 real training samples - the
   mechanism that makes 1-3 possible, not merely correlated with them.

A model with a correct `Axx` and an inert excess realization would show a
Hankel cliff at rank 6, an `Axx` close to `A_d`, and `||K_s||/||K_x||`
near zero - M3 shows none of the three, and (4) is the reason none of
them can be otherwise: the training objective genuinely cannot
distinguish a correct-Axx/inert-excess solution from the one M3 landed
on, except through a single sample's worth of signal at t=0.

**Caveat, stated rather than smoothed over:** this establishes exact
non-identifiability of `(Axx, Axs)` specifically, given the REST of the
model's parameters (`Asx, Ass, Bs`, and the S4 recursion producing them)
already fixed at their trained values - it does not by itself prove
anything about identifiability of the FULL parameter set jointly (a much
larger claim, not attempted here). It also does not explain why gradient
descent, among the entire flat orbit, lands where it does rather than
nearer the safe point - "no gradient signal either way" is consistent
with landing anywhere on the orbit, including by chance near the good
point, and M3 never does (91.7% median, minimum 56.9% across 30
checkpoints) - that specific question (is there a soft bias, e.g. from
initialization or optimizer dynamics, toward one region of the orbit) is
not answered here.

GPU: 0 (pure CPU; the training-trajectory regeneration matches real
identification's `DATA_SEED=42` exactly, and every extracted matrix was
already sitting on disk from earlier GPU work this session).

## 2026-08-18 — TASK B (fifth round): the nu-gap integer-guard bug hunt found a real one - the MAJORITY of "failed" rows are not confirmed winding violations

sha: (pending commit) | `docs/nugap_integer_check.csv`

`tools/nu_gap_analysis.py`'s `nu_gap()` calls the comparison invalid
(forcing `delta_nu=1.0`) whenever `abs(wno + eta_Phat - eta_P) >= 0.4`,
without distinguishing a clean nonzero integer (genuine winding
violation - `delta_nu=1` is the metric's own correct maximal verdict, not
a missing value) from a non-integer reading (the winding-number
computation itself broke down - the honest report is indeterminate, not
1.0). Reran the exact same comparison (`tools/nugap_integer_check.py`,
after a performance rewrite noted below) and binned every failed row by
distance to the nearest integer.

**Bug hunt result: real, not a false alarm, matching the shape of this
project's prior bugs-as-results.** Of 27/90 rows that failed the current
guard: 8 are a clean nonzero integer (genuine violation, correctly
reported); 19 - the majority - are NOT close to any integer (distances
0.10 to 0.47, several sitting right at the edge of the current 0.4
threshold itself). Winding numbers of a continuous curve avoiding the
origin are topologically forced to be integers - a reading that lands
0.4-0.47 away from the nearest one (as far as a value in this range CAN
be from an integer) is not a fractional winding number, it is the
computation failing to resolve one. **The current implementation's
`delta_nu=1` verdicts, when they fire, should not be trusted uncritically
- most of them are not confirmed as genuine winding violations by this
check.**

```
(a) clean NONZERO integer (delta_nu=1 correct):        8/27
(b) clean ZERO but still failed the guard (never seen): 0/27
(c) not close to any integer (indeterminate):          19/27
```

**Performance note, not a separate finding but required to run this at
all:** the original `nu_gap()`'s `_freq_response` does one direct
`np.linalg.solve` per frequency point; at d1030 x 2000 points x ~90
checkpoints this did not complete in practical time locally (killed after
11 CPU-minutes with zero checkpoints finished). Reimplemented with one
eigendecomposition per checkpoint (`H(z) = (CV) diag(1/(z-lambda)) (V^-1
B)` over the whole frequency grid at once, same technique as this
session's `free_response_test.py`) - ~11.7s/checkpoint, ~18 minutes
total. Same math, verified against the original formula; only the
per-frequency solve was replaced.

**Not yet determined: whether a finer frequency grid resolves the 19
indeterminate rows into clean integers (a resolution/numerical-precision
fix) or whether they stay non-integer at higher resolution (a deeper
problem with the phase-unwrapping approach itself).** Flagged as the
natural next step, not attempted here - this entry's job was the bug
hunt, not the fix. Whatever the resolution, **every `delta_nu=1` value
already used elsewhere in this project's nu-gap analysis
(`docs/nu_gap_analysis.csv`) should be treated as provisional until this
is resolved**, since most of them were likely computed under the same
unresolved guard.

GPU: 0 (pure CPU; ~18 minutes wall-clock after the eigendecomposition
rewrite).

## 2026-08-18 — TASK C (fifth round): the dither cure works - exactly, completely, and confirms the pre-registered prediction with no hedging needed

sha: (pending commit) | `docs/dither_cure_test.csv`

**Implementability, decided before running anything, per instruction.**
M3's S4 layer state is a linear recursion's hidden vector - no gating
nonlinearity anywhere in M3 (no norm/activation/glu at all), unlike a
GRU/LSTM's tanh-bounded state. A linear system's state space has no
implicit manifold constraint; any point in that vector space is a
mathematically valid state for the same step function, and this
project's own `zero_states` helper (`s4dpc/diagnostics.py`) already
treats it as a plain overwritable array (that's how cold-start `s0=0` is
implemented everywhere). **Conclusion: genuinely independent random
sampling of `s` is implementable here, not merely perturbation** - the
one thing requiring care is the SCALE, calibrated below from the real
empirical RMS of `s` reached during actual teacher-forced training
(~0.32, measured directly), not an arbitrary range.

**Method - no GPU, no gradient descent, per TASK A's exact-linearity
result:** M3's x-prediction is exactly linear in `(Axx, Axs, Bx)` holding
`s_t` fixed, so re-fitting this sub-block under a modified data
presentation is ordinary least squares. Combined design matrix: the REAL
on-manifold `(x_t, s_t, u_t) -> x_true_{t+1}` data (same trajectory M3
actually trained on) plus `N_DITHER` synthetic off-manifold samples
`(x_synth, s_synth, u_synth) -> x_true_next`, where `x_synth`/`u_synth`
are drawn from realistic ranges and `s_synth` is drawn INDEPENDENTLY at
the real empirical scale; the target is computed from the TRUE plant
directly (`A_d@x_synth + B_d@u_synth`) - well-defined regardless of
`s_synth` because the true plant has no internal state for `s` to stand
in for. Re-solved OLS for `(Axx, Axs, Bx)` on the combined data; kept the
real M3's own `(Asx, Ass, Bs, C)` - the S4 recursion itself - completely
unchanged. Reduced from the originally-planned 5-point dither sweep to 2
points (0, 2000) after timing: a single DARE solve at d1030 did not
finish within 100s locally, and 150 solves (5 x 30 checkpoints) would
have taken multiple hours; 2 points across all 6 cases x 5 seeds (60
solves, ~2 hours) still answers the pre-registered question.

**Result - exact, all 30 checkpoints, no exceptions:**

```
                median axx_rel_err   median ||Axs||   transfer-stable   median ratio
n_dither=0      0.8037               0.681             6/30             4,548x
n_dither=2000   0.0000 (2.8e-15)     0.000 (1.6e-14)   30/30             1.005x
```

**Pre-registered prediction confirmed exactly, not approximately:** with
enough dither, `Axx -> A_d` and `Axs -> 0` to floating-point precision
(2.8e-15, 1.6e-14 - machine noise, not "close"), and EVERY checkpoint
transfers at essentially oracle-optimal cost (median 1.005x, i.e. 0.5%
above the true LQR-optimal cost). This is not a partial improvement like
every other mitigation tried this session (dimension reduction, the
stability-hinge penalty) - it is a complete fix, and it follows directly
from TASK A's exact-linearity result: with 2000 richly-independent
synthetic samples dominating 100 real ones, the only `(Axx,Axs,Bx)`
achieving near-zero residual on ALL of them simultaneously is the one
where `Axs` carries no signal (since `s_synth` carries none) and `Axx,Bx`
match the true plant exactly (since that's what the synthetic samples
directly measure). Not a coincidence; the theorem from this round's TASK
A entry predicts precisely this once `s`'s independence from `x` is
actually enforced by the data.

**Two things this result does NOT show, stated plainly so it isn't
overclaimed:**

1. **`Axs -> 0` effectively reduces M3's x-prediction, for this fully-
   observed/Markov-in-x setting, to exactly M1's own linear regression**
   (`Axx~A_d, Bx~B_d`, no memory contribution at all). The fix works by
   making the surrogate stop using the very thing (S4 memory) that
   distinguishes it from M1 - not by teaching the memory to be used
   correctly. This is not a weakness of the finding (it's the CORRECT,
   provably loss-consistent outcome once `s` genuinely can't help predict
   `x_next` beyond what `x_t` itself already gives), but it means this
   round's result validates the MECHANISM (dither restores
   identifiability), not a general recipe for "how to make S4 memory
   trustworthy" - for THIS class of plant, the theorem-optimal answer is
   to not lean on memory for the x-readout at all.
2. **The synthetic targets were computed directly from the known `A_d,
   B_d`** - valid here because this project's "true plant" is a known,
   queryable simulator (exactly the setting active system identification
   with a digital twin, or hardware-in-the-loop testing, provides) - NOT
   a recipe usable against a genuinely unknown black-box real plant from
   passively-collected data alone, where dither would have to come from
   actual richer physical experiments, not a formula.

**SCOPE NOTE for the paper, per instruction - the real boundary this
result sets up, not a footnote:** this works because every plant in this
project is fully observed and Markov in `x` - the true state IS the
6-dimensional physical state, so a memory path that carries nothing
`x_t` doesn't already contain costs literally nothing to zero out.
**For a partially observed system, `s` must carry genuine information
(the part of the true state not visible in `x_t` alone) and cannot be
randomized freely without destroying real signal - the cure as stated
here is unavailable.** This makes the partially-observed case the actual
open problem this session's results point toward, not a generalization
of what's been fixed here.

**Not yet done, natural next steps:** validating this via actual
gradient-descent retraining of the full S4 model with dithered
presentation (this entry used a closed-form OLS re-fit of just the
readout block, which is the theorem-exact test but not yet confirmed to
survive the messier, jointly-optimized, non-convex full-model setting);
a finer dither sweep to find the minimum `N_DITHER` needed (only 0 and
2000 were tested); and whether this generalizes to the generic linear
SSM and dimension-swept S4 checkpoints from earlier this session, not
only fullM3.

GPU: 0 (pure CPU OLS + DARE solves; ~2 hours wall-clock, dominated
entirely by 60 DARE solves at d1030, ~100-150s each - the same cost this
session already established for every d1030 LQR-transfer construction).

## 2026-08-18 — FRAMING CORRECTION for TASK C, before it goes anywhere near the paper

Flagged by the user immediately after TASK C landed, before drafting -
recorded as a correction to how the result is described, not a change to
the result itself (the 30/30 exact-precision transfer numbers above
stand unchanged).

**The problem with the natural framing:** "Axs->0, Axx->A_d" makes M3's
x-readout functionally equivalent to M1's plain linear regression (TASK
C's own entry already said this). Described as "we fixed the learned
surrogate," a reviewer has an immediate, correct rebuttal: the
demonstrated cure for "an over-parameterized surrogate fails" is "make it
behave like the minimal model" - classical order selection / model-order
reduction, a solved problem, not a new result. Framed that way, TASK C
would be claiming to have solved something system identification already
solved decades ago.

**The defensible claim, narrower and still real - write it this way, not
the broader one:** this project did not discover that small models
transfer better than big ones (already known, and this project's own
dimension-sweep entry already found reducing SIZE alone does not fix
this - 5-8.5 unstable modes essentially flat from d64 to d1030). What
this session's TASK A/TASK C pair actually shows, and what a reviewer
cannot wave away with "that's just order selection":

1. **The missing constraint is identified precisely, not guessed at:**
   `(Axx, Axs)` is EXACTLY non-identifiable from teacher-forced data for
   every timestep after the first (TASK A, proven, not observed).
2. **The objective has NO GRADIENT toward the safe point of that
   ambiguity** - not "gradient descent failed to find it," but "the loss
   surface is exactly flat along the entire orbit containing it," proven
   by the same Gauss-Newton-null-space argument, not inferred from
   training curves.
3. **A MODIFIED objective (dither-augmented data) recovers the safe
   point exactly** - this is the actual result: identifiability, once
   restored, produces the safe gauge; the fact that the safe gauge here
   happens to coincide with M1's model is a property of these particular
   (fully-observed, Markov-in-`x`) plants, not the mechanism being
   demonstrated.

**Do not claim "we fixed learned surrogates."** The claim this project
can defend is: a specific, provable identifiability failure in how these
surrogates are trained explains a control-relevant failure mode that
prediction-accuracy metrics cannot see, and restoring identifiability
(by any means - dither here, truncation if TASK D below confirms it)
removes the failure. That the restored model happens to be small is a
consequence of these plants' structure, not the point being proven.

**How this changes what TASK D (next entry) is actually testing:** not
"does a smaller model work" (already answered, no) but "does restoring
identifiability via a DIFFERENT mechanism (removing excess, unidentified
content post-hoc via truncation, no retraining) ALSO produce the safe
gauge, or is dither-based retraining the only route to it." Framed this
way, TASK D is a genuine second test of the SAME mechanism, not a rerun
of the dimension-sweep question.

GPU: 0.

## 2026-08-18 — TASK D: fidelity-matched truncation - the deferred experiment, run - the gauge is the whole explanation, and truncation is cleanly ruled out

sha: (pending commit) | `docs/fidelity_matched_truncation.csv`, `docs/fidelity_matched_truncation_spectrum.csv`

The experiment flagged as outstanding since the 2026-08-13 balanced-
truncation entry ("the only experiment that would actually test the
hypothesis - not yet run"), now framed precisely by the TASK C correction
above: does removing EXCESS realization content (regardless of which
gauge produced it) fix transfer, or is the gauge the whole story?

**Method:** Hankel-SVD/ERA (`tools/balanced_truncation.py`'s machinery,
reimplemented directly on the already-extracted `docs/nu_gap_export`
matrices - no jax/flax needed). For each fullM3 checkpoint: `target_fidelity`
= that SPECIFIC checkpoint's own max-abs Markov-parameter error vs the
true plant, at FULL 1030 dimension, over the h=1..40 window
(`N_HANKEL=20` requires `G_1..G_40`) - stated precisely because this
number is NOT the "~1e-6" figure quoted elsewhere in this project (that
figure is from a shorter horizon/different metric); at this h<=40 window
it ranges ~0.018-0.70 across checkpoints (M3's own mild instability
compounds error over 40 steps far more than over 1 or 10). Sweep
truncation order `r=6..60`, take the smallest `r` whose ERA-reconstructed
model reaches that target; run the identical observer/LQR-transfer
construction used everywhere else this session on the truncated system.
M1 (already 6-dim) and M0_S4 (`Abar[:6,6:]=0` exactly, so full-dimension
fidelity is already near machine precision) as controls. **Bug found and
fixed en route:** `balanced_truncation.py`'s `to_output_normal_form`
assumed `Cr` square, only valid at `r=6` - crashed the moment any
checkpoint needed `r>6`. Generalized to `to_partial_output_normal_form`
(similarity transform so the first 6 coordinates of the reduced state
read out as the physical output exactly, remaining `r-6` coordinates
whatever basis completes it - standard construction, verified against
the `r=6` case first).

**Result - clean, decisive, matches the "gauge is the whole explanation"
branch:**

```
          median r_chosen   transfer-stable   median ratio
fullM3    6                 2/30              1.21e13
M0_S4     6                 30/30             1.005x
M1        6                 30/30             1.005x
```

26/30 checkpoints already reach their own fidelity target at `r=6` - the
TRUE plant's own order, zero excess dimensions by definition; the
remaining 4 need only `r=7, 7, 12, 20`. **Regardless of `r`, 28/30
catastrophically fail** (ratios from ~280x up to 8.08e129), and the 2
successes are themselves marginal (65.8x, 80.6x - both well above
oracle-optimal, barely under the 100x success threshold). Since the
fidelity target used here is LOOSER than the "~1e-6" figure elsewhere
(making `r=6` easier to reach than a stricter target would), this is if
anything a GENEROUS test in truncation's favor, and it still fails almost
everywhere.

**Verdict: excess realization content is not the culprit. A completely
independent construction method (Hankel-SVD/ERA on Markov parameters,
nothing to do with the (Axx,Axs) sub-block extraction TASK A/C worked
with) applied at the MINIMAL possible order, fidelity-matched to M3's own
achieved accuracy, reproduces the same catastrophic failure. This rules
out truncation as a general-purpose second cure, cleanly - not "truncation
didn't help enough," but "truncation, done as well as this project knows
how, changes almost nothing."** This also generalizes TASK A's finding
past the specific `(Axx,Axs)` gauge: the problem isn't confined to one
particular decomposition of M3's parameters - ANY reduction of M3's own
data to a compact realization, by any of the two independent methods this
project has now tried, lands somewhere that doesn't transfer. TASK C's
dither cure remains the one route that worked, and per the FRAMING
CORRECTION above, that is because it directly restores identifiability
(supplies the information teacher-forced data structurally cannot) - not
because it happens to produce a small model. TASK D confirms the second
half of that reasoning by elimination: producing a small, fidelity-matched
model WITHOUT restoring identifiability does not work.

GPU: 0 (pure CPU; ~4 minutes total for all 90 checkpoints x the full
r-sweep - truncation orders stay small, so DARE solves are fast, unlike
this session's d1030 solves).

## 2026-08-19 — NU-GAP CLEANUP: finer grid resolves most of the flagged rows to clean integers; 7/90 stay indeterminate, mostly explained, and are dropped

sha: (pending commit) | `docs/nugap_integer_check.csv` (N_FREQ=20000), `docs/nugap_integer_check_coarse_n2000.csv` (original N_FREQ=2000, kept for comparison)

Full 90-checkpoint rerun at `N_FREQ=20000` (10x the original 2000),
following spot checks (previous entry) that showed a MIXED picture at
higher resolution - some flagged rows converging cleanly to integers,
at least one drifting further away even at 100,000 points. This resolves
which is which for the whole population, not just the 3 spot-checked
rows.

```
                          n_freq=2000 (prior entry)   n_freq=20000 (this entry)
rows failing the guard:   27/90                       27/90
  (a) clean integer:       8/27                        20/27
  (b) clean zero:           0/27                        0/27
  (c) indeterminate:       19/27                        7/27
```

**Per the user's own decision rule, applied per-row since the actual
result is mixed rather than uniform:** the 20 rows that now resolve
cleanly (within 0.1 of a nonzero integer, several within 0.01-0.03) get
the theory-backed reading - `delta_nu=1` is correct, reported at the
refined grid. Full list in `docs/nugap_integer_check.csv`.

**The remaining 7 stay non-integer even at 10x finer resolution - and
spot checks pushing to 100,000 points (50x the original) on 3 of them
confirm this is not "needs a bit more resolution":** case3/seed0 and
case7/seed4 (both in the CLEAN group at n=20000) converged smoothly as
resolution increased (11.90->11.97->12 and -2.17->-2.00->-2.00
respectively); case1/seed0 (still indeterminate at n=20000) kept
DRIFTING - 10.57 (n=2000) -> 16.44 (n=20000) -> 16.44 (n=50000) ->
16.50 (n=100000), approaching 16.5 (the WORST possible position,
maximally far from either 16 or 17), not settling near any single
integer across a 50x resolution range, while `min_det` (how close the
comparison curve passes to the origin) shrank monotonically the whole
time (0.336 -> 0.114 -> 0.064 -> 0.043 -> 0.024) - consistent with a
genuine near-singular point in the comparison that a finite, uniform
grid cannot resolve at any practical resolution, not a numerical-
precision issue that more points eventually fixes.

**4 of the 7 indeterminate rows are ALL case1** (seeds 0,1,2,4 - only
seed3 resolved cleanly). Not a coincidence: **case1's true plant
(`A_d`) has THREE eigenvalues at EXACTLY `z=1`**, verified directly
(`[0.9656, 1.0, 0.9656, 1.0, 0.9493, 1.0]`) - already flagged in
`nu_gap_analysis.py`'s own existing comments as a known edge case (the
half-step frequency-grid offset exists specifically to avoid landing
exactly ON that pole). A marginal pole exactly on the unit circle is
exactly the kind of feature that makes a winding-number integral
numerically fragile near that frequency, regardless of grid density -
this is a plausible, well-supported mechanism for case1's persistent
non-convergence, not a mystery. The remaining 3 indeterminate rows
(case3/seed2, case5/seed0, case5/seed4) don't share this specific
explanation and are reported as indeterminate without a proposed cause.

**Verdict, per instruction: report the theory-backed reading where the
grid refinement earned it (20/27), report indeterminacy and drop
`delta_nu=1` where it didn't (7/27 - listed by name below), and put
every binned value in the appendix regardless.** `docs/nu_gap_analysis.csv`
(this project's original, still-uncorrected nu-gap output, used
elsewhere) should be treated as reflecting the coarse-grid,
UNRESOLVED numbers for these same rows - if `nu_gap_analysis.csv`'s
figures are cited anywhere in the paper, they need the same correction
applied (a fresh run at the refined grid, or exclusion of the affected
rows), not a blanket "provisional" caveat as this session's previous
entry left it.

**Indeterminate rows, drop `delta_nu=1` for these specifically:**
fullM3/case1/seed0 (cond=16.44, dist 0.44), fullM3/case1/seed1
(cond=10.25, dist 0.25), fullM3/case1/seed2 (cond=5.32, dist 0.32),
fullM3/case1/seed4 (cond=5.50, dist 0.50 - the worst case, exactly
between two integers), fullM3/case3/seed2 (cond=1.11, dist 0.11),
fullM3/case5/seed0 (cond=-1.14, dist 0.14), fullM3/case5/seed4
(cond=-4.23, dist 0.23).

GPU: 0 (pure CPU; the fine-grid rerun took materially longer than the
coarse one - not precisely timed due to two `/tmp`-clearing environment
restarts interrupting earlier attempts, restarted with `nohup`+`disown`
to survive a third; the per-frequency-point Python loop, not the
per-checkpoint eigendecomposition, is the dominant cost at high
`n_freq` and would be the next thing to vectorize if an even finer grid
is ever needed).

## 2026-08-19 — TASK 0 (new session, gating check): conv/step modes agree to machine precision on all 30 real checkpoints - AND a major, unrelated finding: established scripts modeled M3 as linear when it is affine

sha: (pending commit) | `docs/task0_parity_summary.csv`, `docs/task0_parity_stepcurve.csv` | `tools/task0_decode_mode_parity.py`

Before any citation work: does `decode=False` (conv, used for
identification) agree with `decode=True` (stepped, used for every
control rollout and every LQR-transfer computation this session) on
REAL TRAINED checkpoints, not fresh-init params? Run on all 30 real
fullM3 msgpack checkpoints in `docs/nu_gap_export/ckpt/` - the actual
exports every LQR-transfer, gauge-freedom, dither-cure, and truncation
result this session was built from.

**Environment, stated precisely since it's not the exact pin:** this
machine has only Python 3.10 (no sudo to install 3.11); jax==0.7.2/
flax==0.11.2/s4-nnx all hard-require Python>=3.11 at the package level
(not just the CLAUDE.md pin - confirmed by pip's own dependency
resolver). Used `uv python install 3.11` (a standalone, no-root Python
build) to create a venv with the EXACT pinned `jax==0.7.2,
flax==0.11.2, optax==0.2.8, numpy==2.0.2, scipy==1.16.3`, and
`s4-nnx@v0.2.0` installed from git - the real pin, not an approximation.
Single-threaded via `XLA_FLAGS`/`OMP_NUM_THREADS=1` set before any jax
import. `jax_enable_x64=True` throughout. `model.init_state()` was
found to hardcode `complex64` regardless of the x64 flag (already
documented in this project's `s4dpc/diagnostics.py`) - used
`zero_states` instead, per that module's own existing fix.

**(A) conv vs step, s0=0, steps 1-100 (the only length conv mode accepts
at all - see (B)): clean pass, all 30 checkpoints, no exceptions.**

```
max_abs   median=2.598e-13   worst=1.886e-12  (case2/seed4)
max_rel   median=1.078e-11   worst=1.278e-10  (case1/seed0)
```

Every value is at float64 machine-precision scale - the expected level
of disagreement between two DIFFERENT code paths (FFT-based causal
convolution vs sequential `jax.lax.scan` recursion) computing the SAME
mathematical quantity. Full per-checkpoint table in
`docs/task0_parity_summary.csv`; per-step deviation curve (which step
index the max occurs at, no growth pattern toward step 100) in
`docs/task0_parity_stepcurve.csv`.

**(B) - established by direct inspection of `s4-nnx`'s actual source
(`s4_nnx/s4.py`, `S4LayerEnsemble.__call__`) before running anything,
then empirically confirmed on all 30 checkpoints:**

```python
if not self.decode:
    if inputs.shape[0] != self.l_max:
        raise ValueError("Convolution mode currently requires sequence "
                          f"length to equal l_max={self.l_max}; got {inputs.shape[0]}")
    ...
    outputs = causal_convolution(inputs, kernel) + self.D.value * inputs
    return outputs, previous_state   # <- returned UNCHANGED, never used above
```

**Conv mode does not truncate, pad, or wrap past `l_max` - it raises a
hard `ValueError`. There is no way to run it past 100 steps at all, in a
single call, ever.** And `previous_state` is accepted as an argument but
never read to compute `outputs` - it is passed through unmodified. Conv
mode is therefore, by construction, ONLY the zero-initial-state impulse
response (standard S4/SSM kernel theory - the causal-convolution kernel
IS the response from rest), for exactly L=100. **Empirically confirmed
on all 30 checkpoints: running conv mode twice with two different random
nonzero `previous_state` values gives BIT-IDENTICAL output, `0.000e+00`
difference, every single checkpoint, no exceptions.**

**(D) - decode=True DOES depend on s0, substantially, confirming the two
modes are ONLY equivalent at s0=0, exactly as (B) predicts:**

```
step_out(s0=random, scale=0.32) vs step_out(s0=0): max_abs median=4.209, min=1.568, all 30 checkpoints
```

Order-1-to-10 magnitude changes from a modest (empirically-scaled)
nonzero `s0` - not a subtle effect. **Conclusion for (A)+(B)+(D)
together: the two modes are equivalent ONLY at s0=0, and this is not
a bug to fix - it is how conv mode is defined, and this project has
never once called it with a nonzero state (conv mode is only ever used
by `identify.py`, which always cold-starts at s0=0; every nonzero-s0
computation this session - the free-response test, the LQR-transfer
observer construction, the dither cure - always used decode=True or the
Abar/Bbar linear-algebra abstraction, never conv mode).** Since conv
mode structurally cannot represent a nonzero state, and this project
never asked it to, this is a verified architectural boundary, not a
disagreement between two things that were supposed to agree.

**Since conv mode can't run past l_max, the >100-step question needed a
different construction: does decode=True's REAL, free-running output
(matching `s4dpc.control.rollout_learned`'s exact construction - `x` fed
back from the model's OWN prior output, never the true trajectory past
t=0) match the closed-form prediction from M3's own augmented linear
operator (Abar, Bbar - extracted via jacfwd, the SAME construction
`tools/m3_spurious_modes.py` and every downstream script this session
used)? First attempt: NO, badly (median max_abs=178.6, worst 2.24e11) -
investigated rather than dismissed, and this surfaced something bigger
than Task 0 itself was asking about.**

**MAJOR FINDING, not what Task 0 set out to check: M3 is affine, not
linear, and established session scripts modeled it as linear.** `f(z=0,
u=0) != 0` for M3 - this project's OWN `equilibrium_drift` diagnostic
already names and measures exactly this quantity
(`s4dpc/diagnostics.py`, "the TRUE plant satisfies F_true(0,0)=0
exactly... this returns the model's raw deviation"). Measured directly
on all 30 real checkpoints: `||c0_x||` (the physical-state component of
`f(0,0)`) has median **0.83**, range **0.49-1.33** - real, substantial,
not a rounding-level artifact. Adding this single constant term to the
closed-form simulation (`z_{k+1} = Abar@z_k + Bbar@u_k + c0`, instead of
dropping the `+c0`) closes the gap from median 178.6/worst 2.24e11 down
to **median 1.265e-12, worst 1.13e-3** (two checkpoints, case2/seed2 and
case7/seed3, land in the 1e-3-to-1e-8 absolute range rather than 1e-12 -
both are checkpoints whose free-running trajectory itself blows up
astronomically without `c0` either, `7.9e10` and `2.2e11` - ordinary
float64 rounding compounding through 200 steps of a highly unstable
recursion, confirmed by their RELATIVE error staying at the same
~1e-12-to-1e-13 level as every other checkpoint, not a separate issue).
**With `c0` included, this is machine-precision agreement across the
full population, no exceptions - decode=True's real output is exactly
the affine relationship the augmented-operator formalism describes, c0
included.**

**Established session scripts that build `z_{k+1} = Acl @ z_k` (or
`Abar @ z_k`) with NO `+c0` term, confirmed by direct grep, not
assumed:** `tools/lqr_transfer_to_true_plant.py` (`z = z @ Acl.T`),
`tools/fidelity_matched_truncation.py` (`z = z @ Acl.T`),
`tools/free_response_test.py` (`z = z @ Abar.T`),
`tools/dither_cure_test.py` (`z = z @ Acl.T`). Every one of these
propagates M3's dynamics as LINEAR when the real, verified relationship
is AFFINE.

**Impact assessment, precise about what is and isn't at risk - not
everything built on these scripts is affected equally:**

1. **Every EIGENVALUE-based verdict is exactly unaffected, provably, not
   just probably.** `c0` never enters an eigenvalue computation
   (`rho_transfer`, `n_unstable`, PBH controllability/observability,
   the Hankel-SVD spectrum, the `(Axx,Axs)` non-identifiability theorem's
   Gauss-Newton null-space argument). This covers the MAJORITY of this
   session's headline claims: the 0/30 (fullM3) vs 30/30 (M0_S4, M1)
   LQR-transfer stability split, TASK A's spurious-mode counts and the
   gauge-freedom theorem, TASK D's truncation-still-fails verdict at
   `r=6`. All of these stand exactly as reported.
2. **Catastrophic-failure MAGNITUDES (300x to 8e129x) are qualitatively
   robust but not exactly numerically correct.** An unstable
   (`rho_transfer>1`) closed loop diverges to infinity whether or not a
   BOUNDED per-step forcing term is included - "catastrophic, many
   orders of magnitude worse than oracle" survives regardless. The exact
   reported ratio (e.g. "25,300x" specifically, vs whatever the
   `+c0`-corrected number would be) has NOT been re-verified and should
   be treated as approximate in its last few significant figures, not
   as a bug.
3. **The controls (M0_S4, M1) are unaffected by construction, not by
   luck.** M0_S4's `out.kernel=0, out.bias=0` (already established,
   Task 2 Part A entries) zero its OWN `c0` exactly - no bias path
   exists. M1's `fit_least_squares` fits `[A_hat|B_hat]` with NO
   intercept column, and the TRUE plant genuinely has zero bias
   (`A_d@0+B_d@0=0`) - M1 is fitting toward a target that is actually
   bias-free, so omitting an intercept is correct for M1, not a gap.
4. **The genuinely AT-RISK numbers are the STABLE (`rho_transfer<1`),
   near-the-100x-threshold successes - the minority of checkpoints this
   session's mitigations were measured by.** A stable closed loop with a
   real bias term settles to a nonzero equilibrium `z* = (I-Acl)^{-1}@c0`,
   not to zero - `simulate_cost`'s sum-of-`||x||^2` implicitly assumed
   convergence to exactly zero. This could shift the exact cost ratio for:
   TASK C's dither cure (30/30 near-1.005x - though the dither-cured
   `(Axx,Axs,Bx)` block was ITSELF re-fit via OLS with no intercept
   against a genuinely-zero-bias target, `A_d@x+B_d@u`, so the dither
   cure's OWN `c0_x` should be exactly zero by construction; only the
   UNCHANGED `Asx/Ass/Bs` path's small residual bias, `s`-component norm
   ~0.06-0.13 in the general population, measured above, is a candidate
   for a small remaining correction - not independently verified on the
   dither-cured checkpoints specifically, since they were never saved as
   loadable params, only as `(Axx,Axs,Bx)` numpy arrays); the handful of
   marginal successes in TASK A's stability-hinge run (case1/seed3,
   6.5x) and TASK D's truncation (65.8x, 80.6x) - these were NOT
   re-verified with `c0` included and are flagged, not corrected, here.

**Scope of what this entry does and does not cover, stated precisely:**
only the 30 real **fullM3** checkpoints have saved, loadable params
(`docs/nu_gap_export/ckpt/*.msgpack`) - this is the common ancestor of
nearly every decisive result this session (LQR-transfer, the
gauge-freedom theorem, TASK A/B/C/D, the nu-gap work), so it is the
highest-value target and the one Task 0 asked to verify first. **M6,
M0_S4 (as an actual instantiated model, not just its extracted
matrices), and the "dither-cured model" have NO saved checkpoint
anywhere in this repository - M6 was never checkpointed by any script
that still has its weights; M0_S4 is a deterministic hand-construction
that was never saved as msgpack either (regenerable without training,
not attempted here); the dither cure never instantiated an actual nnx
model at all - `tools/dither_cure_test.py` only ever produced raw numpy
`(Axx,Axs,Bx)` arrays via closed-form OLS, combined with the real M3's
UNCHANGED `Asx/Ass/Bs/C`, never assembled into a loadable
`StackedModel`.** Task 0 as literally posed ("every case, every seed,
every variant... M3, M6, M0_S4, and the dither-cured model") could only
be run on M3, for this concrete, checkable reason - not skipped by
choice.

**Verdict per the user's own decision rule: PASS - conv and step modes
agree to float64 precision at every step tested, on every real
checkpoint, at the only initial condition conv mode can represent
(s0=0); the s0-dependence divergence is a verified architectural
boundary this project has never crossed, not a bug. Recorded as a
verified invariant. Proceeding to the citation sweep is warranted on
THIS specific question.** The affine-vs-linear finding is a SEPARATE,
also-important discovery that does not block Task 0's own verdict (no
established EIGENVALUE-based claim is affected) but does mean a
specific, bounded set of MAGNITUDE numbers (mostly the near-threshold
marginal successes) should be treated as approximate pending a
`+c0`-corrected re-run, flagged explicitly rather than silently
carried into the paper.

GPU: 0 (pure CPU, py311 venv with the exact pinned jax/flax/s4-nnx via
`uv python install 3.11` - no sudo available in this sandbox, so a
standalone build was used instead of apt; ~41 minutes wall-clock for
all 30 checkpoints, dominated by unjitted per-step Python loops x 200
steps x free-running + teacher-forced + jacfwd extraction, not by any
single expensive operation).

## 2026-08-19 — TASK C (bias-term round): the affine offset is exactly non-identifiable too, same manifold, and the dither cure drives it to zero along with Axs - the cure story is strengthened, not incomplete

sha: (pending commit)

The Axx/Axs gauge-freedom theorem (fifth-round TASK A, earlier this
session) concerned the LINEAR part of the one-step map only. Does the
SAME manifold argument extend to `c0_x` (the affine offset just found
to be real, substantial, and omitted from four established scripts)?

**Extended the exact test, same construction as the original theorem:**
for `t>=1` (99 of 100 real training samples), does `S`'s column space
ALSO exactly reproduce the CONSTANT (all-ones) direction, not just
arbitrary physical-state perturbation patterns? **Yes - median relative
residual 3.98e-15, worst 1.03e-13, machine precision, all 30
checkpoints.** `c0_x` is exactly as non-identifiable as `(Axx, Axs)`,
by the identical argument: the training objective, restricted to
`t>=1`, cannot distinguish "the model has a real nonzero equilibrium
offset" from "that offset has been fully absorbed into `Axs`'s action
on `s`" - a THIRD quantity on the same flat orbit, not a separate issue.

**Does the dither cure ALSO drive `c0_x` to zero? Yes, exactly, at the
same `n_dither=2000` that already drove `Axs->0`.** Re-ran
`dither_cure_test.py`'s regression WITH an explicit intercept column
(the original never fit one - forcing `c0_x=0` by omission of a term,
not by result):

```
n_dither=0:    median ||c0_x_recovered|| = 0.338   (nonzero - the ambiguity is real without dither)
n_dither=2000: median ||c0_x_recovered|| = 2.18e-15  (exactly zero, machine precision, all 30 checkpoints)
```

**This strengthens the cure story, per the user's own stated branch: the
dither cure recovers the FULL correct point of the gauge orbit -
`Axx->A_d`, `Axs->0`, AND `c0_x->0` - not two out of three.** The
`n_dither=0` baseline's nonzero `c0_x` (median 0.338, smaller than real
M3's own ~0.83 but still real and nonzero) confirms the ambiguity is
genuine at this construction too, not merely theoretical: an
intercept-free regression (like every established session script)
implicitly fixes `c0_x=0` by NOT MODELING it at all, which happens to
be closer to correct for a bias-free target than the alternative, but
is not the same thing as the dither actually resolving an otherwise-real
ambiguity - `n_dither=0`'s recovered value, once an intercept IS
modeled, shows the ambiguity was there the whole time.

**Carried through to the actual closed-loop cost (TASK A, next entry):
the corrected dither-cure ratio is 1.0050x, identical to the original
figure to 4 decimal places** - the small residual bias from the
UNCHANGED `Asx/Ass/Bs` path (median norm ~0.06-0.13, not addressed by
the dither cure since it never touches those parameters) turns out to
contribute negligibly to the closed-loop fixed point for this
construction, verified directly, not assumed.

GPU: 0 (pure CPU, reusing the same training-trajectory regeneration and
already-exported `Asx/Ass/Bs` this session's other CPU-only scripts use).

## 2026-08-19 — TASK A (bias-term round): every claimed success re-verified with the affine offset included - the dither cure is completely unaffected, the two truncation successes need no correction (structurally cannot have one), the stability-hinge success could not be re-derived without new GPU work

sha: (pending commit) | `docs/bias_corrected_dither_cure.csv`, `docs/bias_corrected_lqr_transfer.csv`, `docs/bias_corrected_free_response.csv`

Per instruction: does a claimed "success" actually converge to the
origin, or to a nonzero fixed point `z* = (I-Acl)^-1@c_open` that the
uncorrected cost calculation never checked for? Every claimed success
or near-success this session, checked in turn.

**TASK C's dither cure re-fit (`Axx,Axs,Bx,c0_x` via OLS WITH an
intercept, `n_dither=2000`) carried through to the actual closed-loop
DARE synthesis and cost - result: EXACT, complete confirmation, all 30
checkpoints, no exceptions:**

```
c0_x_recovered: every checkpoint in [1.1e-16, 4.5e-15] - machine precision, exactly zero
rho: every checkpoint in [0.9987, 0.9999] - stable, matches original
||z*_x||: 0.000000 for all 30 checkpoints - EXACT, not "small"
ratio_corrected: 1.0050e+00 for all 30 checkpoints - IDENTICAL to the original figure to 4 decimals
```

This DOES need a fresh DARE solve per checkpoint (the intercept-corrected
`Axx/Axs/Bx` is a genuinely different matrix from the original `Abar`'s
x-block, so the cached `K_lqr` does not apply here - unlike the
LQR-transfer/free-response correction below, which reuses `fullM3`'s
own unchanged `Abar/Bbar` and therefore its cached gain) - 30 fresh
d1030 DARE solves, ~75 minutes wall-clock, the dominant cost of this
entry. The residual `c0_s` (the UNCHANGED `Asx/Ass/Bs` path's own small
bias, median norm ~0.06-0.13) is included in the closed-loop simulation,
not assumed away. **The user's own stated test -
"if a success converges to a large offset, it is not a success" - comes
back with the offset being EXACTLY zero, not merely small, for every
one of the 30 checkpoints.** The dither cure's headline claim is
unconditionally confirmed, not merely "probably fine."

**The two truncation successes (66x, 81x, TASK D) need no correction,
and this is a structural fact about the method, not a numerical
result:** Hankel-SVD/ERA (`tools/balanced_truncation.py`'s `era()`)
reconstructs `(Ar, Br, Cr)` PURELY from Markov parameters
(`Cr@Ar^(h-1)@Br` matched to `C@Abar^(h-1)@Bbar`) via the Kung/ERA
algorithm's own matrix formulas - there is no additive/intercept degree
of freedom anywhere in that construction (verified by direct inspection
of `era()`'s code, not re-run: `Ar`, `Br`, `Cr` are each built from
`U`/`S`/`Vt` of the block-Hankel matrix alone). A truncated model built
this way is bias-free BY CONSTRUCTION - `z=z@Acl.T` (no `+c0`) is the
mathematically correct simulation for what ERA actually produces, not
an omission. Worth noting as an independent, additional way truncation
loses information about M3 beyond the already-established gauge/
eigenvalue story (ERA cannot represent M3's own real, nonzero
equilibrium at all, on top of everything else) - not investigated
further, since it does not change TASK D's verdict either way.

**The stability-hinge success (case1/seed3, 6.5x, TASK A second round)
could NOT be re-derived exactly - stated as a real limitation, not
skipped quietly.** `tools/identify_stability_constrained.py` never
saved raw checkpoint params (only the extracted `Abar/Bbar/K_lqr` in
`docs/stability_constrained/*.npz`) - `c0` is not recoverable from
those alone (it is a separate quantity from the Jacobian, not implied
by `Abar/Bbar`). Exactly re-deriving it would need re-running that
identification with checkpoint saving added, real GPU time, which
TASK B below deliberately scoped out ("CPU only, existing exports
where possible"). Flagged for the user: this ONE result - a single
marginal success (`rho_transfer=0.999812`, ratio 6.5x) already
described as "the best of a bad set, not a clean win" when first
reported - remains uncorrected; the paper's actual positive claim (the
dither cure) does not depend on it.

GPU: 0 (pure CPU throughout; ~75 minutes wall-clock, entirely from 30
fresh d1030 DARE solves for the dither-cure correction - the
LQR-transfer/free-response correction, TASK B below, reuses cached
`K_lqr` and needed no new DARE solves at all).

## 2026-08-19 — TASK B (bias-term round): original-vs-corrected side by side, all four affected scripts, 30 fullM3 checkpoints each - no verdict flips, magnitudes move 5-30% except one script that needed no correction at all

sha: (pending commit) | `docs/bias_corrected_lqr_transfer.csv`, `docs/bias_corrected_free_response.csv`,
`docs/bias_corrected_dither_cure.csv`, `docs/lqr_transfer_to_true_plant.csv` (orig),
`docs/free_response_test.csv` (orig), `docs/dither_cure_test.csv` (orig),
`docs/fidelity_matched_truncation.csv` (unmodified, no correction applies)

Per instruction: put the original and `+c0`-corrected numbers next to
each other, state plainly which headline numbers move. All four
scripts, medians/extrema over the same 30 fullM3 checkpoints unless
noted:

| script | quantity | original | corrected | verdict |
|---|---|---|---|---|
| `lqr_transfer_to_true_plant.py` | `cost_ratio` median | 25,300x | 30,266x | unchanged: catastrophic failure, all 30 |
| | `cost_ratio` max | 9.67e12 | 8.74e12 | unchanged |
| | `rho>1` count | 30/30 | 30/30 | unchanged - closed loop is unstable, not merely offset |
| | `\|\|z*\|\|` | n/a (not computed) | undefined for all 30 (`rho>=1`, no fixed point) | the correction this quantity was FOR does not even apply here - these diverge, they do not settle at an offset |
| `free_response_test.py` | `err_t1` median | 0.7690 | 0.7526 | unchanged - ~77% first-step error is robust to the correction |
| | `err_t200` median | 1.045 | 1.016 | unchanged |
| | `err_t200` max | 2.57e10 | 2.14e10 | unchanged - same 2 checkpoints diverge astronomically in both |
| `dither_cure_test.py` | `ratio_transfer`/`ratio_corrected` | 1.0050x (all 30) | 1.0050x (all 30) | **unchanged to 4 decimal places, exactly** - see TASK A above |
| | `\|\|z*_x\|\|` | n/a | 0.000000 (all 30) | zero, not small - offset is exactly recovered away |
| `fidelity_matched_truncation.py` | `ratio_transfer` (2 successes) | 65.8x, 80.6x | **no correction applies** | ERA is structurally bias-free (see TASK A above) - the "original" number already was the correct one |

**What moves and what doesn't, stated plainly:** the LQR-transfer and
free-response numbers shift 5-30% in either direction per checkpoint
(visible in the underlying per-row CSVs) but every median, every
extremum, and every pass/fail verdict is unchanged after correction.
The dither cure needed correction least of all (identically 1.0050x)
because its own re-fit already drives `c0_x` to machine-zero as part of
the same procedure that fixes `Axx`/`Axs` (TASK C above) - there is no
separate bias to correct for. Truncation needed no correction because
ERA cannot produce a nonzero `c0` in the first place. **No conclusion
in this session flips under the corrected accounting - only the LQR-
transfer/free-response magnitudes move, and only within their own
existing catastrophic-failure or robust-error ranges, never across a
pass/fail boundary.**

One incidental finding surfaced while assembling this table, not asked
for but worth recording: `fidelity_matched_truncation.py`'s case4/seed3
"success" (65.8x) has `rho_transfer=1.011 > 1` - i.e. that reduced
closed loop is technically UNSTABLE, and its "success" cost is a
finite-horizon artifact (the LQR cost horizon does not run long enough
to reveal the divergence), not a converged low-cost trajectory. Only
the other truncation success (case7/seed2, 80.6x, `rho_transfer=0.9994`)
is a genuinely stable closed loop. This does not change TASK D's
verdict (truncation is still ruled out as a general cure - 28/30 fail
regardless), but it means only one of the two previously-reported
"successes" was ever a clean one; the other was already a partial
artifact of the evaluation horizon, independent of anything to do with
`c0`.

GPU: 0 (pure CPU; this entry assembles numbers already computed by TASK
A above and by `tools/bias_corrected_reverify.py`/
`tools/bias_corrected_dither_cure.py` - no new computation beyond
summary statistics and the `rho_transfer` cross-check on the truncation
CSV).

## 2026-08-19 — BOOKKEEPING: the fifth artifact, and naming the pattern behind all five

This session's affine-offset discovery (TASK 0 above) is the fifth
time in this project's history that a result looked settled, then
turned out to rest on an unstated assumption a different part of the
project's own tooling had already measured and contradicted. Naming
the pattern, per instruction, since it recurs:

1. The kink hypothesis (§1's original claim) - LayerNorm curvature
   looked like the mechanism because M3's near-machine-precision Markov
   fidelity was read as "M3's dynamics are correct," when Markov
   fidelity is a forced-response-from-rest quantity and says nothing
   about free-running stability.
2. M0_S4's exact-realization result - initially read as clearing the
   S4/BPTT machinery AND the realization itself at once, when it
   actually changed two things simultaneously (exact I/O and zero
   `obs_norm`) and cleared only the first.
3. The balanced-truncation experiment - initially called "cleanly
   falsified" when it never matched M3's own achieved fidelity, so it
   couldn't have tested the spurious-mode hypothesis either way.
4. `Axx` vs `A_d` (91.7% off) - initially read as "the model learned
   the wrong dynamics," when `S`'s own column space (already implied by
   the training data, not a new measurement) made it a gauge/
   non-identifiability fact instead.
5. **This round: `equilibrium_drift`.** `s4dpc/diagnostics.py` has
   computed `|F(0,0,s)|` since early in this project (`docs/DECISIONS.md`
   line ~1200, values 1.08-2.97 logged during original identification
   sweeps) and it is a VALIDATED diagnostic, not a new measurement -
   yet four downstream analysis scripts (`lqr_transfer_to_true_plant.py`,
   `free_response_test.py`, `dither_cure_test.py`,
   `fidelity_matched_truncation.py`, all written and relied on across
   this project's most recent several rounds) simulated the closed
   loop as `z_next = z @ Acl.T`, silently assuming the fixed point is
   the origin. The diagnostic had been reporting a median `c0_x` norm
   of 0.83 the entire time these scripts were in use.

**The common shape, stated once instead of five times:** in every
case, the thing that was "missed" was not actually unmeasured - it was
measured correctly in one place (a diagnostic, a training-data
property, a construction detail) and silently assumed away in another
(an analysis script, a framing paragraph, a decision rule). The fix
each time was never a new experiment - it was cross-checking existing
tools against each other before writing the conclusion down. Unlike
1-4, this one, when checked (TASK A/B above), changed no verdict - the
closest call so far to "measured correctly everywhere, assumed away
nowhere that mattered."

**Methodology-note-for-the-paper, as requested:** this project's
diagnostic suite (`s4dpc/diagnostics.py`) and its analysis scripts
(the `tools/*.py` one-off experiments layered on top across the
project's life) were not written against a shared contract - a
diagnostic can report a nonzero quantity for months before an analysis
script that assumes it is zero gets written, with nothing to catch the
mismatch except manual re-derivation. Worth a sentence in the paper's
methodology section: the diagnostics were right the whole time; the
gap was between subsystems that never checked each other's assumptions,
not a modeling error in either one individually.

GPU: 0 (bookkeeping only).

## 2026-08-19 — TASK 0 EXTENSION: M6's conv/step parity does NOT hold to machine precision like fullM3's does - real, systematic, on all 30 trained checkpoints. STOPPING per the standing decision rule to report this precisely before anything else.

sha: (pending commit) | `docs/task0_parity_summary.csv` (90 rows: 30 fullM3 + 30 M0_S4 + 30 M6), `docs/task0_parity_stepcurve.csv`

**Why this check is only happening now:** the original Task 0 run
(`docs/DECISIONS.md`'s TASK 0 entry above) covered fullM3 only - M6 and
M0_S4 checkpoints didn't exist as saved artifacts until this session's
checkpoint-reproducibility work (TASK D). A combined fullM3+M0_S4+M6
sweep was launched afterward, but the sweep process had already been
started (and had the module's `VARIANT_CONFIGS` loaded into memory)
BEFORE the M6 entry was added to that dict on disk - Python doesn't
hot-reload a running process's source, so that run silently only ever
covered fullM3+M0_S4 (60 rows, confirmed by inspecting the completed
output), despite the file on disk already listing all three. Re-run as
an M6-only pass (same script, same real checkpoints, same x64/single-
thread settings) to get the missing 30 rows; merged into the real
`docs/task0_parity_summary.csv`/`stepcurve.csv`, now 90 rows total.

**The result, all 30 real M6 checkpoints, x64, teacher-forced conv-vs-
step over the first 100 steps (check (A), same construction that gave
fullM3 ~1e-13 across the board):**

```
A_max_abs:  median=3.168e-05  min=3.726e-06  max=5.881e-04
A_max_rel:  median=9.367e-04  max=1.332e-01 (13.3%)
distribution: 2/30 checkpoints > 5% relative error, 5/30 > 1%, 13/30 > 0.1%
A_argmax_step: median=81.5 (error concentrates in the LATTER half of the 100-step window, not at t=0)
```

fullM3, same check, same 30-checkpoint population size, for direct
comparison: `A_max_abs` median 2.598e-13, max 1.886e-12 - **M6's median
is ~8 orders of magnitude looser, and this is not one or two outlier
checkpoints: all 30 M6 checkpoints sit above fullM3's max.** (B) and
(D) - conv-mode's own state-invariance and step-mode's own nonzero-s0
sensitivity - behave exactly as expected for M6 too (0.000 and >>0
respectively), so this is specifically a conv-vs-step DISAGREEMENT, not
a sign either check itself is broken.

**This is not an artifact of the `model.init_state()` dtype bug the
just-pulled TACC-branch commit (`9b695b0`) happened to fix in
`s4dpc/model.py` - checked directly, not assumed:**
`tools/task0_decode_mode_parity.py` has always used
`s4dpc.diagnostics.zero_states(model)` (which already defaulted to
`complex128` and was written specifically to route around that bug)
for every state construction in this script, never `model.init_state()`
itself - grepped to confirm before writing this entry. The result above
was computed identically before and after pulling that fix.

**What the newly-pulled commit DOES make relevant, and appears to
partially explain the mechanism (not yet confirmed as the full
explanation):** the same commit updated
`tests/test_control_decode_parity.py` with a new docstring stating
"the Cauchy-kernel evaluation in kernel_DPLR is known to be numerically
sensitive (roots-of-unity evaluation points can land close to a
channel's poles)" and reports, for a model trained only 50 epochs
(barely off random init): M3 ~1.0e-1 and M6 ~1.3e-2 relative mismatch
under float32, both collapsing to ~1e-14/~1e-15 under x64. **Our real
M6 checkpoints are trained for 40,000 epochs (identify.py's real
recipe) and show max relative error up to 13.3% UNDER x64** - roughly
the same magnitude as that test's FLOAT32 case for a barely-trained
model, not its x64 floor. The natural reading: extensive training
pushes some channel's pole toward a numerically sensitive Cauchy-kernel
configuration that a near-random-init model never reaches, and M6's
LayerNorm/GELU/GLU stack (present) vs. fullM3's absence of all three
is what makes this specific to M6 - fullM3 trains for the same 40,000
epochs on the same identify.py pipeline and shows no such effect. This
is a plausible, not yet directly tested, mechanism - stated as a
hypothesis, not a finding.

**What this means for the paper, stated precisely per the standing
decision rule ("if they disagree anywhere, STOP and report precisely,
do not proceed"):** the VERIFIED INVARIANT in `CLAUDE.md` is scoped to
fullM3 and is NOT invalidated - it never claimed to cover M6. But the
broader goal stated when M6/M0_S4 checkpoints were added to Task 0
("the verified invariant covers the models the paper actually relies
on rather than fullM3 alone") is NOT yet satisfied - M6 is exactly the
"full model" cell of the variant ladder the paper's headline DPC-
failure numbers are drawn from, and `rollout_learned` (decode=True,
every control/LQR-transfer result) is now demonstrated to deploy a
measurably different function than `identify.py` trained it as
(decode=False conv mode) for M6 specifically, by up to 13% on the worst
of 30 real checkpoints. **This does not on its own explain the M6 DPC-
failure magnitude** (M6's established failure is 2-6 orders of
magnitude, this discrepancy is at most ~13% on the worst checkpoint and
~0.1% at the median) - it is far too small to be the mechanism this
paper's central claim rests on, but it is large enough that a reviewer
who checks train/deploy parity as a first-principles sanity check
(exactly what this test file exists to do) would find a real, non-
machine-precision gap for M6 that the fullM3-only invariant does not
speak to. **Not yet checked:** whether A_max_rel or A_max_abs
correlates with the per-checkpoint DPC cost-ratio severity (the same
question TASK C above asked and left open for the free-response error);
whether this is present in the dither-cured model's inherited
`Asx/Ass/Bs` path (unlikely to apply - that path is copied from fullM3,
not M6, per `tools/save_dither_cured_checkpoints.py`'s docstring, so
should be unaffected, but not directly verified here).

**Stopping here per instruction rather than proceeding to Tasks 1-4.**
This is a genuine, reproducible, previously-uncaught disagreement on
real trained checkpoints for a model this paper's central claim
depends on - it needs the user's read on whether it's worth chasing
(e.g., correlating with DPC severity, or re-deriving the M6 control
numbers under corrected/native-precision Cauchy kernel evaluation)
before treating M6's story as fully settled.

GPU: 0 (CPU-only re-run of the M6 phase of the already-CPU-only Task 0
sweep, ~35 minutes wall-clock).

## 2026-08-19 — TASK A (ninth round, code-reading half) + TASK B: the Cauchy-kernel numerics reading is doubted correctly - here is the structural picture, and the closed-loop-trajectory check the user asked for as a cross-check

sha: (pending commit) | `s4_nnx/s4.py` (read, not modified), `s4dpc/blocks.py` (read),
`docs/task_b_m6_closedloop/trajectory.npz`, `parity_stepcurve.csv`, `parity_summary.csv`

**Code-reading, per instruction, before any more speculation.**
`ConfigurableBlock.__call__` (s4dpc/blocks.py) is IDENTICAL PYTHON CODE
regardless of `decode` - norm, GELU, GLU, and the residual are all
applied POINTWISE IN TIME (LayerNorm over the last/feature axis per
timestep, GELU/GLU elementwise, residual an elementwise add), so none
of them contain a decode-mode branch or a differently-shaped tensor
between conv and step. The ONLY code that branches on `decode` is
inside `S4LayerEnsemble.__call__` (s4_nnx/s4.py) itself: conv mode
evaluates the transfer function at `L` roots-of-unity frequency points
via a Cauchy-matrix product (`kernel_dplr`) and inverse-FFTs the result
into a time-domain kernel; step mode instead bilinear-discretizes the
same continuous DPLR parameters (`discrete_dplr`, forward/backward
Euler with the low-rank Woodbury correction) and runs `jax.lax.scan`.
These are two ARITHMETICALLY DIFFERENT PROCEDURES for the same exact
discrete LTI system - identical in exact arithmetic, but not
guaranteed identical under floating point, and this applies to M3
equally (M3 uses the exact same `S4LayerEnsemble` core M6 does - the
architectural difference is entirely in what `ConfigurableBlock` wraps
around it). **This confirms the numerics-sensitivity mechanism is
architecturally PRESENT in every variant, not M6-specific by
construction - which is exactly why the user's skepticism about it
being the WHOLE explanation is well-founded: M3's own conv/step gap is
~1e-13 on the identical S4 core, so whatever amplifies M6's gap to
13% has to be something ConfigurableBlock adds, not the core itself.**
Given the norm/act/glu code is pointwise and decode-invariant, the
most likely amplification path (not yet tested directly) is LayerNorm's
division by a per-timestep standard deviation: if that std is small at
some timestep, a tiny (S4-core-level, ~1e-13) discrepancy between conv
and step gets divided by a small number and comes out large -
consistent with a SMOOTH severity distribution across checkpoints
(matching what was actually measured) rather than the sporadic,
all-or-nothing signature a pole-near-a-specific-root-of-unity story
would predict. Stated as a hypothesis pending TASK A's empirical
M4/M5 result (separate entry, pending that run's completion), not
asserted as settled.

**TASK B, run to completion, decisive: the discrepancy does NOT grow
under real closed-loop excitation - the "too small to explain M6's
failure" scoping claim is earned, not merely assumed.** Trained one
real GRU-DPC controller (BPTT, full curriculum, same recipe as
`tools/task2b_m6_reality_gap.py`) against the actual committed
`M6_case3_seed0` checkpoint (verified NOT to match that old script's
uncommitted `results/all_cases/ckpt` weights before reusing anything -
replay check showed ~4-6 abs diff at step 1, so a fresh GPU run was
needed for a self-consistent check; ~18 min, Kaggle T4). Controller
converged normally (curriculum loss 599→112) and stabilizes cleanly
against the surrogate it was trained on (cost 121.5, `||x||` stays in
[2.0, 12.6] throughout - no blowup). For 10 of the 100 eval x0's,
replayed the REAL recorded `(x_t, u_t)` pairs teacher-forced through
both conv and step mode (two independent 100-step segments, steps
0-100 and 100-200, each from `s0=0` - the same construction check(A)
used, generalized from the APRBS identification trajectory to a
genuine closed-loop one):

```
max_abs:  median=5.638e-05  min=4.857e-05  max=6.963e-05   (tightly clustered - NOT growing between the two 100-step halves)
max_rel:  median=3.204e-03  min=2.956e-04  max=2.023e-01   (one outlier: member8/half1 at 20.2%)
4/20 segments > 1% relative error, 1/20 > 5%
```

Compare to check(A)'s original in-distribution (APRBS) result for the
SAME checkpoint population: `A_max_abs` median 3.168e-05, max 5.881e-04;
`A_max_rel` median 9.367e-04, max 1.332e-01. **The closed-loop numbers
sit in the same order of magnitude, not a new regime** - `max_abs` is
actually MORE tightly bounded here (never exceeds 7e-5, versus check A's
population max of 5.9e-4) and `max_rel`'s single 20.2% outlier is only
~1.5x check A's own worst-case 13.3%, not a qualitative jump. The
argmax step location also replicates check A's pattern exactly (errors
concentrate in the LATTER part of each 100-step window: steps 84-99 for
half 0, 192-199 for half 1) - the same underlying behavior, not a
different one triggered by leaving the identification support. **Per
the user's own decision rule: it stayed bounded, so the scoping claim
("too small to explain M6's 2-6-order-of-magnitude DPC failure") is
earned and kept**, not weakened - a single 20% relative-error segment
noted honestly but not chased further, since it does not change the
order-of-magnitude comparison against the 2-6-orders-of-magnitude
quantity it was being scoped against.

GPU: 16.84 min (Kaggle T4, one GRU-DPC controller trained via BPTT,
full 6-phase curriculum, single case/seed).

## 2026-08-25 — EXTERNAL REPRODUCTION: the central failure is independently confirmed; our explanation of it is retracted. New mechanism: the A_xs block and the margin identity. Two bugs in our own code found and verified. No new experiments run this entry - recorded from the external report pending our own re-verification.

sha: (pending commit) | no new data this entry - `tools/lqr_transfer_to_true_plant.py`,
`s4dpc/control.py`, `s4dpc/identify.py` (read, not modified, to verify the claims below)

An independent group reimplemented this project from scratch (same
plants, Tustin dt=0.01, d_model=16/N=32/n_layers=1/l_max=100, matching
param counts - 3638 M3 / 3942 M6, confirmed against this machine's own
`env_probe.py` model_canary below - same variant ladder, same
LQR-transfer construction: `A_open=[[A_d,0],[A_sx,A_ss]]`, `s0=0`,
`u=-K_x x - K_s s`, DARE ignoring `c0`, `Q=5I`, `R=0.1I`, `Qf=50`,
horizon 200, 100 ICs, `rho(A_cl)<1`, n=30 as 6 cases x 5 seeds, case 6
excluded). This entry is written from their report, not a re-run on
our side - per instruction, no new experiments this round.

**WHAT REPLICATED, independently: the central failure is real.** 0/30
stable transfers. Coupling `||A_xs||/||A_xx||` 5.08-16.20 against our
own 5-15. This is now confirmed by a from-scratch reimplementation, not
resting on this codebase alone.

**WHAT DID NOT REPLICATE, AND WHY - our explanation of the failure was
wrong.** They train on 320 trajectories (32,000 transitions); we train
on `batch_size=1` - ONE 100-step trajectory, 100 samples (confirmed
directly this entry - see TASK 4 below). At B=320 they get `A_xx` rel
err 3.2e-3 and drift 3.9e-4. At B=1 they reproduce OUR numbers almost
exactly (0.572 and 1.027 vs our 0.698 and 1.076). **Our `A_xx` and
`equilibrium_drift` figures are artifacts of training on a single
trajectory, not properties of what M3 learned.** With N=100 samples
against a 1024-dim latent, `rank(S) >= rank(X)` automatically, so `Axx`
is formally unidentifiable at B=1 - our own gauge-freedom proof said
exactly this, correctly, but its INTERPRETATION was wrong: "the internal
state's column space contains an exact compensation for 99 of 100
timesteps" reduces, at one sample per timestep, to "a nonzero vector in
R^1024 absorbs any vector in R^6" - true for ANY `s`, including pure
noise. It is not evidence of a meaningful learned gauge symmetry; it is
a restatement of a linear-algebra fact about severely underdetermined
regression. `rank(S)` saturates near 429, so identifiability returns at
B>=5. **Critically, this does not explain away the failure - it only
retracts our explanation of it:** transfer fails at EVERY B tested
(`rho` 1.0206 -> 1.0172 from B=1 to B=320), and the coupling
`||A_xs||/||A_xx||` never moves. More data cures the SYMPTOM we
measured (`Axx` error, drift) and leaves the disease (the transfer
failure itself) untouched.

**FORMAL RETRACTIONS, per instruction - retracted, not softened, same
standard as the kink refutation (§1's "REFUTATION" entry):**

1. **The 91.7% `Axx` vs `A_d` relative-error figure** (`docs/DECISIONS.md`
   line 4150, "TASK A (fourth round)") **is RETRACTED as evidence about
   what M3 learned.** The raw measurement may be reproducible at B=1,
   but it does not characterize M3's dynamics - it characterizes what an
   underdetermined regression on 100 samples against a 1024-dim latent
   does, which is not a property of the model, the architecture, or the
   training objective in any way that would persist at reasonable data
   volume.

2. **The `equilibrium_drift=0.83` figure** (`docs/DECISIONS.md` line
   4803, "TASK 0", and CLAUDE.md §1's "affine, not linear" finding)
   **is RETRACTED as evidence of a real, dynamically meaningful
   displaced equilibrium.** Same reasoning as (1) - `c0` is read off the
   same severely underdetermined fit. The underlying fact that `f(0,0)`
   was computed and is nonzero at B=1 is not disputed; what is retracted
   is treating that number as characterizing the model's true
   equilibrium rather than a B=1 identification artifact.

3. **The gauge-freedom proof as currently worded** (`docs/DECISIONS.md`
   line 4292, "TASK A (fifth round)") **is RETRACTED as an argument
   about the model, and should be read going forward as a fact about
   the DATA regime only.** The linear algebra is not wrong - `S`'s
   column space genuinely does contain an exact compensation at B=1 -
   but stated as "the model has an exploitable gauge symmetry" it
   overclaims: the identical proof holds for a state vector containing
   nothing but noise, so it demonstrates non-identifiability of the
   REGRESSION PROBLEM at N=100, not a property discovered about S4 or
   about what training does. `rank(S)` saturating near 429 at higher B
   means this is a data-volume statement, not an architectural one.

4. **The dither cure as described** (`docs/DECISIONS.md` line 4475,
   "TASK C (fifth round)", extended at line 5021/5076 with the affine
   offset) **is RETRACTED as a cure for the actual mechanism.** Two
   independent problems, both fatal to the claim as stated: (a) it was
   built and verified to solve the B=1 non-identifiability specifically
   (dithering the regressor to break the exact rank-deficiency at
   B=1) - since that non-identifiability is now understood to be a
   B=1 artifact rather than the real disease, curing it does not
   address transfer failure at realistic data volumes, where the
   external group's own B=320 result shows the disease persists
   regardless (`rho` still >1, coupling unchanged); (b) more
   fundamentally, per the external group's note (which we did not
   catch ourselves): an unconstrained `(6,1024)` `Axs` produced by an
   OLS refit corresponds to NO SET OF S4 WEIGHTS - a realisable `Axs`
   is constrained to `rank<=6` with rigid block structure inherited
   from the S4 recursion, not a free `6x1024` matrix. **The
   network-level version of the dither cure (re-training the actual
   S4 weights, not an unconstrained linear-algebra readout of
   `(Axx,Axs,Bx)`) was never run** - both our own record and the
   external group's notes agree on this. Every dither-cure result in
   this document (30/30 near-oracle, `c0_x -> 0`, etc.) describes a
   closed-form OLS construction that is NOT achievable by any actual
   S4 model, and should not be read as evidence a retrained network
   would transfer.

**THE MECHANISM WE SHOULD ADOPT INSTEAD, per the external group's
report (not yet independently re-verified by us - flagged as external
until we do):** the cause is the `A_xs` block specifically, not a
gauge/identifiability story at all. Zeroing `A_xs` at synthesis time -
no retraining, no other weight touched - converts a 356x failure into
1.0013x (near-oracle). Discarding `K_s` instead (dropping the gain on
the S4-state channel, keeping `K_x` only) makes it WORSE (1.6e113x),
which refutes the "gain wasted on a phantom state" reading this
project floated earlier: Riccati co-designs `K_x` and `K_s` jointly for
a plant that does not exist (the augmented `(A,B)` fit to B=1 data),
and neither half of that joint design is usable alone once you remove
the coupling it was designed against. **The margin identity:** at
`alpha=0` (meaning: coupling zeroed) the augmented closed-loop system
is block-triangular, so `rho = 1 - max|eig(A_ss)|` exactly - reported
to match to `1.33e-15`. Read plainly: the ENTIRE stability margin
available to absorb the `A_xs` coupling IS the damping of the
least-damped S4 internal mode, and S4's own HiPPO-derived spectrum
places 462 of 1024 modes within `1e-2` of the unit circle - a system
with almost no spare margin to give away. Their proposed cure is a
scale-normalized `||C||^2` penalty applied DURING TRAINING (not a
post-hoc linear-algebra readout): reported 30/30 at oracle cost while
ALSO improving one-step MSE by 24 orders of magnitude - i.e., a cure
that does not trade fidelity for stability, unlike this project's own
earlier "explicit internal-stability hinge" attempt
(`docs/DECISIONS.md`'s RECONCILED entry, 2026-08-18), which reduced
`n_unstable` but left transfer at 1/30. **This has NOT been run on our
side.** Per instruction, no reruns this entry - the record is being
set straight first.

**TASK 2, verified directly: the M1/M0_S4 "1.005x" figure is a real
bug, not floating-point noise, and the external group's diagnosis is
exactly right - numerator and denominator DO come from different code
paths.** In `tools/lqr_transfer_to_true_plant.py`:
- **Numerator** (`simulate_cost`, lines 110-133): propagates the closed-
  loop matrix directly (`z = z @ Acl.T`) for `EVAL_HORIZON=200` steps,
  normalizes at line 131: `cost = (stage + terminal) / EVAL_HORIZON`
  - divides by **200**.
- **Denominator** (`oracle_cost`, computed at line 169 via
  `true_quadratic_cost(x_hist_lqr, u_hist_lqr, ...)`, the local
  function defined at lines 155-158): consumes `x_hist_lqr` from
  `rollout_lqr_true` (lines 144-152), whose `xs = [x0]` then appends
  one entry per step, giving `x_hist.shape[0] = EVAL_HORIZON + 1 =
  201`. Line 158 normalizes by `x_hist.shape[0]` - divides by **201**.

`201 / 200 = 1.005` exactly, matching the reported "1.0050000000000003"
/ "1.0049999999999997" / "1.0050000000002886" figures (small residuals
beyond the exact 1.005 come from M1/M0_S4's own gain not being bit-
identical to the oracle's, not from anything else). This is not a
one-off slip either - `s4dpc/control.py` itself already carries BOTH
conventions simultaneously: `rollout_linear`/`rollout_learned` (the
actual TRAINING-time loss, lines 129 and 198) normalize by
`horizon_N` (200 - the numerator's convention), while `control.py`'s
OWN canonical `true_quadratic_cost` (lines 240-247, the function
`tools/lqr_transfer_to_true_plant.py`'s denominator re-implements
locally) normalizes by `x_hist.shape[0]` (201 - the denominator's
convention). `tools/lqr_transfer_to_true_plant.py` mixed the two
established-but-different conventions within one ratio. **Every
M1/M0_S4 "near-1.0x" claim in this project's history that used this
construction is off by a confirmed, exact, quotable 0.5% multiplicative
factor** - not large enough to change any qualitative verdict this
project has drawn (M1/M0_S4 were never claimed to be anything but
"near-oracle," and 1.000x vs 1.005x doesn't change that), but a real
bug, now on the record. Not fixed this entry - verification only, per
instruction.

**TASK 4, confirmed directly from source: identification really does
use `batch_size=1`.** `s4dpc/identify.py:44-54`'s `case_data()` calls
`generate_microgrid_trajectory(batch_size=1, length=l_max, ...)` and
returns `batch_inputs[0], batch_targets[0]` - a single trajectory, not
a batch. `run_identify` (`s4dpc/identify.py:299` on) confirms this is
exactly what real training sees: `data = {c: case_data(c, l_max, ...)
for c in cases}` builds ONE trajectory per case, and
`inputs_grid = jnp.stack([data[c][0] for c in flat_cases])` STACKS THE
SAME single trajectory once per seed-member of the vmapped ensemble -
every seed within a case trains on the identical 100-sample trajectory,
differing only in weight initialization. **This is a documented,
load-bearing limitation, not a bug to fix reflexively:** the whole
identification recipe this project has used throughout - all 30 fullM3
checkpoints, M0_S4, M4, M5, M6, the dither-cured artifacts - is built
on this single-trajectory convention. Any claim resting on `Axx`,
`equilibrium_drift`, or the OLS-based dither/gauge constructions
inherits this limitation; any claim resting on the LQR-transfer
failure itself, the coupling ratio, or the margin identity does not
(those are properties of the augmented `(A,B)` regardless of how it
was identified, and the external group's B=320 result shows the
failure survives the data-volume fix).

**What this changes and what it doesn't, stated plainly per
instruction:** the paper's central empirical claim - learned S4
surrogates fail DPC transfer catastrophically despite good one-step
fidelity - is UNCHANGED and now independently strengthened. What
changes is the MECHANISM chapter: the gauge/non-identifiability/dither-
cure narrative (this project's entire 2026-08-18 "fifth-round" through
"bias-term-round" arc) is retracted as the explanation, replaced by a
coupling/margin story (`A_xs` causally implicated, margin identity,
HiPPO spectral placement) that is currently an EXTERNAL, not yet
independently re-verified, claim. The truncation result (TASK D,
"excess realization content is not the culprit") is UNAFFECTED by any
of this - it never depended on the gauge proof or the dither cure, and
zeroing `A_xs` (the new mechanism) is consistent with, not
contradicted by, truncation failing (a generic order-reduction method
has no reason to find the SPECIFIC zero-`A_xs` direction).

GPU: 0 (this entry is a written record of an external report plus
direct code-reading verification on our own side - no reruns).

## 2026-08-25 — TASK 1: the 201/200 bug audit, every reported ratio in this project, mapped by numerator/denominator function - two clean families, no partial cases

sha: (pending commit) | code-reading only, no data this entry

Per instruction: for every reported ratio, does the numerator and
denominator use the SAME cost function or DIFFERENT ones. Two clean
families emerged - every script fell entirely into one or the other,
no mixed cases.

**FAMILY A - SAME function both sides, bug CANCELS, numbers unaffected:**

- **The 310x-family DPC numbers** (`tools/controller_oracles.py`,
  `tools/controller_surrogates.py` - the ORIGINAL BPTT/GRU-DPC "Task 3
  sweep", real trained neural controllers, not the LQR-transfer
  construction). Numerator: `tools/controller_oracles.py:190-193`'s
  `_evaluate()` calls `true_quadratic_cost` (imported from
  `s4dpc.control`) at line 193. Denominator:
  `tools/controller_oracles.py:231` / `tools/controller_surrogates.py:291`
  both call the SAME `co.true_quadratic_cost`. Identical function, same
  `x_hist.shape[0]` normalization on both sides - no bug.
- **The horizon sweep** (4.16x/2.55x/1.45x/1.04x/1.02x and the M3
  column - `tools/horizon_sweep_oracle.py`,
  `tools/horizon_sweep_surrogate.py`). Denominator:
  `horizon_sweep_oracle.py:68` / `horizon_sweep_surrogate.py:134`, both
  `co.true_quadratic_cost`. Numerator: `result["cost"]` from the same
  `co._evaluate()` as above (`horizon_sweep_oracle.py:94`,
  `horizon_sweep_surrogate.py:183` divide `result["cost"]` by
  `oracle_costs[case]`). Identical function - no bug.

**FAMILY B - DIFFERENT functions, bug APPLIES, exact factor 201/200 =
1.005 (or the checkpoint-specific residual around it from the two
controllers not being bit-identical):**

- **The 25,300x LQR transfer** (`tools/lqr_transfer_to_true_plant.py`) -
  already found and reported last round. Numerator: `simulate_cost`,
  lines 110-133, normalizes by `EVAL_HORIZON` (line 131, =200).
  Denominator: `true_quadratic_cost`, lines 155-158, normalizes by
  `x_hist.shape[0]` (line 158, =201, since `rollout_lqr_true`'s
  `xs=[x0]` then appends `horizon_N` more).
- **The truncation results** (`tools/fidelity_matched_truncation.py`).
  Numerator: `simulate_cost`, lines 149-165, `/ EVAL_HORIZON` at line
  163. Denominator: `true_quadratic_cost`, lines 178-181, `/
  x_hist.shape[0]` at line 181. Same bug, same file structure as
  `lqr_transfer_to_true_plant.py` (copy-pasted pattern).
- **The stability-hinge row (6.5x)**
  (`tools/identify_stability_constrained.py`). Numerator:
  `simulate_transfer_cost`, lines 154-172, `/ horizon_N` at line 170
  (`horizon_N` defaults to 200). Denominator: `true_quadratic_cost`,
  lines 148-151, `/ x_hist.shape[0]` at line 151. Same bug.
- **The dither rows, both the original and the bias-corrected version**
  (`tools/dither_cure_test.py`, `tools/bias_corrected_dither_cure.py`).
  Numerator: `simulate_cost`/`simulate_cost_biased`, `/ EVAL_HORIZON`
  (`dither_cure_test.py:176`, `bias_corrected_dither_cure.py:114`).
  Denominator: `true_quadratic_cost`, `/ x_hist.shape[0]`
  (`dither_cure_test.py:192`, `bias_corrected_dither_cure.py:130`).
  **Worth stating plainly: the dither cure's headline "1.0050x, not
  quite 1.0" figure is now suspect as being driven ENTIRELY by this
  bug** - `Axx` recovers to machine precision in that construction, so
  genuinely-oracle-quality control would read as ~1.000x if computed
  correctly; "1.0050x" is exactly what a perfect match would read as
  UNDER this specific normalization mismatch. This doesn't change the
  retraction from the previous entry (the OLS-readout construction
  itself is not achievable by any real S4 model regardless), but it
  means even the closed-form number was never showing "very close to
  oracle, not exact" - it may have been showing "exactly oracle,
  measured with a biased ruler."
- **The bias-corrected LQR-transfer figure (25,300x -> 30,266x)**
  (`tools/bias_corrected_reverify.py`). Numerator:
  `simulate_cost_biased`, `/ EVAL_HORIZON` (line 102). Denominator:
  `true_quadratic_cost`, `/ x_hist.shape[0]` (line 82). Same bug on
  BOTH the "original" and "corrected" figure equally (both computed by
  this same script's functions) - the 5-30% shift documented between
  them is a real effect of adding `+c0`, not related to this bug, since
  the bug's own ~0.5% contribution is present identically in both.
- **The generic-linear-SSM-baseline and dimension-sweep comparisons**
  (`tools/linear_ssm_baseline.py`, `tools/dimension_sweep.py`) -
  identical structure: `simulate_transfer_cost(..., horizon_N=200)` (`/
  horizon_N`) vs `true_quadratic_cost` (`/ x_hist.shape[0]`), same line
  numbers as `identify_stability_constrained.py` (all three files share
  near-identical copy-pasted bodies). Same bug.

**Root cause, stated once instead of per-file:** this project has two
GENERATIONS of evaluation code. Generation 1
(`controller_oracles.py`/`controller_surrogates.py`/`horizon_sweep_*.py`)
evaluates a REAL trained GRU-DPC controller via
`evaluate_controller_on_true` + `s4dpc.control.true_quadratic_cost` on
BOTH the surrogate-trained and oracle controllers uniformly - safe by
construction, since there's only one cost function in play. Generation
2 (`lqr_transfer_to_true_plant.py` and everything that copy-pasted its
body: `fidelity_matched_truncation.py`, `identify_stability_constrained.py`,
`dither_cure_test.py`, `bias_corrected_dither_cure.py`,
`bias_corrected_reverify.py`, `linear_ssm_baseline.py`,
`dimension_sweep.py`) is the pure-linear-algebra LQR-transfer
construction (no neural controller, direct `Acl` propagation) -
introduced its OWN locally-defined `simulate_cost`/
`simulate_transfer_cost` for the numerator (normalizing by the loop
count, matching `s4dpc.control.rollout_linear`/`rollout_learned`'s
TRAINING-time convention) while reusing `true_quadratic_cost` for the
denominator (normalizing by array length, `s4dpc.control`'s own
EVALUATION-time convention) - two conventions that already coexist,
legitimately, elsewhere in this codebase (`s4dpc/control.py` itself has
both), mixed inconsistently within one ratio in every Generation-2
script.

**What this changes, stated plainly:** nothing qualitative. Every
Family-B ratio is a large-magnitude failure (6.5x to 9.67e12x) where a
0.5% multiplicative bias changes no verdict - except the dither cure,
where the bug may be the ENTIRE story behind "1.0050x" rather than a
negligible correction to a real near-miss. Not fixed this entry -
verification and mapping only, per instruction.

GPU: 0 (code-reading only).

## 2026-08-25 — TASK 4: B=320 identification reproduces the external group's numbers closely - the pipelines agree, and the mechanism can be built on. One self-caught bug (a tautological 30/30-stable false positive) and one genuinely new, unexplained finding (n_unstable up 5x, not down) along the way.

sha: (pending commit) | `tools/identify_b320.py`, `tools/lqr_transfer_b320.py`,
`docs/b320/b320_summary.csv`, `docs/b320/lqr_transfer_b320.csv`, 30 checkpoints in `docs/b320/ckpt/`

Per instruction: rerun M3 identification at B=320 (320 independent
trajectories, 32,000 transitions, `generate_microgrid_trajectory`'s own
`batch_size` parameter already draws independent realizations - no new
data-generation code needed) instead of B=1, same cases/seeds/epochs
(40000) as every other real checkpoint in this project, for direct
comparability. New training function `train_ensemble_multi_traj`
(`tools/identify_b320.py`) mirrors `s4dpc.identify._train_ensemble`
exactly, adding one inner `jax.vmap` over the trajectory-batch axis
inside each ensemble member's loss - `s4dpc/identify.py` itself is
untouched, matching this project's established pattern of standalone
`tools/*.py` scripts for one-off identification variants
(`dimension_sweep.py`, `linear_ssm_baseline.py`, etc.).

**Timing, validated before committing to the full run:** a 2000-epoch/
full-30-checkpoint-ensemble timing test on Kaggle T4 took 108.9s
(3.63s/checkpoint) - extrapolated to 40000 epochs (~36 min), and the
full run's actual identification wall time was 1913.2s (31.89 min, from
the kernel log directly). Logged in `gpu_ledger.csv`
(`s4dpc-identify-b320-full`, 31.89 T4-min; the timing test and the
env-probe kernel from the prior round logged as estimates, their exact
kernel logs having already been cleaned up locally when this was
written).

**Result, all 30 checkpoints, compared against both our own B=1 numbers
and the external group's reported B=320 numbers:**

| metric | our B=1 | our B=320 (median) | external B=1 | external B=320 |
|---|---|---|---|---|
| `axx_rel_err` | ~0.917 | **3.54e-3** (range 6.0e-4 - 1.70e-2) | 0.698 | ~3.2e-3 |
| `equilibrium_drift` | ~0.83 | **7.68e-4** (range 1.59e-4 - 3.11e-3) | 1.076 | ~3.9e-4 |
| coupling `‖A_xs‖/‖A_xx‖` | 5-15 | **6.93** (range 4.05-14.19) | 5.08-16.20 | unchanged |
| transfer stable count | 0/30 | **0/30** | - | still failing |
| `rho(A_cl)` | >1 (unstable) | **1.010-1.032** (all >1) | ~1.0206 | ~1.0172 |
| cost_ratio (LQR-transfer) | median ~25,300x | **167.9x** (min 32.1x, max 1.51e5x) | - | - |

**Our pipeline reproduces the external group's B=320 numbers closely on
every axis they reported.** `axx_rel_err` and `equilibrium_drift` land
within a factor of ~2 of their figures (well within checkpoint-to-
checkpoint spread - our own range spans a full order of magnitude on
`axx_rel_err` alone), the coupling ratio sits inside their reported
5.08-16.20 band and visibly does NOT move from the B=1 range (matching
their "coupling unchanged" claim), and transfer still fails at 0/30
with `rho` just over 1 in almost the same place they report (1.010-1.032
here vs their ~1.0172-1.0206). **Per the standing decision rule: we
reproduce their finding, so our pipeline and theirs agree, and the
A_xs/margin mechanism from the retraction entry above is safe to build
on**, not merely an untested external claim anymore.

**A real bug in THIS ROUND's own code, caught before being reported -
worth recording precisely, same discipline as everything else this
project has caught this way.** The first `tools/lqr_transfer_b320.py`
run reported **30/30 STABLE** (`rho<1` for every checkpoint,
cost_ratio median ~31x) - immediately suspicious since it directly
contradicted the external group's "still failing" claim. Root cause:
the script ran the transfer simulation on the model's own learned
`Abar`/`Bbar` directly, instead of substituting the TRUE plant's
`(A_d, B_d)` into the physical-state block the way
`tools/lqr_transfer_to_true_plant.py`'s original fullM3/M0_S4
construction does (`A_open = block([[A_true, 0], [Asx, Ass]])` - DARE
is solved on the model's own `(A,B)` to get `K_lqr`, matching what the
model would design, but the TRANSFER test has to run on the true
physical dynamics or it isn't testing transfer at all). Using the
model's own `Abar` for both steps is tautological - a model's own LQR
gain trivially stabilizes the model's own dynamics, for any
stabilizable pair, regardless of whether the model matches reality.
Fixed to match the established construction exactly; the DARE cache
(keyed on the model's own `(A,B,C)`, unaffected by the fix) was
expected to still be valid but the corrected rerun re-solved anyway
(~50 min, not investigated - possibly a cache-path mismatch between
the two script versions, cosmetic either way since the DARE solve
itself was unaffected by the bug or its fix). Corrected result is the
0/30-stable table above.

**A genuinely new, unexplained finding from this round, NOT part of
the external group's report - flagged honestly rather than folded in
silently:** `n_unstable` (open-loop `Abar` eigenvalues with `|lambda|>1`,
before any control) has a B=320 median of **40** (range 21-56) -
roughly 5x this project's earlier-established B=1 baseline of ~8.5
(the RECONCILED-round finding, `docs/DECISIONS.md`, 2026-08-18). Every
other metric in the table above moves in the expected direction with
more data (Axx/drift shrink toward the true values; coupling stays
flat) - `n_unstable` moves the OPPOSITE way, growing substantially
worse. Not chased further this round per instruction (no new
experiments beyond what was asked). Worth flagging for whoever picks
this up: given the adopted mechanism ties the whole stability margin
to `rho = 1 - max|eig(A_ss)|` at zero coupling, a large increase in
open-loop instability at B=320 is exactly the kind of thing that could
matter for that story, or could be a separate, unrelated artifact of
higher data volume interacting with training dynamics - genuinely open.

GPU: 31.89 T4-min (identify_b320.py's full run) + ~4.0 T4-min (timing
test + env-probe estimates) - LQR-transfer analysis itself was CPU-only
(30 fresh 1030-dim DARE solves, ~50 min wall-clock, run twice due to
the bug above).
