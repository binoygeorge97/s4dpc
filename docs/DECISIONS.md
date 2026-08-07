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
