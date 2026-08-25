# CLAUDE.md — s4dpc

Operating guide for Claude Code in this repository.

---

## 1. What this project is

Research code for a conference paper on **why differentiable predictive control (DPC)
fails when the plant model is a learned S4 surrogate**, even when one-step prediction
MSE is low.

**Correction, 2026-08-13 (kept, not silently rewritten — see `docs/DECISIONS.md`'s
"REFUTATION" entry and the 10-seed/kink-magnitude entries around it for the full
evidence trail):** the paragraph below was this project's original central claim.
It has been tested directly on control-side data and **refuted**. M3 has exactly
zero kink by construction (no norm/activation/glu) and still fails DPC by
2-6 orders of magnitude on every case, including ones where it recovers Markov
parameters to ~1e-6; Spearman(kink magnitude, M6/M3 cost ratio) = -0.54 at n=6 —
wrong sign, not just non-significant. **Original claim, kept for the record:**
*control-relevant derivative fidelity*, not prediction error, determines DPC
success, via LayerNorm (degree-0 homogeneous, so it cannot represent a linear map)
distorting the learned Jacobian near the regulation setpoint — the "kink".

**Current working picture, updated 2026-08-13 with a decisive result (also
`docs/DECISIONS.md` — check there before citing a specific mechanism as settled,
since this remains under active investigation):** the S4/BPTT/`decode=True` control
machinery itself is innocent. "M0_S4" — an S4 hand-constructed (block-zeroed) to
realize the exact true `(A_d, B_d)` to ~1e-17, then deployed through the identical
`rollout_learned`/`jax.lax.scan` machinery M3/M6 controllers train through — matches
the true-plant training baseline essentially exactly, at every curriculum horizon
tested (5 to 200) and on every one of the 6 control cases (>=5 seeds each, most
agreeing to 3-4 decimal places). So the 300x-700,000x M3/M6 failure has nothing to
do with being an S4 surrogate, with BPTT through the stepped/scan machinery, or with
any training-mechanics candidate — it is entirely about what identification LEARNS
when fit to data instead of being handed exactly. The current best candidate for
what that learned difference is: M3's augmented state-transition operator (physical
state + S4 hidden state) has ~300 near-unit-circle modes (of 1030 dims) where the
true system has only 3-6, with real one-step leverage on the output and often
orders-of-magnitude-excess transient growth — present in every real M3 checkpoint,
absent by construction in M0_S4 (which was never asked to LEARN a realization).
This does not yet explain the case-by-case severity gradient (no tested spectral
quantity correlates with the per-case DPC ratio) or why the same spurious-mode
signature shows up: warm-starting the S4 hidden state on genuine burned-in history
makes M3's free-running prediction monotonically WORSE, not better, arguing against
a simpler inconsistent-initial-condition story and toward the internal modes being
fundamentally unmoored from the physical state rather than merely mis-initialized.

**Sharpened, 2026-08-13 (see `docs/DECISIONS.md`'s CORRECTION entry after Task 2
Part C):** M0_S4 changes TWO things relative to a real M3 checkpoint at once —
exact I/O (M3's is ~1e-6) and zero *observable* internal state (M3's `obs_norm`
is ~10-20, real). M0_S4's result therefore does NOT distinguish which one matters
— it only rules out the S4/BPTT machinery. Since M3's ~1e-6 Markov error was
already too small to explain a 300x+ blowup under ordinary error propagation
(`rho~1.02` over 200 steps compounds 1e-6 to only ~5e-5), realization/observable-
spurious-modes remains the standing, not-yet-refuted candidate.

**Tested directly via balanced truncation (Hankel-SVD/ERA, since the augmented
operator is marginally unstable so Gramian-based truncation doesn't apply) —
result decided NOTHING about the spurious-mode hypothesis (corrected
2026-08-13 in `docs/DECISIONS.md` — an earlier write-up called this "cleanly
falsified," which overstated it).** Truncating M3 to 6 states does NOT recover
M3's own ~1e-6 fidelity (lands at ~1e-2 instead) because M3's Hankel singular
values have no cliff at rank 6 the way the true system's do (verified:
self-check on the true system gives an EXACT cliff to 0 past index 6; M3's
spectrum decays smoothly) — a real, unconfounded finding about M3's own
realization. But truncated M3 then failing DPC 3-41x worse than full M3 is
NOT informative about spurious modes specifically: a ~1e-2-fidelity system
was already expected to fail badly given this project's own established
M1(~1e-14→~1x)-vs-M3(~1e-6→300x+) relationship, independent of any mode-based
story. The experiment never isolated the variable it needed to. A
fidelity-matched truncation (larger r, chosen so `err_vs_m3_markov` reaches
M3's own ~1e-6 floor) is the only experiment that would actually test the
hypothesis — not yet run.

**Update, 2026-08-18 (`docs/DECISIONS.md`'s RECONCILED/TASK A/TASK B entries —
read there for the full evidence trail):** the "~300 near-unit-circle modes"
figure above and later single-digit figures are NOT in tension — same 30
d1030 checkpoints, `|lambda|>0.99` gives ~300 (median 302.5), strict
`|lambda|>1.0` gives median 8.5; ~97% of the ~300 are stable modes sitting
close to, not over, the boundary. The mechanism is tighter than originally
stated: a handful of augmented modes (single digits to low teens) land on
the wrong side of the unit circle, not hundreds. This count/severity does
NOT scale with internal dimension (5.0 → 7.5 → 8.5 unstable modes across a
64→256→1024 sweep, while fidelity improves ~500x) — an objective property
(teacher-forced one-step MSE cannot penalize a weakly-excited mode's
stability either way), not a capacity property. **Directly tested and
refuted as a fixable lever:** re-identifying M3 with an explicit hinge
penalty on the internal spectral radius reduces `n_unstable` (median
8.5 → 3.0) at comparable fidelity (~2x teacher_mse, not confounded), but
transfer success stays at 1/30 — and two checkpoints that hit
`n_unstable=0` EXACTLY (exact eigendecomposition, not the training-time
proxy) still failed transfer (`rho_transfer` 1.002, 1.021). Population
correlation between `n_unstable` and transfer outcome is statistically
indistinguishable from zero (Spearman ≈0.17–0.23, p>0.2, n=30). **Also
directly tested: not S4-specific.** A fully generic dense linear SSM (no S4
structure, no HiPPO, no clip) fit by gradient descent on the identical
objective produces the same spurious-unstable-mode/transfer-failure
pattern (6/30 transfer success — better than S4 at matched 70-dim capacity,
but with far heavier-tailed failures when it does fail, unexplained).
**Correction, 2026-08-18: the "linearization-mismatch" wording above (in an
earlier draft of this update) was ill-posed and has been retracted, not
softened.** `Abar` is 1030-dim, `(A_d, B_d)` is 6-dim — there is no direct
matrix comparison to make, so "mismatch between them" was not a real claim.
The actual paradox, unresolved: the quantity that DOES admit a fair
comparison — induced input-output behavior from rest — already agrees to
`~1e-6` (M3's teacher-forced Markov fidelity). Internal stability, also
directly testable, has been driven to exactly zero unstable modes on two
checkpoints. Both of the only two things measured so far say M3 is fine;
transfer still fails on both. **Resolved, 2026-08-18 (`docs/DECISIONS.md`'s TASK B, "third round" —
read there for the full method and evidence): free response from a
nonzero initial condition is the discriminating quantity, and it is not
subtle.** Driven from a matched nonzero `x0` with `u=0` (both a paired
observer-derived and a `s0=0` internal state tried — the choice barely
matters), M3's own free-running prediction is already `~77%` relative
error at the VERY FIRST step, versus `~1e-18`/`~1e-13` for the M0_S4/M1
controls (same harness, so this validates the method rather than being a
bug). The frequency-domain transfer function `H_M3(z)` vs `H_true(z)`
disagrees by a median `27x` at each checkpoint's worst frequency and
`~3.2x` even near DC — broad, not confined to a narrow band. **This is
not in tension with the `~1e-6` teacher-forced/Markov fidelity — it is
the mechanism that explains how both are true at once.** Every fidelity
metric used elsewhere in this project drives the model through `Bbar`
from rest; a mode that couples weakly to `Bbar` costs nothing in that
forced evaluation regardless of its own stability (same "objective blind
spot" as the RECONCILED entry, restated in input-space rather than
eigenvalue terms) — but a raw nonzero physical `x0`, injected directly
into the state rather than reached by driving `Bbar`, has no reason to
respect that same weak coupling, and once excited, the near-unit-circle
dynamics amplify it immediately. **Does not explain the case-by-case
severity gradient** — checked directly, Spearman between free-response/
frequency-response error and the per-checkpoint LQR-transfer cost ratio
(the pure-linear-algebra construction this session's work runs against,
`docs/lqr_transfer_to_true_plant.csv` — NOT yet checked against the
original BPTT/GRU-DPC cost ratio, a related but distinct outcome measure;
`docs/DECISIONS.md`'s TASK C entry) is weak and non-significant throughout
(best case rho=0.325, p=0.079, n=30) — that gradient remains open. A
companion population-level check (do checkpoints that measure better on
`n_unstable`/`||K_s||/||K_x||`/etc. actually
transfer more often?) found a real but weak association when pooling
across every architecture tried this session, which does NOT replicate
within the flagship d1030 M3 population and is not, on Task A's own
direct-intervention evidence above, actionable there.

**Sharpened further, 2026-08-18 (`docs/DECISIONS.md`'s fourth-round TASK
A):** the free-response failure isn't only a multi-step or s0-choice
effect. M3's raw one-step x-to-x Jacobian (`Axx`, a single constant
matrix since M3 is exactly affine — no evaluation-point ambiguity) is
`~92%` off from `A_d` in relative Frobenius norm at every one of 30
checkpoints (range 57%-140%) — already wrong at the most local, single-
step level, not a pattern that emerges over a horizon. `||Axs||`
(sensitivity to the internal S4 state) is 5-15x larger than `||Axx||`
in every checkpoint: M3's x-prediction is dominated by the internal-state
channel, not the x-input channel. **A prior "J_x ~ A_d, error ~0.003"
measurement was referenced but could not be located anywhere in this
repository or its git history — superseded by the 91.7% figure above
(`docs/DECISIONS.md`'s BOOKKEEPING entry), not left as two contradictory
numbers.**

**Reframed and proven, 2026-08-18 (`docs/DECISIONS.md`'s fifth-round TASK
A):** not "the model got the dynamics wrong" but gauge freedom, and this
is now a proven fact, not a hypothesis. `x_next = Axx@x + Axs@s + Bx@u` is
exactly linear in `(Axx, Axs)` holding `s_t` fixed, so the Gauss-Newton
null space restricted to those coordinates is exactly the null space of
the stacked training data `Z=[X|S]` — a real theorem, not an analogy.
Direct test on the real training trajectory (`s4dpc.data`,
`DATA_SEED=42`, matching real identification exactly): for every
timestep `t>=1` (99 of the 100 training samples), `S`'s column space
contains an EXACT combination (residual ~1e-16 to 2e-5) reproducing any
specific physical-state perturbation — the ONLY sample that constrains
`Axx` at all is `t=0`, where `s_0=0` by the project's own cold-start
convention structurally forbids any compensation. Confirmed identical for
M0_S4: the same gauge orbit exists there too — M0_S4 doesn't avoid the
ambiguity, it sits by hand construction at the orbit's one transfer-safe
point (`Axs=0` exactly), and identification has no gradient toward that
point because the objective is exactly constant across the whole orbit.
Four measurements are one fact, seen four ways: no Hankel cliff at rank 6
(the excess ~1024 dims aren't inert), `Axx` 91.7% off `A_d` (which point
of the orbit training landed on), `||K_s||/||K_x||` ~91x median (the LQR
gain inherits the same imbalance), and this entry's exact t>=1
compensation (the mechanism that makes the other three possible, not
merely correlated with them). Caveat: this proves non-identifiability of
`(Axx,Axs)` given the rest of the model fixed, not joint identifiability
of the whole parameter set — and it does not explain why training lands
91.7% off rather than by chance nearer the safe point (a possible soft
optimizer/init bias, not investigated).

**IDENTIFIABILITY RESTORED, 2026-08-18 (`docs/DECISIONS.md`'s fifth-round
TASK C, framing corrected same day before this reached the paper —
read the FRAMING CORRECTION entry for the full reasoning).** Not "we
fixed learned surrogates" — that framing invites a fair rebuttal (the
result, read broadly, is indistinguishable from classical order
selection / model-order reduction, already solved). **The defensible,
narrower claim:** the missing constraint is identified precisely
(`(Axx,Axs)` exactly non-identifiable past `t=0`, TASK A, proven not
observed); the objective has NO GRADIENT toward the safe point of that
ambiguity (exactly flat, not merely hard to find); and a MODIFIED
objective (dither-augmented data — Gustavsson/Ljung/Söderström's cure
for collinear closed-loop regressors, which is exactly teacher forcing's
structural situation here) recovers the safe point exactly. Mechanically:
re-fit `(Axx, Axs, Bx)` by closed-form OLS (no GPU, no gradient descent)
on real training data plus synthetic `(x, s_random, u) -> true_x_next`
samples, target computed directly from `A_d`/`B_d`. Result, all 30 fullM3
checkpoints: `Axx -> A_d`, `Axs -> 0` to floating-point precision, every
checkpoint transfers at essentially oracle-optimal cost (median 1.005x).
**That the recovered gauge happens to equal M1's model is a property of
these particular fully-observed, Markov-in-`x` plants, not the mechanism
being demonstrated** — restoring identifiability is the result; a small
recovered model is this dataset's consequence of it, not the claim.
Other caveats: synthetic targets used known `A_d,B_d` directly (valid
for this project's known-simulator setting, not a recipe against an
unknown black-box real plant); not yet validated under actual end-to-end
gradient-descent retraining, only a closed-form readout re-fit.
**SCOPE BOUNDARY, load-bearing for the paper:** this works because these
plants are fully observed and Markov in `x` — a partially-observed
system needs `s` to carry genuine information and cannot have it
randomized freely without destroying real signal. The
partially-observed case is the real open problem this result points
toward, not something this fix already covers.

**TRUNCATION RULED OUT, 2026-08-18 (`docs/DECISIONS.md`'s TASK D) — the
deferred fidelity-matched-truncation experiment, finally run, closing
the last open question about excess content vs. gauge.** Hankel-SVD/ERA
(independent of the (Axx,Axs) construction TASK A/C used) reduced each
fullM3 checkpoint to the smallest order reaching that SAME checkpoint's
own achieved Markov-parameter fidelity (not a generic "~1e-6" — at this
experiment's h<=40 window, M3's own fidelity is looser, ~0.02-0.7,
making this if anything a generous test for truncation). 26/30
checkpoints already reach their own fidelity target at `r=6` — the true
plant's own order, zero excess dimensions. Regardless of `r`: **28/30
still catastrophically fail** (ratios up to 8e129; the 2 successes are
marginal, 66x/81x). M1/M0_S4 controls: 30/30 each, as expected.
**Verdict: excess realization content is not the culprit — a second,
independent reduction method at minimal order and matched fidelity
reproduces the same failure. Truncation is cleanly ruled out as a
general-purpose cure; TASK C's dither cure remains the one route that
worked, and by elimination this confirms it works BECAUSE it restores
identifiability, not because it happens to produce a small model.**

**VERIFIED INVARIANT, 2026-08-19 (`docs/DECISIONS.md`'s TASK 0) —
conv-mode (identification) and stepped-mode (every control/LQR-transfer
result above) agree to float64 machine precision, on all 30 real trained
fullM3 checkpoints, at s0=0 (the only state conv mode can represent —
verified directly from `s4-nnx` source: conv mode hard-rejects any
sequence length != l_max, and its `previous_state` argument is accepted
but never used, confirmed empirically bit-identical across random
states on every checkpoint).** Every result above that depends on this
equivalence is safe. **Companion finding: M3 is affine, not linear —
`f(0,0)!=0` (`equilibrium_drift`, median norm 0.83 across the 30
checkpoints) — and `lqr_transfer_to_true_plant.py`, `free_response_test.py`,
`dither_cure_test.py`, and `fidelity_matched_truncation.py` all
originally propagated `z=z@Acl.T` with no `+c0` term.** Every
EIGENVALUE-based verdict in this document (`rho_transfer`, `n_unstable`,
PBH, the Hankel spectrum, the gauge-freedom theorem) is exactly
unaffected — `c0` never enters an eigenvalue computation. **RE-VERIFIED,
2026-08-19 (`docs/DECISIONS.md`'s bias-term-round TASK A/B/C entries) —
every headline claim in this document is confirmed unchanged after
adding `+c0`, not merely "expected to be robust."** The correct impact
bound is sharper than "eigenvalue verdicts survive": with an affine
term the closed-loop fixed point is `z*=(I-Acl)^-1@c_open`, so a stable
`rho<1` no longer implies regulation to the origin — it could mean
convergence to a nonzero offset, which would not be a real success.
Checked directly for every claimed success: the dither cure's 30/30
near-oracle result recovers `c0_x` to machine-zero (range
[1.1e-16,4.5e-15]) as part of the SAME re-fit that fixes `Axx`/`Axs`,
giving `||z*_x||=0.000000` exactly on all 30 and `ratio_corrected`
identical to 4 decimal places — the cure was never masking an offset.
The two truncation successes (66x, 81x) need no correction at all —
Hankel-SVD/ERA is structurally incapable of representing an additive
term (verified by direct inspection of `era()`, not re-run), so its
"original" ratio already was the `+c0`-correct one; a companion check
also found one of those two (case4/seed3, 65.8x) has `rho_transfer>1`
— i.e. was already an unstable-loop finite-horizon artifact, not a
converged low-cost trajectory, independent of anything to do with
`c0`. The LQR-transfer (median 25,300x → 30,266x) and free-response
(median err_t1 0.769 → 0.753, err_t200 1.045 → 1.016) numbers shift
5-30% per checkpoint but no median, extremum, or pass/fail verdict
changes — `rho>1` for all 30 in both the original and corrected LQR-
transfer accounting, meaning those trajectories diverge outright and
never reach a fixed point to check for an offset in the first place.
The one exception, honestly flagged rather than silently dropped: TASK
A's stability-hinge success (6.5x, a single checkpoint from an earlier
round) could not be re-derived, since that identification run never
saved raw params and `c0` is not recoverable from the cached
`Abar/Bbar/K_lqr` alone — re-deriving it would need new GPU time, which
this correction round deliberately scoped out (CPU-only). The paper's
actual positive claim (the dither cure) does not depend on that one
result. **No conclusion in this document flips under the corrected
accounting.**

**RETRACTED, 2026-08-25, by an independent reproduction — the entire
gauge/non-identifiability/dither-cure mechanism chapter above (every
paragraph from "Reframed and proven" through "TRUNCATION RULED OUT" that
rests on `Axx`, `equilibrium_drift`, or the dither cure) — full record in
`docs/DECISIONS.md`'s "EXTERNAL REPRODUCTION" entry, retracted with the
same standard as the kink refutation, not softened.** An outside group
reimplemented this project from scratch and independently CONFIRMED the
central failure (0/30 stable transfers, coupling `||A_xs||/||A_xx||`
5-16) — that claim stands and is now stronger, not weaker. But they
train identification on 320 trajectories; this codebase trains on
`batch_size=1` — ONE 100-step trajectory, 100 samples against a
1024-dim latent (confirmed directly, `s4dpc/identify.py`'s
`case_data`/`run_identify`) — and at B=1, `rank(S)>=rank(X))`
automatically, making `Axx` formally unidentifiable by construction.
Our 91.7% `Axx` error and 0.83 `equilibrium_drift` figures reproduce
almost exactly at their B=1 setting and shrink by orders of magnitude
at B=320 — they are B=1 data-volume artifacts, not properties of what
M3 learned. The gauge-freedom proof's linear algebra was not wrong, but
its reading was: "the internal state's column space contains an exact
compensation" is, at one sample per timestep, just "a nonzero vector in
R^1024 absorbs any vector in R^6" — true for pure noise, not evidence
of a meaningful learned symmetry. The dither cure inherits both
problems, plus a separate, independently fatal one neither our record
nor the external group's caught until now: an unconstrained `(6,1024)`
`Axs` from an OLS refit corresponds to no set of S4 weights at all — a
realisable `Axs` has `rank<=6` with rigid block structure — so the
dither cure's 30/30 near-oracle result was never achievable by an
actual retrained network. **Critically, none of this explains away the
failure — transfer fails at every B tested (`rho` 1.0206→1.0172, B=1
to B=320) and the coupling ratio never moves; more data cures the
symptoms we measured and leaves the disease untouched.**

**THE MECHANISM WE ARE ADOPTING INSTEAD (external claim, not yet
independently re-verified on our side — flagged as such until it is):**
the `A_xs` coupling block itself, not identifiability. Zeroing `A_xs` at
synthesis (no retraining) converts 356x into 1.0013x; discarding `K_s`
instead makes it WORSE (1.6e113x) — refuting a "wasted gain" reading,
since Riccati co-designs `K_x`/`K_s` jointly for a plant that doesn't
exist and neither half works alone. The margin identity: at zero
coupling the augmented closed loop is block-triangular, so
`rho = 1 - max|eig(A_ss)|` exactly (matches to 1.33e-15) — the entire
margin available to absorb `A_xs` IS the least-damped S4 mode's own
damping, and HiPPO places 462 of 1024 modes within `1e-2` of the unit
circle. Their proposed cure — a scale-normalized `||C||^2` penalty
during TRAINING (not a post-hoc linear-algebra readout) — reportedly
hits 30/30 at oracle cost while improving one-step MSE by 24 orders of
magnitude. Not yet run here.

**Two of our own bugs found and verified during this correction, kept
distinct from the retraction above since they are real, independent of
the mechanism story:** (1) the M1/M0_S4 "1.005x" figure throughout this
document is a genuine bug, not noise — `tools/lqr_transfer_to_true_plant.py`'s
numerator (`simulate_cost`) normalizes by `EVAL_HORIZON=200`, its
denominator (`true_quadratic_cost` via `rollout_lqr_true`) normalizes
by `x_hist.shape[0]=201` — `201/200=1.005` exactly, matching the
reported figures to the residual digit. Verified, not yet fixed. (2)
M6's conv/step parity gap (`docs/DECISIONS.md`'s "TASK 0 EXTENSION"
and follow-on "TASK A"/"TASK B" entries, 2026-08-19) is UNAFFECTED by
any of this retraction — it is an independent numerics finding about
`s4-nnx`'s Cauchy-kernel-vs-scan code paths, not part of the gauge/
dither chapter, and stays on the record as-is.

Plants are 7 discrete-time linear systems, all `A: (6,6)`, `B: (6,3)`. Ground truth
`A_d`, `B_d`, Markov parameters and an oracle LQR controller are all computable, which
is the entire reason for using linear plants.

**Timeline: 7 days.** Bias every judgement call toward "ships this week".

---

## 2. Repo layout

```
s4dpc/
  requirements.lock          # EXACT pins. Do not edit without re-running parity.
  env_probe.py               # version + device + param-tree + numerics canary
  s4dpc/
    model.py                 # blocks + stacked model (consumes s4-nnx)
    blocks.py                # configurable block: norm / activation / glu
    systems.py               # get_discrete_system(case) -> A_d, B_d, name
    identify.py              # system identification, all variants
    control.py               # GRU controller + DPC loss (action IS bounded)
    diagnostics.py           # Markov params, gradient cosine, equilibrium drift
    sweep.py                 # THE entry point. argparse + vmapped harness + CSV
    logging.py               # Logger (online/offline/off), row schema, classifier
  tests/
    test_parity.py           # refactor == original, BIT-EXACT
    fixtures/reference_model.msgpack
  launch/
    kaggle/                  # kernel-metadata.json + runner notebook/script
    tacc/                    # job.slurm, build_env.sh, submit_all.sh
  docs/
    00_protocol.md           # frozen tables + figure captions + kill criteria
    DECISIONS.md             # append-only, dated
    LOG.md                   # append-only, one entry per batch of runs
  results/                   # gitignored, CSV output
  gpu_ledger.csv             # Kaggle GPU quota tracking
```

The S4 layer itself lives in a **separate pinned package** (`s4-nnx`, by git tag).
Never vendor or edit it from here — see §7.

---

## 3. Golden rules

1. **`sweep.py` is the only entry point.** Notebooks, Kaggle kernels and SLURM jobs
   all invoke `python -m s4dpc.sweep` with different flags. There is no
   platform-specific training code. If a platform seems to need its own version, the
   port is wrong — fix the port.
2. **Never edit code on TACC.** Develop locally/Kaggle, push, `git pull --ff-only` on
   the login node.
3. **Never bump a version in `requirements.lock`** without re-running `tests/test_parity.py`
   on all platforms.
4. **Never split one experiment arm across platforms.** All seeds of a variant run on
   the same hardware — XLA lowering differs by backend, and BPTT through unstable
   dynamics amplifies float differences.
5. **CSV is ground truth. W&B is a lens. Figures come from the CSV.**
6. **Diverged runs are results, not failures.** Tag, never delete.
7. Stamp `git_sha` and `lockfile_sha` into every CSV row.

---

## 4. Kaggle CLI workflow

Kaggle is for **development and parity testing**, not production sweeps. Fast feedback,
small configs, `--epochs 50`.

### Kernel metadata

`launch/kaggle/kernel-metadata.json`:

```json
{
  "id": "USERNAME/s4dpc-smoke",
  "title": "s4dpc smoke",
  "code_file": "runner.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

`enable_internet` must be true — the kernel clones the repo and pip-installs the lock file.

### Commands

```bash
# push
kaggle kernels push -p launch/kaggle

# poll until "complete" or "error"
kaggle kernels status USERNAME/s4dpc-smoke

# pull results + log
kaggle kernels output USERNAME/s4dpc-smoke -p ./out
```

### The runner cell

```python
!git clone https://github.com/USER/s4dpc.git /kaggle/working/s4dpc
!pip install -q -r /kaggle/working/s4dpc/requirements.lock
# RESTART REQUIRED if kaggle preinstalled a different jax
!cd /kaggle/working/s4dpc && python env_probe.py
!cd /kaggle/working/s4dpc && python -m s4dpc.sweep \
    --variant M6 --cases 3 --n_seeds 2 --epochs 50 \
    --wandb off --out /kaggle/working/out.csv
```

### GPU ledger — REQUIRED after every run

After every successful `kaggle kernels output`, parse the total runtime from the `.log`
in `./out` and append one line to `gpu_ledger.csv`:

```
date,kernel_slug,accelerator,minutes
```

**Before pushing any GPU job**, sum the current week's rows (quota resets **Saturday
00:00 UTC**) and report the running total to the user. Do not push if the remaining
budget looks tight without saying so first.

### Colab CLI — for INDEPENDENT experiment arms only, when Kaggle's 2 slots are full

The `colab` CLI (`uv tool`-installed, symlinked at `~/.local/bin/colab` — add that to
`PATH` the same way `kaggle` needs it) provisions and drives real Google Colab VMs
from the shell: `colab new -s <name> --gpu T4`, then run code on it, then
`colab stop -s <name>`. Full usage is in `colab skill` / `colab readme`.

**Only for a genuinely separate experiment arm** — never to finish or split an arm
already started on Kaggle (CLAUDE.md §3 rule 4: XLA lowering differs by backend, and
BPTT through unstable dynamics amplifies float differences across platforms). If Task
X is running on Kaggle and Task Y is independent and ready to launch, and both Kaggle
slots are occupied, Colab is where Y goes — not a shortcut around the 2-slot limit for
splitting Task X.

**`colab exec` was broken on CLI 0.6.0, root cause found and hand-patched locally
(2026-08-14):** `AttributeError: module 'jupyter_kernel_client' has no attribute
'KernelClient'`. Not a `colab-cli` bug per se — `google-colab-cli`'s dependency on
`jupyter-kernel-client` is unpinned, and the installed `jupyter-kernel-client==1.0.1`
renamed the class `KernelClient` → `JupyterKernelClient` (constructor signature
unchanged — pure rename). Fixed by editing the one call site directly in the
installed venv:
`~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/runtime.py:106`,
`jupyter_kernel_client.KernelClient(` → `jupyter_kernel_client.JupyterKernelClient(`
(clear `colab_cli/__pycache__` after editing). **This is a local hotfix to a
uv-tool venv, not an upstream fix** — a `uv tool upgrade`/reinstall of
`google-colab-cli` will silently revert it and the same `AttributeError` will come
back. If `colab exec` breaks again with this exact error, look here before
re-diagnosing from scratch; if the reappeared error is different, it's a new issue.
`colab exec -s <name> -f script.py` (or piped stdin) is now the preferred way to
run code — prefer it over the `console`+tmux workaround below when it's available.

**Fallback if `exec` is broken again**: `colab console -s <name>`, piping a full
shell command in (`echo "cd /content && git clone ... && pip install ... && python
script.py" | colab console -s <name>`), matching the Kaggle notebook cell pattern
(`!git clone && !pip install && !python`).

**Long jobs**: the piped command keeps running in the session's tmux pane even if the
local `colab console` connection drops (e.g. a Bash tool timeout) — reconnecting later
with another `colab console -s <name>` finds it still going. Do **not** poll
`colab status` to check progress on a `console`-launched job — that reflects the
*Jupyter kernel's* idle/busy state, not the tmux shell's, and reads IDLE the whole
time regardless of what's running in the pane. Instead have the job write its final
output to a known path and poll for that file with `colab download <remote> <local>`
(safe to call while the pane is busy — it's a separate file-transfer channel, not
console input). Do not pipe a status-check command into a *busy* console — with no
free shell prompt to run it, the keystrokes just queue silently instead of executing.

**Always `colab stop -s <name>` when the arm is done.** The keep-alive daemon prevents
idle timeout specifically so a forgotten session does *not* get auto-reclaimed — that
means a session left running keeps burning compute units indefinitely until someone
stops it.

### `launch/colab/orchestrate.py` — for jobs that must survive a VM death

Built 2026-08-14 after three straight lost long-running Colab jobs. `python
launch/colab/orchestrate.py --session <name> --script tools/some_script.py` bootstraps
a session, launches the target script backgrounded (`python -u`, unbuffered — plain
`python` leaves stdout fully buffered when redirected to a file, so a log can sit
empty for many minutes of genuine progress, confirmed directly), then loops: a real
`colab exec` health check every cycle (not "is the local keep-alive pid still
running" — confirmed directly that a live local daemon pid is NOT evidence the remote
VM is still there), and a verified **per-file** backup sync remote→local into
`docs/nu_gap_export/` (never a directory-level `colab download`/`upload` — both are
single-file-only in this CLI version; `ContentsClient.download` raises
`IsADirectoryError` on a directory path, and three straight backup attempts before
this script existed had exactly that call piped to `/dev/null`, so it was failing on
every cycle regardless of session health). On a detected death (failed health check,
or N consecutive cycles with no new verified artifact), it stops the session,
relaunches, **re-uploads the local backup before re-running the script** so the
script's own resume logic (per-case `case_done()` in `tools/nu_gap_export.py`) sees
prior progress. `python launch/colab/status.py <name>` gives a one-line status from
another shell.

**Known, structural limitation — read before trusting this for an unattended run:**
this only survives the *Colab VM* dying. It cannot survive *this Claude Code session's
own host environment* going away (laptop sleep/shutdown, sandbox teardown) — every
watcher and the local keep-alive daemon are child processes of that same session.
Confirmed directly: a run was lost to a ~9.5 hour gap in the session's own execution
(not a Colab-side failure at all — `colab sessions` still showed the VM assigned
right up until a `colab exec` health check finally 404'd it). No amount of
in-session monitoring fixes that; it needs either a genuinely persistent host, or
accepting the exposure and preferring Kaggle (whose kernels run fully server-side,
per CLAUDE.md §4 above) for anything that must survive many hours unattended.

**Stall-detection threshold gotcha**: pick the "N cycles with no new artifact ⇒
treat as dead" threshold from *measured* per-phase wall-clock time, not a guess — a
too-short threshold kills healthy jobs (confirmed directly: killed a genuinely
working M1 run twice by guessing 3-4 min and 15 min tolerances against a job whose
curriculum phases individually ran 10-15+ min each). Killing a session this way
mid-`colab new`/`colab stop` can also leave an **orphaned VM assignment** with no
local record (`colab sessions` shows it as `[?]`) that `colab stop` can't target
(it needs an auth token only the killed process had) — these have so far always
self-cleared within under an hour (no keep-alive daemon ever started for them), but
can also transiently exhaust the account's concurrent-assignment slots
(`TooManyAssignmentsError`, HTTP 412) until they do. Only measure a phase's real
duration by watching an **undisturbed** run (no auto-relaunch armed) before trusting
a threshold on a real job.

---

## 5. Parity testing — DONE (2026-08-08, `docs/DECISIONS.md`); kept as reference

No longer the current task — the refactor passed L1/L2/L3 below and control-side work
(`s4dpc/control.py`, the M0/M1/M3/M6 DPC comparisons) has been running on top of it since.
Current focus is the control-failure mechanism itself (§1's correction) — see
`docs/DECISIONS.md`'s most recent entries for what's actually being worked on. This
section stays as reference for what to do if a future dependency bump (§7) requires
re-running parity.

Goal: the refactored code (which calls `s4-nnx` for the SSM core) must reproduce the
original notebook code **exactly**.

**Acceptance criteria — bit-exact, `rtol=0, atol=0`:**

- **L1** — load `tests/fixtures/reference_model.msgpack` into both; same input →
  identical output.
- **L2** — identical gradients.
- **L3** — both `decode=False` (conv, `L == l_max`) and `decode=True` (stepped).

`rtol=1e-5` hides real bugs. Use zero tolerance. If the op sequence is unchanged, XLA
emits the same kernel and the outputs are bit-identical.

**Do not compare by training both and eyeballing loss curves.** `nnx.Rngs` is stateful,
so any change to key-draw order changes every parameter, and you cannot distinguish a
bug from a different random draw. Parity is tested against a **saved checkpoint**,
independent of initialization.

If param trees differ under the same seed, that is expected — record it in
`docs/DECISIONS.md` and note that pre/post-refactor runs are not seed-comparable.

---

## 6. Variant ladder

Set by config, never by branching code:

| Variant | `norm` | `activation` | `glu` | Isolates |
|---|---|---|---|---|
| M0 | — | — | — | true `(A_d, B_d)` oracle |
| M1 | — | — | — | least-squares `(Â, B̂)` |
| M3 | none | none | False | LTI S4 — is capacity the issue? |
| M4 | none | gelu | True | activation curvature |
| M5 | layer | none | False | normalization / kink source (refuted as DPC cause — see correction below) |
| M6 | layer | gelu | True | current full model |
| M6_fix | static | gelu | True | fixed standardization (the proposed fix) |

**M3 is the pivotal cell.** A linear S4 is an LTI system and should recover the Markov
parameters to ~1e-6. If it does, capacity is not the problem and the nonlinearities are.
If it doesn't, the paper pivots to the spurious-memory/realization story — flag this
immediately.

**Correction, 2026-08-13: this decision rule fired, and the branch it pointed to was
wrong.** M3 DID recover Markov parameters to ~1e-6 (`docs/DECISIONS.md`'s 10-seed
identification entry) — by the rule as written, that meant "capacity is not the
problem, the nonlinearities are," which pointed straight at M5/M6's LayerNorm kink.
The kink was real (M6 has one, M3 provably can't) and looked like a clean confirmation
on identification-side data alone. But when actually tested on control-side data (M3
vs M6 through DPC), M3 fails just as catastrophically as M6 despite its exact
near-machine-precision Markov fidelity and zero kink — see §1's correction and
`docs/DECISIONS.md`'s "REFUTATION" entry. **The rule's premise was incomplete: "M3
recovers Markov parameters" answers whether M3's *local linearization* matches the
true plant, not whether M3 is *safe to backprop a controller through* — those turned
out to be different questions, and the ladder as designed had no cell that tested the
second one directly.** That gap is what the current working picture (§1) and the
session-by-session entries in `docs/DECISIONS.md` are now closing (M0_S4 — a
hand-constructed S4 forced to realize the true plant, deployed through the same
BPTT/control machinery — is the closest thing to a corrected version of this rule).

---

## 7. Versions

Pinned in `requirements.lock`. Tested combination:

```
jax==0.7.2   flax==0.11.2   optax==0.2.8
numpy  scipy  pandas  matplotlib  tqdm  wandb
```

**Do not move to flax 0.12.x this week.** 0.12.0 introduced breaking NNX changes around
Modules holding Arrays; `StackedModelRegression` uses a plain `self.layers = []` +
`.append()` pattern that sits in that blast radius.

`venv` + pip only. **No conda** — the `jax[cuda12]` wheels bundle their own CUDA, and a
conda `cudatoolkit` shadows them and produces linker errors that look like hardware
faults.

`s4-nnx` is pinned by git tag:
```toml
dependencies = ["s4-nnx @ git+https://github.com/USER/s4-nnx.git@v0.2.0"]
```
It is frozen for the week. If you want to change it, move the code into `s4dpc` instead.

---

## 8. W&B

One flag, three modes: `--wandb {online,offline,off}`.

- `online` — Kaggle / Colab
- `offline` — TACC compute nodes (no outbound internet); `wandb sync` later from the login node
- `off` — tests, smoke runs, parity. **This is the default.**

Never scatter `if use_wandb:` through the code. Use the null-object `Logger` in
`s4dpc/logging.py`.

**Projects:**
- `s4dpc-dev` — all development, debugging, smoke runs. Default.
- `s4dpc` — the real sweep. Nothing enters until parity passes and the schema is frozen.

Runs are created **post-hoc from CSVs** by `push_to_wandb.py`, with a deterministic
`id` so re-pushing updates rather than duplicates. This is what makes ~35 vmapped
logical runs per job into 35 separately-filterable W&B runs.

**Never commit a W&B or GitHub token.** They load from the environment.

---

## 9. Parallelism

`jax.vmap` over `(case × seed)` inside one process; `sbatch` across configs. That is the
whole strategy.

**Do not add Ray, joblib, or multiprocessing.** All 7 systems share shapes, so 35 runs
fuse into single kernels — no process overhead, no VRAM contention, no second scheduler
fighting SLURM.

For readable tracebacks during debugging, `sweep.py` accepts `--no-vmap`, which loops in
Python. Same function, same outputs. That is the "single run" path — a flag, not a
second script.

---

## 10. When you finish a batch of runs

Append to `docs/LOG.md` with three coordinates — **sha, wandb group, csv path** — then
what was concluded and any anomaly:

```markdown
## 2026-08-11 — M3/M5/M6 identification, 18 archs × 10 seeds
sha: a3f91c2 | wandb: M5_case* | results/M5_*.csv

M3 recovers Markov params to ~1e-6 on all cases → capacity is not the issue.
M5 markov_err_h10 jumps 2-3 orders on cases 4/6 → supports the norm mechanism.
Anomaly: case 7 seed 3 diverged in identification, tagged bad_id, not rerun.
```

Any decision that changes what enters the paper goes in `docs/DECISIONS.md`, dated,
with the reasoning. Reviewers ask exactly these questions.

---

## 11. Things that will waste the week

- Debugging on a batch queue instead of Kaggle (20-min feedback loops)
- Refactoring `s4-nnx` mid-week
- Adding the nonlinear benchmark on day 3 because the compute is there — it costs the
  writing days; it is future work
- Manually transcribing numbers into LaTeX instead of generating `\input`-able tables
- Writing the research log retroactively on day 6

---

## 12. TACC operational flow

Added 2026-08-18 alongside `launch/tacc/` — the Lonestar6 execution layer.
Full detail lives in `launch/tacc/README.md`; this section is the short
version so it's visible without leaving CLAUDE.md. This section documents
operational flow only — it changes nothing in §1-§11 above.

**The path a run takes:**

```
EE-119537 (local orchestration)
  → GitHub (push to main)
  → TACC: git pull --ff-only (launch/tacc/sync_tacc.sh, never edits source there)
  → gpu-a100-dev smoke (launch/tacc/smoke_test.sh)
  → gpu-a100-small production (launch/tacc/submit_all.sh)
  → Slurm (launch/tacc/job.slurm — platform-only wrapper around `python -m s4dpc.sweep`)
  → CSV/result retrieval (launch/tacc/pull_results.sh, into $WORK then locally)
  → docs/LOG.md (append git SHA + CSV path + conclusions/anomalies, same as Kaggle runs)
```

Also holds, same as every other golden rule in this file:

- TACC's Python is `python/3.12.11` via `module load` — the default
  `python3`/3.9.7 on TACC must never be used for this project.
- Scientific code is never edited on TACC — only cloned/pulled. `s4dpc/sweep.py`
  remains the only entry point on every platform (§3 rule 1).
- The user controls scientific experiment definitions (variant, cases, seeds,
  epochs, architecture, `--wandb` mode). Claude does not invent or alter these.
- Claude may monitor and collect the results of approved jobs (`status.sh`,
  `pull_results.sh`).
- Claude may not independently launch follow-up experiments on TACC — every
  `sbatch` submission is something the user explicitly asked for.
- Resource escalation — a bigger queue (`gpu-a100`/`gpu-h100`), more nodes,
  longer walltime, or more concurrent jobs than already approved — requires
  explicit user approval before Claude submits it.

**Validated 2026-08-19 (full narrative + exact numbers in the top-level
README.md's "TACC / Lonestar6 execution layer" section) — three more
invariants specific to the TACC execution path, load-bearing, do not
revert:**

- Real sweeps require `jax_enable_x64=True` — S4 conv/recurrent parity
  was off by ~0.67 (M3) / ~0.089 (M6) under float32 on TACC's A100s,
  collapsing to ~1e-13/~1e-14 under x64. Float32 is not a supported
  production mode for this codebase's S4 path.
- `launch/tacc/job.slurm` must run `module reset && module load
  python/3.12.11` before activating the venv — a batch job's inherited
  module state was found to silently swap the venv's Python 3.12.11 for
  3.12.13 otherwise.
- `launch/tacc/job.slurm` must derive its repo root from
  `$SLURM_SUBMIT_DIR`, never `BASH_SOURCE` — Slurm executes a spooled
  copy of the batch script, so `BASH_SOURCE[0]` pointed into the spool
  directory rather than the actual checkout (this is exactly what made
  smoke job 3377181 fail immediately with "fatal: not a git repository").

---

## 13. Multi-machine sync protocol

Added 2026-08-25 once three local machines (ThinkPad P53, Lab PC, Home PC —
full detail in `docs/ENVIRONMENTS.md`) were in regular use alongside Kaggle/
Colab/TACC, all pushing to `main` directly (no branch protection is
configured on this repo — confirmed via the GitHub API, not assumed). The
risk isn't malicious, it's silent divergence: two machines editing the same
files without syncing, or a result row from one machine's backend getting
mixed into an analysis that assumes another's (CLAUDE.md §3 rule 4).

**The three rules:**

1. **Pull before starting work, on ANY machine, every time** — `git pull
   --ff-only`. If it's not a clean fast-forward, STOP and look at what
   diverged before doing anything else; don't force past it.
2. **Commit and push before switching machines** — uncommitted work on one
   machine is invisible to every other one. Don't leave a machine holding
   the only copy of something overnight.
3. **Never force-push** to `main`, from any machine, under any
   circumstances covered by this file. (The repo-wide git safety rules
   already forbid this by default — this section is not a new exception
   path, just restating it applies identically across every machine.)

**One-line "what state am I in" check**, run this before assuming a
machine's checkout is current or its working tree is clean:

```bash
git fetch origin && git status --short && git log -1 --oneline && \
  git rev-list --left-right --count main...origin/main
```

Empty `git status --short` output = clean tree. `0	0` from the last
command = local `main` and `origin/main` are identical (no unpushed local
commits, nothing to pull). Anything else — investigate before running
`sweep.py`, launching a Kaggle/TACC job, or committing.
