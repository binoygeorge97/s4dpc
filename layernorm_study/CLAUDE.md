# CLAUDE.md — layernorm_study

Scoped override of the parent repo's `CLAUDE.md` §4/§12 compute policy,
for this sub-project specifically. Everything else in the parent
`CLAUDE.md` (git safety rules, sync protocol, golden rules) still
applies unchanged - this file only overrides where and how experiments
run.

## Compute policy (added 2026-08-28, after a direct policy conflict was raised and resolved)

**layernorm_study's scalar-plant experiments run LOCALLY. Not TACC.**

The parent repo's CLAUDE.md §4/§12 says "main runs go on TACC via
SLURM, Kaggle for smoke tests only." That policy was written for the
parent project's real cases - 6-state DPC systems, `d_model` up to
1024, runs that need a GPU and can take hours. It does not fit this
sub-project: every arm here is a scalar (or near-scalar) plant at
`d_model=8-16`, and a single 60000-epoch training run finishes in
well under a minute of CPU time. Queuing for an A100 to do that would
be strictly worse than running it locally - the GPU would sit idle
and the run would pay TACC's queue latency for no benefit.

**Concretely:**
- layernorm_study's own experiments (everything under
  `layernorm_study/experiments/`) run on local CPU by default. This
  was true for every number produced in rounds 1, 1.5, and 2 - see
  NOTES.md's provenance note.
- TACC via SLURM (parent CLAUDE.md §12, `layernorm_study/slurm/`) is
  for anything that (a) touches the parent project's multi-state DPC
  cases, or (b) genuinely needs a GPU or a run longer than roughly 30
  minutes. `layernorm_study/slurm/run_arm.slurm` and `submit_all.sh`
  exist for when this sub-project actually reaches that scale (e.g. a
  full multi-state postnorm-boundedness study); they were written in
  round 1 and have not needed to be used yet.
- Kaggle stays smoke-tests-only, per the parent policy, unchanged.

**The judgment call that produced this file:** the parent policy was
followed by omission for three rounds without anyone noticing the
mismatch - not by a deliberate, disclosed decision. When a stated
policy doesn't fit what the work actually needs, the right move is to
raise it explicitly and get a decision, not to quietly follow it or
quietly depart from it. This file is that decision, made once and
recorded, instead of a per-round judgment call repeated silently.

## Environment

`layernorm_study/requirements.txt` - exact pins, captured directly
from the environment that produced every existing number (2026-08-28).
Python 3.11.15, CPU only. Recreate via:

```bash
python3.11 -m venv layernorm_study/.venv
source layernorm_study/.venv/bin/activate
pip install -r layernorm_study/requirements.txt
```

`layernorm_study/.venv/` is gitignored (the venv itself isn't
committed - only the pin list) but is the persistent, in-repo home for
this sub-project's environment going forward, replacing an earlier
session-scoped `/tmp` location that was not durable (confirmed
directly - it was wiped once already mid-round-2).
