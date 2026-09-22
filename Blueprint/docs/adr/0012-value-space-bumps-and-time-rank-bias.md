# ADR-0012 — Bins are value-space bumps; temporal bias uses within-bump time rank

**Status:** Accepted · **Date:** 2026-09-22 · **Supersedes:** SPEC v0.7.0 §4.2 and §4.3
· **Evidence:** [AUDIT], gated by G3 + G4 (ADR-0011)

## Context

v0.7.0 read Algorithm 1 as producing **time-contiguous** runs of rare/normal cases, and
the temporal bias as `p = i/n` with `i` the position in the full series. Its Node 4 brief
called the within-bin rank "the single most likely bug".

## Decision

**Bumps (SPEC §4.2).**
- Order cases by `y` and cut where φ crosses `t_R`:
  - under and smote use a sign-change rule with `>`;
  - over uses `≥`.
- Each bump is classified by its **mean φ**:
  - under and smote: rare iff mean `> t_R`, normal iff mean `< t_R`;
  - over: rare iff mean `≥ t_R`.
- Typical results are 2 bumps (one-sided φ) or 3 (low-rare, normal, high-rare).

**Bias (SPEC §4.3).** Inside a bump of `r` cases ordered chronologically, the case with time
rank `j` gets preference `j/r` (temporal), times `φ_j` for temporal+phi. The rank is within
the bump, not the position in the series.

## Evidence

- These are the R conventions per the 2026-09-22 audit.
- **Corroboration [ORACLE-indirect]:** an implementation using them reproduces the
  canonical `lm` + resampler mean F1 to 0.009–0.012 across 20 datasets, which is seed-noise
  level. `lm` is deterministic given its training set, so the resampled training sets must
  have essentially the right composition.
- Time-contiguous runs would produce dozens of bins per split and completely different
  target sizes, which is irreconcilable with that agreement.
- Not directly observable in recorded output (training sets were not saved), so G3 pins
  the structure on worked examples and G4 checks the effect.

## Consequences

- `_bins.py` sorts by value; the time index enters only through `_prefs.py` and SMOTE
  neighbour choice.
- The G1 bin property changes: bumps partition the value-sorted order; there is no
  "alternating rarity" in time.
- `Blueprint/docs/REPLICATION.md` §5 step 3 is inverted accordingly.
