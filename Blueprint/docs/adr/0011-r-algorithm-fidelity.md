# ADR-0011 — Fidelity to the original R algorithms is the default

**Status:** Accepted · **Date:** 2026-09-22 · **Amends:** ADR-0004, ADR-0006, ADR-0007;
frames ADR-0012, ADR-0013

## Context

The paper's pseudocode and the authors' R code (`Exps.R` resamplers, `uba` relevance and
utility) disagree in several places. v0.7.0 resolved each ambiguity by reading the paper,
and so drifted from the R behaviour that produced the published and canonical numbers. The
project owner stated the goal explicitly (2026-09-22): the library must be **as close as
possible to the original R algorithms**. The canonical R replication
(`jpmsilva1/ts-resampling-replication`) is the instrument used to tune and verify the spec,
not something replicated for its own sake.

## Decision

1. **R behaviour is the default** wherever the paper and the R code differ, including
   behaviour that looks like a bug: the SMOTE index-ordering quirk, last-predictor target
   weights, truncating sample sizes.
2. A paper-intent reading may exist **only as an opt-in flag** (`r_quirks=False`, ADR-0013),
   never as the default.
3. **Robustness deviations** are allowed only where R produces *no result* (it raises or
   yields NaN). Each is documented in SPEC §4 and tested:
   - The no-bump rule (return input unchanged + warning; matches the canonical harness's
     D3 patch).
   - Zero-range SMOTE target weights.
   - Too few positive-probability cases in a no-replacement draw.
4. **Harness choices are not library behaviour.** `complete.cases` on DS12/13, the SVM
   training-row cap, `auto.arima(method="CSS")` and the learner implementations configure
   how gates read the oracle (`Blueprint/docs/REPLICATION.md` §4). They never shape `src/`.
5. **Order of output rows** may differ from R (SPEC §2.2 ordering guarantee), because order
   changes neither which cases are present nor how they were drawn.

## Evidence standard

Conventions are tagged **[ORACLE]** (verified against recorded canonical numbers) or
**[AUDIT]** (established by the 2026-09-22 audit of a colleague's Python port of the R code,
the maintainer's private audit report, not in this repo). An [AUDIT] convention must be confirmed by a gate before its
node is done: G3 worked examples for structure, G4 for effect. Where the canonical output
cannot observe a convention (e.g. which neighbour was chosen), G4 is the check.

## Consequences

- SPEC v0.8.0 §4.1–4.7 are rewritten to the R conventions (SPEC §0.1 #12–#22).
- The kill criterion (SPEC §7) now names the [AUDIT] conventions, not ADR-0004's literal
  reading, as the first thing to re-check when G4 fails.
- Users wanting "the paper's algorithm as written" get `r_quirks=False` and a docs section
  listing every deviation, with its ADR.

## Amendment — 2026-10-26: tied control points are not a robustness deviation

The list above named "degenerate φ control points" as a case where R produces no result. That
was wrong. Run directly, R `uba` 0.7.7 (and `UBL`) returns φ for tied knots without error: it
keeps them, and a value takes the segment of the last knot ≤ it. v0.8.0 instead returned φ ≡ 0
with a warning, which disagreed with R whenever `min(y) == med` (on 82 of 2,230 sampled real
series from the imbalance-ts-hub catalog, φ ≡ 0 where R found rare cases) and for `IQR == 0`
with an outlier (R scores every value rare). `phi()` now follows R in these cases; strictly
increasing knots are unchanged, bit for bit. Evidence: recorded `uba` output in
`tests/test_algorithms.py::TIED_KNOTS_R`.
