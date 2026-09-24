# ADR-0016 — `relevance` accepts φ's control points; the utility metrics require them

**Status:** Accepted · **Date:** 2026-09-24 · **Extends:** SPEC §2.2, §2.3, ADR-0009, ADR-0014

## Context

SPEC §2.3 gives the metrics the resampler's three `relevance` forms: `"auto"`, a per-case
φ array, or a callable. Two facts make the last two insufficient for
`precision_phi`/`recall_phi`/`f1_phi`:

- precision selects on and normalises by **φ(ŷ)**, and a per-case array holds φ(y) only;
- the utility surface `U` (ADR-0014) is built from the **bumps of φ's control points**,
  which neither an array nor an opaque callable exposes.

SPEC §4.7 also says the metrics' φ is the one fit on the **training** target (ADR-0008),
which `"auto"` (fit on `y_true`) cannot express. Gate G0b needs exactly that: R's recorded
per-split control points.

## Decision

1. `relevance` additionally accepts φ's **control points**: an array of shape `(3, 2)`
   `[(x, φ), …]` (low, median, high), as returned by `control_points(y)`. A trailing
   derivative column (shape `(3, 3)`, R's `phi.control` layout) is accepted and must be 0.
   2-D input is always read as control points, so it cannot be confused with a per-case φ
   array.
2. `tsresample.metrics.control_points(y)` is public: the way to fit φ on a training target
   and score a test set with it.
3. `precision_phi`, `recall_phi` and `f1_phi` accept `"auto"` or control points. A per-case
   array or a callable raises `ValueError` naming both accepted forms. `sera` accepts all
   four forms, since it needs φ(y) only.
4. `TimeSeriesResampler(relevance=…)` accepts control points too, for symmetry, so the same
   object serves both.

## Consequences

- The public surface grows by one function (`metrics.control_points`) and one accepted
  argument shape. SPEC §2.2 and §2.3 are updated.
- Recommended evaluation: `cp = control_points(y_train)`, resample with `relevance=cp`,
  score with `relevance=cp`. One φ, shared.
