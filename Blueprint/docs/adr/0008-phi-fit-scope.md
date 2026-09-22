# ADR-0008 — φ is fit only on the `y` passed to `fit_resample`

**Status:** Accepted · **Date:** 2026-09-03

## Context

The relevance function's control points are derived from quantiles of `y`. If those
quantiles are computed over the whole dataset and the resampler is then applied inside a
cross-validation fold, **test-set information leaks into the training transformation**.
The result is optimistic and irreproducible, and — because φ only shifts *which* cases are
called rare — it fails quietly rather than loudly.

Spec v0.6.0 did not state a fit scope at all.

## Decision

φ is fit on exactly the `y` array handed to `fit_resample`, and nothing else. There is no
`fit()` that persists control points across calls, and there is no way to pass a
pre-fitted φ except by supplying `relevance=` explicitly — which is the documented escape
hatch for users who deliberately want a fixed relevance across folds.

`TimeSeriesResampler` is therefore stateless between calls: two invocations on different
data share nothing.

## Evidence this matches the reference

The recovered oracle (`Blueprint/tests/fixtures/phi_oracle.json`) stores control points **per Monte
Carlo iteration**, and for most datasets they differ between iterations — e.g. DS01
iteration 1 is `(−0.123043, 0.003750, 0.125000)` and iteration 2 is
`(−0.123333, 0.003297, 0.124167)`. The reference implementation re-fits φ on each
training split. Datasets whose control points are constant across iterations (DS05, DS06,
DS11, DS13) are heavily tied series where every split yields the same quartiles — that is
a property of the data, not evidence of a global fit.

## Consequences

- Using the resampler inside a scikit-learn `Pipeline` with `TimeSeriesSplit` is correct
  by construction; no extra care is needed from the user.
- The replication suite must re-fit φ per split, and gate G0 compares against the
  per-iteration oracle rather than a single global fit.
- Users who *want* a global φ must opt in explicitly via `relevance=`, and the docstring
  warns that doing so inside cross-validation leaks.
