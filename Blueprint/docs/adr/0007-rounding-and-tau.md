# ADR-0007 — (Superseded 2026-09-22 — see amendment and ADR-0013) Half-up rounding for target counts; `τ_i = i/k` for neighbour weighting

**Status:** **Superseded 2026-09-22**. Part A by the amendment below; part B by ADR-0013 · **Date:** 2026-09-03

Two small unspecified constants that would each silently change output.

## A. Rounding of fractional target counts

**Context.** `tgtNr = N / n_bins` is rarely an integer. The paper does not say how to
round it. R's `sample(x, size)` truncates a non-integer `size`.

**Decision.** `round_half_up(x) = floor(x + 0.5)`, applied once per bin target.

**Why not `np.rint`.** NumPy's `rint` (and Python's built-in `round`) implement
banker's rounding: `rint(0.5) == 0`, `rint(1.5) == 2`, `rint(2.5) == 2`. On a dataset
where `N / n_bins` lands on an exact half — entirely possible with small integer bin
counts — the two conventions disagree by one case per bin, and the disagreement is
data-dependent rather than systematic, which makes it very hard to spot later. Pinning
half-up removes a whole class of "off by a few rows" replication mismatches.

**Implementation.** One shared helper, used everywhere a count is derived. Never
`round()`, never `np.rint`, never `astype(int)` on a positive float without an explicit
`+ 0.5`.

## B. Temporal weight `τ` in Algorithm 13

**Context.** Algorithm 13 selects the neighbour maximising `φ_i · τ_i`, where `τ`
weights the `k` nearest neighbours by recency. The paper does not give `τ`'s
normalisation. Two readings are available: `τ_i = i/k` and `τ_i = (i−1)/(k−1)`.

**Decision.** `τ_i = i / k`, for the `i`-th nearest neighbour ordered **nearest to
farthest**, `i ∈ {1..k}`.

**Why.** `(i−1)/(k−1)` assigns `τ_1 = 0`, so the nearest neighbour has weight zero and
can never be selected regardless of its relevance. That directly contradicts §3.2 of the
paper, whose stated motivation is to *bias* selection, not to exclude a candidate
outright. `i/k` biases without excluding, and keeps `τ ∈ (0, 1]`.

**Tie-breaking.** Ties in `argmax(φ·τ)` — and ties in the "most recent" rule of
Algorithm 9 — resolve to the **smallest original index**. Without a stated rule,
`np.argmax` behaviour would be an implementation detail leaking into results, and runs
with the same `random_state` could differ across NumPy versions.

## Consequences

Both choices are marked in the code with a comment pointing back to this ADR, because
both are places where a future contributor would reasonably "clean up" the code into
something subtly different. Gate G3 tests each with a worked example whose arithmetic is
written out in the test file.

## Amendment — v0.8.0 (2026-09-22)

**Part A superseded.** R truncates: `sample(x, size)` floors a non-integer `size`. The
library follows R (ADR-0011):
- **Sample sizes:** `trunc`.
- **Per-bump ratios** in under/over `"balance"`: rounded to 5 decimals first.
- **smote bump size** `N / #bumps`: R's `round`, which is round-half-to-**even**.

Half-up rounding is therefore removed. `np.rint` is now *correct* for the smote bump size
alone, and it must go through one shared helper, never an inline call.

**Part B superseded by ADR-0013.** Algorithm 13's `τ` is recency: the neighbour's time
index over the largest time index among the `k` candidates. It is not nearness rank, and
the "nearest neighbour gets weight 0" argument above no longer applies.

**Evidence:** [AUDIT] (ADR-0011).
