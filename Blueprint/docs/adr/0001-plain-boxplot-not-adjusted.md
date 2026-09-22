# ADR-0001 — φ uses the plain Tukey boxplot, not the adjusted boxplot

**Status:** Accepted, **amended 2026-09-22** (see end) · **Date:** 2026-09-03 · **Supersedes:** spec v0.6.0 §4.A

## Context

Spec v0.6.0 specified that the relevance function's control points sit at the fences of
the **adjusted boxplot** of Hubert & Vandervieren (2008), i.e.
`Q₁ − 1.5·e^{a·MC}·IQR` and `Q₃ + 1.5·e^{b·MC}·IQR` with the medcouple `MC`. That would
have made `statsmodels` (for `stats.stattools.medcouple`) a hard runtime dependency.

The claim was inherited, not verified. It matters: the adjusted boxplot is strongly
asymmetric on skewed data, so the two conventions disagree about which cases are rare.

## Decision

φ's control points sit at the **plain Tukey fences**:

```
lo = Q₁ − 1.5·IQR      hi = Q₃ + 1.5·IQR      (quartiles = R quantile type 7)
```

No medcouple. No `statsmodels`. No `robustbase`.

## Evidence

Two independent lines, neither requiring GPL source to be read.

**1. Numerical.** Our own R experiment runs wrote the fitted control points to
`mc.*_phi_ctrl.csv` for all 20 datasets × 50 Monte Carlo iterations. Comparing the
per-dataset median oracle fences against both candidate conventions computed on the same
series:

| Dataset | oracle lo / hi | plain lo / hi | adjusted lo / hi |
|---|---|---|---|
| DS01 | −0.12417 / 0.12417 | −0.12708 / 0.12958 | −0.14692 / 0.10830 |
| DS03 | −0.20088 / 0.19714 | −0.21371 / 0.21355 | −0.18199 / 0.24238 |
| DS16 | −0.02609 / 0.02535 | −0.02566 / 0.02728 | −0.02189 / 0.03068 |
| DS18 | −0.02539 / 0.03024 | −0.03136 / 0.03302 | −0.02272 / 0.04253 |
| DS19 | −0.02661 / 0.02704 | −0.02657 / 0.02841 | −0.02095 / 0.03396 |
| DS20 | −0.02111 / 0.02581 | −0.02191 / 0.02342 | −0.02389 / 0.02110 |

The oracle is close to symmetric on every skewed dataset and tracks the plain fences.
The adjusted fences are visibly asymmetric in the opposite direction on DS01, DS16, DS18,
DS19. The residual plain-vs-oracle gap is expected and explained: the oracle is fitted per
Monte Carlo *training split* (~50 % of rows), the comparison column on the full series.

**2. Packaging.** `uba` 0.7.7 (`DESCRIPTION`, April 2017) declares no dependency on
`robustbase`, which is the only R package providing a medcouple implementation. An
adjusted boxplot was therefore not available to `phi.control` at all.

## Consequences

- `statsmodels` is removed from the dependency set; runtime deps are `numpy`, `scipy`,
  `scikit-learn`.
- Any downstream comparison against a *modern* UBL/SMOGN implementation may disagree,
  because later versions of that lineage did adopt the adjusted boxplot. Fidelity to the
  2017 paper is the stated goal, so this is the correct trade.
- `Blueprint/tests/fixtures/phi_oracle.json` pins the expected control points so the convention
  cannot silently regress.

## Amendment — v0.8.0 (2026-09-22): where the plain-boxplot control points sit

Plain-vs-adjusted stands. But the **Decision block above is superseded**. The control points
are not at the fences, and the quartiles are not type 7.

| | v0.7.0 (this ADR, above) | v0.8.0 |
|---|---|---|
| quartiles | R `quantile` type 7 | **Tukey hinges** (`fivenum`) |
| outer control points | the fences `Q₁ ∓ 1.5·IQR`, φ = 1 | the **whisker ends**: most extreme observations inside the fences, φ = 1 |
| side with no outliers | not handled (φ = 1 at the fence regardless) | `(min or max, φ = 0)` |
| outside the endpoints | φ ≡ 1 | constant at the endpoint's φ |

**Evidence [ORACLE].**
- 90 training splits (18 NA-free datasets × 5 iterations) reconstructed by matching the
  recorded test targets to the series. The new convention reproduces R's recorded
  control points in **85/90**; the fences in 4/54 on a first sample.
- The 5 misses are all DS05, whose training values lie exactly on the ±0.08 fences. The
  outcome there depends on last-bit precision lost in the CSV export (open item,
  REPLICATION §2).
- φ = 0 endpoints appear in the canonical control points of every DS10 split and of DS19
  and DS21–24.
- `%Rare` vs paper Table 1 on the embedded full series: **0.17 pp MAE** (v0.7.0: 0.82 pp,
  with one dataset at 4.20 pp, beyond the 4.0 pp gate).

**Why the table above looked right.** The "oracle" column is a per-dataset *median* over
splits. Whisker ends sit just inside the fences, so medians landed near the fences, and
v0.7.0's G0 fixture was then generated from the fence formula itself (SPEC §0 #13), which
made the gate unable to fail.
