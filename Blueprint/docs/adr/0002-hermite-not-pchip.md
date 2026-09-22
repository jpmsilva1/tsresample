# ADR-0002 — φ interpolates with CubicHermiteSpline (dydx = 0), not PchipInterpolator

**Status:** Accepted · **Date:** 2026-09-03 · **Supersedes:** spec v0.6.0 §4.A

## Context

Spec v0.6.0 said to build φ with `scipy.interpolate.PchipInterpolator` over the three
control points `(lo, 1), (med, 0), (hi, 1)`. PCHIP is the natural reading of "monotone
piecewise cubic Hermite", and the reference implementation does ship a `pchip.c`, so the
choice looked safe.

It is not the same function. `PchipInterpolator` *derives* the knot slopes from the data
using the Fritsch–Carlson rule; at an interior extremum it produces slope 0, but at the
two **outer** knots it uses a one-sided three-point formula that yields a **nonzero**
slope. The reference stores the slopes explicitly instead.

## Decision

φ is a cubic Hermite spline with **explicitly supplied zero derivatives at all three
knots**:

```python
CubicHermiteSpline(x=[lo, med, hi], y=[1.0, 0.0, 1.0], dydx=[0.0, 0.0, 0.0])
```

Outside `[lo, hi]`, φ ≡ 1 by extension — never by extrapolating the cubic.

## Evidence

**1. Direct.** The recovered control-point files carry a derivative column, and it is zero
at every knot of every dataset and every iteration:

```
"iteration","point_index","ctrl_x","ctrl_phi","ctrl_deriv"
1,1,-0.123043,1,0
1,2,0.00374999999999998,0,0
1,3,0.125,1,0
```

`ctrl_deriv = 0` at `point_index` 1 and 3 is exactly what PCHIP would *not* produce.

**2. Consequential.** Reproducing the paper's Table 1 `%Rare` column (share of cases with
φ > 0.9) across all 20 datasets, using the oracle control points and varying only the
interpolant:

| Interpolant | mean absolute error vs. Table 1 |
|---|---|
| Cubic Hermite, dydx = 0 | **0.86 pp** |
| PCHIP | 2.94 pp |

Exact agreement under Hermite on DS05 (3.5 / 3.5), DS06 (4.8 / 4.8), DS07 (12.5 / 12.5),
DS11 (13.3 / 13.3); DS01 lands at 10.0 vs. the paper's 9.9. PCHIP is wrong on every one.

## Consequences

- With zero slopes at both ends of each half, φ on `[med, hi]` is the smoothstep
  `3t² − 2t³`. The φ > 0.9 boundary therefore sits at `med + 0.7286·(hi − med)`, a fact
  worth knowing when reading bin boundaries.
- Both halves remain monotone, so the property tests in SPEC §5 still hold.
- The residual 0.86 pp is attributed to control points being fitted per training split
  rather than on the full series; it is not treated as an unexplained discrepancy, but
  gate G0 pins the tolerance so it cannot drift.

## Amendment — v0.8.0 (2026-09-22)

Hermite with `dydx = 0` stands. Two clarifications:
- The endpoint φ values are 1 **or 0** (ADR-0001 amendment).
- Outside the endpoints φ is constant at the endpoint's value, which is what linear
  extrapolation with zero endpoint slope gives. This was confirmed on a reference
  implementation of the R behaviour.
