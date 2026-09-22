# ADR-0014 — The utility surface `U(ŷ, y)` and the bump initialisation

**Status:** Accepted · **Date:** 2026-09-22 · **Closes:** task M0 (b) · **Extends:** ADR-0009

## Context

`precision_φ`/`recall_φ` (SPEC §4.7) need `u_i = U(ŷ_i, y_i)`. ADR-0009's amendment knew the
surface was exactly recoverable because a GPL-derived reference reproduced the recorded values,
but it did not transcribe that surface. M0 had to derive `U` cleanly from Ribeiro (2011) and
verify it against the recorded R output.

**Provenance.** Sources: the thesis (R. P. Ribeiro, *Utility-based Regression*, PhD thesis, University of Porto, 2011; not redistributed here); the recorded CSVs
under `Results (Clean)/Results Data/` (`raw_iterations_by_dataset_v2`, `raw_predictions_by_dataset`,
`raw_sera_by_dataset`); and the harness script `Results (Clean)/Eval_Metrics Code/SERA Metric/sera_metric.py`
(project-authored, PROVENANCE §2). Oracle hygiene: MANIFEST.csv has no `provenance` column; its
`note` column marks `oracles/metrics/prec_rec_f1` "VALID oracle - R uba::util(), correct phi" and
`sera_pchip` CONTAMINATED. The M0 session opened **no** quarantined path (PROVENANCE §1). It also
did not open the port-audit report (the maintainer's private audit report, not in this repo), which came out of a tainted session.

The probe is `Blueprint/replication/probes/m0_utility.py` (`sweep`, `sera`, `fixture`).

## Decision — the surface (Confirmed unless marked)

The thesis defines `U` in §3.3–3.4; §4.3 uses it. With the extremes φ of SPEC §4.1, the control
points are `(x₁, φ₁), (med, 0), (x₃, φ₃)`, all with derivative 0.

1. **Loss.** `L(ŷ, y) = |ŷ − y|` (thesis p. 88, "e.g. absolute deviation").
2. **Bumps** (Def 3.10–3.11, Alg 3.2, p. 100). Each bump is `⟨b⁻ᵢ, b*ᵢ⟩`: left edge and maximum.
   Bump `i` covers `[b⁻ᵢ, b⁻ᵢ₊₁)`, with `b⁻` after the last bump `= +∞`. Algorithm 3.2 runs over
   the critical points (derivative 0). Constant-φ runs are averaged.
   **Initialisation — the new ambiguity.** The printed initialisation (`r ← 1`, `inBump ← true`)
   never assigns `b⁻₁`, yet Example 3.6 yields `⟨1.1, 5.5⟩`. The oracle pins it:
   - if φ's first change across the control points is a **decrease**, bump 1 is open from
     `b⁻₁ = −∞`, and there is no `⟨−∞, −∞⟩` bump;
   - otherwise bumps start closed. The leading constant-φ run is averaged into `b⁻₁`, and bump 1
     is preceded by `⟨−∞, −∞⟩`. This also reproduces Example 3.6.

   For the shapes that occur:

   | φ at control points | Partition |
   |---|---|
   | (1, 0, 1) two-sided | `⟨−∞, x₁⟩, ⟨med, x₃⟩` |
   | (0, 0, 1) one-sided | `⟨−∞, −∞⟩, ⟨(x₁+med)/2, x₃⟩` |
   | (1, 0, 0) one-sided | `⟨−∞, x₁⟩, ⟨(med+x₃)/2, +∞⟩` — **does not match R, see Residual** |

3. **Maximum admissible loss** (Def 3.12): `b^Δᵢ = 2·min(|b⁻ᵢ − b*ᵢ|, |b*ᵢ − b⁻ᵢ₊₁|)`. A
   non-finite value, including `|−∞ − (−∞)|`, takes the adjacent bump's value (p. 85).
4. **Thresholds**, with `γ(y)` = the bump containing y:
   - `L̇_B = min(b^Δ_γ, |y − b⁻_γ|)` if `ŷ < y`, else `min(b^Δ_γ, |y − b⁻_{γ+1}|)` (Defs 3.14–3.15).
   - `L̇_C = min(b^Δ_γ, |y − b*_{γ−1}|)` if `ŷ < y`, else `min(b^Δ_γ, |y − b*_{γ+1}|)` (Def 3.18).
   - A missing neighbour's `b*` is `∓∞`.
5. **Bounded loss** (Def 3.14): `Γ = L/L̇` if `L < L̇`, else 1.
6. **Utility** (Def 3.20, Eq 3.19): `U = φ(y)(1 − Γ_B) − φ_p(ŷ, y)·Γ_C` with
   `φ_p = (1 − p)φ(ŷ) + p·φ(y)`, `p = 0.5`. At p = 0.5, which of y and ŷ p weights is immaterial.
7. **Metric conventions** are unchanged from SPEC §4.7 / ADR-0009 (`≥ t_E`, `|1+u|`, `1e-5`).
   Note that the thesis's own Def 4.3 differs from them: it uses calibrated `ẑ` and a
   `2 − p(1 − φ)` denominator. The recorded R output follows SPEC §4.7, not Def 4.3.

**Inferred, not separately probed:** a y exactly on a bump edge belongs to the upper bump, and
`ŷ = y` takes the `ŷ ≥ y` branch (the thesis's case split). No recorded split was found where
flipping either would change a value at 1e-6.

## Evidence

prec/rec/F1 were checked against `raw_iterations_by_dataset_v2/` on every recorded split: 24
datasets × 52 workflows × 50 iterations, using R's recorded per-split control points. Pass means
all three measures are within 1e-6 absolute.

| φ shape | Datasets | Splits | Pass | Worst abs error |
|---|---|---|---|---|
| (1, 0, 1) | DS01–DS09, DS11–DS20 (DS19 in part), DS21–DS23 (part) | 51,532 | 51,532 | 6.8e-15 |
| (0, 0, 1) | DS10, DS21–DS24 (part) | 10,556 | 10,556 | 8.7e-15 |
| (1, 0, 0) | DS19 iterations 25–30 | 312 | 0 | 8.3e-01 |
| **Total** | | **62,400** | **62,088** | |

Per-dataset worst errors are in the probe output. This covers the §3.4 regimes: DS05 3.5 %, DS01
9.9 % and DS09 21.1 %, all 2,600/2,600, across all six model families. It also covers the NA
datasets DS12/13.

The recorded `prec = 1e-05` rows (6,371 rows, 306 dataset/workflow pairs) are reproduced too. They
test `U` only through recall: with no `φ(ŷ) ≥ t_E`, precision is the floor whatever `U` is.

**Rejected candidates** (the worst-case error varies from split to split, so each is a wrong
convention, not a missing normalisation):

| Candidate | Where tested | Result |
|---|---|---|
| Printed init with `b⁻₁ = y₁` (two-sided → `⟨−∞,−∞⟩, ⟨x₁,x₁⟩, ⟨med,x₃⟩`) | DS01, DS05 it 1–3 | rec off 0.15–0.18 |
| Init `r ← 0, inBump ← false` everywhere (drops the low bump) | DS01, DS05 it 1–3 | off 2.4e-4 to 3.0e-2 |
| One-sided (0,0,1) left edge `b⁻₁ = x₁` | DS10, 2,600 splits | all fail, worst 0.12 |
| One-sided (0,0,1) left edge `b⁻₁ = med` | DS10, 2,600 splits | all fail, worst 0.21 |

**SERA** (for the fixture; the finding itself is ADR-0009 amendment 2). `sera(step=0.01)` matched
`raw_sera_by_dataset/` on two-sided splits only: 10,452/10,452 on DS01, DS04, DS05, DS09 and DS21
(the two-sided subset), worst relative error 1.3e-15. This needed one numerical detail that SPEC
§4.1's recipe `spline(np.clip(y, x_low, x_high))` does not guarantee:

- scipy's `CubicHermiteSpline`, evaluated on an array, can return `1 − 4e-16` at the upper knot;
- it can return `−2.2e-16` near the median (DS01 iteration 21, every workflow).

Those values drop cases out of the `t = 1` and `t = 0` grid points. 1,138 DS01 splits and 1,872
DS09 splits were off by up to 0.5 % until φ was made to return the **exact endpoint value at and
beyond the knots** and was **clipped to [0, 1]**, as the harness does. prec/rec/F1 are unaffected
at 1e-6, since `t_E = 0.9` is far from both edges.

## Residual — DS19, φ = (1, 0, 0) (Unconfirmed; documented, not fitted)

The mirrored one-sided shape occurs only in DS19 iterations 25–30 (312 splits). No reading tried
reproduces R there:

- The rule above gives worst error 0.83. R's recall is near its floor (e.g. 0.036 recorded
  against 0.867 computed), so R gives these low extremes almost no benefit.
- The printed initialisation (`b⁻₁ = y₁`, so `b^Δ = 0` and `u = −φ_p` on the low side) matches
  199/312. It misses exactly the splits where some rare y has a prediction in the far low tail,
  and there R awards more utility than `−φ_p`.
- A brute-force search over partitions `⟨b⁻₁, b*₁⟩, ⟨b⁻₂, b*₂⟩`, with every value drawn from
  `{−∞, x₁, med, (x₁+med)/2, (med+x₃)/2, x₃, +∞}`, with and without a leading `⟨−∞,−∞⟩` bump:
  best worst-case error 0.053 on a 40-split sample.

So the gap is not a bump-placement question within the thesis's `U`. The ADR-0009 reference
check (45 cases: DS01, DS04, DS05, DS09, DS10) never covered this shape either.

**Consequence.** `_utility.py` implements the rule above for every shape. On a (1, 0, 0) φ, the
docstring of `precision_phi`/`recall_phi`/`f1_phi` must say that agreement with R has not been
established for that shape. G0b's fixture excludes it. Next probe, if anyone wants to close it:
isolate single-case contributions on DS19 splits with exactly one rare y in the far tail.

## Acceptance (REPLICATION §3.4)

- `precision_phi`, `recall_phi`, `f1_phi` within 1e-6 absolute on DS05, DS01 and DS09 (and on all
  other datasets except the residual), across six model families: **met.**
- `sera(step=0.01)` within 1e-6 relative on two-sided splits: **met.**
- `Blueprint/tests/fixtures/metric_oracle.json` frozen: 8 cases, 49.4 KB. The cases are DS10 (4.8 %,
  one-sided, prec/rec/F1 only), DS01 (9.9 %), DS04 (13.3 %) and DS09 (21.1 %), over `lm`, `svm`,
  `rf` and `rpart` workflows. Values are the recorded ones, and each was asserted against the
  probe at build time. DS05 is omitted from the fixture only because one split is about 170 KB;
  it passes in the probe.

## Consequences

- Node 8 is unblocked. `_utility.py` implements items 1–6. Its tests take expected values from
  `metric_oracle.json`, never from this probe.
- The metrics' φ path must return exact endpoint φ and clip to [0, 1] (SERA section above). Node
  2a's `_relevance.py` is the shared φ; it should do this, since G0b's SERA check depends on it.
- DS19 (1, 0, 0) stays a documented residual (REPLICATION §3.4, "documented as different").
