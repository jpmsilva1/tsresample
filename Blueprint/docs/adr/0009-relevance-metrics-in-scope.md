# ADR-0009 — Relevance-based evaluation metrics are in scope

**Status:** Accepted · **Date:** 2026-09-03 · **Reverses:** spec v0.6.0 "no metrics
module ships"; supersedes the v0.6.0 kit's `fact-check-request.md` Items 2–3 ("out of
scope")

## Context

Spec v0.6.0 excluded metrics on the grounds that resampling and evaluation are separable
concerns. That is true architecturally and false in practice: a user who resamples an
imbalanced time series and then evaluates with RMSE has learned nothing, because RMSE is
dominated by the ~90 % of normal cases the resampling was designed to de-emphasise. The
paper itself reports **precision_φ, recall_φ and F1_φ**, and the successor line
(Ribeiro & Moniz 2020) reports **SERA**. Shipping the resampler without them hands users
a tool whose effect they cannot measure.

## Decision

A `tsresample.metrics` module ships, with four public functions, all relevance-based:

```python
precision_phi(y_true, y_pred, *, relevance="auto", rel_threshold=0.9, ...) -> float
recall_phi   (y_true, y_pred, *, relevance="auto", rel_threshold=0.9, ...) -> float
f1_phi       (y_true, y_pred, *, relevance="auto", rel_threshold=0.9, beta=1.0, ...) -> float
sera         (y_true, y_pred, *, relevance="auto", return_curve=False, ...) -> float
```

Non-relevance-based metrics (plain RMSE, MAE, R²) are **not** provided. `sklearn.metrics`
already has them and there is no value in a second copy. The module's reason to exist is
precisely the relevance weighting.

`relevance` accepts the same three forms as `TimeSeriesResampler`: `"auto"` (fit φ from
`y_true` per ADR-0001/0002/0008), an array, or a callable. φ is shared code — `_relevance.py`
— so the metrics and the resampler can never disagree about what "rare" means. That
shared definition is the main architectural argument for shipping them together rather
than as a second package.

## Formulas

**Utility-based precision / recall** (Torgo & Ribeiro 2009; Ribeiro 2011 §4):

```
recall_φ    = Σ_{i : φ(y_i)  ≥ t_E} |1 + u_i| / Σ_{i : φ(y_i)  ≥ t_E} |1 + φ(y_i)|
precision_φ = Σ_{i : φ(ŷ_i) ≥ t_E} |1 + u_i| / Σ_{i : φ(ŷ_i) ≥ t_E} |1 + φ(ŷ_i)|
F_β,φ       = (1 + β²) · precision_φ · recall_φ / (β² · precision_φ + recall_φ)
```

where `u_i = U(ŷ_i, y_i)` is the utility of the prediction under the relevance function
and a loss surface.

**SERA** (Squared Error-Relevance Area; Ribeiro & Moniz 2020):

```
SER_t = Σ_{i : φ(y_i) ≥ t} (y_i − ŷ_i)²          for t ∈ [0, 1]
SERA  = ∫₀¹ SER_t dt
```

`SER_t` is a step function of `t`. The integral is computed on a **uniform grid**
(`t ∈ {0, step, …, 1}`, `step = 0.001` default, trapezoidal rule) — **not** exactly over
the `φ(y_i)` breakpoints, which was the original plan here and is now known to be wrong
(see Verification status below). `return_curve=True` yields `(t, SER_t)` for plotting,
which is the diagnostic users actually want.

## Verification status — read this before implementing

The two formula families are **not** at the same confidence level, and the implementer
must not treat them as if they were.

| Component | Status | Basis |
|---|---|---|
| φ itself | **Confirmed** | ADR-0001, ADR-0002; two independent derivations |
| SERA integrand and shape | **Confirmed** | Ribeiro & Moniz 2020 |
| SERA integration family | **Confirmed** (2026-09-22) | grid-trapezoidal, not exact breakpoints — see below |
| SERA exact numeric match | **Confirmed** (2026-09-22) | recorded values reproduced to 1e-15 with the harness's `step = 0.01`; see amendment 2 |
| Metric conventions (`≥ t_E`, empty → `1e-5`, `|1+u|`, `p = 0.5`) | **Confirmed** (2026-09-22) | [ORACLE], see amendment |
| `u_i` utility surface | **Confirmed** (2026-09-22), except φ = (1,0,0) | derived cleanly from Ribeiro (2011) §3.3–3.4; 62,088/62,400 recorded splits within 1e-6 — **ADR-0014** |

**The original probe below (first two rows of the table, ratios 1.28–1.57) was invalid.**
It compared against `Blueprint/replication/oracles/metrics/sera_pchip/DS05.csv`, which
`Blueprint/replication/MANIFEST.csv` already flags as `CONTAMINATED - computed with
PchipInterpolator phi` — a pre-v0.7.0 recomputation, not the recorded R output. The actual
oracle, per `Blueprint/docs/REPLICATION.md` §1, is
`Results (Clean)/Results Data/raw_sera_by_dataset/DS*.csv` under
`TSRESAMPLE_REPLICATION_ROOT`. None of the five candidates originally listed here (`≥` vs
`>`, split mismatch, normalisation, mean-vs-sum, scale) was the actual cause — the lesson
for future M0-style probes is to check `MANIFEST.csv`'s provenance column before trusting
a file that merely has the right name and shape.

Re-run against the correct oracle, with φ = `CubicHermiteSpline(dydx=0)` (the confirmed
ADR-0001/0002 formula, not PCHIP) and grid-trapezoidal integration:

| Dataset / workflow | %Rare | iteration | computed | recorded | ratio |
|---|---|---|---|---|---|
| DS05 / `mc.lm_OVERB` | 3.5% | 1 | 1.520267 | 1.521788 | 0.9990 |
| DS05 / `mc.lm_OVERB` | 3.5% | 2 | 1.473369 | 1.474985 | 0.9989 |
| DS05 / `mc.lm_OVERB` | 3.5% | 3 | 1.394431 | 1.395699 | 0.9991 |
| DS01 / `mc.lm_OVERB` | 9.9% | 1 | 0.595811 | 0.596132 | 0.9995 |

All four are within 0.11% at `step = 0.001`. This does **not** shrink toward zero as
`step` is tightened — 0.0001 gives 0.14%, 0.00001 gives 0.15% — so the residual is *not*
pure grid-quantization error on our side; something else, worth ~0.15%, is still
unexplained *at the time of writing*. **Explained, see amendment 2.** A boundary-extrapolation hypothesis (φ clipped to 1 outside the control points vs. extrapolated) was checked on 2026-09-22 and **refuted**: the reference relevance function extrapolates linearly with zero endpoint slope, i.e. constantly at the endpoint φ — identical to clipping on DS05 (φ = 1 at both endpoints).

Exact-breakpoint integration (the ADR's original plan) was also re-tested against the
correct oracle and is unambiguously wrong regardless of this residual: ratio 1.22 on DS05
iteration 1, an order of magnitude off. `Blueprint/docs/SPEC.md` §4.7 has been corrected accordingly
(grid-trapezoidal, `step = 0.001`).

**This resolves the SERA integration *family* — the M0 task is not fully closed.** The
convention (φ = CubicHermiteSpline(dydx=0), `≥`, grid-trapezoidal) is confirmed; the
sub-0.2% residual against `1e-6` relative (`Blueprint/ops/QUALITY_GATES.md` G0b) is not. The residual's source was found later the same day (amendment 2). `precision_φ`/`recall_φ`/`f1_φ` are unaffected by any of
this and still block separately on the `u_i` utility surface, which is unrecovered.
`Blueprint/tests/fixtures/metric_oracle.json` should not be frozen yet: §3.4 of
`Blueprint/docs/REPLICATION.md` requires prec/rec/F1 matches too, and those are still open.

The `u_i` utility surface requires `uba::loss.control(y)`'s parameterisation, which has
not been recovered. **It must be reverse-engineered from the oracle the same way φ was**
— never guessed. The full procedure, oracle file paths and acceptance tolerance are
task **M0** in `Blueprint/ops/TASK_GRAPH.md`, and it **blocks** the metrics module the way the
IRonPy question used to block `_relevance.py`.

## Consequences

- `Blueprint/docs/REPLICATION.md` gate G4 gets much stronger: with metrics in-library, the
  replication compares our `precφ`/`recφ`/`F1φ`/`sera` against 20 datasets × workflows ×
  50 iterations of recorded R output, rather than comparing resampled row counts.
- The GPL quarantine widens. `external_repos/ImbalanceMetrics` (GPL-3) implements SERA and
  is now *topically* relevant, which makes it more tempting to open and therefore more
  important to keep closed. It is on the quarantine list in `Blueprint/docs/PROVENANCE.md`.
- Ship order: metrics land **after** the resampler grid (SPEC §7 roadmap), because M0 is
  research and the resampler is not blocked on it.

## Amendment — v0.8.0 (2026-09-22): metric conventions

- **Threshold:** `φ ≥ t_E`, case by case (not `>`; not the bump rule of ADR-0012).
- **Empty selection:** if no case clears `t_E`, precision (or recall) is **`1e-5`**. It is
  not `0.0`, not NaN, and there is no warning. Recorded canonical rows show `prec = 1e-05`,
  `F1 = 1.99996e-05`. `F_β = 0` only if a measure is exactly 0.
- **Sums** use `|1 + u_i|` and `|1 + φ|`; the utility uses `p = 0.5`.

**Evidence [ORACLE].** A reference implementation with these conventions reproduced the
recorded canonical prec/rec/F1 to **1e-15** on 45 cases: DS01, DS04, DS05, DS09 and DS10 ×
{`lm_OVERB`, `svm_baseline`, `rf_SMOTET`, `rpart_UNDERTPhi`} × 3 iterations.

**What this means for M0.** The utility surface is exactly recoverable. That reference was
GPL-derived (PROVENANCE §5), so its internals are **not** transcribed. M0 derives `U` from
Ribeiro (2011) §4 and verifies it on the oracle against G0b's tolerance
(`Blueprint/ops/QUALITY_GATES.md` §1). Agreement near machine precision is known to be reachable. The SERA residual is resolved separately (amendment 2).

## Amendment 2 — v0.8.0 (2026-09-22): the SERA residual is explained

The recorded values come from the harness script
`Results (Clean)/Eval_Metrics Code/SERA Metric/sera_metric.py`, not from R. It uses
`step = 0.01` and forces φ = 1 at and beyond **both** endpoints, including a one-sided
φ = 0 endpoint. Recomputing with those settings matches every recorded SERA to **1e-15**
(45 cases: DS01, DS05, DS09, DS10, DS21 × {`mc.lm`, `mc.lm_OVERB`, `mc.rf_SMOTET`} × 3
iterations).

So the 0.05–0.15 % gap, and its variation by dataset, were the coarser 0.01 grid on the
harness side. It was not seed noise: the deterministic `lm` baseline showed the same gap.
Nor was it grid construction: `arange`, `i·step` and `linspace` are identical.

**Decisions.**
- SERA is a harness metric, not one of the paper's R algorithms, so ADR-0011 does not
  bind it.
- The library default stays **`step = 0.001`** (the SERA reference default; also MetaIR's)
  with §4.1's φ, including a one-sided φ = 0.
- **G0b** checks `sera(..., step=0.01)` against `raw_sera_by_dataset/` on **two-sided splits
  only**, within the G0b tolerance.
- One-sided splits (both-endpoint φ ≠ 1: DS10, some DS19, DS21–24) are excluded. There the
  recorded values carry the harness's φ bug.
- Recommended outside this repo: fix the harness script's one-sided φ and regenerate
  `raw_sera_by_dataset/`. Until then, treat SERA on one-sided datasets in the canonical
  report with caution.
