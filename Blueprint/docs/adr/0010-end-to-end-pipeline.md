# ADR-0010 — The library covers the workflow end to end; what to borrow from `imbalance_eval`

**Status:** Accepted, **amended 2026-09-22** (imputation, threshold, iblr rationale; see end) · **Date:** 2026-09-03

## Context

Two public symbols (`embed`, `TimeSeriesResampler`) plus four metric functions is a
correct *library* and an incomplete *tool*. A user arriving with a CSV of a time series
still has to write the loading, the differencing, the imputation, the embedding, the
temporal split, the resampling, the fitting, the scoring and the summary table themselves
— and every one of those steps has a way to get it subtly wrong that the paper's own
pipeline gets right.

The user has already built and validated part of this: `imbalance_eval`
(`github.com/jpmsilva1/imbalance_eval`), a single-module library + CLI that loads CSVs,
KNN-imputes, time-delay-embeds, computes φ via `ImbalancedLearningRegression`, and emits
the paper's Table 1 (`N`, `n_normal`, `n_rare`, `IR`, `%Rare`). It has a test suite, a
manifest-driven batch mode, and a README documenting validation against the paper.

## Decision

`tsresample` covers the pipeline end to end, in three layers, each usable alone:

**Layer 1 — primitives** (already specified): `embed`, `TimeSeriesResampler`, `metrics`.

**Layer 2 — pipeline helpers** (`tsresample.pipeline`):

```python
load_series(path, *, target, date_col=None, diff=False, impute="knn") -> np.ndarray
imbalance_summary(y, *, rel_threshold=0.9, relevance="auto") -> dict
    # {N, n_normal, n_rare, IR, pct_rare} — the paper's Table 1, one row
temporal_split(X, y, *, train_size=0.5, test_size=0.25, n_reps=50, random_state=None)
    # Monte Carlo temporal splits: contiguous windows, train strictly before test
evaluate(estimator, X, y, *, strategies=..., metrics=..., splitter=...) -> DataFrame
    # the full grid × splits × metrics table, tidy long format
```

**Layer 3 — CLI** (`tsresample` console script): single-file and manifest batch modes,
mirroring `imbalance_eval`'s argument surface so existing invocations keep working.

### What is borrowed from `imbalance_eval`, and what is not

| Borrowed | How it changes |
|---|---|
| `time_delay_embedding` shape (`lag_k … lag_1, target`) | Confirms ADR-0005's mapping. Reimplemented on `embed()` with `horizon`. |
| `load_csv` with `date_col` sort-and-drop | Kept. Sorting by date before embedding is a real correctness step, not a convenience. |
| The `{N, n_normal, n_rare, IR, %Rare}` summary | Becomes `imbalance_summary`. |
| Manifest CSV batch mode (`id,name,granularity,path,target[,threshold,k]`) | Kept verbatim, including the required-column validation that names the missing columns. |
| The `diff` / `no-embed` / `xtrm-type` / `coef` flags | Kept as pipeline kwargs. |
| Its error-message discipline — `ValueError` naming the available columns, refusing a length-1 series instead of reporting `IR=0` | Adopted as a general standard for this library. |

| **Not** borrowed | Why |
|---|---|
| The `ImbalancedLearningRegression` dependency | `tsresample` owns φ so the resampler and metrics share one definition (SPEC §3), and iblr 0.0.2 is not numpy-2 compatible. (Its φ is *correct*: it matches R's control points 85/90 — see amendment.) |
| `_iblr_numpy_quantile_compat` | Dead with the dependency. Deleting a monkey-patch of `np.quantile` is a strict improvement. |
| pandas as a **required** dependency | `pandas` becomes an optional extra (`tsresample[io]`) used by Layer 2/3 only. Layers 1 stays numpy/scipy/sklearn so the primitives install anywhere. |
| `impute_dataframe` | **Bug (found 2026-09-22):** on a univariate series `KNNImputer` has no other features, so every gap gets the column mean. Replaced by lag-window kNN (amendment below). |

## Consequences

- `imbalance_eval`'s validated `%Rare` numbers become a **third** independent check on φ,
  alongside the R oracle and the paper's Table 1. Its test suite is a source of test cases.
- Layering is enforced by import direction: `pipeline` may import `metrics` and the
  primitives; nothing may import `pipeline`. A CI check asserts it, because this is the
  boundary that erodes first.
- `pandas` and the CLI are optional. `pip install tsresample` stays lightweight;
  `pip install tsresample[io]` gets the end-to-end path.
- The user's existing `imbalance_eval` invocations should keep working through the CLI
  compatibility surface. Where behaviour intentionally differs (strict threshold, φ
  source), `docs/MIGRATION.md` states it explicitly rather than letting numbers move
  silently.

## Amendment — v0.8.0 (2026-09-22)

**1. `impute="knn"` is lag-window kNN, not `KNNImputer` on the raw column.**
`impute_dataframe` runs `KNNImputer` on the numeric columns, but a series CSV has one
numeric column. With no features to measure distance on, every gap is filled with the
column mean:

| | gaps | `impute_dataframe` fills with | %Rare | lag-window kNN %Rare | paper |
|---|---|---|---|---|---|
| DS12 | 197 | −0.135 (the mean), all 197 | 14.65 | **10.99** | 11.0 |
| DS13 | 201 | −0.0884 (the mean), all 201 | 9.19 | 9.19 | 11.1 |

**The rule** (the paper applies `knnImputation` to the embedded frame; SPEC §2.4):
- Walk the series in time order.
- For each missing `y_t`, take its lag window `y_{t−9..t−1}`. Standardise with column
  means and SDs of the embedded frame.
- Find the `k = 10` nearest fully-observed embedded rows on the observed lags (Euclidean),
  and fill with their targets weighted `exp(−d)`.
- Values filled earlier count as observed for later gaps. That fills long runs and is
  deterministic.
- Leading gaps with no observed lag are dropped with a warning.

It reproduces DS12 exactly. DS13 does not reproduce under any tested variant (9.19
sequential, 9.71 non-sequential, 10.67 dropping gaps), so it stays an open item. That is
tolerable because imputation is harness-adjacent (ADR-0011) and G0/G4 run DS12/DS13
separately (REPLICATION §2, §4.2b). `impute="drop"` and `impute=None` (raise on NaN) are
also offered.

**2. `%Rare` threshold.** `imbalance_summary` counts `φ ≥ t` (as `imbalance_eval` does, and
as the metrics' event rule does, SPEC §4.7), not strict `>`. On all 18 NA-free datasets
the two give identical `%Rare`, so no data distinguishes them. `≥` removes a migration
difference.

**3. φ corroboration.** `imbalance_eval`'s φ (via iblr, black-box tested) reproduces R's
per-split control points in **85/90** splits, the same as SPEC v0.8.0 §4.1, and `%Rare` to
0.169 pp. It was right before the SPEC was; v0.7.0 had cited it while specifying fences.
