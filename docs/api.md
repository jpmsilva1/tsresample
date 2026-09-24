# API reference

Generated from the docstrings by `tests/test_docs.py` (`TSRESAMPLE_UPDATE_DOCS=1 pytest tests/test_docs.py`).

## `tsresample`

### `embed(series, k, *, horizon=1, exog=None)`

```text
Build a supervised matrix from a univariate series by time-delay embedding.

Row t of X is [y_t, y_{t-1}, ..., y_{t-k}] (k+1 lagged values, most recent
first). The paired target is y[t] = series[t + horizon].

Parameters
----------
series : array-like of shape (n,)
    Univariate series, ordered oldest to newest. Must be 1-D and finite.
k : int
    Number of *additional* lags beyond the current value, so each row holds
    k+1 columns. Must satisfy k >= 1.
horizon : int, default 1
    Forecast horizon. Must be >= 1.
exog : array-like of shape (n, m), optional
    Exogenous columns, aligned to `series` by position. Row t receives
    exog[t] (contemporaneous only -- no lagging is applied). Appended to the
    right of the lag block.

Returns
-------
X : ndarray of shape (n - k - horizon, k + 1 + m)
y : ndarray of shape (n - k - horizon,)

Raises
------
ValueError
    If `series` is not 1-D, contains NaN or inf, `k < 1`, `horizon < 1`,
    `exog` length does not match `series`, or the series is too short to
    produce at least one row.

Notes
-----
Mapping to the reference implementation: the paper's R helper
``create.data(ts, m)`` builds ``embed(ts, m)[, m:1]`` and treats the last
column as the target, giving ``m - 1`` predictors and target ``y_t``.
The equivalent call here is ``embed(series, k=m - 2, horizon=1)``: ``k + 1``
predictors, so ``k = m - 2``. The paper's ``create.data(ts, 10)`` is
``embed(series, k=8, horizon=1)``, whose targets are ``series[9:]``.
See Blueprint/docs/adr/0005-embed-convention.md (and its amendment).

Examples
--------
>>> X, y = embed([0.0, 1.0, 2.0, 3.0, 4.0], k=1)
>>> X
array([[1., 0.],
       [2., 1.],
       [3., 2.]])
>>> y
array([2., 3., 4.])
```

### `TimeSeriesResampler(strategy='smote', bias=None, *, rel_threshold=0.9, relevance='auto', k=5, o=None, u=None, r_quirks=True, random_state=None)`

```text
Resample an imbalanced time-series regression training set.

Implements the nine strategies of Moniz, Branco & Torgo (2017): random
undersampling, random oversampling and SMOTE, each with no bias, a temporal
bias, or a temporal+relevance bias (SPEC §4.6). Defaults reproduce the
original R code (ADR-0011).

Parameters
----------
strategy : {"under", "over", "smote"}, default "smote"
bias : {None, "temporal", "temporal+phi"}, default None
rel_threshold : float, default 0.9
    ``t_R``; bumps are rare by their mean relevance (SPEC §4.2).
relevance : "auto", array of shape (n,), or callable, default "auto"
    ``"auto"`` fits phi on ``y`` (SPEC §4.1). An array gives phi per case; a
    callable is called as ``f(y)``. Values must lie in [0, 1].
k : int, default 5
    SMOTE neighbour count; ignored otherwise.
o, u : float or None, default None
    R's ``C.perc``: ``None`` balances (SPEC §4.4); otherwise ``o`` scales
    rare bumps and ``u`` normal bumps.
r_quirks : bool, default True
    Reproduce R's SMOTE quirks (ADR-0013). ``False`` gives the paper reading.
random_state : int, RandomState or None, default None

Notes
-----
Rows of ``X`` must be in time order: position is the time index. Output rows
are in time order, synthetic cases right after their seed. Stateless: phi is
refit on every call (ADR-0008).

Examples
--------
>>> import numpy as np
>>> from tsresample import TimeSeriesResampler, embed
>>> series = np.random.RandomState(0).standard_t(3, size=300)  # heavy tails
>>> X, y = embed(series, k=4)
>>> X.shape
(295, 5)
>>> resampler = TimeSeriesResampler("smote", "temporal", random_state=0)
>>> X_res, y_res = resampler.fit_resample(X, y)
>>> X_res.shape
(293, 5)
```

#### `fit_resample(X, y)`

```text
Return the resampled ``(X, y)``; rows in time order (SPEC §2.2).

If phi yields no rare or no normal bump, the input is returned
unchanged with a ``UserWarning`` (SPEC §4.2, ADR-0011).
```

## `tsresample.metrics`

### `precision_phi(y_true, y_pred, *, relevance='auto', rel_threshold=0.9)`

```text
Utility-based precision over cases with ``phi(y_pred) >= rel_threshold``.

``sum |1 + u_i| / sum |1 + phi(y_pred_i)|``; ``1e-5`` if no case qualifies.
```

### `recall_phi(y_true, y_pred, *, relevance='auto', rel_threshold=0.9)`

```text
Utility-based recall over cases with ``phi(y_true) >= rel_threshold``.

``sum |1 + u_i| / sum |1 + phi(y_i)|``; ``1e-5`` if no case qualifies.
```

### `f1_phi(y_true, y_pred, *, relevance='auto', rel_threshold=0.9, beta=1.0)`

```text
F-beta of :func:`precision_phi` and :func:`recall_phi` (0 if either is 0).

Examples
--------
>>> import numpy as np
>>> y = np.array([0.1, -0.2, 3.0, 0.0, 0.3, -0.1, 0.2, -2.5])
>>> round(f1_phi(y, y), 6)  # a perfect forecast
1.0
```

### `sera(y_true, y_pred, *, relevance='auto', step=0.001, return_curve=False)`

```text
Squared error-relevance area (Ribeiro & Moniz 2020).

``SER_t = sum over phi(y_i) >= t of (y_i - y_pred_i)^2``, integrated over
``t`` in [0, 1] on a uniform grid of spacing ``step`` (trapezoidal rule).
With ``return_curve=True`` returns ``(t, SER_t)`` instead.
```

### `control_points(y)`

```text
Fit phi's three control points ``(x, phi)`` on ``y`` (SPEC §4.1 Steps 1-3).

Rows are low, median and high. The outer points sit at the whisker ends with
phi = 1, or at ``min``/``max`` with phi = 0 on a side with no outliers.
NaNs are dropped before fitting.
```

## `tsresample.pipeline`

### `load_series(path, *, target, date_col=None, diff=False, impute='knn')`

```text
Load one series from a CSV column, oldest first.

Parameters
----------
path : path to a CSV file with a header row.
target : column holding the series.
date_col : optional column to sort by (then dropped); rows are otherwise
    taken in file order.
diff : if True, return first differences (one value shorter).
impute : ``"knn"`` fills gaps by lag-window kNN (SPEC §2.4); ``"drop"``
    removes them; ``None`` raises if any value is missing.
```

### `imbalance_summary(y, *, rel_threshold=0.9, relevance='auto')`

```text
One row of the paper's Table 1: ``N, n_normal, n_rare, IR, pct_rare``.

A case is rare iff ``phi(y) >= rel_threshold``. ``IR = n_rare / n_normal``
(``inf`` if every case is rare) and ``pct_rare = 100 * n_rare / N``.

Examples
--------
>>> imbalance_summary([4, 1, 9, 2, 100, 3, 8, 5, 7, 6])["n_rare"]
2
```

### `temporal_split(X, y, *, train_size=0.5, test_size=0.25, n_reps=50, random_state=None)`

```text
Yield ``(X_train, y_train, X_test, y_test)`` for ``n_reps`` Monte Carlo splits.

Each split is a contiguous training window of ``trunc(train_size * n)`` rows
followed immediately by a test window of ``trunc(test_size * n)`` rows, at a
random position (the paper's ``MonteCarlo(nReps=50, szTrain=.5,
szTest=.25)``). Rows are never shuffled.
```

### `evaluate(estimator, X, y, *, strategies=None, metrics=None, splitter=None, random_state=0)`

```text
Score ``estimator`` under each resampling strategy on temporal splits.

For every split, phi is fit on the *training* target only (ADR-0008); the
same control points drive the resampler and the metrics (ADR-0016), so the
grid is leakage-free by construction.

Parameters
----------
estimator : scikit-learn regressor; cloned for every fit.
X, y : the embedded series (``embed``), rows in time order.
strategies : name -> ``TimeSeriesResampler`` (or ``None`` for no
    resampling). Default: ``"baseline"`` plus the nine SPEC §4.6 cells.
metrics : name -> ``f(y_true, y_pred, *, relevance=cp)``. Default:
    ``precision_phi``, ``recall_phi``, ``f1_phi``, ``sera``.
splitter : ``f(X, y)`` yielding ``(X_train, y_train, X_test, y_test)``.
    Default: ``temporal_split(X, y, random_state=random_state)`` (50 reps).
random_state : seeds the default splitter, and one stream from which every
    resampler without its own seed draws a fresh one per split, so a run is
    reproducible and splits stay independent.

Returns
-------
DataFrame with columns ``strategy, split, metric, value``: one row per
(strategy, split, metric).
```
