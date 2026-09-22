# ADR-0005 — `embed(series, k, horizon)` and its mapping to the paper's `create.data`

**Status:** Accepted · **Date:** 2026-09-03

## Context

The paper builds its supervised matrices with

```r
create.data <- function(ts, embed) {
  t <- index(ts)[-(1:(embed-1))]
  e <- embed(ts, embed)[, embed:1]
  colnames(e) <- paste('V', 1:embed, sep='')
  d <- xts(e, t); as.data.frame(d)
}
```

and calls it with `embed = 10`. R's `embed(ts, m)` yields `m` columns; reversing with
`[, m:1]` puts the oldest lag first and the **current** value last, and the experiment
treats that last column as the target. So `create.data(ts, 10)` gives **9 predictors and
target `y_t`** — a nowcast of the current value from the previous nine.

A naive `embed(series, k=10)` in this library's own convention (k+1 lag columns, target at
`t + horizon`) yields **11 predictors and target `y_{t+1}`**. Silently using `k=10` for
both would compare two different learning problems and quietly break replication.

The user's `imbalance_eval` library independently uses the same shape as the paper
(`lag_k … lag_1, target` = `k` predictors + target), confirming the reading.

## Decision

The library's convention is explicit and horizon-aware:

```python
embed(series, k, *, horizon=1, exog=None) -> (X, y)
# row t of X is [y_t, y_{t-1}, ..., y_{t-k}]   (k+1 columns)
# y[t] is series[t + horizon]
# X has shape (n - k - horizon, k + 1 + m)
```

The mapping to the paper is stated in the docstring and in SPEC §2.1:

```
paper's create.data(ts, m)   ≡   embed(series, k=m-1, horizon=1)
paper's create.data(ts, 10)  ≡   embed(series, k=9,  horizon=1)
```

The replication suite uses `k=9, horizon=1` and asserts the resulting row count matches
the recorded experiment's `N`.

## Consequences

- The library's default is *not* the paper's default. That is deliberate: `horizon` is the
  parameter users actually need, and hiding a forecast horizon of 0 inside an
  off-by-one lag count would be a worse API.
- Both `k` and `horizon` are keyword-documented with the mapping, so a reader coming from
  the paper cannot make the substitution by accident.
- `exog` columns are contemporaneous only — `exog[t]` joins row `t`. Lagging exogenous
  inputs is the caller's job, because only the caller knows which are known-in-advance.
