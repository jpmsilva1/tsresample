# tsresample

Resampling strategies for **imbalanced time-series forecasting**, and the metrics to judge
them. When the values you care about most (demand spikes, floods, price crashes) are rare,
an ordinary regressor learns to ignore them. `tsresample` rebalances the training set
before you fit, using the nine strategies of Moniz, Branco & Torgo (2017): random
undersampling, random oversampling and SMOTE, each available with no bias, a temporal
bias, or a temporal+relevance bias. It also ships the relevance-aware metrics
(`precision_phi`, `recall_phi`, `f1_phi`, `sera`) that show whether the rare cases got
better.

It reproduces the **original R implementation**, quirks included, and is verified against
recorded R output (see [Replication](#replication)).

## Install

```bash
pip install tsresample            # numpy, scipy, scikit-learn only
pip install "tsresample[io]"      # + pandas: CSV loading, evaluation grid, CLI
```

Python 3.10–3.13.

## 30-second example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from tsresample import TimeSeriesResampler, embed
from tsresample.metrics import control_points, f1_phi

# A heavy-tailed series: most values are ordinary, a few are extreme.
series = np.random.RandomState(0).standard_t(3, size=600)

X, y = embed(series, k=8)  # 9 lagged values -> next value
X_train, y_train, X_test, y_test = X[:400], y[:400], X[400:], y[400:]

cp = control_points(y_train)  # "what counts as rare", fit on train only
resampler = TimeSeriesResampler("smote", "temporal", relevance=cp, random_state=0)
X_res, y_res = resampler.fit_resample(X_train, y_train)

for name, (Xf, yf) in {"as is": (X_train, y_train), "SMOTE-T": (X_res, y_res)}.items():
    pred = LinearRegression().fit(Xf, yf).predict(X_test)
    print(f"{name:8s} F1phi = {f1_phi(y_test, pred, relevance=cp):.3f}")
```

`TimeSeriesResampler` is a scikit-learn estimator (`get_params`, `set_params`, `clone`).
Its rows come back in time order, with each synthetic case placed right after the case it
was grown from.

## The three layers

| Layer | Import | Needs | Use it for |
|---|---|---|---|
| 1. Primitives | `tsresample.embed`, `tsresample.TimeSeriesResampler`, `tsresample.metrics` | numpy/scipy/sklearn | Resampling and scoring inside your own code |
| 2. Pipeline | `tsresample.pipeline`: `load_series`, `imbalance_summary`, `temporal_split`, `evaluate` | `[io]` | CSV to results table in a few lines, leakage-free |
| 3. CLI | `tsresample` | `[io]` | The paper's Table 1 (how imbalanced is this series?) from the shell |

### From a CSV to a results table

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tsresample import embed
from tsresample.pipeline import evaluate, imbalance_summary, load_series, temporal_split

# Any CSV with a date column and a value column (gaps are filled by lag-window kNN).
dates = pd.date_range("2020-01-01", periods=500, freq="D")
values = np.random.RandomState(1).standard_t(3, size=500)
pd.DataFrame({"date": dates, "demand": values}).to_csv("demand.csv", index=False)

series = load_series("demand.csv", target="demand", date_col="date")
print(imbalance_summary(series))  # N, n_normal, n_rare, IR, pct_rare

X, y = embed(series, k=8)
results = evaluate(
    LinearRegression(),
    X,
    y,
    splitter=lambda X, y: temporal_split(X, y, n_reps=5, random_state=0),
)
print(results.groupby(["strategy", "metric"])["value"].mean().unstack().round(3))
```

`evaluate` refits φ on each training window and uses it for both resampling and scoring,
so nothing from the test window leaks in.

### Command line

```bash
tsresample demand.csv --target demand --date-col date
tsresample --config manifest.csv --output table1.csv   # many series at once
```

Coming from `imbalance_eval`? Your commands translate unchanged; see
[docs/MIGRATION.md](docs/MIGRATION.md).

## Choosing a strategy

| `strategy` | What it does | Output size |
|---|---|---|
| `"under"` | Drops normal cases until the normal side matches the rare total | smaller |
| `"over"` | Keeps everything and adds copies of rare cases | larger |
| `"smote"` (default) | Aims every bump at `N / #bumps`: synthesises rare cases by interpolation, undersamples normal ones | about the same |

`bias=None` samples uniformly. `"temporal"` prefers recent cases. `"temporal+phi"`
prefers cases that are both recent and highly relevant. The relevance threshold
(`rel_threshold=0.9`), SMOTE's `k=5`, and explicit percentages `o`/`u` are all
constructor arguments. See `help(tsresample.TimeSeriesResampler)` or
[docs/api.md](docs/api.md).

## Relevance φ

"Rare" is defined by a relevance function φ(y) ∈ [0, 1]. The default (`relevance="auto"`)
is the paper's boxplot-based φ: cases beyond the whiskers get φ = 1, and the median gets 0.
Fit it on training data with `tsresample.metrics.control_points(y_train)` and pass the
result as `relevance=` to both the resampler and the metrics. A per-case φ array or a
callable also works for the resampler and `sera`.

## Replication

Gate G4 compares the library against the recorded canonical R experiment: 24 datasets,
50 splits, `lm`. All nine strategies reproduce the paper's conclusion, since resampling
significantly beats the baseline on F1φ (R5). Resampled training-set sizes match the
specification exactly (10,800/10,800). Per-dataset rankings agree with R (mean Spearman
ρ = 0.88).

One diagnostic misses its bar. On DS19 the nine strategies are statistically tied, and the
ranking correlation there is 0.47 against a per-dataset floor of 0.6. Full numbers are in
[`replication_report.md`](replication_report.md).

Where the library knowingly differs from the paper's pseudocode, it is because it follows
the R code instead. Each difference is listed with its reason in
[docs/deviations.md](docs/deviations.md). `r_quirks=False` switches SMOTE to the paper's
reading.

## Citation

If you use `tsresample`, please cite the paper whose algorithms it implements (and the
software, see [`CITATION.cff`](CITATION.cff)):

> The resampling strategies implemented here are those of Moniz, N., Branco, P., &
> Torgo, L. (2017). *Resampling strategies for imbalanced time series forecasting.*
> International Journal of Data Science and Analytics 3(3), 161–181. The relevance
> function follows Ribeiro, R. (2011), *Utility-based Regression*, PhD thesis, University
> of Porto. SERA follows Ribeiro, R. & Moniz, N. (2020), *Imbalanced regression and
> extreme value prediction*, Machine Learning 109, 1803–1835. This is an independent
> implementation and is not affiliated with or endorsed by those authors.

## Contributing

The library is built from a written plan in [`Blueprint/`](Blueprint/). Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md), which includes the reading order for new sessions.

MIT licence.
