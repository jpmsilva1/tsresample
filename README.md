# tsresample

[![CI](https://github.com/jpmsilva1/tsresample/actions/workflows/ci.yml/badge.svg)](https://github.com/jpmsilva1/tsresample/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-informational)](src/tsresample/py.typed)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

**Resampling strategies for imbalanced time-series forecasting.**

Forecasting models are trained to be right on average, so they learn to ignore the rare,
extreme values that usually matter most: demand spikes, floods, price crashes.
`tsresample` rebalances the training data before you fit, using the strategies of
[Moniz, Branco & Torgo (2017)](https://doi.org/10.1007/s41060-017-0044-3). It also
provides the relevance-aware metrics that tell you whether the rare cases actually
improved.

## Highlights

- **Nine resampling strategies.** Undersampling, oversampling and SMOTE, each with no
  bias, a temporal bias, or a temporal + relevance bias.
- **Faithful to the reference.** Reproduces the authors' original R implementation and is
  verified against recorded R output ([Validation](#validation)).
- **scikit-learn compatible.** `TimeSeriesResampler` supports `get_params`, `set_params`
  and `clone`. Output rows stay in time order.
- **Relevance-aware metrics.** `precision_phi`, `recall_phi`, `f1_phi` and `sera`, sharing
  one definition of "rare" with the resampler.
- **Leakage-free evaluation.** Temporal Monte Carlo splits, with relevance refit on every
  training window.
- **Lightweight.** The core needs only numpy, scipy and scikit-learn. It is fully typed
  (`mypy --strict`) and tested on Linux, macOS and Windows.

## Installation

```bash
pip install tsresample             # core: numpy, scipy, scikit-learn
pip install "tsresample[io]"       # + pandas: CSV loading, evaluation grid, CLI
```

Requires Python 3.10 or newer.

## Quickstart

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from tsresample import TimeSeriesResampler, embed
from tsresample.metrics import control_points, f1_phi

# A heavy-tailed series: mostly ordinary values, a few extreme ones.
series = np.random.RandomState(0).standard_t(3, size=600)

X, y = embed(series, k=8)                     # 9 lagged values -> next value
X_train, y_train, X_test, y_test = X[:400], y[:400], X[400:], y[400:]

cp = control_points(y_train)                  # what counts as "rare", fit on train only
resampler = TimeSeriesResampler("smote", "temporal", relevance=cp, random_state=0)
X_res, y_res = resampler.fit_resample(X_train, y_train)

for name, (Xf, yf) in {"baseline": (X_train, y_train), "SMOTE-T": (X_res, y_res)}.items():
    pred = LinearRegression().fit(Xf, yf).predict(X_test)
    print(f"{name:9s} F1phi = {f1_phi(y_test, pred, relevance=cp):.3f}")
```

```text
baseline  F1phi = 0.000
SMOTE-T   F1phi = 0.380
```

The baseline model never predicts a rare value. After resampling, it does.

## Usage

The library has three layers. Use only what you need.

| Layer | Entry point | Requires |
|---|---|---|
| Primitives | `tsresample.embed`, `tsresample.TimeSeriesResampler`, `tsresample.metrics` | core |
| Pipeline | `tsresample.pipeline`: `load_series`, `imbalance_summary`, `temporal_split`, `evaluate` | `[io]` |
| Command line | `tsresample` | `[io]` |

### Evaluate every strategy on a CSV

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tsresample import embed
from tsresample.pipeline import evaluate, imbalance_summary, load_series, temporal_split

# Example data; use your own CSV with a date column and a value column.
dates = pd.date_range("2020-01-01", periods=500, freq="D")
values = np.random.RandomState(1).standard_t(3, size=500)
pd.DataFrame({"date": dates, "demand": values}).to_csv("demand.csv", index=False)

series = load_series("demand.csv", target="demand", date_col="date")  # gaps: lag-window kNN
print(imbalance_summary(series))              # N, n_normal, n_rare, IR, pct_rare

X, y = embed(series, k=8)
results = evaluate(
    LinearRegression(), X, y,
    splitter=lambda X, y: temporal_split(X, y, n_reps=5, random_state=0),
)
print(results.groupby(["strategy", "metric"])["value"].mean().unstack().round(3))
```

`evaluate` returns a tidy frame with one row per (strategy, split, metric). The default
grid is a baseline plus all nine strategies.

### Command line

```bash
tsresample demand.csv --target demand --date-col date        # one series
tsresample --config manifest.csv --output table1.csv         # many series
```

This prints the paper's Table 1 imbalance summary. Users of `imbalance_eval` can keep their
commands; see the [migration guide](docs/MIGRATION.md).

## How it works

**Relevance.** A relevance function φ(y) ∈ [0, 1] scores how extreme each target value is.
The default is the paper's boxplot-based φ: values beyond the whiskers score 1 and the
median scores 0. Cases with φ ≥ 0.9 are *rare*. Fit φ on training data with
`metrics.control_points(y_train)` and pass the result as `relevance=` to both the resampler
and the metrics.

**Strategies.** Cases are grouped into rare and normal "bumps" of the target's value range,
then rebalanced:

| `strategy` | Effect | Output size |
|---|---|---|
| `"under"` | Removes normal cases until they match the rare total | smaller |
| `"over"` | Keeps every case and adds copies of rare ones | larger |
| `"smote"` (default) | Synthesises rare cases by interpolation and undersamples normal ones, aiming each bump at `N / #bumps` | about the same |

| `bias` | Which cases are preferred when sampling |
|---|---|
| `None` | all equally |
| `"temporal"` | recent ones |
| `"temporal+phi"` | recent and highly relevant ones |

Other options, all documented in the [API reference](docs/api.md):
- the relevance threshold;
- SMOTE's `k`;
- explicit sampling percentages `o` and `u`;
- `r_quirks`.

## Validation

The library is checked against the recorded output of the canonical R experiment: 24
datasets, 50 temporal splits each, linear models.

| Check | Result |
|---|---|
| Relevance control points vs R | 85 of 90 recorded splits exact; the other 5 are excluded (a precision loss in the export) |
| Precision/recall/F1 and SERA vs R | within 1e-6 on every recorded case in the test fixture |
| Resampled training-set sizes | 10,800 / 10,800 exact |
| Paper's conclusion (resampling beats the baseline, Wilcoxon) | reproduced for all 9 strategies |
| Strategy rankings vs R (Spearman) | mean ρ = 0.885 |

On four datasets the strategies are statistically tied. Their order is noise, so they are
not held to the per-dataset ranking floor
([ADR-0017](Blueprint/docs/adr/0017-r1-floor-noise-limited-datasets.md)). Full results are
in [`replication_report.md`](replication_report.md).

Where the paper's pseudocode and the R code disagree, the library follows R. Every such
choice is listed in [Deviations from the paper](docs/deviations.md). `r_quirks=False`
selects the paper's reading for SMOTE.

## Documentation

- [API reference](docs/api.md), generated from the docstrings
- [Deviations from the paper's pseudocode](docs/deviations.md)
- [Migrating from `imbalance_eval`](docs/MIGRATION.md)
- [Replication report](replication_report.md)
- [Specification and design decisions](Blueprint/README.md)

## Citation

If you use `tsresample` in your research, please cite the paper whose algorithms it
implements:

```bibtex
@article{moniz2017resampling,
  title   = {Resampling strategies for imbalanced time series forecasting},
  author  = {Moniz, Nuno and Branco, Paula and Torgo, Lu{\'i}s},
  journal = {International Journal of Data Science and Analytics},
  volume  = {3},
  number  = {3},
  pages   = {161--181},
  year    = {2017},
  doi     = {10.1007/s41060-017-0044-3}
}
```

To cite the software itself, use [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository").

The relevance function follows Ribeiro, R. (2011), *Utility-based Regression*, PhD thesis,
University of Porto. SERA follows Ribeiro, R. & Moniz, N. (2020), *Imbalanced regression
and extreme value prediction*, Machine Learning 109, 1803–1835. `tsresample` is an
independent implementation and is not affiliated with or endorsed by those authors.

## Contributing

Contributions are welcome. The library is built from a written specification, so please
read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. To set up a
development environment:

```bash
git clone https://github.com/jpmsilva1/tsresample.git
cd tsresample
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE) © João P. M. Silva and tsresample contributors.
