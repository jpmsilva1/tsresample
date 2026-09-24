"""Layer 2 pipeline: load_series, imbalance_summary, temporal_split, evaluate.

SPEC §2.4; ADR-0010. Needs the ``io`` extra: ``pip install tsresample[io]``.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "tsresample.pipeline and the tsresample CLI need pandas: "
        'pip install "tsresample[io]"'
    ) from None
from numpy.typing import ArrayLike
from sklearn.base import clone

from tsresample import _relevance
from tsresample.metrics import f1_phi, precision_phi, recall_phi, sera
from tsresample.pipeline.io import load_series
from tsresample.pipeline.splits import temporal_split
from tsresample.resampler import TimeSeriesResampler

__all__ = ["evaluate", "imbalance_summary", "load_series", "temporal_split"]


def imbalance_summary(
    y: ArrayLike, *, rel_threshold: float = 0.9, relevance: Any = "auto"
) -> dict[str, float]:
    """One row of the paper's Table 1: ``N, n_normal, n_rare, IR, pct_rare``.

    A case is rare iff ``phi(y) >= rel_threshold``. ``IR = n_rare / n_normal``
    (``inf`` if every case is rare) and ``pct_rare = 100 * n_rare / N``.

    Examples
    --------
    >>> imbalance_summary([4, 1, 9, 2, 100, 3, 8, 5, 7, 6])["n_rare"]
    2
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    if len(ya) < 2:
        raise ValueError(f"imbalance_summary: need at least 2 values; got {len(ya)}.")
    if not np.isfinite(ya).all():
        raise ValueError(
            "imbalance_summary: y contains NaN or inf; impute or drop them first "
            "(load_series does)."
        )
    n_rare = int((_relevance.resolve(relevance, ya) >= rel_threshold).sum())
    n_normal = len(ya) - n_rare
    return {
        "N": len(ya),
        "n_normal": n_normal,
        "n_rare": n_rare,
        "IR": n_rare / n_normal if n_normal else float("inf"),
        "pct_rare": 100.0 * n_rare / len(ya),
    }


def _default_strategies() -> dict[str, TimeSeriesResampler | None]:
    # SPEC §4.6 workflow labels; "baseline" trains on the data as given.
    bias = {"B": None, "T": "temporal", "TPhi": "temporal+phi"}
    grid: dict[str, TimeSeriesResampler | None] = {"baseline": None}
    for strategy in ("under", "over", "smote"):
        for tag, b in bias.items():
            grid[f"{strategy.upper()}{tag}"] = TimeSeriesResampler(strategy, b)  # type: ignore[arg-type]
    return grid


def evaluate(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    *,
    strategies: Mapping[str, TimeSeriesResampler | None] | None = None,
    metrics: Mapping[str, Callable[..., Any]] | None = None,
    splitter: Callable[..., Iterator[tuple[Any, ...]]] | None = None,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Score ``estimator`` under each resampling strategy on temporal splits.

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
    """
    strategies = _default_strategies() if strategies is None else strategies
    metrics = metrics or {
        f.__name__: f for f in (precision_phi, recall_phi, f1_phi, sera)
    }
    splits = (
        splitter(X, y)
        if splitter is not None
        else temporal_split(X, y, random_state=random_state)
    )
    rng = np.random.RandomState(random_state)  # one stream: independent draws
    rows = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(splits):
        cp = _relevance.control_points(y_tr)
        for name, resampler in strategies.items():
            Xf, yf = X_tr, y_tr
            if resampler is not None:
                r = clone(resampler).set_params(relevance=cp)
                if r.random_state is None:
                    r.set_params(random_state=rng.randint(2**31 - 1))
                Xf, yf = r.fit_resample(X_tr, y_tr)
            pred = clone(estimator).fit(Xf, yf).predict(X_te)
            for metric, f in metrics.items():
                rows.append((name, i, metric, float(f(y_te, pred, relevance=cp))))
    return pd.DataFrame(rows, columns=["strategy", "split", "metric", "value"])
