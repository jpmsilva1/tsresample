"""CSV and manifest loading; lag-window kNN imputation (ADR-0010)."""

import warnings
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Impute = Literal["knn", "drop"] | None


def load_series(
    path: str | PathLike[str],
    *,
    target: str,
    date_col: str | None = None,
    diff: bool = False,
    impute: Impute = "knn",
) -> NDArray[np.float64]:
    """Load one series from a CSV column, oldest first.

    Parameters
    ----------
    path : path to a CSV file with a header row.
    target : column holding the series.
    date_col : optional column to sort by (then dropped); rows are otherwise
        taken in file order.
    diff : if True, return first differences (one value shorter).
    impute : ``"knn"`` fills gaps by lag-window kNN (SPEC §2.4); ``"drop"``
        removes them; ``None`` raises if any value is missing.
    """
    df = pd.read_csv(path)
    for col in (target, date_col):
        if col is not None and col not in df.columns:
            raise ValueError(
                f"column {col!r} not found in {path}; available: {list(df.columns)}"
            )
    if date_col is not None:
        # format="mixed": exports often write midnight as a bare date (DS21-24)
        df = df.sort_values(
            date_col, kind="stable", key=lambda c: pd.to_datetime(c, format="mixed")
        )
    raw = df[target]
    s = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
    bad = raw[np.isnan(s) & raw.notna().to_numpy()]
    if len(bad):
        raise ValueError(
            f"column {target!r} has {len(bad)} non-numeric value(s), e.g. "
            f"{bad.astype(str).head(3).tolist()}; fix the file (thousands separators, "
            "decimal commas) or leave missing cells empty."
        )
    s = _impute(s, impute)
    return np.diff(s) if diff else s


def _impute(s: NDArray[np.float64], how: Impute) -> NDArray[np.float64]:
    if not np.isnan(s).any():
        return s
    if how is None:
        raise ValueError(
            f"series has {int(np.isnan(s).sum())} missing values; pass "
            "impute='knn' or impute='drop'."
        )
    if how == "drop":
        return np.asarray(s[~np.isnan(s)], dtype=np.float64)
    return _knn_fill(s)


_WINDOW = 9  # lags y_{t-9..t-1}: the paper's create.data(ts, 10) frame (ADR-0005)
_K = 10  # neighbours, as in the paper's knnImputation call (ADR-0010 amendment)


def _knn_fill(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lag-window kNN imputation (SPEC §2.4, ADR-0010 amendment).

    Walk the series in time order. For a missing y_t, standardise its lag
    window (earlier fills count as observed) with the embedded frame's column
    means and SDs, find the k nearest *originally* fully observed rows on the
    observed lags, and fill with their targets weighted exp(-d). Leading gaps
    with no observed lag are dropped with a warning. Donors come from the
    original series: that reproduces DS12's 10.99 %Rare (filled donors: 9.33).
    """
    if len(s) <= _WINDOW:
        raise ValueError("impute='knn' needs at least one gap-free window of 10.")
    frame = np.lib.stride_tricks.sliding_window_view(s, _WINDOW + 1)
    donors = frame[~np.isnan(frame).any(axis=1)]
    if len(donors) == 0:
        raise ValueError("impute='knn' needs at least one gap-free window of 10.")
    mu = np.nanmean(frame[:, :_WINDOW], axis=0)
    sd = np.nanstd(frame[:, :_WINDOW], axis=0, ddof=1)
    sd[sd == 0] = 1.0
    z_donors = (donors[:, :_WINDOW] - mu) / sd
    out = s.copy()
    for t in np.flatnonzero(np.isnan(s)):
        q = np.full(_WINDOW, np.nan)
        lags = out[max(0, t - _WINDOW) : t]
        q[_WINDOW - len(lags) :] = lags
        seen = ~np.isnan(q)
        if not seen.any():
            continue  # leading gap: dropped below
        zq = (q[seen] - mu[seen]) / sd[seen]
        d = np.sqrt(((z_donors[:, seen] - zq) ** 2).sum(axis=1))
        nn = np.argsort(d, kind="stable")[:_K]
        w = np.exp(-d[nn])
        out[t] = (w * donors[nn, _WINDOW]).sum() / w.sum()
    lead = int(np.isnan(out).sum())
    if lead:
        warnings.warn(
            f"impute='knn': dropped {lead} leading missing values with no "
            "observed lag.",
            UserWarning,
            stacklevel=3,
        )
    return np.asarray(out[~np.isnan(out)], dtype=np.float64)
