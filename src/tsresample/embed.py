"""Time-delay embedding of a univariate series (SPEC §2.1, ADR-0005)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def embed(
    series: ArrayLike,
    k: int,
    *,
    horizon: int = 1,
    exog: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a supervised matrix from a univariate series by time-delay embedding.

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
    """
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"embed: series must be 1-D; got shape {s.shape}.")
    if not np.all(np.isfinite(s)):
        raise ValueError(
            "embed: series contains NaN or inf; impute or drop them first "
            "(tsresample.pipeline imputes with lag-window kNN)."
        )
    if k < 1:
        raise ValueError(f"embed: expected k >= 1; got k={k}.")
    if horizon < 1:
        raise ValueError(f"embed: expected horizon >= 1; got horizon={horizon}.")
    n = len(s) - k - horizon
    if n < 1:
        raise ValueError(
            f"embed: series too short: {len(s)} values give no rows with "
            f"k={k}, horizon={horizon}; need at least {k + horizon + 1}."
        )
    cols = [s[k - j : k - j + n] for j in range(k + 1)]
    if exog is not None:
        e = np.asarray(exog, dtype=np.float64)
        if len(e) != len(s):
            raise ValueError(
                f"embed: exog has {len(e)} rows but series has {len(s)}; "
                "they must align by position."
            )
        cols.append(e[k : k + n])
    return np.column_stack(cols), s[k + horizon : k + horizon + n]
