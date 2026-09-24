"""Relevance function phi: control points and Hermite evaluation.

SPEC §4.1; ADR-0001, ADR-0002.
"""

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicHermiteSpline

_COEF = 1.5  # boxplot whisker coefficient of uba's "extremes" method (ADR-0001)


def _hinges(x: NDArray[np.float64]) -> tuple[float, float, float]:
    # Tukey hinges as R's fivenum computes them (ADR-0001 amendment), not type-7
    # quantiles; 1-based positions d, averaged over floor/ceil.
    n = len(x)
    n4 = math.floor((n + 3) / 2) / 2
    q1, med, q3 = (
        0.5 * (x[math.floor(d) - 1] + x[math.ceil(d) - 1])
        for d in (n4, (n + 1) / 2, n + 1 - n4)
    )
    return float(q1), float(med), float(q3)


def control_points(y: ArrayLike) -> NDArray[np.float64]:
    """Fit phi's three control points ``(x, phi)`` on ``y`` (SPEC §4.1 Steps 1-3).

    Rows are low, median and high. The outer points sit at the whisker ends with
    phi = 1, or at ``min``/``max`` with phi = 0 on a side with no outliers.
    NaNs are dropped before fitting.
    """
    x = np.sort(np.asarray(y, dtype=np.float64).ravel())
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("control_points: y has no non-NaN values; expected >= 1.")
    q1, med, q3 = _hinges(x)
    iqr = q3 - q1
    inside = x[(x >= q1 - _COEF * iqr) & (x <= q3 + _COEF * iqr)]
    lw, uw = inside[0], inside[-1]
    low = (lw, 1.0) if x[0] < lw else (x[0], 0.0)
    high = (uw, 1.0) if x[-1] > uw else (x[-1], 0.0)
    return np.array([low, (med, 0.0), high], dtype=np.float64)


def phi(y: ArrayLike, cp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate phi at ``y`` given control points ``cp`` (SPEC §4.1 Step 4).

    Cubic Hermite with zero slope at every knot (ADR-0002), constant at the
    endpoint's phi at and beyond the outer knots, clipped to [0, 1].
    """
    y = np.asarray(y, dtype=np.float64)
    (x_lo, p_lo), (med, _), (x_hi, p_hi) = cp
    if not x_lo < med < x_hi:
        # R's spline constructor raises here; we degrade to "nothing is rare"
        # (documented robustness deviation, SPEC §4.1, ADR-0011).
        warnings.warn(
            f"phi: control-point x = ({x_lo}, {med}, {x_hi}) is not strictly "
            "increasing (e.g. IQR = 0 with no outliers); phi is 0 everywhere.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(y)
    spline = CubicHermiteSpline([x_lo, med, x_hi], [p_lo, 0.0, p_hi], np.zeros(3))
    # Exact endpoint values and the clip are both required: scipy can return
    # 1 - 4e-16 at the knot, which drops cases out of SERA's t = 1 grid point
    # (ADR-0014).
    inside = np.clip(spline(np.clip(y, x_lo, x_hi)), 0.0, 1.0)
    out: NDArray[np.float64] = np.where(
        y <= x_lo, p_lo, np.where(y >= x_hi, p_hi, inside)
    )
    return out


def resolve(relevance: object, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """phi for ``y`` from a ``relevance`` argument: "auto", an array or a callable.

    Shared by the resampler and the metrics, so both call the same cases rare.
    """
    if isinstance(relevance, str):
        if relevance != "auto":
            raise ValueError(
                "relevance: expected 'auto', an array or a callable; "
                f"got {relevance!r}."
            )
        return phi(y, control_points(y))
    got = relevance(y) if callable(relevance) else relevance
    p = np.asarray(got, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"relevance: expected shape {y.shape}; got {p.shape}.")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("relevance: values must lie in [0, 1] (and not be NaN).")
    return p
