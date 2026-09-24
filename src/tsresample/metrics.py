"""Relevance-aware metrics: precision_phi, recall_phi, f1_phi, sera.

SPEC §2.3, §4.7; ADR-0009, ADR-0014, ADR-0016. Score a test set with the phi fit
on the *training* target::

    cp = control_points(y_train)
    f1_phi(y_test, y_pred, relevance=cp)

Known residual: for a phi whose control-point values are (1, 0, 0) (low side
rare, high side not), agreement of precision/recall/F1 with R's ``uba`` has not
been established (ADR-0014, "Residual"). Every other shape matches R to 1e-6.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid

from tsresample import _relevance, _utility
from tsresample._relevance import control_points

__all__ = ["control_points", "f1_phi", "precision_phi", "recall_phi", "sera"]

_FLOOR = 1e-5  # R's value for an empty selection (SPEC §4.7), never 0 or NaN

Relevance = str | ArrayLike | Callable[..., Any]


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    y = np.asarray(y_true, dtype=np.float64)
    yh = np.asarray(y_pred, dtype=np.float64)
    if y.ndim != 1 or y.shape != yh.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D and the same length; got {y.shape} "
            f"and {yh.shape}."
        )
    if not (np.isfinite(y).all() and np.isfinite(yh).all()):
        raise ValueError("y_true and y_pred must be finite (no NaN or inf).")
    return y, yh


def _cp(relevance: Relevance, y: NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(relevance, str) and relevance == "auto":
        return control_points(y)
    if not isinstance(relevance, str) and not callable(relevance):
        cp = np.asarray(relevance, dtype=np.float64)
        if cp.ndim == 2:
            return _relevance.as_control_points(cp)
    raise ValueError(
        "precision/recall/F1 need relevance='auto' or phi's control points of "
        "shape (3, 2) (see metrics.control_points); a per-case phi array or a "
        "callable cannot give phi(y_pred) or the utility bumps (ADR-0016)."
    )


def _ratio(
    sel: NDArray[np.bool_], u: NDArray[np.float64], phi: NDArray[np.float64]
) -> float:
    if not sel.any():
        return _FLOOR
    return float(np.abs(1 + u[sel]).sum() / np.abs(1 + phi[sel]).sum())


def recall_phi(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    relevance: Relevance = "auto",
    rel_threshold: float = 0.9,
) -> float:
    """Utility-based recall over cases with ``phi(y_true) >= rel_threshold``.

    ``sum |1 + u_i| / sum |1 + phi(y_i)|``; ``1e-5`` if no case qualifies.
    """
    y, yh = _pair(y_true, y_pred)
    cp = _cp(relevance, y)
    phi_y = _relevance.phi(y, cp)
    return _ratio(phi_y >= rel_threshold, _utility.utility(yh, y, cp), phi_y)


def precision_phi(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    relevance: Relevance = "auto",
    rel_threshold: float = 0.9,
) -> float:
    """Utility-based precision over cases with ``phi(y_pred) >= rel_threshold``.

    ``sum |1 + u_i| / sum |1 + phi(y_pred_i)|``; ``1e-5`` if no case qualifies.
    """
    y, yh = _pair(y_true, y_pred)
    cp = _cp(relevance, y)
    phi_h = _relevance.phi(yh, cp)
    return _ratio(phi_h >= rel_threshold, _utility.utility(yh, y, cp), phi_h)


def f1_phi(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    relevance: Relevance = "auto",
    rel_threshold: float = 0.9,
    beta: float = 1.0,
) -> float:
    """F-beta of :func:`precision_phi` and :func:`recall_phi` (0 if either is 0).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.1, -0.2, 3.0, 0.0, 0.3, -0.1, 0.2, -2.5])
    >>> round(f1_phi(y, y), 6)  # a perfect forecast
    1.0
    """
    kw: dict[str, Any] = {"relevance": relevance, "rel_threshold": rel_threshold}
    p = precision_phi(y_true, y_pred, **kw)
    r = recall_phi(y_true, y_pred, **kw)
    if p == 0 or r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def sera(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    relevance: Relevance = "auto",
    step: float = 0.001,
    return_curve: bool = False,
) -> float | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Squared error-relevance area (Ribeiro & Moniz 2020).

    ``SER_t = sum over phi(y_i) >= t of (y_i - y_pred_i)^2``, integrated over
    ``t`` in [0, 1] on a uniform grid of spacing ``step`` (trapezoidal rule).
    With ``return_curve=True`` returns ``(t, SER_t)`` instead.
    """
    n_steps = round(1 / step) if 0 < step <= 1 else 0
    if n_steps == 0 or abs(n_steps * step - 1) > 1e-9:
        raise ValueError(f"sera: step must be 1/m for an integer m >= 1; got {step}.")
    y, yh = _pair(y_true, y_pred)
    phi_y = _relevance.resolve(relevance, y)
    t = np.linspace(0.0, 1.0, n_steps + 1)
    # SER_t = sum of squared errors over phi >= t: sort by phi once, then read
    # suffix sums at each threshold (O(n log n) instead of O(n / step)).
    order = np.argsort(phi_y)
    suffix = np.concatenate([np.cumsum(((y - yh) ** 2)[order][::-1])[::-1], [0.0]])
    ser = suffix[np.searchsorted(phi_y[order], t, side="left")]
    if return_curve:
        return t, ser
    return float(trapezoid(ser, t))
