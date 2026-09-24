"""Shared input validation."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_xy(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``X`` 2-D and ``y`` 1-D, same length, finite, as float arrays."""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features); got shape {Xa.shape}."
        )
    if ya.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {ya.shape}.")
    if len(Xa) != len(ya):
        raise ValueError(f"X has {len(Xa)} rows but y has {len(ya)}.")
    if len(ya) < 2:
        raise ValueError(f"need at least 2 cases to resample; got {len(ya)}.")
    if not (np.isfinite(Xa).all() and np.isfinite(ya).all()):
        raise ValueError("X and y must be finite (no NaN or inf); impute first.")
    return Xa, ya


def check_choice(name: str, value: Any, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}: expected one of {allowed}; got {value!r}.")
