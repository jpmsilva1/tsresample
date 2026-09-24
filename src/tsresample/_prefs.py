"""Preference vectors for the temporal biases (SPEC §4.3, ADR-0012)."""

import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsresample._bins import Bump

Bias = Literal["temporal", "temporal+phi"] | None


def preference(
    bump: Bump, time_index: ArrayLike, phi: ArrayLike, bias: Bias
) -> NDArray[np.float64]:
    """Sampling probabilities for ``bump``'s cases, aligned with ``bump.idx``.

    ``temporal`` weighs the case with chronological rank ``j`` *within the bump*
    (not its position in the series) by ``j / r``; ``temporal+phi`` multiplies
    by its phi; ``None`` is uniform. Falls back to uniform when the weights sum
    to zero.
    """
    r = len(bump.idx)
    if bias is None:
        return np.full(r, 1.0 / r)
    t = np.asarray(time_index)[bump.idx]
    j = np.argsort(np.argsort(t, kind="stable"), kind="stable") + 1.0
    p = j / r
    if bias == "temporal+phi":
        p = p * np.asarray(phi, dtype=np.float64)[bump.idx]
    total = p.sum()
    return p / total if total > 0 else np.full(r, 1.0 / r)


def draw(
    p: NDArray[np.float64], size: int, *, replace: bool, rng: np.random.RandomState
) -> NDArray[np.intp]:
    """Draw ``size`` positions of ``p`` with probabilities ``p``.

    Without replacement, if fewer than ``size`` cases have ``p > 0`` (possible
    for ``temporal+phi``), R raises; we take every positive case, fill the rest
    uniformly from the zero-probability cases, and warn (SPEC §4.3, ADR-0011).
    """
    positive = np.flatnonzero(p > 0)
    if replace or size <= len(positive):
        return rng.choice(len(p), size=size, replace=replace, p=p)
    warnings.warn(
        f"draw: only {len(positive)} cases have positive preference but {size} "
        "are needed without replacement; the rest are drawn uniformly.",
        UserWarning,
        stacklevel=2,
    )
    zeros = np.flatnonzero(p <= 0)
    fill = rng.choice(zeros, size=size - len(positive), replace=False)
    return np.concatenate([positive, fill])
