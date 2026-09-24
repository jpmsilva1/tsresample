"""Target counts and under/over selection (SPEC §4.4, ADR-0004, ADR-0006, ADR-0007).

Every count goes through ``trunc`` / ``round5`` / ``round_even`` (ADR-0007
amendment): R's ``sample(size=)`` truncates, per-bump ratios are rounded to 5
decimals, and the smote bump size uses R's ``round`` (half to even). The float
arithmetic order follows R's, so e.g. ``trunc(0.33333 * 15) == 4``.
"""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tsresample._bins import Bump
from tsresample._prefs import Bias, draw, preference

Strategy = Literal["under", "over", "smote"]


def trunc(x: float) -> int:
    return math.floor(x)


def round5(x: float) -> float:
    return round(x, 5)


def round_even(x: float) -> int:
    return round(x)  # Python's round is half-to-even, like R's


def ratios(
    bumps: list[Bump],
    N: int,
    strategy: Strategy,
    o: float | None,
    u: float | None,
) -> list[float]:
    """Per-bump multiplier ``c_B`` of SPEC §4.4 (1.0 means "keep whole")."""
    if strategy == "over" and o is not None and o < 1:
        raise ValueError(f"over: expected o >= 1 (R rejects smaller); got o={o}.")
    for name, v in (("o", o), ("u", u)):
        if v is not None and v < 0:
            raise ValueError(f"{strategy}: expected {name} >= 0; got {name}={v}.")
    rare = [b for b in bumps if b.rare]
    normal = [b for b in bumps if b.normal]
    n_r = sum(len(b.idx) for b in rare)
    n_u = sum(len(b.idx) for b in normal)
    out = []
    for b in bumps:
        size = len(b.idx)
        if strategy == "under":
            c = (
                (u if u is not None else round5((n_r / len(normal)) / size))
                if b.normal
                else 1.0
            )
        elif strategy == "over":
            c = (
                (o if o is not None else round5((n_u / len(rare)) / size))
                if b.rare
                else 0.0
            )
        elif o is None and u is None:
            c = round_even(N / len(bumps)) / size
        else:
            given = o if b.rare else u if b.normal else None
            c = 1.0 if given is None else given
        out.append(c)
    return out


def targets(
    bumps: list[Bump],
    N: int,
    strategy: Strategy,
    o: float | None,
    u: float | None,
) -> list[int]:
    """Output size of each bump after resampling (SPEC §4.4, §4.5 counts)."""
    out = []
    for b, c in zip(bumps, ratios(bumps, N, strategy, o, u), strict=True):
        size = len(b.idx)
        if strategy == "over":
            out.append(size + trunc(c * size))  # originals kept, copies appended
        elif size == 1 or c == 1 or (strategy == "under" and c >= 1):
            out.append(size)
        elif c < 1:
            out.append(trunc(c * size))
        else:  # smote synthesis: nexs per seed + extra seeds (SPEC §4.5)
            nexs = trunc(c - 1)
            out.append(size + nexs * size + trunc(size * (c - 1 - nexs)))
    return out


def resample(
    bumps: list[Bump],
    N: int,
    strategy: Strategy,
    o: float | None,
    u: float | None,
    time_index: NDArray[np.intp],
    phi: NDArray[np.float64],
    bias: Bias,
    rng: np.random.RandomState,
) -> tuple[NDArray[np.intp], list[tuple[Bump, float]]]:
    """Select cases per SPEC §4.4; return positions and the smote synthesis jobs.

    Positions index the input (duplicates allowed). Each job ``(bump, c)`` asks
    ``_synth`` for the synthetic cases of a bump with ``c > 1``; its originals
    are already among the positions. Replacement: under without, over and
    smote's undersampling with (ADR-0006 amendment).
    """
    parts: list[NDArray[np.intp]] = []
    jobs: list[tuple[Bump, float]] = []
    counts = targets(bumps, N, strategy, o, u)
    for b, c, n_out in zip(
        bumps, ratios(bumps, N, strategy, o, u), counts, strict=True
    ):
        size = len(b.idx)
        if strategy == "over":
            p = preference(b, time_index, phi, bias)
            extra = draw(p, n_out - size, replace=True, rng=rng)
            parts += [b.idx, b.idx[extra]]
        elif n_out < size:
            p = preference(b, time_index, phi, bias)
            parts.append(b.idx[draw(p, n_out, replace=strategy == "smote", rng=rng)])
        else:
            parts.append(b.idx)
            if n_out > size:
                jobs.append((b, c))
    return np.concatenate(parts), jobs


def assemble(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    idx: NDArray[np.intp],
    X_syn: NDArray[np.float64],
    y_syn: NDArray[np.float64],
    seeds: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rows in time order, each synthetic case right after its seed (SPEC §2.2)."""
    # Originals/copies sort at their time index, synthetic cases at seed + 0.5;
    # the stable sort keeps generation order among ties.
    key = np.concatenate([idx.astype(np.float64), seeds + 0.5])
    order = np.argsort(key, kind="stable")
    return np.vstack([X[idx], X_syn])[order], np.concatenate([y[idx], y_syn])[order]
