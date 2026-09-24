"""SMOTE-style synthetic case generation (SPEC §4.5, ADR-0013).

Defaults reproduce R (``r_quirks=True``): one lambda per synthetic case, the
target weighted by the most recent lag only, and for the temporal biases the
index mismatch between the time-ordered neighbour search and the value-ordered
rows the values are read from. ``r_quirks=False`` reads everything in time order
and weights the target by full-vector distances.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from tsresample._bins import Bump
from tsresample._prefs import Bias


def _neighbours(X: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    # Row i: the k nearest other rows, nearest first; ties to the lowest index.
    # Exact per-pair distances (cdist) in row chunks; per row only the candidates
    # within the k-th smallest distance are sorted (stable, so ties keep index
    # order). Chunks hold ~16M distances (128 MB, plus the partition copy).
    # ponytail: O(r^2) time; a KD-tree if rare bumps reach ~1e5-1e6 rows.
    out = np.empty((len(X), k), dtype=np.intp)
    rows = max(1, 2**24 // len(X))
    for lo in range(0, len(X), rows):
        d = cdist(X[lo : lo + rows], X)
        d[np.arange(len(d)), np.arange(lo, lo + len(d))] = np.inf
        kth = np.partition(d, k - 1, axis=1)[:, k - 1 : k]
        for i, row in enumerate(d):
            cand = np.flatnonzero(row <= kth[i])
            out[lo + i] = cand[np.argsort(row[cand], kind="stable")[:k]]
    return out


def synthesize(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    bump: Bump,
    c: float,
    time_index: NDArray[np.intp],
    phi: NDArray[np.float64],
    bias: Bias,
    k: int,
    rng: np.random.RandomState,
    *,
    r_quirks: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Synthetic cases for ``bump`` with multiplier ``c > 1`` (SPEC §4.5).

    Column 0 of ``X`` is taken as the most recent lag (``embed``'s layout); under
    ``r_quirks`` it alone weights the target (ADR-0013 amendment).

    Returns ``(X_new, y_new, seeds)``; ``seeds`` are the input rows each case
    was grown from (the resampler places a synthetic case after its seed).
    """
    t_y = bump.idx  # value order (the bump's own order, SPEC §4.2)
    t_t = t_y[np.argsort(time_index[t_y], kind="stable")]  # time order
    r = len(t_y)
    # The order neighbours are searched in, and the order values are read from.
    search = t_y if (bias is None and r_quirks) else t_t
    read = t_y if r_quirks else t_t
    k_eff = r - 1 if r <= k else k
    nn = _neighbours(X[search], k_eff)

    nexs = math.floor(c - 1)
    extra = math.floor(r * (c - 1 - nexs))
    # Every seed position makes nexs cases, then `extra` distinct seeds one more.
    plan = np.concatenate(
        [np.repeat(np.arange(r), nexs), rng.choice(r, size=extra, replace=False)]
    )
    cand = nn[plan]
    if bias is None:  # Alg. 4: uniform among the k
        pick = cand[np.arange(len(plan)), rng.randint(k_eff, size=len(plan))]
    elif bias == "temporal":  # Alg. 9: most recent (largest T_t position)
        pick = cand.max(axis=1)
    else:  # Alg. 13: recency tau relative to the most recent candidate, times phi
        # read in time order; argmax ties -> the nearer neighbour (ADR-0013).
        tau = (cand + 1.0) / (cand.max(axis=1, keepdims=True) + 1.0)
        pick = cand[np.arange(len(plan)), np.argmax(tau * phi[t_t[cand]], axis=1)]
    seed, other = read[plan], read[pick]
    lam = rng.uniform(size=(len(plan), 1))  # one lambda per synthetic case
    x_new = X[seed] + lam * (X[other] - X[seed])
    if r_quirks:  # weights from the most recent lag, column 0 (ADR-0013)
        d1 = np.abs(X[seed, 0] - x_new[:, 0])
        d2 = np.abs(X[other, 0] - x_new[:, 0])
    else:
        d1 = np.linalg.norm(X[seed] - x_new, axis=1)
        d2 = np.linalg.norm(X[other] - x_new, axis=1)
    with np.errstate(invalid="ignore"):  # d1 = d2 = 0 is the midpoint branch
        weighted = (d2 * y[seed] + d1 * y[other]) / (d1 + d2)
    # d1 == d2, including R's NaN case (zero column range): the midpoint.
    y_new = np.where(d1 == d2, (y[seed] + y[other]) / 2, weighted)
    return x_new, y_new, seed
