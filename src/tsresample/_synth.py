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
    # Row i: the k nearest other rows, nearest first; ties to the lowest index
    # (stable sort over row order). cdist is exact per pair, so ties stay ties.
    # ponytail: O(r^2) time in 1024-row chunks; a KD-tree if bumps reach ~1e5 rows.
    out = np.empty((len(X), k), dtype=np.intp)
    for lo in range(0, len(X), 1024):
        d = cdist(X[lo : lo + 1024], X)
        d[np.arange(len(d)), np.arange(lo, lo + len(d))] = np.inf
        out[lo : lo + 1024] = np.argsort(d, axis=1, kind="stable")[:, :k]
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

    def choose(s: int) -> int:
        cand = nn[s]
        if bias is None:  # Alg. 4: uniform
            return int(cand[rng.randint(k_eff)])
        if bias == "temporal":  # Alg. 9: most recent (largest T_t position)
            return int(cand.max())
        # Alg. 13: recency tau relative to the most recent candidate, times phi
        # of the candidate read in time order; argmax ties -> nearer (ADR-0013).
        tau = (cand + 1.0) / (cand.max() + 1.0)
        return int(cand[np.argmax(tau * phi[t_t[cand]])])

    nexs = math.floor(c - 1)
    extra = math.floor(r * (c - 1 - nexs))
    plan = [s for s in range(r) for _ in range(nexs)]
    plan += rng.choice(r, size=extra, replace=False).tolist()
    xs, ys, seeds = [], [], []
    for s in plan:
        seed, other = read[s], read[choose(s)]
        lam = rng.uniform()
        x_new = X[seed] + lam * (X[other] - X[seed])
        if r_quirks:  # weights from the most recent lag, column 0 (ADR-0013)
            d1 = abs(X[seed, 0] - x_new[0])
            d2 = abs(X[other, 0] - x_new[0])
        else:
            d1 = float(np.linalg.norm(X[seed] - x_new))
            d2 = float(np.linalg.norm(X[other] - x_new))
        if d1 == d2:  # also R's NaN case (zero column range): midpoint
            y_new = (y[seed] + y[other]) / 2
        else:
            y_new = (d2 * y[seed] + d1 * y[other]) / (d1 + d2)
        xs.append(x_new)
        ys.append(y_new)
        seeds.append(seed)
    return (
        np.array(xs, dtype=np.float64).reshape(-1, X.shape[1]),
        np.array(ys, dtype=np.float64),
        np.array(seeds, dtype=np.intp),
    )
