"""
Resampling strategies for imbalanced regression on time series.

Port of the nine functions defined in Exps.R (P. Branco, 2016):

    randUnderRegressB / randUnderRegressT / randUnderRegressTPhi
    randOverRegressB  / randOverRegressT  / randOverRegressTPhi
    smoteRegressB     / smoteRegressT     / smoteRegressTPhi

Suffixes: B = the original (bias-free) strategy, T = temporal bias (more
recent cases have a higher probability of being selected / used as
neighbours), TPhi = temporal bias weighted by the relevance phi.

Conventions
-----------
* `data` is a pandas DataFrame whose index is a DatetimeIndex (the R row names
  are timestamps; time ordering uses this index).
* The target column (R: form[[2]]) is given by `tgt` (name or position);
  it defaults to the last column, as in the experiments (V10 ~ .).
* `rng` is a numpy Generator (or seed) replacing R's global RNG.
* `C_perc` follows R: "balance", "extreme" or a list of percentages (one per
  bump). For the SMOTE variants a dict {"un": u, "ov": o} applies u to every
  normal bump and o to every rare bump (Algorithm 5 of the article), which is
  what the u/o parameters of the search (OptParmsSearch.R) mean.

Notes on the R source (see README):
* smote.exsRegressT / smote.exsRegressTPhi build the numeric matrix T *before*
  re-ordering the data by time but compute the neighbours *after*, so the
  R code mixes row positions of two different orderings. The default
  (`r_index_quirk=True`) reproduces this exactly, because it is what the
  published results were computed with; `r_index_quirk=False` builds the
  matrix after the temporal re-ordering (the evident intent).
* In smote.exsRegress* the target of a synthetic case is a weighted average
  whose weights are computed inside a loop that overwrites (instead of
  accumulating) d1/d2; only the last non-target column matters. This is
  reproduced as is (it is the same in DMwR/UBL).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .uba import phi, phi_control

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _tgt_pos(data: pd.DataFrame, tgt) -> int:
    if tgt is None:
        return data.shape[1] - 1
    if isinstance(tgt, (int, np.integer)):
        return int(tgt)
    return list(data.columns).index(tgt)


def _relevance(y: np.ndarray, rel):
    if isinstance(rel, np.ndarray) and rel.ndim == 2:
        return phi_control(y, method="range", control_pts=rel)
    if isinstance(rel, dict):
        return rel
    if isinstance(rel, str) and rel == "auto":
        return phi_control(y, method="extremes")
    raise ValueError("future work!")


def _bumps(y: np.ndarray, pc: dict, thr_rel: float, crossing: str):
    """Split the sorted target into "bumps" (classes).

    Returns (obs_ind, imp): obs_ind is a list of arrays with the positions
    (into `y`) of the cases in each bump, in increasing order of y;
    imp is the mean relevance of each bump.
    crossing = "sign": under-sampling and SMOTE rule
               (relevance > thr negated; boundary where the product < 0)
    crossing = "thr":  over-sampling rule (boundary where >= thr changes)
    """
    order = np.argsort(y, kind="stable")
    s_y = y[order]
    temp = phi(s_y, pc)
    if not (temp < 1).any():
        raise ValueError("All the points have relevance 1. Please, redefine your relevance function!")
    if not (temp > 0).any():
        raise ValueError("All the points have relevance 0. Please, redefine your relevance function!")
    if crossing == "sign":
        t = temp.copy()
        t[temp > thr_rel] = -t[temp > thr_rel]
        bumps = np.where(t[:-1] * t[1:] < 0)[0]
    elif crossing == "thr":
        hi = temp >= thr_rel
        bumps = np.where(hi[:-1] != hi[1:])[0]
    else:
        raise ValueError(crossing)
    obs_ind = []
    last = 0
    for b in bumps:
        obs_ind.append(order[last:b + 1])
        last = b + 1
    obs_ind.append(order[last:])
    imp = [float(np.mean(phi(y[o], pc))) for o in obs_ind]
    return obs_ind, imp


def _time_order(data: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    """positions `pos` re-ordered chronologically (R: order(rownames))."""
    times = data.index[pos]
    return pos[np.argsort(times.values, kind="stable")]


def _sample(rng, pool: np.ndarray, size: float, replace: bool, prob=None) -> np.ndarray:
    """R's sample(x, size, replace, prob) on a pool of positions.
    size is truncated to an integer (as R does)."""
    size = int(size)
    if size <= 0:
        return np.array([], dtype=int)
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        if (prob < 0).any() or not np.isfinite(prob).all():
            raise ValueError("invalid sampling probabilities")
        if not replace and (prob > 0).sum() < size:
            raise ValueError("too few positive probabilities")
        prob = prob / prob.sum()
    if not replace and size > len(pool):
        raise ValueError("cannot take a sample larger than the population when 'replace = FALSE'")
    return rng.choice(pool, size=size, replace=replace, p=prob)


def _r_round(x: float, digits: int) -> float:
    return float(round(x, digits))


def _under_percs(C_perc, obs_ind, und, ove):
    if isinstance(C_perc, (list, tuple)):
        if len(und) > 1 and len(C_perc) == 1:
            return [C_perc[0]] * len(und)
        if len(und) > len(C_perc) and len(C_perc) > 1:
            raise ValueError("The number of under-sampling percentages must be equal to the number of bumps below the threshold defined!")
        if len(und) < len(C_perc):
            raise ValueError("the number of under-sampling percentages must be at most the number of bumps below the threshold defined!")
        return list(C_perc)
    if C_perc == "balance":
        B = sum(len(obs_ind[k]) for k in ove)
        obj = B / len(und)
        return [_r_round(obj / len(obs_ind[k]), 5) for k in und]
    if C_perc == "extreme":
        Bove = sum(len(obs_ind[k]) for k in ove) / len(ove)
        return [_r_round((Bove ** 2 / len(obs_ind[k])) / len(obs_ind[k]), 5) for k in und]
    raise ValueError("C_perc must be 'balance', 'extreme' or a list of percentages")


def _over_percs(C_perc, obs_ind, und, ove):
    if isinstance(C_perc, (list, tuple)):
        if len(ove) > 1 and len(C_perc) == 1:
            return [C_perc[0]] * len(ove)
        if len(ove) > len(C_perc) and len(C_perc) > 1:
            raise ValueError("The number of over-sampling percentages must be equal to the number of bumps above the threshold defined!")
        if len(ove) < len(C_perc):
            raise ValueError("The number of over-sampling percentages must be at most the number of bumps above the threshold defined!")
        return list(C_perc)
    if C_perc == "balance":
        B = sum(len(obs_ind[k]) for k in und)
        obj = B / len(ove)
        return [_r_round(obj / len(obs_ind[k]), 5) for k in ove]
    if C_perc == "extreme":
        Bund = sum(len(obs_ind[k]) for k in und) / len(und)
        return [_r_round((Bund ** 2 / len(obs_ind[k])) / len(obs_ind[k]), 5) for k in ove]
    raise ValueError("C_perc must be 'balance', 'extreme' or a list of percentages")


# ---------------------------------------------------------------------------
# Random under-sampling
# ---------------------------------------------------------------------------
def _rand_under(data: pd.DataFrame, tgt, rel, thr_rel, C_perc, repl, rng, variant: str) -> pd.DataFrame:
    rng = _rng(rng)
    t = _tgt_pos(data, tgt)
    y = data.iloc[:, t].to_numpy(dtype=float)
    pc = _relevance(y, rel)
    obs_ind, imp = _bumps(y, pc, thr_rel, "sign")
    und = [k for k, v in enumerate(imp) if v < thr_rel]
    ove = [k for k, v in enumerate(imp) if v > thr_rel]
    if not ove or not und:
        raise ValueError("randUnderRegress: no bumps above/below the relevance threshold")
    parts = [data.iloc[obs_ind[k]] for k in ove]  # start with the rare "classes"
    C = _under_percs(C_perc, obs_ind, und, ove)
    for j, k in enumerate(und):
        o = obs_ind[k]
        if variant == "B":
            if len(o) == 1 or C[j] * len(o) > len(o):
                sel = o
            else:
                sel = _sample(rng, o, C[j] * len(o), repl)
        else:
            so = _time_order(data, o)  # sdata: examples of the bump ordered by time
            r = len(so)
            probs = np.arange(1, r + 1) / r  # more recent -> higher probability
            if variant == "TPhi":
                probs = probs * phi(data.iloc[so, t].to_numpy(dtype=float), pc)
            if r == 1 or C[j] * r > r:
                sel = so
            else:
                sel = _sample(rng, so, C[j] * r, repl, prob=probs)
        parts.append(data.iloc[sel])
    newdata = pd.concat(parts, axis=0)
    if variant != "B":
        newdata = newdata.iloc[np.argsort(newdata.index.values, kind="stable")]
    return newdata


def rand_under_regress_B(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=False, rng=None):
    """R: randUnderRegressB(form, data, rel, thr.rel, C.perc, repl)"""
    return _rand_under(data, tgt, rel, thr_rel, C_perc, repl, rng, "B")


def rand_under_regress_T(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=False, rng=None):
    """R: randUnderRegressT  (probability of keeping a case grows with recency)"""
    return _rand_under(data, tgt, rel, thr_rel, C_perc, repl, rng, "T")


def rand_under_regress_TPhi(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=False, rng=None):
    """R: randUnderRegressTPhi  (recency x relevance)"""
    return _rand_under(data, tgt, rel, thr_rel, C_perc, repl, rng, "TPhi")


# ---------------------------------------------------------------------------
# Random over-sampling
# ---------------------------------------------------------------------------
def _rand_over(data: pd.DataFrame, tgt, rel, thr_rel, C_perc, repl, rng, variant: str) -> pd.DataFrame:
    rng = _rng(rng)
    if isinstance(C_perc, (list, tuple)) and any(c < 1 for c in C_perc):
        raise ValueError("The over-sampling percentages provided in parameter C.perc can not be lower than 1!")
    t = _tgt_pos(data, tgt)
    y = data.iloc[:, t].to_numpy(dtype=float)
    pc = _relevance(y, rel)
    obs_ind, imp = _bumps(y, pc, thr_rel, "thr")
    ove = [k for k, v in enumerate(imp) if v >= thr_rel]
    und = [k for k, v in enumerate(imp) if v < thr_rel]
    if not ove or not und:
        raise ValueError("randOverRegress: no bumps above/below the relevance threshold")
    C = _over_percs(C_perc, obs_ind, und, ove)
    parts = [data]
    for j, k in enumerate(ove):
        o = obs_ind[k]
        if variant == "B":
            sel = _sample(rng, o, C[j] * len(o), repl)
        else:
            so = _time_order(data, o)
            r = len(so)
            probs = np.arange(1, r + 1) / r
            if variant == "TPhi":
                probs = probs * phi(data.iloc[so, t].to_numpy(dtype=float), pc)
            sel = _sample(rng, so, C[j] * r, repl, prob=probs)
        parts.append(data.iloc[sel])
    return pd.concat(parts, axis=0)


def rand_over_regress_B(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=True, rng=None):
    """R: randOverRegressB(form, dat, rel, thr.rel, C.perc, repl)"""
    return _rand_over(data, tgt, rel, thr_rel, C_perc, repl, rng, "B")


def rand_over_regress_T(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=True, rng=None):
    """R: randOverRegressT"""
    return _rand_over(data, tgt, rel, thr_rel, C_perc, repl, rng, "T")


def rand_over_regress_TPhi(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", repl=True, rng=None):
    """R: randOverRegressTPhi"""
    return _rand_over(data, tgt, rel, thr_rel, C_perc, repl, rng, "TPhi")


# ---------------------------------------------------------------------------
# SMOTE for regression
# ---------------------------------------------------------------------------
_DIST_CODES = {"Chebyshev": "chebyshev", "Manhattan": "cityblock", "Euclidean": "euclidean",
               "Canberra": "canberra", "p-norm": "minkowski"}


def neighbours(tgt: int, T: np.ndarray, dist: str = "Euclidean", p: int = 2, k: int = 5) -> np.ndarray:
    """Port of UBL::neighbours for numeric data.

    Returns an (n, k) array with the positions of the k nearest neighbours of
    each row (the row itself excluded; ties broken by the lowest position, as
    in the Fortran code). Distances use all columns except `tgt`, unscaled.
    """
    if p < 1:
        raise ValueError("The parameter p must be >=1!")
    if dist not in _DIST_CODES:
        raise ValueError("Distance measure not available!")
    T = np.asarray(T, dtype=float)
    X = np.delete(T, tgt, axis=1)
    if dist == "p-norm":
        D = cdist(X, X, metric="minkowski", p=p)
    else:
        D = cdist(X, X, metric=_DIST_CODES[dist])
    np.fill_diagonal(D, np.inf)
    n = D.shape[0]
    if k > n - 1:
        raise ValueError("k must be smaller than the number of cases")
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def _smote_exs(bump: pd.DataFrame, tgt: int, N: float, k: int, dist: str, p: int, variant: str,
               pc: dict, rng: np.random.Generator, r_index_quirk: bool = True) -> pd.DataFrame:
    """smote.exsRegressB / smote.exsRegressT / smote.exsRegressTPhi.

    Generates (N-1)*nrow(bump) synthetic cases (plus `extra` for the
    fractional part) with rare values on the target.

    For the T/TPhi variants the R code fills the numeric matrix `T` from the
    bump *as received* (rows in increasing order of the target) and only then
    re-orders the data by time to compute the neighbours, so the neighbour
    indices (time order) are applied to rows of `T` (target order).
    `r_index_quirk=True` (default) reproduces that exactly, which is what the
    published results were computed with; `r_index_quirk=False` builds `T`
    from the time-ordered data (the evident intent).
    """
    T = bump.to_numpy(dtype=float)
    if variant in ("T", "TPhi"):
        # order by time so that the most recent examples have the higher line numbers
        T_time = bump.iloc[np.argsort(bump.index.values, kind="stable")].to_numpy(dtype=float)
        if not r_index_quirk:
            T = T_time
    else:
        T_time = T
    nT, nC = T.shape
    ranges = T.max(axis=0) - T.min(axis=0)
    kNNs = neighbours(tgt, T_time, dist, p, k)
    nexs = int(N - 1)
    extra = int(nT * (N - 1 - nexs))
    idx = _sample(rng, np.arange(nT), extra, replace=False)
    new = np.empty((nexs * nT + extra, nC), dtype=float)
    non_tgt = np.array([c for c in range(nC) if c != tgt])
    x_last = non_tgt[-1]  # the R loop overwrites d1/d2, so only the last column counts

    def pick_neighbour(i: int) -> int:
        row = kNNs[i]
        if variant == "B":
            return int(row[rng.integers(k)])  # sample(1:k, 1)
        if variant == "T":
            return int(row[np.argmax(row)])  # the most recent neighbour
        # TPhi: the neighbour more recent and with higher phi (R uses 1-based positions;
        # the relevance is taken from the time-ordered data, as in R)
        y_rel = phi(T_time[row, tgt], pc)
        pos_eval = ((row + 1) / (row.max() + 1)) * y_rel
        return int(row[np.argmax(pos_eval)])

    def generate(dest: int, i: int):
        nn = pick_neighbour(i)
        difs = T[nn, non_tgt] - T[i, non_tgt]
        new[dest, non_tgt] = T[i, non_tgt] + rng.random() * difs
        if ranges[x_last] > 0:
            d1 = abs(T[i, x_last] - new[dest, x_last]) / ranges[x_last]
            d2 = abs(T[nn, x_last] - new[dest, x_last]) / ranges[x_last]
        else:  # R would produce NaN (0/0) here; treat as equidistant
            d1 = d2 = 0.0
        if d1 == d2:
            new[dest, tgt] = (T[i, tgt] + T[nn, tgt]) / 2
        else:
            new[dest, tgt] = (d2 * T[i, tgt] + d1 * T[nn, tgt]) / (d1 + d2)

    if nexs:
        for i in range(nT):
            for n_ in range(nexs):
                generate(i * nexs + n_, i)
    if extra:
        for count, i in enumerate(idx):
            generate(nexs * nT + count, int(i))
    out = pd.DataFrame(new, columns=bump.columns,
                       index=pd.DatetimeIndex([pd.NaT] * new.shape[0], name=bump.index.name))
    return out


def _smote_regress(data: pd.DataFrame, tgt, rel, thr_rel, C_perc, k, repl, dist, p, rng,
                   variant: str, r_index_quirk: bool = True) -> pd.DataFrame:
    rng = _rng(rng)
    if data.isna().to_numpy().any():
        raise ValueError("The data set provided contains NA values!")
    t = _tgt_pos(data, tgt)
    ncol = data.shape[1]
    cols = list(range(ncol))
    if t < ncol - 1:  # move the target to the last column
        cols[t], cols[ncol - 1] = cols[ncol - 1], cols[t]
        data = data.iloc[:, cols]
        t = ncol - 1
    if thr_rel is None or (isinstance(thr_rel, float) and np.isnan(thr_rel)):
        raise ValueError("Future work!")
    y = data.iloc[:, t].to_numpy(dtype=float)
    pc = _relevance(y, rel)
    obs_ind, imp = _bumps(y, pc, thr_rel, "sign")
    nbump = len(obs_ind)
    if isinstance(C_perc, dict):
        # Algorithm 5 of the article: the under-sampling percentage u applies to
        # every bin of normal cases and the over-sampling percentage o to every
        # bin of rare cases (the R code only accepts one value per bump).
        un = C_perc.get("un", 1)
        ov = C_perc.get("ov", 1)
        C = [ov if v > thr_rel else un for v in imp]
    elif isinstance(C_perc, (list, tuple)):
        if len(C_perc) != nbump:
            raise ValueError("The percentages provided must be the same length as the number of bumps!")
        C = list(C_perc)
    elif C_perc == "balance":
        B = round(data.shape[0] / nbump)
        C = [B / len(o) for o in obs_ind]
    elif C_perc == "extreme":
        B = round(data.shape[0] / nbump)
        sizes = np.array([len(o) for o in obs_ind], dtype=float)
        rescale = nbump * B / np.sum(B ** 2 / sizes)
        obj = np.round((B ** 2 / sizes) * rescale, 2)
        C = list(np.round(obj / sizes, 1))
    else:
        raise ValueError("C_perc must be 'balance', 'extreme' or a list")
    parts = []
    for i, o in enumerate(obs_ind):
        if len(o) == 1 or C[i] == 1:
            parts.append(data.iloc[o])
        elif C[i] > 1:
            kk = len(o) - 1 if len(o) <= k else k
            new_exs = _smote_exs(data.iloc[o], t, C[i], kk, dist, p, variant, pc, rng, r_index_quirk)
            parts.append(new_exs)
            parts.append(data.iloc[o])
        elif C[i] < 1:
            if variant == "B":
                sel = _sample(rng, o, int(C[i] * len(o)), repl)
            else:
                so = _time_order(data, o)
                r = len(so)
                probs = np.arange(1, r + 1) / r
                if variant == "TPhi":
                    probs = probs * phi(data.iloc[so, t].to_numpy(dtype=float), pc)
                sel = _sample(rng, so, C[i] * r, repl, prob=probs)
            parts.append(data.iloc[sel])
    newdata = pd.concat(parts, axis=0)
    if cols != list(range(ncol)):
        newdata = newdata.iloc[:, cols]
    return newdata


def smote_regress_B(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", k=5, repl=False,
                    dist="Euclidean", p=2, rng=None):
    """R: smoteRegressB(form, data, rel, thr.rel, C.perc, k, repl, dist, p)"""
    return _smote_regress(data, tgt, rel, thr_rel, C_perc, k, repl, dist, p, rng, "B")


def smote_regress_T(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", k=5, repl=False,
                    dist="Euclidean", p=2, rng=None, r_index_quirk=True):
    """R: smoteRegressT (synthetic cases use the most recent neighbour;
    under-sampling biased towards recent cases). See _smote_exs for r_index_quirk."""
    return _smote_regress(data, tgt, rel, thr_rel, C_perc, k, repl, dist, p, rng, "T", r_index_quirk)


def smote_regress_TPhi(data, tgt=None, rel="auto", thr_rel=0.5, C_perc="balance", k=5, repl=False,
                       dist="Euclidean", p=2, rng=None, r_index_quirk=True):
    """R: smoteRegressTPhi (recency x relevance). See _smote_exs for r_index_quirk."""
    return _smote_regress(data, tgt, rel, thr_rel, C_perc, k, repl, dist, p, rng, "TPhi", r_index_quirk)


RESAMPLERS = {
    "UNDERB": rand_under_regress_B, "UNDERT": rand_under_regress_T, "UNDERTPhi": rand_under_regress_TPhi,
    "OVERB": rand_over_regress_B, "OVERT": rand_over_regress_T, "OVERTPhi": rand_over_regress_TPhi,
    "SMOTEB": smote_regress_B, "SMOTET": smote_regress_T, "SMOTETPhi": smote_regress_TPhi,
}
