"""
Port of the parts of the R package `uba` (Rita P. Ribeiro, v0.7.7) used by
the experiments:

    phi.control(y, method="extremes")   -> phi_control(y, method="extremes")
    phi(y, phi.parms)                    -> phi(y, phi_parms)
    loss.control(y)                      -> loss_control(y)
    util(ypred, y, ph, ls, util.control(umetric=..., event.thr=..., beta=...))
                                         -> util(ypred, y, ph, ls, umetric=..., event_thr=..., beta=...)

The relevance function phi is a piecewise cubic Hermite interpolating
polynomial (pchip) with Fritsch-Carlson monotone slopes, evaluated with
linear extrapolation outside the control points (C code: pchip.c, phi.c).
The utility of a prediction follows util.c / bump.c; the metrics P, R and Fm
follow utilMetrics.c.  The `phi` function of the UBL package (which masks the
uba one in the original R session) is the same algorithm (phi.f90).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

PHI_METHODS = ("extremes", "range")
UTIL_METRICS = ("MU", "NMU", "AUCROC", "AUCPR", "MAP11", "BFM", "P", "R", "Fm")
UTIL_TYPES = {"MU": "base", "NMU": "norm", "AUCROC": "rank", "AUCPR": "rank", "MAP11": "rank",
              "BFM": "rank", "P": "rank", "R": "rank", "Fm": "rank"}
DELTA = 0.00001  # utilMetrics.c: value returned when there are no positives


# ---------------------------------------------------------------------------
# R helpers: fivenum / boxplot.stats
# ---------------------------------------------------------------------------
def fivenum(x) -> np.ndarray:
    """R's stats::fivenum (Tukey five number summary)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.full(5, np.nan)
    n4 = math.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n], dtype=float)
    lo = np.floor(d).astype(int) - 1
    hi = np.ceil(d).astype(int) - 1
    return 0.5 * (x[lo] + x[hi])


def boxplot_stats(x, coef: float = 1.5):
    """R's grDevices::boxplot.stats -> (stats, out)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    stats = fivenum(x)
    iqr = stats[3] - stats[1]
    if coef == 0:
        return stats, np.array([], dtype=float)
    out = (x < stats[1] - coef * iqr) | (x > stats[3] + coef * iqr)
    if out.any():
        stats[0] = x[~out].min()
        stats[4] = x[~out].max()
    return stats, x[out]


# ---------------------------------------------------------------------------
# phi.control
# ---------------------------------------------------------------------------
def _phi_extremes(y, extr_type="both", coef=1.5):
    if extr_type not in ("both", "high", "low"):
        raise ValueError("extr_type must be one of both/high/low")
    stats, out = boxplot_stats(y, coef=coef)
    r = (np.min(y), np.max(y))
    pts = []
    if extr_type in ("both", "low") and (out < stats[0]).any():
        pts.append((stats[0], 1.0, 0.0))  # adjL
    else:
        pts.append((r[0], 0.0, 0.0))  # min
    pts.append((stats[2], 0.0, 0.0))  # median
    if extr_type in ("both", "high") and (out > stats[4]).any():
        pts.append((stats[4], 1.0, 0.0))  # adjH
    else:
        pts.append((r[1], 0.0, 0.0))  # max
    return np.asarray(pts, dtype=float)


def _phi_range(y, control_pts):
    if isinstance(control_pts, dict):
        control_pts = np.asarray(control_pts["control_pts"], float).reshape(control_pts["npts"], 3)
    cp = np.asarray(control_pts, dtype=float)
    if cp.ndim != 2 or cp.shape[1] not in (2, 3):
        raise ValueError("The control.pts must be given as a matrix in the form: < x, y, m > or < x, y >")
    npts = cp.shape[0]
    dx = np.diff(cp[:, 0])
    if np.isnan(dx).any() or (dx == 0).any():
        raise ValueError("'x' must be *strictly* increasing (non - NA)")
    if ((cp[:, 1] > 1) | (cp[:, 1] < 0)).any():
        raise ValueError("phi relevance function maps values only in [0,1]")
    cp = cp[np.argsort(cp[:, 0], kind="stable")]
    if cp.shape[1] == 2:
        dx = np.diff(cp[:, 0])
        dy = np.diff(cp[:, 1])
        Sx = dy / dx
        m = np.concatenate([[0.0], (Sx[1:] + Sx[:-1]) / 2, [0.0]])
        cp = np.column_stack([cp, m])
    return cp


def phi_control(y, method: str = "extremes", extr_type: str = "both", coef: float = 1.5,
                control_pts=None) -> dict:
    """R: uba::phi.control(y, method=..., ...).  Returns the `phi.parms` list
    as a dict {method, npts, control_pts} (control_pts flattened row-wise as in R)."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if method == "extremes":
        cp = _phi_extremes(y, extr_type=extr_type, coef=coef)
    elif method == "range":
        cp = _phi_range(y, control_pts)
    else:
        raise ValueError("method must be one of " + ", ".join(PHI_METHODS))
    return {"method": method, "npts": int(cp.shape[0]), "control_pts": cp.ravel().tolist()}


# ---------------------------------------------------------------------------
# pchip (pchip.c)
# ---------------------------------------------------------------------------
def pchip_slope_monoFC(m: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Fritsch & Carlson slope modification (pchip_slope_monoFC in pchip.c)."""
    m = np.array(m, dtype=float, copy=True)
    n = len(m)
    for k in range(n - 1):
        Sk = delta[k]
        k1 = k + 1
        if abs(Sk) == 0:
            m[k] = m[k1] = 0.0
        else:
            alpha = m[k] / Sk
            beta = m[k1] / Sk
            if abs(m[k]) != 0 and alpha < 0:
                m[k] = -m[k]
                alpha = m[k] / Sk
            if abs(m[k1]) != 0 and beta < 0:
                m[k1] = -m[k1]
                beta = m[k1] / Sk
            a2b3 = 2 * alpha + beta - 3
            ab23 = alpha + 2 * beta - 3
            if a2b3 > 0 and ab23 > 0 and alpha * (a2b3 + ab23) < a2b3 * a2b3:
                tauS = 3 * Sk / math.sqrt(alpha * alpha + beta * beta)
                m[k] = tauS * alpha
                m[k1] = tauS * beta
    return m


@dataclass
class HermiteSpline:
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    @property
    def npts(self) -> int:
        return len(self.x)

    @classmethod
    def from_control_points(cls, x, y, m) -> "HermiteSpline":
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.asarray(m, float)
        n = len(x)
        h = x[1:] - x[:-1]
        if (h == 0).any():
            raise ValueError("phi control points must have strictly increasing x "
                             f"(got {x.tolist()}); the target has a degenerate distribution")
        delta = (y[1:] - y[:-1]) / h
        new_m = pchip_slope_monoFC(m, delta)
        c = np.zeros(n)
        d = np.zeros(n)
        c[:-1] = (3 * delta - 2 * new_m[:-1] - new_m[1:]) / h
        d[:-1] = (new_m[:-1] - 2 * delta + new_m[1:]) / (h * h)
        return cls(x=x, a=y.copy(), b=new_m, c=c, d=d)

    def value(self, xval, extrapol: int = 0):
        """pchip_val: returns (yval, yvald, yvaldd) arrays."""
        xv = np.asarray(xval, dtype=float)
        scalar = xv.ndim == 0
        xv = np.atleast_1d(xv)
        n = self.npts
        # R's findInterval: i in 0..n with x[i-1] <= xval < x[i]
        i = np.searchsorted(self.x, xv, side="right")
        outside = (i == 0) | (i == n)
        ii = np.where(i == n, n - 1, i)  # if(i == npts) i--   (for the extrapolation branch)
        # linear extrapolation branch (extrapol == 0 and outside)
        s_lin = xv - self.x[ii]
        y_lin = self.a[ii] + self.b[ii] * s_lin
        yd_lin = self.b[ii]
        # cubic branch: i-- then s = xval - x[i]
        ic = np.clip(i - 1, 0, n - 1)
        s = xv - self.x[ic]
        y_cub = self.a[ic] + s * (self.b[ic] + s * (self.c[ic] + s * self.d[ic]))
        yd_cub = self.b[ic] + s * (2 * self.c[ic] + s * (3 * self.d[ic]))
        ydd_cub = 2 * self.c[ic] + s * 6 * self.d[ic]
        if extrapol == 0:
            yv = np.where(outside, y_lin, y_cub)
            ydv = np.where(outside, yd_lin, yd_cub)
            yddv = np.where(outside, 0.0, ydd_cub)
        else:
            yv, ydv, yddv = y_cub, yd_cub, ydd_cub
        if scalar:
            return float(yv[0]), float(ydv[0]), float(yddv[0])
        return yv, ydv, yddv


def _spline_from_parms(phi_parms: dict) -> HermiteSpline:
    cp = np.asarray(phi_parms["control_pts"], dtype=float).reshape(int(phi_parms["npts"]), 3)
    return HermiteSpline.from_control_points(cp[:, 0], cp[:, 1], cp[:, 2])


def phi(y, phi_parms: dict, only_phi: bool = True):
    """R: phi(y, phi.parms)  (uba) / phi(y, control.parms) (UBL)."""
    H = _spline_from_parms(phi_parms)
    yv, ydv, yddv = H.value(np.asarray(y, dtype=float).ravel(), extrapol=0)
    if only_phi:
        return yv
    return {"y_phi": yv, "yd_phi": ydv, "ydd_phi": yddv}


# ---------------------------------------------------------------------------
# loss.control (loss.R)
# ---------------------------------------------------------------------------
def loss_control(y, ymin=None, ymax=None, tloss=None, epsilon: float = 0.1) -> dict:
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    r = (float(np.min(y)), float(np.max(y)))
    if ymin is None:
        ymin = r[0]
    if ymax is None:
        ymax = r[1]
    if not (ymax - ymin > 0):
        ymin, ymax = r
    if tloss is None:
        # Cherkassky and Ma, 2002
        n = len(y)
        tau = 3
        tL = np.abs(np.mean(y) - y)
        sd = float(np.std(tL, ddof=1)) if n > 1 else 0.0
        tloss = tau * sd * math.sqrt(math.log(n) / n)
    return {"ymin": float(ymin), "ymax": float(ymax), "tloss": float(tloss), "epsilon": float(epsilon)}


# ---------------------------------------------------------------------------
# bumps (bump.c)
# ---------------------------------------------------------------------------
@dataclass
class PhiBumps:
    n: int
    bleft: np.ndarray
    bmax: np.ndarray
    bloss: np.ndarray


def bumps_set(H: HermiteSpline, tloss: float) -> PhiBumps:
    """Port of bumps_set() in bump.c (loss_args[2] is tloss)."""
    npts = H.npts
    size = npts + 2
    bleft = np.zeros(size)
    bmax = np.zeros(size)
    bloss = np.zeros(size)
    bleft[0] = -np.inf
    bmax[0] = -np.inf
    bloss[0] = np.inf
    critical = [i for i in range(npts) if abs(H.b[i]) == 0]
    j = len(critical)
    if j == 0:
        raise ValueError("bumps_set: no critical points in the relevance function")
    Bn = 0
    inBump = 1
    sum_b = H.x[critical[0]]
    nb = 1
    i = 0
    while i < j - 1:
        d1 = H.a[critical[i + 1]] - H.a[critical[i]]
        if d1 == 0:
            sum_b += H.x[critical[i + 1]]
            nb += 1
        else:
            if d1 < 0 and inBump:
                bmax[Bn] = sum_b / nb
                if np.isfinite(bmax[Bn]) and np.isfinite(bleft[Bn]):
                    bloss[Bn] = abs(bmax[Bn] - bleft[Bn])
                inBump = 0
            elif d1 > 0 and (not inBump or not Bn):
                Bn += 1
                bleft[Bn] = sum_b / nb
                if np.isfinite(bmax[Bn - 1]) and np.isfinite(bleft[Bn]):
                    delta = abs(bmax[Bn - 1] - bleft[Bn])
                    if delta < bloss[Bn - 1]:
                        bloss[Bn - 1] = 2 * delta
                    else:
                        bloss[Bn - 1] = 2 * bloss[Bn - 1]
                inBump = 1
            sum_b = H.x[critical[i + 1]]
            nb = 1
        i += 1
    if Bn > 0:
        if inBump:
            bmax[Bn] = sum_b / nb
            if np.isfinite(bmax[Bn]) and np.isfinite(bleft[Bn]):
                bloss[Bn] = 2 * abs(bmax[Bn] - bleft[Bn])
        else:
            Bn += 1
            bleft[Bn] = sum_b / nb
            bmax[Bn] = np.inf
            if np.isfinite(bmax[Bn - 1]) and np.isfinite(bleft[Bn]):
                delta = abs(bmax[Bn - 1] - bleft[Bn])
                if delta < bloss[Bn - 1]:
                    bloss[Bn - 1] = 2 * delta
                else:
                    bloss[Bn - 1] = 2 * bloss[Bn - 1]
        # constant extrapolation -> bloss outside the range of control points
        if not np.isfinite(bmax[0]):
            bloss[0] = bloss[1]
        if not np.isfinite(bmax[Bn]):
            bloss[Bn] = bloss[Bn - 1]
    else:  # standard regression
        bloss[0] = tloss
    Bn += 1
    return PhiBumps(n=Bn, bleft=bleft[:Bn].copy(), bmax=bmax[:Bn].copy(), bloss=bloss[:Bn].copy())


# ---------------------------------------------------------------------------
# utility (util.c)
# ---------------------------------------------------------------------------
def benefcost_lin(y: np.ndarray, ypred: np.ndarray, bumps: PhiBumps):
    """Vectorised port of benefcost_lin(): returns (lb, lc)."""
    y = np.asarray(y, float)
    ypred = np.asarray(ypred, float)
    n = bumps.n
    if n > 1:
        i = np.searchsorted(bumps.bleft, y, side="right")  # findInterval
    else:
        i = np.ones(len(y), dtype=int)
    i = np.where(i > 0, i - 1, i)
    under = ypred <= y

    def _finite_at(arr, idx, valid):
        out = np.full(len(y), np.inf)
        ok = valid & np.isfinite(np.where(valid, arr[np.clip(idx, 0, len(arr) - 1)], np.nan))
        out[ok] = np.abs(y[ok] - arr[idx[ok]])
        return out

    # benefits loss tolerance
    lossA = np.where(under,
                     _finite_at(bumps.bleft, i, (i > 0)),
                     _finite_at(bumps.bleft, i + 1, (i + 1 < n)))
    lb = np.minimum(lossA, bumps.bloss[i])
    # costs loss tolerance
    lossA = np.where(under,
                     _finite_at(bumps.bmax, i - 1, (i > 0)),
                     _finite_at(bumps.bmax, i + 1, (i + 1 < n)))
    lc = np.minimum(lossA, bumps.bloss[i])
    return lb, lc


def util_values(ypred, y, phi_parms: dict, loss_parms: dict, p: float = 0.5, utype: str = "rank"):
    """util_core(): per-prediction utility u, plus phi(y) and phi(ypred)."""
    y = np.asarray(y, dtype=float).ravel()
    ypred = np.asarray(ypred, dtype=float).ravel()
    if len(y) != len(ypred):
        raise ValueError("y and ypred must have the same length")
    H = _spline_from_parms(phi_parms)
    bumps = bumps_set(H, loss_parms["tloss"])
    y_phi = H.value(y)[0]
    ypred_phi = H.value(ypred)[0]
    lb, lc = benefcost_lin(y, ypred, bumps)
    l = np.abs(y - ypred)
    with np.errstate(divide="ignore", invalid="ignore"):
        benef = np.where((lb == 0) | (l > lb), 0.0, 1.0 - l / lb)
        cost = np.where((lc == 0) | (l > lc), 1.0, l / lc)
    jphi = p * y_phi + (1 - p) * ypred_phi
    uv = y_phi * benef - jphi * cost
    if utype == "norm":
        uv = (uv + 1) / 2
    return uv, y_phi, ypred_phi


def _ksum(v) -> float:
    return math.fsum(np.asarray(v, dtype=float).tolist())


def precision(u, ypred_phi, event_thr: float, use_util: bool = True, wt=None) -> float:
    u = np.asarray(u, float)
    ypred_phi = np.asarray(ypred_phi, float)
    wt = np.ones(len(u)) if wt is None else np.asarray(wt, float)
    pos = ypred_phi >= event_thr
    if not pos.any():
        return DELTA
    if use_util:
        num = np.abs(1 + u[pos]) * wt[pos]
        den = np.abs(1 + ypred_phi[pos]) * wt[pos]
    else:
        num = wt[pos]
        den = wt[pos]
    return _ksum(num) / _ksum(den)


def recall(u, y_phi, event_thr: float, use_util: bool = True, wt=None) -> float:
    u = np.asarray(u, float)
    y_phi = np.asarray(y_phi, float)
    wt = np.ones(len(u)) if wt is None else np.asarray(wt, float)
    pos = y_phi >= event_thr
    if not pos.any():
        return DELTA
    if use_util:
        den = np.abs(1 + y_phi[pos]) * wt[pos]
        num = np.abs(1 + u[pos]) * wt[pos]
    else:
        den = wt[pos]
        num = wt[pos]
    return _ksum(num) / _ksum(den)


def harmonic_mean(x: float, y: float, beta: float) -> float:
    if abs(x) == 0 or abs(y) == 0:
        return 0.0
    return ((beta ** 2 + 1) * x * y) / (beta ** 2 * x + y)


def fmeasure(u, y_phi, ypred_phi, event_thr: float, beta: float = 1.0, use_util: bool = True) -> float:
    prec = precision(u, ypred_phi, event_thr, use_util)
    rec = recall(u, y_phi, event_thr, use_util)
    return harmonic_mean(prec, rec, beta)


def util_control(umetric: str = "MU", p: float = 0.5, Bmax: float = 1.0, event_thr: float = 1.0,
                 beta: float = 1.0, use_util: bool = True, **kw) -> dict:
    """R: util.control(umetric=..., event.thr=..., beta=...)."""
    if umetric not in UTIL_METRICS:
        raise ValueError(f"unknown umetric {umetric}")
    utype = UTIL_TYPES[umetric]
    if not (0 <= p <= 1):
        p = 0.5
    if utype == "rank" and event_thr == 0:
        event_thr = 0.01
    if not (0 <= event_thr <= 1):
        event_thr = 1.0
    return {"umetric": umetric, "utype": utype, "use_util": use_util, "p": p, "Bmax": Bmax,
            "event_thr": event_thr, "beta": beta}


def util(ypred, y, phi_parms: dict, loss_parms: dict, util_parms: dict | None = None,
         return_uv: bool = False, **kw) -> float | np.ndarray:
    """R: util(ypred, y, phi.parms, loss.parms, util.parms, return.uv)."""
    up = util_control(**({} if util_parms is None else util_parms), **kw) if (util_parms is None or kw) else util_parms
    u, y_phi, ypred_phi = util_values(ypred, y, phi_parms, loss_parms, p=up["p"], utype=up["utype"])
    if return_uv:
        return u
    m = up["umetric"]
    if m in ("MU", "NMU"):
        return float(np.mean(u))
    if m == "P":
        return precision(u, ypred_phi, up["event_thr"], up["use_util"])
    if m == "R":
        return recall(u, y_phi, up["event_thr"], up["use_util"])
    if m == "Fm":
        return fmeasure(u, y_phi, ypred_phi, up["event_thr"], up["beta"], up["use_util"])
    raise NotImplementedError(f"utility metric {m} is not ported (not used by the experiments)")


class UtilityEvaluator:
    """Convenience: compute u once and derive several metrics (eval.stats)."""

    def __init__(self, phi_parms: dict, loss_parms: dict, p: float = 0.5):
        self.phi_parms = phi_parms
        self.loss_parms = loss_parms
        self.p = p

    def metrics(self, ypred, y, event_thr: float = 0.9, betas: Sequence[float] = (0.5, 1, 2)) -> dict:
        u, y_phi, ypred_phi = util_values(ypred, y, self.phi_parms, self.loss_parms, p=self.p)
        prec = precision(u, ypred_phi, event_thr)
        rec = recall(u, y_phi, event_thr)
        out = {"prec": prec, "rec": rec}
        for b in betas:
            key = "F1" if b == 1 else f"F{b}".replace("0.5", "05")
            out[key] = harmonic_mean(prec, rec, b)
        return out
