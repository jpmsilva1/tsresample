"""
Multivariate Adaptive Regression Splines: a port of the R package `earth`.

Stand-in for `earth(form, train, nk, degree, thresh)` used by the mc.mars*
workflows (py-earth does not build on current Python versions).  This is a
line-by-line port of the forward pass of earth.c (earth 5.3.6, Stephen
Milborrow, derived from Friedman's MARS) and of the pruning pass of
earth.fit.R, so that the same data give the same model:

* knots are taken from the cases in the order of the predictor (not from the
  unique values): the top and bottom `endspan` cases are never knots
  (endspan = 3 + log2(20) + log2(p), tripled for interaction terms), and only
  every `minspan`-th case with a non-zero parent term is a candidate
  (minspan = (-log(-log(0.95)) + log(p * n_parent)) / (2.5 log 2));
* for each parent term and predictor the candidate "linear" term
  parent * x is evaluated first; a knot only replaces it when it reduces the
  RSS more, and the pair of hinges is rejected when its orthogonalised
  column is nearly collinear with the model (tolerance 0.01 for the first 15
  terms, 1e-5 afterwards) or when its RSS reduction exceeds
  min(1.01 * RSS, 10 * previous reduction);
* the candidate linear column is dropped (and only one hinge is added) when
  its sum of squares after orthogonalisation is <= 0.01 (`MIN_BX_SOS`), an
  absolute threshold, so that the scale of the predictors matters, as in
  earth;
* a term is added as a single hinge when the "form" already exists
  (`GetNewFormFlag`), the parents are visited in the Fast MARS queue order
  (`fast.k`, `fast.beta`), a linear term consumes two slots of `nk`, and the
  pass stops when RSq changes by less than `thresh`, RSq >= 1 - thresh,
  GRSq < -10 or nk is reached;
* backward pruning over the linearly independent terms, choosing the subset
  with the lowest GCV, GCV = RSS/n / (1 - (nterms + penalty*(nterms-1)/2)/n)^2,
  penalty = 3 if degree > 1 else 2.

The port was checked against earth 5.3.6 on the Monte Carlo windows of the
experiments: same terms, knots and coefficients.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Factor = Tuple[int, float, int]  # (variable, knot, sign)  sign: +1 max(0,x-t), -1 max(0,t-x), 0 linear

ALMOST_ZERO = 1e-10
MIN_BX_SOS = 0.01
MIN_GRSQ = -10.0
QR_TOL = 1e-8
MAX_DEGREE = 100


def _maybe_zero(x: float) -> float:
    return 0.0 if -ALMOST_ZERO < x < ALMOST_ZERO else x


def _gcv(nterms: int, n: int, rss: float, penalty: float) -> float:
    if penalty == -1:
        cost = 0.0
    else:
        cost = (nterms + penalty * (nterms - 1) / 2.0) / n
    return math.inf if cost >= 1 else rss / (n * (1 - cost) ** 2)


class MARS:
    """earth(x, y, nk, degree, thresh, penalty, minspan, endspan, fast.k,
    fast.beta, newvar.penalty, Adjust.endspan, Auto.linpreds)"""

    def __init__(self, nk: int = 21, degree: int = 1, thresh: float = 0.001, penalty: float | None = None,
                 minspan: int = 0, endspan: int = 0, fast_k: int = 20, fast_beta: float = 1.0,
                 newvar_penalty: float = 0.0, adjust_endspan: float = 2.0, auto_linpreds: bool = True):
        self.nk = int(nk)
        self.degree = int(degree)
        self.thresh = float(thresh)
        self.penalty = (3.0 if self.degree > 1 else 2.0) if penalty is None else float(penalty)
        self.minspan = int(minspan)
        self.endspan = int(endspan)
        self.fast_k = int(fast_k)
        self.fast_beta = float(fast_beta)
        self.newvar_penalty = float(newvar_penalty)
        self.adjust_endspan = float(adjust_endspan)
        self.auto_linpreds = bool(auto_linpreds)
        self.terms_: List[List[Factor]] = []
        self.coef_: np.ndarray | None = None

    # ------------------------------------------------------------------ spans
    def _end_span(self, p: int, degree: int, n: int) -> int:
        if self.endspan > 0:
            e = self.endspan
        elif self.endspan == 0:
            e = int(7.32193 + math.log(p) / 0.69315)  # 3 + log2(20) + log2(p)
        else:
            raise ValueError("endspan < 0")
        if degree >= 2:
            e += int(self.adjust_endspan * e + 0.5)
        if e > n // 2 - 1:
            e = n // 2 - 1
        return max(1, e)

    def _span_params(self, n: int, p: int, degree: int, parent: int):
        end = self._end_span(p, degree, n)
        if self.minspan < 0:  # negative minspan = number of knots
            ms = int(math.ceil(n / (1.0 - self.minspan)))
            start = ms
            while start < end:
                start += ms
            start -= 1
            start = max(1, start)
        else:
            if self.minspan > 0:
                ms = self.minspan
            else:
                nused = int(np.count_nonzero(self._bx[:, parent] > 0))
                ms = int((2.9702 + math.log(p * nused)) / 1.7329) if nused > 0 else 1
            ms = max(1, ms)
            avail = max(0, n - 2 * end)
            start = avail // 2
            if avail > ms:
                ndiv = avail // ms
                start = ms // 2 if avail == ndiv * ms else (avail - ndiv * ms) // 2
            start = max(1, end + start)
        return ms, end, start

    # ------------------------------------------------------------ orthogonal
    def _orthog_residuals(self, col: np.ndarray, nterms: int) -> np.ndarray:
        """OrthogResiduals: sequential projection on the used orthonormal columns."""
        r = col.astype(float).copy()
        for t in range(nterms):
            if self._full[t]:
                q = self._bxorth[:, t]
                r -= (q @ r) * q
        return r

    def _init_bxorth_col(self, col: np.ndarray, t: int, thresh: float) -> bool:
        """InitBxOrthCol: column t of bxorth = normalised residuals of `col` on
        the lower columns. Returns False (and zeroes the column) when the sum
        of squares is <= thresh."""
        n = len(col)
        if t == 0:
            self._bxorth[:, 0] = 1 / math.sqrt(n)
            self._bxorth_mean[0] = 1 / math.sqrt(n)
            return True
        if t == 1:
            r = col - col.mean()
        else:
            r = self._orthog_residuals(col, t)
        sos = float(r @ r)
        good = sos > MIN_BX_SOS
        if sos > thresh:
            self._bxorth_mean[t] = float(r.mean())  # as in earth: the mean before normalising
            self._bxorth[:, t] = r / math.sqrt(sos)
        else:
            self._bxorth_mean[t] = 0.0
            self._bxorth[:, t] = 0.0
        return good

    # ------------------------------------------------------------- fast MARS
    def _init_q(self, nmax: int):
        self._q_nterms = np.full(nmax, -99, dtype=int)
        self._q_delta = np.full(nmax, -1.0)
        self._nq = 0
        self._sorted_q: List[int] = []

    def _add_term_to_q(self, iterm: int, nterms: int, delta: float, sort: bool):
        i = self._nq
        self._q_nterms[i] = nterms
        self._q_delta[i] = max(self._q_delta[iterm], delta)
        self._nq += 1
        if sort:
            idx = list(range(self._nq))
            idx.sort(key=lambda j: (-self._q_delta[j], j))
            if self.fast_beta > 0:
                aged = {j: rank + self.fast_beta * (nterms - self._q_nterms[j]) for rank, j in enumerate(idx)}
                idx.sort(key=lambda j: (aged[j], -self._q_delta[j], j))
            self._sorted_q = idx

    # -------------------------------------------------------------- new form
    def _new_form_flag(self, pred: int, parent: int, nterms: int) -> bool:
        dirs = self._dirs
        is_new = True
        for i in range(1, nterms):
            if self._full[i]:
                is_new = False
                if dirs[i, pred] == 0:
                    return True
                for j in range(dirs.shape[1]):
                    if j != pred and (dirs[i, j] != 0) != (dirs[parent, j] != 0):
                        return True
        return is_new

    # -------------------------------------------------------------- find knot
    def _find_knot(self, parent: int, pred: int, inew: int, Qc: np.ndarray, ycbo: np.ndarray,
                   unadj_lin: float, adj: float, pair_delta: float, max_legal: float,
                   start: int, ms: int, end: int):
        """FindKnot: scan the cases of `pred` in descending order, evaluating
        the RSS reduction of the hinge pair (parent, pred, knot) for every
        admissible knot.  Returns (best case index in the order, best delta)."""
        X, n = self._X, self._n
        order = self._xorder[:, pred]
        xs = X[order, pred]
        c = self._bx[order, parent]
        yq = self._yc[order]
        rev = slice(None, None, -1)
        x1 = xs[rev][:-1]
        x0 = xs[rev][1:]
        bx1 = c[rev][:-1]
        y1 = yq[rev][:-1]
        Qr = Qc[order][rev][:-1]
        xd = x1 - x0
        cov_sx = np.cumsum(Qr * bx1[:, None], axis=0)
        cov_col = np.cumsum(xd[:, None] * cov_sx, axis=0)
        bx_sum = np.cumsum(bx1)
        bxsq = bx1 * bx1
        bxsq_sum = np.cumsum(bxsq)
        bxx_sum = np.cumsum(bx1 * x1)
        bxsqx_sum = np.cumsum(bxsq * x1)
        st = bxx_sum - bx_sum * x0
        su = np.concatenate([[0.0], st[:-1]])
        cov_new = np.cumsum(xd * (2 * bxsqx_sum - bxsq_sum * (x0 + x1)) + (su * su - st * st) / n)
        ybx_sum = np.cumsum(y1 * bx1)
        ycbo_new = np.cumsum(xd * ybx_sum)
        smax = n - 2 - end  # i = n-2-s >= end
        if smax < 0:
            return -1, pair_delta
        mask = (bx1 > 0) & (cov_new > 0)
        mask[smax + 1:] = False
        k = np.cumsum(mask)
        evaluated = mask & ((k == start) | ((k > start) & ((k - start) % ms == 0)))
        if not evaluated.any():
            return -1, pair_delta
        tol = 0.01 if inew < 15 else 1e-5
        temp1 = ycbo_new - cov_col @ ycbo[:inew]
        temp2 = cov_new - np.einsum("ij,ij->i", cov_col, cov_col)
        with np.errstate(divide="ignore", invalid="ignore"):
            tol_good = evaluated & (temp2 / cov_new > tol)
            hinge = np.where(tol_good, temp1 * temp1 / temp2, 0.0)
        delta = adj * (unadj_lin + hinge)
        valid = evaluated & (delta > pair_delta) & (delta < max_legal)
        if not valid.any():
            return -1, pair_delta
        cand = np.where(valid, delta, -np.inf)
        s = int(np.argmax(cand))
        return n - 2 - s, float(delta[s])

    # -------------------------------------------------------------- find term
    def _find_term(self, nterms: int, rss_delta_prev: float, rss: float):
        n, p = self._n, self._p
        best = dict(case=-1, pred=-1, parent=-1, delta=0.0, new_form=False, lin=False)
        max_legal = min(1.01 * rss, 10 * rss_delta_prev)
        ycbo = self._yc @ self._bxorth[:, :nterms + 1]  # column nterms is filled per candidate
        ycbo[nterms] = 0.0
        fast_k = self.fast_k
        if fast_k <= 0:
            fast_k = 10001
        if fast_k < 3:
            fast_k = 3
        for iq in range(min(self._nq, fast_k)):
            parent = self._sorted_q[iq]
            best_for_parent = -1.0
            if self._deg[parent] >= self.degree:
                continue
            ms, end, start = self._span_params(n, p, self._deg[parent] + 1, parent)
            for pred in range(p):
                if self._dirs[parent, pred] != 0:
                    continue
                adj = 1.0 / (1.0 + (self.newvar_penalty if self._nuses[pred] == 0 else 0.0))
                new_form = self._new_form_flag(pred, parent, nterms)
                unadj_lin = 0.0
                delta_lin = 0.0
                if new_form:
                    xbx = self._X[:, pred] * self._bx[:, parent]
                    new_form = self._init_bxorth_col(xbx, nterms, MIN_BX_SOS)
                    qn = self._bxorth[:, nterms]
                    ycbo[nterms] = float(self._yc @ qn)
                    unadj_lin = float(self._ys @ qn) ** 2
                    delta_lin = adj * unadj_lin
                    if delta_lin > best_for_parent:
                        best_for_parent = delta_lin
                    if delta_lin > best["delta"]:
                        best.update(case=0, pred=pred, parent=parent, delta=delta_lin, lin=True)
                else:
                    self._bxorth[:, nterms] = 0.0
                    self._bxorth_mean[nterms] = 0.0
                    ycbo[nterms] = 0.0
                inew = nterms + 1 if new_form else nterms
                Qc = self._bxorth[:, :inew] - self._bxorth_mean[:inew]
                case, pair_delta = self._find_knot(parent, pred, inew, Qc, ycbo, unadj_lin, adj, delta_lin,
                                                   max_legal, start, ms, end)
                if pair_delta > best_for_parent:
                    best_for_parent = pair_delta
                if pair_delta > best["delta"]:
                    best.update(case=case, pred=pred, parent=parent, delta=pair_delta, lin=False, new_form=new_form)
            self._q_nterms[parent] = nterms
            self._q_delta[parent] = best_for_parent
        self._bxorth[:, nterms] = 0.0
        self._bxorth_mean[nterms] = 0.0
        return best

    # ---------------------------------------------------------- add term pair
    def _add_term_pair(self, nterms: int, parent: int, case: int, pred: int, lin: bool, new_form: bool):
        X = self._X
        t0, t1 = nterms, nterms + 1
        self._dirs[t0] = self._dirs[t1] = self._dirs[parent]
        self._cuts[t0] = self._cuts[t1] = self._cuts[parent]
        self._deg[t0] = self._deg[t1] = self._deg[parent] + 1
        self._dirs[t0, pred] = 2 if lin else 1
        self._dirs[t1, pred] = -1
        order = self._xorder[:, pred]
        cut = float(X[order[case], pred])
        self._cuts[t0, pred] = self._cuts[t1, pred] = cut
        self._bx[:, t0] = 0.0
        self._bx[:, t1] = 0.0
        if lin:
            self._bx[:, t0] = self._bx[:, parent] * X[:, pred]
        else:
            xs = X[order, pred]
            above = np.arange(self._n) > case
            self._bx[order[above], t0] = self._bx[order[above], parent] * (xs[above] - cut)
            self._bx[order[~above], t1] = self._bx[order[~above], parent] * (cut - xs[~above])
        self._nuses[pred] += 1
        self._full[t0] = True
        self._init_bxorth_col(self._bx[:, t0], t0, 0.0)
        self._full[t1] = (not lin) and new_form
        if self._full[t1]:
            self._init_bxorth_col(self._bx[:, t1], t1, 0.0)
        else:
            self._deg[t1] = MAX_DEGREE + 1
            self._bxorth[:, t1] = 0.0
            self._bxorth_mean[t1] = 0.0

    # ----------------------------------------------------------------- fit
    def fit(self, X, y) -> "MARS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self._X, self._n, self._p = X, n, p
        tss = float(np.sum((y - y.mean()) ** 2))
        self._dirs = np.zeros((1, p), dtype=int)
        self._cuts = np.zeros((1, p))
        if tss <= 0 or n < 3 or self.nk < 3:
            self.selected_ = [0]
            self.coef_ = np.array([y.mean()])
            self.terms_ = [[]]
            self.gcv_ = _gcv(1, n, tss, self.penalty)
            return self
        # earth scales y in the forward pass (Scale.y); the selection is scale free
        sd = float(y.std(ddof=1))
        ys = (y - y.mean()) / sd if sd > 0 else y.copy()
        self._ys = ys
        nmax = self.nk
        self._xorder = np.argsort(X, axis=0, kind="stable")
        self._bx = np.zeros((n, nmax))
        self._bxorth = np.zeros((n, nmax))
        self._bxorth_mean = np.zeros(nmax)
        self._dirs = np.zeros((nmax, p), dtype=int)
        self._cuts = np.zeros((nmax, p))
        self._deg = np.zeros(nmax, dtype=int)
        self._nuses = np.zeros(p, dtype=int)
        self._full = np.zeros(nmax, dtype=bool)
        self._bx[:, 0] = 1.0
        self._init_bxorth_col(self._bx[:, 0], 0, 0.0)
        self._full[0] = True
        ymean = float(ys.mean())
        self._yc = ys - ymean
        rss_null = float(self._yc @ self._yc)
        rss, rss_delta, rsq, rsq_delta = rss_null, rss_null, 0.0, 0.0
        nused = 1
        gcv_null = _gcv(nused, n, rss_null, self.penalty)
        self._init_q(nmax)
        self._add_term_to_q(0, 1, rss_null, True)
        nterms = 1
        self.termcond_ = 7
        if nmax >= 3:
            while True:
                best = self._find_term(nterms, rss_delta, rss)
                case, lin, new_form = best["case"], best["lin"], best["new_form"]
                if not self.auto_linpreds:
                    lin = False
                rss_delta = best["delta"]
                if case >= 0:
                    self._add_term_pair(nterms, best["parent"], case, best["pred"], lin, new_form)
                is_pair = case > 0 and new_form
                nused += 2 if is_pair else 1
                rss = _maybe_zero(rss - rss_delta)
                gcv = _gcv(nused, n, rss, self.penalty)
                old = rsq
                rsq = 1 - rss / rss_null
                rsq_delta = _maybe_zero(rsq - old)
                if case < 0:
                    self._full[nterms] = self._full[nterms + 1] = False
                    self.termcond_ = 6
                    break
                if self.thresh != 0 and rsq_delta < self.thresh:
                    self._full[nterms] = self._full[nterms + 1] = False
                    self.termcond_ = 4
                    break
                grsq = 1 - gcv / gcv_null
                if self.thresh != 0 and grsq < MIN_GRSQ:
                    self._full[nterms] = self._full[nterms + 1] = False
                    self.termcond_ = 3
                    break
                if not lin and new_form:
                    self._add_term_to_q(nterms, nterms, math.inf, False)
                    self._add_term_to_q(nterms + 1, nterms, math.inf, True)
                else:
                    self._add_term_to_q(nterms, nterms, math.inf, True)
                nterms += 2
                if rsq >= 1 - self.thresh:
                    self.termcond_ = 5
                    break
                if nterms >= nmax - 1:
                    self.termcond_ = 7
                    break
        # RegressAndFix: drop linearly dependent columns (dqrdc2 with tol 1e-8)
        used = [t for t in range(nmax) if self._full[t]]
        keep = []
        basis = []
        for t in used:
            col = self._bx[:, t]
            r = col.copy()
            for q in basis:
                r -= (q @ r) * q
            nr = float(np.linalg.norm(r))
            if t == 0 or nr >= QR_TOL * float(np.linalg.norm(col)):
                keep.append(t)
                basis.append(r / nr if nr > 0 else r)
        self.forward_terms_ = keep
        self._dirs = self._dirs[keep]
        self._cuts = self._cuts[keep]
        # pruning pass (backward elimination, subset with the lowest GCV), on the original y
        B = self._bx[:, keep]
        sel, self.gcv_ = self._prune(B, y)
        self.selected_ = sel
        self._dirs = self._dirs[sel]
        self._cuts = self._cuts[sel]
        Bs = B[:, sel]
        self.coef_ = np.linalg.lstsq(Bs, y, rcond=None)[0]
        self.terms_ = self._terms_from_dirs()
        self.rss_ = float(np.sum((y - Bs @ self.coef_) ** 2))
        for a in ("_bx", "_bxorth", "_bxorth_mean", "_full", "_deg", "_nuses", "_xorder", "_ys", "_yc", "_X"):
            if hasattr(self, a):
                delattr(self, a)
        return self

    def _prune(self, B: np.ndarray, y: np.ndarray):
        """pruning.pass with pmethod="backward": earth calls the BAKWRD routine
        of the leaps package (Alan Miller's AS 274) on the basis matrix with the
        intercept forced in.  BAKWRD keeps the columns in an ordering, starts
        from the full model and at each step drops the column whose removal
        increases the RSS the least, moving it to the end of the current model
        (VMOVE); the best subset of every size is the lowest-RSS subset among
        the *prefixes* of the ordering seen along the way (INITR records the
        initial prefixes, REPORT the prefixes touched by every move), which is
        not always the greedy path itself.  The subset size with the lowest GCV
        is selected (first minimum)."""
        n, m = B.shape

        def rss(cols):
            if not cols:
                return float(y @ y)
            beta = np.linalg.lstsq(B[:, cols], y, rcond=None)[0]
            r = y - B[:, cols] @ beta
            return float(r @ r)

        order = list(range(m))  # position -> column; column 0 (intercept) is forced in
        best = {k: (rss(order[:k]), list(order[:k])) for k in range(1, m + 1)}  # INITR
        for pos in range(m, 1, -1):  # BAKWRD: POS = LAST .. FIRST+1 (1-based)
            model = order[:pos]
            base = rss(model)
            ss = [rss([c for c in model if c != model[j]]) - base for j in range(1, pos)]
            jmin = 1 + int(np.argmin(ss))  # first minimum, as DROP1
            if jmin < pos - 1:  # JMIN < POS: VMOVE + REPORT of the prefixes JMIN..POS-1
                v = order.pop(jmin)
                order.insert(pos - 1, v)
                for k in range(jmin + 1, pos):
                    r = rss(order[:k])
                    if r < best[k][0]:
                        best[k] = (r, list(order[:k]))
        sizes = range(1, m + 1)
        gcvs = [_gcv(k, n, best[k][0], self.penalty) for k in sizes]
        kbest = int(np.argmin(gcvs)) + 1
        self.rss_per_subset_ = [best[k][0] for k in sizes]
        self.gcv_per_subset_ = gcvs
        return sorted(best[kbest][1]), gcvs[kbest - 1]

    # --------------------------------------------------------------- predict
    def _terms_from_dirs(self) -> List[List[Factor]]:
        terms = []
        for t in range(self._dirs.shape[0]):
            term = []
            for v in range(self._dirs.shape[1]):
                d = self._dirs[t, v]
                if d == 2:
                    term.append((v, float(self._cuts[t, v]), 0))
                elif d != 0:
                    term.append((v, float(self._cuts[t, v]), int(d)))
            terms.append(term)
        return terms

    @staticmethod
    def _factor_values(X: np.ndarray, f: Factor) -> np.ndarray:
        v, t, s = f
        x = X[:, v]
        if s > 0:
            return np.maximum(0.0, x - t)
        if s < 0:
            return np.maximum(0.0, t - x)
        return x

    def _basis(self, X: np.ndarray, terms) -> np.ndarray:
        B = np.ones((X.shape[0], len(terms)))
        for j, term in enumerate(terms):
            for f in term:
                B[:, j] *= self._factor_values(X, f)
        return B

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._basis(X, self.terms_) @ self.coef_

    def summary(self) -> str:
        lines = []
        for term, c in zip(self.terms_, self.coef_):
            if not term:
                lines.append(f"(Intercept) {c:.4g}")
                continue
            parts = []
            for v, t, s in term:
                if s > 0:
                    parts.append(f"h(x{v}-{t:.4g})")
                elif s < 0:
                    parts.append(f"h({t:.4g}-x{v})")
                else:
                    parts.append(f"x{v}")
            lines.append(" * ".join(parts) + f"  {c:.4g}")
        return "\n".join(lines)
