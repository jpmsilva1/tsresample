"""
Workflows (the mc.* functions of Exps.R / Exps_Time.R) and evaluation.

Every workflow has the signature

    wf(train: DataFrame, test: DataFrame, rng=None, **pars) -> dict

and returns {"evaluation": {"prec", "rec", "F1"}, "traintime", "trainpredtime",
"traintime_elapsed", "trainpredtime_elapsed"} (Exps.R only keeps
`evaluation`; Exps_Time.R adds the two proc.time() differences - here both
are always recorded: process CPU time like R's user time, plus wall time).

Learner mapping (R -> Python):
    lm                                  -> sklearn LinearRegression (OLS)
    e1071::svm(cost, gamma)             -> sklearn SVR(rbf, C=cost, gamma, epsilon=0.1) on
                                           standardised X and y (e1071 scale=TRUE)
    earth(nk, degree, thresh)           -> tsresamp.mars.MARS (a port of earth's forward and
                                           pruning passes: same model for the same data)
    randomForest(mtry, ntree)           -> RandomForestRegressor(max_features=mtry,
                                           n_estimators=ntree, min_samples_split=6, min_samples_leaf=1)
    rpart(minsplit, cp)                 -> DecisionTreeRegressor(min_samples_split=minsplit,
                                           min_samples_leaf=round(minsplit/3),
                                           ccp_alpha=cp*var(y), max_depth=30)
    forecast::auto.arima + Arima(model) -> pmdarima.auto_arima + statsmodels apply()
    BDES (bagged rpart on embeddings)   -> ported as is (see mc_BDES)

The target is the last column of `train` (R: train[, ncol(train)] and the
formula V10 ~ .).
"""
from __future__ import annotations

import time
import warnings
from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .mars import MARS
from .resampling import RESAMPLERS
from .uba import UtilityEvaluator, loss_control, phi, phi_control


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def target_of(df: pd.DataFrame) -> str:
    return df.columns[-1]


def resp(df: pd.DataFrame, tgt: Optional[str] = None) -> np.ndarray:
    """R: resp(form, data) / responseValues(form, data)"""
    return df[target_of(df) if tgt is None else tgt].to_numpy(dtype=float)


def split_xy(df: pd.DataFrame, tgt: Optional[str] = None):
    tgt = target_of(df) if tgt is None else tgt
    X = df.drop(columns=[tgt]).to_numpy(dtype=float)
    y = df[tgt].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# evaluation statistics (utility-based regression framework)
# ---------------------------------------------------------------------------
def eval_stats(train: pd.DataFrame, test: pd.DataFrame, preds, ph: dict, ls: dict,
               tgt: Optional[str] = None, event_thr: float = 0.9) -> Dict[str, float]:
    """R: eval.stats(form, train, test, preds, ph, ls) -> c(prec, rec, F1)"""
    trues = resp(test, tgt)
    preds = np.asarray(preds, dtype=float).ravel()
    ev = UtilityEvaluator(ph, ls).metrics(preds, trues, event_thr=event_thr, betas=(1,))
    return {"prec": float(ev["prec"]), "rec": float(ev["rec"]), "F1": float(ev["F1"])}


def extended_stats(train, test, preds, ph, ls, tgt=None, event_thr=0.9) -> Dict[str, float]:
    """All the quantities computed inside R's eval.stats (only prec/rec/F1 are
    returned there); provided for convenience."""
    trues = resp(test, tgt)
    preds = np.asarray(preds, dtype=float).ravel()
    ev = UtilityEvaluator(ph, ls).metrics(preds, trues, event_thr=event_thr, betas=(0.5, 1, 2))
    ph_t = phi(trues, ph)
    err = trues - preds
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "prec": ev["prec"], "rec": ev["rec"], "F05": ev["F05"], "F1": ev["F1"], "F2": ev["F2"],
            "mad": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err ** 2)),
            "mape": float(np.mean(np.abs(err) / trues) * 100),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae_phi": float(np.mean(ph_t * np.abs(err))),
            "mape_phi": float(np.mean(ph_t * np.abs(err) / trues) * 100),
            "mse_phi": float(np.mean(ph_t * err ** 2)),
            "rmse_phi": float(np.sqrt(np.mean(ph_t * err ** 2))),
        }
    return out


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------
class LMLearner:
    """R: lm(form, train)"""

    def __init__(self, **kw):
        self.m = LinearRegression()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class SVMLearner:
    """R: e1071::svm(form, train, cost, gamma)  (eps-regression, radial kernel,
    epsilon=0.1, scale=TRUE for x and y)."""

    def __init__(self, cost: float, gamma: float, epsilon: float = 0.1, tol: float = 0.001,
                 cache_size: float = 40, **kw):
        self.cost, self.gamma, self.epsilon, self.tol, self.cache_size = cost, gamma, epsilon, tol, cache_size

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.xm = X.mean(axis=0)
        self.xs = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
        self.xs = np.where(self.xs > 0, self.xs, 1.0)
        self.ym = y.mean()
        ys = y.std(ddof=1) if len(y) > 1 else 1.0
        self.ys = ys if ys > 0 else 1.0
        self.m = SVR(kernel="rbf", C=self.cost, gamma=self.gamma, epsilon=self.epsilon, tol=self.tol,
                     cache_size=self.cache_size, shrinking=True)
        self.m.fit((X - self.xm) / self.xs, (y - self.ym) / self.ys)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.m.predict((X - self.xm) / self.xs) * self.ys + self.ym


class MARSLearner:
    """R: earth(form, train, nk, degree, thresh)"""

    def __init__(self, nk: int = 21, degree: int = 1, thresh: float = 0.001, **kw):
        self.m = MARS(nk=nk, degree=degree, thresh=thresh)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.m.predict(X)).ravel()


class TreeRanks:
    """Monotone transform of the predictors for the sklearn trees: each column is
    mapped to the rank of its value among the distinct training values (new values
    by linear interpolation between the neighbouring training values, clamped
    outside the training range).

    Why: sklearn's splitter treats two sorted feature values as equal when they
    differ by less than FEATURE_THRESHOLD = 1e-7 (an absolute constant) and never
    tries a split between them, whereas rpart and randomForest split between any
    two distinct doubles (`x[i+1] != x[i]`).  The series of the experiments contain
    such pairs (e.g. 0.0633339999999999 and 0.063334 in DS1, 30 pairs per column),
    and the R trees do split between them.  A tree depends on the predictors only
    through their order, so after the rank transform the same split positions are
    available to both implementations; sklearn's threshold between ranks i and
    i+1 is i + 0.5, and a new value between the two training values interpolates
    to a rank on the same side of i + 0.5 as it is of the R midpoint threshold.
    Used by the rpart, randomForest and BDES stand-ins at fit and predict time."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.levels_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for j, u in enumerate(self.levels_):
            out[:, j] = np.interp(X[:, j], u, np.arange(len(u), dtype=float)) if len(u) > 1 else 0.0
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class RFLearner:
    """R: randomForest(form, train, mtry, ntree)  (nodesize=5 for regression)

    nodesize semantics (randomForest regTree.c): a node with nodesize or fewer
    cases is not split; the children of a split may be of any size (down to 1).
    That is sklearn's min_samples_split = nodesize + 1 with min_samples_leaf = 1
    (`nodesize_mode="split"`, the default). `nodesize_mode="leaf"` uses
    min_samples_leaf = nodesize instead (leaves of at least 5 cases)."""

    def __init__(self, mtry: int, ntree: int = 500, seed: Optional[int] = None, n_jobs: int = 1,
                 nodesize: int = 5, nodesize_mode: str = "split", **kw):
        self.mtry, self.ntree, self.seed, self.n_jobs = mtry, ntree, seed, n_jobs
        self.nodesize, self.nodesize_mode = nodesize, nodesize_mode

    def fit(self, X, y):
        p = np.asarray(X).shape[1]
        if self.nodesize_mode == "leaf":
            size_kw = dict(min_samples_leaf=self.nodesize)
        else:
            size_kw = dict(min_samples_split=self.nodesize + 1, min_samples_leaf=1)
        self.m = RandomForestRegressor(n_estimators=self.ntree, max_features=min(int(self.mtry), p),
                                       bootstrap=True, random_state=self.seed, n_jobs=self.n_jobs, **size_kw)
        self.ranks_ = TreeRanks()
        self.m.fit(self.ranks_.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.ranks_.transform(X))


def rpart_regressor(y, minsplit: int = 20, cp: float = 0.01, maxdepth: int = 30, seed=None,
                    pruning: str = "ccp") -> DecisionTreeRegressor:
    """rpart(control=rpart.control(minsplit, cp)) as a sklearn tree.

    What rpart does with `cp` (rpart/src/partition.c, bsplit.c): the tree is grown
    with minsplit / minbucket = round(minsplit/3) / maxdepth and, for each node, the
    complexity g = (SS_node - SS_leaves_of_subtree) / n_splits_of_subtree is computed
    bottom-up, collapsing first the children with the smaller g (the weakest link);
    a subtree is kept only if g > alpha = cp * SS_root.  That is minimal
    cost-complexity pruning; in sklearn's units (impurity = SS / N) it is
    ccp_alpha = cp * SS_root / N = cp * var(y)  (`pruning="ccp"`, the default).
    `pruning="pre"` is the rule the rpart documentation seems to describe ("any split
    that does not decrease the overall lack of fit by a factor of cp is not
    attempted"): min_impurity_decrease = cp * var(y).  On the Monte Carlo windows of
    the experiments the "ccp" rule reproduces the number of leaves of the R trees in
    28/50 windows of DS1 (12/50 for "pre") and 36/50 of DS10 (24/50); the remaining
    differences are in the growth phase (see README)."""
    var = float(np.var(np.asarray(y, float)))
    kw = dict(min_impurity_decrease=cp * var) if pruning == "pre" else dict(ccp_alpha=cp * var)
    return DecisionTreeRegressor(min_samples_split=max(2, int(minsplit)),
                                 min_samples_leaf=max(1, int(round(minsplit / 3))),
                                 max_depth=maxdepth, random_state=seed, **kw)


class RPartLearner:
    """R: rpart(form, train, control=rpart.control(minsplit, cp))"""

    def __init__(self, minsplit: int = 20, cp: float = 0.01, seed: Optional[int] = None, pruning: str = "ccp", **kw):
        self.minsplit, self.cp, self.seed, self.pruning = minsplit, cp, seed, pruning

    def fit(self, X, y):
        self.ranks_ = TreeRanks()
        self.m = rpart_regressor(y, self.minsplit, self.cp, seed=self.seed, pruning=self.pruning)
        self.m.fit(self.ranks_.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.ranks_.transform(X))


LEARNERS = {"lm": LMLearner, "svm": SVMLearner, "mars": MARSLearner, "rf": RFLearner, "rpart": RPartLearner}

# resampler settings used by the workflows in Exps.R
_RESAMPLER_KW = {
    "UNDERB": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=False),
    "UNDERT": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=False),
    "UNDERTPhi": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=False),
    "OVERB": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=True),
    "OVERT": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=True),
    "OVERTPhi": dict(rel="auto", thr_rel=0.9, C_perc="balance", repl=True),
    "SMOTEB": dict(rel="auto", thr_rel=0.9, C_perc="balance", k=5, repl=True, dist="Euclidean", p=2),
    "SMOTET": dict(rel="auto", thr_rel=0.9, C_perc="balance", k=5, repl=True, dist="Euclidean", p=2),
    "SMOTETPhi": dict(rel="auto", thr_rel=0.9, C_perc="balance", k=5, repl=True, dist="Euclidean", p=2),
}


def apply_resampler(key: Optional[str], train: pd.DataFrame, rng: np.random.Generator,
                    un: Optional[float] = None, ov: Optional[float] = None) -> pd.DataFrame:
    """Apply the resampling strategy `key` with the settings of Exps.R.
    `un`/`ov` (OptParmsSearch.R) replace C.perc="balance" by list(un), list(ov)
    or list(un, ov)."""
    if key is None:
        return train
    kw = dict(_RESAMPLER_KW[key])
    if key.startswith("UNDER") and un is not None:
        kw["C_perc"] = [un]
    elif key.startswith("OVER") and ov is not None:
        kw["C_perc"] = [ov]
    elif key.startswith("SMOTE") and (un is not None or ov is not None):
        # u for the normal bumps, o for the rare bumps (Algorithm 5 of the article);
        # R's list(un, ov) only works when there are exactly two bumps
        kw["C_perc"] = {"un": 1 if un is None else un, "ov": 1 if ov is None else ov}
    return RESAMPLERS[key](train, tgt=None, rng=rng, **kw)


def _timed_fit_predict(model, X, y, Xt):
    t0, w0 = time.process_time(), time.perf_counter()
    model.fit(X, y)
    t1, w1 = time.process_time(), time.perf_counter()
    p = np.asarray(model.predict(Xt), dtype=float).ravel()
    t2, w2 = time.process_time(), time.perf_counter()
    times = {"traintime": t1 - t0, "trainpredtime": t2 - t0,
             "traintime_elapsed": w1 - w0, "trainpredtime_elapsed": w2 - w0}
    return p, times


def learner_workflow(learner: str, resampler: Optional[str], train: pd.DataFrame, test: pd.DataFrame,
                     rng=None, un=None, ov=None, **pars) -> dict:
    """Generic mc.<learner>[_<resampler>] workflow."""
    rng = _rng(rng)
    ycol = train.iloc[:, -1].to_numpy(dtype=float)  # R: train[, ncol(train)]
    ph = phi_control(ycol, method="extremes")
    ls = loss_control(ycol)
    train = apply_resampler(resampler, train, rng, un=un, ov=ov)
    tgt = target_of(train)
    X, y = split_xy(train, tgt)
    Xt = test[[c for c in train.columns if c != tgt]].to_numpy(dtype=float)
    kw = dict(pars)
    if learner in ("rf", "rpart"):
        kw.setdefault("seed", _seed(rng))
    model = LEARNERS[learner](**kw)
    p, times = _timed_fit_predict(model, X, y, Xt)
    ev = eval_stats(train, test, p, ph, ls, tgt)
    return {"evaluation": ev, **times, "n_train": int(len(train))}


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------
class AutoARIMA:
    """forecast::auto.arima(trainY) then fitted(Arima(c(trainY, trues), model=m))."""

    def __init__(self, **kw):
        self.kw = dict(seasonal=False, stepwise=True, information_criterion="aicc", test="kpss",
                       max_p=5, max_q=5, max_d=2, start_p=2, start_q=2, max_order=5,
                       suppress_warnings=True, error_action="ignore")
        self.kw.update(kw)

    def fit(self, y):
        import pmdarima as pm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = pm.auto_arima(np.asarray(y, float), **self.kw)
        return self

    def fitted_on(self, y_full) -> np.ndarray:
        """One-step-ahead in-sample fitted values of the estimated model
        applied (without re-estimation) to the full series."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.m.arima_res_.apply(np.asarray(y_full, float))
        return np.asarray(res.fittedvalues, dtype=float)


def mc_arima(train: pd.DataFrame, test: pd.DataFrame, rng=None, **pars) -> dict:
    ycol = train.iloc[:, -1].to_numpy(dtype=float)
    ph = phi_control(ycol, method="extremes")
    ls = loss_control(ycol)
    trainY = resp(train)
    trues = resp(test)
    t0, w0 = time.process_time(), time.perf_counter()
    m = AutoARIMA(**pars).fit(trainY)
    data = np.concatenate([trainY, trues])
    t1, w1 = time.process_time(), time.perf_counter()
    p = m.fitted_on(data)[len(trainY):]
    t2, w2 = time.process_time(), time.perf_counter()
    ev = eval_stats(train, test, p, ph, ls)
    return {"evaluation": ev, "traintime": t1 - t0, "trainpredtime": t2 - t0,
            "traintime_elapsed": w1 - w0, "trainpredtime_elapsed": w2 - w0,
            "order": tuple(int(o) for o in m.m.order), "n_train": int(len(train))}


# ---------------------------------------------------------------------------
# BDES (bagged decision trees on embeddings, Cerqueira et al.)
# ---------------------------------------------------------------------------
def embed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """R: embedStats(data) - appends the row-wise mean and (sample) variance.

    NOTE: as in the R code the statistics are computed over *all* columns,
    including the target (the first column after the re-ordering done in
    mc.BDES). This is reproduced faithfully; see README."""
    vals = df.to_numpy(dtype=float)
    out = df.copy()
    out["mean"] = vals.mean(axis=1)
    out["var"] = vals.var(axis=1, ddof=1)
    return out


def bootstrap(n: int, rng: np.random.Generator) -> np.ndarray:
    """R: sample(n, n, replace=TRUE)"""
    return rng.integers(0, n, size=n)


def bagged_trees(train: pd.DataFrame, embedding_dimension: int = 10, nstats: int = 2, ntrees: int = 500,
                 rng=None, pruning: str = "ccp"):
    """R: baggedtrees(form, data, embedding.dimension, nstats, learner.pars)

    Returns a list of (tree, feature_columns, rank_transform). The target is the FIRST column
    of `train` (mc.BDES re-orders the columns to V10, V9, ..., V1, mean, var).
    """
    rng = _rng(rng)
    tgt = train.columns[0]
    cols = list(train.columns)
    embed_split_at = [embedding_dimension, embedding_dimension / 2, embedding_dimension / 4]
    n = len(train)
    n_ = int(np.floor(ntrees / (2 * len(embed_split_at))))
    models = []
    ncol = len(cols)
    for K in embed_split_at:
        k = int(K)  # seq_len(K)
        preds1 = cols[:k]
        preds2 = cols[:k] + cols[ncol - nstats:]
        for predictors in (preds1, preds2):
            feats = [c for c in predictors if c != tgt]
            for _ in range(n_):
                idx = bootstrap(n, rng)
                sub = train.iloc[idx]
                X = sub[feats].to_numpy(dtype=float)
                y = sub[tgt].to_numpy(dtype=float)
                ranks = TreeRanks()
                tree = rpart_regressor(y, minsplit=20, cp=0.01, seed=_seed(rng), pruning=pruning).fit(ranks.fit_transform(X), y)
                models.append((tree, feats, ranks))
    return models


def mc_BDES(train: pd.DataFrame, test: pd.DataFrame, rng=None, ntrees: int = 500, pruning: str = "ccp", **pars) -> dict:
    rng = _rng(rng)
    ycol = train.iloc[:, -1].to_numpy(dtype=float)
    ph = phi_control(ycol, method="extremes")
    ls = loss_control(ycol)
    tgt = target_of(train)
    order = list(train.columns)[::-1]  # c(10,9,...,1): target first
    tr = embed_stats(train[order])
    te = embed_stats(test[order])
    t0, w0 = time.process_time(), time.perf_counter()
    models = bagged_trees(tr, embedding_dimension=len(order), nstats=2, ntrees=ntrees, rng=rng, pruning=pruning)
    t1, w1 = time.process_time(), time.perf_counter()
    preds = np.column_stack([tree.predict(ranks.transform(te[feats].to_numpy(dtype=float))) for tree, feats, ranks in models])
    p = preds.mean(axis=1)
    t2, w2 = time.process_time(), time.perf_counter()
    ev = eval_stats(tr, te, p, ph, ls, tgt)
    return {"evaluation": ev, "traintime": t1 - t0, "trainpredtime": t2 - t0,
            "traintime_elapsed": w1 - w0, "trainpredtime_elapsed": w2 - w0, "n_train": int(len(train))}


# ---------------------------------------------------------------------------
# registry (same names and order as the `models` vector of GetResults.R)
# ---------------------------------------------------------------------------
RESAMPLER_KEYS = [None, "UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT", "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
LEARNER_ORDER = ["lm", "svm", "mars", "rf", "rpart"]

WORKFLOWS: Dict[str, Callable] = {}
for _L in LEARNER_ORDER:
    for _R in RESAMPLER_KEYS:
        _name = f"mc.{_L}" + (f"_{_R}" if _R else "")
        WORKFLOWS[_name] = partial(learner_workflow, _L, _R)
WORKFLOWS["mc.arima"] = mc_arima
WORKFLOWS["mc.BDES"] = mc_BDES
WORKFLOW_NAMES = list(WORKFLOWS.keys())


def run_workflow(name: str, train: pd.DataFrame, test: pd.DataFrame, rng=None, **pars) -> dict:
    """R: runWorkflow(Workflow(name, pars), form, train, test)"""
    if name not in WORKFLOWS:
        raise KeyError(f"unknown workflow {name}")
    return WORKFLOWS[name](train, test, rng=rng, **pars)
