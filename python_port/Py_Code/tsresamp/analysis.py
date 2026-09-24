"""
Result tables and statistical comparisons.

    GetResults.R          -> get_results(res)
    GetResults_RunTime.R  -> get_results_runtime(res_time)
    pairedComparisons()   -> paired_comparisons(res, baseline, maxs, p_value)
    PairedComparisons.R   -> WLdef(res, base, measure, sig)
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import METRICS, ComparisonResults

# the `models` vector of GetResults.R
MODELS = [f"mc.{L}{s}" for L in ("lm", "svm", "mars", "rf", "rpart")
          for s in ("", "_UNDERB", "_UNDERT", "_UNDERTPhi", "_OVERB", "_OVERT", "_OVERTPhi", "_SMOTEB", "_SMOTET", "_SMOTETPhi")]
MODELS += ["mc.arima", "mc.BDES"]


# ---------------------------------------------------------------------------
# GetResults.R / GetResults_RunTime.R
# ---------------------------------------------------------------------------
def get_results(res: ComparisonResults, models: Optional[Sequence[str]] = None, task=None,
                n_reps: Optional[int] = None, summary: str = "mean", F1_summary: Optional[str] = None) -> pd.DataFrame:
    """Table with the mean prec/rec/F1 over the iterations of each workflow.
    (OptParmsSearch.R uses the median for F1: F1_summary="median".)"""
    task = res.task_names()[0] if task is None else task
    models = res.workflow_names(task) if models is None else list(models)
    f = {"mean": np.nanmean, "median": np.nanmedian}
    rows = []
    for wf in models:
        sc = res.iterations_scores(task, wf, METRICS)
        if n_reps is not None:
            sc = sc[:n_reps]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"model": wf,
                         "prec": float(f[summary](sc[:, 0])),
                         "rec": float(f[summary](sc[:, 1])),
                         "F1": float(f[F1_summary or summary](sc[:, 2]))})
    return pd.DataFrame(rows)


def get_results_runtime(res: ComparisonResults, models: Optional[Sequence[str]] = None, task=None,
                        n_reps: int = 10, key: str = "traintime") -> pd.DataFrame:
    """Mean training / prediction / total time (first n_reps iterations)."""
    task = res.task_names()[0] if task is None else task
    models = res.workflow_names(task) if models is None else list(models)
    tot_key = key.replace("traintime", "trainpredtime")
    rows = []
    for wf in models:
        its = res.tasks[task][wf].iterations[:n_reps]
        tr = np.array([float(it.get(key, np.nan)) for it in its])
        tot = np.array([float(it.get(tot_key, np.nan)) for it in its])
        rows.append({"model": wf, "tr.time": float(np.nanmean(tr)), "pr.time": float(np.nanmean(tot - tr)),
                     "tot.time": float(np.nanmean(tot))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# statistical tests (R defaults)
# ---------------------------------------------------------------------------
def wilcox_test_paired(x, y) -> float:
    """p-value of R's wilcox.test(x, y, paired=TRUE) with default arguments:
    zeros are dropped; the exact distribution is used when n < 50 and there
    are no ties/zeros, otherwise the normal approximation with continuity
    correction."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    d = x - y
    zeroes = (d == 0).any()
    d_nz = d[d != 0]
    n = len(d_nz)
    if n == 0:
        return np.nan
    ties = len(np.unique(np.abs(d_nz))) < n
    exact = (n < 50) and not ties and not zeroes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.wilcoxon(d_nz, zero_method="wilcox", correction=True, method="exact" if exact else "approx")
    return float(r.pvalue)


def t_test_paired(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_rel(x[ok], y[ok]).pvalue)



def common_workflows(res: ComparisonResults) -> list:
    """Workflows present in every task (R assumes the same workflows in all
    tasks; results merged from different runs may differ), in the order of
    the first task."""
    names = res.workflow_names(res.task_names()[0])
    for t in res.task_names()[1:]:
        present = set(res.workflow_names(t))
        names = [n for n in names if n in present]
    return names


# ---------------------------------------------------------------------------
# pairedComparisons (resultsAnalysis.R)
# ---------------------------------------------------------------------------
def paired_comparisons(res: ComparisonResults, baseline: Optional[str] = None, maxs: Optional[Sequence[bool]] = None,
                       p_value: float = 0.05) -> dict:
    ts = res.task_names()
    ws = common_workflows(res)
    ms = res.metric_names()
    nts, nws = len(ts), len(ws)
    if nws < 2:
        raise ValueError("Paired comparisons only make sense with more than one workflow!")
    if baseline is not None and baseline not in ws:
        raise ValueError(f"baseline {baseline!r} is not present in every task")
    if maxs is None:
        maxs = [False] * len(ms)
    out = {}
    for p, metric in enumerate(ms):
        comp = {"setup": {"nTasks": nts, "nWorkflows": nws}}
        avg = pd.DataFrame(index=ts, columns=ws, dtype=float)
        med = pd.DataFrame(index=ts, columns=ws, dtype=float)
        for t in ts:
            for w in ws:
                sc = res.iterations_scores(t, w, ms)[:, p]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    avg.loc[t, w] = np.nanmean(sc)
                    med.loc[t, w] = np.nanmedian(sc)
        comp["avgScores"] = avg
        comp["medScores"] = med
        rk_src = -avg if maxs[p] else avg
        comp["rks"] = pd.DataFrame(np.vstack([stats.rankdata(r, method="average", nan_policy="omit") for r in rk_src.to_numpy()]),
                                   index=ts, columns=ws)
        comp["avgRksWFs"] = comp["rks"].mean(axis=0)
        if baseline is None:
            base = comp["avgRksWFs"].idxmin()
        else:
            base = baseline
        comp["baseline"] = base
        other = [w for w in ws if w != base]
        order = [base] + other
        wt, tt = {}, {}
        for t in ts:
            W = pd.DataFrame(np.nan, index=order, columns=["MedScore", "DiffMedScores", "p.value"])
            T = pd.DataFrame(np.nan, index=order, columns=["AvgScore", "DiffAvgScores", "p.value"])
            W.loc[base, "MedScore"] = med.loc[t, base]
            T.loc[base, "AvgScore"] = avg.loc[t, base]
            xb = res.iterations_scores(t, base, ms)[:, p]
            for o in other:
                xo = res.iterations_scores(t, o, ms)[:, p]
                W.loc[o, "DiffMedScores"] = med.loc[t, base] - med.loc[t, o]
                W.loc[o, "MedScore"] = med.loc[t, o]
                try:
                    W.loc[o, "p.value"] = wilcox_test_paired(xo, xb)
                except Exception:
                    W.loc[o, "p.value"] = np.nan
                T.loc[o, "DiffAvgScores"] = avg.loc[t, base] - avg.loc[t, o]
                T.loc[o, "AvgScore"] = avg.loc[t, o]
                try:
                    T.loc[o, "p.value"] = t_test_paired(xo, xb)
                except Exception:
                    T.loc[o, "p.value"] = np.nan
            wt[t] = W
            tt[t] = T
        comp["WilcoxonSignedRank.test"] = wt
        comp["t.test"] = tt
        if nts > 1:
            rks = comp["avgRksWFs"].to_numpy()
            chi = 12 * nts / (nws * (nws + 1)) * (np.sum(rks ** 2) - (nws * (nws + 1) ** 2) / 4)
            with np.errstate(divide="ignore", invalid="ignore"):
                FF = (nts - 1) * chi / (nts * (nws - 1) - chi)
            # NB: the R code uses df() (the F density) here, reproduced as is
            critVal = stats.f.pdf(1 - p_value, nws - 1, (nws - 1) * (nts - 1))
            rejNull = bool(FF > critVal)
            comp["F.test"] = {"chi": chi, "FF": FF, "critVal": critVal, "rejNull": rejNull}
            comp["Nemenyi.test"] = None
            comp["BonferroniDunn.test"] = None
            if rejNull:
                CD_n = stats.studentized_range.ppf(1 - p_value, nws, 1e6) / np.sqrt(2) * np.sqrt(nws * (nws + 1) / (6 * nts))
                allRkDifs = np.abs(rks[:, None] - rks[None, :])
                comp["Nemenyi.test"] = {"critDif": CD_n, "rkDifs": pd.DataFrame(allRkDifs, index=ws, columns=ws),
                                        "signifDifs": pd.DataFrame(allRkDifs >= CD_n, index=ws, columns=ws)}
                CD_bd = stats.studentized_range.ppf(1 - (p_value / (nws - 1)), 2, 1e6) / np.sqrt(2) * np.sqrt(nws * (nws + 1) / (6 * nts))
                pb = ws.index(base)
                diffs2base = pd.Series(np.abs(np.delete(rks, pb) - rks[pb]), index=other)
                comp["BonferroniDunn.test"] = {"critDif": CD_bd, "baseline": base, "rkDifs": diffs2base,
                                               "signifDifs": diffs2base >= CD_bd}
        else:
            comp["F.test"] = comp["Nemenyi.test"] = comp["BonferroniDunn.test"] = None
        out[metric] = comp
    return out


def WLdef(res: ComparisonResults, base: str, measure: str, sig: float = 0.05) -> pd.DataFrame:
    """Port of WLdef() in PairedComparisons.R: wins / significant wins /
    losses / significant losses / ties of every workflow against `base`,
    counted over the tasks (data sets)."""
    pres = paired_comparisons(res, base, maxs=[True] * len(res.metric_names()), p_value=sig)
    ws = common_workflows(res)
    others = [w for w in ws if w != base]
    WL = pd.DataFrame(0, index=others, columns=["Win", "sigWin", "Loss", "SigLoss", "Tie"], dtype=int)
    for t in res.task_names():
        W = pres[measure]["WilcoxonSignedRank.test"][t]
        for nm in others:
            diff = W.loc[nm, "DiffMedScores"]
            pv = W.loc[nm, "p.value"]
            if diff < 0:
                WL.loc[nm, "Win"] += 1
                if pv < sig:
                    WL.loc[nm, "sigWin"] += 1
            elif diff == 0:
                WL.loc[nm, "Tie"] += 1
            else:
                WL.loc[nm, "Loss"] += 1
                if pv < sig:
                    WL.loc[nm, "SigLoss"] += 1
    return WL
