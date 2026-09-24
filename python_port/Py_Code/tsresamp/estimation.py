"""
Experimental methodology: port of the parts of the R package
`performanceEstimation` (L. Torgo) used by the experiments.

    Workflow("mc.svm", cost=cost, gamma=gamma)         -> Workflow("mc.svm", dict(cost=..., gamma=...))
    workflowVariants("mc.svm_UNDERB", cost=c(..), ...)  -> workflow_variants("mc.svm_UNDERB", cost=[...], ...)
    MonteCarlo(nReps=50, szTrain=.5, szTest=.25)        -> MonteCarlo(nReps=50, szTrain=.5, szTest=.25)
    EstimationTask("totTime", method=MonteCarlo(...))   -> EstimationTask(method=MonteCarlo(...))
    performanceEstimation(PredTask(form, ds), wfs, et)  -> performance_estimation("ds1", ds, wfs, et)
    getIterationsInfo(exp, workflow=wf, task=1, it=i)   -> res.get_iterations_info(task, wf, it)

Monte Carlo estimation (mcEstimates in experiments.R): with n rows,
train.size = floor(n*szTrain), test.size = floor(n*szTest), nReps starting
points are sampled (without replacement) from (train.size+1):(n-test.size+1)
and sorted; iteration i trains on the train.size rows before the starting
point and tests on the test.size rows from it.  The R random stream cannot
be reproduced; numpy's default generator seeded with `seed` is used.

Deviation: a failing iteration (e.g. a resampler that cannot be applied to a
particular split) is recorded with NaN scores and the error message instead
of aborting the whole experiment.
"""
from __future__ import annotations

import itertools
import pickle
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

METRICS = ("prec", "rec", "F1")


# ---------------------------------------------------------------------------
# workflows and estimation settings
# ---------------------------------------------------------------------------
@dataclass
class Workflow:
    func: str
    pars: dict = field(default_factory=dict)
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.func

    def __repr__(self):
        ps = ", ".join(f"{k}={v}" for k, v in self.pars.items())
        return f"Workflow({self.name}: {self.func}({ps}))"


def workflow_variants(func: str, varsRootName: Optional[str] = None, **vars) -> List[Workflow]:
    """R: workflowVariants(wf, ...). Parameters given as lists/tuples with more
    than one value generate variants (expand.grid order: the first parameter
    varies fastest); the variants are named <root>.v1, <root>.v2, ..."""
    root = func if varsRootName is None else varsRootName
    names = list(vars.keys())
    values = []
    for k in names:
        v = vars[k]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 1:
            values.append(list(v))
        elif isinstance(v, (list, tuple, np.ndarray)):
            values.append([v[0]])
        else:
            values.append([v])
    # expand.grid: first factor varies fastest -> product over reversed lists
    grid = [tuple(reversed(t)) for t in itertools.product(*reversed(values))]
    out = []
    for i, combo in enumerate(grid, start=1):
        pars = dict(zip(names, combo))
        out.append(Workflow(func, pars, name=f"{root}.v{i}"))
    return out


@dataclass
class MonteCarlo:
    nReps: int = 10
    szTrain: float = 0.25
    szTest: float = 0.25
    seed: int = 1234


@dataclass
class EstimationTask:
    metrics: Sequence[str] = METRICS
    method: MonteCarlo = field(default_factory=MonteCarlo)


@dataclass
class Split:
    it: int          # 1-based iteration number
    start: int       # R's 1-based starting point of the test window
    train: range     # 0-based positions
    test: range


def mc_splits(n: int, mc: MonteCarlo) -> List[Split]:
    """Port of the split generation in mcEstimates()."""
    train_size = int(n * mc.szTrain) if mc.szTrain < 1 else int(mc.szTrain)
    test_size = int(n * mc.szTest) if mc.szTest < 1 else int(mc.szTest)
    if n - test_size + 1 <= train_size + 1:
        raise ValueError("mcEstimates:: Invalid train/test sizes.")
    rng = np.random.default_rng(mc.seed)
    selection_range = np.arange(train_size + 1, n - test_size + 2)  # (train.size+1):(n-test.size+1)
    if mc.nReps > len(selection_range):
        raise ValueError("nReps larger than the number of admissible starting points")
    starts = np.sort(rng.choice(selection_range, size=mc.nReps, replace=False))
    out = []
    for it, start in enumerate(starts, start=1):
        s0 = int(start) - 1
        out.append(Split(it=it, start=int(start), train=range(s0 - train_size, s0), test=range(s0, s0 + test_size)))
    return out


# ---------------------------------------------------------------------------
# results container (ComparisonResults / EstimationResults)
# ---------------------------------------------------------------------------
@dataclass
class WorkflowResults:
    workflow: Workflow
    iterations: List[dict]
    splits: List[Split]

    def scores(self, metrics: Sequence[str] = METRICS) -> np.ndarray:
        """R: @iterationsScores  (nReps x nMetrics, NaN for failed iterations)"""
        rows = []
        for it in self.iterations:
            ev = it.get("evaluation", {}) or {}
            rows.append([float(ev.get(m, np.nan)) for m in metrics])
        return np.asarray(rows, dtype=float).reshape(len(self.iterations), len(metrics))


class ComparisonResults:
    """dict-like: results[task][workflow_name] -> WorkflowResults"""

    def __init__(self, est_task: EstimationTask):
        self.est_task = est_task
        self.tasks: Dict[str, Dict[str, WorkflowResults]] = {}

    # --- R accessors
    def task_names(self) -> List[str]:
        return list(self.tasks.keys())

    def workflow_names(self, task: Optional[str] = None) -> List[str]:
        t = self.task_names()[0] if task is None else task
        return list(self.tasks[t].keys())

    def metric_names(self) -> List[str]:
        return list(self.est_task.metrics)

    def __getitem__(self, task):
        if isinstance(task, int):
            task = self.task_names()[task - 1]
        return self.tasks[task]

    def get_iterations_info(self, task, workflow, it: int) -> dict:
        """R: getIterationsInfo(obj, workflow, task, it) (1-based it)."""
        if isinstance(task, int):
            task = self.task_names()[task - 1]
        if isinstance(workflow, int):
            workflow = self.workflow_names(task)[workflow - 1]
        return self.tasks[task][workflow].iterations[it - 1]

    def iterations_scores(self, task, workflow, metrics: Optional[Sequence[str]] = None) -> np.ndarray:
        if isinstance(task, int):
            task = self.task_names()[task - 1]
        if isinstance(workflow, int):
            workflow = self.workflow_names(task)[workflow - 1]
        return self.tasks[task][workflow].scores(self.metric_names() if metrics is None else metrics)

    # --- persistence / merging
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path) -> "ComparisonResults":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def add(self, task: str, wres: WorkflowResults):
        self.tasks.setdefault(task, {})[wres.workflow.name] = wres

    def merge(self, other: "ComparisonResults") -> "ComparisonResults":
        for t, wfs in other.tasks.items():
            for name, wres in wfs.items():
                self.add(t, wres)
        return self

    @staticmethod
    def merge_all(results: Sequence["ComparisonResults"]) -> "ComparisonResults":
        out = ComparisonResults(results[0].est_task)
        for r in results:
            out.merge(r)
        return out

    def summary(self, task: Optional[str] = None) -> pd.DataFrame:
        t = self.task_names()[0] if task is None else task
        rows = []
        for name, wres in self.tasks[t].items():
            sc = wres.scores(self.metric_names())
            row = {"workflow": name}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row.update({m: float(np.nanmean(sc[:, j])) for j, m in enumerate(self.metric_names())})
            row["failed"] = int(np.isnan(sc[:, 0]).sum())
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run_iteration(wf: Workflow, ds: pd.DataFrame, split: Split, seed: int, wf_index: int = 0) -> dict:
    from .models import run_workflow  # lazy: keeps this module importable without sklearn etc.
    train = ds.iloc[list(split.train)]
    test = ds.iloc[list(split.test)]
    rng = np.random.default_rng([int(seed), int(wf_index), int(split.it)])
    t0 = time.perf_counter()
    try:
        res = run_workflow(wf.func, train, test, rng=rng, **wf.pars)
    except Exception as e:  # deviation: record the failure, do not abort the experiment
        res = {"evaluation": {m: np.nan for m in METRICS}, "traintime": np.nan, "trainpredtime": np.nan,
               "traintime_elapsed": np.nan, "trainpredtime_elapsed": np.nan,
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    res["train"] = split.train
    res["test"] = split.test
    res["start"] = split.start
    res["it"] = split.it
    res["elapsed"] = time.perf_counter() - t0
    return res


def mc_estimates(wf: Workflow, ds: pd.DataFrame, est_task: EstimationTask, n_jobs: int = 1,
                 verbose: bool = True, wf_index: int = 0) -> WorkflowResults:
    """R: mcEstimates(wf, task, estTask)"""
    mc = est_task.method
    splits = mc_splits(len(ds), mc)
    if verbose:
        print(f"\n\n##### {wf.name}  ({wf.func} {wf.pars})\n"
              f"{mc.nReps}x Monte Carlo (seed={mc.seed}, train={mc.szTrain}, test={mc.szTest})", flush=True)
    if n_jobs == 1:
        its = []
        for sp in splits:
            if verbose:
                print(f"Repetition {sp.it}\n\t start test = {sp.start}; test size = {len(sp.test)}", flush=True)
            its.append(run_iteration(wf, ds, sp, mc.seed, wf_index))
    else:
        from joblib import Parallel, delayed
        its = Parallel(n_jobs=n_jobs)(delayed(run_iteration)(wf, ds, sp, mc.seed, wf_index) for sp in splits)
    if verbose:
        sc = WorkflowResults(wf, its, splits).scores(est_task.metrics)
        means = ", ".join(f"{m}={np.nanmean(sc[:, j]):.4f}" for j, m in enumerate(est_task.metrics))
        nerr = sum(1 for r in its if "error" in r)
        print(f"  -> {means}" + (f"   [{nerr} failed iterations]" if nerr else ""), flush=True)
    return WorkflowResults(wf, its, splits)


def performance_estimation(task: str, ds: pd.DataFrame, workflows: Sequence[Workflow], est_task: EstimationTask,
                           n_jobs: int = 1, verbose: bool = True, checkpoint_dir=None) -> ComparisonResults:
    """R: performanceEstimation(PredTask(form, ds), c(Workflow(...), ...), EstimationTask(...))

    With `checkpoint_dir` every finished workflow is pickled to
    <checkpoint_dir>/<task>__<workflow>.pkl and skipped when re-run.
    """
    res = ComparisonResults(est_task)
    for i, wf in enumerate(workflows):
        ck = None
        if checkpoint_dir is not None:
            ck = Path(checkpoint_dir) / f"{task}__{wf.name}.pkl"
            if ck.exists():
                with open(ck, "rb") as fh:
                    res.add(task, pickle.load(fh))
                if verbose:
                    print(f"##### {wf.name}: loaded from checkpoint", flush=True)
                continue
        wres = mc_estimates(wf, ds, est_task, n_jobs=n_jobs, verbose=verbose, wf_index=i)
        res.add(task, wres)
        if ck is not None:
            ck.parent.mkdir(parents=True, exist_ok=True)
            with open(ck, "wb") as fh:
                pickle.dump(wres, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return res
