"""Shared helpers for the experiment scripts (data loading, workflow lists, CLI)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tsresamp.data import create_data, default_sizes, get_dataset, handle_na, load_data  # noqa: E402
from tsresamp.estimation import EstimationTask, MonteCarlo, Workflow  # noqa: E402
from tsresamp.models import WORKFLOW_NAMES  # noqa: E402

RESULTS_DIR = ROOT / "results"

# EXAMPLE PARAMETRIZATION (Exps.R)
DEFAULT_PARS = dict(cost=150, gamma=0.001, nk=17, degree=2, thresh=0.001, mtry=7, ntree=500, minsplit=10, cp=0.001)

# Table 7 of the article: optimal parametrisation of each regression algorithm
# in each data set (svm cost/gamma, mars nk/degree/thresh, rf mtry/ntree, rpart minsplit/cp)
_T7 = {
    1: (300, 0.01, 17, 1, 0.001, 5, 1500, 10, 0.01),
    2: (300, 0.01, 17, 2, 0.001, 7, 750, 10, 0.001),
    3: (300, 0.01, 17, 1, 0.001, 7, 500, 10, 0.001),
    4: (150, 0.01, 10, 1, 0.001, 7, 750, 10, 0.1),
    5: (300, 0.001, 10, 2, 0.001, 7, 750, 20, 0.001),
    6: (300, 0.01, 17, 2, 0.001, 5, 500, 10, 0.001),
    7: (300, 0.01, 10, 1, 0.001, 7, 750, 30, 0.001),
    8: (300, 0.01, 17, 2, 0.001, 7, 750, 30, 0.001),
    9: (10, 0.01, 10, 2, 0.001, 5, 750, 30, 0.001),
    10: (300, 0.01, 17, 2, 0.001, 7, 500, 10, 0.001),
    11: (10, 0.01, 17, 1, 0.001, 7, 500, 20, 0.001),
    12: (300, 0.01, 17, 1, 0.001, 7, 750, 10, 0.001),
    13: (150, 0.01, 17, 2, 0.001, 7, 750, 10, 0.001),
    14: (150, 0.01, 17, 2, 0.001, 7, 1500, 10, 0.001),
    15: (300, 0.01, 17, 2, 0.001, 5, 1500, 10, 0.001),
    16: (300, 0.01, 17, 2, 0.001, 7, 750, 10, 0.001),
    17: (300, 0.01, 17, 2, 0.001, 7, 500, 10, 0.001),
    18: (300, 0.01, 17, 2, 0.001, 5, 500, 10, 0.001),
    19: (150, 0.01, 17, 1, 0.01, 5, 500, 10, 0.001),
    20: (300, 0.01, 17, 2, 0.001, 7, 500, 10, 0.001),
    21: (150, 0.001, 17, 2, 0.001, 7, 500, 10, 0.001),
    22: (150, 0.001, 10, 2, 0.001, 7, 500, 10, 0.001),
    23: (10, 0.001, 10, 1, 0.001, 5, 500, 10, 0.001),
    24: (150, 0.01, 17, 1, 0.001, 7, 750, 10, 0.001),
}
_KEYS = ("cost", "gamma", "nk", "degree", "thresh", "mtry", "ntree", "minsplit", "cp")
PAPER_PARS = {ds: dict(zip(_KEYS, v)) for ds, v in _T7.items()}


def paper_pars(dataset: int) -> dict:
    """Optimal parameters of Table 7 of the article for data set `dataset`."""
    return dict(PAPER_PARS[dataset])


def load_embedded(dataset: int, embed: int = 10, na: str = "knn", data=None):
    """ds <- create.data(data[[i]], 10)  (+ NA handling, see README)."""
    ts = get_dataset(dataset, data)
    ds = create_data(ts, embed)
    if ds.isna().any().any():
        ds = handle_na(ds, na)
    return ds


def learner_pars(name: str, pars: dict) -> dict:
    if ".svm" in name:
        return {k: pars[k] for k in ("cost", "gamma")}
    if ".mars" in name:
        return {k: pars[k] for k in ("nk", "degree", "thresh")}
    if ".rf" in name:
        return {k: pars[k] for k in ("mtry", "ntree")}
    if ".rpart" in name:
        return {k: pars[k] for k in ("minsplit", "cp")}
    return {}


def default_workflows(pars: dict | None = None, names=None) -> list[Workflow]:
    """The 52 workflows of Exps.R, in the same order."""
    pars = {**DEFAULT_PARS, **(pars or {})}
    names = WORKFLOW_NAMES if names is None else names
    return [Workflow(n, learner_pars(n, pars)) for n in names]


def select_workflows(spec: str) -> list[str]:
    """'all' | comma separated names | comma separated learner/strategy filters
    (e.g. 'lm,rpart' or 'svm_UNDER' or 'SMOTE')."""
    if spec in (None, "", "all"):
        return list(WORKFLOW_NAMES)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in WORKFLOW_NAMES:
            out.append(tok)
        else:
            hits = [n for n in WORKFLOW_NAMES if tok in n]
            if not hits:
                raise SystemExit(f"unknown workflow filter: {tok}")
            out.extend(h for h in hits if h not in out)
    return out


def add_common_args(p: argparse.ArgumentParser, time_default: bool = False):
    p.add_argument("--dataset", "-d", type=int, default=1, help="data set number 1..24 (R: i <- 1)")
    p.add_argument("--nreps", type=int, default=50, help="Monte Carlo repetitions (R: nReps=50)")
    p.add_argument("--sztrain", type=float, default=None, help="train size (default: 0.5; 0.1 for 21/22; 0.2 for 23/24)")
    p.add_argument("--sztest", type=float, default=None, help="test size (default: 0.25; 0.05 for 21/22; 0.1 for 23/24)")
    p.add_argument("--seed", type=int, default=1234, help="Monte Carlo seed (performanceEstimation default)")
    p.add_argument("--embed", type=int, default=10, help="embedding dimension (R: create.data(..., 10))")
    p.add_argument("--na", choices=["knn", "complete", "none"], default="knn",
                   help="NA handling for data sets with missing values (R: knnImputation / complete.cases)")
    p.add_argument("--workflows", "-w", default="all", help="'all', names, or filters such as 'lm,rpart' or 'SMOTE'")
    p.add_argument("--n-jobs", type=int, default=1, help="parallel jobs over the Monte Carlo repetitions")
    p.add_argument("--out", "-o", default=None, help="output pickle (default: results/exp[.time]_ds<i>.pkl)")
    p.add_argument("--checkpoint-dir", default=None, help="directory to checkpoint every finished workflow")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--pars", choices=["paper", "example"], default="paper",
                   help="learner parameters: 'paper' = Table 7 of the article for the data set (default); "
                        "'example' = the example parametrisation of Exps.R")
    for k, v in DEFAULT_PARS.items():
        p.add_argument(f"--{k}", type=type(v), default=None, help=f"override the learner parameter {k}")
    return p


def learner_pars_from_args(args) -> dict:
    pars = paper_pars(args.dataset) if args.pars == "paper" else dict(DEFAULT_PARS)
    for k in DEFAULT_PARS:
        v = getattr(args, k, None)
        if v is not None:
            pars[k] = v
    return pars


def estimation_task_from_args(args) -> EstimationTask:
    tr, te = default_sizes(args.dataset)
    sztrain = tr if args.sztrain is None else args.sztrain
    sztest = te if args.sztest is None else args.sztest
    return EstimationTask(method=MonteCarlo(nReps=args.nreps, szTrain=sztrain, szTest=sztest, seed=args.seed))
