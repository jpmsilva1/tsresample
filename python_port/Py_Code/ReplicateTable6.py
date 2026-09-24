"""
Replication of Table 6 of the article (Moniz, Branco & Torgo, JDSA 2017):
SVM models and resampling strategies with the optimised parameters of
Tables 7 (baseline) and 8 (resampling) on data sets 4, 10 and 12, mean
utility-based F1 over 50 Monte Carlo repetitions (50%/25%).

    python Py_Code/ReplicateTable6.py [--datasets 4,10,12] [--nreps 50] [--n-jobs 4]

Not part of the original R code; added to check the port against the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exps_common import RESULTS_DIR, load_embedded  # noqa: E402
from tsresamp.analysis import get_results  # noqa: E402
from tsresamp.estimation import EstimationTask, MonteCarlo, Workflow, performance_estimation  # noqa: E402

# Table 7 (baseline svm) and Table 8 (cost, gamma, under u / over o) of the article
BASELINE = {4: dict(cost=150, gamma=0.01), 10: dict(cost=300, gamma=0.01), 12: dict(cost=300, gamma=0.01)}
TABLE8 = {
    4: {"UNDERB": (10, 0.01, .4), "UNDERT": (10, 0.01, .4), "UNDERTPhi": (10, 0.01, .8),
        "OVERB": (10, 0.001, 5), "OVERT": (150, 0.001, 2), "OVERTPhi": (150, 0.01, 2),
        "SMOTEB": (150, 0.001, .8, 2), "SMOTET": (150, 0.001, .6, 2), "SMOTETPhi": (10, 0.001, .8, 2)},
    10: {"UNDERB": (10, 0.001, .1), "UNDERT": (150, 0.001, .1), "UNDERTPhi": (300, 0.001, .1),
         "OVERB": (10, 0.001, 2), "OVERT": (150, 0.001, 2), "OVERTPhi": (150, 0.001, 2),
         "SMOTEB": (10, 0.001, .8, 10), "SMOTET": (10, 0.001, .6, 5), "SMOTETPhi": (300, 0.001, .6, 3)},
    12: {"UNDERB": (150, 0.001, .2), "UNDERT": (300, 0.001, .2), "UNDERTPhi": (150, 0.001, .2),
         "OVERB": (10, 0.001, 10), "OVERT": (150, 0.001, 3), "OVERTPhi": (150, 0.001, 3),
         "SMOTEB": (10, 0.001, .2, 3), "SMOTET": (10, 0.001, .8, 5), "SMOTETPhi": (150, 0.001, .4, 2)},
}
# Table 6 (paper): mean F1
PAPER = {
    "mc.svm": (0.584, 0.638, 0.554), "mc.svm_UNDERB": (0.668, 0.652, 0.610), "mc.svm_UNDERT": (0.659, 0.643, 0.614),
    "mc.svm_UNDERTPhi": (0.651, 0.647, 0.630), "mc.svm_OVERB": (0.653, 0.651, 0.611), "mc.svm_OVERT": (0.650, 0.652, 0.615),
    "mc.svm_OVERTPhi": (0.651, 0.652, 0.611), "mc.svm_SMOTEB": (0.662, 0.675, 0.609), "mc.svm_SMOTET": (0.656, 0.698, 0.600),
    "mc.svm_SMOTETPhi": (0.649, 0.721, 0.620),
}


def workflows_for(ds: int) -> list[Workflow]:
    wfs = [Workflow("mc.svm", dict(BASELINE[ds]))]
    for key, p in TABLE8[ds].items():
        pars = dict(cost=p[0], gamma=p[1])
        if key.startswith("UNDER"):
            pars["un"] = p[2]
        elif key.startswith("OVER"):
            pars["ov"] = p[2]
        else:
            pars["un"], pars["ov"] = p[2], p[3]
        wfs.append(Workflow(f"mc.svm_{key}", pars))
    return wfs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="4,10,12")
    p.add_argument("--nreps", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--na", default="knn")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", default="", help="suffix for the output files")
    args = p.parse_args(argv)
    cols = {}
    for ds in [int(x) for x in args.datasets.split(",")]:
        data = load_embedded(ds, na=args.na)
        et = EstimationTask(method=MonteCarlo(nReps=args.nreps, szTrain=.5, szTest=.25, seed=args.seed))
        exp = performance_estimation(f"ds{ds}", data, workflows_for(ds), et, n_jobs=args.n_jobs, verbose=False)
        exp.save(RESULTS_DIR / f"table6_ds{ds}{args.tag}.pkl")
        tab = get_results(exp, n_reps=args.nreps).set_index("model")
        cols[f"DS{ds} port"] = tab["F1"].round(3)
        cols[f"DS{ds} paper"] = pd.Series({m: v[[4, 10, 12].index(ds)] for m, v in PAPER.items()})
        cols[f"DS{ds} failed"] = exp.summary().set_index("workflow")["failed"]
    out = pd.DataFrame(cols)
    with pd.option_context("display.width", 160):
        print(out.to_string())
    out.to_csv(RESULTS_DIR / f"table6_comparison{args.tag}.csv")
    return out


if __name__ == "__main__":
    main()
