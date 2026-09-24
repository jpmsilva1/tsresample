"""
Port of OptParmsSearch.R: optimal parametrisation search for the SVM
workflows (cost, gamma and the under/over-sampling percentages), with
10 Monte Carlo repetitions (50%/25%).

    python Py_Code/OptParmsSearch.py --dataset 1 [--n-jobs 8]

Writes results/exp.best_ds<i>.pkl and prints the table `globalres`
(mean prec, mean rec, median F1 - as in the R script).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exps_common import RESULTS_DIR, load_embedded  # noqa: E402
from tsresamp.analysis import get_results  # noqa: E402
from tsresamp.estimation import EstimationTask, MonteCarlo, Workflow, performance_estimation, workflow_variants  # noqa: E402


def build_workflows() -> list[Workflow]:
    wfs = [Workflow("mc.svm", dict(cost=150, gamma=0.01))]
    costs, gammas = [10, 150, 300], [0.01, 0.001]
    for s in ("UNDERB", "UNDERT", "UNDERTPhi"):
        wfs += workflow_variants(f"mc.svm_{s}", cost=costs, gamma=gammas, un=[.1, .2, .4, .6, .8])
    for s in ("OVERB", "OVERT", "OVERTPhi"):
        wfs += workflow_variants(f"mc.svm_{s}", cost=costs, gamma=gammas, ov=[2, 3, 5, 10])
    for s in ("SMOTEB", "SMOTET", "SMOTETPhi"):
        wfs += workflow_variants(f"mc.svm_{s}", cost=costs, gamma=gammas, un=[.05, .1, .2, .4, .6, .8], ov=[2, 3, 5, 10])
    return wfs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", "-d", type=int, default=1)
    p.add_argument("--nreps", type=int, default=10)
    p.add_argument("--sztrain", type=float, default=0.5)
    p.add_argument("--sztest", type=float, default=0.25)
    p.add_argument("--na", choices=["knn", "complete", "none"], default="knn")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N workflows (for testing)")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    ds = load_embedded(args.dataset, na=args.na)
    wfs = build_workflows()
    if args.limit:
        wfs = wfs[:args.limit]
    et = EstimationTask(method=MonteCarlo(nReps=args.nreps, szTrain=args.sztrain, szTest=args.sztest))
    out = Path(args.out) if args.out else RESULTS_DIR / f"exp.best_ds{args.dataset}.pkl"
    print(f"{len(wfs)} workflow variants (R: 1 + 3*30 + 3*24 + 3*144 = 595)")
    t0 = time.perf_counter()
    exp_best = performance_estimation(f"ds{args.dataset}", ds, wfs, et, n_jobs=args.n_jobs, verbose=not args.quiet,
                                      checkpoint_dir=args.checkpoint_dir)
    exp_best.save(out)
    print(f"saved {out} ({time.perf_counter() - t0:.1f}s)")
    globalres = get_results(exp_best, n_reps=args.nreps, summary="mean", F1_summary="median")
    globalres["pars"] = [str(w.pars) for w in wfs]
    with pd.option_context("display.max_rows", 1000, "display.width", 160):
        print(globalres.to_string(index=False))
    return globalres


if __name__ == "__main__":
    main()
