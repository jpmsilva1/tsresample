"""
Port of GetResults_RunTime.R: mean training time, prediction time and total
time of every workflow (first 10 iterations, as in the R script).

    python Py_Code/GetResults_RunTime.py results/exp.time_ds1.pkl [...] [--elapsed] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsresamp.analysis import MODELS, get_results_runtime  # noqa: E402
from tsresamp.estimation import ComparisonResults  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="pickle(s) written by Exps_Time.py / Exps.py")
    p.add_argument("--nreps", type=int, default=10, help="iterations to average (R: 1:10)")
    p.add_argument("--elapsed", action="store_true", help="use wall-clock time instead of CPU time")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)
    tables = []
    for f in args.results:
        exp = ComparisonResults.load(f)
        for task in exp.task_names():
            names = [m for m in MODELS if m in exp.workflow_names(task)] or exp.workflow_names(task)
            tab = get_results_runtime(exp, models=names, task=task, n_reps=args.nreps,
                                      key="traintime_elapsed" if args.elapsed else "traintime")
            tab.insert(0, "task", task)
            tables.append(tab)
    globalres_time = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.max_rows", 2000, "display.width", 140):
        print(globalres_time.to_string(index=False))
    if args.csv:
        globalres_time.to_csv(args.csv, index=False)
    return globalres_time


if __name__ == "__main__":
    main()
