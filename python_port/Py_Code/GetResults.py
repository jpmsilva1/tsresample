"""
Port of GetResults.R: table with the mean prec / rec / F1 of every workflow
over the Monte Carlo iterations (R object `globalres`).

    python Py_Code/GetResults.py results/exp_ds1.pkl [results/exp_ds2.pkl ...] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsresamp.analysis import MODELS, get_results  # noqa: E402
from tsresamp.estimation import ComparisonResults  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="pickle(s) written by Exps.py")
    p.add_argument("--nreps", type=int, default=50, help="iterations to average (R: 1:50)")
    p.add_argument("--csv", default=None, help="write the table to this CSV file")
    args = p.parse_args(argv)
    tables = []
    for f in args.results:
        exp = ComparisonResults.load(f)
        for task in exp.task_names():
            names = [m for m in MODELS if m in exp.workflow_names(task)] or exp.workflow_names(task)
            tab = get_results(exp, models=names, task=task, n_reps=args.nreps)
            tab.insert(0, "task", task)
            tables.append(tab)
    globalres = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.max_rows", 2000, "display.width", 140):
        print(globalres.to_string(index=False))
    if args.csv:
        globalres.to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return globalres


if __name__ == "__main__":
    main()
