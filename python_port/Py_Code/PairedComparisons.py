"""
Port of PairedComparisons.R: paired comparisons (Wilcoxon signed rank test,
p-value < 0.05) of the resampling strategies against their baselines,
counted over all the data sets (tasks).

    python Py_Code/PairedComparisons.py results/exp_ds*.pkl

The result pickles of the different data sets are merged into one
multi-task object (R: `exp` with all the tasks).  The tables printed are the
ones requested in the R script, e.g. WLdef(exp,"mc.lm","F1",0.05)[c(1,4,7),].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsresamp.analysis import WLdef, common_workflows  # noqa: E402
from tsresamp.estimation import ComparisonResults  # noqa: E402

# The R script selects rows of the WL matrix by position, e.g.
#   WLdef(exp,"mc.lm","F1",0.05)[c(1,4,7),]        -> mc.lm_UNDERB, mc.lm_OVERB, mc.lm_SMOTEB
#   WLdef(exp,"mc.lm_UNDERB","F1",0.05)[c(2,3),]   -> mc.lm_UNDERT, mc.lm_UNDERTPhi
#   WLdef(exp,"mc.lm_OVERB","F1",0.05)[c(5,6),]    -> mc.lm_OVERT, mc.lm_OVERTPhi
#   WLdef(exp,"mc.lm_SMOTEB","F1",0.05)[c(8,9),]   -> mc.lm_SMOTET, mc.lm_SMOTETPhi
# (and the same for svm, mars, rf, rpart); mc.arima and mc.BDES print the whole table.
# The same rows are selected here by name, which also works with a subset of workflows.
LEARNERS = ("lm", "svm", "mars", "rf", "rpart")


def requests():
    out = []
    for L in LEARNERS:
        out.append((f"mc.{L}", [f"mc.{L}_UNDERB", f"mc.{L}_OVERB", f"mc.{L}_SMOTEB"]))
    for L in LEARNERS:
        out.append((f"mc.{L}_UNDERB", [f"mc.{L}_UNDERT", f"mc.{L}_UNDERTPhi"]))
    for L in LEARNERS:
        out.append((f"mc.{L}_OVERB", [f"mc.{L}_OVERT", f"mc.{L}_OVERTPhi"]))
    for L in LEARNERS:
        out.append((f"mc.{L}_SMOTEB", [f"mc.{L}_SMOTET", f"mc.{L}_SMOTETPhi"]))
    out += [("mc.arima", None), ("mc.BDES", None)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="pickles written by Exps.py (one per data set)")
    p.add_argument("--measure", default="F1", choices=["prec", "rec", "F1"])
    p.add_argument("--sig", type=float, default=0.05)
    p.add_argument("--all", action="store_true", help="print the full WL table for every base instead of the R selections")
    args = p.parse_args(argv)
    exp = ComparisonResults.merge_all([ComparisonResults.load(f) for f in args.results])
    print(f"{len(exp.task_names())} task(s): {exp.task_names()}")
    names = common_workflows(exp)
    print(f"{len(names)} workflow(s) common to all tasks")
    out = {}
    for base, rows in requests():
        if base not in names:
            continue
        WL = WLdef(exp, base, args.measure, args.sig)
        if rows is not None and not args.all:
            rows = [r for r in rows if r in WL.index]
            if not rows:
                continue
            WL = WL.loc[rows]
        print(f"\nWLdef(exp, \"{base}\", \"{args.measure}\", {args.sig})")
        print(WL.to_string())
        out[base] = WL
    return out


if __name__ == "__main__":
    main()
