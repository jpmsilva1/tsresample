"""
Port of Exps.R: run the experiments for one data set.

    python Py_Code/Exps.py --dataset 1                # all 52 workflows, 50 Monte Carlo reps
    python Py_Code/Exps.py -d 21 --n-jobs 8           # 10%/5% sizes chosen automatically
    python Py_Code/Exps.py -d 1 -w lm,rpart --nreps 2 # quick check

The result (R object `exp`) is saved as results/exp_ds<i>.pkl and can be
summarised with GetResults.py.  Every workflow also records training and
prediction times, so Exps_Time.py is the same run under another file name.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exps_common import (RESULTS_DIR, add_common_args, default_workflows,  # noqa: E402
                         estimation_task_from_args, learner_pars_from_args, load_embedded, select_workflows)
from tsresamp.analysis import get_results  # noqa: E402
from tsresamp.estimation import performance_estimation  # noqa: E402


def main(argv=None, time_run: bool = False):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args(argv)

    ds = load_embedded(args.dataset, embed=args.embed, na=args.na)
    et = estimation_task_from_args(args)
    pars = learner_pars_from_args(args)
    wfs = default_workflows(pars, select_workflows(args.workflows))
    tag = "exp.time" if time_run else "exp"
    out = Path(args.out) if args.out else RESULTS_DIR / f"{tag}_ds{args.dataset}.pkl"

    print(f"Data set {args.dataset}: {len(ds)} rows after embedding ({args.embed}); NA handling = {args.na}")
    print(f"Monte Carlo: nReps={et.method.nReps} szTrain={et.method.szTrain} szTest={et.method.szTest} seed={et.method.seed}")
    print(f"{len(wfs)} workflows; parameters ({args.pars}) {pars}")
    t0 = time.perf_counter()
    exp = performance_estimation(f"ds{args.dataset}", ds, wfs, et, n_jobs=args.n_jobs, verbose=not args.quiet,
                                 checkpoint_dir=args.checkpoint_dir)
    exp.save(out)
    print(f"\nsaved {out}  ({time.perf_counter() - t0:.1f}s)")
    with __import__("pandas").option_context("display.max_rows", 100, "display.width", 120):
        print(get_results(exp, n_reps=et.method.nReps).to_string(index=False))
    return exp


if __name__ == "__main__":
    main()
