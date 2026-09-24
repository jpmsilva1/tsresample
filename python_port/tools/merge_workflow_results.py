"""Merge per-data-set result pickles: start from the full run (results/apuana/exp_ds<i>.pkl) and
replace the workflows found in the override directory (results/<dir>/exp_ds<i>_*.pkl, e.g. the
mc.mars* workflows re-run locally with the earth port).  Writes results/<out>/exp_ds<i>.pkl so
that GetResults.py, PairedComparisons.py and tools/compare_full_run.py work unchanged.

    .venv/bin/python tools/merge_workflow_results.py results/apuana results/mars_v2 results/apuana_v2
"""
import glob, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Py_Code"))
from tsresamp.estimation import ComparisonResults  # noqa: E402

base, override, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(base, "exp_ds*.pkl")), key=lambda f: int(re.search(r"ds(\d+)", f).group(1))):
    ds = int(re.search(r"ds(\d+)", f).group(1))
    res = ComparisonResults.load(f)
    replaced = []
    for g in sorted(glob.glob(os.path.join(override, f"exp_ds{ds}_*.pkl"))):
        o = ComparisonResults.load(g)
        for t in o.task_names():
            for wf, wres in o.tasks[t].items():
                if t in res.tasks:
                    res.tasks[t][wf] = wres; replaced.append(wf)
    res.save(os.path.join(out, f"exp_ds{ds}.pkl"))
    print(f"ds{ds}: {len(replaced)} workflows replaced ({', '.join(sorted(replaced)) if replaced else '-'})")
