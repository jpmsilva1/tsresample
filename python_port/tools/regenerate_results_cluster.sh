#!/bin/bash
# Rebuilds every file of results_cluster/ from the per-data-set result pickles:
#   results/apuana/exp_ds<i>.pkl      the full cluster run (job 15477, 2026-09-18)
#   results/mars_v2/  rpart_v2/  rf_v2/   workflows re-run locally after the audit of 2026-09-22
#     (mc.mars* with the earth port; mc.rpart* and mc.BDES with cost-complexity pruning and
#      rank-transformed predictors; mc.rf* with rank-transformed predictors, 24 data sets)
# Usage:  bash tools/regenerate_results_cluster.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results_cluster
MERGED=results/apuana_v2
rm -rf "$MERGED"
$PY tools/merge_workflow_results.py results/apuana results/mars_v2 "$MERGED" > /dev/null
for d in results/rpart_v2 results/rf_v2; do
  [ -d "$d" ] && $PY tools/merge_workflow_results.py "$MERGED" "$d" "$MERGED" > /dev/null
done
$PY tools/merge_workflow_results.py "$MERGED" results/mars_v2 "$MERGED" | sed 's/^/merge: /' | head -3
$PY tools/compare_full_run.py "$MERGED" --out "$OUT" > /dev/null
$PY Py_Code/GetResults.py "$MERGED"/exp_ds*.pkl --csv "$OUT/globalres_all.csv" > /dev/null
$PY Py_Code/PairedComparisons.py "$MERGED"/exp_ds*.pkl > "$OUT/paired_comparisons.txt"
$PY - <<'PYEOF'
import pandas as pd
w = pd.read_csv("results_cluster/F1_by_dataset.csv", index_col=0)
L = ["lm", "svm", "mars", "rf", "rpart"]; S = ["UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT", "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
rows = []
for l in L:
    rows.append([l, round(w[f"mc.{l}"].mean(), 4)] + [round(w[f"mc.{l}_{s}"].mean(), 4) for s in S])
pd.DataFrame(rows, columns=["", "baseline"] + S).to_csv("results_cluster/meanF1_model_strategy.csv", index=False)
PYEOF
$PY tools/compare_fig7_all.py > "$OUT/comparison_fig7.txt" 2>/dev/null || echo "compare_fig7_all.py failed (needs the article PDF); kept the previous comparison_fig7.txt"
echo "results_cluster/ regenerated from $MERGED"
