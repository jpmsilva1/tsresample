Python port of the code used to load, model, run experiments and obtain results.
Every R script has a Python counterpart with the same name; the shared code
lives in the package `tsresamp/`.

| R file | Python file | Notes |
|---|---|---|
| `Exps.R` | `Exps.py` | `python Py_Code/Exps.py --dataset i`. Monte Carlo estimates, 50 repetitions, 50%/25% (10%/5% for data sets 21 and 22, 20%/10% for 23 and 24 - chosen automatically). Saves `results/exp_ds<i>.pkl`. |
| `GetResults.R` | `GetResults.py` | Table with the overall results (mean prec / rec / F1). |
| `Exps_Time.R` | `Exps_Time.py` | Same experiments recording training / prediction time; saves `results/exp.time_ds<i>.pkl`. |
| `GetResults_RunTime.R` | `GetResults_RunTime.py` | Table with the time required to train / predict. |
| `PairedComparisons.R` | `PairedComparisons.py` | Paired comparisons with Wilcoxon signed rank tests (p-value < 0.05) over all data sets. |
| `OptParmsSearch.R` | `OptParmsSearch.py` | Optimal parametrisation search for the SVM workflows. |
| (data loading) | `convert_rdata.py` | Converts `Data/*.Rdata` into a pickle of pandas Series (pure Python, no R needed). |

Package `tsresamp` (R source -> module):

| R | Python module |
|---|---|
| `uba` package: `phi.control`, `phi`, `loss.control`, `util` (P, R, Fm) | `tsresamp/uba.py` |
| `create.data`, `load(...Rdata)`, `knnImputation` | `tsresamp/data.py` |
| `randUnderRegress{B,T,TPhi}`, `randOverRegress{B,T,TPhi}`, `smoteRegress{B,T,TPhi}`, `UBL::neighbours` | `tsresamp/resampling.py` |
| `earth` (MARS) | `tsresamp/mars.py` (port of `earth.c` 5.3.6 and of the `leaps` backward pruning) |
| `mc.*` workflows, `eval.stats`, `mc.arima`, `mc.BDES` | `tsresamp/models.py` |
| `performanceEstimation`, `MonteCarlo`, `Workflow`, `workflowVariants` | `tsresamp/estimation.py` |
| `pairedComparisons`, `WLdef`, result tables | `tsresamp/analysis.py` |

Quick checks (no full experiment needed):

```
python -m pytest tests -q                              # unit tests of every module
python Py_Code/Exps.py -d 1 -w lm,rpart --nreps 2      # 2 repetitions, cheap learners
python Py_Code/GetResults.py results/exp_ds1.pkl
```
