Resampling Strategies for Imbalanced Time Series Forecasting (Code + Data) - **Python port**

This repository is a Python re-implementation of
[nunompmoniz/TSResampStrat_JDSA2017](https://github.com/nunompmoniz/TSResampStrat_JDSA2017),
the code and data used to execute the experimental process (originally designed
for **R**) of the article "Resampling Strategies for Imbalanced Time Series
Forecasting" by Nuno Moniz, Paula Branco and Luís Torgo (International Journal
of Data Science and Analytics, Springer).

The structure mirrors the original repository:

* `Data/` - the 24 time series (original `.Rdata` plus a converted pickle) and the data description;
* `Py_Code/` - the Python counterparts of `R_Code/` (see `Py_Code/README.md`) and the package `tsresamp`;
* `tests/` - unit tests of every component (they run in seconds, without fitting the full experiments).

**Installation**

```
python3 -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install -r requirements.txt
python Py_Code/convert_rdata.py                          # Data/*.Rdata -> Data/*.pkl (pure python, no R needed)
python -m pytest tests -q
```

**Running the experiments**

```
python Py_Code/Exps.py --dataset 1                       # 52 workflows x 50 Monte Carlo repetitions
python Py_Code/Exps.py --dataset 21 --n-jobs 8 --checkpoint-dir results/ckpt
python Py_Code/GetResults.py results/exp_ds1.pkl
python Py_Code/Exps_Time.py --dataset 1
python Py_Code/GetResults_RunTime.py results/exp.time_ds1.pkl
python Py_Code/PairedComparisons.py results/exp_ds*.pkl
python Py_Code/OptParmsSearch.py --dataset 1
```

The full experiments are computationally heavy (in particular data sets 21-24)
and are meant to run as a batch job (the SLURM job used on the Apuana cluster of CIn/UFPE is in
the standalone repository `TSResampStrat_Python`, `cluster/apuana/`); `--checkpoint-dir` saves every finished
workflow so that a job can be resumed, `--n-jobs` parallelises the Monte Carlo
repetitions and `--workflows` restricts the run (e.g. `-w lm,rpart` or `-w SMOTE`).

**Requirements** - Python >= 3.10 with numpy, pandas, scipy, scikit-learn,
statsmodels, pmdarima, rdata and joblib (`requirements.txt`). The R packages
`uba`, `UBL`, `DMwR`, `performanceEstimation`, `e1071`, `earth`, `randomForest`,
`rpart` and `forecast` are **not** needed: the parts of them used by the
experiments were ported (see below).

**What was ported and how**

| R | Python |
|---|---|
| `uba::phi.control(method="extremes")`, `phi`, `loss.control`, `util` (precision, recall, F-measure of the utility-based framework) | `tsresamp/uba.py` - line-by-line port of `phi.R`, `pchip.c`, `bump.c`, `util.c`, `utilMetrics.c` |
| `randUnderRegress{B,T,TPhi}`, `randOverRegress{B,T,TPhi}`, `smoteRegress{B,T,TPhi}` (Exps.R) and `UBL::neighbours` | `tsresamp/resampling.py` |
| `lm` | `sklearn.linear_model.LinearRegression` |
| `e1071::svm(cost, gamma)` | `sklearn.svm.SVR` (rbf, epsilon=0.1) on standardised x and y, as `svm(scale=TRUE)` |
| `earth(nk, degree, thresh)` | `tsresamp/mars.py`, a port of the forward pass of `earth.c` (earth 5.3.6: minspan/endspan knot placement, candidate linear terms, `MIN_BX_SOS`, Fast MARS queue, stopping rules) and of the pruning pass (leaps `BAKWRD` backward elimination + GCV); verified to give the same terms, knots and coefficients as `earth` on 74 Monte Carlo windows and resampled training sets of 8 data sets (`tests/test_mars.py`, fixtures in `tests/data/earth/`). py-earth no longer builds |
| `randomForest(mtry, ntree)` | `RandomForestRegressor(max_features=mtry, n_estimators=ntree, min_samples_split=6, min_samples_leaf=1)` - regression `nodesize=5` in regTree.c means "do not split nodes with 5 or fewer cases", leaves can be smaller. The predictors are rank-transformed before every sklearn tree (`TreeRanks` in `models.py`): sklearn never splits between two values closer than 1e-7 (an absolute tolerance), R does; DS1 has 30 such pairs per column. Checked against `randomForest` on the same training sets: mean F1 equal within the forest's own noise (DS1, DS2/DS3 with over-sampling) |
| `rpart(minsplit, cp)` | `DecisionTreeRegressor(min_samples_split=minsplit, min_samples_leaf=round(minsplit/3), ccp_alpha=cp*var(y), max_depth=30)` - in rpart's source (`partition.c`) `cp` is minimal cost-complexity pruning: a subtree is kept only if its weakest-link complexity (SS decrease per split) exceeds cp * SS_root; in sklearn's units that is `ccp_alpha = cp * var(y)`. Checked on the Monte Carlo windows: same number of leaves as R in 28/50 (DS1) and 36/50 (DS10) windows, mean F1 within 0.003 of R on the same windows; the residual differences are in the growth phase: with the rank transform (see the `randomForest` row) 38/50 DS1 windows have the same number of leaves; on DS10 (an integer-valued series) ties in the split criterion between predictors are broken by column order in rpart and by a random feature order in sklearn, which cannot be changed |
| `forecast::auto.arima` + `fitted(Arima(data, model=m))` | `pmdarima.auto_arima(seasonal=False, ic="aicc")` + statsmodels `apply()` (same model re-applied to train+test, one-step-ahead fitted values) |
| `mc.BDES` (bagged trees on embeddings with mean/var statistics) | `tsresamp/models.py: mc_BDES` |
| `performanceEstimation` / `MonteCarlo(nReps, szTrain, szTest)` / `getIterationsInfo` | `tsresamp/estimation.py` (same split rule: sorted random starting points, train window before, test window after) |
| `pairedComparisons` (Wilcoxon signed rank, paired t-test, Friedman/Nemenyi/Bonferroni-Dunn) and `WLdef` | `tsresamp/analysis.py` (R's `wilcox.test` conventions: exact for n < 50 without ties/zeros, else normal approximation with continuity correction) |

**Parameters** - by default `Exps.py` uses, for each data set, the optimal
parametrisation of Table 7 of the article (`--pars paper`); `--pars example`
selects the example parametrisation written in `Exps.R` (cost=150,
gamma=0.001, nk=17, degree=2, thresh=0.001, mtry=7, ntree=500, minsplit=10,
cp=0.001), and any parameter can be overridden (e.g. `--ntree 1500`).
`Py_Code/ReplicateTable6.py` repeats the experiment of Table 6 of the article
(SVM with the parameters of Table 8 on data sets 4, 10 and 12) and prints the
port's values next to the published ones.

**Documented differences with respect to the R code**

1. *Random numbers.* R's random stream (`set.seed(1234)` in `performanceEstimation`, `sample`, `runif`) cannot be reproduced in Python; the Monte Carlo starting points and all the resampling draws use numpy's default generator seeded from the same seed. Results are reproducible run to run, but not bit-identical to R.
2. *Missing values.* Data sets 12, 13, 23 and 24 contain NAs. `Exps.R` has `knnImputation(ds)` and `ds[complete.cases(ds),]` commented out; here `--na knn` (default, a port of `DMwR::knnImputation`), `--na complete` or `--na none` select the treatment.
3. *`smote.exsRegressT` / `smote.exsRegressTPhi`.* The R code fills the numeric matrix `T` *before* re-ordering the cases by time and computes the neighbours *after*, so seed cases and neighbour indices refer to different orderings. The port reproduces this by default (`r_index_quirk=True`), because it is what the published results were computed with (verified against the original R code, see `R_replication/`); `r_index_quirk=False` re-orders first (the evident intent). Everything else in the SMOTE generation - including the target interpolation weights computed only from the last non-target column, a consequence of the loop overwriting `d1`/`d2` in the original (and in DMwR/UBL) - is reproduced as is.
4. *SMOTE percentages `u`/`o`.* `OptParmsSearch.R` calls the SMOTE variants
   with `C.perc=list(un, ov)`, which the R functions only accept when the data
   set has exactly two relevance bumps; data sets with extreme values on both
   sides (three bumps, e.g. DS4 and DS12) would stop with an error. Following
   Algorithm 5 of the article, the port applies `u` to every bump of normal
   cases and `o` to every bump of rare cases (`C_perc={"un": u, "ov": o}`).
5. *`mc.BDES`.* `embedStats` adds the row-wise mean and variance of **all** the embedding columns, including the target (`V10`), for both train and test - the test target therefore leaks into two features. This is reproduced faithfully because it is what the published experiments ran; it is flagged here so that users are aware of it.
6. *Learners.* `lm` and `svm` are exact equivalents (least squares; the same libsvm with the same scaling), and MARS is a port of `earth` itself (same model for the same data, see the table and `COMPARISON.md` section 10). `rf`, `rpart` and `auto.arima` are the closest scikit-learn / pmdarima equivalents; their parameters were mapped to the same meaning (`cp` -> `ccp_alpha = cp * var(y)`, `mtry` -> `max_features`, `nodesize` -> `min_samples_split`, etc.) but the implementations differ (tree-growing details, bootstrap draws, ARIMA order selection). Until 2026-09-22 the MARS was an own numpy implementation (quantile knots, no `MIN_BX_SOS` rule); it produced F1 values up to 0.3 above `earth` on some baselines and under-sampled workflows (DS10, DS15, DS16, DS19), which is why it was replaced by the port.
7. *Failures.* A workflow iteration that raises (for instance a resampler with too few relevant cases in one split) is recorded with NaN scores and the error message, instead of aborting the whole experiment; `GetResults.py` averages over the remaining iterations and `summary()` reports the number of failed ones.
8. *`pairedComparisons`.* The Friedman critical value uses R's `df()` (the F density) exactly as in the original source.
9. *Times.* `traintime` / `trainpredtime` are CPU process times (R's `proc.time()[1]`, user time); wall-clock times are also stored (`*_elapsed`).
10. *Timestamps.* Series 5-8, 21 and 22 have a few duplicated timestamps; rows are addressed by position, so this has no effect (in R the row names would have to be unique).

**Comparison with the article and with the original R code** - see `COMPARISON.md` (relevance function, data set 1 and Table 6 replicated against the published values) and `REVIEW.md` (what is a translation, what is a substitution, and why the remaining differences exist). The original R code was also run locally (folder `R_replication/`, R 4.6.1 with the original packages) and compared three-way, article x original R x Python: the port reproduces the original code to within 0.01 of F1 on average (156 workflow/data set pairs), while the published code itself departs from several published numbers (DS4, DS10, Table 6).

**Data** - identical to the original repository (see `Data/README.md` in the original repository). The article's additional figures are in the original repository (`Figures/`) and are not copied here.

## Resultados do run completo (Apuana, 18/09/2026)

Os 24 data sets x 52 workflows x 50 repetições rodaram no cluster Apuana em 9 h (job 15477,
48 CPUs). Sumários versionados em `results_cluster/` (F1 por data set e workflow, tabelas de
vitórias/derrotas, comparação com as Tabelas 3, 4 e 5 do artigo e com a replicação
independente em R). Leitura dos resultados: `COMPARISON.md`, seção 9. O job SLURM e o passo a passo do cluster
ficam no repositório standalone `TSResampStrat_Python` (`cluster/apuana/`).

**Auditoria de 22/09/2026** - learner a learner, R e Python nos mesmos conjuntos de treino: o MARS
passou a ser um port do `earth` (antes era a única divergência sistemática), o `cp` do `rpart` passou
a poda por custo-complexidade e os preditores das árvores passam para postos. Detalhes em
`COMPARISON.md` seção 10; ferramentas em `tools/` (`merge_workflow_results.py`,
`regenerate_results_cluster.sh`); proveniência dos resultados em `results_cluster/PROVENANCE.md`.

**Relatório de replicação** - `RELATORIO_REPLICACAO.md`: o que o artigo afirma, o protocolo
seguido passo a passo, os resultados originais lado a lado com os nossos (Fig. 7, Tabelas 3 a 6,
hipóteses H1 a H3) e a explicação de cada diferença.
