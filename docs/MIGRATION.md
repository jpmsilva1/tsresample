# Migrating from `imbalance_eval`

`tsresample` includes everything `imbalance_eval` did (the paper's Table 1 imbalance summary,
single-file and manifest batch modes), plus the resampling strategies and the metrics.
Most invocations translate unchanged:

```bash
# before
python imbalance_eval.py day.csv --target temp --date-col dteday --diff
python imbalance_eval.py --config manifest.csv --output table1.csv
# after (pip install "tsresample[io]")
tsresample day.csv --target temp --date-col dteday --diff
tsresample --config manifest.csv --output table1.csv
```

The flags `--target --id --name --granularity --date-col --config --threshold --k
--no-embed --diff --output` mean the same thing. The manifest format is the same too:
`id,name,granularity,path,target[,threshold,k]`, with paths relative to the working
directory. So is the output table (`ID Dataset N Granularity n_normal n_rare IR %Rare`,
where `IR = n_rare / n_normal` and both are rounded to 2 decimals). `--k K` still
counts rare cases on `series[K:]`.

## Intentional behavioural differences

| | `imbalance_eval` | `tsresample` | Effect on numbers |
|---|---|---|---|
| **Missing values** | `KNNImputer` on the CSV's numeric columns. On a one-column series this fills every gap with the **column mean**. | Lag-window kNN on the series (k = 10, `exp(−d)` weights; ADR-0010). `--impute drop` and `--impute none` are also available. | Series with gaps change. DS12 %Rare goes from 14.65 to **10.99** (paper: 11.0). Gap-free series are unaffected. |
| **φ source** | `ImbalancedLearningRegression` (`iblr`) | The library's own φ (SPEC §4.1): Tukey hinges, whisker ends, cubic Hermite. | None expected. Both reproduce R's control points in 85/90 recorded splits and %Rare to 0.17 pp (ADR-0010 amendment). |
| **Rare threshold** | `φ ≥ t` | `φ ≥ t`, the same rule the metrics use. | None. Strict `>` gives identical %Rare on all 18 NA-free datasets. |
| **`--xtrm-type`, `--coef`** | `both/high/low`; any coefficient | Only `both` and `1.5`. Any other value exits with code 2 and a message. | φ follows the paper's R `extremes` method, which has no such options. |
| **Every φ is 0** | Prints a warning and reports `n_rare = 0` | Reports `n_rare = 0` with no warning. A resampler run on the same data warns ("no rare bump"). | None. |
| **Dependencies** | `iblr`, pandas, scikit-learn; patched `np.quantile` for NumPy 2 | numpy, scipy and scikit-learn. pandas only with the `[io]` extra. No monkey-patching. | None. |

## Python API

| `imbalance_eval` | `tsresample` |
|---|---|
| `compute_imbalance(y, rel_thres=0.9)` → `{N, n_normal, n_rare, IR, %Rare}` (rounded) | `tsresample.pipeline.imbalance_summary(y, rel_threshold=0.9)` → `{N, n_normal, n_rare, IR, pct_rare}` (unrounded) |
| `load_csv(path, date_col)` + `impute_dataframe(df)` | `tsresample.pipeline.load_series(path, target=..., date_col=..., diff=..., impute="knn")` |
| `time_delay_embedding(series, k)` → `lag_k … lag_1, target` | `tsresample.embed(series, k=k - 1)`: the same rows with the most recent lag **first**. The paper's `create.data(ts, 10)` is `embed(series, k=8)` (ADR-0005). |
| `evaluate_datasets` / `evaluate_csv_files` | the CLI, or a loop over `load_series` + `imbalance_summary` |
