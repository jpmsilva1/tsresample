# Auditoria learner a learner (22/09/2026)

R e Python ajustados **nos mesmos conjuntos de treino e teste**, para separar a diferença de
implementação do sorteio das janelas. Resultados em `../../COMPARISON.md`, seção 10.

1. Gravar as janelas Monte Carlo (e, se quiser, os conjuntos reamostrados) do port em CSV,
   `<nome>_train.csv` / `<nome>_test.csv`, com `mc_splits` e `apply_resampler` de `tsresamp`.
2. Ajustar em R:
   * `Rscript earth_all.R <dir> <nk> <degree> <thresh> <prefixo>` grava, para cada janela,
     `<nome>_earth_pred.csv` e `<nome>_earth_model.csv` (dirs, cuts, termos escolhidos,
     coeficientes). Os fixtures de `tests/data/earth/` foram gerados assim.
   * `Rscript fit_others.R <dir> <prefixo> rpart <minsplit> <cp>` (ou `rf <mtry> <ntree>`,
     ou `arima`) grava `<nome>_<learner>_pred.csv`.
3. Comparar: `../../.venv/bin/python compare_paired.py <dir> rpart rf arima` (F1 médio de R e
   Python nas mesmas janelas; os parâmetros por data set estão em `PARS` no script). Para
   conjuntos reamostrados, grave também `<nome>_orig.csv` (janela original), de onde vem a
   função de relevância, como no `Exps.R`.
