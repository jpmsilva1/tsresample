# Proveniência dos resultados versionados em `results_cluster/`

Todos os arquivos desta pasta são gerados por `tools/regenerate_results_cluster.sh` a partir dos
pickles por data set (`results/`, não versionados, 52 workflows x 50 repetições cada), na
seguinte composição:

| workflows | origem | código |
|---|---|---|
| `mc.lm*`, `mc.svm*`, `mc.arima` | run completo no cluster Apuana, job 15477, 18/09/2026 (`results/apuana/`, log em `joblogs/`) | código idêntico ao atual para lm, svm e arima |
| `mc.mars*` (24 data sets) | reexecução local, 22/09/2026 (`results/mars_v2/`) | `mars.py` = port do `earth` (COMPARISON.md, seção 10) |
| `mc.rpart*`, `mc.BDES` (24 data sets) | reexecução local, 22/09/2026 (`results/rpart_v2/`) | `cp` como poda de custo-complexidade + preditores em postos |
| `mc.rf*` (24 data sets) | reexecução local, 22 a 24/09/2026 (`results/rf_v2/`) | preditores em postos (`TreeRanks`) |

Mesmos parâmetros em todos os casos: Tabela 7 do artigo, `knnImputation`, semente 1234,
janelas 50 %/25 % (10 %/5 % em DS21-22, 20 %/10 % em DS23-24). As janelas Monte Carlo são
as mesmas em todas as execuções (mesma semente e mesmo gerador), portanto a substituição de
um workflow não altera a comparabilidade com os demais.

Nada mais depende do código antigo: os workflows que vêm do cluster (lm, svm, arima) usam
código que não mudou desde o run.
