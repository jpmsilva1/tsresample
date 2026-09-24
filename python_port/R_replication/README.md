# Replicação com o código R original

O código R publicado pelos autores foi executado nesta máquina (R 4.6.1, Homebrew) para
verificar (1) se ele reproduz os números do artigo e (2) se o port Python reproduz o
código. As funções do `Exps.R` e do `OptParmsSearch.R` são carregadas **sem alteração**;
os drivers só acrescentam o que o artigo descreve e o script deixa em aberto: os
parâmetros das Tabelas 7 e 8 e a imputação `knnImputation`.

| arquivo | função |
|---|---|
| `install_packages.R` | instala os pacotes (CRAN, `DMwR` do arquivo do CRAN, `uba` do GitHub) |
| `run_exps.R` | 52 workflows x 50 repetições de um dataset: `Rscript run_exps.R <ds> 50 all results` |
| `run_table6.R` | Tabela 6 (SVM com `un`/`ov` da Tabela 8): `Rscript run_table6.R <ds> 50 results` |
| `compare.py` | **não copiado para o `tsresample`** (precisa dos CSVs de `results/`): tabela de três colunas artigo x R original x Python |
| `results/` | **não copiado para o `tsresample`**: scores por repetição de cada workflow (`.csv`, `.err` = falha do R) e a comparação completa ficam no repositório `TSResampStrat_Python`; os resumos estão em `../COMPARISON.md` |

Ajustes de ambiente (sem tocar na lógica): `uba` compilado com `-std=gnu17` e
`typedef enum {false,true} bool` trocado por `#include <stdbool.h>` (C moderno); `UBL`
saiu do CRAN e arrasta dependências geoespaciais, então foi instalado um `UBL` mínimo com
o fonte oficial de `neighbours`, `phi` e `phi.control`, que é tudo o que o `Exps.R` usa.

Resumo (F1, 50 repetições, média das diferenças absolutas):

| | Python x R original | R original x artigo |
|---|---|---|
| DS1, 52 workflows | 0,009 (máx 0,03) | 0,015 (máx 0,05) |
| DS4, 52 workflows | 0,011 (máx 0,03) | 0,074 (máx 0,39) |
| DS10, 52 workflows | 0,013 (máx 0,27: baseline `earth` x MARS próprio; resolvido em 22/09, o `mars.py` é agora um port do `earth`, ver `../COMPARISON.md` seção 10) | 0,086 (máx 0,50) |
| Tabela 6, DS12 / DS4 / DS10 | 0,004 / 0,012 / 0,02 | 0,007 / 0,063 / 0,245; SMOTE **falha** em DS4 e DS12 |
