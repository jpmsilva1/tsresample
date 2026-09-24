# Relatório de replicação

**Artigo replicado:** Moniz, N., Branco, P., Torgo, L. (2017). *Resampling strategies for
imbalanced time series forecasting.* International Journal of Data Science and Analytics,
3, 161-181. Código e dados originais: github.com/nunompmoniz/TSResampStrat_JDSA2017 (R).

**Replicação:** port completo para Python (`TSResampStrat_Python`) e execução dos 24 data
sets, 52 workflows e 50 repetições Monte Carlo no cluster Apuana (CIn/UFPE), 18/09/2026.
Relatório de 21/09/2026, atualizado em 22/09/2026 (seção 7).

---

## 1. Resumo

O experimento do artigo foi reexecutado por inteiro com uma reimplementação em Python do
código R dos autores. A comparação foi feita em três frentes: contra o artigo (Fig. 7 e
Tabelas 3 a 6), contra o código R original executado localmente com os pacotes originais,
e contra uma replicação independente em R (relatório de J. P. Miranda, `main.pdf`).

| comparação | escopo | resultado |
|---|---|---|
| Fig. 7 do artigo (F1φ por data set e workflow) | 1232 pontos lidos da figura | desvio mediano 0,012; 87 % dos pontos a 0,05 ou menos; viés -0,003 (antes da seção 7: 0,013 e 84 %) |
| Tabelas 3, 4 e 5 (vitórias/derrotas, Wilcoxon) | 135 células | direção coincide em 87 %, 83 % e 100 % das células; diferença média de 1 a 2 vitórias em 24 |
| Hipóteses H1, H2 e H3 do artigo | conclusões | as três se sustentam com os nossos números |
| Código R original, rodado localmente | 156 pares (DS1, DS4, DS10, Tabela 6) + teste pareado por learner nos mesmos dados (seção 7) | Python = R original a 0,01 de F1 (ruído de semente); MARS idêntico ao `earth` |
| Replicação independente em R (`main.pdf`) | 1100 células (24 data sets) | 19 de 24 data sets a menos de 0,025; os outros 5 divergem por decisões daquele relatório |

Veredito: **a replicação é bem-sucedida.** O port reproduz o código publicado ao nível do
ruído aleatório, e reproduz o artigo em magnitude e em conclusões. As diferenças que
restam têm três origens, todas identificadas: (1) o próprio código R publicado não gera
alguns números do artigo (baselines de DS4 e DS10, Tabela 6 de DS4 e DS10); (2) uma
substituição de biblioteca sem equivalente exato em Python (MARS do pacote `earth`), que
afetava uma única célula relevante e foi eliminada em 22/09/2026 com um port do `earth`
(seção 7); (3) a semente do gerador aleatório, que o artigo não
informa e que muda contagens de vitórias em data sets onde os workflows quase empatam.

---

## 2. O artigo original

### 2.1 Problema e hipóteses

Previsão de séries temporais em que o interesse do usuário está nos valores raros e
extremos (picos de consumo, cheias, quedas de bolsa). O artigo propõe adaptar três
estratégias de reamostragem para regressão (undersampling aleatório, oversampling
aleatório e SmoteR) ao contexto temporal, com duas variantes cada: viés temporal (T:
preferência pelos casos mais recentes de cada bin) e viés temporal e de relevância (TPhi:
preferência pelos casos recentes e relevantes). São nove estratégias: U_B, U_T, U_TPhi,
O_B, O_T, O_TPhi, SM_B, SM_T, SM_TPhi.

Três hipóteses são testadas:

- **H1.** Reamostragem melhora significativamente a acurácia preditiva nos casos raros em
  relação ao modelo de regressão sem reamostragem.
- **H2.** As variantes com viés (T, TPhi) melhoram em relação à versão base (B) da mesma
  estratégia.
- **H3.** Modelos de regressão com reamostragem superam modelos específicos de séries
  temporais (ARIMA e BDES).

### 2.2 Desenho experimental (resumo)

24 séries reais de 6 fontes (Tabela 1 do artigo), embedding com k = 10 defasagens,
função de relevância automática por estatísticas de box plot com limiar 0,9, cinco
modelos de regressão (LM, SVM, MARS, RF, RPART) com hiperparâmetros otimizados por data
set (Anexo 1, Tabela 7 do artigo), mais ARIMA (`auto.arima`) e BDES (bagging de árvores
com múltiplos embeddings). Estimação Monte Carlo com 50 repetições, janelas de 50 %
treino e 25 % teste (10 %/5 % em DS21 e DS22; 20 %/10 % em DS23 e DS24). Métrica
principal: F1φ, a média harmônica de precisão e recall baseados em utilidade. Teste de
Wilcoxon pareado (p < 0,05) sobre os 24 data sets para cada comparação.

### 2.3 O que o artigo publica e o que não publica

Publicado: Fig. 7 (F1φ médio de cada um dos 52 workflows em cada data set, apenas em
gráfico), Tabelas 3, 4 e 5 (número de vitórias e derrotas, com as significativas entre
parênteses), Tabela 6 (F1φ do SVM em DS4, DS10 e DS12 com os percentuais de
reamostragem da Tabela 8), Tabela 7 (hiperparâmetros ótimos por data set), Tabela 1
(percentual de casos raros por série).

Não publicado: os valores numéricos da Fig. 7; a semente do gerador aleatório; as versões
dos pacotes R. No repositório de código, os hiperparâmetros da Tabela 7 não aparecem (o
script traz uma "parametrização de exemplo" com um único conjunto para todos os data
sets) e a imputação de valores faltantes, que o artigo diz ter usado (`knnImputation`),
está comentada.

---

## 3. Protocolo de replicação

Este é o procedimento seguido, passo a passo, para produzir os resultados deste
relatório. Cada passo indica o que o artigo determina, o que o código R original faz e o
que a replicação fez.

### 3.1 Dados

Os 24 data sets vêm do arquivo `data_NM_PB_LT_DSAA2016.Rdata` do repositório original,
lido em Python por um parser de `.Rdata` e convertido para pickle (`convert_rdata.py`).
As 24 séries, seus tamanhos e índices temporais foram conferidos contra a Tabela 1 (tamanhos do arquivo: DS1 a DS4 730, DS5 a DS8 17.378, DS9 1.095, DS10 a DS13 1.457, DS14 a DS20 536, DS21 e DS22 239.602, DS23 e DS24 51.208).

| ID | série | fonte | granularidade | valores | % raros (artigo) |
|---|---|---|---|---|---|
| DS1 a DS4 | temperatura, umidade, vento, aluguéis | Bike Sharing | diária | 730 (a Tabela 1 diz 731) | 9,9 / 9,3 / 7,8 / 13,3 |
| DS5 a DS8 | idem | Bike Sharing | horária | 17.378 (a Tabela 1 imprime 7.379) | 3,5 / 4,8 / 12,5 / 17,6 |
| DS9 | vazão do rio Vatnsdalsá | Icelandic River | diária | 1.095 | 21,1 |
| DS10 a DS13 | temp. mín., temp. máx., vento constante, rajada | Porto weather | diária | 1.457 | 4,8 / 13,3 / 11,0 / 11,1 |
| DS14 a DS20 | SP, DAX, FTSE, NIKKEI, BOVESPA, EU, emergentes | Istanbul Stock Exchange | diária | 536 | 16,3 / 11,4 / 9,7 / 11,6 / 10,1 / 8,2 / 6,8 |
| DS21, DS22 | demanda total, preço recomendado | Australian electricity load | meia hora | 239.602 | 1,8 / 10,2 |
| DS23, DS24 | Pedrouços, Rotunda AEP | Água do Porto | meia hora | 51.208 | 0,08 / 3,4 |

Valores faltantes existem em DS12, DS13, DS23 e DS24. O artigo (seção 4.1) diz que foram
imputados com `knnImputation` do pacote DMwR; no `Exps.R` a chamada está comentada. A
replicação porta `knnImputation` e a aplica ao dataframe embutido. Verificação: com a
imputação, o percentual de casos raros de DS12 dá 10,99 %, igual aos 11,0 % da Tabela 1;
sem ela dá 5,2 %.

### 3.2 Embedding

Cada série é transformada por `create.data(série, 10)`: cada linha tem os 10 valores
consecutivos V1 a V10, o alvo é V10 (o mais recente) e os preditores são V1 a V9. A série
entra em nível, sem diferenciação nem normalização, como no código original (o embedding
remove as 9 primeiras linhas). Port: `tsresamp/data.py::create_data`, testado contra
exemplos manuais.

### 3.3 Função de relevância φ

`uba::phi.control(y, method="extremes")` deriva pontos de controle das estatísticas de
box plot (`boxplot.stats`, cerca de 1,5 vezes o intervalo interquartil): relevância 1 nos
extremos adjacentes, 0 na mediana. `uba::phi` interpola com spline cúbica de Hermite
monotônica (pchip, declives de Fritsch-Carlson). Limiar de relevância tR = 0,9. Port:
`tsresamp/uba.py`, tradução linha a linha do C do pacote `uba`. Verificação: o percentual
de casos raros (φ > 0,9) coincide com a Tabela 1 nos 24 data sets dentro de 1 ponto
percentual (`COMPARISON.md`, seção 1).

### 3.4 Estratégias de reamostragem

As nove estratégias do artigo (Algoritmos 2 a 13) foram portadas das funções do `Exps.R`
(`randUnderRegress`, `randOverRegress`, `smoteRegress` e as variantes T e TPhi), incluindo
a formação de bins por relevância, a ordenação temporal, as preferências i/|OrdB| (T) e
i/|OrdB| × φ(y) (TPhi), os vizinhos do SmoteR (k = 5, distância euclidiana, `UBL::neighbours`)
e a escolha do vizinho mais recente (T) ou de maior relevância × posição temporal (TPhi).

Percentuais de reamostragem: o artigo (seção 5) usa um método de inferência que **balanceia**
o número de casos normais e raros no treino (`C.perc = "balance"` no código). A Tabela 6
usa percentuais explícitos da Tabela 8, com u aplicado aos bins normais e o aos bins raros
(Algoritmo 5); o código R publicado falha nesse caso em data sets com extremos dos dois
lados, e a replicação segue o algoritmo do artigo.

Duas peculiaridades do código R são reproduzidas tal como estão, porque os resultados
publicados foram gerados com elas: no `smote.exsRegressT/TPhi` a matriz numérica é
preenchida antes da reordenação temporal e os vizinhos calculados depois (índices de duas
ordenações diferentes); e o peso da interpolação do alvo sintético usa só a última coluna
não alvo. As duas estão documentadas em `README.md` e `tsresamp/resampling.py`.

### 3.5 Modelos e hiperparâmetros

| modelo | R original | replicação | equivalência |
|---|---|---|---|
| LM | `stats::lm` | `sklearn LinearRegression` | exata (mínimos quadrados) |
| SVM | `e1071::svm` (libsvm, RBF, eps 0,1, escala de x e y) | `sklearn SVR` (o mesmo libsvm, mesma escala) | exata dado o mesmo conjunto de treino |
| MARS | `earth` | port do `earth.c` 5.3.6 e da poda `leaps::BAKWRD` (`tsresamp/mars.py`; até 21/09 era um MARS próprio) | exata: mesmos termos, nós e coeficientes nos mesmos dados (seção 7) |
| RF | `randomForest` (`nodesize=5` = não dividir nós com 5 ou menos casos) | `RandomForestRegressor(min_samples_split=6, min_samples_leaf=1)`, preditores em postos | mesmo algoritmo; F1 igual ao R nos mesmos dados dentro do ruído da floresta (seção 7) |
| RPART | `rpart` (`cp` = poda por custo-complexidade, `minbucket = minsplit/3`, profundidade 30) | `DecisionTreeRegressor(ccp_alpha=cp·var(y), ...)`, preditores em postos | F1 a 0,002 do R nos mesmos dados (seção 7) |
| ARIMA | `forecast::auto.arima` + `Arima(model=)` | `pmdarima.auto_arima` (AICc, stepwise) + `apply` do statsmodels | aproximada: ordens diferem, F1 a 0,004 do R nos mesmos dados (seção 7) |
| BDES | bagging de `rpart` com embeddings kmax, kmax/2, kmax/4 e média/variância como features | port fiel com árvores do scikit-learn, inclusive o vazamento do alvo em `embedStats` | tradução + substituição |

Hiperparâmetros: os da Tabela 7 do artigo, transcritos para `exps_common.py` e conferidos
pela camada de texto do PDF (Anexo A deste relatório). Como determina a seção 6 do
artigo, os mesmos parâmetros são usados no modelo sem reamostragem e em todas as
variantes reamostradas do mesmo modelo.

**Nenhum limite de linhas é aplicado a nenhum modelo.** O SVM, em particular, treina na
janela de treino inteira de cada repetição: 23.959 linhas em DS21 e DS22, 10.239 em DS23
e DS24, 8.684 em DS5 a DS8. Os tempos de treino gravados nos resultados confirmam:

| data set | linhas de treino | teste | SVM sem reamostragem | SVM O_B | SVM SM_B |
|---|---|---|---|---|---|
| DS5 | 8.684 | 4.342 | 32 s | 248 s | 35 s |
| DS8 | 8.684 | 4.342 | 92 s | 367 s | 99 s |
| DS21 | 23.959 | 11.979 | 67 s | 262 s | 22 s |
| DS22 | 23.959 | 11.979 | 10 s | 91 s | 24 s |
| DS23 | 10.239 | 5.119 | 24 s | 75 s | 16 s |
| DS24 | 10.239 | 5.119 | 64 s | 419 s | 72 s |

(tempo médio de CPU por repetição; o oversampling aumenta o treino, daí os tempos maiores.)

### 3.6 Estimação Monte Carlo

Port de `performanceEstimation::MonteCarlo`: tamanho de treino = int(n × szTrain), de
teste = int(n × szTest); 50 pontos de partida sorteados sem reposição entre
train+1 e n−test+1, ordenados; treino = as `train` linhas anteriores ao ponto, teste = as
`test` linhas seguintes. Semente 1234 (o padrão do pacote R). Tamanhos: 50 %/25 %
(DS1 a DS20), 10 %/5 % (DS21, DS22), 20 %/10 % (DS23, DS24), como na seção 5 do
artigo. Todos os 52 workflows de um data set usam exatamente as mesmas 50 janelas, o que
torna as comparações pareadas.

O gerador aleatório do R não é reproduzível em Python: as janelas e os sorteios da
reamostragem usam o gerador do numpy com a mesma semente. Efeito medido (DS1, 52
workflows, semente 1234 contra 4321): diferença média de F1 de 0,008, máxima 0,05 (RF).

### 3.7 Métricas

Precisão, recall e F1 de utilidade (equações 1 a 3 do artigo) com a superfície de
utilidade de Ribeiro (`uba::util`, `benefcost_lin`, perda máxima admissível pelos pontos de
controle), limiar 0,9. Port: `tsresamp/uba.py::UtilityEvaluator`. O F1 reportado é a
média das 50 repetições (`GetResults`), como na Fig. 7. Uma repetição que falha (por
exemplo, janela de treino sem nenhum caso raro: o resampler para com "All the points have
relevance 0", o mesmo `stop()` do R) é registrada como NaN e a média usa as repetições
válidas; isso ocorreu apenas em DS23 (6 repetições, 10 na U_TPhi).

### 3.8 Testes estatísticos

Port de `pairedComparisons` e `WLdef` do `PairedComparisons.R`: para cada data set e cada
par (workflow, baseline), teste de Wilcoxon pareado sobre as 50 repetições, com as
convenções do `wilcox.test` do R; vitória se a mediana das diferenças favorece o workflow,
significativa se p < 0,05; contagem sobre os 24 data sets. As Tabelas 3, 4 e 5 são
reconstruídas com as mesmas seleções de linhas do script R.

### 3.9 Implementação e ambiente

Estrutura do port espelha a do repositório R: `Py_Code/Exps.py` (experimento),
`GetResults.py`, `PairedComparisons.py`, `OptParmsSearch.py`, `ReplicateTable6.py`,
pacote `tsresamp` (data, uba, resampling, mars, models, estimation, analysis). 122
testes unitários cobrem cada componente (spline, box plot, bins, cada resampler, janelas
Monte Carlo, Wilcoxon, learners).

Ambiente: Python 3.12.11, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1,
pmdarima 2.1.1, statsmodels 0.15.0 (`requirements-lock.txt`). As mesmas versões foram
usadas localmente (macOS arm64) e no cluster (Linux x86_64); DS1 rodado nos dois dá F1
idêntico a menos de 10⁻⁴ nos modelos determinísticos.

Referência em R: R 4.6.1 com os pacotes originais (`uba` compilado do GitHub, `DMwR` do
arquivo do CRAN, `performanceEstimation`, `e1071`, `randomForest`, `rpart`, `earth`,
`forecast`, `UBL` mínimo). As funções do `Exps.R` foram carregadas sem alteração; o
driver só acrescenta a Tabela 7 e a imputação (`R_replication/`).

### 3.10 Execução no cluster Apuana

Job SLURM 15477, partição `short-simple`, 1 nó (cluster-node6: 96 CPUs, 512 GB), 48 CPUs
e 500 GB reservados, das 00:17 às 09:16 de 18/09/2026 (9 h). Dentro do job, três
"faixas" do `run_datasets.sh` rodam em paralelo, cada uma com 16 workers do joblib sobre
as 50 repetições de um workflow (1 thread de BLAS por worker), tomando o próximo data set
livre de uma lista compartilhada (reserva atômica por `mkdir`). Checkpoint por workflow e
por data set, o que permite retomar após cancelamento ou requeue. Scripts em
`cluster/apuana/` (`run_all.slurm`, `run_datasets.sh`, `setup_env.sh`, `watch_run.sh`).

Tempo por data set (min): DS22 308, DS24 224, DS7 220, DS6 218, DS21 215, DS8 188, DS5 80,
DS23 57, DS1 11, DS2 9, DS3 a DS20 entre 2 e 8 cada. Os data sets grandes (5 a 8, 21 a
24) dominam; neles o MARS e o SVM com oversampling são os workflows mais caros.

### 3.11 Verificações antes do run completo

1. Testes unitários (122) de cada componente.
2. Percentual de casos raros contra a Tabela 1 (24 de 24).
3. Tabela 6 do artigo em DS12: 10 de 10 linhas a 0,01 (relevância, utilidade, os nove
   resamplers, Monte Carlo e SVM juntos).
4. DS1, DS4 e DS10 completos contra a Fig. 7 e contra o código R original rodado
   localmente: Python = R a 0,01 de F1 em 156 pares.
5. Job de 20 min na partição `debug` (DS1, 2 repetições, lm e rpart) antes do job real.
6. DS1 do cluster contra DS1 local: idêntico a menos de ponto flutuante.

### 3.12 Reprodução passo a passo

```bash
# ambiente local
uv venv .venv --python 3.12 && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python Py_Code/convert_rdata.py          # Data/*.Rdata -> pickle
.venv/bin/python -m pytest tests -q                 # 127 testes

# um data set (52 workflows, 50 repetições, Tabela 7, knnImputation)
.venv/bin/python Py_Code/Exps.py --dataset 1 --nreps 50 --workflows all \
    --n-jobs 8 --out results/exp_ds1.pkl

# os 24 no Apuana (ver cluster/apuana/README.md para VPN, ssh e ambiente)
sbatch cluster/apuana/run_all.slurm                 # 48 CPUs, 3 faixas, checkpoints
rsync -az apuana:~/TSResampStrat_Python/results/ results/apuana/

# tabelas e comparações
.venv/bin/python Py_Code/GetResults.py results/apuana/exp_ds*.pkl --csv results_cluster/globalres_all.csv
.venv/bin/python Py_Code/PairedComparisons.py results/apuana/exp_ds*.pkl
.venv/bin/python tools/compare_full_run.py results/apuana   # Tabelas 3, 4, 5 e relatório R
.venv/bin/python tools/compare_fig7_all.py                  # Fig. 7, 24 data sets
```

---

## 4. Resultados: artigo x replicação

> Os números desta seção são os do run do cluster de 18/09/2026. A seção 7 descreve a
> auditoria de 22/09 (MARS = `earth`, árvores corrigidas) e traz as mesmas comparações
> recalculadas; as conclusões não mudam.

Formato das células: **nosso | artigo**.

### 4.1 Fig. 7: F1φ por data set e workflow

A Fig. 7 é o único lugar em que o artigo mostra o F1φ de cada workflow em cada data set.
Os valores foram lidos da figura renderizada do PDF (`tools/compare_fig7_all.py`: eixo x
calibrado nas linhas de grade de DS1 e DS24, séries tracejadas interpoladas nos vértices),
com precisão da ordem de ±0,03. 1232 dos 1248 pontos foram recuperados.

Concordância geral: |desvio| médio 0,027, mediano 0,013; 84 % dos pontos a 0,05 ou menos,
93 % a 0,10 ou menos; viés (nosso menos artigo) -0,003.

| modelo | pontos | \|desvio\| médio | viés | pontos a ≤ 0,05 |
|---|---|---|---|---|
| SVM | 239 | 0,014 | -0,004 | 96 % |
| RPART | 240 | 0,018 | -0,007 | 92 % |
| RF | 240 | 0,019 | -0,006 | 93 % |
| MARS | 238 | 0,024 | +0,002 | 90 % |
| LM | 230 | 0,050 | -0,002 | 65 % |
| BDES | 24 | 0,081 | -0,002 | 54 % |
| ARIMA | 21 | 0,109 | +0,007 | 33 % |

F1φ médio sobre os data sets, nosso | artigo. Todos os 50 pares modelo × estratégia ficam a
0,02 ou menos um do outro:

| | sem | U_B | U_T | U_TPhi | O_B | O_T | O_TPhi | SM_B | SM_T | SM_TPhi |
|---|---|---|---|---|---|---|---|---|---|---|
| LM | 0,26 \| 0,26 | 0,49 \| 0,49 | 0,49 \| 0,50 | 0,49 \| 0,50 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,50 \| 0,50 | 0,49 \| 0,49 | 0,49 \| 0,49 |
| SVM | 0,45 \| 0,46 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,49 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,48 | 0,47 \| 0,47 | 0,46 \| 0,46 |
| MARS | 0,36 \| 0,36 | 0,44 \| 0,42 | 0,45 \| 0,43 | 0,45 \| 0,44 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,48 |
| RF | 0,27 \| 0,29 | 0,41 \| 0,42 | 0,41 \| 0,42 | 0,41 \| 0,42 | 0,37 \| 0,38 | 0,36 \| 0,36 | 0,35 \| 0,36 | 0,50 \| 0,50 | 0,48 \| 0,48 | 0,48 \| 0,49 |
| RPART | 0,44 \| 0,46 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,49 | 0,47 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,49 | 0,46 \| 0,47 | 0,47 \| 0,48 |

ARIMA 0,275 | 0,268; BDES 0,116 | 0,118.

Por data set, |desvio| médio sobre os workflows lidos:

| DS | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | 0,012 | 0,018 | 0,019 | 0,034 | 0,028 | 0,020 | 0,026 | 0,035 | 0,042 | 0,060 | 0,030 | 0,022 |
| **DS** | **13** | **14** | **15** | **16** | **17** | **18** | **19** | **20** | **21** | **22** | **23** | **24** |
| | 0,019 | 0,035 | 0,021 | 0,021 | 0,019 | 0,020 | 0,019 | 0,023 | 0,028 | 0,036 | 0,047 | 0,016 |

Nenhum data set foge do padrão. Os pontos que mais divergem são baselines de DS4, DS5,
DS7 e DS10 (seção 5.2).

### 4.2 Tabela 3: estratégia base vs modelo sem reamostragem (H1)

Vitórias (significativas) / derrotas (significativas) sobre os 24 data sets.

| | LM | SVM | MARS | RF | RPART |
|---|---|---|---|---|---|
| U_B | 19 (19)/5 (4) \| 19 (18)/5 (4) | 11 (10)/13 (10) \| 8 (6)/16 (8) | 17 (16)/7 (5) \| 15 (12)/9 (7) | 19 (17)/5 (4) \| 18 (17)/6 (2) | 11 (9)/13 (5) \| 12 (8)/12 (3) |
| O_B | 19 (19)/5 (1) \| 18 (17)/6 (3) | 14 (11)/10 (6) \| 7 (6)/17 (10) | 21 (19)/3 (3) \| 17 (17)/7 (4) | 21 (14)/3 (3) \| 20 (15)/4 (1) | 12 (11)/12 (7) \| 11 (9)/13 (8) |
| SM_B | 20 (19)/4 (3) \| 19 (18)/5 (3) | 11 (9)/13 (8) \| 7 (6)/17 (10) | 21 (18)/3 (2) \| 18 (17)/6 (4) | 21 (20)/3 (1) \| 20 (20)/4 (1) | 12 (10)/12 (6) \| 10 (10)/14 (7) |

LM, RF e RPART coincidem a 1 ou 2 vitórias. MARS fica 2 a 4 vitórias acima do artigo. SVM
é a única célula em que a direção muda: o artigo tem 7 a 8 vitórias em 24 (reamostragem
perde), nós temos 11 a 14 (empate ou leve ganho).

### 4.3 Tabela 4: viés temporal e de relevância vs versão base (H2)

| | LM | SVM | MARS | RF | RPART |
|---|---|---|---|---|---|
| U_T | 18 (2)/6 (0) \| 14 (2)/10 (0) | 12 (2)/12 (1) \| 10 (0)/14 (0) | 15 (3)/9 (1) \| 11 (0)/13 (2) | 13 (1)/11 (2) \| 12 (1)/12 (2) | 11 (0)/13 (1) \| 14 (4)/10 (0) |
| U_TPhi | 14 (6)/10 (2) \| 15 (10)/9 (3) | 16 (8)/8 (3) \| 11 (5)/13 (4) | 11 (7)/13 (1) \| 17 (6)/7 (1) | 13 (5)/11 (3) \| 16 (6)/8 (3) | 14 (6)/10 (2) \| 16 (7)/8 (5) |
| O_T | 12 (4)/12 (7) \| 14 (8)/10 (9) | 8 (4)/16 (7) \| 12 (5)/12 (6) | 12 (4)/12 (6) \| 11 (3)/13 (4) | 7 (1)/17 (5) \| 8 (4)/16 (4) | 11 (3)/13 (1) \| 12 (3)/12 (2) |
| O_TPhi | 10 (5)/14 (7) \| 14 (9)/10 (7) | 7 (4)/17 (10) \| 12 (4)/12 (7) | 10 (4)/14 (3) \| 11 (3)/13 (2) | 6 (0)/18 (4) \| 8 (2)/16 (5) | 10 (2)/14 (0) \| 14 (3)/10 (2) |
| SM_T | 6 (3)/18 (14) \| 6 (5)/18 (13) | 10 (3)/14 (13) \| 10 (5)/14 (10) | 13 (4)/11 (10) \| 9 (6)/15 (10) | 6 (4)/18 (13) \| 9 (3)/15 (10) | 5 (1)/19 (12) \| 8 (1)/15 (11) |
| SM_TPhi | 6 (2)/18 (10) \| 6 (4)/18 (11) | 11 (10)/13 (10) \| 9 (6)/15 (12) | 15 (4)/9 (7) \| 12 (4)/12 (6) | 6 (6)/18 (9) \| 12 (5)/12 (10) | 7 (4)/17 (10) \| 6 (4)/17 (9) |

Padrões que coincidem: U_T sem vantagem clara e quase sem significância (como o artigo
observa); U_TPhi com vitórias significativas; SmoteR com viés pior que SM_B em quase todas
as colunas. Padrão que difere: no oversampling com viés (O_T, O_TPhi) o artigo é
ligeiramente positivo em LM e RPART (14 vitórias) e nós ligeiramente negativo (10 a 12).

### 4.4 Tabela 5: modelos com reamostragem vs ARIMA e vs BDES (H3)

90 células (tabela completa no Anexo B). A direção coincide em 100 % delas; |Δvitórias|
médio 1,1; 83 % das células a 2 vitórias ou menos do artigo. Faixas de vitórias em 24:

| modelo | vs ARIMA, nosso | vs ARIMA, artigo | vs BDES, nosso | vs BDES, artigo |
|---|---|---|---|---|
| LM | 20 a 21 | 18 a 21 | 22 | 22 |
| SVM | 18 a 23 | 19 a 21 | 22 | 20 a 22 |
| MARS | 18 a 21 | 18 a 23 | 21 a 22 | 21 a 22 |
| RF | 13 a 22 | 19 a 22 | 15 a 22 | 15 a 22 |
| RPART | 17 a 19 | 17 a 22 | 21 a 22 | 22 a 23 |

A maior diferença é RF com O_T e O_TPhi contra ARIMA (13 e 14 vitórias contra 19).

### 4.5 Tabela 6: SVM com os percentuais da Tabela 8 (DS4, DS10, DS12)

Executada localmente com 50 repetições (Python, R original e artigo):

| | DS12 artigo / R / Python | DS4 artigo / R / Python | DS10 artigo / R / Python |
|---|---|---|---|
| SVM | 0,554 / 0,551 / 0,557 | 0,584 / 0,560 / 0,549 | 0,638 / 0,609 / 0,589 |
| U_B, U_T, U_TPhi | 0,61-0,63 / 0,61-0,62 / 0,61-0,62 | 0,65-0,67 / 0,60 / 0,58-0,59 | 0,64-0,65 / **0,12**, 0,65, 0,67 / **0,14**, 0,63, 0,67 |
| O_B, O_T, O_TPhi | 0,61 / 0,60-0,61 / 0,61 | 0,65 / 0,61, 0,61, 0,50 / 0,60, 0,60, 0,50 | 0,65 / **0,00, 0,09, 0,09** / **0,00, 0,10, 0,09** |
| SM_B, SM_T, SM_TPhi | 0,60-0,62 / falha / 0,60-0,61 | 0,65-0,66 / falha / 0,57-0,60 | 0,68-0,72 / 0,68, 0,65, 0,70 / 0,68, 0,60, 0,71 |

DS12 coincide nas 10 linhas. Em DS4 e DS10, Python e R original coincidem entre si e
divergem do artigo da mesma forma; o SMOTE com percentuais explícitos falha no R
original nos data sets com extremos dos dois lados.

### 4.6 As três hipóteses com os nossos números

- **H1, confirmada.** U_B, O_B e SM_B vencem o modelo sem reamostragem em 17 a 21 dos 24
  data sets em LM, MARS e RF (quase todas significativas), com margem menor em RPART
  (11 a 12) e SVM (11 a 14). O artigo tem o mesmo quadro, com o SVM um pouco mais
  desfavorável à reamostragem.
- **H2, confirmada para undersampling com TPhi; não confirmada para SmoteR.** Como no
  artigo: U_TPhi vence U_B com vitórias significativas na maioria dos modelos; U_T sozinho
  não ajuda; SM_T e SM_TPhi perdem para SM_B. O oversampling com viés não mostra vantagem
  nos nossos números (o artigo o descreve como "sem vantagem clara").
- **H3, confirmada.** Todo modelo com reamostragem vence ARIMA em 13 a 23 e BDES em 15 a
  22 dos 24 data sets, quase sempre com significância.

---

## 5. Diferenças e explicações

### 5.1 De onde vêm as diferenças

1. **Substituição de bibliotecas.** LM e SVM têm equivalentes exatos (mínimos quadrados;
   o mesmo libsvm com a mesma escala). RF, RPART, MARS e ARIMA são o mesmo algoritmo com
   implementações diferentes; cada detalhe (regra de tamanho de nó do RF, critério de
   parada do MARS, aproximações do `auto.arima`) vale alguns centésimos de F1 e, em
   baselines que quase nunca preveem extremos, pode virar zero ou meio ponto. (Atualização
   de 22/09/2026, seção 7: o MARS passou a ser um port do `earth`, e RF, RPART e ARIMA foram
   verificados contra o R nos mesmos conjuntos de treino: sem diferença sistemática.)
2. **Gerador aleatório.** As janelas Monte Carlo e os sorteios da reamostragem não são
   bit a bit os do R. Efeito medido: 0,01 de F1 em média, 0,05 no RF. Em contagens de
   vitórias, um data set em que dois workflows quase empatam pode mudar de lado.
3. **O artigo não é só o código.** Hiperparâmetros só no PDF, imputação comentada,
   SMOTE com percentuais explícitos que não roda em 3 bumps, índices trocados no SMOTE
   temporal. Em cada ponto a replicação seguiu o que o artigo descreve.
4. **O código publicado não reproduz alguns números do artigo.** Verificado rodando o R
   original: baselines de DS4 (LM, ARIMA, BDES), RF e ARIMA de DS10, Tabela 6 de DS4 e
   DS10. O que gerou aquelas linhas não é exatamente o que está no repositório.

### 5.2 Casos específicos

| onde | artigo | nosso | R original | explicação |
|---|---|---|---|---|
| DS4: LM, ARIMA, BDES sem reamostragem | 0,24; 0,28; 0,19 | 0,58; 0,57; 0,63 | 0,59; 0,58; 0,63 | o código publicado dá o mesmo que o Python; discrepância é do artigo com o seu código |
| DS10: LM com reamostragem | 0,43-0,47 | 0,65-0,67 | 0,65-0,67 | idem |
| DS10: RF e ARIMA sem reamostragem | 0,50; 0,47 | 0,00; 0,00 | 0,00; 0,00 | idem |
| DS10: MARS sem reamostragem | 0,16 | 0,49 → **0,16** (seção 7) | 0,22 | era a substituição do `earth`, a única célula em que o Python diferia das duas execuções em R; o `mars.py` é agora um port do `earth` e a célula coincide |
| Tabela 3, SVM | 7-8 vitórias | 11-14 | (relatório independente: 58 % de vitórias) | as três execuções do código dão mais vitórias ao SVM reamostrado do que o artigo |
| Tabela 4, O_T/O_TPhi em LM e RPART | 14 vitórias | 10-12 | - | empates apertados que mudam com a semente |
| Tabela 5, RF O_T/O_TPhi vs ARIMA | 19 | 13-14 | - | idem; RF com oversampling é o workflow mais fraco (0,36 de F1) |
| DS23: 6 a 10 repetições NaN | 50 repetições | 44 válidas | mesmo `stop()` | janelas de treino sem casos raros (0,08 % de raros); o código R para do mesmo modo |

### 5.3 Triangulação com outras execuções

| par | escopo | diferença média de F1 |
|---|---|---|
| Python × R original (local) | DS1, DS4, DS10, Tabela 6: 156 pares | 0,009 a 0,013 (um outlier, MARS baseline em DS10, eliminado na seção 7) |
| R original (local) × replicação independente em R (`main.pdf`) | DS1, DS4, DS10: 150 pares | 0,001 (LM) a 0,009 (RF) |
| Python × replicação independente em R | 24 data sets, 1100 células | mediana 0,008; 19 de 24 data sets a menos de 0,025 |

Os cinco data sets em que o Python e a replicação independente divergem (DS12, DS20,
DS21, DS22, DS24) se explicam por decisões daquele relatório que não estão no artigo:
`complete.cases` em vez de `knnImputation` (DS12), teto de cerca de 8.000 linhas de treino
após a reamostragem em DS21 a DS24 (aqui não há teto: 23.959 e 10.239 linhas), e
baselines zeradas em DS20 que o próprio relatório registra como defeito em aberto.

### 5.4 Ruído ou diferença sistemática?

Regra usada: diferenças de F1 abaixo de 0,02 (0,05 para RF) e diferenças de contagem de
1 a 3 vitórias em 24 são indistinguíveis do efeito da semente (medido na seção 3.6).
Acima disso, a diferença foi investigada e tem explicação na tabela 5.2. Não resta
nenhuma diferença sem explicação atribuível ao port.

---

## 6. Limitações e recomendações

- A leitura da Fig. 7 tem precisão de ±0,03 e não distingue pontos sobrepostos; para SVM,
  RPART e RF a concordância está no limite do que a figura permite medir.
- A semente e as versões de pacotes do artigo são desconhecidas: contagens exatas de
  vitórias não são reproduzíveis por ninguém, incluindo os autores com o código atual.
- Para o TCC: as comparações **entre estratégias de reamostragem dentro de um mesmo
  modelo** (Tabelas 3 e 4) são reproduzidas com fidelidade e podem ser usadas
  diretamente. As comparações **entre modelos** e contra ARIMA/BDES carregam a diferença
  de implementação de RF, MARS e ARIMA e devem ser apresentadas como tal.
- (Superado em 22/09/2026: o MARS é agora um port do `earth`, verificado nos mesmos dados;
  ver seção 7.)
- Os arquivos de resultados (`results_cluster/`) permitem recalcular qualquer tabela com
  outra métrica (precisão, recall) ou outro α sem reexecutar nada.

---

## 7. Atualização de 22/09/2026: auditoria learner a learner e correção do MARS

Depois do relatório de 21/09 restava uma pergunta: o que ainda separava o port do código R
além do sorteio das janelas? A resposta exigia um teste que não tinha sido feito: ajustar o R
e o Python **nos mesmos conjuntos de treino e teste**, learner a learner, para isolar a
implementação do acaso. As janelas Monte Carlo e os conjuntos reamostrados do port foram
gravados em CSV e ajustados também em R (`earth`, `rpart`, `randomForest`, `auto.arima`).

### 7.1 O MARS não era o `earth`, e passou a ser

O `tsresamp/mars.py` era um MARS "de livro" (nós nos quantis de cada variável, termo linear
sempre candidato, poda gulosa por GCV). O `earth` faz quatro coisas diferentes que mudam o
modelo: (1) escolhe os nós entre os **casos** ordenados, nunca nos `endspan` casos de cada
extremo e só a cada `minspan` casos; (2) descarta o candidato linear quando a sua soma de
quadrados ortogonalizada é ≤ 0,01 em valor absoluto (`MIN_BX_SOS`), o que em séries de
valores pequenos (DS15, DS16, DS19, valores ~0,01) e em conjuntos sob-amostrados reduz o
modelo a poucos hinges; (3) rejeita pares de hinges quase colineares e limita a redução de
RSS por passo; (4) poda com a rotina `BAKWRD` do pacote `leaps`, que guarda o melhor
subconjunto de cada tamanho entre todos os prefixos visitados, não só o caminho guloso. O
nosso MARS ajustava mais termos e extrapolava mais nos extremos, exatamente onde o F1φ é
medido.

O `mars.py` foi reescrito como port do `earth.c` (5.3.6) e da poda do `earth.fit.R`/`leaps`.
Verificação: em 74 conjuntos de treino/teste de 10 data sets (com e sem reamostragem) o port
escolhe os mesmos termos, nós e coeficientes que o `earth`, com predições iguais a 1e-14; nas
50 janelas Monte Carlo de DS10 e de DS1 o F1 de `mc.mars` é igual iteração a iteração
(0,1624 = 0,1624 e 0,2308 = 0,2308). Quatro desses casos ficaram como fixtures de teste
(`tests/data/earth/`).

| F1 médio, 50 repetições | MARS antigo | MARS = `earth` | R original | relatório R |
|---|---|---|---|---|
| DS10 `mc.mars` | 0,490 | **0,162** | 0,219 | 0,195 |
| DS15 UNDER B / T / TPhi | 0,279 / 0,292 / 0,267 | **0,151 / 0,171 / 0,170** | - | 0,143 / 0,121 / 0,138 |
| DS16 UNDER B / T / TPhi | 0,301 / 0,272 / 0,258 | **0,089 / 0,132 / 0,135** | - | 0,085 / 0,094 / 0,092 |
| DS19 UNDER B / T / TPhi | 0,275 / 0,271 / 0,274 | **0,126 / 0,154 / 0,136** | - | 0,098 / 0,169 / 0,158 |

Nos 19 data sets comparáveis com o relatório independente, o |desvio| médio das 190 células
MARS cai de 0,022 para 0,011 e as células a mais de 0,05 caem de 20 para 5; o viés das
famílias UNDER (+0,02 a +0,03) desaparece. O que resta (0,162 contra 0,219 na baseline do
DS10) é o sorteio das janelas: o erro padrão da média de 50 janelas dessa baseline é 0,036.

### 7.2 Os outros learners: sem diferença sistemática, dois ajustes de mapeamento

| learner | conjuntos | F1 médio R | F1 médio Python | diferença (erro padrão) |
|---|---|---|---|---|
| `rpart` | DS1 / DS4 / DS10, 50 janelas | 0,433 / 0,533 / 0,548 | 0,432 / 0,533 / 0,546 | -0,001 (0,007) / 0 / -0,002 (0,003) |
| `randomForest` | DS1; DS2 e DS3 com OVER B, 50 conjuntos | 0,006 / 0,200 / 0,243 | 0,006 / 0,201 / 0,249 | 0 / +0,001 (0,009) / +0,007 (0,017) |
| `auto.arima` | DS4 / DS10 / DS1, 50 janelas | 0,572 / 0,000 / 0,009 | 0,568 / 0,000 / 0,000 | -0,004 (0,003) / 0 / uma janela |

Os dois ajustes, ambos lidos no fonte do R e confirmados pelo teste pareado:

1. **`cp` do `rpart` é poda por complexidade.** Em `rpart/src/partition.c` a subárvore de
   um nó só fica se a sua complexidade (redução de SS por divisão, calculada de baixo para
   cima colapsando primeiro os filhos mais fracos) excede `cp × SS_raiz`; `bsplit.c` não
   aplica limiar por divisão. Isso é `ccp_alpha = cp·var(y)` no scikit-learn, e não a
   pré-poda que a documentação do `rpart` sugere e que a versão anterior usava. O número de
   folhas passa a coincidir com o do R em 28/50 janelas do DS1 (antes 12/50) e a diferença
   de F1 vai de -0,010 para +0,002. O BDES usa as mesmas árvores (efeito: +0,001).
2. **Empates a 1e-7 no scikit-learn.** As árvores do scikit-learn nunca dividem entre dois
   valores a menos de 1e-7 um do outro; `rpart` e `randomForest` dividem entre quaisquer
   doubles distintos, e o DS1 tem 30 pares por coluna a 1 ulp de distância. Os preditores
   passam agora por uma transformação para postos antes de cada árvore (`TreeRanks`), que
   preserva a ordem e o lado do ponto médio. Com isso o DS1 chega a 38/50 janelas com o
   mesmo número de folhas e o `randomForest` do DS1 dá o mesmo F1 em todas as janelas.

O que fica como diferença de implementação sem correção possível: empates exatos do critério
de divisão entre variáveis no `rpart` em séries de inteiros (DS10), desempatados pela ordem
das colunas no R e por ordem aleatória no scikit-learn (0,002 de F1). O `auto.arima` do
`pmdarima` escolhe ordens diferentes das do `forecast` em 40 a 70 % das janelas, mas as
predições um passo à frente têm correlação 0,999 e o F1 não muda.

### 7.3 Tabelas recalculadas

Os workflows MARS, RPART, BDES e RF (24 data sets) foram reexecutados
localmente com o código atual, nas mesmas janelas do run do cluster, e as tabelas de
`results_cluster/` foram regeradas (`tools/regenerate_results_cluster.sh`; proveniência em
`results_cluster/PROVENANCE.md`).

| comparação | antes (run do cluster) | depois (MARS = `earth`, árvores corrigidas) |
|---|---|---|
| Fig. 7 do artigo, 1232 pontos: \|desvio\| médio / mediana / a ≤ 0,05 | 0,027 / 0,013 / 84 % | 0,025 / 0,012 / 87 % |
| Fig. 7, só MARS (238 pontos): \|desvio\| médio / a ≤ 0,05 | 0,024 / 90 % | **0,015 / 97 %** |
| Fig. 7, só RPART (240 pontos): \|desvio\| médio / a ≤ 0,05 | 0,018 / 92 % | 0,014 / 97 % |
| Tabela 3: \|Δvitórias\| média / mesma direção | 2,13 / 93 % | 1,60 / 87 % |
| Tabela 4: \|Δvitórias\| média / mesma direção | 2,60 / 77 % | 2,20 / 83 % |
| Tabela 5: \|Δvitórias\| média / mesma direção | 1,09 / 100 % | 1,17 / 100 % |
| relatório R independente, 19 data sets comparáveis: \|desvio\| médio / mediana / a ≤ 0,05 | 0,0123 / 0,0062 / 96 % | 0,0101 / 0,0054 / 97 % |
| idem, só MARS | 0,0215 | **0,0108** |

A melhora na Fig. 7 é uma confirmação independente: nada foi ajustado olhando o artigo, só o
código R. As contagens de vitórias das Tabelas 3 e 4 mudam dentro do ruído de semente já
descrito na seção 5.4: a diferença média cai, e uma célula da Tabela 3 troca de direção
(SVM, em que o próprio R original diverge do artigo), enquanto duas da Tabela 4 passam a
coincidir. O RF de DS21 a DS24 também foi reexecutado (terminou em 24/09/2026); todas as tabelas
acima já o incluem.

F1 médio sobre os 24 data sets depois da atualização (`results_cluster/meanF1_model_strategy.csv`):

| | base | U_B | U_T | U_TPhi | O_B | O_T | O_TPhi | SM_B | SM_T | SM_TPhi |
|---|---|---|---|---|---|---|---|---|---|---|
| mars | 0,349 | 0,423 | 0,428 | 0,428 | 0,492 | 0,494 | 0,491 | 0,494 | 0,484 | 0,480 |
| rf | 0,273 | 0,406 | 0,406 | 0,414 | 0,373 | 0,364 | 0,363 | 0,495 | 0,480 | 0,482 |
| rpart | 0,452 | 0,481 | 0,478 | 0,480 | 0,480 | 0,479 | 0,479 | 0,483 | 0,468 | 0,472 |

(Artigo, média lida da Fig. 7 — MARS: 0,36 / 0,42 / 0,43 / 0,44 / 0,49 / 0,49 / 0,49 / 0,49 / 0,49 / 0,48.)


---

## Anexo A: hiperparâmetros da Tabela 7 (transcritos e usados)

| DS | SVM cost | SVM gamma | MARS nk | degree | thresh | RF mtry | ntree | RPART minsplit | cp |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 0,01 | 17 | 1 | 0,001 | 5 | 1500 | 10 | 0,01 |
| 2 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 750 | 10 | 0,001 |
| 3 | 300 | 0,01 | 17 | 1 | 0,001 | 7 | 500 | 10 | 0,001 |
| 4 | 150 | 0,01 | 10 | 1 | 0,001 | 7 | 750 | 10 | 0,1 |
| 5 | 300 | 0,001 | 10 | 2 | 0,001 | 7 | 750 | 20 | 0,001 |
| 6 | 300 | 0,01 | 17 | 2 | 0,001 | 5 | 500 | 10 | 0,001 |
| 7 | 300 | 0,01 | 10 | 1 | 0,001 | 7 | 750 | 30 | 0,001 |
| 8 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 750 | 30 | 0,001 |
| 9 | 10 | 0,01 | 10 | 2 | 0,001 | 5 | 750 | 30 | 0,001 |
| 10 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 500 | 10 | 0,001 |
| 11 | 10 | 0,01 | 17 | 1 | 0,001 | 7 | 500 | 20 | 0,001 |
| 12 | 300 | 0,01 | 17 | 1 | 0,001 | 7 | 750 | 10 | 0,001 |
| 13 | 150 | 0,01 | 17 | 2 | 0,001 | 7 | 750 | 10 | 0,001 |
| 14 | 150 | 0,01 | 17 | 2 | 0,001 | 7 | 1500 | 10 | 0,001 |
| 15 | 300 | 0,01 | 17 | 2 | 0,001 | 5 | 1500 | 10 | 0,001 |
| 16 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 750 | 10 | 0,001 |
| 17 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 500 | 10 | 0,001 |
| 18 | 300 | 0,01 | 17 | 2 | 0,001 | 5 | 500 | 10 | 0,001 |
| 19 | 150 | 0,01 | 17 | 1 | 0,01 | 5 | 500 | 10 | 0,001 |
| 20 | 300 | 0,01 | 17 | 2 | 0,001 | 7 | 500 | 10 | 0,001 |
| 21 | 150 | 0,001 | 17 | 2 | 0,001 | 7 | 500 | 10 | 0,001 |
| 22 | 150 | 0,001 | 10 | 2 | 0,001 | 7 | 500 | 10 | 0,001 |
| 23 | 10 | 0,001 | 10 | 1 | 0,001 | 5 | 500 | 10 | 0,001 |
| 24 | 150 | 0,01 | 17 | 1 | 0,001 | 7 | 750 | 10 | 0,001 |

## Anexo B: Tabela 5 completa (nosso | artigo)

| modelo, estratégia | vs ARIMA | vs BDES |
|---|---|---|
| LM U_B | 20 (19)/4 (4) \| 18 (18)/6 (3) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| LM U_T | 21 (20)/3 (3) \| 18 (18)/6 (3) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| LM U_TPhi | 20 (19)/4 (4) \| 18 (18)/6 (5) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| LM O_B | 21 (20)/3 (3) \| 21 (18)/3 (2) | 22 (21)/2 (2) \| 22 (22)/2 (2) |
| LM O_T | 21 (20)/3 (2) \| 18 (18)/6 (3) | 22 (21)/2 (2) \| 22 (22)/2 (2) |
| LM O_TPhi | 21 (20)/3 (2) \| 18 (18)/6 (3) | 22 (21)/2 (2) \| 22 (22)/2 (2) |
| LM SM_B | 21 (21)/3 (3) \| 20 (18)/4 (3) | 22 (21)/2 (2) \| 22 (22)/2 (2) |
| LM SM_T | 20 (19)/4 (3) \| 18 (17)/6 (5) | 22 (21)/2 (2) \| 22 (20)/2 (2) |
| LM SM_TPhi | 20 (19)/4 (4) \| 18 (18)/6 (5) | 22 (21)/2 (2) \| 22 (20)/2 (2) |
| SVM U_B | 21 (20)/3 (1) \| 21 (21)/3 (3) | 22 (22)/2 (1) \| 22 (22)/2 (1) |
| SVM U_T | 21 (20)/3 (1) \| 21 (21)/3 (3) | 22 (22)/2 (1) \| 22 (22)/2 (1) |
| SVM U_TPhi | 23 (19)/1 (1) \| 20 (20)/4 (4) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| SVM O_B | 20 (20)/4 (1) \| 21 (21)/3 (1) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| SVM O_T | 20 (20)/4 (2) \| 21 (21)/3 (3) | 22 (22)/2 (1) \| 22 (22)/2 (2) |
| SVM O_TPhi | 20 (20)/4 (2) \| 21 (21)/3 (3) | 22 (22)/2 (1) \| 22 (22)/2 (2) |
| SVM SM_B | 20 (20)/4 (2) \| 19 (19)/5 (1) | 22 (22)/2 (1) \| 22 (22)/2 (2) |
| SVM SM_T | 20 (20)/4 (3) \| 20 (20)/4 (3) | 22 (21)/2 (2) \| 20 (20)/4 (2) |
| SVM SM_TPhi | 18 (18)/6 (4) \| 19 (19)/5 (4) | 22 (22)/2 (2) \| 22 (20)/2 (2) |
| MARS U_B | 20 (18)/4 (3) \| 23 (18)/1 (1) | 21 (21)/3 (3) \| 21 (20)/3 (3) |
| MARS U_T | 20 (18)/4 (4) \| 20 (18)/4 (2) | 21 (21)/3 (3) \| 21 (19)/3 (2) |
| MARS U_TPhi | 19 (18)/5 (4) \| 22 (19)/2 (2) | 21 (21)/3 (3) \| 21 (21)/3 (3) |
| MARS O_B | 21 (18)/3 (2) \| 19 (18)/5 (1) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| MARS O_T | 21 (17)/3 (2) \| 18 (18)/6 (2) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| MARS O_TPhi | 21 (18)/3 (3) \| 18 (18)/6 (2) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| MARS SM_B | 20 (18)/4 (2) \| 19 (19)/5 (1) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| MARS SM_T | 18 (18)/6 (5) \| 19 (19)/5 (4) | 22 (22)/2 (1) \| 22 (22)/2 (2) |
| MARS SM_TPhi | 19 (17)/5 (4) \| 19 (19)/5 (4) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| RF U_B | 17 (16)/7 (5) \| 19 (18)/5 (1) | 18 (18)/6 (3) \| 19 (18)/5 (2) |
| RF U_T | 18 (16)/6 (5) \| 21 (18)/3 (2) | 18 (18)/6 (3) \| 19 (18)/5 (2) |
| RF U_TPhi | 19 (19)/5 (4) \| 21 (17)/3 (2) | 18 (18)/6 (3) \| 18 (18)/6 (2) |
| RF O_B | 17 (13)/7 (3) \| 20 (17)/4 (2) | 17 (16)/7 (3) \| 18 (16)/6 (2) |
| RF O_T | 13 (12)/11 (3) \| 19 (17)/5 (2) | 15 (15)/9 (7) \| 15 (15)/9 (3) |
| RF O_TPhi | 14 (12)/10 (4) \| 19 (16)/5 (2) | 15 (15)/9 (8) \| 15 (15)/9 (3) |
| RF SM_B | 22 (21)/2 (2) \| 22 (22)/2 (1) | 22 (22)/2 (2) \| 22 (22)/2 (2) |
| RF SM_T | 21 (20)/3 (1) \| 20 (20)/4 (2) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| RF SM_TPhi | 20 (19)/4 (3) \| 20 (20)/4 (2) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| RPART U_B | 19 (17)/5 (4) \| 22 (20)/2 (2) | 22 (21)/2 (2) \| 22 (22)/2 (1) |
| RPART U_T | 19 (17)/5 (4) \| 22 (20)/2 (2) | 21 (21)/3 (2) \| 22 (22)/2 (1) |
| RPART U_TPhi | 19 (16)/5 (4) \| 20 (18)/4 (1) | 22 (22)/2 (2) \| 23 (22)/1 (1) |
| RPART O_B | 18 (15)/6 (3) \| 20 (20)/4 (1) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| RPART O_T | 18 (16)/6 (4) \| 20 (20)/4 (1) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| RPART O_TPhi | 19 (16)/5 (3) \| 21 (20)/3 (1) | 21 (21)/3 (3) \| 22 (22)/2 (2) |
| RPART SM_B | 19 (16)/5 (3) \| 22 (18)/2 (1) | 21 (21)/3 (2) \| 22 (22)/2 (1) |
| RPART SM_T | 17 (15)/7 (5) \| 17 (17)/7 (4) | 21 (20)/3 (3) \| 22 (22)/2 (2) |
| RPART SM_TPhi | 17 (16)/7 (5) \| 19 (18)/5 (3) | 21 (21)/3 (3) \| 22 (22)/2 (2) |

## Anexo C: arquivos gerados

| arquivo | conteúdo |
|---|---|
| `results_cluster/F1_by_dataset.csv` | F1φ médio dos 52 workflows em cada um dos 24 data sets |
| `results_cluster/globalres_all.csv` | precisão, recall e F1φ médios por data set e workflow |
| `results_cluster/paired_comparisons.txt` | saída do `PairedComparisons.py` (todas as tabelas WLdef) |
| `results_cluster/comparison_tables_3_4_5.txt` | Tabelas 3, 4 e 5, nosso \| artigo, com resumo de concordância |
| `results_cluster/fig7_article_readings.csv` | valores lidos da Fig. 7 do artigo |
| `results_cluster/desvios_python_cluster_vs_artigo_fig7.csv` | desvio por célula contra a Fig. 7 |
| `results_cluster/desvios_python_cluster_vs_relatorio.csv` | desvio por célula contra a replicação independente |
| `results_cluster/durations.csv` | tempo de cada data set no cluster |
| `results/apuana/exp_ds*.pkl` (não versionados) | as 50 repetições de cada workflow, com janelas, tempos e erros |
| `COMPARISON.md`, `REVIEW.md` | análises detalhadas que este relatório resume |
| `R_replication/` | execução do código R original e comparação em três colunas |
