# Revisão completa do port: o que foi traduzido, o que foi substituído, e por que os números diferem

Pergunta que motivou esta revisão: "se só traduzimos para Python, não devia dar
tudo igual?" Resposta curta: **não**, porque só uma parte do trabalho é
tradução. O que é tradução linha a linha ficou igual e foi verificado. O que
é substituição de biblioteca (os modelos de regressão e o ARIMA) e o que é
aleatório (janelas Monte Carlo, sorteios da reamostragem) produz números
próximos, não idênticos. E há três lugares em que o código R publicado não
consegue produzir os números do artigo tal como está.

## 1. Inventário: tradução x substituição

| componente | R original | no port | tipo | evidência de equivalência |
|---|---|---|---|---|
| leitura dos dados | `load(.Rdata)` (xts) | parser puro Python (`rdata`) | tradução | 24 séries, tamanhos, índices e valores conferidos |
| embedding | `create.data` (`embed`) | `create_data` | tradução | teste com exemplo manual; V10 = valor mais recente |
| valores faltantes | `knnImputation` (comentado no `Exps.R`; o artigo diz que usou) | `knn_imputation`, port do `DMwR` | tradução | DS12 imputado dá 10,99% de raros; Tabela 1 diz 11,0% |
| relevância `phi` | `uba::phi.control` + spline pchip em C | `tsresamp/uba.py`, linha a linha do C | tradução | % de raros bate com a Tabela 1 nos 24 datasets |
| utilidade, precisão, recall, F1 | `uba::util` em C | `tsresamp/uba.py` | tradução | previsão perfeita dá 1, constante dá 0; DS12 Tabela 6 bate em 10 de 10 |
| under/over/SMOTE, B/T/TPhi | funções do `Exps.R` + `UBL::neighbours` | `tsresamp/resampling.py` | tradução (com sorteio próprio) | tamanhos, ordenação temporal, viés de recência e vizinhos testados; DS12 Tabela 6 bate nas 9 estratégias |
| Monte Carlo | `performanceEstimation::MonteCarlo` | `tsresamp/estimation.py` | tradução (com sorteio próprio) | mesma regra de janelas; testes de fronteira |
| tabelas e Wilcoxon | `GetResults.R`, `pairedComparisons`, `WLdef` | `tsresamp/analysis.py` | tradução | convenções do `wilcox.test` do R reproduzidas |
| lm | `stats::lm` | `LinearRegression` | substituição equivalente | mínimos quadrados: mesma solução |
| svm | `e1071::svm` (libsvm) | `sklearn.svm.SVR` (libsvm) | substituição equivalente | mesmo libsvm, mesma escala de x e y; DS12 bate em 0,01 |
| mars | `earth` | MARS próprio em numpy (até 22/09; desde então port do `earth.c`, ver seção 10) | ~~substituição aproximada~~ → tradução verificada | mesmo modelo que o `earth` nos mesmos dados (seção 10) |
| rf | `randomForest` | `RandomForestRegressor` | substituição equivalente | mesmo algoritmo; regra de tamanho de nó corrigida nesta revisão (ver 4); preditores em postos e teste pareado na seção 10 |
| rpart | `rpart` | `DecisionTreeRegressor` | substituição equivalente | regra do `cp` traduzida como pré-poda nesta revisão (ver 4) e corrigida para poda por complexidade em 22/09 (seção 10) |
| arima | `forecast::auto.arima` + `Arima(model=)` | `pmdarima.auto_arima` + `apply` do statsmodels | **substituição aproximada** | mesmo procedimento de Hyndman-Khandakar, implementação diferente; ordens diferem, F1 igual nos mesmos dados (seção 10) |
| BDES | bagging de `rpart` com estatísticas do embedding | port fiel, árvores do sklearn | tradução + substituição | inclusive o vazamento do alvo em `embedStats` |
| hiperparâmetros | Tabela 7 e Tabela 8 do PDF (não estão no repositório) | `exps_common.py`, `ReplicateTable6.py` | transcrição | conferida pela camada de texto do PDF |

Onde o código R publicado **não consegue** produzir os números do artigo:

1. `smoteRegress*` com `C.perc=list(un, ov)` para em datasets com extremos dos
   dois lados (3 bumps: DS4, DS12 e outros). O artigo reporta SMOTE nesses
   datasets. O port segue o Algoritmo 5 do artigo (u nos bins normais, o nos
   raros).
2. `smote.exsRegressT/TPhi` monta a matriz antes de reordenar por tempo e
   calcula os vizinhos depois: os índices não correspondem. O port reordena
   antes.
3. Datasets com `NA` (12, 13, 23, 24) quebram `phi.control`; a imputação
   está comentada no `Exps.R`. O port imputa (o artigo diz que imputou).

## 2. Tudo o que foi executado

| execução | datasets | workflows | repetições | parâmetros | finalidade |
|---|---|---|---|---|---|
| testes unitários | sintéticos + fixtures do `earth` | todos os componentes | - | - | 127 testes (121 na época) |
| smoke | 1 | 52 | 2 | exemplo do `Exps.R` | pipeline de ponta a ponta |
| smoke NA | 12 | lm, rpart | 2 | exemplo | imputação e complete cases |
| tempo | 21 | 9 | 1 | exemplo | custo por modelo no maior dataset |
| DS1 completo | 1 | 52 | 50 | exemplo | comparação com a Fig. 7 (superado) |
| DS1 completo | 1 | 52 | 50 | Tabela 7 | comparação com a Fig. 7 |
| Tabela 6 | 4, 10, 12 | 10 de svm | 50 | Tabelas 7 e 8 | comparação numérica |
| grade de busca | 10 | U_B, O_B, O_T (78 variantes) | 10 | grade do Anexo 2 | diagnóstico do DS10 |
| série em nível | 10 | 10 de svm | 50 | Tabela 8 | hipótese descartada |
| revisão: DS1 v2 | 1 | 52 | 50 | Tabela 7, mapeamentos corrigidos | efeito do RF e do rpart |
| revisão: outra semente | 1 e 4, 12 | 52 e 10 de svm | 50 | Tabela 7 e 8 | ruído da aleatoriedade |
| revisão: grade completa | 4 | 595 variantes de svm | 10 | grade do Anexo 2 | hipótese do otimismo de seleção |
| revisão: DS4 e DS10 completos | 4, 10 | 52 | 50 | Tabela 7 | comparação com a Fig. 7 (leitura automática) |

Nenhuma execução abrange os 24 datasets; isso é o job do cluster.

## 3. O que foi verificado igual ao artigo

| verificação | resultado |
|---|---|
| % de casos raros por dataset (Tabela 1) | 24 de 24 dentro de 1 ponto (DS12 após imputação kNN: 10,99% x 11,0%) |
| Tabela 6, DS12, 10 configurações de SVM com reamostragem | 10 de 10 dentro de 0,01 |
| Fig. 7, DS1, 52 workflows, parâmetros da Tabela 7 | todos dentro da margem de leitura do gráfico (cerca de 0,03) após a correção do RF |
| ruído de semente (DS1, 52 workflows; Tabela 6 com outra semente) | média 0,008, máximo 0,05 (RF); Tabela 6 muda no máximo 0,015 |

A Tabela 6 do DS12 é a prova mais forte: as nove estratégias de reamostragem,
a função de relevância, a utilidade, o Monte Carlo e o SVM produzem os mesmos
números a duas casas decimais sem nenhum ajuste.

## 4. Os resultados estranhos, um a um

**4.1 RF com under-sampling no DS1: 0,17-0,20 contra 0,30-0,40 no artigo.**
Causa: erro de mapeamento meu, não do R nem do artigo. Em regressão, o
`nodesize=5` do `randomForest` (regTree.c) significa "não dividir nós com 5
casos ou menos"; as folhas resultantes podem ter 1 caso. Eu tinha mapeado
como `min_samples_leaf=5` (folhas com no mínimo 5 casos), o que obriga cada
folha a ser média de 5 pontos e suaviza os extremos, justamente o que
importa aqui. Corrigido para `min_samples_split=6, min_samples_leaf=1`:

| DS1 | mapeamento antigo | corrigido | outra semente | Fig. 7 |
|---|---|---|---|---|
| rf UNDER B / T / TPhi | 0,17 / 0,18 / 0,20 | 0,26 / 0,26 / 0,35 | 0,30 / 0,30 / 0,38 | 0,30-0,40 |
| rf OVER B / T / TPhi | 0,22 / 0,20 / 0,20 | 0,20 / 0,18 / 0,17 | 0,25 / 0,19 / 0,22 | 0,20-0,25 |
| rf SMOTE B / T / TPhi | 0,49 / 0,45 / 0,49 | 0,48 / 0,48 / 0,48 | 0,47 / 0,48 / 0,48 | 0,45 |

Na mesma revisão o `cp` do `rpart` passou a ser aplicado como pré-poda
(`min_impurity_decrease = cp * var(y)`, que é exatamente a regra "a divisão
tem de reduzir o R^2 de falta de ajuste em pelo menos cp"), em vez de
pós-poda por complexidade. Efeito no DS1: no máximo 0,01, dentro do ruído.
**Nota de 22/09:** a leitura do fonte do `rpart` (`partition.c`) mostrou que o `cp` é
exatamente a pós-poda por complexidade; o mapeamento voltou a `ccp_alpha` (seção 10).

**4.2 SVM sem reamostragem no DS1: 0,00 contra 0,40.** Causa: parâmetros. A
primeira rodada usou a "parametrização de exemplo" do `Exps.R` (gamma
0,001), que não é a do artigo. Com os parâmetros da Tabela 7 (custo 300,
gamma 0,01) dá 0,41. Os parâmetros ótimos não estão no repositório R; só no
PDF.

**4.3 DS10, Tabela 6: UNDER B e a família OVER em 0,00 a 0,14 contra 0,65.**
Não é bug: é um fio da navalha em gamma. Com gamma 0,001 o kernel RBF é quase
linear e, no conjunto reamostrado, nem uma regressão linear alcança o limiar
de relevância (previsão máxima 2,8 quando é preciso 4). A grade de busca do
artigo, reproduzida para essas famílias, chega a 0,66-0,68 (o patamar do
artigo) com gamma 0,01 ou com oversampling forte. Descartado por teste: série
em nível, mediana de 10 repetições, erro de transcrição da Tabela 8 (conferida
pela camada de texto do PDF). O que não se reproduz é a combinação exata de
parâmetros que a Tabela 8 lista para essas linhas.

**4.4 DS4, Tabela 6: tudo 0,03 a 0,08 abaixo, mesmo padrão.** Não é ruído de
semente (muda no máximo 0,015). A hipótese de que a Tabela 6 seria o
resultado otimista da própria busca (10 repetições, mediana, melhor de cada
família) foi testada rodando a grade completa do artigo no DS4 (595
variantes): o *melhor* de cada família fica em 0,55-0,62, ainda 0,05 abaixo
dos 0,65-0,67 publicados, e em duas configurações idênticas às da Tabela 8
(baseline e O_B) o artigo tem 0,035 e 0,064 a mais. A busca explica a
uniformidade dos 0,65 do DS10 (nossos máximos lá: 0,67-0,68), mas não o DS4.

**4.5 DS4, experimento principal (Fig. 7, Tabela 7, "balance", 50
repetições).** Para tirar a leitura a olho da equação, os valores da Fig. 7
foram extraídos por processamento de imagem (`Py_Code/tools/read_fig7.py`,
saída em `results/fig7_readings.txt`; a coluna lm, que carrega o eixo, foi
lida à mão). DS4:

| DS4 | artigo (Fig. 7) | port | diferença |
|---|---|---|---|
| lm: sem / under / over / SMOTE | 0,58 / 0,55 / 0,55 / 0,56 (mão) | 0,58 / 0,55 / 0,56 / 0,54 | nenhuma |
| svm: sem / under / over / SMOTE | 0,54 / 0,57 / 0,56 / 0,55 | 0,55 / 0,45 / 0,44 / 0,42 | reamostrados 0,11-0,13 abaixo |
| mars: sem / under / over / SMOTE | 0,53 / 0,58 / 0,58 / 0,58 | 0,51 / 0,45 / 0,49 / 0,49 | reamostrados 0,09-0,13 abaixo |
| rf: sem / under / over / SMOTE | 0,59 / 0,63 / 0,60 / 0,63 | 0,56 / 0,55 / 0,56 / 0,55 | reamostrados 0,04-0,08 abaixo |
| rpart: sem / under / over / SMOTE | 0,57 / 0,58 / 0,58 / 0,57 | 0,53 / 0,58 / 0,58 / 0,56 | nenhuma |
| ARIMA / BDES | 0,37 / 0,23 | 0,57 / 0,63 | port 0,2-0,4 **acima** |

Os baselines batem (o SVM em 0,55 contra 0,54). O que difere é o efeito da
reamostragem: no artigo ela melhora o SVM e o MARS em 0,03-0,05; no port ela
piora em 0,10. Mecanismo, visível nas métricas: a precisão de relevância do
SVM cai de 0,61 para 0,40 e o recall não sobe, ou seja, treinado no conjunto
balanceado (84 casos, metade extremos dos dois lados) o modelo passa a
prever extremos demais. Na grade do DS4, com os parâmetros da Tabela 7 (custo
150, gamma 0,01) qualquer quantidade de under ou oversampling reduz o F1, e
quanto mais reamostragem menor a precisão; com custo 10 a reamostragem ajuda
(0,55 para 0,59). É portanto uma interação entre a regularização do SVM e o
conjunto reamostrado, específica deste dataset (no DS1 e no DS12 o mesmo
código dá o resultado do artigo). O mesmo pipeline com percentuais
explícitos no DS12 reproduz a Tabela 6 em 10 de 10 linhas, o que exclui erro
no caminho de 3 bumps. Não foi possível identificar, sem o ambiente R, por
que o SVM do artigo não perde precisão no DS4.

**4.6 ARIMA e BDES no DS4: port bem acima do artigo.** São as duas
substituições mais distantes do original: `pmdarima` não é o
`forecast::auto.arima` (critério e aproximações diferentes na seleção de
ordem), e o BDES usa árvores do scikit-learn. No DS4 o BDES do port é o
melhor modelo (0,63), o que é consistente com o vazamento do alvo nas
features de média e variância do `embedStats`, reproduzido fielmente; o BDES
do artigo, com o mesmo código, fica em 0,23. Isso sugere que o BDES que gerou
os resultados do artigo não é exatamente o do repositório, mas não há como
confirmar. No DS1 os dois dão zero.

**4.7 DS10, experimento principal (Fig. 7, Tabela 7, "balance", 50
repetições).** Leitura automática da figura (lm à mão):

| DS10 | artigo (Fig. 7) | port | diferença |
|---|---|---|---|
| lm: sem / under / over / SMOTE | 0,00 / 0,65 / 0,67 / 0,65 (mão) | 0,00 / 0,65 / 0,67 / 0,66 | nenhuma |
| svm | 0,51 / 0,63-0,65 / 0,64-0,66 / 0,56-0,65 | 0,59 / 0,57-0,61 / 0,59-0,60 / 0,59-0,62 | baseline +0,08; reamostrados -0,05 |
| mars | 0,51 / 0,49-0,54 / 0,66 / 0,66 | 0,49 / 0,46-0,62 / 0,61 / 0,61 | over e SMOTE -0,05 |
| rf | 0,50 / 0,50-0,56 / 0,51-0,55 / 0,64-0,71 | 0,00 / 0,65 / 0,53-0,60 / 0,62-0,65 | baseline -0,50; under +0,10 |
| rpart | 0,55 / 0,60-0,67 / 0,65 / 0,61-0,66 | 0,54 / 0,60-0,62 / 0,56 / 0,57-0,61 | over -0,09 |
| ARIMA / BDES | 0,47 / 0,10 | 0,00 / 0,00 | ARIMA -0,47 |

O lm, que é substituição exata (mínimos quadrados), bate ao centésimo: isso
confirma relevância, utilidade, reamostragem e janelas no DS10. O SVM e o
rpart ficam dentro de 0,05-0,09. As diferenças grandes estão no RF sem
reamostragem (o do R prevê extremos, o nosso nunca) e no ARIMA (o
`forecast::auto.arima` prevê extremos com F1 0,47; o `pmdarima` nunca): são
as substituições mais distantes do original, e nos dois casos a
implementação R produz previsões mais extremas que a do port. A Tabela 6 do
DS10 (0,64-0,72) coincide com estes valores da Fig. 7 para o SVM com
reamostragem, o que reforça que os 0,65 daquela tabela são o patamar normal
do dataset e não dependem da combinação exata de parâmetros listada.

**4.8 ARIMA, BDES e os baselines lm e rf em 0,00 no DS1.** Não é erro: sem
reamostragem esses modelos nunca preveem uma variação extrema, a precisão de
relevância é o valor mínimo (0,00001) e o F1 vai a zero. O artigo mostra o
mesmo (Fig. 7, linhas pontilhadas em zero na maioria dos datasets).

## 5. Balanço por componente, com os quatro datasets verificados (antes da replicação em R; ver a seção 7 para o veredito final)

| componente | DS1 | DS4 | DS10 | DS12 (Tab. 6) | veredito |
|---|---|---|---|---|---|
| relevância, utilidade, reamostragem, Monte Carlo (via lm e Tab. 1) | bate | bate | bate | bate | tradução correta |
| svm | bate | baseline bate; reamostrado -0,10 | +-0,05 a 0,08 | bate (0,01) | equivalente; interação com reamostragem no DS4 sem explicação |
| rpart | bate | bate | -0,09 no over | - | aproximação boa |
| mars | bate | reamostrado -0,10 | -0,05 | - | aproximação razoável |
| rf | bate (após correção) | reamostrado -0,06 | baseline -0,50, under +0,10 | - | aproximação; o RF do R prevê extremos mais facilmente |
| arima | bate (0) | +0,20 | -0,47 | - | substituição distante; ordem selecionada difere |
| BDES | bate (0) | +0,40 | bate (0) | - | substituição distante; código do artigo pode não ser o do repositório |

## 6. Então por que não dá "tudo igual"?

1. **Cinco modelos e o ARIMA são substituições, não traduções.** `earth`,
   `randomForest`, `rpart`, `e1071` e `forecast` são pacotes R com código
   próprio; não existe versão Python deles. O port usa os equivalentes do
   scikit-learn e do pmdarima e uma implementação própria do MARS. Para lm e
   SVM a equivalência é exata (mesma solução de mínimos quadrados; o mesmo
   libsvm por baixo). Para RF, rpart, MARS e ARIMA é o mesmo algoritmo com
   detalhes de implementação diferentes; cada detalhe, como a regra de
   tamanho de nó do RF, vale alguns centésimos de F1.
2. **Aleatoriedade.** As janelas Monte Carlo, os sorteios do under/over e a
   interpolação do SMOTE dependem do gerador de números aleatórios; o do R não
   é reproduzível em Python. Medido: 0,01 de ruído típico, 0,05 no RF.
3. **O artigo não é só o código.** Os hiperparâmetros vêm do PDF; a imputação
   está comentada no script; três pontos do código R publicado não conseguem
   gerar os números do artigo como estão (SMOTE em datasets de 3 bumps, índice
   do SMOTE temporal, NA). O port tomou a decisão indicada pelo artigo em cada
   caso, e documentou.
4. **Parte dos números do artigo tem otimismo de seleção.** Onde o artigo
   reporta o melhor de uma busca, a réplica com configuração fixa e 50
   repetições tende a ficar abaixo por construção (4.4).

Onde nada disso entra (relevância, utilidade, reamostragem, Monte Carlo, lm e
SVM com parâmetros fixos, como no DS12 e no lm do DS10), os números coincidem
a duas casas. Onde entra uma biblioteca substituída, as diferenças vão de
alguns centésimos (rpart, MARS) a meio ponto de F1 em datasets específicos
(RF e ARIMA no DS10). Para o TCC isso significa: as comparações *entre
estratégias de reamostragem* dentro de cada modelo são reproduzidas com
fidelidade; as comparações *entre modelos* (e contra ARIMA/BDES) carregam a
diferença de implementação e devem ser apresentadas como tal.

## 7. Verificação definitiva: o código R original, rodado aqui

O R foi instalado nesta máquina e o código dos autores foi executado sem
alteração (pasta `R_replication/`, detalhes no README dela) nos mesmos
datasets e com os mesmos parâmetros usados no Python. Três colunas: artigo,
R original, Python.

| F1 médio, 50 repetições | Python x R original | R original x artigo |
|---|---|---|
| DS1, 52 workflows | média 0,009, máximo 0,03 | média 0,015, máximo 0,05 |
| DS4, 52 workflows | média 0,011, máximo 0,03 | média 0,074, máximo 0,39 |
| DS10, 52 workflows | média 0,013 (um outlier: `earth` 0,22 x MARS próprio 0,49 no baseline) | média 0,086, máximo 0,50 |
| Tabela 6 (DS12 / DS4 / DS10) | 0,004 / 0,012 / 0,02 | 0,007 / 0,063 / 0,245 |

Conclusões:

1. **O port reproduz o código R original.** Nos 156 pares de workflows a
   diferença média é de 0,01, o nível do ruído de semente, incluindo RF,
   rpart, ARIMA e BDES. A única substituição que produz número diferente é o
   MARS sem reamostragem no DS10 (`earth` 0,22, nosso 0,49; o artigo mostra
   0,51). Com reamostragem os dois MARS coincidem.
2. **O código R publicado não reproduz várias linhas do artigo.** No DS4, o
   R original dá o mesmo que o Python: SVM e MARS com reamostragem 0,10 abaixo
   da Fig. 7, BDES 0,63 contra 0,24 e ARIMA 0,58 contra 0,38. No DS10 dá RF
   baseline 0,00 (artigo 0,50), ARIMA 0,00 (0,49) e MARS baseline 0,22
   (0,51). Na Tabela 6, dá OVER B 0,000 e UNDER B 0,116 no DS10 (artigo
   0,65) e **falha** com a mensagem "The percentages provided must be the same
   length as the number of bumps!" nas nove linhas de SMOTE de DS4 e DS12.
3. Portanto as discrepâncias das seções 3 e 4 são entre o artigo e o seu
   próprio código publicado, não entre o Python e o R. O que o artigo rodou
   para produzir essas linhas não é exatamente o que está no repositório.
4. Uma consequência prática: a "correção" do descasamento de índices do SMOTE
   temporal, que eu tinha aplicado, afastava o port do R (0,535 contra 0,654
   no SM_T do DS10). O comportamento original voltou a ser o padrão
   (`r_index_quirk=True`), com a correção como opção.

## 8. Comparação com uma replicação independente em R (relatório `main.pdf`, J. P. Miranda)

O relatório "Resampling Strategies for Time Series Forecasting (Moniz et al.
2017)" reexecuta o pipeline R original nos 24 datasets, 52 workflows, 50
folds, com os hiperparâmetros da Tabela 7, em outra máquina. Os valores das
Tabelas 15 a 19 do relatório (F1 médio por modelo, estratégia e dataset;
extraídos da camada de texto do PDF para
`R_replication/results/joao_report_F1.json`) foram comparados com o nosso R
original nos 150 pares disponíveis (DS1, DS4, DS10 x 5 modelos x 10
estratégias):

| modelo | média de \|relatório - nosso R\| | máximo | células acima de 0,02 |
|---|---|---|---|
| lm | 0,001 | 0,006 | 0 de 30 |
| svm | 0,003 | 0,010 | 0 de 30 |
| mars | 0,006 | 0,040 | 3 de 30 |
| rf | 0,009 | 0,047 | 7 de 30 |
| rpart | 0,004 | 0,019 | 0 de 30 |

As duas execuções do código R original, em máquinas diferentes, são
idênticas ao nível do sorteio aleatório (lm e svm, determinísticos dada a
janela, coincidem ao milésimo; rf, o mais estocástico, difere no máximo
0,05). O relatório chega às mesmas conclusões desta revisão quanto ao
artigo: a "Table 7" dele (SVM em DS4/DS10/DS12) reporta os mesmos valores
do nosso R (DS4 baseline 0,560, U_B 0,450, O_B 0,444; DS10 baseline 0,609,
O_B 0,589) e registra que a Tabela 3 do artigo não se reproduz para o SVM
(58% de vitórias contra 33% no artigo).

Consequência para o port Python: as três execuções (R do relatório, R
nosso, Python) concordam entre si a 0,01; o único ponto em que o Python se
afasta das duas execuções em R é o MARS sem reamostragem no DS10 (0,49
contra 0,22 e 0,20), a implementação própria do MARS.

## 9. Run completo no Apuana (18/09/2026)

Os 24 data sets rodaram no cluster (job 15477, 9 h, 48 CPUs) com o port em Python, Tabela 7 e
knnImputation. Contra as Tabelas 3, 4 e 5 do artigo (vitórias/derrotas de Wilcoxon sobre os 24
data sets), a direção de cada comparação coincide em 93 %, 77 % e 100 % das células e a
diferença média é de 1 a 3 vitórias em 24, o que é o ruído esperado da semente (seção 6). As
três hipóteses do artigo saem confirmadas com os nossos números. Contra a Fig. 7 (F1 por data
set, 1232 pontos lidos da figura), o desvio mediano é 0,013 e 84 % dos pontos ficam a 0,05 ou
menos; as médias por modelo e estratégia coincidem a 0,02. Os poucos pontos discrepantes
(baselines em DS4/DS10) são iguais no R original, isto é, vêm do código publicado, não do port;
a exceção é o MARS baseline em DS10 (substituto do `earth`). Contra a replicação em R do
relatório `main.pdf`, 19 dos 24 data sets ficam a menos de 0,025 de F1; os outros cinco (12,
20, 21, 22, 24) divergem por decisões do relatório que não estão no artigo (complete.cases,
teto de 8.000 linhas de treino em DS21 a 24, baselines zeradas em DS20). Tabelas e explicação
completas em `COMPARISON.md`, seção 9; dados em `results_cluster/`.

## 10. Auditoria de 22/09/2026: substituições que viraram traduções

A pergunta desta revisão era "se só traduzimos, não devia dar tudo igual?". Depois do run
completo, sobrava um único componente com desvio sistemático em relação ao R: o MARS
(seção 7, conclusão 1: `earth` 0,22 contra 0,49 na baseline do DS10; e, contra o relatório
independente, as famílias UNDER de DS15/16/19 a +0,13 a +0,22). A auditoria (detalhes e
tabelas em `COMPARISON.md`, seção 10) fez o teste que faltava: **R e Python ajustados nos
mesmos conjuntos de treino e teste**, learner a learner.

| componente | antes | agora | evidência |
|---|---|---|---|
| mars | MARS próprio (nós nos quantis, sem as regras do `earth`) | **port do `earth.c` 5.3.6** (minspan/endspan, termo linear com `MIN_BX_SOS`, fila Fast MARS, regras de parada) e da poda `leaps::BAKWRD` | 74 conjuntos de 10 data sets: mesmos termos, nós e coeficientes, predições a 1e-14; 50 janelas de DS10 e DS1: F1 igual iteração a iteração |
| rpart | `cp` como pré-poda (`min_impurity_decrease`) | `cp` como poda de custo-complexidade (`ccp_alpha = cp·var(y)`), lido em `rpart/src/partition.c`; preditores em postos (`TreeRanks`) para anular a tolerância de 1e-7 do scikit-learn | mesmo número de folhas em 38/50 janelas do DS1 (antes 12/50); F1 a 0,001 do R (antes 0,010) |
| rf | `RandomForestRegressor` com `nodesize` corrigido (seção 4.1) | idem + preditores em postos | F1 igual ao `randomForest` nos mesmos conjuntos (DS1 exato; DS2/DS3 com oversampling a 0,001/0,007, dentro do ruído das florestas) |
| BDES | árvores com pré-poda | árvores com `ccp` + postos | +0,001 de F1 em DS4 |
| arima | `pmdarima` | inalterado | F1 a 0,004 do `forecast` nas mesmas janelas; ordens diferem, predições não |
| lm, svm | exatos | inalterados | - |

Por que o MARS antigo dava mais? Duas regras do `earth` que um MARS "de livro" não tem:
(1) os nós são escolhidos entre os **casos** ordenados, com `endspan` casos proibidos em cada
extremo e um nó a cada `minspan` casos, o que impede hinges nos extremos da variável, e
(2) o candidato linear é descartado quando a sua soma de quadrados ortogonalizada é ≤ 0,01 em
valor absoluto, o que em séries de valores pequenos (DS15/16/19, valores ~0,01) e em conjuntos
sob-amostrados (menos linhas, soma menor) reduz o modelo a poucos hinges. O nosso MARS
ajustava mais termos e extrapolava mais nos extremos, exatamente onde o F1φ é medido.

O que fica registrado como diferença de implementação sem correção possível dentro do
scikit-learn: empates exatos do critério de divisão entre variáveis no `rpart` (DS10, série de
inteiros), que o R desempata pela ordem das colunas e o scikit-learn por uma ordem aleatória.
Efeito medido: 0,002 de F1.

Consequência para o balanço da seção 5: as linhas "mars" e "rpart" passam de "aproximação" a
"tradução verificada nos mesmos dados"; "rf" e "arima" ficam como equivalentes verificados
dentro do ruído próprio de cada método. As tabelas do run completo foram recalculadas com os
workflows MARS, RPART, BDES e RF reexecutados localmente com este código (`results_cluster/`,
proveniência em `results_cluster/PROVENANCE.md`).
