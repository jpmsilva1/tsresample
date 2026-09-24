# Comparação do port Python com os resultados publicados

Artigo: Moniz, Branco & Torgo, "Resampling strategies for imbalanced time
series forecasting", Int J Data Sci Anal (2017) 3:161-181
(`../s41060-017-0044-3.pdf`). Todos os números abaixo foram obtidos com
`Py_Code/Exps.py` (parâmetros da Tabela 7) e `Py_Code/ReplicateTable6.py`
(parâmetros da Tabela 8), 50 repetições Monte Carlo, 50%/25%.

## 1. Função de relevância (Tabela 1, coluna "% Rare")

Percentual de casos raros (phi > 0,9) em cada série, artigo x port:

| DS | artigo | port | DS | artigo | port | DS | artigo | port |
|---|---|---|---|---|---|---|---|---|
| 1 | 9,9 | 10,0 | 9 | 21,1 | 21,6 | 17 | 11,6 | 11,9 |
| 2 | 9,3 | 9,2 | 10 | 4,8 | 4,9 | 18 | 10,1 | 11,0 |
| 3 | 7,8 | 6,8 | 11 | 13,3 | 13,3 | 19 | 8,2 | 7,8 |
| 4 | 13,3 | 13,2 | 12 | 11,0 | 5,2 | 20 | 6,8 | 7,6 |
| 5 | 3,5 | 3,5 | 13 | 11,1 | 10,9 | 21 | 1,8 | 1,8 |
| 6 | 4,8 | 4,8 | 14 | 16,3 | 15,7 | 22 | 10,2 | 10,2 |
| 7 | 12,5 | 12,5 | 15 | 11,4 | 10,8 | 23 | 0,08 | 0,1 |
| 8 | 17,6 | 17,9 | 16 | 9,7 | 9,9 | 24 | 3,4 | 3,4 |

23 das 24 séries coincidem dentro de 1 ponto na série bruta. A exceção
(DS12, 5,2%) tem valores em falta: aplicando `knnImputation` ao dataframe
embutido, como o comentário do `Exps.R` e a seção 4.1 do artigo indicam, o
percentual passa a 10,99%, igual aos 11,0% da Tabela 1. Conclusão:
`phi.control`/`phi` do `uba` estão corretamente portados e a imputação é a
mesma do artigo.

## 2. Dataset 1 (Fig. 7 do artigo, F1 médio; leitura visual, +-0,03)

| workflow | artigo | port (B / T / TPhi) |
|---|---|---|
| lm | 0,00 | 0,00 |
| lm under / over / SMOTE | 0,52-0,55 / 0,51-0,55 / 0,50-0,52 | 0,53 0,53 0,53 / 0,53 0,54 0,54 / 0,52 0,51 0,50 |
| svm | 0,40 | 0,41 |
| svm under / over / SMOTE | 0,45 / 0,42 / 0,42 | 0,45 0,46 0,47 / 0,45 0,45 0,45 / 0,44 0,44 0,45 |
| mars | 0,25 | 0,26 |
| mars under / over / SMOTE | 0,50 / 0,50-0,52 / 0,50 | 0,48 0,47 0,49 / 0,52 0,51 0,51 / 0,49 0,49 0,49 |
| rf | 0,02 | 0,00 |
| rf under / over / SMOTE | 0,30-0,40 / 0,20-0,25 / 0,45 | 0,26 0,26 0,35 / 0,20 0,18 0,17 / 0,48 0,48 0,48 |
| rpart | 0,43 | 0,43 |
| rpart under / over / SMOTE | 0,45 / 0,45 / 0,45 | 0,48 0,46 0,47 / 0,45 0,46 0,44 / 0,47 0,45 0,46 |
| arima / BDES | 0,00 / 0,00 | 0,00 / 0,00 |

Tudo dentro da margem de leitura do gráfico. Dois ajustes foram necessários
para chegar aqui: os parâmetros da Tabela 7 (com a parametrização de exemplo
do `Exps.R` o SVM sem reamostragem fica em 0,00) e a semântica correta do
`nodesize` do `randomForest` (ver `REVIEW.md`; com folhas de no mínimo 5
casos o RF com under-sampling ficava em 0,17-0,20).

## 2b. Dataset 4 (Fig. 7, leitura automática da figura; F1 médio, 50 repetições)

| DS4 | artigo | port |
|---|---|---|
| lm: sem / under B T TPhi / over B T TPhi / SMOTE B T TPhi | 0,58 / ~0,55 / ~0,55 / ~0,56 | 0,58 / 0,55 0,55 0,54 / 0,56 0,56 0,56 / 0,54 0,54 0,53 |
| svm | 0,54 / 0,57 0,58 0,56 / 0,58 0,55 0,56 / 0,56 0,55 0,54 | 0,55 / 0,45 0,46 0,45 / 0,44 0,44 0,43 / 0,42 0,43 0,42 |
| mars | 0,53 / 0,58 0,57 0,59 / 0,57 0,59 0,59 / 0,59 0,57 0,57 | 0,51 / 0,45 0,46 0,45 / 0,48 0,50 0,49 / 0,49 0,49 0,48 |
| rf | 0,59 / 0,61 0,63 0,64 / 0,62 0,58 0,60 / 0,67 0,60 0,62 | 0,56 / 0,55 0,56 0,55 / 0,57 0,56 0,56 / 0,55 0,55 0,53 |
| rpart | 0,57 / 0,56 0,60 0,58 / 0,60 0,56 0,58 / 0,59 0,57 0,57 | 0,53 / 0,58 0,58 0,57 / 0,59 0,58 0,58 / 0,57 0,57 0,55 |
| ARIMA / BDES | 0,37 / 0,23 | 0,57 / 0,63 |

Ver `REVIEW.md` 4.5 e 4.6.

## 2c. Dataset 10 (Fig. 7, leitura automática; F1 médio, 50 repetições)

| DS10 | artigo | port |
|---|---|---|
| lm: sem / under B T TPhi / over B T TPhi / SMOTE B T TPhi | 0,00 / ~0,65 / ~0,67 / ~0,65 | 0,00 / 0,65 0,65 0,65 / 0,67 0,67 0,67 / 0,67 0,66 0,66 |
| svm | 0,51 / 0,63 0,64 0,65 / 0,65 0,66 0,64 / 0,56 0,64 0,65 | 0,59 / 0,58 0,57 0,61 / 0,59 0,60 0,60 / 0,59 0,59 0,62 |
| mars | 0,51 / 0,49 0,54 0,52 / 0,66 0,67 0,66 / 0,66 0,66 0,66 | 0,49 / 0,48 0,46 0,62 / 0,61 0,61 0,61 / 0,61 0,61 0,62 |
| rf | 0,50 / 0,54 0,50 0,56 / 0,51 0,55 0,53 / 0,71 0,64 0,68 | 0,00 / 0,66 0,65 0,65 / 0,53 0,60 0,53 / 0,62 0,65 0,65 |
| rpart | 0,55 / 0,60 0,62 0,67 / 0,65 0,66 0,65 / 0,61 0,66 0,64 | 0,54 / 0,60 0,60 0,62 / 0,56 0,56 0,56 / 0,57 0,57 0,61 |
| ARIMA / BDES | 0,47 / 0,10 | 0,00 / 0,00 |

Ver `REVIEW.md` 4.7.

## 3. Tabela 6 (SVM com parâmetros da Tabela 8; F1 médio)

| estratégia | DS4 artigo | DS4 port | DS10 artigo | DS10 port | DS12 artigo | DS12 port |
|---|---|---|---|---|---|---|
| svm | 0,584 | 0,549 | 0,638 | 0,589 | 0,554 | 0,557 |
| U_B | 0,668 | 0,583 | 0,652 | 0,141 | 0,610 | 0,609 |
| U_T | 0,659 | 0,588 | 0,643 | 0,632 | 0,614 | 0,606 |
| U_TPhi | 0,651 | 0,581 | 0,647 | 0,673 | 0,630 | 0,621 |
| O_B | 0,653 | 0,597 | 0,651 | 0,000 | 0,611 | 0,611 |
| O_T | 0,650 | 0,596 | 0,652 | 0,101 | 0,615 | 0,609 |
| O_TPhi | 0,651 | 0,495 | 0,652 | 0,085 | 0,611 | 0,606 |
| SM_B | 0,662 | 0,585 | 0,675 | 0,681 | 0,609 | 0,600 |
| SM_T | 0,656 | 0,600 | 0,698 | 0,535 | 0,600 | 0,603 |
| SM_TPhi | 0,649 | 0,568 | 0,721 | 0,694 | 0,620 | 0,610 |

* DS12: as 10 linhas coincidem com diferença máxima de 0,01.
* DS4: todas as linhas 0,03-0,08 abaixo (O_TPhi 0,16), com o mesmo padrão
  (reamostragem acima do baseline). Extremos dos dois lados; os pontos de
  partida Monte Carlo diferem dos do R.
* DS10: U_T, U_TPhi, SM_B e SM_TPhi coincidem; U_B, O_B, O_T e O_TPhi
  colapsam (precisão nula em todas as repetições). Verificado: a estrutura
  de bumps e o percentual de raros coincidem com o artigo, a transcrição da
  Tabela 8 está correta (camada de texto do PDF) e, no conjunto reamostrado,
  nem uma regressão linear prevê acima de 2,8 quando o limiar de relevância
  exige cerca de 4; um SVM com gamma 0,001 é quase linear. O scikit-learn usa
  o mesmo libsvm do e1071, portanto com as mesmas entradas o resultado seria
  o mesmo. Hipóteses testadas e descartadas: série em nível em vez de
  diferenças (dá 0,33 em tudo); mediana de 10 repetições em vez de média de
  50 (O_B dá zero em 50 de 50 janelas).
* DS10, grade de busca (`OptParmsSearch`, 10 repetições, mediana do F1),
  reproduzida para U_B, O_B e O_T: o F1 é um fio da navalha em gamma. Com
  gamma 0,01, ou com gamma 0,001 e o = 10, U_B/O_B/O_T chegam a 0,66-0,68,
  o mesmo patamar dos 0,65 do artigo; com gamma 0,001 e o <= 3 dão zero.
  Ou seja, o resultado do artigo para o DS10 (SVM com reamostragem em torno
  de 0,65, acima do baseline) é reproduzido pela busca; o que não se reproduz
  é a combinação específica de parâmetros listada na Tabela 8 para essas
  linhas. Resultados em `results/search_ds10_subset.pkl`.

## 4. Leitura geral

Relevância, utilidade, divisão Monte Carlo, estratégias de reamostragem e
os learners reproduzem os resultados publicados na grande maioria dos casos
(DS12 quase exato; DS1 e DS10 consistentes na maior parte). As conclusões
qualitativas do artigo (Hipóteses 1 a 3) aparecem nos nossos números. As
diferenças que restam concentram-se em configurações em que o conjunto de
treino reamostrado continua dominado por casos normais.

## 5. Ruído da aleatoriedade (semente do Monte Carlo)

DS1, 52 workflows, parâmetros da Tabela 7, semente 1234 contra 4321:
diferença média de F1 de 0,008, mediana 0,005, máxima 0,05 (variantes de RF).
Tabela 6 nos DS4 e DS12 com semente 4321: nenhuma linha muda mais de 0,015.
Como 50 janelas de 182 possíveis se sobrepõem muito, a média é estável.
Consequência: diferenças abaixo de 0,02 (0,05 para RF) são indistinguíveis de
ruído; o desvio sistemático do DS4 (0,03 a 0,08) não é ruído.

## 6. Verificação definitiva: o código R original rodado localmente

O R 4.6.1 foi instalado nesta máquina com os pacotes originais (`uba` compilado
do GitHub, `UBL` mínimo, `DMwR` do arquivo do CRAN, `performanceEstimation`,
`e1071`, `randomForest`, `rpart`, `earth`, `forecast`). As funções do
`Exps.R` e do `OptParmsSearch.R` foram carregadas **sem nenhuma alteração**
(`../R_replication/run_exps.R`, `run_table6.R`); só se acrescentou o que o
artigo descreve e o script deixa em aberto: parâmetros das Tabelas 7 e 8 e
`knnImputation`. Comparação em três colunas (`../R_replication/compare.py`).

### Tabela 6: artigo x R original x Python (F1 médio, 50 repetições)

| DS12 | artigo | R original | Python |
|---|---|---|---|
| svm | 0,554 | 0,551 | 0,557 |
| U_B / U_T / U_TPhi | 0,610 / 0,614 / 0,630 | 0,605 / 0,605 / 0,623 | 0,609 / 0,606 / 0,621 |
| O_B / O_T / O_TPhi | 0,611 / 0,615 / 0,611 | 0,612 / 0,603 / 0,602 | 0,611 / 0,609 / 0,606 |
| SM_B / SM_T / SM_TPhi | 0,609 / 0,600 / 0,620 | **falha** (3 bumps) | 0,600 / 0,603 / 0,610 |

| DS4 | artigo | R original | Python |
|---|---|---|---|
| svm | 0,584 | 0,560 | 0,549 |
| U_B / U_T / U_TPhi | 0,668 / 0,659 / 0,651 | 0,597 / 0,603 / 0,599 | 0,583 / 0,588 / 0,581 |
| O_B / O_T / O_TPhi | 0,653 / 0,650 / 0,651 | 0,613 / 0,607 / 0,498 | 0,597 / 0,596 / 0,495 |
| SM_B / SM_T / SM_TPhi | 0,662 / 0,656 / 0,649 | **falha** (3 bumps) | 0,585 / 0,600 / 0,568 |

| DS10 | artigo | R original | Python |
|---|---|---|---|
| svm | 0,638 | 0,609 | 0,589 |
| U_B / U_T / U_TPhi | 0,652 / 0,643 / 0,647 | **0,116** / 0,654 / 0,670 | 0,141 / 0,632 / 0,673 |
| O_B / O_T / O_TPhi | 0,651 / 0,652 / 0,652 | **0,000** / **0,087** / **0,087** | 0,000 / 0,101 / 0,085 |
| SM_B / SM_T / SM_TPhi | 0,675 / 0,698 / 0,721 | 0,683 / 0,654 / 0,704 | 0,681 / 0,602 / 0,710 |

|Python - R original|: DS12 média 0,004; DS4 0,012; DS10 0,02 (SM_T 0,05).
|R original - artigo|: DS12 0,007; DS4 0,063; DS10 0,245.

Leitura: o port reproduz o código R original ao nível do ruído de semente.
O que o código R publicado **não** reproduz é o artigo: no DS4 fica 0,02 a
0,15 abaixo em todas as linhas; no DS10 produz exatamente o mesmo colapso de
U_B, O_B, O_T e O_TPhi que o port; e o SMOTE com percentuais explícitos não
roda em datasets com extremos dos dois lados (DS4, DS12). As discrepâncias
apontadas nas seções 3 e 4 são portanto do artigo em relação ao seu próprio
código, e não do port.

### Experimento principal (52 workflows, Tabela 7, balance): artigo (Fig. 7) x R original x Python

Tabela completa em `R_replication/results/comparison_article_R_python.txt`. Resumo das
diferenças absolutas médias de F1: Python x R original 0,009 (DS1), 0,011 (DS4), 0,013
(DS10); R original x artigo 0,015 (DS1), 0,074 (DS4), 0,086 (DS10). As linhas em que o R
original mais se afasta do artigo são as mesmas em que o Python se afastava: SVM e MARS
com reamostragem no DS4 (R 0,43-0,50 contra 0,55-0,59), BDES e ARIMA no DS4 (R 0,63 e
0,58 contra 0,24 e 0,38), RF e ARIMA sem reamostragem no DS10 (R 0,00 contra 0,50 e 0,49).

## 7. Replicação independente em R (relatório de J. P. Miranda, `main.pdf`)

Tabelas 15-19 do relatório contra o nosso R original, F1 médio de 50 repetições
(célula = relatório / nosso R):

| DS1 | lm | svm | mars | rf | rpart |
|---|---|---|---|---|---|
| sem reamostragem | 0,000 / 0,000 | 0,429 / 0,429 | 0,268 / 0,268 | 0,011 / 0,017 | 0,432 / 0,432 |
| UNDER B / T / TPhi | 0,527/0,530 · 0,529/0,529 · 0,523/0,522 | 0,444/0,446 · 0,465/0,455 · 0,477/0,472 | 0,488/0,465 · 0,474/0,475 · 0,493/0,500 | 0,276/0,272 · 0,287/0,275 · 0,403/0,376 | 0,457/0,457 · 0,460/0,458 · 0,465/0,468 |
| OVER B / T / TPhi | 0,531/0,532 · 0,539/0,542 · 0,539/0,541 | 0,451/0,445 · 0,448/0,449 · 0,447/0,451 | 0,497/0,502 · 0,498/0,497 · 0,500/0,499 | 0,215/0,168 · 0,170/0,150 · 0,160/0,150 | 0,453/0,457 · 0,447/0,446 · 0,450/0,443 |
| SMOTE B / T / TPhi | 0,508/0,508 · 0,505/0,508 · 0,504/0,503 | 0,442/0,438 · 0,454/0,451 · 0,444/0,446 | 0,493/0,490 · 0,497/0,505 · 0,503/0,510 | 0,470/0,475 · 0,505/0,489 · 0,497/0,503 | 0,459/0,470 · 0,465/0,465 · 0,463/0,453 |

| DS4 | lm | svm | mars | rf | rpart |
|---|---|---|---|---|---|
| sem reamostragem | 0,589 / 0,589 | 0,560 / 0,560 | 0,523 / 0,523 | 0,573 / 0,573 | 0,520 / 0,521 |
| UNDER B / T / TPhi | 0,553/0,555 · 0,561/0,561 · 0,543/0,546 | 0,450/0,444 · 0,453/0,452 · 0,457/0,460 | 0,462/0,462 · 0,475/0,470 · 0,467/0,462 | 0,563/0,566 · 0,566/0,564 · 0,563/0,564 | 0,585/0,566 · 0,585/0,587 · 0,587/0,573 |
| OVER B / T / TPhi | 0,575/0,575 · 0,568/0,568 · 0,568/0,567 | 0,444/0,442 · 0,437/0,436 · 0,435/0,432 | 0,486/0,491 · 0,509/0,503 · 0,507/0,505 | 0,583/0,583 · 0,578/0,580 · 0,580/0,580 | 0,602/0,606 · 0,593/0,596 · 0,592/0,595 |
| SMOTE B / T / TPhi | 0,548/0,545 · 0,562/0,564 · 0,551/0,548 | 0,424/0,425 · 0,452/0,449 · 0,442/0,441 | 0,487/0,487 · 0,507/0,506 · 0,504/0,502 | 0,560/0,563 · 0,572/0,571 · 0,562/0,560 | 0,585/0,582 · 0,574/0,573 · 0,573/0,573 |

| DS10 | lm | svm | mars | rf | rpart |
|---|---|---|---|---|---|
| sem reamostragem | 0,000 / 0,000 | 0,609 / 0,609 | 0,195 / 0,219 | 0,000 / 0,000 | 0,569 / 0,569 |
| UNDER B / T / TPhi | 0,650/0,645 · 0,650/0,649 · 0,648/0,647 | 0,569/0,576 · 0,578/0,580 · 0,614/0,615 | 0,484/0,492 · 0,524/0,485 · 0,623/0,626 | 0,646/0,653 · 0,649/0,649 · 0,648/0,648 | 0,597/0,599 · 0,604/0,598 · 0,619/0,621 |
| OVER B / T / TPhi | 0,668/0,668 · 0,667/0,667 · 0,667/0,667 | 0,589/0,589 · 0,601/0,599 · 0,601/0,599 | 0,613/0,606 · 0,604/0,606 · 0,604/0,606 | 0,518/0,543 · 0,569/0,593 · 0,569/0,593 | 0,559/0,563 · 0,555/0,552 · 0,555/0,552 |
| SMOTE B / T / TPhi | 0,666/0,666 · 0,647/0,647 · 0,645/0,645 | 0,586/0,588 · 0,580/0,583 · 0,605/0,605 | 0,614/0,609 · 0,601/0,606 · 0,613/0,617 | 0,619/0,643 · 0,620/0,613 · 0,633/0,635 | 0,559/0,568 · 0,555/0,564 · 0,599/0,599 |

Diferença média por modelo: lm 0,001, svm 0,003, mars 0,006, rf 0,009, rpart 0,004.

## 9. Run completo no cluster Apuana (24 data sets, 52 workflows, 50 repetições)

Job 15477 (`short-simple`, 48 CPUs, 500 GB, 3 faixas de 16 workers), 18/09/2026 00:17 a 09:16
(9 h). Parâmetros da Tabela 7, NA por knnImputation, semente 1234. Saídas versionadas em
`results_cluster/` (`globalres_all.csv`, `F1_by_dataset.csv`, `paired_comparisons.txt`,
`comparison_tables_3_4_5.txt`, `desvios_python_cluster_vs_relatorio.csv`, `durations.csv`).
Script: `tools/compare_full_run.py`.

### 9.1 Tabelas 3 e 4 do artigo: nosso | artigo

Formato: vitórias (significativas) / derrotas (significativas), teste de Wilcoxon pareado sobre
os 24 data sets, F1φ, α = 0,05.

**Tabela 3** (estratégia base vs modelo sem resampling)

| | LM | SVM | MARS | RF | RPART |
|---|---|---|---|---|---|
| U_B | 19 (19)/5 (4) \| 19 (18)/5 (4) | 11 (10)/13 (10) \| 8 (6)/16 (8) | 17 (16)/7 (5) \| 15 (12)/9 (7) | 19 (17)/5 (4) \| 18 (17)/6 (2) | 11 (9)/13 (5) \| 12 (8)/12 (3) |
| O_B | 19 (19)/5 (1) \| 18 (17)/6 (3) | 14 (11)/10 (6) \| 7 (6)/17 (10) | 21 (19)/3 (3) \| 17 (17)/7 (4) | 21 (14)/3 (3) \| 20 (15)/4 (1) | 12 (11)/12 (7) \| 11 (9)/13 (8) |
| SM_B | 20 (19)/4 (3) \| 19 (18)/5 (3) | 11 (9)/13 (8) \| 7 (6)/17 (10) | 21 (18)/3 (2) \| 18 (17)/6 (4) | 21 (20)/3 (1) \| 20 (20)/4 (1) | 12 (10)/12 (6) \| 10 (10)/14 (7) |

**Tabela 4** (viés temporal / relevância vs a versão base da mesma estratégia)

| | LM | SVM | MARS | RF | RPART |
|---|---|---|---|---|---|
| U_T | 18 (2)/6 (0) \| 14 (2)/10 (0) | 12 (2)/12 (1) \| 10 (0)/14 (0) | 15 (3)/9 (1) \| 11 (0)/13 (2) | 13 (1)/11 (2) \| 12 (1)/12 (2) | 11 (0)/13 (1) \| 14 (4)/10 (0) |
| U_TPhi | 14 (6)/10 (2) \| 15 (10)/9 (3) | 16 (8)/8 (3) \| 11 (5)/13 (4) | 11 (7)/13 (1) \| 17 (6)/7 (1) | 13 (5)/11 (3) \| 16 (6)/8 (3) | 14 (6)/10 (2) \| 16 (7)/8 (5) |
| O_T | 12 (4)/12 (7) \| 14 (8)/10 (9) | 8 (4)/16 (7) \| 12 (5)/12 (6) | 12 (4)/12 (6) \| 11 (3)/13 (4) | 7 (1)/17 (5) \| 8 (4)/16 (4) | 11 (3)/13 (1) \| 12 (3)/12 (2) |
| O_TPhi | 10 (5)/14 (7) \| 14 (9)/10 (7) | 7 (4)/17 (10) \| 12 (4)/12 (7) | 10 (4)/14 (3) \| 11 (3)/13 (2) | 6 (0)/18 (4) \| 8 (2)/16 (5) | 10 (2)/14 (0) \| 14 (3)/10 (2) |
| SM_T | 6 (3)/18 (14) \| 6 (5)/18 (13) | 10 (3)/14 (13) \| 10 (5)/14 (10) | 13 (4)/11 (10) \| 9 (6)/15 (10) | 6 (4)/18 (13) \| 9 (3)/15 (10) | 5 (1)/19 (12) \| 8 (1)/15 (11) |
| SM_TPhi | 6 (2)/18 (10) \| 6 (4)/18 (11) | 11 (10)/13 (10) \| 9 (6)/15 (12) | 15 (4)/9 (7) \| 12 (4)/12 (6) | 6 (6)/18 (9) \| 12 (5)/12 (10) | 7 (4)/17 (10) \| 6 (4)/17 (9) |

**Tabela 5** (cada modelo + estratégia vs ARIMA e vs BDES; 90 células, tabela completa em
`results_cluster/comparison_tables_3_4_5.txt`): a direção (quem ganha) coincide em 100 % das
células; |Δvitórias| média 1,1, e 83 % das células ficam a 2 vitórias ou menos do artigo. As
maiores diferenças são RF O_T e O_TPhi vs ARIMA (nosso 13 e 14 vitórias, artigo 19).

Resumo da concordância:

| tabela | células | \|Δvitórias\| média | máx | células a ≤ 2 vitórias | mesma direção |
|---|---|---|---|---|---|
| 3 | 15 | 2,1 | 7 | 67 % | 93 % |
| 4 | 30 | 2,6 | 6 | 50 % | 77 % |
| 5 | 90 | 1,1 | 6 | 83 % | 100 % |

As conclusões do artigo se sustentam nos nossos números: (H1) as três estratégias base vencem o
modelo sem resampling em lm, mars, rf e, com margem menor, rpart; em svm o artigo dá 7 a 8
vitórias em 24 e nós damos 11 a 14 (o único desvio de direção da Tabela 3); (H2) TPhi ajuda
no undersampling e o SmoteR com viés perde para o SM_B, como no artigo; o oversampling com viés
fica mais fraco nos nossos números do que no artigo (O_TPhi em LM/RPART: 10 vitórias contra 14);
(H3) todos os modelos com resampling vencem ARIMA e BDES em 17 a 23 dos 24 data sets, como no
artigo. As diferenças de 2 a 7 vitórias vêm de data sets em que a diferença entre os dois
workflows é pequena e o sinal muda com a semente do Monte Carlo (seção 5); não há como
reproduzir a contagem exata sem a semente original.

### 9.2 F1 por data set vs a replicação independente em R (relatório `main.pdf`)

1100 células (24 data sets x 50 workflows, sem arima/BDES): |desvio| mediano 0,008, 75 % das
células a 0,02 ou menos, 87 % a 0,05 ou menos. Em 19 dos 24 data sets (1 a 11, 13 a 19, 23) o
|desvio| médio fica entre 0,003 e 0,024. Os cinco restantes divergem por decisões que o próprio
relatório lista no apêndice A.2 e que não estão no artigo nem no nosso port:

| data set | \|desvio\| médio | causa (apêndice A.2 do relatório) |
|---|---|---|
| DS12 | 0,119 | item 3: `complete.cases` após o embed em vez de knnImputation (o artigo usa knnImputation; com ela o % raro bate com a Tabela 1). Lá lm, rf e mars sem resampling dão 0,00; aqui 0,43 a 0,48 |
| DS20 | 0,126 | o relatório tem 0,000 em lm, lm_UNDERB, lm_UNDERTPhi e svm e valores muito baixos em svm/rpart com resampling; é o "defeito em aberto" que ele mesmo registra (precisão do LM ≈ 0 em 12 dos 24 data sets). Aqui a linha LM da Tabela 3 bate com o artigo |
| DS21, DS22, DS24 | 0,070, 0,121, 0,069 | item 6: teto de ~8.000 linhas de treino após o resampling em DS21 a 24 (o artigo usa 10 %/5 % e 20 %/10 % da série inteira, ~24.000 linhas em DS21). As famílias inteiras deslocam: rpart com OVER/SMOTE em DS21 (0,89 vs 0,56), mars em DS22, lm/mars em DS24 |

DS23 tem 6 repetições (10 no UNDERTPhi) em que a janela de treino não tem casos raros: os
resamplers param com "All the points have relevance 0", o mesmo `stop()` do código R original,
e a iteração fica NaN (a média usa as 44 válidas). O relatório trocou esse `stop()` por um
no-op (item 5), por isso lá são 50 repetições.

### 9.3 F1 médio sobre os 24 data sets (nosso run)

| | baseline | U_B | U_T | U_TPhi | O_B | O_T | O_TPhi | SM_B | SM_T | SM_TPhi |
|---|---|---|---|---|---|---|---|---|---|---|
| lm | 0,254 | 0,489 | 0,492 | 0,494 | 0,492 | 0,488 | 0,489 | 0,503 | 0,490 | 0,487 |
| svm | 0,447 | 0,479 | 0,479 | 0,479 | 0,484 | 0,483 | 0,483 | 0,476 | 0,469 | 0,464 |
| mars | 0,357 | 0,445 | 0,451 | 0,445 | 0,486 | 0,487 | 0,488 | 0,489 | 0,482 | 0,479 |
| rf | 0,274 | 0,412 | 0,410 | 0,414 | 0,371 | 0,360 | 0,354 | 0,496 | 0,482 | 0,483 |
| rpart | 0,445 | 0,478 | 0,477 | 0,477 | 0,475 | 0,477 | 0,475 | 0,480 | 0,465 | 0,468 |

ARIMA 0,240; BDES 0,116. (O artigo não publica esta média; a Fig. 7 traz os valores por data set.)

### 9.4 Fig. 7 do artigo (F1 por data set) lida por completo: nosso x artigo

A Fig. 7 traz o F1 dos 52 workflows em cada um dos 24 data sets. O leitor
`tools/compare_fig7_all.py` lê a figura renderizada do PDF (eixo x calibrado nas linhas de
grade de DS1 e DS24, linhas tracejadas interpoladas nos vértices) e recupera 1232 dos 1248
pontos (99 %); precisão da leitura por volta de ±0,03. Arquivos:
`results_cluster/fig7_article_readings.csv`, `desvios_python_cluster_vs_artigo_fig7.csv`,
`comparison_fig7.txt`.

**Resultado geral:** |desvio| médio 0,027, mediano 0,013; 84 % das células a 0,05 ou menos e
93 % a 0,10 ou menos; viés (nosso menos artigo) de -0,003. Por modelo:

| modelo | células | \|desvio\| médio | viés | a ≤ 0,05 |
|---|---|---|---|---|
| svm | 239 | 0,014 | -0,004 | 96 % |
| rpart | 240 | 0,018 | -0,007 | 92 % |
| rf | 240 | 0,019 | -0,006 | 93 % |
| mars | 238 | 0,024 | +0,002 | 90 % |
| lm | 230 | 0,050 | -0,002 | 65 % |
| BDES | 24 | 0,081 | -0,002 | 54 % |
| ARIMA | 21 | 0,109 | +0,007 | 33 % |

F1 médio sobre os data sets, nosso | artigo (todos os 50 pares modelo x estratégia ficam a
0,02 ou menos um do outro):

| | base | U_B | U_T | U_TPhi | O_B | O_T | O_TPhi | SM_B | SM_T | SM_TPhi |
|---|---|---|---|---|---|---|---|---|---|---|
| lm | 0,26 \| 0,26 | 0,49 \| 0,49 | 0,49 \| 0,50 | 0,49 \| 0,50 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,50 \| 0,50 | 0,49 \| 0,49 | 0,49 \| 0,49 |
| svm | 0,45 \| 0,46 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,49 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,48 | 0,47 \| 0,47 | 0,46 \| 0,46 |
| mars | 0,36 \| 0,36 | 0,44 \| 0,42 | 0,45 \| 0,43 | 0,45 \| 0,44 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,49 \| 0,49 | 0,48 \| 0,49 | 0,48 \| 0,48 |
| rf | 0,27 \| 0,29 | 0,41 \| 0,42 | 0,41 \| 0,42 | 0,41 \| 0,42 | 0,37 \| 0,38 | 0,36 \| 0,36 | 0,35 \| 0,36 | 0,50 \| 0,50 | 0,48 \| 0,48 | 0,48 \| 0,49 |
| rpart | 0,44 \| 0,46 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,49 | 0,47 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,48 | 0,48 \| 0,49 | 0,46 \| 0,47 | 0,47 \| 0,48 |

ARIMA 0,275 | 0,268; BDES 0,116 | 0,118.

Por data set, o |desvio| médio fica entre 0,012 (DS1) e 0,060 (DS10); nenhum data set foge
do padrão. Os pontos isolados que divergem mais são baselines em DS4, DS5, DS7 e DS10, e o
código R original rodado localmente (seção 6) dá o mesmo que o Python neles, ou seja, é o
código publicado que não reproduz a figura nesses pontos, não o port:

| célula | Python | R original (local) | artigo (Fig. 7) |
|---|---|---|---|
| DS4 BDES | 0,627 | 0,633 | 0,194 |
| DS4 lm | 0,577 | 0,589 | 0,244 |
| DS4 ARIMA | 0,568 | 0,581 | 0,279 |
| DS10 lm_SMOTEB | 0,666 | 0,666 | 0,434 |
| DS10 lm_UNDERB | 0,647 | 0,645 | 0,430 |
| DS10 lm | 0,000 | 0,000 | 0,114 |
| DS10 mars | 0,490 | 0,219 | 0,163 |

A única exceção era o MARS sem resampling em DS10, em que o MARS próprio (numpy) ficava em 0,49
contra 0,22 do `earth` e 0,16 do artigo. **Corrigido em 22/09 (seção 10):** o `mars.py` é agora um
port do `earth` e essa célula passa a 0,16; as tabelas desta seção foram recalculadas (10.6).

## 10. Auditoria de 22/09/2026: o MARS era a única divergência sistemática, e foi corrigida

Pergunta: o que ainda separava o port do código R original, além do sorteio das janelas?
Método: para cada learner, o R original e o Python foram ajustados **nos mesmos conjuntos
de treino e teste** (janelas Monte Carlo e conjuntos reamostrados gerados pelo port e
gravados em CSV), de modo que a diferença medida é só de implementação. Scripts e fixtures:
`tests/test_mars.py`, `tests/data/earth/`, `tools/compare_mars_v2.py`.

### 10.1 O que estava errado: o MARS próprio não era o `earth`

O `tsresamp/mars.py` anterior era um MARS "de livro" (nós nos quantis de cada variável,
até 30 por variável; termo linear sempre candidato; parada por `nk`/`thresh`; poda gulosa por
GCV). O `earth` faz outra coisa em quatro pontos que mudam o modelo:

1. os nós candidatos são **casos** na ordem da variável, nunca os `endspan` casos de cada
   extremo (endspan = 3 + log2(20) + log2(p) = 10 com 9 preditores, 30 em termos de
   interação) e só de `minspan` em `minspan` casos com termo pai não nulo
   (minspan = (2,97 + ln(p·n_pai))/1,73, 6 a 8 nas nossas janelas);
2. o termo linear `pai × x` é sempre avaliado primeiro e um nó só o substitui se reduzir mais
   o RSS; mas o candidato linear é **descartado quando a sua soma de quadrados após a
   ortogonalização é ≤ 0,01** (`MIN_BX_SOS`, um limiar absoluto): em séries de valores
   pequenos (DS15, DS16, DS19: valores ~0,01) isso elimina quase todos os termos lineares e
   metade dos pares de hinges, e o modelo fica muito mais pobre do que o nosso;
3. um par de hinges é rejeitado quando a sua coluna é quase colinear com o modelo (tolerância
   0,01 nos primeiros 15 termos) ou quando reduz o RSS mais do que
   min(1,01·RSS, 10·redução anterior); um termo linear consome dois slots de `nk`; a fila do
   Fast MARS define a ordem dos pais;
4. a poda usa a rotina `BAKWRD` do pacote `leaps`, que além do caminho guloso de eliminação
   regressiva registra os prefixos intermediários de cada movimentação e guarda, para cada
   tamanho, o melhor subconjunto visto (na janela 33 do DS15 isso escolhe 6 termos, a
   eliminação gulosa pura escolhia 4).

O `mars.py` foi reescrito como port do `earth.c` (5.3.6) e do `earth.fit.R`/`leaps`.

### 10.2 Verificação: mesmo modelo que o `earth` nos mesmos dados

| verificação | resultado |
|---|---|
| 74 conjuntos de treino/teste de 10 data sets (1, 4, 5, 9, 10, 15, 16, 19, 21, 23), com e sem reamostragem (UNDER B/TPhi, OVER B, SMOTE B), parâmetros da Tabela 7 | os mesmos termos, os mesmos nós, os mesmos coeficientes; predições iguais a 1e-14 (1e-11 em DS4/DS21, cujos valores são milhares) |
| as 50 janelas Monte Carlo do port em DS10 e em DS1, `mc.mars` | F1 igual em cada iteração (diferença 1e-15): média 0,1624 = 0,1624 (DS10) e 0,2308 = 0,2308 (DS1) |
| casos difíceis guardados como fixtures (`tests/data/earth/`): série pequena (DS16), série inteira sob-amostrada com termos lineares e hinges soltos (DS10 UNDERTPhi), poda não gulosa (DS15), `thresh` = 0,01 (DS19) | 4 testes de regressão no `pytest` |

### 10.3 Efeito nos workflows MARS (50 repetições; F1 médio)

| | MARS antigo | **MARS = earth** | R original | relatório R |
|---|---|---|---|---|
| DS10 `mc.mars` | 0,490 | **0,162** | 0,219 | 0,195 |
| DS10 UNDER B / T / TPhi | 0,480 / 0,458 / 0,621 | 0,515 / 0,466 / 0,628 | 0,492 / 0,485 / 0,626 | 0,484 / 0,524 / 0,623 |
| DS15 UNDER B / T / TPhi | 0,279 / 0,292 / 0,267 | **0,151 / 0,171 / 0,170** | - | 0,143 / 0,121 / 0,138 |
| DS16 UNDER B / T / TPhi | 0,301 / 0,272 / 0,258 | **0,089 / 0,132 / 0,135** | - | 0,085 / 0,094 / 0,092 |
| DS19 UNDER B / T / TPhi | 0,275 / 0,271 / 0,274 | **0,126 / 0,154 / 0,136** | - | 0,098 / 0,169 / 0,158 |
| DS1, 10 workflows: média de \|Python - R original\| | 0,009 | 0,015 | | |
| DS4, idem | 0,008 | 0,005 | | |
| DS10, idem | 0,033 (máx 0,271) | 0,012 (máx 0,057) | | |
| DS15 / DS16 / DS19, média de \|Python - relatório\| | 0,055 / 0,067 / 0,049 (máx 0,22) | 0,018 / 0,017 / 0,017 (máx 0,05) | | |

O que resta (por exemplo 0,162 contra 0,219 na baseline do DS10) é o sorteio das janelas:
o F1 dessa baseline por janela tem desvio padrão 0,25, logo a média de 50 janelas tem erro
padrão 0,036, e nas janelas do port o `earth` dá exatamente 0,162.

Nos 19 data sets comparáveis com o relatório independente (todos menos 12, 20, 21, 22, 24),
o \|desvio\| médio das 190 células MARS cai de 0,022 para 0,011, as células a mais de 0,05
caem de 20 para 5, e o viés das famílias UNDER (antes +0,02 a +0,03) desaparece (≤ 0,004).
Tabelas 3, 4, 5 e Fig. 7 recalculadas com o MARS novo: seção 10.6.

### 10.4 Os outros learners, no mesmo teste pareado

| learner | conjuntos | F1 médio R | F1 médio Python | diferença (erro padrão) |
|---|---|---|---|---|
| `rpart` | DS1, 50 janelas | 0,433 | 0,432 | -0,001 (0,007) |
| `rpart` | DS4, 50 janelas | 0,533 | 0,533 | 0 (predições idênticas nas 50) |
| `rpart` | DS10, 50 janelas | 0,548 | 0,546 | -0,002 (0,003) |
| `randomForest` | DS1, 50 janelas | 0,006 | 0,006 | 0 |
| `randomForest` | DS2 OVER B, 50 conjuntos | 0,200 | 0,201 | +0,001 (0,009) |
| `randomForest` | DS3 OVER B, 50 conjuntos | 0,243 | 0,249 | +0,007 (0,017) |
| `auto.arima` | DS4, 50 janelas | 0,572 | 0,568 | -0,004 (0,003) |
| `auto.arima` | DS10 / DS1, 50 janelas | 0,000 / 0,009 | 0,000 / 0,000 | 0 / uma janela |

Nenhuma diferença sistemática. Dois ajustes saíram desta auditoria, ambos no mapeamento das
árvores, e ambos verificados pelo fonte do R e pelo teste pareado:

* **`cp` do `rpart` é poda por complexidade, não pré-poda.** Em `rpart/src/partition.c` a
  complexidade de cada nó é (SS do nó - SS das folhas da subárvore) / número de divisões da
  subárvore, calculada de baixo para cima colapsando primeiro os filhos mais fracos, e a
  subárvore só fica se a complexidade excede `cp × SS_raiz`; `bsplit.c` não aplica nenhum
  limiar por divisão. Isso é a poda de custo-complexidade mínima, `ccp_alpha = cp·var(y)` no
  scikit-learn, e não o `min_impurity_decrease` que a documentação do `rpart` sugere e que a
  revisão anterior tinha adotado. Com `ccp` o número de folhas coincide com o do R em 28/50
  janelas do DS1 (12/50 com pré-poda) e 36/50 do DS10 (24/50); a diferença de F1 no DS1 vai
  de -0,010 para +0,002. O BDES usa as mesmas árvores; efeito medido em DS4: +0,001.
* **Empates a 1e-7 no scikit-learn.** O `DecisionTreeRegressor` (e o `RandomForestRegressor`)
  nunca divide entre dois valores a menos de 1e-7 um do outro (constante absoluta
  `FEATURE_THRESHOLD`); `rpart` e `randomForest` dividem entre quaisquer dois doubles
  distintos. O DS1 tem 30 pares por coluna a 1 ulp de distância (0,0633339999999999 e
  0,063334) e o `rpart` divide entre eles. Os preditores passam agora por uma transformação
  monótona para postos (`TreeRanks`: posto entre os valores distintos de treino; valores
  novos por interpolação linear, o que mantém cada valor do mesmo lado do ponto médio do R).
  Com isso o DS1 passa a 38/50 janelas com o mesmo número de folhas e a média de F1 fica a
  0,001 do R; no `randomForest` do DS1 as duas implementações passam a dar o mesmo F1 em
  todas as janelas. O que resta no `rpart` (DS10, série de inteiros) são empates exatos do
  critério de divisão entre variáveis, que o `rpart` desempata pela ordem das colunas e o
  scikit-learn por uma ordem aleatória de variáveis, sem opção de mudar.

O `auto.arima` do `forecast` e o do `pmdarima` escolhem ordens diferentes em 40 a 70 % das
janelas (por exemplo ARIMA(0,0,2) contra (1,0,1)), mas as predições um passo à frente ficam
com correlação 0,999 e o F1 não muda; não há ajuste a fazer.

### 10.5 O que não muda

`lm` e `svm` já eram exatos (seções 6 e 7). Relevância, utilidade, reamostragem, Monte Carlo e
testes estatísticos continuam como traduções verificadas. As conclusões da seção 6 sobre o
artigo (o código publicado não reproduz várias células do artigo) não dependem do MARS.

### 10.6 Tabelas do run completo recalculadas

Workflows MARS, RPART, BDES e RF (24 data sets) reexecutados com o código atual
nas mesmas janelas; o resto vem do run do cluster (`results_cluster/PROVENANCE.md`).

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
