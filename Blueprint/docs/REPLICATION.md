# Replication procedures

How we prove `tsresample` reproduces Moniz, Branco & Torgo (2017) — and what we do if it
doesn't.

The premise of this document is that **we already own the ground truth.** A prior project
re-ran the paper's experiment in R and kept everything: fitted φ control points, per-split
predictions, and the resulting metrics for 24 datasets × 52 workflows × 50 Monte Carlo
iterations (G0's fixture covers DS01–DS20). That turns replication from an argument into a numerical comparison.

---

## 1. The oracle

**Root:** a clone of the public replication repository, pinned to the commit the oracles
were verified against:

```bash
git clone https://github.com/jpmsilva1/ts-resampling-replication.git
git -C ts-resampling-replication checkout b1c7e57d48200212841ac1bb6b054f6fbda77c8b
export TSRESAMPLE_REPLICATION_ROOT="$PWD/ts-resampling-replication"
```

Gate G4 skips cleanly if `TSRESAMPLE_REPLICATION_ROOT` is unset. G0 and G0b never need it:
their fixtures are committed in `Blueprint/tests/fixtures/`. The clone is large (about 4 GB of
per-split predictions). Its `scratch/uba/` is GPL and quarantined (`Blueprint/docs/PROVENANCE.md` §1).

| Path (under the root) | Contents | Used by |
|---|---|---|
| `data/paper_datasets_csv/DS01..DS24_*.csv` | `time_index,target`; the paper's 24 series (identical copies in `Blueprint/replication/datasets/`) | G0, G4 |
| `Results (Clean)/Results Data/raw_predictions_by_dataset/DS*/mc.*_phi_ctrl.csv` | `iteration,point_index,ctrl_x,ctrl_phi,ctrl_deriv` — **fitted φ control points, per split** | G0 |
| `…/raw_predictions_by_dataset/DS*/mc.*_predictions.csv` | `iteration,obs_index,y_true,y_pred` | G0b, G4 |
| `…/raw_iterations_by_dataset_v2/DS*.csv` | `dataset_id,dataset_name,workflow,iteration,prec,rec,F1` | G0b, G4 |
| `…/raw_sera_by_dataset/DS*.csv` | `dataset_id,workflow,model_family,strategy,iteration,sera` | G0b, G4 |
| `…/raw_rmse_by_dataset/` | RMSE per split | context only |
| `…/original_paper_reference_results/` | figures/tables transcribed from the paper | G4 sanity |

Workflow labels are `mc.<model>_<STRATEGY>` with `STRATEGY ∈ {baseline, UNDERB, UNDERT,
UNDERTPhi, OVERB, OVERT, OVERTPhi, SMOTEB, SMOTET, SMOTETPhi}` — the SPEC §4.6 grid plus
the unresampled baseline. Models are `lm`, `svm`, `mars`, `rpart`, `rf`, `BDES`.

**None of these files is GPL.** They are numerical output produced by our own experiment
runs. Reading them is not reading the R implementation. See `Blueprint/docs/PROVENANCE.md` — that
distinction is what keeps the MIT licence defensible, and it is the reason this whole
approach works.

---

## 2. Gate G0 — the φ oracle (committed, offline, runs in CI)

**Fixture:** `Blueprint/tests/fixtures/phi_oracle.json` (~29 KB, rebuilt 2026-09-22). Per dataset:
the paper's Table 1 `%Rare`, a truncated SHA-256 of the target column, R's iteration-1 and
median control points, and **`r_splits`**. `r_splits` holds R's recorded control points
(`ctrl_x`, `ctrl_phi`) for workflow `mc.lm_OVERB`, iterations 1–5, each with the raw-series
slice `[a, b)` of that split's training targets. The slice was reconstructed by matching
the recorded test targets to the series. DS12/DS13 (NAs) have no `r_splits`. The full
50-iteration R control points for every dataset are in
`Blueprint/replication/oracles/phi_control_points/` (identical across workflows).

v0.7.0's `full_series_control_x` was **removed**. It had been generated from the SPEC's own
fence formula (it matched that formula 20/20 and R 5/20), so its "Test A" could not fail.
That is exactly the tautology CLAUDE.md rule 6 forbids (SPEC §0 #13).

**Test A — per-split control points (the exact test).** For every `r_splits` entry, slice
`Blueprint/replication/datasets/DS<nn>_*.csv` `target[a:b]`, compute control points per SPEC §4.1,
and compare `(x, φ)` to the recorded values within `1e-6`. This checks the hinges, the
whisker rule and the one-sided φ = 0 rule directly against R.
- Current result: **85/90 exact**.
- **DS05 is excluded** and tracked as an open item. Its training values sit exactly on the
  ±0.08 fences, and whether they count as outliers depends on last-bit precision that the
  CSV export lost (21 of 22 "−0.08" values are exact in the CSV, one is `−0.0799…`). Close
  it by re-exporting DS05 at full precision from the R session, never by loosening the
  tolerance.

**Test B — `%Rare` against the paper.** For each NA-free dataset, fit φ per §4.1 on the
**embedded** target (`series[9:]`, the `create.data(ts, 10)` target) and count `φ ≥ 0.9`
(identical to `> 0.9` on all 18 datasets; ADR-0010 amendment).
Compare to `paper_pct_rare`. **Tolerance: MAE ≤ 0.25 pp over the 18 NA-free datasets, and
≤ 2.5 pp on any single one** (`Blueprint/ops/QUALITY_GATES.md` §1).

| Convention | MAE | worst | Test B |
|---|---|---|---|
| v0.8.0 §4.1 (hinges, whiskers, Hermite) | **0.169 pp** | 2.44 (DS05) | pass |
| v0.7.0 fences, same Hermite | 0.756 pp | 4.19 | fail |
| v0.8.0 control points + `PchipInterpolator` | 3.109 pp | 5.34 | fail |

Test B is the gate that can catch a *wrong formula*, and it fails on both wrong
conventions above. Node 2a must demonstrate that.

**Test C — one-sided φ.** DS10 has φ = 0 at one endpoint in every recorded split. Assert
that the fixture's DS10 splits reproduce it, and that φ is constant beyond that endpoint.

---

## 3. Task M0 → Gate G0b — the metric oracle

**M0 is a research task, not an implementation task.** It blocks `_utility.py`. (The SERA
constant is settled, §3.2.) It follows exactly the procedure that pinned φ: form a hypothesis,
compute it from `y_true`/`y_pred`, compare against the recorded value, and iterate on the
*convention* rather than fitting a fudge factor.

### 3.1 Inputs

Pick one dataset/workflow with a stable φ as the working case — **DS05 / `mc.lm_OVERB`**
is recommended, because its oracle control points are constant across all 50 iterations at
`(−0.06, 0, 0.06)`, which removes the split-identification problem entirely.

```
predictions : raw_predictions_by_dataset/DS05/mc.lm_OVERB_predictions.csv
sera        : raw_sera_by_dataset/DS05.csv          (workflow == "mc.lm_OVERB")
prec/rec/F1 : raw_iterations_by_dataset_v2/DS05*.csv (workflow == "mc.lm_OVERB")
phi ctrl    : raw_predictions_by_dataset/DS05/mc.lm_OVERB_phi_ctrl.csv
```

### 3.2 SERA — resolved (2026-09-22)

History: the first probe compared against `Blueprint/replication/oracles/metrics/sera_pchip/`, a file
`Blueprint/replication/MANIFEST.csv` flags `CONTAMINATED`. **Always check a candidate oracle file's
provenance column first.** Against the real `raw_sera_by_dataset/`, `step = 0.001` left a
0.05–0.15 % gap that varied by dataset. The cause: the recorded values come from the harness script
`Results (Clean)/Eval_Metrics Code/SERA Metric/sera_metric.py`, not from R. It uses
`step = 0.01` and forces φ = 1 at and beyond **both** endpoints, including a one-sided
φ = 0 endpoint. Recomputing with those settings matches every recorded SERA to **1e-15**
(45 cases: DS01, DS05, DS09, DS10, DS21 × {`mc.lm`, `mc.lm_OVERB`, `mc.rf_SMOTET`} × 3
iterations).

It was not seed noise (the deterministic `lm` baseline had the same gap), not grid
construction, and not φ extrapolation. Exact-breakpoint integration is wrong regardless
(~22 % off). The library keeps `step = 0.001` (SPEC §4.7); G0b tests `step=0.01` on
two-sided splits (§3.4).

### 3.3 Then the utility surface

`precφ`/`recφ` need `u_i = U(ŷ_i, y_i)`. As of 2026-09-22 the problem is known to be
exactly solvable: a reference implementation of R's conventions reproduced the recorded
prec/rec/F1 to **1e-15** on 45 cases (ADR-0009 amendment). The metric-level conventions are
pinned in SPEC §4.7: `≥ t_E`, the `1e-5` empty-selection value, `|1+u|`, `p = 0.5`.

What M0 still has to do cleanly is derive `U` itself from **Ribeiro (2011) §4**. That is the
benefit/cost surface over φ's bumps, with the loss tolerances it defines. Verify it against
`raw_iterations_by_dataset_v2/`. That reference implementation is GPL-derived and
quarantined (PROVENANCE §1). Do not seek it out; the thesis and the oracle are sufficient.

**Resolved (2026-09-22): ADR-0014.** `U` was derived from the thesis §3.3–3.4 and matches
within 1e-6 on 62,088/62,400 recorded splits. The residual is DS19's φ = (1,0,0) splits (312),
which are documented and excluded from G0b. Probe: `Blueprint/replication/probes/m0_utility.py`.

Run the free check first: `F1 = 2·prec·rec/(prec+rec)` must hold on the recorded triples,
and it does whenever neither measure is the `1e-5` floor. Recorded rows with
`prec = 1e-05` pin the empty-selection convention and make good first test cases.

### 3.4 Acceptance

M0 is done when, for at least **three** datasets covering different `%Rare` levels
(suggested: DS05 at 3.5 %, DS01 at 9.9 %, DS09 at 21.1 %) and at least two model families:

- `sera(..., step=0.01)` matches recorded within **1e-6 relative** on every split whose
  recorded control points have φ = 1 at both endpoints (one-sided splits are excluded:
  the harness forced φ = 1 there, §3.2), and
- `precision_phi`, `recall_phi`, `f1_phi` match recorded within **1e-6 absolute**.

**Result (2026-09-22): both met** (ADR-0014). The fixture is frozen: 8 cases, 49.4 KB, from DS10,
DS01, DS04 and DS09; DS05 passes in the probe but is too large to embed. DS19 φ = (1,0,0) is the
documented residual.

Then, and only then, freeze `Blueprint/tests/fixtures/metric_oracle.json` with a sample of
`(y_true, y_pred, φ control points) → (sera, prec, rec, F1)` tuples and wire gate G0b.
Keep it under ~50 KB by sampling iterations, not by rounding values.

**If M0 cannot reach the prec/rec/F1 tolerance**, the metrics still ship — but their
docstrings say plainly which convention they implement, and `Blueprint/docs/REPLICATION.md` records
the residual. A metric that is
*documented as different* is usable. A metric that is *silently different* is a trap.

---

## 4. Gate G4 — end-to-end replication

Opt-in (`pytest -m replication`), nightly and on release tags. Needs
`TSRESAMPLE_REPLICATION_ROOT`.

### 4.1 Procedure, per dataset

1. Load `DS<nn>_*.csv`; take the `target` column in time order.
2. Embed with `embed(series, k=9, horizon=1)` — the paper's `create.data(ts, 10)`
   (ADR-0005). Assert the row count matches the recorded experiment's `N`.
3. For each of 50 Monte Carlo temporal splits (`train_size=0.5, test_size=0.25`;
   DS21–22 `0.1/0.05`, DS23–24 `0.2/0.1`; contiguous, train before test), for each of the
   9 strategies plus baseline:
   a. `fit_resample` the training block. φ refits inside (ADR-0008).
   b. Fit the estimator; predict the test block.
   c. Score with `precision_phi`, `recall_phi`, `f1_phi`, `sera`, φ fit on the training
      target.
4. Compare the resulting 50-value distribution against the recorded one for the same
   `(dataset, workflow)`.

### 4.2 What is compared, and how

Exact per-iteration equality is **not** achievable and is not the target — the R run's
RNG stream, split boundaries and learner implementations are not reproducible from Python.
Requiring it would guarantee failure and teach us nothing.

What must hold, in order of strength:

| # | Assertion | Tolerance |
|---|---|---|
| R1 | **Ranking.** Per dataset, the Spearman correlation between our mean-`F1φ` ranking of the 10 workflows and the recorded ranking. | ρ ≥ 0.7, and ≥ 0.6 on every individual dataset |
| R2 | **Direction.** For every `(dataset, strategy)` where the recorded run shows resampling beating baseline on `F1φ`, ours does too. | ≥ 85 % agreement |
| R3 | **Location.** Per `(dataset, workflow)`, our median `F1φ` sits inside the recorded interquartile range. | ≥ 75 % of cells |
| R4 | **Row counts.** Resampled training-set sizes equal the SPEC §4.4 counts for the split's bumps (under shrinks to the rare total, over grows by the normal total, smote `|len − N| ≤ 2·#bumps`). | exact |
| R5 | **Paper-level conclusion.** The Wilcoxon signed-rank test over datasets reproduces the paper's headline: `SMOTE*` and `OVER*` significantly beat baseline on `F1φ`; `UNDER*` does not consistently. | sign and significance at α = 0.05 |

R4 is exact and cheap; run it first.

### 4.2b Harness configuration (ADR-0011: harness choices are not library behaviour)

G4 reads an oracle produced by a specific harness. Configure the comparison to match it,
and never change `src/` to match it:

| Canonical harness choice | Affects | G4 handling |
|---|---|---|
| `complete.cases` (paper and some ports impute with kNN) | DS12, DS13 | run these two in complete-cases mode, or exclude them from R1–R3 |
| SVM training cap: last 10,000 rows, OVER/SMOTE only | `svm` on DS05–08 (and DS21–24) | exclude from strict checks |
| `auto.arima(method="CSS")` | `arima` | out of scope; learner-only |
| learners (`e1071`, `earth`, `randomForest`, `rpart`) | all non-lm cells | **`lm` is the strict learner.** A correct independent implementation lands at ≈0.01 mean ΔF1 on lm with occasional 0.1–0.4 outliers on rf/mars (the maintainer's private audit report, not in this repo) |
| per-dataset split sizes (above) | DS21–24 | use the same sizes | R5 is the one that actually matters for the claim
"this library replicates the paper" — R1–R3 are the diagnostics that tell you *where* it
broke when R5 fails.

### 4.3 Reporting

G4 writes `replication_report.md`: the R1–R5 table, per-dataset ranking correlations, and
a scatter of our mean `F1φ` against recorded. It is committed on each release so the claim
is auditable rather than asserted.

---

## 5. If replication fails

Work the list in order. Each step is cheaper than the one after it.

1. **G0 red** → stop. φ is wrong, and everything downstream inherits it.
2. **R4 fails** → the §4.4 counts: check the per-strategy `"balance"` rule (ADR-0004
   amendment), truncation vs rounding (ADR-0007 amendment), and the bump classification
   rule for the strategy (`>` vs `≥`, ADR-0012).
3. **`lm` fails on `*B` for under/over** → bumps are wrong (value space, mean-φ
   classification, ADR-0012) before anything else.
4. **Fails on `*T` and `*TPhi` but not `*B`** → the preference vectors (SPEC §4.3). The
   likely culprit is using the **position in the full series** instead of the time rank
   **within the bump**. v0.7.0 specified the wrong one.
5. **Fails on `SMOTE*` only** → `r_quirks` must be `True` for G4. Then check the single λ,
   the last-predictor target weights, TPhi's recency `τ`, and the `nexs`/`extra` counts
   (ADR-0013).
6. **Fails uniformly across every strategy including baseline** → not a resampling bug.
   Suspect the embedding (ADR-0005), the split geometry, or a harness row from §4.2b.

**Kill criterion.** If R5 cannot be met after step 1 has been exhausted as an explanation,
the library ships as **"implements the strategies described in Moniz et al. (2017)"** and
not as **"replicates Moniz et al. (2017)"**. The README says so in its first paragraph,
`replication_report.md` ships with the failing numbers, and the discrepancy gets an ADR.

Shipping a library that silently disagrees with the paper it cites is the one outcome
worse than shipping nothing.
