# Progress

**This file is the single record of what is done and what is next.** Every session reads it
first and updates it last (rules at the bottom). Node definitions, tiers and briefs are in
`Blueprint/ops/TASK_GRAPH.md`; this file tracks status only.

Status values: `Not started` · `In progress (YYYY-MM-DD, @handle)` ·
`Done (YYYY-MM-DD, @handle)` · `Blocked: <reason>`. How to claim a step and open its PR:
`CONTRIBUTING.md`.

## Phase 0 — Planning and verification

| Step | What | Status | Evidence |
|---|---|---|---|
| P1 | SPEC v0.8.0 + ADR-0001–0013, corrected to R behaviour | Done (2026-09-22) | `Blueprint/docs/SPEC.md` §0 |
| P2 | φ oracle rebuilt from R's recorded per-split control points | Done (2026-09-22) | `Blueprint/tests/fixtures/phi_oracle.json`; 85/90 exact (`Blueprint/docs/REPLICATION.md` §2) |
| P3 | SERA recorded-value mechanism explained | Done (2026-09-22) | ADR-0009 amendment 2; `Blueprint/docs/REPLICATION.md` §3.2 |
| M0 | Clean derivation of the utility surface `U` | Done (2026-09-22) | ADR-0014; `Blueprint/tests/fixtures/metric_oracle.json`; 62,088/62,400 splits within 1e-6 |
| P4 | φ exact-endpoint + [0, 1] clip rule carried into SPEC §4.1 and the node 2a brief | Done (2026-09-22) | SPEC §4.1 Step 4 |
| P5 | Library repo created with `Blueprint/`, agent entry points and the collaboration workflow | Done (2026-09-22) | https://github.com/jpmsilva1/tsresample; `CONTRIBUTING.md` |

## Phase 1 — Build

Do the first row whose status is `Not started` and whose dependencies are all `Done`.
Rows in the same wave can run in parallel sessions.

| Step | Node | Wave | Depends on | Status | Evidence |
|---|---|---|---|---|---|
| 0 | **Setup (maintainer)**: PyPI Trusted Publishing configured for `tsresample`; branch protection on `main` (PRs required, code-owner review) | 0 | — | Not started | |
| 1 | Scaffold | 1 | P5 | Done (2026-09-24, @jpmsilva1) | local branch `node/1-scaffold`; gates L, T, A, C green (27 tests, 100 % cov); reviewer: merge after fixes, fixes applied |
| 2a | φ (`_relevance.py`, gate G0) | 2 | 1 | Done (2026-09-24, @jpmsilva1) | local branch `node/2a-phi`; G0 A 85/85, B MAE 0.169 pp (worst 2.44), C 5/5; PCHIP (3.109 pp) and v0.7.0 fences (0.756 pp) both fail B; inline self-review (not independent-context) |
| 2b | Embedding (`embed.py`) | 2 | 1 | Done (2026-09-24, @jpmsilva1) | local branch `node/2b-embed`; 39 tests, 100 % cov; ADR-0005 amended (create.data(ts, m) = embed(k=m-2)); inline self-review (not independent-context) |
| 3 | Bumps (`_bins.py`) | 3 | 2a | Done (2026-09-24, @jpmsilva1) | local branch `node/3-bumps`; G1 partition property + G3 worked examples (both rules, two- and one-sided, no-bump); 144 tests, 100 % cov; inline self-review |
| 4 | Preferences (`_prefs.py`) | 4 | 3 | Done (2026-09-24, @jpmsilva1) | local branch `node/4-prefs`; G3 worked examples for all three biases, zero-sum and too-few-positive fallbacks; new seam `_prefs.draw` recorded in QUALITY_GATES §2; 150 tests, 100 % cov; inline self-review |
| 5 | Sampling (`_sample.py`) | 5 | 4 | Done (2026-09-24, @jpmsilva1) | local branch `node/5-sampling`; G3 worked examples for all three strategies (balance + explicit o/u, R float-truncation and half-even cases) and G1 size properties; ADR-0015 (singleton-bump size property); new seam `_sample.resample`; 164 tests, 100 % cov; inline self-review |
| 6 | Synthesis (`_synth.py`) | 6 | 5 | Done (2026-09-24, @jpmsilva1) | local branch `node/6-synthesis`; G3 hand-worked neighbour choice for T and TPhi under both r_quirks, B with k=1, k_eff path, tie midpoint, extra seeds, determinism; mutations (quirk off, nearness tau, per-attribute lambda, no midpoint) all caught; ADR-0013 amendment (column a = X[:,0]); 172 tests, 100 % cov; inline self-review |
| 7 | Resampler (`resampler.py`, gate G2) | 7 | 6 | Done (2026-09-24, @jpmsilva1) | local branch `node/7-resampler`; G2 green (4 sklearn checks + clone/get_params round-trip); full 3x3 grid on DS01; ordering guarantee, no-bump passthrough, relevance forms, validation; 204 tests, 100 % cov; inline self-review |
| 8 | Metrics (`metrics.py`, `_utility.py`, gate G0b) | 8 | M0, 7 | Done (2026-09-24, @jpmsilva1) | local branch `node/8-metrics`; G0b green: prec/rec/F1 8/8 recorded cases within 1e-6 abs, SERA 7/7 within 1e-6 rel (step=0.01); spans %Rare 4.8-21.1 (DS05 3.5 % is probe-only, ADR-0014); ADR-0016 (control-point relevance); mutations caught except one equivalent mutant; 233 tests, 100 % cov; inline self-review |
| 9a | Pipeline I/O | 8 | 7 | Done (2026-09-24, @jpmsilva1) | local branch `node/9a-pipeline-io`; CSV->series->embed round-trip hand-checked; G3 DS12 kNN %Rare 10.99 (mean-imputation mutation fails it); contiguous non-shuffled Monte Carlo splits with the recorded sizes; ADR-0010 amendment (donor rows); no pandas in Layer 1 (gate A); 243 tests, 100 % cov; inline self-review |
| 9b | Pipeline API | 9 | 8, 9a | Done (2026-09-24, @jpmsilva1) | local branch `node/9b-pipeline-api`; tidy frame one row per (strategy, split, metric); phi refit per training window (test-phi mutation caught); imbalance_summary uses >= (ADR-0010) and matches Table 1 at MAE <= 0.25 pp on 18 datasets, no count shift vs imbalance_eval's rule; evaluate gained random_state (SPEC §2.4); 249 tests, 100 % cov; inline self-review |
| 10 | CLI + `MIGRATION.md` | 10 | 9b | Done (2026-09-24, @jpmsilva1) | local branch `node/10-cli`; CLI mirrors imbalance_eval flags + manifest; --k translates 1:1 (DS01 N=720); usage errors exit 2 with messages; docs/MIGRATION.md lists every behavioural difference; 255 tests, 100 % cov; inline self-review |
| 11 | Replication (gate G4 + `replication_report.md`) | 10 | 8, 9b | Done (2026-09-24, @jpmsilva1) | local branch `node/11-replication`; `replication_report.md`: R4 10800/10800 exact, R2 96.2 %, R3 95.0 %, R5 9/9 strategies match (paper conclusion reproduced), R1 mean 0.880 but DS19 0.467 < 0.6 (reported, open item); lm learner, r_quirks=True; found and fixed a mixed-date-format bug in load_series |
| 12 | Docs | 11 | 10 | Not started | |
| 13 | Release | 12 | 0, 11, 12 | Not started | |

## Open items

Known issues that don't block any step. Close one by stating the resolution and date here.

| Item | Detail | Where |
|---|---|---|
| DS19 φ = (1, 0, 0) utility residual | 312 splits don't match R (worst 0.83). Excluded from `metric_oracle.json`; the metric docstrings must say so. | ADR-0014 "Residual" |
| DS05 φ control points | 5/90 splits miss on last-bit precision at the ±0.08 fences; needs a full-precision re-export from R. Excluded from G0 Test A. | `Blueprint/docs/REPLICATION.md` §2 |
| DS13 imputation | Lag-window kNN gives 9.19–10.67 % rare vs the paper's 11.1 %. G4 excludes DS13 from strict checks. | ADR-0010 amendment; `Blueprint/docs/REPLICATION.md` §4.2b |
| `[AUDIT]`-tagged SPEC items | Bins, counts, bias, SMOTE, replacement. Each is confirmed by its node's G3/G4 check. | ADR-0012, 0013; amended 0004/0006/0007 |
| Gate C command hardcodes the floor | `Blueprint/ops/QUALITY_GATES.md` §3 writes `--cov-fail-under=90`, contradicting §1 ("stated once", read from `pyproject.toml`). CI reads `fail_under` from `pyproject.toml`; the §3 text should drop the flag. Found in step 1 review. | `Blueprint/ops/QUALITY_GATES.md` §3 |
| G4 R1 per-dataset floor fails on DS19 | ρ = 0.467 < 0.6 (mean over datasets 0.880 passes). DS19's nine strategy means span 0.028 against a 0.011 standard error, and ρ moves 0.19-0.56 with the split seed alone; R2-R5 hold. DS19 also carries the ADR-0014 φ = (1, 0, 0) residual. Not fixed by changing tolerance; closing it needs either the ADR-0014 residual resolved or an ADR on noise-limited rankings. Found in step 11. | `replication_report.md`; `Blueprint/docs/REPLICATION.md` §4.2 |

## Rules

1. **Read this file before anything else.** Pick your step from it, not from memory or from
   an older handoff note.
2. **When you start a step,** claim it: set it to `In progress (date, @handle)` in the
   first commit of your `step/` branch and open a draft PR at once (`CONTRIBUTING.md`).
   If an open draft PR already claims the step, pick another.
3. **When you finish a step,** set it to `Done (date, @handle)` and fill **Evidence** (PR link, gate
   output location, or ADR) in the same PR as the work. A step is finished only when
   `Blueprint/ops/QUALITY_GATES.md` §4 holds, including the independent `reviewer` pass.
4. **If you cannot finish,** set `Blocked: <reason>` and say what would unblock it.
5. **New open items** found during a step go in the Open items table. Don't fix unrelated
   items inside a step's PR.
6. Dates are the day the step's PR merged (or the day the work was verified, for Phase 0),
   in `YYYY-MM-DD`.
