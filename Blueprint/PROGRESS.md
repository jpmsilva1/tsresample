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
| 1 | Scaffold | 1 | P5 | Not started | |
| 2a | φ (`_relevance.py`, gate G0) | 2 | 1 | Not started | |
| 2b | Embedding (`embed.py`) | 2 | 1 | Not started | |
| 3 | Bumps (`_bins.py`) | 3 | 2a | Not started | |
| 4 | Preferences (`_prefs.py`) | 4 | 3 | Not started | |
| 5 | Sampling (`_sample.py`) | 5 | 4 | Not started | |
| 6 | Synthesis (`_synth.py`) | 6 | 5 | Not started | |
| 7 | Resampler (`resampler.py`, gate G2) | 7 | 6 | Not started | |
| 8 | Metrics (`metrics.py`, `_utility.py`, gate G0b) | 8 | M0, 7 | Not started | |
| 9a | Pipeline I/O | 8 | 7 | Not started | |
| 9b | Pipeline API | 9 | 8, 9a | Not started | |
| 10 | CLI + `MIGRATION.md` | 10 | 9b | Not started | |
| 11 | Replication (gate G4 + `replication_report.md`) | 10 | 8, 9b | Not started | |
| 12 | Docs | 11 | 10 | Not started | |
| 13 | Release | 12 | 0, 11, 12 | Not started | |

## Open items

Known issues that don't block any step. Close one by stating the resolution and date here.

| Item | Detail | Where |
|---|---|---|
| DS19 φ = (1, 0, 0) utility residual | 312 splits don't match R (worst 0.83). Excluded from `metric_oracle.json`; the metric docstrings must say so. | ADR-0014 "Residual" |
| DS05 φ control points | 5/90 splits miss on last-bit precision at the ±0.08 fences; needs a full-precision re-export from R. Excluded from G0 Test A. | `Blueprint/docs/REPLICATION.md` §2 |
| DS13 imputation | Lag-window kNN gives 9.19–10.67 % rare vs the paper's 11.1 %. G4 excludes DS13 from strict checks. | ADR-0010 amendment; `Blueprint/docs/REPLICATION.md` §4.2b |
| `python_port/` vendored | The GPL-derived Python replication now lives in `python_port/` (quarantined). Step 1 must keep it out of the library's build, test collection (`testpaths`), coverage, lint and type checks. | `python_port/QUARANTINE.md`; PROVENANCE §1 |
| `[AUDIT]`-tagged SPEC items | Bins, counts, bias, SMOTE, replacement. Each is confirmed by its node's G3/G4 check. | ADR-0012, 0013; amended 0004/0006/0007 |

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
