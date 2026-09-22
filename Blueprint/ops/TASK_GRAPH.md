# Task graph

One node = one agent session (CLAUDE.md rule 7). Claim a node before starting it
(`CONTRIBUTING.md`); its status lives in `Blueprint/PROGRESS.md`. Nodes on the same row of the dependency
list run in parallel.

**Model routing below names Claude models.** With another agent tool, use its strongest
model wherever this table says Opus, and a fast model wherever it says Sonnet. No cross-family review — the adversarial split is by
*role and context*, not by vendor. A reviewer agent that never saw the implementer's
reasoning and works only from the spec and the gates is the load-bearing property; running
it on a different model family was a proxy for that, not the thing itself.

| Role | Model | Effort | Why |
|---|---|---|---|
| Implementation, AUTO nodes | Sonnet 5 | medium | Pinned by a gate; the spec does the thinking |
| Implementation, GATED nodes | Opus 5 | high | Synthesis, ambiguity, licence questions |
| Research probes (M0, any new ADR) | Opus 5 | xhigh | This is where wrong answers are expensive |
| Adversarial review | Opus 5 | high | Must be able to out-think the implementer |
| Replication runs | Sonnet 5 | medium | Mechanical; long-running |

---

## Nodes

Status and completion dates are tracked only in `Blueprint/PROGRESS.md`, not here.

| # | Node | Deliverable | Tier | Agent | Depends on |
|---|---|---|---|---|---|
| **1** | Scaffold | `pyproject.toml`, `src/` layout, CI workflow, `ruff`/`mypy` config, empty `py.typed` | AUTO | `implementer` | — |
| **2a** | φ | `_relevance.py` + gate G0 | AUTO | `implementer` | 1 |
| **2b** | Embedding | `embed.py` | AUTO | `implementer` | 1 |
| **3** | Bumps | `_bins.py` (value-space bumps, ADR-0012) | AUTO | `implementer` | 2a |
| **4** | Preferences | `_prefs.py` | AUTO | `implementer` | 3 |
| **5** | Sampling | `_sample.py` — target counts, under/over draws | GATED | `implementer` | 4 |
| **6** | Synthesis | `_synth.py` — Algorithms 4/9/13 | GATED | `implementer` | 5 |
| **7** | Resampler | `resampler.py` — dispatch + sklearn protocol, gate G2 | GATED | `implementer` | 6 |
| **M0** | **Metric oracle probe** | Clean derivation of `U`; `metric_oracle.json`; ADR update | GATED | `researcher` | 2a |
| **8** | Metrics | `metrics.py`, `_utility.py` + gate G0b | GATED | `implementer` | M0, 7 |
| **9a** | Pipeline I/O | `pipeline/io.py`, `pipeline/splits.py` | AUTO | `implementer` | 7 |
| **9b** | Pipeline API | `pipeline/__init__.py` — `evaluate`, `imbalance_summary` | AUTO | `implementer` | 8, 9a |
| **10** | CLI | `pipeline/cli.py` + `MIGRATION.md` | AUTO | `implementer` | 9b |
| **11** | Replication | Gate G4 suite + `replication_report.md` | GATED | `replicator` | 8, 9b |
| **12** | Docs | README, API docs, examples, `CITATION.cff` | AUTO | `implementer` | 10 |
| **13** | Release | Trusted Publishing, tag, PyPI | GATED | `implementer` | 11, 12 |

Every node is followed by a `reviewer` pass before merge. That pass is not optional and is
not a node — it is part of the definition of done (`Blueprint/ops/QUALITY_GATES.md` §4).

### Parallel waves

```
wave 1:  1
wave 2:  2a  2b
wave 3:  3   M0            ← start M0 early; it is the long pole
wave 4:  4
wave 5:  5
wave 6:  6
wave 7:  7
wave 8:  8   9a
wave 9:  9b
wave 10: 10  11
wave 11: 12
wave 12: 13
```

**Start M0 in wave 3, not wave 8.** It is research with an unknown completion time and it
gates the entire metrics half of the library. Everything else on the critical path is
implementation against a written spec. If M0 is left until its dependents are ready, it
becomes the schedule.

---

## Node briefs

Each brief is what you paste to start that session. They are deliberately short: the
detail lives in the spec, and duplicating it here would create exactly the drift problem
that made the previous kit's `spec.md` a byte-identical copy of the plan.

**Every brief implicitly includes:** read `CLAUDE.md`, read `Blueprint/docs/SPEC.md` §0 and the
sections your node touches, read the ADRs your node touches, work TDD in vertical slices,
report failures verbatim.

---

### Node 1 — Scaffold · AUTO
Build the package skeleton per SPEC §3 and §6. First `git mv Blueprint/pyproject.toml
pyproject.toml`: the library lives at the repo root (`src/`, `tests/`), and tests read
oracles from `Blueprint/tests/fixtures/`. `hatchling`, `src/` layout, version
single-sourced from `__init__.py`. CI matrix: CPython 3.10–3.13 × ubuntu/macos/windows.
Wire `ruff`, `mypy --strict`, `pytest` with markers `replication` and `slow`, and a
coverage floor of 90 % read from `pyproject.toml`. Add the layering check from SPEC §3 as
a test. No library code — every module is a stub with a docstring.
**Done when:** `pip install -e .[dev]` works, `pytest` collects zero failures, CI is green.

### Node 2a — φ · AUTO
Implement `_relevance.py` per SPEC §4.1. **Read ADR-0001 (including its v0.8.0 amendment)
and ADR-0002 first.** There are four natural wrong choices, and each passes a naive test:
`medcouple`, `PchipInterpolator`, `np.percentile` quartiles, and control points at the
fences. The right ones are Tukey hinges, whisker ends, φ = 0 on a side with no outliers, and
constant extension beyond the endpoints. φ must return the **exact** endpoint value at and
beyond the knots and be clipped to [0, 1] (SPEC §4.1 Step 4, ADR-0014); node 8's SERA gate
depends on it, and G0 alone will not catch its absence.
Implement gate G0 per `Blueprint/docs/REPLICATION.md` §2 against the committed
`Blueprint/tests/fixtures/phi_oracle.json`. Tolerances are in `Blueprint/ops/QUALITY_GATES.md` §1. Test A
compares against R's **recorded** per-split control points (`r_splits`), never against
anything computed by our own code.
Handle the degenerate path (control-point `x` not strictly increasing → φ ≡ 0,
`UserWarning`); it is reachable.
**Done when:** G0 Tests A, B and C are green, and swapping in `PchipInterpolator`, and
separately the v0.7.0 fence formula, each makes Test B fail. Demonstrate both; a gate
that cannot fail is not a gate.

### Node 2b — Embedding · AUTO
Implement `embed()` per SPEC §2.1. Read ADR-0005 — the mapping to the paper's
`create.data` is an off-by-one that must be in the docstring.
Property tests: no row contains a value at or after its target's time; exact shapes;
`exog` alignment; every documented `ValueError`.
**Done when:** property tests pass and the docstring states the `create.data` mapping.

### Node 3 — Bumps · AUTO
Implement `_bins.py` per SPEC §4.2 and ADR-0012. Bumps are formed in **value space** (sort
by `y`), not time. There are two crossing rules: under/smote use sign-change with `>`, over
uses `≥`. Classification is by **mean φ** per bump. Implement the no-bump rule
(unchanged + `UserWarning`).
Property test: bumps partition the value-sorted order exactly, with no gaps and no
overlaps. G3 worked examples: a two-sided and a one-sided φ, for both crossing rules.
**Done when:** the partition property holds under Hypothesis and the worked examples pass.

### Node 4 — Preferences · AUTO
Implement `_prefs.py` per SPEC §4.3 and ADR-0012. The preference is the case's **time rank
within its bump**, `j / r` (× φ for temporal+phi). It is **not** the position in the full
series; v0.7.0's brief had this inverted, and `Blueprint/docs/REPLICATION.md` §5 step 4 names it as
the likely bug.
Handle the zero-sum fallback to uniform, and the too-few-positive-probabilities fallback
for no-replacement draws.
**Done when:** a worked example with hand-written arithmetic passes for all three biases.

### Node 5 — Sampling · GATED
Implement `_sample.py` per SPEC §4.4. Read ADR-0004, ADR-0006 and ADR-0007, **including
their v0.8.0 amendments**, and ADR-0011. `"balance"` differs per strategy:
- under: each normal bump → rare total / #normal;
- over: each rare bump gains normal total / #rare copies;
- smote: each bump → `round_even(N / #bumps)`.

Sizes are truncated. `trunc`, `round5` and `round_even` are shared helpers. smote's
undersampling draws **with** replacement.
**Plan to post before implementing:** the three `"balance"` computations on one worked
example with a two-sided φ, and how explicit `o`/`u` map to them.
**Done when:** G3 worked examples pass for all three strategies, and the size properties of
SPEC §4.4 hold across randomised inputs.

### Node 6 — Synthesis · GATED
Implement `_synth.py` per SPEC §4.5 and ADR-0013. Defaults follow R (`r_quirks=True`):
- one λ per synthetic case;
- target weights from the last predictor;
- T/TPhi read values in bump order with neighbours found in time order (the index quirk);
- TPhi `τ` = time index over the max time index among the `k`;
- `nexs` + `extra` counts.

`r_quirks=False` switches to full-vector weights and time order throughout. The
bump-smaller-than-`k+1` and tied-last-predictor paths are reachable and must be tested.
**Plan to post before implementing:** neighbour selection for each bias, the index-quirk
mechanics, and how you stay deterministic under a fixed `random_state`.
**Done when:** determinism holds across two runs, and hand-worked examples match for each
bias under **both** `r_quirks` settings.

### Node 7 — Resampler · GATED
Implement `resampler.py`. Dispatch only — **no arithmetic in this file**. sklearn contract
per SPEC §2.2: `__init__` stores arguments unmodified and validates nothing;
`check_random_state` is called inside `fit_resample`.
Gate G2: `sklearn.utils.estimator_checks` plus `clone`/`get_params` round-trip.
Implement the time-ordering guarantee (synthetic cases follow their seed).
**Done when:** G2 green and the full 3 × 3 grid produces output for every cell.

### Node M0 — Metric oracle probe · GATED · `researcher` · Opus 5, xhigh
**This is research, not implementation. Do not write `metrics.py`.**
Follow `Blueprint/docs/REPLICATION.md` §3 exactly.

**(a) SERA — done (2026-09-22).** The recorded values come from the harness script
(`step = 0.01`, φ forced to 1 at both ends) and are reproduced to 1e-15. See ADR-0009
amendment 2 and `Blueprint/docs/REPLICATION.md` §3.2. Nothing left to research.

**(b) Utility surface `U(ŷ, y)` — known solvable, clean derivation open.** The metric
conventions are pinned (SPEC §4.7: `≥ t_E`, `1e-5` empty selection, `|1+u|`, `p = 0.5`). A
reference implementation reproduced recorded prec/rec/F1 to 1e-15, but it is GPL-derived
and quarantined. Derive `U` from Ribeiro (2011) §4 and verify it on the oracle
(`Blueprint/docs/REPLICATION.md` §3.3).
CLAUDE.md rule 3 applies with full force: `uba`'s source would answer this in thirty
seconds and you may not open it. **Before trusting any candidate oracle file, check its
provenance column in `Blueprint/replication/MANIFEST.csv`** — the SERA probe's first pass burned a
cycle comparing against a file already flagged `CONTAMINATED` there.
**Deliverables:** an ADR update recording the `u_i` evidence table, `Blueprint/tests/fixtures/metric_oracle.json`
(covering both SERA, two-sided splits at `step=0.01`, and prec/rec/F1 — do not freeze it on SERA alone), and a written
statement of the acceptance result against §3.4's tolerances.
**If you cannot reach the tolerance:** say so plainly and update the ADR documenting what
the metrics *do* implement. A documented difference is shippable; a silent one is not.

### Node 8 — Metrics · GATED
Implement `metrics.py` and `_utility.py` per SPEC §4.7 and M0's ADR. Gate G0b.
SERA integrates on a **uniform grid** (default `step = 0.001`, trapezoidal) with §4.1's φ;
G0b calls it with `step=0.01` on two-sided splits to match the recorded values (SPEC §4.7,
ADR-0009 amendment 2).
Metric conventions per SPEC §4.7: `φ ≥ t_E`; an empty selection gives **`1e-5`** (no
warning, never `nan`, never `0.0`); `|1+u|` and `|1+φ|` in the sums.
**Done when:** G0b green on three datasets spanning `%Rare` from 3.5 % to 21.1 %.

### Node 9a — Pipeline I/O · AUTO
`pipeline/io.py` and `pipeline/splits.py` per SPEC §2.4 and ADR-0010. Borrow
`load_csv` and the manifest reader from `imbalance_eval`. Do **not** borrow
`impute_dataframe`: it mean-imputes univariate series. Implement lag-window kNN per SPEC
§2.4 and the ADR-0010 amendment, with a G3 test that DS12 reaches %Rare 10.99. Do not
borrow the `ImbalancedLearningRegression` dependency or its `np.quantile` monkey-patch.
`temporal_split` yields contiguous windows, train strictly before test. Never shuffles.
**Done when:** a round-trip from CSV to `(X, y)` matches a hand-checked expectation, and
`pandas` appears in no Layer 1 import.

### Node 9b — Pipeline API · AUTO
`imbalance_summary` and `evaluate`. `evaluate` refits φ inside each split.
Cross-check `imbalance_summary` against `imbalance_eval`'s published Table 1 numbers —
noting SPEC §4.2's strict `>` may shift a count by one; if it does, say so rather than
loosening the comparison.
**Done when:** the tidy output frame has one row per (strategy, split, metric).

### Node 10 — CLI · AUTO
`pipeline/cli.py`, console script `tsresample`. Mirror `imbalance_eval`'s argument surface.
Write `docs/MIGRATION.md` stating every intentional behavioural difference (strict
threshold, φ source).
**Done when:** an `imbalance_eval` invocation translates 1:1 and differences are documented.

### Node 11 — Replication · GATED · `replicator`
Build gate G4 per `Blueprint/docs/REPLICATION.md` §4. Assertions R1–R5. **Run R4 first** — it is
exact, cheap, and its failure means ADR-0004 is wrong, which changes everything downstream.
Emit `replication_report.md`.
**Done when:** R1–R5 evaluated and reported. Note that "done" is *reported*, not *passed*
— if it fails, §5's kill criterion applies and the honest report is the deliverable.

### Node 12 — Docs · AUTO
README (installation, 30-second example, the three layers, the citation, and any
replication caveat from node 11), API reference from docstrings, `CITATION.cff`, examples,
and a **"Deviations from the paper's pseudocode"** page listing every R-over-paper choice
(ADR-0004, 0006, 0007, 0012, 0013) and every robustness deviation (ADR-0011) with its ADR.
The README currently opens with the agent start-here guide; keep that guide (it may move
to `CONTRIBUTING.md` with a link from the README), since `Blueprint/PROGRESS.md` stays the entry
point for future work.
**Done when:** a reader who has not read the spec can resample a CSV in under five minutes.

### Node 13 — Release · GATED
PyPI Trusted Publishing (OIDC, no long-lived token). Tag-triggered. `sdist` + pure wheel.
Verify the built wheel installs clean in a fresh venv and `import tsresample` works with
only `numpy`/`scipy`/`scikit-learn` present.
**Done when:** `pip install tsresample` works from PyPI and the wheel has no `pandas`
requirement.
