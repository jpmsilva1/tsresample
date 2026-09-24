# Quality gates

What "done" means. A node is not done when the code works; it is done when the gates below
are green and a reviewer who did not write it agrees.

---

## 1. The numbers, stated once

These appear here and in `Blueprint/docs/SPEC.md` §6. Nowhere else. If you find a different value in
another file, **that file is the bug** — the previous kit shipped with the φ tolerance at
`1e-6` in one file and `1e-5` in another, and coverage at 90 % in one and 85 % in another,
which meant neither number was real.

| Thing | Value |
|---|---|
| Coverage floor, `src/tsresample/` | **90 %** |
| G0 Test A — R per-split control points `(x, φ)` | `1e-6` absolute (DS05 excluded, open item) |
| G0 Test B — `%Rare` vs paper, embedded target | ≤ **0.25 pp** MAE across the 18 NA-free datasets; ≤ 2.5 pp worst case |
| G0 Test C — one-sided φ (DS10) | exact |
| G0b — metrics vs recorded values | `1e-6` relative (`sera` at `step=0.01`, two-sided splits only), `1e-6` absolute (prec/rec/F1) |
| G4 R1 — ranking correlation | Spearman ρ ≥ 0.7 mean, ≥ 0.6 per dataset |
| G4 R2 — direction agreement | ≥ 85 % |
| G4 R3 — median inside recorded IQR | ≥ 75 % of cells |
| G4 R4 — resampled row count | exact: equals SPEC §4.4 counts (smote: `|len − N| ≤ 2·#bumps`) |
| Python versions | 3.10, 3.11, 3.12, 3.13 |
| Platforms | ubuntu, macos, windows |

The coverage floor is read from `pyproject.toml` by CI, not hardcoded in a workflow file.

---

## 2. Seams

Tests live at these boundaries and nowhere else (CLAUDE.md rule 5). Confirmed up front so
testing effort lands on the critical paths rather than on every private helper.

| Seam | Tested via | Not tested |
|---|---|---|
| `embed(series, k, horizon, exog) -> (X, y)` | public call | internal slicing |
| `_relevance.control_points(y) -> ndarray (3, 2)` of `(x, φ)` | direct, against the oracle | the hinge helper alone |
| `_relevance.phi(y, cp) -> ndarray` | direct, against the oracle | the spline object |
| `_bins.bumps(y, phi, t_R, rule) -> list[Bump]` (value-space; `rule` ∈ {under/smote, over}) | direct | the sort |
| `_prefs.preference(bump, time_index, phi, bias) -> ndarray` (within-bump time rank) | direct | normalisation helper |
| `_prefs.draw(p, size, replace, rng) -> ndarray` (added in step 4: the too-few-positive fallback of SPEC §4.3 needs a draw to test) | direct | — |
| `_sample.targets(bumps, N, strategy, o, u) -> list[int]` | direct | the `trunc` / `round5` / `round_even` helpers alone |
| `_sample.resample(bumps, N, strategy, o, u, time_index, phi, bias, rng) -> (idx, smote jobs)` (added in step 5: performs the draws so `resampler.py` stays arithmetic-free) | direct | — |
| `_synth.synthesize(..., r_quirks) -> (X_new, y_new)` | direct, both `r_quirks` settings | distance helper |
| `TimeSeriesResampler.fit_resample(X, y)` | public call | dispatch internals, `_sample.assemble` (ordering is asserted through the public call) |
| `metrics.{precision_phi,recall_phi,f1_phi,sera}` | public call | `_utility` internals |
| `pipeline.{load_series,imbalance_summary,temporal_split,evaluate}` | public call | CSV parsing details |
| CLI | subprocess, exit codes + stdout | argparse wiring |

`_relevance.control_points` and `_relevance.phi` are private modules exposed as seams
deliberately: they are where the oracle comparison happens, and routing G0 through
`fit_resample` would make a formula regression show up as a resampling failure three
layers away.

---

## 3. Gate definitions

| Gate | Runs | Command | Blocking |
|---|---|---|---|
| **G0** φ oracle | every push | `pytest tests/test_phi_oracle.py` | yes |
| **G0b** metric oracle | every push, once node 8 lands | `pytest tests/test_metric_oracle.py` | yes |
| **G1** properties | every push | `pytest tests/test_properties.py` | yes |
| **G2** sklearn protocol | every push | `pytest tests/test_sklearn_contract.py` | yes |
| **G3** algorithmic | every push | `pytest tests/test_algorithms.py` | yes |
| **G4** replication | nightly + tags | `pytest -m replication` | release only |
| **L** lint/format | every push | `ruff check . && ruff format --check .` | yes |
| **T** types | every push | `mypy --strict src/` | yes |
| **C** coverage | every push | `pytest --cov=tsresample --cov-fail-under=90` | yes |
| **A** layering | every push | `pytest tests/test_layering.py` | yes |
| **P** provenance | every gate, by review | see §5 | yes |

G4 skips with a clear message when `TSRESAMPLE_REPLICATION_ROOT` is unset. It never
silently passes — a skipped replication gate reports as skipped, and the release workflow
treats a skip as a failure.

---

## 4. Definition of done, per node

A node is done when **all** of these hold:

1. The gates listed for its deliverable are green, and the output is pasted in the report.
2. A `reviewer` pass has run and its findings are addressed or explicitly declined with a
   reason.
3. Every ambiguity encountered is either covered by an existing ADR or has a new one.
4. No quarantined path was opened (§5).
5. The spec and the code agree. If the node revealed a spec error, the spec is updated in
   the same PR.
6. Test names read as behaviour, not as structure — `test_over_retains_every_original_rare_case`,
   not `test_sample_2`.

Item 2 is not a formality. The reviewer runs in a **fresh session with no memory of the
implementation**, working from `Blueprint/docs/SPEC.md`, the ADRs, and this file. That independence
is the actual mechanism — same model family is fine, shared context is not.

---

## 5. The provenance check (gate P)

At every gate, answer explicitly: **did this session open any path in
`Blueprint/docs/PROVENANCE.md` §1?**

The answer is auditable from the tool-call record, so answer it honestly and specifically —
"no quarantined path was opened" or "opened `Exps.R` lines 1–120 only, which is the
permitted protocol region." A vague answer is treated as a failure.

This is the only gate that cannot be automated and the only one where a false pass is
unrecoverable: a tainted implementation cannot be un-tainted, it can only be rewritten.

---

## 6. Anti-patterns that fail review automatically

- A test that recomputes its expected value the way the implementation does.
- A tolerance loosened to make a test pass, without an ADR saying why.
- A number that contradicts §1.
- A target count computed without the shared `trunc` / `round5` / `round_even` helpers (ADR-0007 amendment), or half-up rounding anywhere.
- `PchipInterpolator`, `medcouple` or `np.percentile`/`np.quantile` for φ's quartiles anywhere in `src/` (ADR-0001, ADR-0002).
- A paper-intent reading as the default where the R behaviour differs (ADR-0011).
- A G0 expected value computed from our own formula instead of taken from R's recorded output (SPEC §0 #13).
- `import pandas` in a Layer 1 module.
- An abstraction with one implementation.
- A green summary over a red run (CLAUDE.md rule 8).
- A resolved ambiguity with no ADR (CLAUDE.md rule 2).
