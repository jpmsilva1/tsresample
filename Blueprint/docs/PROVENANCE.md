# Provenance and the cleanroom rule

`tsresample` is MIT-licensed. The reference implementations of these algorithms are
GPL-licensed. This document records how we stay on the right side of that line, and what
evidence exists that we did.

## The rule

**Algorithms come from the paper. Constants come from numerical artifacts. Neither comes
from reading GPL source.**

The paper (Moniz, Branco & Torgo 2017) publishes Algorithms 1–13 as pseudocode. Publishing
pseudocode in a journal puts the *ideas* in the open literature; it does not license the
authors' R implementation. So we implement from the pseudocode, and where the pseudocode
is ambiguous we resolve the ambiguity **from data we produced ourselves**, not from the
reference source.

That second half is the part that needs care, and it is what §3 is about.

## 1. Quarantine — paths an implementation agent must never open

These contain GPL-licensed source. Reading them taints the implementation.

| Path | Licence | Why it is tempting |
|---|---|---|
| `external_repos/ImbalanceMetrics/` | GPL-3 | Implements SERA — directly relevant since ADR-0009 |
| `external_repos/IRonPy/src/`, `external_repos/IRonPy/iron/` | vendored GPL-2 + LGPL | Wraps `uba`'s `phi.c`/`pchip.c`; see ADR-0003 |
| `<replication repo>/scratch/uba/` | GPL-2 | The reference φ and resamplers, in full |
| `<replication repo>/scratch/uba/R/phi.R` | GPL-2 | Would answer §4.1 directly. **Never opened.** |
| `<replication repo>/scratch/uba/src/phi.c`, `pchip.c` | GPL-2 | Same |
| `<replication repo>/src/original/R_Code/Exps.R` lines ~1700–2450 | UBL-derived, GPL | Vendored resampler bodies |
| `<replication repo>/src/adapted/Exps.R` (same region) | UBL-derived, GPL | Same |
| any `UBL`, `uba`, `smogn` source checkout | GPL | — |
| `python_port/Py_Code/`, `python_port/tests/`, `python_port/tools/`, `python_port/R_replication/*.R`, `python_port/R_replication/audit/` (colleague's Python port, vendored in this repo; previously `TSResampStrat_Python/Py_Code/tsresamp/` on Drive) | GPL-derived (line-by-line port of `uba` C, `Exps.R` bodies, `UBL::neighbours`; MARS ported from `earth`, GPL-3) | Complete working port. Its `*.md` reports, `results_cluster/` and `R_replication/results/` are numbers + prose and are permitted (`python_port/QUARANTINE.md`) |

The `Exps.R` entry is a **partial** quarantine: the file's top-level experiment protocol
(function signatures, default arguments, `EstimationTask` configuration) is ordinary
scientific method description and is fair to read. The vendored resampler *bodies* in that
line range are not. ADR-0006 was written from signatures and defaults only.

**Enforcement.** The rule is stated in `CLAUDE.md` rule 3, listed by path here, and
checked at review time — an agent's tool-call record is auditable, so "did anyone open a
quarantined path" is a question with a real answer. Ask it at every gate.

## 2. Permitted sources

| Source | Status |
|---|---|
| The paper's PDF and its Algorithms 1–13 | Primary. Use freely. |
| Ribeiro (2011) thesis, Torgo & Ribeiro (2009), Ribeiro & Moniz (2020) | Primary literature. |
| `Results (Clean)/**` — our own R experiment **output** | Numerical artifacts. See §3. |
| `data/paper_datasets_csv/**` | Public datasets (UCI etc.). |
| `imbalance_eval` (`github.com/jpmsilva1/imbalance_eval`) | The user's own work. Borrow per ADR-0010. |
| `report/pchip-alternatives-research.md` | The user's own research memo. |
| `external_repos/imbalanced-learn` | MIT. Reference for sklearn API conventions only. |
| `external_repos/python-packaging-user-guide` | Packaging conventions. |
| scikit-learn, scipy, numpy docs | — |

## 3. The distinction that makes this work

ADR-0001 and ADR-0002 pinned two formulas that the paper does not state precisely. They
were recovered from `mc.*_phi_ctrl.csv` — files containing five columns of *numbers*
(`iteration, point_index, ctrl_x, ctrl_phi, ctrl_deriv`) written out by our own R runs.

Reading those numbers is not reading `phi.c`. It is the same epistemic act as
black-box-testing a binary: observe inputs and outputs, infer the function. `phi.R` and
`phi.c` sit on disk a few directories away and were **not opened** — that is a checkable
claim about this project's tool-call history, not a promise.

The claim is further strengthened by independent corroboration: the same two conclusions
were reached in `report/pchip-alternatives-research.md` by an entirely different route,
and the plain-boxplot finding is independently stated in `imbalance_eval`'s README. A
conclusion reached three ways, none of which required our implementation to derive from
GPL source, is a robust one.

**Corollary for future work.** When a constant is unknown — the utility surface in task
M0 is the live example — the answer is *always* another oracle probe, never "let me just
check how UBL does it." The probe is slower. It is also the only version that ships.

## 4. Attribution

`tsresample` implements published algorithms and says so, prominently, in the README and
in `CITATION.cff`:

> The resampling strategies implemented here are those of Moniz, N., Branco, P., &
> Torgo, L. (2017). *Resampling strategies for imbalanced time series forecasting.*
> International Journal of Data Science and Analytics 3(3), 161–181. The relevance
> function follows Ribeiro, R. (2011), *Utility-based Regression*, PhD thesis, University
> of Porto. SERA follows Ribeiro, R. & Moniz, N. (2020), *Imbalanced regression and
> extreme value prediction*, Machine Learning 109, 1803–1835. This is an independent
> implementation and is not affiliated with or endorsed by those authors.

Citing the work you implement is both the licence-safe and the honest thing to do. The
"not affiliated" sentence is not boilerplate — the library deviates from the literal
pseudocode in documented places (ADR-0004, ADR-0005), and users must not read our choices
as the authors'.

## 5. Audit log

| Date | Question | Method | Outcome |
|---|---|---|---|
| 2026-09-03 | Is IRonPy usable and MIT? | Read its README, `setup.py`, `iron/phi.py`, `file` on its binaries | No and no. ADR-0003. |
| 2026-09-03 | Plain or adjusted boxplot? | Compared `mc.*_phi_ctrl.csv` against both, 20 datasets | Plain. ADR-0001. |
| 2026-09-03 | PCHIP or Hermite? | `ctrl_deriv` column + `%Rare` vs Table 1, 20 datasets | Hermite, dydx=0. ADR-0002. |
| 2026-09-03 | Replacement policy? | `Exps.R` call signatures only (permitted region) | ADR-0006. |
| 2026-09-03 | Is PyPI `tsresample` free? | HTTP request to the project URL | 404 — available. |
| 2026-09-22 | Utility surface `U`? | Clean M0 session: Ribeiro (2011) §3.3–3.4 + recorded CSVs + the project's own `sera_metric.py`. No quarantined path opened; port-audit report not opened. | Derived; 62,088/62,400 splits within 1e-6. ADR-0014. DS19 φ=(1,0,0) residual. |
| 2026-09-22 | Audit of colleague's Python port | **User-authorised read of the quarantined port's source** (option 1: audit, use as inspiration for behaviour only). Session is tainted. | Findings (the maintainer's private audit report, not in this repo) are behavioural/numerical only and are folded into SPEC v0.8.0 as [AUDIT] items. Implementation sessions work from the SPEC, never the port's source; no code is written from this session. |
| 2026-09-22 | SPEC v0.8.0 corrections | Tainted session (above) wrote SPEC v0.8.0 and ADR-0011–0013 plus amendments, stating **behaviour only**. [ORACLE] items re-verified against recorded R output; φ §4.1 re-implemented from the new SPEC text (not the port) and checked against 90 R splits (85 exact). [AUDIT] items are gated by G3/G4 before their nodes are done. | Implementation sessions use SPEC/ADRs only. |
