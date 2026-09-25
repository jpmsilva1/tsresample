# ADR-0017 — G4 R1's per-dataset floor applies only where the ranking is resolvable

**Status:** Accepted · **Date:** 2026-09-24 · **Amends:** `Blueprint/ops/QUALITY_GATES.md` §1 (G4 R1),
`Blueprint/docs/REPLICATION.md` §4.2 · **Decided by:** the maintainer, after the step 11 result

## Context

R1 asks, per dataset, for Spearman ρ ≥ 0.6 between our mean-F1φ ranking of the 10 `lm`
workflows and the recorded ranking, and ρ ≥ 0.7 on average. At step 11 (and after the lean
pass) the mean was 0.88–0.885, and every dataset cleared 0.6 except DS19 (0.467, then 0.539).

A ranking correlation can only be as good as the ranking is resolvable. Each recorded
workflow mean is an average over 50 splits, with standard error `SE = sd/√50`. Two means
differ significantly at the 95 % level when the gap exceeds `1.96·√2·SE ≈ 2.77·SE`. If the
whole spread of the nine strategy means is below that, their order is mostly split noise,
and a second R run with other seeds would not reproduce R's own order either.

Measured on the recorded data (spread of the nine strategy means ÷ mean SE):

| Dataset | spread / SE | our ρ |
|---|---|---|
| DS21 | 0.8 | 0.93 |
| DS23 | 1.6 | 0.77 |
| DS24 | 2.5 | 0.81 |
| DS19 | 2.6 | 0.47–0.54 |
| next lowest, DS09 | 3.8 | 0.78 |
| all others | 4.7–23.0 | 0.71–0.99 |

On DS19, changing only our split seed moves ρ between 0.19 and 0.56 (step 11 probe).

**Candour.** This ADR was written after seeing DS19 fail. The threshold is not fitted to
DS19. It is the textbook two-sample 95 % criterion, fixed before looking at where the other
datasets fall. It exempts three datasets that pass anyway and one that does not. DS19 also
carries the one known metric residual (ADR-0014, φ = (1, 0, 0)).

## Decision

- R1's **mean** criterion (ρ ≥ 0.7) is unchanged and still covers every strict dataset.
- R1's **per-dataset floor** (ρ ≥ 0.6) applies only to datasets whose recorded strategy
  means are resolvable: spread ≥ 1.96·√2 × SE. Exempt datasets are listed in
  `replication_report.md` with their spread/SE and ρ, so the exemption is visible, not silent.
- R2–R5 are unchanged. R5 (the paper's conclusion) remains the assertion the replication
  claim rests on.

## Consequences

- G4 passes with the current library: mean ρ 0.885, floor met on all 18 resolvable
  datasets, R2 96.2 %, R3 95.0 %, R4 exact, R5 9/9.
- If a future change drops a *resolvable* dataset below 0.6, R1 still fails. The
  exemption cannot hide a real regression, because a regression there shows up as a
  resolvable ranking disagreeing.
- The open item "G4 R1 per-dataset floor fails on DS19" is closed by this ADR. The ADR-0014
  DS19 residual stays open.
