# ADR-0015 — smote's size property with single-case bumps

**Status:** Accepted · **Date:** 2026-09-24 · **Amends:** SPEC §4.4 "Size properties"

## Context

SPEC §4.4 states two things that disagree when a bump holds exactly one case:

- the smote rule "`c_B == 1` or `|B| == 1` → keep" (a single case has no neighbour,
  `k_eff = |B| − 1 = 0`, so it cannot be synthesised, and undersampling it is moot);
- the size property `|len(out) − N| ≤ 2·#bumps`.

Counterexample found by the step 5 Hypothesis property: bumps of sizes 1 (normal) and 11
(rare), `N = 12`, `B* = round_even(12/2) = 6`. The singleton is kept (1), the rare bump
is undersampled to `trunc(6/11 · 11) = 6`, total 7, and `|7 − 12| = 5 > 4`.

## Decision

The per-bump rule stands (it is the R behaviour recorded by the audit, ADR-0011). The size
property is stated precisely:

```
|len(out) − N| ≤ 2·#bumps + Σ_{|B| = 1} |B* − 1|
```

i.e. every bump of size > 1 lands within 2 of `B*`; a singleton stays at 1.

## Consequences

- G1 tests this form. G4 R4 already compares against the exact per-bump counts of SPEC
  §4.4 (`_sample.targets`), which are unaffected; its `≤ 2·#bumps` shorthand is read with
  this correction.
- Real splits have singleton bumps when one extreme value forms its own rare bump.
