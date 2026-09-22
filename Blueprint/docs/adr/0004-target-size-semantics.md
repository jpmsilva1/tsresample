# ADR-0004 — Bin target sizes are balanced to `tgtNr`; total dataset size is preserved

**Status:** Accepted, **Decision superseded 2026-09-22** by the amendment below · **Date:** 2026-09-03

## Context

Algorithm 4 of the paper defines `ng` as the number of synthetic cases to generate **per
existing case**. The callers disagree about what to pass it:

- Alg. 5 line 28 passes `tgtNr − |B|`
- Alg. 8 line 15 passes `tgtNr`
- Alg. 12 line 15 passes `tgtNr`

Read literally, Alg. 8/12 produce `|B| + tgtNr·|B|` rows in a bin that was supposed to
reach `tgtNr`. For a rare bin of 30 cases with `tgtNr = 100` that is 3030 rows, not 100.
This contradicts the paper's own prose ("balance the cases in the bins") and contradicts
the authors' own experiment code, which calls every resampler with `C.perc="balance"`.

## Decision

`ng` is interpreted as a **bin target size**, not a per-case multiplier.

Default (`o is None and u is None`, matching `C.perc="balance"`):

```
tgtNr = N / n_bins                      # N = total cases, n_bins = rare + normal bins
target(B) = round_half_up(tgtNr)        # per bin
```

- `under` — normal bins shrink toward `tgtNr`; rare bins untouched; clamp to `len(B)`.
- `over` — rare bins grow toward `tgtNr`; originals always retained; normal bins untouched.
- `smote` — rare bins grow by synthesis, normal bins shrink, both toward `tgtNr`.

Explicit `o` / `u` override this: rare target = `round_half_up(len(B)·(1+o))`, normal
target = `round_half_up(len(B)·u)`.

## Consequences

- Total output size is approximately `N`, and the test suite asserts
  `abs(len(y_out) − N) <= n_bins` — the slack is exactly per-bin rounding.
- This is a **documented deviation from the literal text of Algorithms 8 and 12**, and it
  must be stated in the library's own docs, not just here. Users comparing against a
  literal transcription of the pseudocode will see different output sizes and deserve to
  know why.
- It is the interpretation most likely to reproduce the published results, because it is
  what the authors' own code does. If gate G4 (replication) fails, **this ADR is the first
  place to look** — re-testing the literal reading is a cheap experiment and is named as
  such in the kill criterion (SPEC §7).

## Amendment — v0.8.0 (2026-09-22): per-strategy "balance"

"Bins are balanced to a target, not multiplied" stands; that rejects the literal `ng`
reading. But `tgtNr = N / n_bins` for every strategy is **superseded**. R's `"balance"`
differs by strategy (SPEC §4.4), and bins are value-space bumps (ADR-0012):

| strategy | R `"balance"` |
|---|---|
| under | each normal bump → `n_rare / #normal_bumps` (without replacement); rare kept |
| over | each rare bump **gains** `n_normal / #rare_bumps` copies (with replacement); all originals kept |
| smote | each bump → `round_even(N / #bumps)`: synthesis above, undersampling (with replacement) below |

Explicit `o` / `u` are R's `C.perc` multipliers, with the per-strategy meaning in SPEC §4.4.
Sizes are truncated (ADR-0007 amendment).

The size property `abs(len − N) ≤ n_bins` now holds for **smote only** (as `≤ 2·#bumps`).
Under shrinks and over grows by construction.

**Evidence:** [AUDIT] (ADR-0011). Corroborated by `lm` + resampler agreement with canonical
at ≈0.01 mean F1 across 20 datasets.
