# ADR-0006 — Replacement policy: `under` without, `over` and `smote` with

**Status:** Accepted, **amended 2026-09-22** (smote row) · **Date:** 2026-09-03

## Context

The paper's pseudocode says "sample" without specifying whether draws are with or without
replacement. The choice is not cosmetic: undersampling with replacement can drop a case
entirely while duplicating another, and oversampling without replacement caps a bin's
growth at its own size, making `tgtNr` unreachable for strongly imbalanced bins.

## Decision

Taken from the authors' own experiment calls in `Exps.R`:

| Strategy | Call | Replacement |
|---|---|---|
| `under` | `randUnderRegressB/T/TPhi(… repl=FALSE)` | **without** |
| `over` | `randOverRegressB/T/TPhi(… repl=TRUE)` | **with** |
| `smote` | `smoteRegressB/T/TPhi(… repl=TRUE)` | **with** (for the neighbour draw) |

Only the function signatures and their default arguments were read from that file. The
resampler *bodies* vendored in the same file are UBL-derived and GPL; they are on the
quarantine list in `Blueprint/docs/PROVENANCE.md` and were not read.

## Consequences

- For `under`, a bin's target is clamped to `len(B)` — it can never be asked for more
  distinct cases than it holds, and it never shrinks below 1.
- For `over`, the original cases in a bin are always retained and only the *surplus*
  `target − len(B)` is drawn with replacement. This guarantees the property test "every
  original rare case survives `over`", which would not hold under a plain
  with-replacement draw of `target` cases.
- Duplicates are a real and expected output of `over`. Downstream users who cannot
  tolerate exact duplicates want `smote`, and the docs say so.

## Amendment — v0.8.0 (2026-09-22)

The smote row above is wrong about **what** `repl=TRUE` governs. In `smoteRegress*` it
controls the **undersampling of normal bumps**: they are drawn *with* replacement. The
SMOTE neighbour is chosen uniformly among the `k` (bias `None`) or deterministically
(T, TPhi). The `extra` seeds for the fractional part are drawn *without* replacement.
under and over rows stand.

**Evidence:** [AUDIT] (ADR-0011). The signature-only reading above could not see this,
because the meaning of an argument lives in the function body.
