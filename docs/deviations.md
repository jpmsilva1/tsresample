# Deviations from the paper's pseudocode

`tsresample` reproduces the **original R code** of Moniz, Branco & Torgo (2017). Where the
paper's Algorithms 1–13 and that code disagree, the library follows the code
([ADR-0011](../Blueprint/docs/adr/0011-r-algorithm-fidelity.md)). Each choice below has
its ADR with the evidence behind it. These are the library's decisions, not the authors'.

## R behaviour chosen over the paper's text

| Topic | Paper (literal reading) | `tsresample` (R behaviour) | ADR |
|---|---|---|---|
| Bin target sizes | Alg. 8/12 pass `tgtNr` as a per-case multiplier | Bins are *balanced*. under: each normal bump goes to the rare total ÷ #normal bumps. over: each rare bump gains the normal total ÷ #rare bumps. smote: every bump goes to `round(N / #bumps)` | [0004](../Blueprint/docs/adr/0004-target-size-semantics.md) |
| Replacement | unstated | under draws without replacement; over with; smote undersamples normal bumps *with* replacement | [0006](../Blueprint/docs/adr/0006-sampling-replacement.md) |
| Rounding | unstated | Sample sizes are truncated; ratios are rounded to 5 dp; the smote bump size is rounded half to even (R's `round`) | [0007](../Blueprint/docs/adr/0007-rounding-and-tau.md) |
| Bins | contiguous runs in time | "Bumps" in **value** space (sort by `y`, cut where φ crosses the threshold), classified by mean φ | [0012](../Blueprint/docs/adr/0012-value-space-bumps-and-time-rank-bias.md) |
| Temporal bias | position in the series | Time rank **within the bump**, `j / r` (× φ for temporal+phi) | [0012](../Blueprint/docs/adr/0012-value-space-bumps-and-time-rank-bias.md) |
| SMOTE λ | one per attribute | One λ per synthetic case | [0013](../Blueprint/docs/adr/0013-smote-generation-r-conventions.md) |
| SMOTE target | distance-weighted over all attributes | Weighted by the most recent lag only (`X[:, 0]`) | [0013](../Blueprint/docs/adr/0013-smote-generation-r-conventions.md) |
| SMOTE T/TPhi | neighbours and values in time order | Neighbours are found in time order but values are read in value order (R's index mismatch) | [0013](../Blueprint/docs/adr/0013-smote-generation-r-conventions.md) |
| Alg. 13 τ | nearness rank | Recency: time index ÷ the most recent candidate's | [0013](../Blueprint/docs/adr/0013-smote-generation-r-conventions.md) |

`TimeSeriesResampler(r_quirks=False)` switches the SMOTE rows to the paper's reading: full-vector
target weights, and time order throughout. The counts and the other rows stay as R has them.

## Robustness deviations (R gives no result here)

| Situation | R | `tsresample` | ADR |
|---|---|---|---|
| No rare bump or no normal bump | `stop()` | Returns the input unchanged with a `UserWarning` | [0011](../Blueprint/docs/adr/0011-r-algorithm-fidelity.md) |
| Degenerate φ control points (e.g. IQR = 0) | the spline constructor errors | φ ≡ 0 with a `UserWarning`, so nothing is resampled | [0011](../Blueprint/docs/adr/0011-r-algorithm-fidelity.md) |
| SMOTE target column with zero range | NaN | Midpoint of seed and neighbour | [0011](../Blueprint/docs/adr/0011-r-algorithm-fidelity.md) |
| Too few positive-probability cases for a draw without replacement | error | Takes all of them, fills the rest uniformly, and warns | [0011](../Blueprint/docs/adr/0011-r-algorithm-fidelity.md) |
| A bump of one case under smote | — | Kept as is; the size bound allows for it | [0015](../Blueprint/docs/adr/0015-smote-size-property-singleton-bumps.md) |

## Interface additions

| Addition | Why | ADR |
|---|---|---|
| `relevance=` accepts φ's control points (shape `(3, 2)`); `metrics.control_points(y)` | Utility-based precision needs φ(ŷ) and the bumps of the training φ | [0016](../Blueprint/docs/adr/0016-control-point-relevance.md) |
| `embed(series, k)` has `k + 1` lag columns | The paper's `create.data(ts, 10)` is `embed(series, k=8)` | [0005](../Blueprint/docs/adr/0005-embed-convention.md) |

## Known residual

For a φ with control-point values (1, 0, 0) (only the low side rare), utility-based
precision/recall have not been shown to match R's `uba`. This affects 312 recorded DS19
splits ([ADR-0014](../Blueprint/docs/adr/0014-utility-surface.md)).
