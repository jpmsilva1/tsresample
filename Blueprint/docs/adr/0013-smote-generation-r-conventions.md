# ADR-0013 — SMOTE generation follows R, quirks included, behind `r_quirks`

**Status:** Accepted · **Date:** 2026-09-22 · **Supersedes:** SPEC v0.7.0 §4.5,
ADR-0007 part B · **Evidence:** [AUDIT], gated by G3 + G4 (ADR-0011)

## Context

v0.7.0 specified per-attribute λ, full-vector distance weights for the synthetic target,
and `τ_i = i/k` by nearness for Algorithm 13. The R code does none of these, and has two
behaviours that are bugs by the paper's intent but produced the published numbers.

## Decision (SPEC §4.5)

| Aspect | R convention (default) | `r_quirks=False` |
|---|---|---|
| λ | one `U(0,1)` draw per synthetic case, shared by all predictors | same |
| synthetic target | weights from the **last predictor only**, so `y_new = y_seed + λ·(y_nn − y_seed)`, or the midpoint if that predictor ties | full-vector Euclidean `d1`, `d2` |
| T/TPhi row order | neighbours found in time order, values read in bump (value) order: **index mismatch** | time order for both |
| T neighbour | most recent of the `k` | same |
| TPhi neighbour | `argmax τ_m·φ(y_m)`, `τ_m = (time idx + 1)/(max time idx among the k + 1)` | same |
| counts | `nexs = floor(c−1)` per seed + `extra = floor(|B|·frac)` distinct seeds without replacement | same |
| neighbours | unscaled Euclidean on predictors, self excluded, ties to lowest index, `k_eff = min(k, |B|−1)` | same |

`τ` encodes **recency** relative to the most recent candidate. ADR-0007B's `i/k` encoded
nearness rank, which carries no recency at all.

## Evidence

- These are the R conventions per the 2026-09-22 audit, which also measured that
  *correcting* the index quirk moved results away from R (SMOTET on DS10: 0.535 vs R 0.654).
- The algebraic reduction of the last-predictor weights to the shared λ is exact. With
  `x_new[a] = x_seed[a] + λΔ`: `d1 = λ|Δ|` and `d2 = (1−λ)|Δ|`.

## Consequences

- G3 carries hand-worked examples for both `r_quirks` settings. G4 runs with the default.
- Docs list both quirks explicitly under "Deviations from the paper's pseudocode".

## Amendment — step 6 (2026-09-24): which column is "the last predictor"

In R's `create.data` layout (`V1…V_m`, oldest lag first, target last) the "last
predictor" is the **most recent lag**. The library's `embed` puts the most recent lag
**first** (`X[:, 0] = y_t`, ADR-0005). The weights therefore use `X[:, 0]`, which is the
same variable as R's column. With user-built `X` (not from `embed`), column 0 is taken
as the most recent lag, and the docstring says so. The column only matters when the
seed and neighbour tie on it (the midpoint rule); otherwise the target reduces to the
shared λ.
