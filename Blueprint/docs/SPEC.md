# tsresample — Specification v0.8.0

> Supersedes v0.7.0 (archived at `archive/v0.7.0-draft/`), which superseded
> `PyUBA_PyUBL_Modernization_Plan.md` v0.6.0. Changes that alter numerical behaviour are
> recorded as ADRs in `Blueprint/docs/adr/`; §0 lists them. **Read §0 before implementing
> anything.** v0.6.0 specified two formulas that were provably wrong. v0.7.0 fixed those
> but specified φ's control points, the bins, the target counts, the temporal bias, the
> SMOTE generator and the metric conventions in ways that disagree with the reference R
> algorithms.

**Governing principle (ADR-0011).** The library reproduces the **original R algorithms**
(the `Exps.R` resamplers and `uba`'s relevance and utility functions), quirks included.
The canonical R replication is the *measuring instrument* used to verify that, not a
target in itself. Where the paper's text and the R behaviour differ, the R behaviour is
the default, and the paper reading is at most an opt-in flag. Choices made by the
experiment harness (imputation mode, SVM row caps, ARIMA estimation, learners) configure
how the gates read the oracle. They never shape the library.

---

## 0. What changed, and why it matters

### 0.1 v0.7.0 → v0.8.0

Evidence tags: **[ORACLE]** = verified here against recorded canonical R numbers.
**[AUDIT]** = established by the 2026-09-22 audit of a colleague's Python port of the R
code (the maintainer's private audit report, not in this repo), corroborated by that port reproducing canonical
`lm`+resampler F1 to ≈0.01. [AUDIT] items carry a gate that checks them (§5) before they
count as verified.

| # | v0.7.0 said | v0.8.0 says | Evidence | ADR |
|---|---|---|---|---|
| 12 | φ control points at the **fences** `Q₁ ∓ 1.5·IQR`, type-7 quartiles; φ = 1 at both ends | Control points at the **whisker ends** (the most extreme observations inside the fences), quartiles from **Tukey hinges**; a side with no outliers gets `(min or max, φ = 0)`; constant extrapolation at the endpoint φ | **[ORACLE]** new §4.1 wording reproduces R's per-split control points in **85/90** reconstructed training splits (the 5 misses are DS05, a fence-precision artefact of the CSV export). Fences: 4/54. `%Rare` vs Table 1: **0.17 pp** MAE (was 0.82). `imbalance_eval`'s φ (via iblr, black-box) gives the identical 85/90 and 0.17 pp. φ = 0 endpoints occur in every DS10 split and in DS19, DS21–24 | [0001](adr/0001-plain-boxplot-not-adjusted.md) (amended) |
| 13 | G0 fixture `full_series_control_x` is an R oracle | It was generated from our own fence formula: 20/20 match to the v0.7.0 SPEC, 5/20 to R. **Removed.** Replaced with R per-split control points plus the reconstructed training slice | **[ORACLE]** | [0001](adr/0001-plain-boxplot-not-adjusted.md) (amended) |
| 14 | Bins = maximal **time-contiguous** runs of `φ > t_R` | Bins ("bumps") are formed in **value space**: sort by `y`, cut where φ crosses `t_R`, classify each bump by its **mean φ** | [AUDIT] | [0012](adr/0012-value-space-bumps-and-time-rank-bias.md) |
| 15 | Temporal bias `p = i/n`, `i` = position in the full series | `p = j/r`, `j` = the case's **time rank within its bump** of size `r` (× φ for TPhi) | [AUDIT] | [0012](adr/0012-value-space-bumps-and-time-rank-bias.md) |
| 16 | `"balance"` = every bin → `N / n_bins`, for all strategies | Per strategy: under → each normal bump to `n_rare / #normal`; over → each rare bump *gains* `n_normal / #rare` copies; smote → each bump to `round(N / #bumps)` | [AUDIT] | [0004](adr/0004-target-size-semantics.md) (amended) |
| 17 | Half-up rounding of target counts | R semantics: sample sizes **truncated**; per-bump ratios rounded to 5 dp; the smote bump size uses R's `round` (half-to-even) | [AUDIT] | [0007](adr/0007-rounding-and-tau.md) (amended) |
| 18 | smote `repl=TRUE` applies to the neighbour draw | It applies to **undersampling the normal bumps** (with replacement). The neighbour draw is uniform | [AUDIT] | [0006](adr/0006-sampling-replacement.md) (amended) |
| 19 | SMOTE: λ per attribute; full-vector distance weights; TPhi `τ = i/k` by nearness | **One λ per synthetic case**; target interpolated from the **last predictor only** (which reduces to the same λ); TPhi `τ` = neighbour's **time rank ÷ max time rank among the k**; T/TPhi keep R's **seed/neighbour index-ordering quirk**. `r_quirks=False` opts out of both quirks | [AUDIT] | [0013](adr/0013-smote-generation-r-conventions.md), [0007](adr/0007-rounding-and-tau.md) (amended) |
| 20 | Metrics: `φ > t_R`; no positives → `0.0` + warning | `φ ≥ t_E` (default 0.9); no positives → precision / recall = **1e-5**; `|1+u|`, `|1+φ|` in the sums | **[ORACLE]** recorded rows show `prec = 1e-05`, `F1 = 1.99996e-05`. A reference implementation of these conventions matches recorded prec/rec/F1 to **1e-15** on 45 cases | [0009](adr/0009-relevance-metrics-in-scope.md) (amended) |
| 21 | SERA by exact breakpoints | Uniform grid, `step = 0.001` default, trapezoidal, φ per §4.1. SERA is a *harness* metric (the 2017 paper does not report it), so ADR-0011 does not bind it; the recorded values used `step = 0.01` and a one-sided-φ bug, reproduced to 1e-15 | **[ORACLE]** | [0009](adr/0009-relevance-metrics-in-scope.md) |
| 22 | No behaviour when a split has no rare (or no normal) bump | Return input unchanged + `UserWarning` (R `stop()`s; the canonical harness patched it to a no-op) | canonical `DIVERGENCE_ANALYSIS.md` D3 | [0011](adr/0011-r-algorithm-fidelity.md) |
| 23 | `impute="knn"` borrowed from `imbalance_eval` (`KNNImputer` on the series) | **Lag-window kNN** in time order (k = 10, `exp(−d)` weights); `imbalance_eval`'s version silently mean-imputes a univariate series | [ORACLE] DS12 %Rare 10.99 vs paper 11.0 (mean-imputation: 14.65); DS13 open | [0010](adr/0010-end-to-end-pipeline.md) (amended) |
| 24 | `imbalance_summary` counts strict `φ > t` | `φ ≥ t` (identical %Rare on all 18 NA-free datasets; matches metrics and `imbalance_eval`) | [ORACLE] | [0010](adr/0010-end-to-end-pipeline.md) (amended) |

### 0.2 v0.6.0 → v0.7.0 (still in force unless amended above)

| # | v0.6.0 said | v0.7.0 says | ADR |
|---|---|---|---|
| 1 | φ uses the **adjusted boxplot** (medcouple) | plain Tukey boxplot (convention refined by #12) | [0001](adr/0001-plain-boxplot-not-adjusted.md) |
| 2 | `PchipInterpolator` | `CubicHermiteSpline`, `dydx = 0` at every knot | [0002](adr/0002-hermite-not-pchip.md) |
| 3 | `IRonPy` is the φ oracle | unusable; the oracle is our own recorded R output | [0003](adr/0003-phi-oracle-replacement.md) |
| 4 | `ng` read literally from Alg. 4 | target-size semantics pinned (refined by #16) | [0004](adr/0004-target-size-semantics.md) |
| 5 | `embed(series, k)` unqualified | `create.data(ts, k_R)` ≡ `embed(series, k=k_R−1, horizon=1)` | [0005](adr/0005-embed-convention.md) |
| 6 | replacement unstated | under without, over and smote with (smote refined by #18) | [0006](adr/0006-sampling-replacement.md) |
| 7–8 | rounding and τ unstated | superseded by #17 and #19 | [0007](adr/0007-rounding-and-tau.md) |
| 9 | φ fit scope unstated | φ is fit only on the `y` passed in | [0008](adr/0008-phi-fit-scope.md) |
| 10 | no metrics module | `precision_phi`, `recall_phi`, `f1_phi`, `sera` | [0009](adr/0009-relevance-metrics-in-scope.md) |
| 11 | primitives only | primitives + pipeline + CLI | [0010](adr/0010-end-to-end-pipeline.md) |

**Corroboration of #1–#2.** ADR-0001 and ADR-0002 were reached twice, independently: once by
reverse-engineering our own R experiment output, and once in
`report/pchip-alternatives-research.md` (2026-08-22). v0.8.0's #12 refines *where* the
plain-boxplot control points sit; it does not reopen plain-vs-adjusted or Hermite-vs-PCHIP.

---

## 1. What this library is

`tsresample` implements the resampling strategies of **Moniz, Branco & Torgo (2017),
"Resampling strategies for imbalanced time series forecasting"** (*Int. J. Data Science
and Analytics*) as a dependency-light, MIT-licensed, scikit-learn-compatible Python
package.

It covers the workflow **end to end** in three layers, each usable on its own (ADR-0010).

**Layer 1 — primitives.** Pure `numpy`/`scipy`/`scikit-learn`.
- `embed()` — time-delay embedding of a univariate series into a supervised matrix.
- `TimeSeriesResampler` — the 3 × 3 strategy grid (Algorithms 1–13 of the paper).
- `tsresample.metrics` — `precision_phi`, `recall_phi`, `f1_phi`, `sera`. All
  relevance-based (ADR-0009).

**Layer 2 — pipeline** (`tsresample.pipeline`, needs the `[io]` extra).
- `load_series`, `imbalance_summary`, `temporal_split`, `evaluate`.

**Layer 3 — CLI** (`tsresample` console script). Single-file and manifest batch modes.

**Scope — what does not ship, and will not:**
- No learners or forecasting models. Bring your own sklearn-compatible estimator.
- No plain RMSE / MAE / R². `sklearn.metrics` has them; a second copy earns nothing.
  The metrics module exists *because* it is relevance-weighted.
- No R dependency, no compiled extension, no C code.

**Dependencies.** Runtime, Layer 1: `numpy`, `scipy`, `scikit-learn` — nothing else.
ADR-0001 removed the only reason `statsmodels` was ever considered; ADR-0010 declined
`ImbalancedLearningRegression`. Layers 2–3 add `pandas` behind the optional `[io]` extra,
so `pip install tsresample` stays light.

**Design goals, in priority order:**
1. **Numerically faithful to the original R algorithms** (ADR-0011), verified against
   recorded R output rather than against prose.
2. **Boring to use** — two public symbols, sklearn conventions, no surprises.
3. **Cheap to install** — pure Python, wheels-only deps, no build step.

---

## 2. Public API

The entire public surface. Anything not listed here is private and may change.

```python
from tsresample import embed, TimeSeriesResampler
```

### 2.1 `embed`

```python
def embed(
    series: ArrayLike,
    k: int,
    *,
    horizon: int = 1,
    exog: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a supervised matrix from a univariate series by time-delay embedding.

    Row t of X is [y_t, y_{t-1}, ..., y_{t-k}] (k+1 lagged values, most recent
    first). The paired target is y[t] = series[t + horizon].

    Parameters
    ----------
    series : array-like of shape (n,)
        Univariate series, ordered oldest to newest. Must be 1-D and finite.
    k : int
        Number of *additional* lags beyond the current value, so each row holds
        k+1 columns. Must satisfy k >= 1.
    horizon : int, default 1
        Forecast horizon. Must be >= 1.
    exog : array-like of shape (n, m), optional
        Exogenous columns, aligned to `series` by position. Row t receives
        exog[t] (contemporaneous only — no lagging is applied). Appended to the
        right of the lag block.

    Returns
    -------
    X : ndarray of shape (n - k - horizon, k + 1 + m)
    y : ndarray of shape (n - k - horizon,)

    Raises
    ------
    ValueError
        If `series` is not 1-D, contains NaN or inf, `k < 1`, `horizon < 1`,
        `exog` length does not match `series`, or the series is too short to
        produce at least one row.

    Notes
    -----
    Mapping to the reference implementation: the paper's R helper
    ``create.data(ts, m)`` builds ``embed(ts, m)[, m:1]`` and treats the last
    column as the target, giving ``m - 1`` predictors and target ``y_t``.
    The equivalent call here is ``embed(series, k=m - 1, horizon=1)``.
    See Blueprint/docs/adr/0005-embed-convention.md.
    """
```

Row `t` is emitted only when both the full lag window and the target exist, so the
first `k` and the last `horizon` positions are dropped. **No row ever contains a value
observed at or after its own target time** — this is the property the tests assert, and
it is what makes the embedding safe for temporal cross-validation.

### 2.2 `TimeSeriesResampler`

```python
class TimeSeriesResampler(BaseEstimator):
    def __init__(
        self,
        strategy: Literal["under", "over", "smote"] = "smote",
        bias: Literal[None, "temporal", "temporal+phi"] = None,
        *,
        rel_threshold: float = 0.9,
        relevance: Literal["auto"] | ArrayLike | Callable = "auto",
        k: int = 5,
        o: float | None = None,
        u: float | None = None,
        r_quirks: bool = True,
        random_state: int | RandomState | None = None,
    ) -> None: ...

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

| Parameter | Meaning |
|---|---|
| `strategy` | `"under"` (Alg. 2/6/10), `"over"` (Alg. 3/7/11), `"smote"` (Alg. 5/8/12). |
| `bias` | `None` = no bias (B variants), `"temporal"` = T variants, `"temporal+phi"` = TPhi variants. |
| `rel_threshold` | `t_R`, default 0.9 (the authors' `thr.rel`). Bumps are classified by their **mean φ** against `t_R` (§4.2), not case by case. |
| `relevance` | `"auto"` fits φ from `y` (§4.1). An array of shape `(n,)` supplies φ directly. A callable is invoked as `f(y) -> ndarray` and must return values in `[0, 1]`. |
| `k` | Nearest-neighbour count for `strategy="smote"`. Ignored otherwise. |
| `o`, `u` | R's `C.perc` multipliers. `None` (default) = `"balance"` (§4.4). Explicit: `u` scales normal bumps, `o` rare bumps, with the per-strategy meaning in §4.4. |
| `r_quirks` | `True` (default) reproduces R's two SMOTE quirks: the seed/neighbour index ordering in T/TPhi and last-predictor target weights (§4.5, ADR-0013). `False` gives the paper-intent reading. |
| `random_state` | Stored verbatim (sklearn contract). Resolved with `check_random_state` **inside** `fit_resample`, never in `__init__`. |

**Contract obligations.** `get_params()` / `set_params()` round-trip; `clone()` yields an
untrained equal estimator; `__init__` performs **no validation and no computation** and
stores every argument unmodified under its own name. Validation happens in
`fit_resample`. This is what `sklearn.utils.estimator_checks` enforces, and the test
suite runs those checks.

`fit_resample` is not `fit` + `transform`: it returns a *different number of rows* than it
received, so it deliberately does not implement the transformer protocol. It is stateless
between calls.

**Ordering guarantee.** The returned rows are sorted by the original time index, with
synthetic cases inserted immediately after their seed case. Downstream temporal splits
therefore remain meaningful. The reference implementation returns rows in bump order
(and appends over-sampled copies at the end). This changes only the *order*, never *which*
cases are present or how they were drawn, so it does not affect fidelity (ADR-0011).

### 2.3 `tsresample.metrics` (ADR-0009)

```python
from tsresample.metrics import precision_phi, recall_phi, f1_phi, sera

def precision_phi(y_true, y_pred, *, relevance="auto", rel_threshold=0.9) -> float: ...
def recall_phi   (y_true, y_pred, *, relevance="auto", rel_threshold=0.9) -> float: ...
def f1_phi       (y_true, y_pred, *, relevance="auto", rel_threshold=0.9,
                  beta=1.0) -> float: ...
def sera         (y_true, y_pred, *, relevance="auto",
                  return_curve=False) -> float | tuple[NDArray, NDArray]: ...
```

`relevance` takes the same three forms as the resampler (`"auto"`, an array, a callable)
and resolves through the same `_relevance.py`, so a case that the resampler treated as
rare is scored as rare. When `"auto"`, φ is fit on `y_true`. For precision and recall,
`rel_threshold` is an **event** threshold applied case by case with `≥` (§4.7). This is not
the bump rule of §4.2.

All four are plain functions, not sklearn scorer objects. Wrap with
`sklearn.metrics.make_scorer(f1_phi, greater_is_better=True)` if you need one; the
library does not ship pre-made scorers because the relevance argument makes the useful
ones application-specific.

`sera(..., return_curve=True)` returns `(t, SER_t)` for plotting the relevance-error
curve, which is the diagnostic users actually want when a single SERA number moves.

### 2.4 `tsresample.pipeline` (ADR-0010, needs `tsresample[io]`)

```python
from tsresample.pipeline import load_series, imbalance_summary, temporal_split, evaluate

def load_series(path, *, target, date_col=None, diff=False,
                impute="knn") -> NDArray: ...
def imbalance_summary(y, *, rel_threshold=0.9, relevance="auto") -> dict: ...
    # {"N", "n_normal", "n_rare", "IR", "pct_rare"} — one row of the paper's Table 1;
    # a case is counted rare iff φ(y) ≥ rel_threshold
def temporal_split(X, y, *, train_size=0.5, test_size=0.25, n_reps=50,
                   random_state=None) -> Iterator[tuple[NDArray, ...]]: ...
def evaluate(estimator, X, y, *, strategies=None, metrics=None,
             splitter=None) -> DataFrame: ...
    # tidy long format: one row per (strategy, split, metric)
```

`load_series(impute=...)`: `"knn"` (default) fills each gap by **lag-window kNN**. Walk
the series in time order; for missing `y_t`, standardise its lag window `y_{t−9..t−1}`;
take the 10 nearest fully-observed embedded rows by Euclidean distance on the observed
lags; fill with their targets weighted `exp(−d)`. Filled values count as observed for later
gaps, and leading gaps with no observed lag are dropped with a warning. `"drop"` removes
gap rows; `None` raises on NaN. Do **not** use `KNNImputer` on the bare series: with no
other features it degenerates to mean imputation (ADR-0010 amendment).

`temporal_split` yields **contiguous** windows with train strictly before test — the
Monte Carlo temporal estimation of the paper (`nReps=50, szTrain=.5, szTest=.25`). It is
not `KFold` and never shuffles; shuffling a time series is the mistake this function
exists to prevent.

`evaluate` refits φ inside each split (ADR-0008), so the whole grid is leakage-free by
construction.

Errors name what went wrong and what was available — a missing target column lists the
columns present, and a length-1 series raises rather than reporting a straight-faced
`IR = 0`. This discipline is inherited from `imbalance_eval` and applies library-wide.

---

## 3. Architecture

```
src/tsresample/
├── __init__.py        # re-exports embed, TimeSeriesResampler, __version__ — nothing else
├── py.typed           # PEP 561 marker
│
│   # ---- Layer 1: primitives (numpy / scipy / sklearn only) ----
├── embed.py           # embed()
├── resampler.py       # TimeSeriesResampler — orchestration only, no math
├── metrics.py         # precision_phi, recall_phi, f1_phi, sera
├── _relevance.py      # φ: control points + Hermite evaluation  [shared by both]
├── _utility.py        # utility surface u(ŷ, y) for precφ / recφ   [blocked on task M0]
├── _bins.py           # Algorithm 1 — relevance bins
├── _prefs.py          # preference vectors (bias)
├── _sample.py         # under/over selection given a preference vector
├── _synth.py          # SMOTE-style synthetic case generation
├── _validate.py       # shared input validation
│
│   # ---- Layer 2: pipeline (adds pandas, optional [io] extra) ----
└── pipeline/
    ├── __init__.py    # load_series, imbalance_summary, temporal_split, evaluate
    ├── io.py          # CSV + manifest loading [from imbalance_eval]; lag-window kNN imputation (ADR-0010)
    ├── splits.py      # Monte Carlo temporal splits
    └── cli.py         # Layer 3 console script
```

Each private module is independently testable and has no upward dependency on
`resampler.py`. `resampler.py` contains **no arithmetic** — it dispatches. This is the
seam structure the test suite is written against (see `Blueprint/ops/QUALITY_GATES.md`).

**Layering is one-directional and enforced.** `pipeline/` may import Layer 1; nothing in
Layer 1 may import `pipeline`, and no Layer 1 module may import `pandas`. A CI check
asserts both, because this is the boundary that erodes first and it is what keeps
`pip install tsresample` light.

`_relevance.py` is shared by the resampler and the metrics *by design* — it is the single
definition of "rare" in the library, so the two can never drift apart. That shared
definition is the main architectural reason the metrics ship here rather than separately.

---

## 4. The mathematics

Everything in this section is normative. Where the paper is ambiguous, the resolution is
an ADR and is marked as such — an implementer must not resolve an ambiguity by judgement.

### 4.1 Relevance function φ (`_relevance.py`) — ADR-0001, ADR-0002

For `relevance="auto"`, φ is the *extremes* relevance of Ribeiro (2011), with the
conventions of `uba::phi.control(y, method="extremes")` recovered from R's recorded
control points. NaNs in `y` are dropped before fitting.

**Step 1: Tukey hinges** (the five-number summary of R's `fivenum`, Tukey 1977), *not*
type-7 quantiles. With `x = sort(y)`, `n = len(x)` and 1-based indexing:

```
n4 = floor((n + 3) / 2) / 2
d  = [1, n4, (n + 1)/2, n + 1 − n4, n]
h_j = 0.5 · (x[floor(d_j)] + x[ceil(d_j)])        # j = 1..5
Q1, med, Q3 = h_2, h_3, h_4
```

**Step 2: whisker ends.** `IQR = Q3 − Q1`. Fences `LF = Q1 − 1.5·IQR`, `UF = Q3 + 1.5·IQR`.
A case is an outlier iff `y < LF` or `y > UF` (strict). The whisker ends are the most
extreme **non-outlier observations**: `lw = min{y : LF ≤ y ≤ UF}`, `uw = max{…}`. They are
observed values, not the fences themselves.

**Step 3: control points** `(x, φ, φ′)`:

```
low  = (lw, 1, 0)       if any y < lw     else (min(y), 0, 0)
mid  = (med, 0, 0)
high = (uw, 1, 0)       if any y > uw     else (max(y), 0, 0)
```

A side with no outliers has **φ = 0 at its endpoint**. That side has no rare cases. This
is common: every DS10 split and some DS19, DS21–24 splits are one-sided.

**Step 4: interpolation.** A cubic Hermite spline through the three points with
`dydx = [0, 0, 0]` (ADR-0002):

```python
spline = CubicHermiteSpline(x=[x_low, med, x_high], y=[φ_low, 0.0, φ_high], dydx=np.zeros(3))
inside = np.clip(spline(np.clip(y, x_low, x_high)), 0.0, 1.0)
phi = np.where(y <= x_low, φ_low, np.where(y >= x_high, φ_high, inside))
```

At and beyond `[x_low, x_high]` φ is **exactly the endpoint's φ** (1 on a two-sided side,
0 on a one-sided side), not extrapolation of the cubic. Both the `np.where` and the
`[0, 1]` clip are required: scipy's spline evaluated on an array can return `1 − 4e-16` at
the upper knot and `−2.2e-16` near the median. That drops cases out of the `t = 1` and
`t = 0` grid points, and it puts SERA off by up to 0.5 % on DS01/DS09 against gate G0b
(ADR-0014, SERA section).

- **Do not use `PchipInterpolator`** (ADR-0002) or the adjusted boxplot / medcouple
  (ADR-0001).
- **Do not use `np.percentile`** for the quartiles, and do not place control points at the
  fences. v0.7.0 did both and matched R's control points in 4 of 54 splits.

**Degenerate inputs.** If the three control-point `x` values are not strictly increasing
(e.g. `IQR == 0` with no outliers, or `min(y) == med`), R's spline constructor raises. The
library instead sets φ ≡ 0 and emits a `UserWarning`, and `fit_resample` returns its input
unchanged (§4.4 "no-bump rule"). This is a documented robustness deviation, since R
produces no result there. It must be tested.

**Fit scope.** φ is fit on exactly the `y` handed in, never on a wider sample (ADR-0008).

**Verification.** A clean implementation of Steps 1–4 reproduces R's recorded per-split
control points in 85 of 90 reconstructed training splits (18 NA-free datasets × 5
iterations, `Blueprint/tests/fixtures/phi_oracle.json` → `r_splits`). The 5 misses are all DS05,
whose training values sit exactly on the ±0.08 fences. Last-bit precision lost in the CSV
export decides whether they count as outliers, so DS05 is excluded from the exact test and
tracked as an open item (`Blueprint/docs/REPLICATION.md` §2).

### 4.2 Relevance bumps (`_bins.py`) — Algorithm 1, ADR-0012

Bins (R: "bumps") are formed in **value space**, not time.

1. Order the cases by `y` ascending (stable), and evaluate φ on that order.
2. Cut between consecutive cases `i`, `i+1` according to the strategy's crossing rule:
   - **under, smote:** let `s = φ` where `φ ≤ t_R` and `s = −φ` where `φ > t_R`; cut
     where `s_i · s_{i+1} < 0`.
   - **over:** cut where `(φ_i ≥ t_R) ≠ (φ_{i+1} ≥ t_R)`.
3. Each run between cuts is a bump. Its importance is the **mean φ** of its cases.
   - **under, smote:** rare iff mean `> t_R`, normal iff mean `< t_R`.
   - **over:** rare iff mean `≥ t_R`, normal otherwise.

With the default extremes φ this gives 2 bumps (one-sided) or 3 (low-rare, normal,
high-rare). The time order of cases plays **no** part in forming bumps. It enters only
through the bias (§4.3) and SMOTE neighbour choice (§4.5).

**No-bump rule.** If every φ is 0, every φ is 1, or there is no rare bump or no normal
bump, return the input unchanged with a `UserWarning`. R `stop()`s here. The canonical
harness patched it to exactly this no-op, and it was measured inert there (ADR-0011).

### 4.3 Bias / preference vectors (`_prefs.py`) — ADR-0012

Used whenever cases are *drawn* from a bump: under (normal bumps), over (rare bumps) and
smote's undersampling of normal bumps. For a bump of `r` cases, order its cases
**chronologically** (by original time index) and let `j = 1..r` be that rank:

| `bias` | preference `p_j` (before normalisation) |
|---|---|
| `None` | uniform |
| `"temporal"` | `j / r` |
| `"temporal+phi"` | `(j / r) · φ_j` |

The rank is **within the bump**, not the position in the full series. v0.7.0 had this
backwards, and so did its Node 4 brief. Normalise with `p / p.sum()`.

**Degenerate cases.** If `p.sum() == 0`, fall back to uniform. If a draw *without*
replacement asks for more cases than have `p > 0` (possible for `"temporal+phi"` when many
φ are 0), R raises. The library instead takes every positive-probability case, fills the
remainder uniformly from the zero-probability cases, and warns. This is a documented
robustness deviation, and both paths must be tested.

### 4.4 Target counts (`_sample.py`) — ADR-0004, ADR-0006, ADR-0007

Notation: `N` = total cases. `R`, `U` = the rare and normal bumps under the strategy's
classification rule (§4.2). `n_R`, `n_U` = total cases in them. `|B|` = bump size.
`trunc(x) = floor(x)` for `x ≥ 0` (R's `sample(size=)` truncates). `round5(x)` = round to 5
decimals. `round_even(x)` = round half to even (R's `round`).

**Default `"balance"`** (`o is None and u is None`):

| strategy | rare bumps | normal bumps |
|---|---|---|
| `under` | kept whole | `c_B = round5((n_R / |U|) / |B|)`. If `|B| == 1` or `c_B ≥ 1`, keep whole; else draw `trunc(c_B · |B|)` **without** replacement (§4.3 bias) |
| `over` | keep all originals, then `c_B = round5((n_U / |R|) / |B|)` and **append** `trunc(c_B · |B|)` draws **with** replacement (§4.3 bias) | kept whole |
| `smote` | `c_B = B* / |B|` with `B* = round_even(N / #bumps)`, applied to every bump. `c_B > 1` → synthesise (§4.5); `c_B < 1` → draw `trunc(c_B · |B|)` **with** replacement (§4.3 bias); `c_B == 1` or `|B| == 1` → keep | ← same rule |

So under brings the normal side down to the rare total, over brings the rare side up by the
normal total, and only smote aims every bump at `N / #bumps`.

**Explicit percentages** (R's `C.perc` list form):
- `under`: `u` replaces `c_B` for every normal bump.
- `over`: `o` replaces `c_B` for every rare bump: `trunc(o · |B|)` copies are appended.
  `o < 1` is a `ValueError` (R rejects it).
- `smote`: `u` for every normal bump, `o` for every rare bump. R's own list form fails on
  three-bump data. This follows the paper's Algorithm 5 instead (one `u`, one `o`), a
  documented extension.

**Size properties.** `under` never grows the data, `over` never shrinks it and retains
every original case, and `smote` satisfies `|len(out) − N| ≤ 2·#bumps`.

### 4.5 Synthetic case generation (`_synth.py`) — Algorithms 4 / 9 / 13, ADR-0013

For a rare bump `B` with multiplier `c > 1` (§4.4). The target is the last column.

**Case order and the index quirk.** Let `T_y` be the bump's rows in bump order (ascending
`y`, §4.2) and `T_t` the same rows in chronological order. Neighbours are always computed
on `T_t`. With `r_quirks=True` (default), `B` uses `T_y` throughout. `T` and `TPhi` read the
seed and neighbour **values** from `T_y` using indices computed on `T_t`, reproducing R's
ordering mismatch. With `r_quirks=False`, `T_t` is used for both.

**Neighbours.** `k_eff = |B| − 1` if `|B| ≤ k`, else `k`. The `k_eff` nearest by
**unscaled Euclidean** distance over all non-target columns, self excluded, ties broken
by lowest index.

**How many.** `nexs = floor(c − 1)` and `extra = floor(|B| · (c − 1 − nexs))`. Every seed
generates `nexs` cases, then `extra` distinct seeds drawn uniformly **without**
replacement generate one more each. The bump's original cases are kept.

**Neighbour choice** (indices into `T_t`, rows sorted nearest-first):

| `bias` | choice |
|---|---|
| `None` (Alg. 4) | uniform over the `k_eff` |
| `"temporal"` (Alg. 9) | the neighbour with the largest time index (most recent) |
| `"temporal+phi"` (Alg. 13) | `argmax_m (τ_m · φ(y_m))` with `τ_m = (idx_m + 1) / (max_m idx_m + 1)`, `idx` the 0-based time index in `T_t` and `y_m` read from `T_t`; ties → the nearer neighbour |

`τ` is **recency** normalised by the most recent candidate, not nearness rank.

**Interpolation.** One `λ ~ U(0, 1)` per synthetic case, shared by all predictors:
`x_new = x_seed + λ · (x_nn − x_seed)`.

**Target.** With `r_quirks=True`, the weights use only the **last predictor** `a` (the
most recent lag): `d1 = |x_seed[a] − x_new[a]|`, `d2 = |x_nn[a] − x_new[a]|` (R divides both
by the column range, which cancels). If `d1 == d2` then `y_new = (y_seed + y_nn) / 2`,
else `y_new = (d2·y_seed + d1·y_nn) / (d1 + d2)`. This reduces to `y_new = y_seed + λ·(y_nn −
y_seed)`, except the midpoint when `x_seed[a] == x_nn[a]`. With `r_quirks=False`, `d1`, `d2`
are full-vector Euclidean distances over all predictors. If the range of column `a` is 0,
R yields NaN; the library uses the midpoint (documented deviation).

### 4.6 The strategy grid

| | `bias=None` | `bias="temporal"` | `bias="temporal+phi"` |
|---|---|---|---|
| `strategy="under"` | Alg. 2 — `UNDERB` | Alg. 6 — `UNDERT` | Alg. 10 — `UNDERTPhi` |
| `strategy="over"` | Alg. 3 — `OVERB` | Alg. 7 — `OVERT` | Alg. 11 — `OVERTPhi` |
| `strategy="smote"` | Alg. 5 — `SMOTEB` | Alg. 8 — `SMOTET` | Alg. 12 — `SMOTETPhi` |

Right-hand names are the workflow labels used in `Results (Clean)`, and the replication
suite keys off them.

### 4.7 Relevance-based metrics (`metrics.py`, `_utility.py`) — ADR-0009

φ is the same function as §4.1, resolved through the same module. The metrics use a
case-by-case **event** threshold `t_E` with `≥`, which is distinct from the bump rule of §4.2.

**SERA** (Ribeiro & Moniz 2020). Threshold-free — it integrates over *all* cutoffs:

```
SER_t = Σ_{i : φ(y_i) ≥ t} (y_i − ŷ_i)²        t ∈ [0, 1]
SERA  = ∫₀¹ SER_t dt
```

`SER_t` is a right-continuous step function of `t`. **Integrate on a uniform grid**,
`t ∈ {0, step, 2·step, …, 1}` with `step = 0.001` by default (a keyword argument),
trapezoidal rule (`numpy.trapezoid`). φ is §4.1's, including φ = 0 beyond a one-sided
endpoint. Do **not** integrate exactly over the `φ(y_i)` breakpoints: against the recorded
values that is off by ~22 %.

**Relation to the recorded values.** SERA is not one of the paper's R algorithms (the
2017 paper does not report it). The canonical numbers were produced by the harness script
`Results (Clean)/Eval_Metrics Code/SERA Metric/sera_metric.py`. It uses `step = 0.01` and
forces φ = 1 at and beyond both endpoints, even a one-sided φ = 0 endpoint. With those two
settings the recorded SERA is reproduced to 1e-15 (ADR-0009 amendment), so there is no
unexplained residual. The library keeps `step = 0.001`, the default of the SERA reference
implementations (also used by the MetaIR `sera.py`), and the correct one-sided φ. Gate G0b
checks against the recorded values with `step=0.01` on two-sided splits only
(`Blueprint/docs/REPLICATION.md` §3.4). On one-sided splits (DS10, some DS19, DS21–24) the recorded
SERA carries the harness's φ bug and is not a valid oracle.

**Utility-based precision and recall** (Torgo & Ribeiro 2009; Ribeiro 2011 §4), with the
conventions of `uba::util` as confirmed against the recorded canonical values:

```
recall_φ    = Σ_{i : φ(y_i)  ≥ t_E} |1 + u_i| / Σ_{i : φ(y_i)  ≥ t_E} |1 + φ(y_i)|
precision_φ = Σ_{i : φ(ŷ_i) ≥ t_E} |1 + u_i| / Σ_{i : φ(ŷ_i) ≥ t_E} |1 + φ(ŷ_i)|
F_β,φ       = (1 + β²)·precision_φ·recall_φ / (β²·precision_φ + recall_φ)
```

- `t_E` is the event threshold (`rel_threshold`, default 0.9), applied case by case with
  **`≥`**.
- **Empty selection:** if no case clears `t_E`, that measure is **`1e-5`**, not 0 and not
  NaN, and no warning. R's recorded output shows exactly this: `prec = 1e-05`,
  `F1 = 1.99996e-05`. A `0.0` here would fail G0b on every such row.
- `F_β,φ = 0` if either measure is exactly 0.
- φ for the metrics is the φ fit on the **training** target, the same φ the resampler
  used (ADR-0008). In the harness that is `phi.control(train_y)` before resampling.

`u_i = U(ŷ_i, y_i)` is the utility of the prediction: the benefit/cost surface of Ribeiro
(2011) §4 over the relevance "bumps" of φ, with `p = 0.5`.

> **Status of `U` (M0 closed, 2026-09-22).** `U` is specified in **ADR-0014**. It was
> derived cleanly from Ribeiro (2011) §3.3–3.4 (Defs 3.10–3.20, Algorithm 3.2 with the
> initialisation the oracle pins), and it reproduces recorded prec/rec/F1 within 1e-6 on
> 62,088 of 62,400 splits (worst 8.7e-15). **Known residual:** a φ with control-point values
> (1, 0, 0) (DS19 only, 312 splits) does not match R; the docstrings must say so. The φ used
> by the metrics must return the exact endpoint value at and beyond the knots, and must be
> clipped to [0, 1] (ADR-0014, SERA section). `_utility.py` is unblocked.

---

## 5. Validation strategy

Five gates, cheapest first. A gate that fails blocks every downstream node — see
`Blueprint/ops/TASK_GRAPH.md`. Full procedures in `Blueprint/docs/REPLICATION.md`.

| Gate | What it proves | Cost | Data needed |
|---|---|---|---|
| **G0 — φ oracle** | φ control points match R's recorded per-split control points; `%Rare` matches the paper's Table 1 | seconds | `Blueprint/tests/fixtures/phi_oracle.json` (committed) |
| **G0b — metric oracle** | `sera` / `precφ` / `recφ` / `F1φ` match recorded R values on held-out predictions | seconds | `Blueprint/tests/fixtures/metric_oracle.json` (produced by task M0) |
| **G1 — properties** | Invariants hold for arbitrary inputs (Hypothesis) | seconds | none |
| **G2 — sklearn protocol** | `check_estimator` passes; `clone`/`get_params` round-trip | seconds | none |
| **G3 — algorithmic** | Bump structure, target counts, bias vectors, neighbour and interpolation rules match hand-worked examples of §4.2–4.5, including both `r_quirks` settings | seconds | hand-computed fixtures |
| **G4 — replication** | End-to-end `precφ`/`recφ`/`F1φ`/`sera` distributions overlap the recorded R results across 20 datasets × workflows × 50 iterations | minutes | external, opt-in |

**G0 is the load-bearing gate.** It is the only one that can catch a *wrong formula* as
opposed to an inconsistent implementation, and it already caught two (ADR-0001, ADR-0002).
It is committed, offline, and runs in CI. v0.7.0's G0 fixture was itself generated from
the SPEC's formula, and so could not fail (§0 #13). The rebuilt fixture holds R's recorded
numbers.

**Anti-tautology rule.** No test may compute its expected value the way the implementation
does. Expected values come from `phi_oracle.json`, from the paper, or from a worked
example written out by hand in the test file with its arithmetic shown in a comment.
A test that reimplements `_relevance.py` inside the test is not a test.

**Property tests (G1)** — the invariants that must hold for *any* valid input:
- `embed`: no row contains a value from time ≥ its target's time; shapes are exactly
  `(n − k − horizon, k + 1 + m)`; round-trips to the original series for `k=1, horizon=1`.
- φ: range `[0, 1]`; φ(med) = 0; each endpoint's φ is 1 if that side has outliers and 0
  otherwise; constant outside the endpoints; monotone non-increasing on `[x_low, med]` and
  non-decreasing on `[med, x_high]`.
- `fit_resample`: output columns == input columns; `strategy="under"` never grows the
  data; `strategy="over"` never shrinks it and every original case survives it;
  `strategy="smote"` satisfies `|len(out) − N| ≤ 2·#bumps`; identical `random_state` ⇒
  identical output arrays.
- Bumps: partition the **value-sorted** order exactly, no gaps, no overlaps; each bump is
  contiguous in `y`.

---

## 6. Packaging & provenance

**Build.** `hatchling`, `src/` layout, `pyproject.toml` only — no `setup.py`, no
`setup.cfg`. `requires-python = ">=3.10"`. Version is single-sourced from
`src/tsresample/__init__.py::__version__`.

**Typing.** Fully annotated, `py.typed` shipped, `mypy --strict` clean on `src/`.

**Lint/format.** `ruff` for both, one config block in `pyproject.toml`.

**CI.** GitHub Actions: matrix over CPython 3.10–3.13 × {ubuntu, macos, windows}. Gates
G0–G3 on every push. G4 nightly and on release tags only. Coverage floor **90 %** on
`src/tsresample/` — one number, stated once, here. Any other number in any other file is
a bug in that file.

**Release.** PyPI Trusted Publishing (OIDC), no long-lived API token in repo secrets.
Tag-triggered. `sdist` + pure-Python `wheel`.

**Name.** `tsresample` — verified available on PyPI (HTTP 404 on the project URL).

**Licence.** MIT.

### 6.1 Provenance — the cleanroom rule

The algorithms come from the **paper** (Moniz, Branco & Torgo 2017) and from **numerical
artifacts we produced ourselves**. They do not come from reading GPL source.

`Blueprint/docs/PROVENANCE.md` holds the full quarantine list and the audit trail. The operative
rule for implementers is in `CLAUDE.md` rule 3 and is enforced by path: certain
directories must never be opened by an agent writing `src/`. The φ convention in §4.1 was
recovered from `mc.*_phi_ctrl.csv` — *output data from our own experiment runs* — not from
`uba`'s `phi.c` or `phi.R`, both of which remain unread. That distinction is what makes
the MIT licence defensible, and it must be preserved.

**v0.8.0 caveat.** The v0.8.0 corrections were found partly through a user-authorised
audit of a colleague's GPL-derived Python port (`Blueprint/docs/PROVENANCE.md` §5). The SPEC states
*behaviour* only, the kind of fact an oracle probe establishes, and every [AUDIT] item has
a gate that verifies it against recorded R output. Implementation sessions work from this
SPEC and the ADRs and must never open the port.

---

## 7. Roadmap

| Version | Contents | Gate to ship |
|---|---|---|
| 0.1.0 | `embed` + φ + bins | G0, G1, G3 green |
| 0.2.0 | `under` + `over`, all three biases | G0–G3 green |
| 0.3.0 | `smote`, all three biases — full grid | G0–G3 green |
| 0.4.0 | `metrics`: `sera`, `precision_phi`, `recall_phi`, `f1_phi` | G0b green |
| 0.5.0 | `pipeline` + CLI — end-to-end path | G0–G3 green |
| 0.9.0 | Docs, examples, `MIGRATION.md`, API freeze | G4 green |
| 1.0.0 | PyPI release | G4 green + a downstream user |

Metrics land at 0.4.0 rather than earlier because task **M0** is research with an unknown
completion time, and the resampler grid is not blocked on it. If M0 stalls, 0.3.0 still
ships something useful.

**Kill criterion.** If G4 cannot be brought within its stated tolerance (see
`Blueprint/docs/REPLICATION.md` §5) after the [AUDIT] conventions of §0.1 (#14–#19) have each been
re-checked against the oracle, the library ships as *"inspired by"* rather than *"replicates"*, the README
says so plainly, and the discrepancy is documented. Shipping a library that silently
disagrees with the paper it cites is the one outcome worse than shipping nothing.

---

## 8. References

1. Moniz, N., Branco, P., Torgo, L. (2017). *Resampling strategies for imbalanced time
   series forecasting.* International Journal of Data Science and Analytics 3(3), 161–181.
2. Branco, P., Torgo, L., Ribeiro, R. (2017). *SMOGN: a pre-processing approach for
   imbalanced regression.* LIDTA/PMLR 74, 36–50.
3. Torgo, L., Ribeiro, R. (2009). *Precision and recall for regression.* Discovery Science.
4. Ribeiro, R. (2011). *Utility-based Regression.* PhD thesis, University of Porto.
   §3.4.1 defines φ by interpolating control points ⟨y_k, φ(y_k), φ′(y_k)⟩; §4 defines the
   utility-based precision and recall of §4.7.
4b. Ribeiro, R., Moniz, N. (2020). *Imbalanced regression and extreme value prediction.*
   Machine Learning 109, 1803–1835. Defines SERA. Note this paper *does* adopt the
   adjusted boxplot — the 2017 work this library replicates does not (ADR-0001).
4c. Dougherty, R., Edelman, A., Hyman, J. (1989). *Nonnegativity-, monotonicity-, or
   convexity-preserving cubic and quintic Hermite interpolation.* Math. Comp. 52(186),
   471–494. The interpolation reference Ribeiro (2011) cites.
5. Takens, F. (1981). *Detecting strange attractors in turbulence.* Springer LNM 898.
6. Fritsch, F., Carlson, R. (1980). *Monotone piecewise cubic interpolation.*
   SIAM J. Numer. Anal. 17(2), 238–246.
7. Hubert, M., Vandervieren, E. (2008). *An adjusted boxplot for skewed distributions.*
   Computational Statistics & Data Analysis 52(12), 5186–5201.
   **Cited for the record only — ADR-0001 establishes it is *not* used.**
