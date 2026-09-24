"""TimeSeriesResampler.fit_resample: the public seam (SPEC §2.2, §4.6)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tsresample import TimeSeriesResampler, embed

DATA = Path(__file__).resolve().parents[1] / "Blueprint" / "replication" / "datasets"


def _ds01() -> tuple[np.ndarray, np.ndarray]:
    path = next(DATA.glob("DS01_*.csv"))
    with path.open(newline="", encoding="utf-8") as f:
        series = [float(r["target"]) for r in csv.DictReader(f)]
    return embed(series[:400], k=8)  # the paper's create.data(ts, 10) (ADR-0005)


GRID = [
    (s, b)
    for s in ("under", "over", "smote")
    for b in (None, "temporal", "temporal+phi")
]


@pytest.mark.parametrize(("strategy", "bias"), GRID)
def test_every_strategy_bias_cell_resamples_real_data(
    strategy: str, bias: str | None
) -> None:
    X, y = _ds01()
    Xr, yr = TimeSeriesResampler(strategy, bias, random_state=0).fit_resample(X, y)  # type: ignore[arg-type]
    assert Xr.shape == (len(yr), X.shape[1]) and np.isfinite(Xr).all()
    rows = {tuple(r) for r in X}
    if strategy == "under":
        assert len(yr) < len(y) and {tuple(r) for r in Xr} <= rows
    elif strategy == "over":
        assert len(yr) > len(y) and rows <= {tuple(r) for r in Xr}
    else:
        assert len({tuple(r) for r in Xr} - rows) > 0  # synthetic cases exist


def _toy(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Column 0 is the time index; the last 6 cases are rare (phi = 1).
    t = np.arange(float(n))
    X = np.column_stack([t, np.sin(t)])
    phi = np.where(t >= n - 6, 1.0, 0.1)  # 0.1: phi = 0 never cuts the sign rule
    return X, t.copy(), phi


@pytest.mark.parametrize("strategy", ["under", "over"])
def test_output_rows_are_in_time_order(strategy: str) -> None:
    X, y, phi = _toy()
    Xr, _ = TimeSeriesResampler(strategy, relevance=phi, random_state=1).fit_resample(
        X, y
    )  # type: ignore[arg-type]
    assert np.all(np.diff(Xr[:, 0]) >= 0)


def test_synthetic_cases_follow_their_seed() -> None:
    X, y, phi = _toy()
    Xr, _ = TimeSeriesResampler(
        "smote", k=2, relevance=phi, random_state=1
    ).fit_resample(X, y)
    synthetic = Xr[:, 0] != np.round(Xr[:, 0])
    assert synthetic.any()
    originals = Xr[~synthetic, 0]
    assert np.all(np.diff(originals) >= 0)
    # Each synthetic run sits after an original from the rare tail (t >= 34), and
    # every synthetic value lies between rare originals.
    for i in np.flatnonzero(synthetic):
        j = i - 1
        while synthetic[j]:
            j -= 1
        assert Xr[j, 0] >= 34 and 34 <= Xr[i, 0] <= 39


def test_no_rare_bump_returns_input_unchanged_with_warning() -> None:
    X, y, _ = _toy()
    with pytest.warns(UserWarning, match="no rare bump"):
        Xr, yr = TimeSeriesResampler("under", relevance=np.zeros(40)).fit_resample(X, y)
    np.testing.assert_array_equal(Xr, X)
    np.testing.assert_array_equal(yr, y)


def test_relevance_callable_is_called_on_y() -> None:
    X, y, phi = _toy()
    seen = []

    def f(v: np.ndarray) -> np.ndarray:
        seen.append(v)
        return phi

    Xr, _ = TimeSeriesResampler("under", relevance=f, random_state=0).fit_resample(X, y)
    np.testing.assert_array_equal(seen[0], y)
    # normal bump |34| -> c = round5(6 / 34) = 0.17647; trunc(5.99998) = 5; + 6 rare
    assert len(Xr) == 11


@pytest.mark.parametrize(
    ("kwargs", "X", "y", "match"),
    [
        ({"strategy": "down"}, np.zeros((5, 2)), np.arange(5.0), "strategy"),
        ({"bias": "recent"}, np.zeros((5, 2)), np.arange(5.0), "bias"),
        ({"k": 0}, np.zeros((5, 2)), np.arange(5.0), "k: expected"),
        ({"relevance": "extremes"}, np.zeros((5, 2)), np.arange(5.0), "relevance"),
        ({"relevance": np.ones(4)}, np.zeros((5, 2)), np.arange(5.0), "shape"),
        ({"relevance": np.full(5, 2.0)}, np.zeros((5, 2)), np.arange(5.0), r"\[0, 1\]"),
        ({}, np.zeros(5), np.arange(5.0), "X must be 2-D"),
        ({}, np.zeros((5, 2)), np.zeros((5, 1)), "y must be 1-D"),
        ({}, np.zeros((4, 2)), np.arange(5.0), "4 rows but y has 5"),
        ({}, np.zeros((1, 2)), np.arange(1.0), "at least 2"),
        ({}, np.full((5, 2), np.nan), np.arange(5.0), "finite"),
    ],
)
def test_invalid_input_raises_in_fit_resample(
    kwargs: dict, X: np.ndarray, y: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TimeSeriesResampler(**kwargs).fit_resample(X, y)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strategy": "over", "o": 0.1}, "o >= 1"),
        ({"strategy": "under", "o": 2.0}, "o applies to"),
        ({"strategy": "over", "u": 0.5}, "u applies to"),
        ({"strategy": "smote", "u": -1.0}, "u >= 0"),
    ],
)
def test_o_and_u_are_validated_even_when_nothing_would_be_resampled(
    kwargs: dict, match: str
) -> None:
    X, y = np.zeros((10, 2)), np.ones(10)  # constant y: no bumps at all
    with pytest.raises(ValueError, match=match):
        TimeSeriesResampler(**kwargs).fit_resample(X, y)
