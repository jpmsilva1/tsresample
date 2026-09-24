"""tsresample.pipeline public seams (SPEC §2.4, ADR-0010)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tsresample import _relevance, embed
from tsresample.pipeline import load_series, temporal_split

DATA = Path(__file__).resolve().parents[1] / "Blueprint" / "replication" / "datasets"


def _csv(tmp_path: Path, rows: list[tuple[str, str]], header=("date", "value")) -> Path:
    p = tmp_path / "s.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return p


def test_load_series_sorts_by_date_then_differences(tmp_path: Path) -> None:
    # Out of order on disk; by date the values are 1, 4, 9, 16 -> diff 3, 5, 7.
    p = _csv(
        tmp_path,
        [
            ("2020-01-03", "9"),
            ("2020-01-01", "1"),
            ("2020-01-04", "16"),
            ("2020-01-02", "4"),
        ],
    )
    np.testing.assert_array_equal(
        load_series(p, target="value", date_col="date"), [1, 4, 9, 16]
    )
    np.testing.assert_array_equal(
        load_series(p, target="value", date_col="date", diff=True), [3, 5, 7]
    )


def test_round_trip_from_csv_to_supervised_matrix(tmp_path: Path) -> None:
    # 0..5 in order; embed(k=1): [1, 0] -> 2, [2, 1] -> 3, [3, 2] -> 4, [4, 3] -> 5.
    p = _csv(tmp_path, [(f"2020-01-0{i + 1}", str(i)) for i in range(6)])
    X, y = embed(load_series(p, target="value", date_col="date"), k=1)
    np.testing.assert_array_equal(X, [[1, 0], [2, 1], [3, 2], [4, 3]])
    np.testing.assert_array_equal(y, [2, 3, 4, 5])


def test_missing_target_column_lists_the_available_ones(tmp_path: Path) -> None:
    p = _csv(tmp_path, [("2020-01-01", "1")])
    with pytest.raises(ValueError, match=r"'temp'.*available: \['date', 'value'\]"):
        load_series(p, target="temp")


def _pct_rare(s: np.ndarray) -> float:
    y = s[9:]  # create.data(ts, 10) target (ADR-0005)
    return 100 * float(np.mean(_relevance.phi(y, _relevance.control_points(y)) >= 0.9))


def test_knn_imputation_reproduces_ds12_pct_rare() -> None:
    # ADR-0010 amendment: lag-window kNN gives 10.99 on DS12 (paper 11.0);
    # the column mean (KNNImputer on one column) gives 14.65.
    s = load_series(
        next(DATA.glob("DS12_*.csv")), target="target", date_col="time_index"
    )
    assert not np.isnan(s).any() and len(s) == 1456
    assert round(_pct_rare(s), 2) == 10.99


def test_leading_gaps_without_observed_lags_are_dropped_with_warning(
    tmp_path: Path,
) -> None:
    rows = [(f"2020-01-{i + 1:02d}", "" if i < 2 else str(i % 5)) for i in range(30)]
    with pytest.warns(UserWarning, match="dropped 2 leading"):
        s = load_series(_csv(tmp_path, rows), target="value", date_col="date")
    assert len(s) == 28 and not np.isnan(s).any()


def test_a_gap_is_filled_from_donors_with_a_matching_lag_window(tmp_path: Path) -> None:
    # Period-3 pattern 0, 1, 2, ...; value 40 (true 40 % 3 = 1) is missing. Its lag
    # window y31..y39 repeats exactly in more than 10 fully observed donor rows,
    # all with target 1, at distance 0 (weight e^0) -> the fill is exactly 1.
    vals = [str(i % 3) for i in range(60)]
    vals[40] = ""
    s = load_series(
        _csv(
            tmp_path,
            [
                (f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", v)
                for i, v in enumerate(vals)
            ],
        ),
        target="value",
        date_col="date",
    )
    assert s[40] == pytest.approx(1.0, abs=1e-12)


def test_impute_none_raises_and_drop_removes_gaps(tmp_path: Path) -> None:
    p = _csv(tmp_path, [("2020-01-01", "1"), ("2020-01-02", ""), ("2020-01-03", "3")])
    with pytest.raises(ValueError, match="1 missing"):
        load_series(p, target="value", impute=None)
    np.testing.assert_array_equal(load_series(p, target="value", impute="drop"), [1, 3])


def test_temporal_split_yields_contiguous_train_then_test_windows() -> None:
    # n = 721 (DS01 embedded): train trunc(0.5 * 721) = 360, test trunc(180.25) = 180,
    # as in the recorded splits (phi_oracle train_slice, metric_oracle test length).
    t = np.arange(721.0)
    X = np.column_stack([t, -t])
    splits = list(temporal_split(X, t, n_reps=50, random_state=0))
    assert len(splits) == 50
    for X_tr, y_tr, _, y_te in splits:
        assert len(y_tr) == 360 and len(y_te) == 180
        assert np.all(np.diff(y_tr) == 1) and np.all(np.diff(y_te) == 1)  # contiguous
        assert y_te[0] == y_tr[-1] + 1  # test right after train, never shuffled
        np.testing.assert_array_equal(X_tr[:, 0], y_tr)
    starts = {s[1][0] for s in splits}
    assert len(starts) > 1  # Monte Carlo: windows move between repetitions


def test_temporal_split_is_reproducible_and_validates_sizes() -> None:
    t = np.arange(100.0)
    X = t[:, None]
    a = [s[1][0] for s in temporal_split(X, t, n_reps=5, random_state=3)]
    b = [s[1][0] for s in temporal_split(X, t, n_reps=5, random_state=3)]
    assert a == b
    with pytest.raises(ValueError, match="train_size \\+ test_size"):
        next(temporal_split(X, t, train_size=0.8, test_size=0.3))
    with pytest.raises(ValueError, match="at least 1"):
        next(temporal_split(X, t, train_size=0.001))


def test_knn_needs_one_gap_free_window(tmp_path: Path) -> None:
    rows = [(f"2020-01-{i + 1:02d}", "" if i % 5 == 0 else "1") for i in range(20)]
    with pytest.raises(ValueError, match="gap-free window"):
        load_series(_csv(tmp_path, rows), target="value")
