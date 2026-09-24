"""tsresample.pipeline public seams (SPEC §2.4, ADR-0010)."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tsresample import _relevance, embed
from tsresample.pipeline import (
    evaluate,
    imbalance_summary,
    load_series,
    temporal_split,
)

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


def test_imbalance_summary_counts_rare_cases_inclusively() -> None:
    # y = 1..9, 100: phi = 1 only at 9 and 100 (control points (1,0),(5.5,0),(9,1),
    # worked in test_algorithms) -> n_rare 2, n_normal 8, IR 2/8, %Rare 20.
    y = np.array([4, 1, 9, 2, 100, 3, 8, 5, 7, 6], dtype=float)
    assert imbalance_summary(y) == {
        "N": 10,
        "n_normal": 8,
        "n_rare": 2,
        "IR": 0.25,
        "pct_rare": 20.0,
    }


def test_imbalance_summary_refuses_a_single_value() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        imbalance_summary([3.0])


def test_imbalance_summary_matches_paper_table_1() -> None:
    # %Rare on the embedded target of the 18 NA-free datasets vs the paper's
    # Table 1 (phi_oracle.json): MAE 0.169 pp, same as gate G0 Test B (SPEC §4.1).
    import json

    oracle = json.loads(
        (DATA.parent.parent / "tests" / "fixtures" / "phi_oracle.json").read_text()
    )["datasets"]
    err = []
    for ds, d in oracle.items():
        if not d.get("r_splits"):
            continue
        s = load_series(next(DATA.glob(f"{ds}_*.csv")), target="target", impute=None)
        _, y = embed(s, k=8)
        err.append(abs(imbalance_summary(y)["pct_rare"] - d["paper_pct_rare"]))
    assert len(err) == 18 and sum(err) / 18 <= 0.25


def _ds01_embedded() -> tuple[np.ndarray, np.ndarray]:
    s = load_series(next(DATA.glob("DS01_*.csv")), target="target", impute=None)
    return embed(s[:300], k=8)


def test_evaluate_returns_one_row_per_strategy_split_metric() -> None:
    from sklearn.linear_model import LinearRegression

    X, y = _ds01_embedded()
    splitter = lambda X, y: temporal_split(X, y, n_reps=3, random_state=0)  # noqa: E731
    df = evaluate(LinearRegression(), X, y, splitter=splitter)
    assert list(df.columns) == ["strategy", "split", "metric", "value"]
    # baseline + the 9 cells of SPEC §4.6, 3 splits, 4 metrics
    assert len(df) == 10 * 3 * 4
    assert set(df["strategy"]) == {
        "baseline",
        "UNDERB",
        "UNDERT",
        "UNDERTPhi",
        "OVERB",
        "OVERT",
        "OVERTPhi",
        "SMOTEB",
        "SMOTET",
        "SMOTETPhi",
    }
    assert set(df["metric"]) == {"precision_phi", "recall_phi", "f1_phi", "sera"}
    assert df.groupby(["strategy", "split", "metric"]).size().eq(1).all()
    assert np.isfinite(df["value"]).all()


def test_evaluate_refits_phi_on_each_training_window() -> None:
    from sklearn.linear_model import LinearRegression

    X, y = _ds01_embedded()
    splits = list(temporal_split(X, y, n_reps=2, random_state=1))
    seen: list[np.ndarray] = []

    def spy(y_true: np.ndarray, y_pred: np.ndarray, *, relevance: np.ndarray) -> float:
        seen.append(relevance)
        return 0.0

    evaluate(
        LinearRegression(),
        X,
        y,
        strategies={"baseline": None},
        metrics={"spy": spy},
        splitter=lambda X, y: iter(splits),
    )
    # phi comes from the training target, never from the test target (ADR-0008)
    for cp, (_, y_tr, _, y_te) in zip(seen, splits, strict=True):
        np.testing.assert_array_equal(cp, _relevance.control_points(y_tr))
        assert not np.array_equal(cp, _relevance.control_points(y_te))


def test_load_series_sorts_mixed_date_formats(tmp_path: Path) -> None:
    # Half-hourly exports write midnight as a bare date (DS21-24).
    rows = [
        ("1999-01-01 01:00:00", "3"),
        ("1999-01-01", "1"),
        ("1999-01-01 00:30:00", "2"),
    ]
    np.testing.assert_array_equal(
        load_series(_csv(tmp_path, rows), target="value", date_col="date"), [1, 2, 3]
    )
