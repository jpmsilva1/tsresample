import numpy as np
import pandas as pd
import pytest

from tsresamp.data import (DEFAULT_RDATA, complete_cases, create_data, default_sizes,
                           get_dataset, handle_na, knn_imputation, load_data, load_rdata)


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_load_24_series(data):
    assert len(data) == 24
    sizes = [len(s) for s in data]
    assert sizes[:4] == [730] * 4
    assert sizes[4:8] == [17378] * 4
    assert sizes[8] == 1095
    assert sizes[9:13] == [1456] * 4
    assert sizes[13:20] == [536] * 7
    assert sizes[20:22] == [239602] * 2
    assert sizes[22:24] == [51208] * 2


def test_index_is_datetime_and_increasing(data):
    for s in data:
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.index.is_monotonic_increasing


def test_known_missing_values(data):
    nans = {s.attrs["dataset"]: int(np.isnan(s.values).sum()) for s in data}
    assert nans[12] == 197 and nans[13] == 201 and nans[23] == 327 and nans[24] == 33
    for i, c in nans.items():
        if i not in (12, 13, 23, 24):
            assert c == 0


def test_daily_series_have_midnight_index(data):
    s = data[0]
    assert (s.index == s.index.normalize()).all()
    assert s.index[0] == pd.Timestamp("2011-01-02")


def test_pickle_matches_rdata(data):
    raw = load_rdata(DEFAULT_RDATA)
    for a, b in zip(raw, data):
        assert np.allclose(a.values, b.values, equal_nan=True)
        assert (a.index == b.index).all()


def test_create_data_embed():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    ts = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    df = create_data(ts, 3)
    # R: embed(ts,3)[,3:1] -> rows (1,2,3),(2,3,4),(3,4,5),(4,5,6); index drops first 2
    assert list(df.columns) == ["V1", "V2", "V3"]
    assert df.shape == (4, 3)
    assert (df.index == idx[2:]).all()
    np.testing.assert_array_equal(df.values, [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    assert (df["V3"].values == ts.values[2:]).all()


def test_create_data_real(data):
    ds = create_data(get_dataset(1, data), 10)
    assert ds.shape == (730 - 9, 10)
    assert ds.columns[-1] == "V10"
    assert ds.index[0] == data[0].index[9]


def test_default_sizes():
    assert default_sizes(1) == (0.5, 0.25)
    assert default_sizes(21) == (0.10, 0.05)
    assert default_sizes(22) == (0.10, 0.05)
    assert default_sizes(23) == (0.20, 0.10)
    assert default_sizes(24) == (0.20, 0.10)


def test_complete_cases_and_knn():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    X[:, 2] = X[:, 0] + X[:, 1]  # make column 2 predictable
    df = pd.DataFrame(X, columns=["a", "b", "c"], index=idx)
    df.iloc[3, 2] = np.nan
    df.iloc[7, 0] = np.nan
    cc = complete_cases(df)
    assert len(cc) == 28 and not cc.isna().any().any()
    imp = knn_imputation(df, k=5)
    assert imp.shape == df.shape and not imp.isna().any().any()
    # untouched values are preserved
    mask = ~df.isna()
    assert np.allclose(imp.values[mask.values], df.values[mask.values])
    # imputed values are within the range of the column
    assert df["c"].min() <= imp.iloc[3, 2] <= df["c"].max()
    assert handle_na(df, "none").isna().sum().sum() == 2
    assert handle_na(df, "knn").isna().sum().sum() == 0


def test_knn_imputation_real(data):
    ds = create_data(get_dataset(12, data), 10)
    assert ds.isna().any().any()
    imp = knn_imputation(ds)
    assert not imp.isna().any().any()
    assert imp.shape == ds.shape
