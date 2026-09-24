"""embed(): worked examples, the create.data mapping, and its ValueErrors."""

import numpy as np
import pytest

from tsresample import embed


def test_rows_are_most_recent_lag_first_with_target_horizon_ahead() -> None:
    # series = 0..9, k = 2, horizon = 1: rows t = 2..8 (first k and last horizon
    # dropped) -> n - k - horizon = 7 rows of k + 1 = 3 columns.
    # t = 2: X = [s2, s1, s0] = [2, 1, 0], y = s3 = 3; t = 8: [8, 7, 6], y = 9.
    X, y = embed(np.arange(10.0), k=2)
    assert X.shape == (7, 3)
    np.testing.assert_array_equal(X[0], [2, 1, 0])
    np.testing.assert_array_equal(X[-1], [8, 7, 6])
    np.testing.assert_array_equal(y, np.arange(3.0, 10.0))


def test_paper_create_data_10_is_k_8_horizon_1() -> None:
    # R: embed(ts, 10) gives 10 columns and n - 9 rows; create.data reverses them
    # and takes the last (current) value as target. On ts = 0..19 that is 11 rows,
    # first row predictors y0..y8 (9 of them), target y9; targets are ts[9:].
    # (ADR-0005 amendment: the mapping is k = m - 2, not m - 1.)
    s = np.arange(20.0)
    X, y = embed(s, k=8, horizon=1)
    assert X.shape == (11, 9)
    np.testing.assert_array_equal(X[0], np.arange(8.0, -1.0, -1.0))
    np.testing.assert_array_equal(y, s[9:])


def test_exog_is_contemporaneous_and_appended_right() -> None:
    # exog[t] joins row t. k = 1: rows t = 1..3 of series 0..4.
    # exog column = 10 * t -> [10, 20, 30] beside lag block [[1,0],[2,1],[3,2]].
    X, _ = embed(np.arange(5.0), k=1, exog=10.0 * np.arange(5.0)[:, None])
    np.testing.assert_array_equal(X, [[1, 0, 10], [2, 1, 20], [3, 2, 30]])


@pytest.mark.parametrize(
    ("series", "kwargs", "match"),
    [
        (np.zeros((4, 2)), {"k": 1}, "1-D"),
        ([0.0, np.nan, 2.0, 3.0], {"k": 1}, "NaN or inf"),
        ([0.0, np.inf, 2.0, 3.0], {"k": 1}, "NaN or inf"),
        (np.arange(5.0), {"k": 0}, "k >= 1"),
        (np.arange(5.0), {"k": 1, "horizon": 0}, "horizon >= 1"),
        (np.arange(5.0), {"k": 1, "exog": np.zeros((4, 1))}, "exog has 4 rows"),
        (np.arange(3.0), {"k": 2}, "too short"),
    ],
)
def test_invalid_input_raises_value_error_naming_the_problem(
    series: object, kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        embed(series, **kwargs)  # type: ignore[arg-type]
