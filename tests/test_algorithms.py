"""G3: algorithmic worked examples. Expected values are hand-derived in comments."""

import numpy as np
import pytest

from tsresample import _relevance


def test_control_points_use_tukey_hinges_and_whisker_ends() -> None:
    # y = 1..9, 100 (n = 10). n4 = floor(13/2)/2 = 3; d = [1, 3, 5.5, 8, 10]
    # Q1 = x[3] = 3, med = (x[5] + x[6])/2 = 5.5, Q3 = x[8] = 8, IQR = 5
    # fences: LF = 3 - 7.5 = -4.5, UF = 8 + 7.5 = 15.5 -> only 100 is an outlier
    # low side: whisker end 1 = min(y), nothing below it -> (min, phi=0)
    # high side: whisker end 9, 100 lies beyond it -> (9, phi=1)
    y = np.array([4, 1, 9, 2, 100, 3, 8, 5, 7, 6], dtype=float)
    np.testing.assert_array_equal(
        _relevance.control_points(y), [[1.0, 0.0], [5.5, 0.0], [9.0, 1.0]]
    )


def test_phi_is_smoothstep_between_knots_and_exact_constant_beyond() -> None:
    # cp = (1, 0), (5.5, 0), (9, 1). On [5.5, 9], dydx = 0 at both knots makes phi
    # the smoothstep 3t^2 - 2t^3 with t = (y - 5.5) / 3.5 (ADR-0002):
    #   y = 7.25  -> t = 0.5  -> 0.75 - 0.25 = 0.5
    #   y = 8.125 -> t = 0.75 -> 1.6875 - 0.84375 = 0.84375
    # At/beyond the knots phi is exactly the endpoint value (SPEC §4.1 Step 4).
    cp = np.array([[1.0, 0.0], [5.5, 0.0], [9.0, 1.0]])
    y = np.array([-3.0, 1.0, 3.0, 5.5, 7.25, 8.125, 9.0, 100.0])
    phi = _relevance.phi(y, cp)
    np.testing.assert_allclose(phi[[4, 5]], [0.5, 0.84375], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(phi[[0, 1, 2, 3, 6, 7]], [0, 0, 0, 0, 1, 1])


def test_degenerate_control_points_give_zero_phi_with_warning() -> None:
    # y = 2, 2, 2, 2, 5: hinges Q1 = med = Q3 = 2, IQR = 0, so 5 is an outlier and
    # the whisker ends are both 2. Low side has nothing below 2 -> (2, 0); high
    # side -> (2, 1). x = (2, 2, 2) is not strictly increasing: R's spline raises;
    # we return phi = 0 everywhere and warn (SPEC §4.1 "Degenerate inputs").
    y = np.array([2.0, 2.0, 5.0, 2.0, 2.0])
    cp = _relevance.control_points(y)
    np.testing.assert_array_equal(cp, [[2, 0], [2, 0], [2, 1]])
    with pytest.warns(UserWarning, match="strictly increasing"):
        np.testing.assert_array_equal(_relevance.phi(y, cp), np.zeros(5))


def test_control_points_drop_nans_and_reject_all_nan_input() -> None:
    # Same worked example as above with NaNs mixed in: NaNs are dropped first.
    y = np.array([4, 1, np.nan, 9, 2, 100, 3, 8, 5, 7, 6, np.nan])
    np.testing.assert_array_equal(
        _relevance.control_points(y), [[1.0, 0.0], [5.5, 0.0], [9.0, 1.0]]
    )
    with pytest.raises(ValueError, match="no non-NaN values"):
        _relevance.control_points([np.nan, np.nan])
