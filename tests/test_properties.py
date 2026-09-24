"""G1: properties that must hold for every input (Hypothesis)."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tsresample import embed


@given(
    n=st.integers(3, 60),
    k=st.integers(1, 10),
    horizon=st.integers(1, 5),
    m=st.integers(0, 3),
)
def test_embed_never_leaks_values_at_or_after_target_time(
    n: int, k: int, horizon: int, m: int
) -> None:
    # Each value *is* its time index, so leakage shows up as X >= y's time.
    t = np.arange(float(n))
    rows = n - k - horizon
    if rows < 1:
        return
    exog = np.repeat(t[:, None], m, axis=1) if m else None
    X, y = embed(t, k=k, horizon=horizon, exog=exog)
    assert X.shape == (rows, k + 1 + m)
    assert y.shape == (rows,)
    assert np.all(y[:, None] > X)
    # Row order is time order and the target sits exactly `horizon` ahead.
    np.testing.assert_array_equal(y - X[:, 0], horizon)
    assert np.all(np.diff(y) > 0)
