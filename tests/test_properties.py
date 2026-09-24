"""G1: properties that must hold for every input (Hypothesis)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tsresample import _bins, embed


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


@pytest.mark.filterwarnings("ignore:bumps")
@given(
    data=st.lists(
        st.tuples(st.integers(-5, 5), st.sampled_from([0.0, 0.3, 0.9, 0.95, 1.0])),
        min_size=1,
        max_size=40,
    ),
    rule=st.sampled_from(["under", "over", "smote"]),
)
def test_bumps_partition_the_value_sorted_order(
    data: list[tuple[int, float]], rule: str
) -> None:
    y = np.array([d[0] for d in data], dtype=float)
    phi = np.array([d[1] for d in data])
    bs = _bins.bumps(y, phi, 0.9, rule)  # type: ignore[arg-type]
    if not bs:  # no-bump rule
        return
    flat = np.concatenate([b.idx for b in bs])
    assert sorted(flat.tolist()) == list(range(len(y)))  # no gaps, no overlaps
    assert all(len(b.idx) for b in bs)
    assert np.all(np.diff(y[flat]) >= 0)  # value order, not time order
    if rule == "over":  # each bump lies wholly on one side of t_R
        assert all(len(set((phi[b.idx] >= 0.9).tolist())) == 1 for b in bs)
