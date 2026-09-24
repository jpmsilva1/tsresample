"""G3: algorithmic worked examples. Expected values are hand-derived in comments."""

import numpy as np
import pytest

from tsresample import _bins, _prefs, _relevance


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


# --- bumps (SPEC §4.2, ADR-0012) -------------------------------------------------
# Two-sided example, t_R = 0.9. Case i (time index) has value y[i] and phi[i]:
#   time: 0    1    2    3    4    5    6
#   y:    5    1    3    9    7    2    8
#   phi:  0.0  1.0  0.2  1.0  0.5  0.9  0.9
# Value order (y ascending) -> time indices 1, 5, 2, 0, 4, 6, 3 with
#   phi 1.0, 0.9, 0.2, 0.0, 0.5, 0.9, 1.0
TWO_SIDED_Y = np.array([5.0, 1, 3, 9, 7, 2, 8])
TWO_SIDED_PHI = np.array([0.0, 1.0, 0.2, 1.0, 0.5, 0.9, 0.9])


def _as_lists(bs: list[_bins.Bump]) -> list[tuple[list[int], bool]]:
    return [(b.idx.tolist(), b.rare) for b in bs]


def test_under_smote_cut_on_strict_sign_change() -> None:
    # s = -phi where phi > 0.9 else phi: -1, .9, .2, 0, .5, .9, -1.
    # s_i * s_{i+1} < 0 only at the two ends -> bumps {1}, {5,2,0,4,6}, {3}.
    # Means: 1.0 (> 0.9 rare), (0.9+0.2+0+0.5+0.9)/5 = 0.5 (normal), 1.0 (rare).
    for rule in ("under", "smote"):
        got = _bins.bumps(TWO_SIDED_Y, TWO_SIDED_PHI, 0.9, rule)
        assert _as_lists(got) == [([1], True), ([5, 2, 0, 4, 6], False), ([3], True)]


def test_over_cuts_where_phi_crosses_t_r_inclusively() -> None:
    # phi >= 0.9: T, T, F, F, F, T, T -> bumps {1,5}, {2,0,4}, {6,3}.
    # Means: 0.95 (>= 0.9 rare), 0.7/3 = 0.233 (normal), 0.95 (rare).
    got = _bins.bumps(TWO_SIDED_Y, TWO_SIDED_PHI, 0.9, "over")
    assert _as_lists(got) == [([1, 5], True), ([2, 0, 4], False), ([6, 3], True)]


@pytest.mark.parametrize("rule", ["under", "over", "smote"])
def test_one_sided_phi_gives_a_normal_and_a_high_rare_bump(rule: str) -> None:
    # time: 0 1 2 3; y = 3, 1, 2, 4; phi = 0.5, 0, 0, 1 (phi = 0 low endpoint).
    # Value order: times 1, 2, 0, 3 with phi 0, 0, 0.5, 1. Both rules cut only
    # before phi = 1 -> {1, 2, 0} mean 0.5/3 = 0.167 (normal), {3} mean 1 (rare).
    got = _bins.bumps([3.0, 1, 2, 4], [0.5, 0, 0, 1], 0.9, rule)  # type: ignore[arg-type]
    assert _as_lists(got) == [([1, 2, 0], False), ([3], True)]
    assert [b.normal for b in got] == [True, False]


@pytest.mark.parametrize(
    ("phi", "rule"),
    [
        ([0.0, 0.0, 0.0], "under"),  # every phi 0: one normal bump, no rare
        ([1.0, 1.0, 1.0], "over"),  # every phi 1: one rare bump, no normal
        # Sign rule: s = -1, 0, 0 -> product never < 0, so one bump of mean 1/3:
        # a rare case swallowed into a normal bump (R's quirk) -> no rare bump.
        ([1.0, 0.0, 0.0], "smote"),
    ],
)
def test_no_rare_or_no_normal_bump_returns_empty_with_warning(
    phi: list[float], rule: str
) -> None:
    with pytest.warns(UserWarning, match="no rare bump|no normal bump"):
        assert _bins.bumps([1.0, 2, 3], phi, 0.9, rule) == []  # type: ignore[arg-type]


# --- preferences (SPEC §4.3, ADR-0012) -------------------------------------------
# The normal bump of the two-sided example: idx = [5, 2, 0, 4, 6] (value order),
# phi there = [0.9, 0.2, 0.0, 0.5, 0.9]. Chronological rank within the bump:
# time 0 -> 1, 2 -> 2, 4 -> 3, 5 -> 4, 6 -> 5 (r = 5);
# aligned with idx: j = [4, 2, 1, 3, 5].
NORMAL_BUMP = _bins.Bump(np.array([5, 2, 0, 4, 6]), rare=False, normal=True)


@pytest.mark.parametrize(
    ("bias", "expected"),
    [
        (None, [0.2] * 5),
        # j / r normalised: [4, 2, 1, 3, 5] / 15
        ("temporal", [4 / 15, 2 / 15, 1 / 15, 3 / 15, 5 / 15]),
        # j * phi = [3.6, 0.4, 0, 1.5, 4.5] (the /r cancels), sum 10
        ("temporal+phi", [0.36, 0.04, 0.0, 0.15, 0.45]),
    ],
)
def test_preference_uses_time_rank_within_the_bump(
    bias: str | None, expected: list[float]
) -> None:
    p = _prefs.preference(NORMAL_BUMP, np.arange(7), TWO_SIDED_PHI, bias)  # type: ignore[arg-type]
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)


def test_temporal_phi_falls_back_to_uniform_when_every_phi_is_zero() -> None:
    # Bump {2, 0}: phi = 0.2, 0.0 -> use phi = 0 everywhere instead.
    bump = _bins.Bump(np.array([2, 0]), rare=False, normal=True)
    p = _prefs.preference(bump, np.arange(7), np.zeros(7), "temporal+phi")
    np.testing.assert_array_equal(p, [0.5, 0.5])


def test_no_replacement_draw_takes_all_positive_cases_then_fills_uniformly() -> None:
    # 2 positive-probability cases, 4 wanted without replacement: R raises; we take
    # both positives, fill 2 more from the zero-probability cases, and warn.
    p = np.array([0.0, 0.7, 0.0, 0.3, 0.0])
    with pytest.warns(UserWarning, match="only 2 cases"):
        got = _prefs.draw(p, 4, replace=False, rng=np.random.RandomState(0))
    assert len(set(got.tolist())) == 4
    assert {1, 3} <= set(got.tolist())


def test_draw_follows_p_and_honours_replacement() -> None:
    rng = np.random.RandomState(0)
    got = _prefs.draw(np.array([0.0, 1.0, 0.0]), 5, replace=True, rng=rng)
    np.testing.assert_array_equal(got, [1] * 5)
    got = _prefs.draw(np.array([0.5, 0.5, 0.0]), 2, replace=False, rng=rng)
    assert sorted(got.tolist()) == [0, 1]
