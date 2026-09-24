import numpy as np
import pytest

from tsresamp.uba import (DELTA, HermiteSpline, PhiBumps, benefcost_lin, boxplot_stats, bumps_set,
                          fivenum, loss_control, phi, phi_control, util, util_values)


def test_fivenum_matches_R():
    np.testing.assert_allclose(fivenum(np.arange(1, 11)), [1, 3, 5.5, 8, 10])
    np.testing.assert_allclose(fivenum([1, 2, 3, 4, 5, 100]), [1, 2, 3.5, 5, 100])
    np.testing.assert_allclose(fivenum([3, 1, 2]), [1, 1.5, 2, 2.5, 3])


def test_boxplot_stats_matches_R():
    stats, out = boxplot_stats([1, 2, 3, 4, 5, 100])
    np.testing.assert_allclose(stats, [1, 2, 3.5, 5, 5])
    np.testing.assert_allclose(out, [100])
    stats, out = boxplot_stats(np.arange(1, 11))
    np.testing.assert_allclose(stats, [1, 3, 5.5, 8, 10])
    assert len(out) == 0


def test_phi_control_extremes_high_only():
    y = np.array([1, 2, 3, 4, 5, 100], float)
    pc = phi_control(y, method="extremes")
    assert pc["method"] == "extremes" and pc["npts"] == 3
    cp = np.array(pc["control_pts"]).reshape(3, 3)
    np.testing.assert_allclose(cp, [[1, 0, 0], [3.5, 0, 0], [5, 1, 0]])


def test_phi_control_extremes_both_sides():
    y = np.concatenate([[-100], np.arange(1, 11), [100]])
    pc = phi_control(y)
    cp = np.array(pc["control_pts"]).reshape(-1, 3)
    assert cp.shape[0] == 3
    assert cp[0, 1] == 1 and cp[2, 1] == 1 and cp[1, 1] == 0
    assert cp[0, 0] == 1 and cp[2, 0] == 10  # adjL / adjH


def test_phi_values_smoothstep_and_extrapolation():
    y = np.array([1, 2, 3, 4, 5, 100], float)
    pc = phi_control(y)
    # zero below the median, one above adjH, smooth (symmetric) in between
    vals = phi(np.array([-50, 1, 3.5, 4.25, 5, 50, 1000]), pc)
    np.testing.assert_allclose(vals, [0, 0, 0, 0.5, 1, 1, 1], atol=1e-12)
    # monotone increasing between median and adjH
    grid = np.linspace(3.5, 5, 50)
    assert np.all(np.diff(phi(grid, pc)) >= 0)
    assert np.all((phi(grid, pc) >= 0) & (phi(grid, pc) <= 1))


def test_phi_range_method_with_two_columns():
    pc = phi_control(np.linspace(0, 10, 100), method="range", control_pts=np.array([[0, 0], [5, 0.5], [10, 1]]))
    v = phi(np.array([0, 5, 10]), pc)
    np.testing.assert_allclose(v, [0, 0.5, 1], atol=1e-12)


def test_hermite_spline_derivatives_zero_at_control_points():
    H = HermiteSpline.from_control_points([0, 1], [0, 1], [0, 0])
    yv, ydv, yddv = H.value(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(yv, [0, 0.5, 1])
    np.testing.assert_allclose(ydv[[0]], [0])
    assert ydv[1] > 0


def test_loss_control():
    y = np.array([1, 2, 3, 4, 5, 100], float)
    ls = loss_control(y)
    assert ls["ymin"] == 1 and ls["ymax"] == 100 and ls["epsilon"] == 0.1
    n = len(y)
    tL = np.abs(y.mean() - y)
    expect = 3 * np.std(tL, ddof=1) * np.sqrt(np.log(n) / n)
    assert ls["tloss"] == pytest.approx(expect)


def test_bumps_high_outliers_only():
    # control points (min,0) (median,0) (adjH,1): expect two "bumps"
    y = np.array([1, 2, 3, 4, 5, 100], float)
    pc = phi_control(y)
    from tsresamp.uba import _spline_from_parms
    B = bumps_set(_spline_from_parms(pc), loss_control(y)["tloss"])
    assert B.n == 2
    assert B.bleft[0] == -np.inf and B.bmax[0] == -np.inf
    assert B.bleft[1] == pytest.approx((1 + 3.5) / 2)
    assert B.bmax[1] == pytest.approx(5)
    L = 2 * abs(5 - (1 + 3.5) / 2)
    assert B.bloss[1] == pytest.approx(L) and B.bloss[0] == pytest.approx(L)


def test_bumps_both_outliers():
    y = np.concatenate([[-100], np.arange(1, 11), [100]])
    pc = phi_control(y)
    from tsresamp.uba import _spline_from_parms
    B = bumps_set(_spline_from_parms(pc), loss_control(y)["tloss"])
    assert B.n == 2
    assert B.bmax[0] == pytest.approx(1)  # adjL
    assert B.bleft[1] == pytest.approx(5.5)  # median
    assert B.bmax[1] == pytest.approx(10)  # adjH
    assert B.bloss[0] == pytest.approx(2 * abs(1 - 5.5))
    assert B.bloss[1] == pytest.approx(2 * abs(10 - 5.5))


def test_bumps_no_outliers_falls_back_to_tloss():
    y = np.arange(1, 11, dtype=float)
    pc = phi_control(y)
    from tsresamp.uba import _spline_from_parms
    ls = loss_control(y)
    B = bumps_set(_spline_from_parms(pc), ls["tloss"])
    assert B.n == 1 and B.bloss[0] == pytest.approx(ls["tloss"])


def _toy():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1, 200), [8, 9, 10, 12]])
    pc = phi_control(y)
    ls = loss_control(y)
    return y, pc, ls


def test_util_perfect_prediction():
    y, pc, ls = _toy()
    u, y_phi, ypred_phi = util_values(y, y, pc, ls)
    # perfect predictions: benefit 1 and cost 0 -> u == phi(y)
    np.testing.assert_allclose(u, y_phi, atol=1e-12)
    assert util(y, y, pc, ls, umetric="P", event_thr=0.9) == pytest.approx(1.0)
    assert util(y, y, pc, ls, umetric="R", event_thr=0.9) == pytest.approx(1.0)
    assert util(y, y, pc, ls, umetric="Fm", event_thr=0.9, beta=1) == pytest.approx(1.0)
    assert util(y, y, pc, ls, umetric="MU") == pytest.approx(float(np.mean(y_phi)))


def test_util_constant_prediction_has_no_positives():
    y, pc, ls = _toy()
    ypred = np.full_like(y, np.median(y))
    prec = util(ypred, y, pc, ls, umetric="P", event_thr=0.9)
    rec = util(ypred, y, pc, ls, umetric="R", event_thr=0.9)
    f1 = util(ypred, y, pc, ls, umetric="Fm", event_thr=0.9, beta=1)
    assert prec == DELTA
    assert 0 < rec < 1
    assert 0 < f1 < 0.01


def test_util_in_unit_interval_and_ranking():
    y, pc, ls = _toy()
    rng = np.random.default_rng(2)
    noisy = y + rng.normal(0, 0.5, len(y))
    worse = y + rng.normal(0, 3, len(y))
    for pred in (noisy, worse):
        for m in ("P", "R"):
            v = util(pred, y, pc, ls, umetric=m, event_thr=0.9)
            assert 0 <= v <= 1
    f_good = util(noisy, y, pc, ls, umetric="Fm", event_thr=0.9)
    f_bad = util(worse, y, pc, ls, umetric="Fm", event_thr=0.9)
    assert f_good > f_bad


def test_benefcost_single_bump_uses_bloss():
    B = PhiBumps(n=1, bleft=np.array([-np.inf]), bmax=np.array([-np.inf]), bloss=np.array([0.3]))
    lb, lc = benefcost_lin(np.array([0.0, 1.0]), np.array([0.5, 0.5]), B)
    np.testing.assert_allclose(lb, [0.3, 0.3])
    np.testing.assert_allclose(lc, [0.3, 0.3])
