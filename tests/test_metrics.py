"""metrics: hand-worked conventions (SPEC §4.7, ADR-0009, ADR-0014, ADR-0016)."""

import numpy as np
import pytest

from tsresample.metrics import control_points, f1_phi, precision_phi, recall_phi, sera

# Two-sided phi with knots at -1 (phi 1), 0 (phi 0), 1 (phi 1).
CP = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])


def test_empty_selection_scores_1e_5_not_zero() -> None:
    # Predictions are all 0 (phi 0): no phi(y_hat) >= 0.9 -> precision = 1e-5.
    # |y_true| <= 0.2 -> phi <= 3(.2)^2 - 2(.2)^3 = 0.104 < 0.9 -> recall = 1e-5.
    y_true = np.array([0.0, 0.1, -0.1, 0.2])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])
    assert precision_phi(y_true, y_pred, relevance=CP) == 1e-5
    assert recall_phi(y_true, y_pred, relevance=CP) == 1e-5
    # F1 = 2 * 1e-5 * 1e-5 / 2e-5 = 1e-5
    assert f1_phi(y_true, y_pred, relevance=CP) == pytest.approx(1e-5)


def test_perfect_prediction_of_rare_cases_scores_one() -> None:
    # y = y_hat = 1 and -1: L = 0 -> Gamma_B = Gamma_C = 0 -> u = phi(y) = 1.
    # recall = (|1+1| + |1+1|) / (|1+1| + |1+1|) = 1; precision the same.
    y = np.array([1.0, -1.0, 0.0])
    assert recall_phi(y, y.copy(), relevance=CP) == pytest.approx(1.0)
    assert precision_phi(y, y.copy(), relevance=CP) == pytest.approx(1.0)
    assert f1_phi(y, y.copy(), relevance=CP) == pytest.approx(1.0)


def test_event_threshold_is_inclusive() -> None:
    # t_E = 1.0 and y, y_hat exactly on phi = 1 knots: phi >= 1 selects them
    # (recall = precision = 1); a strict > would select nothing (1e-5).
    y = np.array([1.0, -1.0, 0.0])
    for f in (recall_phi, precision_phi):
        assert f(y, y.copy(), relevance=CP, rel_threshold=1.0) == pytest.approx(1.0)


def test_sera_of_a_perfect_forecast_is_zero_and_curve_is_returned() -> None:
    y = np.array([1.0, -1.0, 0.0, 0.5])
    assert sera(y, y.copy(), relevance=CP) == 0.0
    t, ser = sera(y, y + 1.0, relevance=CP, step=0.25, return_curve=True)  # type: ignore[misc]
    # Each case errs by 1. phi: 1, 1, 0, 0.5 -> SER_t counts cases with phi >= t:
    # t = 0: 4; 0.25, 0.5: 3; 0.75, 1: 2.
    np.testing.assert_array_equal(t, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_array_equal(ser, [4, 3, 3, 2, 2])


@pytest.mark.parametrize(
    ("relevance", "match"),
    [
        (np.ones(3), "control points"),
        (lambda v: v, "control points"),
        (np.ones((2, 2)), r"shape \(3, 2\)"),
        (np.array([[-1, 1, 0.5], [0, 0, 0], [1, 1, 0]]), "slopes"),
    ],
)
def test_utility_metrics_reject_relevance_without_control_points(
    relevance: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        recall_phi([0.0, 1.0, -1.0], [0.0, 1.0, -1.0], relevance=relevance)  # type: ignore[arg-type]


def test_auto_fits_on_y_true_and_control_points_are_public() -> None:
    y = np.array([4, 1, 9, 2, 100, 3, 8, 5, 7, 6], dtype=float)
    # control_points(y) = (1, 0), (5.5, 0), (9, 1) (worked in test_algorithms).
    assert recall_phi(y, y, relevance="auto") == recall_phi(
        y, y, relevance=control_points(y)
    )


def test_mismatched_or_non_finite_inputs_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        sera([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        sera([1.0, np.nan], [1.0, 2.0])


def test_worst_possible_forecast_of_a_rare_case_scores_zero() -> None:
    # y = 1 predicted as -1 (the opposite extreme). Bumps <-inf, -1>, <0, 1>;
    # y is in bump 2: Delta = 2 min(|0 - 1|, |1 - inf|) = 2. L = 2.
    # Benefit: y_hat < y, L_B = min(2, |1 - 0|) = 1 <= L -> Gamma_B = 1.
    # Cost: L_C = min(2, |1 - b*_1|) = min(2, |1 - (-1)|) = 2 <= L -> Gamma_C = 1.
    # u = 1 * (1 - 1) - 0.5 * (1 + 1) * 1 = -1 -> |1 + u| = 0: recall = precision = 0.
    y, yh = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    assert recall_phi(y, yh, relevance=CP) == 0.0
    assert f1_phi(y, yh, relevance=CP) == 0.0


def test_control_point_phi_must_lie_in_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sera([0.0, 1.0], [0.0, 1.0], relevance=np.array([[-1, 2.0], [0, 0], [1, 1]]))


@pytest.mark.parametrize("step", [0.0, -0.1, 0.3, 1.5])
def test_sera_step_must_divide_the_unit_interval(step: float) -> None:
    with pytest.raises(ValueError, match="step"):
        sera([0.0, 1.0], [0.0, 1.0], relevance=CP, step=step)
