import numpy as np
from scipy.interpolate import PchipInterpolator


def _phi(y, ctrl_pts):
    """Reconstructs R's uba::phi() relevance function from phi.control()'s
    control points. ctrl_pts is a flat (value, phi, derivative)-triple
    sequence -- the exact shape R's phi.control()$control.pts captures
    (derivative is unused: PCHIP derives its own monotone-preserving
    derivatives from the points, same as R's monoH.FC spline family, and
    matching this project's existing calc_phi_pchip() in
    scratch/generate_latex_table.R)."""
    pts = np.asarray(ctrl_pts, dtype=float).reshape(-1, 3)
    interp = PchipInterpolator(pts[:, 0], pts[:, 1])
    return np.clip(interp(y), 0.0, 1.0)


def sera(y_true, y_pred, ctrl_pts, step=0.01):
    """Squared Error-Relevance Area (Ribeiro & Moniz 2020): area under the
    SER_t curve, SER_t = sum of squared errors for cases with phi(y)>=t,
    integrated over t via the trapezoidal rule."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    phi_true = _phi(y_true, ctrl_pts)
    squared_errors = (y_pred - y_true) ** 2

    thresholds = np.arange(0, 1 + step, step)
    ser_t = np.array([squared_errors[phi_true >= t].sum() for t in thresholds])
    return np.trapezoid(ser_t, thresholds)
