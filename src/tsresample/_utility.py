"""Utility surface u(y_hat, y) for the phi-weighted metrics (SPEC §4.7, ADR-0014).

Ribeiro (2011) §3.3-3.4 with the bump initialisation pinned by the recorded R
output. Bumps are ``(b_minus, b_star)`` (left edge, maximum); bump i covers
``[b_minus_i, b_minus_{i+1})``.
"""

import numpy as np
from numpy.typing import NDArray

from tsresample import _relevance

_INF = np.inf
P = 0.5  # uba's p: weight of phi(y) in phi_p (ADR-0014 item 6)


def _bumps(cp: NDArray[np.float64]) -> list[tuple[float, float]]:
    # ADR-0014 item 2, for the extremes shapes (all knots have slope 0).
    (x1, v1), (med, _), (x3, v3) = cp
    if v1 > 0:  # phi first decreases: bump 1 is open from -inf
        high = (med, x3) if v3 > 0 else ((med + x3) / 2, _INF)
        return [(-_INF, x1), high]
    # phi first increases (or never changes): leading constant run averaged,
    # preceded by the <-inf, -inf> bump
    return [(-_INF, -_INF), ((x1 + med) / 2, x3)]


def _max_losses(bumps: list[tuple[float, float]]) -> list[float]:
    # Def 3.12: 2 * min(|b- - b*|, |b* - b-_next|); non-finite takes the
    # adjacent bump's value (thesis p. 85).
    edges = [b[0] for b in bumps] + [_INF]
    with np.errstate(invalid="ignore"):
        raw = [
            2 * min(abs(bm - bs), abs(bs - edges[i + 1]))
            for i, (bm, bs) in enumerate(bumps)
        ]
    for i in range(len(raw)):
        if not np.isfinite(raw[i]):
            nxt = [raw[j] for j in (i + 1, i - 1) if 0 <= j < len(raw)]
            raw[i] = next((v for v in nxt if np.isfinite(v)), _INF)
    return raw


def utility(
    y_hat: NDArray[np.float64], y: NDArray[np.float64], cp: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``u_i = U(y_hat_i, y_i)`` (ADR-0014 items 1-6)."""
    bumps = _bumps(cp)
    delta = np.array(_max_losses(bumps))
    b_minus = np.array([b[0] for b in bumps] + [_INF])
    b_star = np.array([-_INF] + [b[1] for b in bumps] + [_INF])  # padded by 1
    # y exactly on an edge belongs to the upper bump; y_hat == y is "not below".
    g = np.searchsorted(b_minus[:-1], y, side="right") - 1
    below = y_hat < y
    d = delta[g]
    lb = np.minimum(
        d, np.where(below, np.abs(y - b_minus[g]), np.abs(y - b_minus[g + 1]))
    )
    lc = np.minimum(
        d, np.where(below, np.abs(y - b_star[g]), np.abs(y - b_star[g + 2]))
    )
    loss = np.abs(y_hat - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        gam_b = np.where(loss < lb, loss / lb, 1.0)
        gam_c = np.where(loss < lc, loss / lc, 1.0)
    phi_y = _relevance.phi(y, cp)
    phi_p = (1 - P) * _relevance.phi(y_hat, cp) + P * phi_y
    out: NDArray[np.float64] = phi_y * (1 - gam_b) - phi_p * gam_c
    return out
