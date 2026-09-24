"""Value-space relevance bumps (SPEC §4.2, ADR-0012)."""

import warnings
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

Rule = Literal["under", "over", "smote"]


class Bump(NamedTuple):
    """A run of cases in value order between two phi crossings of ``t_R``.

    ``idx`` holds the cases' positions in the input, in ascending-``y`` order.
    Under/smote a bump whose mean phi equals ``t_R`` exactly is neither rare
    nor normal (SPEC §4.2 step 3).
    """

    idx: NDArray[np.intp]
    rare: bool
    normal: bool


def bumps(y: ArrayLike, phi: ArrayLike, t_R: float, rule: Rule) -> list[Bump]:
    """Partition cases into bumps in value space (SPEC §4.2, ADR-0012).

    Cases are sorted by ``y`` (stable) and cut where phi crosses ``t_R``:
    ``rule="over"`` cuts where ``phi >= t_R`` flips; ``"under"``/``"smote"`` cut on
    a strict sign change of ``s = -phi if phi > t_R else phi``. A bump is rare
    by its mean phi (``>= t_R`` for over, ``> t_R`` otherwise).

    Returns ``[]`` with a ``UserWarning`` when there is no rare bump or no
    normal bump: the caller then returns its input unchanged. R stops here; the
    no-op is a documented robustness deviation (ADR-0011).
    """
    order = np.argsort(np.asarray(y, dtype=np.float64), kind="stable")
    p = np.asarray(phi, dtype=np.float64)[order]
    if rule == "over":
        above = p >= t_R
        cuts = np.flatnonzero(above[:-1] != above[1:]) + 1
    else:
        s = np.where(p > t_R, -p, p)
        cuts = np.flatnonzero(s[:-1] * s[1:] < 0) + 1
    out = []
    for idx in np.split(order, cuts):
        m = float(np.mean(np.asarray(phi, dtype=np.float64)[idx]))
        rare = m >= t_R if rule == "over" else m > t_R
        out.append(Bump(idx, rare, m < t_R))
    missing = [
        w
        for w, ok in (
            ("rare", any(b.rare for b in out)),
            ("normal", any(b.normal for b in out)),
        )
        if not ok
    ]
    if missing:
        warnings.warn(
            f"bumps: no {' and no '.join(missing)} bump at t_R={t_R}; "
            "nothing to resample, the input is returned unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return []
    return out
