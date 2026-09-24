"""``TimeSeriesResampler``: sklearn-style dispatch only, no arithmetic (SPEC §2.2)."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from tsresample import _bins, _relevance, _sample, _synth
from tsresample._validate import check_choice, check_xy


class TimeSeriesResampler(BaseEstimator):  # type: ignore[misc]
    """Resample an imbalanced time-series regression training set.

    Implements the nine strategies of Moniz, Branco & Torgo (2017): random
    undersampling, random oversampling and SMOTE, each with no bias, a temporal
    bias, or a temporal+relevance bias (SPEC §4.6). Defaults reproduce the
    original R code (ADR-0011).

    Parameters
    ----------
    strategy : {"under", "over", "smote"}, default "smote"
    bias : {None, "temporal", "temporal+phi"}, default None
    rel_threshold : float, default 0.9
        ``t_R``; bumps are rare by their mean relevance (SPEC §4.2).
    relevance : "auto", array of shape (n,), or callable, default "auto"
        ``"auto"`` fits phi on ``y`` (SPEC §4.1). An array gives phi per case; a
        callable is called as ``f(y)``. Values must lie in [0, 1].
    k : int, default 5
        SMOTE neighbour count; ignored otherwise.
    o, u : float or None, default None
        R's ``C.perc``: ``None`` balances (SPEC §4.4); otherwise ``o`` scales
        rare bumps and ``u`` normal bumps.
    r_quirks : bool, default True
        Reproduce R's SMOTE quirks (ADR-0013). ``False`` gives the paper reading.
    random_state : int, RandomState or None, default None

    Notes
    -----
    Rows of ``X`` must be in time order: position is the time index. Output rows
    are in time order, synthetic cases right after their seed. Stateless: phi is
    refit on every call (ADR-0008).

    Examples
    --------
    >>> import numpy as np
    >>> from tsresample import TimeSeriesResampler, embed
    >>> series = np.random.RandomState(0).standard_t(3, size=300)  # heavy tails
    >>> X, y = embed(series, k=4)
    >>> X.shape
    (295, 5)
    >>> resampler = TimeSeriesResampler("smote", "temporal", random_state=0)
    >>> X_res, y_res = resampler.fit_resample(X, y)
    >>> X_res.shape
    (293, 5)
    """

    def __init__(
        self,
        strategy: Literal["under", "over", "smote"] = "smote",
        bias: Literal["temporal", "temporal+phi"] | None = None,
        *,
        rel_threshold: float = 0.9,
        relevance: Literal["auto"] | ArrayLike | Callable[..., ArrayLike] = "auto",
        k: int = 5,
        o: float | None = None,
        u: float | None = None,
        r_quirks: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.strategy = strategy
        self.bias = bias
        self.rel_threshold = rel_threshold
        self.relevance = relevance
        self.k = k
        self.o = o
        self.u = u
        self.r_quirks = r_quirks
        self.random_state = random_state

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the resampled ``(X, y)``; rows in time order (SPEC §2.2).

        If phi yields no rare or no normal bump, the input is returned
        unchanged with a ``UserWarning`` (SPEC §4.2, ADR-0011).
        """
        rng = check_random_state(self.random_state)
        Xa, ya = check_xy(X, y)
        check_choice("strategy", self.strategy, ("under", "over", "smote"))
        check_choice("bias", self.bias, (None, "temporal", "temporal+phi"))
        if not (isinstance(self.k, int | np.integer) and self.k >= 1):
            raise ValueError(f"k: expected an integer >= 1; got {self.k!r}.")
        phi = _relevance.resolve(self.relevance, ya)
        found = _bins.bumps(ya, phi, self.rel_threshold, self.strategy)
        if not found:
            return Xa.copy(), ya.copy()
        t = np.arange(len(ya))
        idx, jobs = _sample.resample(
            found, len(ya), self.strategy, self.o, self.u, t, phi, self.bias, rng
        )
        syn = [
            _synth.synthesize(
                Xa, ya, b, c, t, phi, self.bias, self.k, rng, r_quirks=self.r_quirks
            )
            for b, c in jobs
        ]
        X_syn = np.vstack([Xa[:0], *(s[0] for s in syn)])
        y_syn = np.concatenate([ya[:0], *(s[1] for s in syn)])
        seeds = np.concatenate([t[:0], *(s[2] for s in syn)])
        return _sample.assemble(Xa, ya, idx, X_syn, y_syn, seeds)
