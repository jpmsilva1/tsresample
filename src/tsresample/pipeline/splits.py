"""Monte Carlo temporal splits (SPEC §2.4)."""

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from tsresample._validate import check_xy


def temporal_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    train_size: float = 0.5,
    test_size: float = 0.25,
    n_reps: int = 50,
    random_state: int | np.random.RandomState | None = None,
) -> Iterator[tuple[NDArray[np.float64], ...]]:
    """Yield ``(X_train, y_train, X_test, y_test)`` for ``n_reps`` Monte Carlo splits.

    Each split is a contiguous training window of ``trunc(train_size * n)`` rows
    followed immediately by a test window of ``trunc(test_size * n)`` rows, at a
    random position (the paper's ``MonteCarlo(nReps=50, szTrain=.5,
    szTest=.25)``). Rows are never shuffled.
    """
    Xa, ya = check_xy(X, y)
    n = len(ya)
    if train_size + test_size > 1:
        raise ValueError(
            f"train_size + test_size must be <= 1; got {train_size} + {test_size}."
        )
    n_tr, n_te = math.floor(train_size * n), math.floor(test_size * n)
    if min(n_tr, n_te) < 1:
        raise ValueError(
            f"windows need at least 1 row each; {n} rows give train {n_tr}, "
            f"test {n_te}."
        )
    rng = check_random_state(random_state)
    for _ in range(n_reps):
        a = int(rng.randint(n - n_tr - n_te + 1))
        b = a + n_tr
        yield Xa[a:b], ya[a:b], Xa[b : b + n_te], ya[b : b + n_te]
