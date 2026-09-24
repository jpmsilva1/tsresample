"""G2: the scikit-learn estimator contract (SPEC §2.2)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

from tsresample import TimeSeriesResampler

# The checks that apply to an estimator with no fit(): fit_resample returns a
# different number of rows, so it is not a transformer (SPEC §2.2).
CHECKS = [
    check_parameters_default_constructible,
    check_no_attributes_set_in_init,
    check_get_params_invariance,
    check_set_params,
]


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_sklearn_estimator_check(check) -> None:  # type: ignore[no-untyped-def]
    check("TimeSeriesResampler", TimeSeriesResampler())


def test_init_stores_arguments_unmodified_and_validates_nothing() -> None:
    rs = np.random.RandomState(3)
    est = TimeSeriesResampler("bogus", "nope", k=-1, o=-5.0, random_state=rs)
    assert est.strategy == "bogus" and est.bias == "nope" and est.k == -1
    assert est.random_state is rs


def test_clone_and_get_params_round_trip() -> None:
    est = TimeSeriesResampler("over", "temporal", k=3, o=2.0, r_quirks=False)
    twin = clone(est)
    assert twin is not est and twin.get_params() == est.get_params()
    assert TimeSeriesResampler(**est.get_params()).get_params() == est.get_params()
