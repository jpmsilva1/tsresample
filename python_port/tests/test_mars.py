import numpy as np
import pytest

from tsresamp.mars import MARS


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def test_recovers_single_hinge():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(400, 3))
    y = 2 * np.maximum(0, X[:, 0] - 0.5) + rng.normal(0, 0.05, 400)
    m = MARS(nk=17, degree=1, thresh=0.001).fit(X, y)
    assert r2(y, m.predict(X)) > 0.95
    # a knot close to 0.5 on variable 0 is used
    knots = [t for term in m.terms_ for (v, t, s) in term if v == 0 and s != 0]
    assert any(abs(t - 0.5) < 0.1 for t in knots)
    assert len(m.terms_) <= 17


def test_degree_two_captures_interaction():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(600, 2))
    y = 5 * np.maximum(0, X[:, 0] - 0.3) * np.maximum(0, X[:, 1] - 0.4) + rng.normal(0, 0.02, 600)
    m1 = MARS(nk=17, degree=1).fit(X, y)
    m2 = MARS(nk=17, degree=2).fit(X, y)
    assert r2(y, m2.predict(X)) > r2(y, m1.predict(X))
    assert r2(y, m2.predict(X)) > 0.95
    assert any(len(t) == 2 for t in m2.terms_)


def test_predict_shape_and_generalisation():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = np.sin(X[:, 1]) + 0.5 * X[:, 2] + rng.normal(0, 0.1, 300)
    m = MARS(nk=17, degree=2, thresh=0.001).fit(X, y)
    Xt = rng.normal(size=(50, 4))
    p = m.predict(Xt)
    assert p.shape == (50,)
    assert np.isfinite(p).all()
    assert r2(y, m.predict(X)) > 0.8


def test_constant_target():
    X = np.random.default_rng(3).normal(size=(30, 2))
    y = np.full(30, 4.0)
    m = MARS().fit(X, y)
    np.testing.assert_allclose(m.predict(X), 4.0)


def test_nk_bound_and_thresh_stop():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(200, 5))
    y = rng.normal(size=200)  # pure noise -> few or no terms
    m = MARS(nk=5, degree=1, thresh=0.05).fit(X, y)
    assert len(m.terms_) <= 5


# --- fidelity to the R package earth (5.3.6) --------------------------------
# tests/data/earth/*: Monte Carlo windows of the experiments (train/test), the
# predictions of earth(V10 ~ ., train, nk, degree, thresh) and its model
# (dirs, cuts, selected terms, coefficients).  Generated with R; see
# README.md ("What was ported and how").
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "earth"
FIXTURES = {  # name: (nk, degree, thresh)
    "ds16_it7": (17, 2, 0.001),           # small-valued series: MIN_BX_SOS drops linear candidates
    "ds10_it7UNDERTPhi": (17, 2, 0.001),  # integer series after under-sampling: linear terms, single hinges
    "ds15_it33": (17, 2, 0.001),          # pruning where the greedy backward path is not the best subset
    "ds19_it7UNDERB": (17, 1, 0.01),      # thresh = 0.01, additive
}


def _read(name):
    rows = list(csv.reader(open(name)))
    return np.array(rows[1:], dtype=float)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_matches_earth(name):
    nk, degree, thresh = FIXTURES[name]
    tr = _read(DATA / f"{name}_train.csv")
    te = _read(DATA / f"{name}_test.csv")
    pe = _read(DATA / f"{name}_earth_pred.csv").ravel()
    m = MARS(nk=nk, degree=degree, thresh=thresh).fit(tr[:, :-1], tr[:, -1])
    p = m.predict(te[:, :-1])
    scale = np.max(np.abs(pe)) + 1e-12
    np.testing.assert_allclose(p, pe, rtol=0, atol=1e-7 * scale)
    # same selected terms (variable, direction, knot) and coefficients
    model = list(csv.DictReader(open(DATA / f"{name}_earth_model.csv")))
    npred = tr.shape[1] - 1
    earth_terms = {}
    for r in model:
        if r["selected"] == "1":
            key = tuple(sorted((v, int(float(r[f"dir{v + 1}"])), round(float(r[f"cut{v + 1}"]), 6))
                               for v in range(npred) if float(r[f"dir{v + 1}"]) != 0))
            earth_terms[key] = float(r["coef"])
    port_terms = {}
    for term, c in zip(m.terms_, m.coef_):
        key = tuple(sorted((v, 2 if s == 0 else s, round(t, 6)) for v, t, s in term))
        port_terms[key] = float(c)
    assert set(port_terms) == set(earth_terms)
    for k in earth_terms:
        assert abs(port_terms[k] - earth_terms[k]) <= 1e-6 * (abs(earth_terms[k]) + 1e-9) + 1e-9 * scale
