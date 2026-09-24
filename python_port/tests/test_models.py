import numpy as np
import pandas as pd
import pytest

from tsresamp.data import create_data
from tsresamp.models import (WORKFLOW_NAMES, WORKFLOWS, AutoARIMA, RPartLearner, SVMLearner, embed_stats,
                             eval_stats, extended_stats, learner_workflow, mc_BDES, mc_arima, run_workflow)
from tsresamp.uba import loss_control, phi_control


def make_ds(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-1, 1, n)
    spikes = rng.choice(n, 25, replace=False)
    y[spikes] += rng.uniform(5, 9, 25)
    # some autocorrelation so that models have something to learn
    y = y + 0.5 * np.roll(y, 1)
    ts = pd.Series(y, index=pd.date_range("2015-01-01", periods=n, freq="D"))
    return create_data(ts, 10)


@pytest.fixture(scope="module")
def split():
    ds = make_ds()
    return ds.iloc[:250], ds.iloc[250:350]


SMALL = dict(cost=10, gamma=0.01, nk=9, degree=2, thresh=0.001, mtry=3, ntree=15, minsplit=10, cp=0.001)


def _pars_for(name):
    if "svm" in name:
        return {k: SMALL[k] for k in ("cost", "gamma")}
    if "mars" in name:
        return {k: SMALL[k] for k in ("nk", "degree", "thresh")}
    if "rf" in name:
        return {k: SMALL[k] for k in ("mtry", "ntree")}
    if "rpart" in name:
        return {k: SMALL[k] for k in ("minsplit", "cp")}
    if name == "mc.BDES":
        return {"ntrees": 30}
    return {}


def test_registry_order_matches_GetResults():
    assert len(WORKFLOW_NAMES) == 52
    assert WORKFLOW_NAMES[0] == "mc.lm" and WORKFLOW_NAMES[9] == "mc.lm_SMOTETPhi"
    assert WORKFLOW_NAMES[10] == "mc.svm" and WORKFLOW_NAMES[20] == "mc.mars"
    assert WORKFLOW_NAMES[30] == "mc.rf" and WORKFLOW_NAMES[40] == "mc.rpart"
    assert WORKFLOW_NAMES[50] == "mc.arima" and WORKFLOW_NAMES[51] == "mc.BDES"


@pytest.mark.parametrize("name", [n for n in WORKFLOW_NAMES if n not in ("mc.arima",)])
def test_every_workflow_runs(name, split):
    train, test = split
    res = run_workflow(name, train, test, rng=123, **_pars_for(name))
    ev = res["evaluation"]
    assert set(ev) == {"prec", "rec", "F1"}
    for v in ev.values():
        assert 0 <= v <= 1
    assert res["traintime"] >= 0 and res["trainpredtime"] >= res["traintime"]
    assert res["n_train"] > 0


def test_arima_workflow(split):
    train, test = split
    res = mc_arima(train, test, rng=1)
    assert set(res["evaluation"]) == {"prec", "rec", "F1"}
    assert len(res["order"]) == 3


def test_arima_fitted_on_full_series_has_right_length():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=120))
    m = AutoARIMA().fit(y[:80])
    f = m.fitted_on(y)
    assert f.shape == (120,)
    assert np.isfinite(f[80:]).all()


def test_eval_stats_perfect_prediction(split):
    train, test = split
    ph = phi_control(train["V10"].to_numpy())
    ls = loss_control(train["V10"].to_numpy())
    ev = eval_stats(train, test, test["V10"].to_numpy(), ph, ls)
    assert ev["prec"] == pytest.approx(1) and ev["rec"] == pytest.approx(1) and ev["F1"] == pytest.approx(1)
    ext = extended_stats(train, test, test["V10"].to_numpy(), ph, ls)
    assert ext["mad"] == 0 and ext["rmse"] == 0 and ext["F05"] == pytest.approx(1)


def test_svm_scaling_like_e1071():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * np.array([1, 100, 0.01])
    y = 1000 + 50 * X[:, 0] + rng.normal(size=200)
    m = SVMLearner(cost=150, gamma=0.001).fit(X, y)
    p = m.predict(X)
    # predictions are on the original scale of y
    assert abs(p.mean() - y.mean()) < 20
    assert np.corrcoef(p, y)[0, 1] > 0.8


def test_rpart_cp_controls_tree_size():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] ** 2 + rng.normal(0, 0.1, 300)
    big = RPartLearner(minsplit=10, cp=0.0).fit(X, y).m
    small = RPartLearner(minsplit=10, cp=0.2).fit(X, y).m
    assert big.get_n_leaves() > small.get_n_leaves()
    assert small.get_n_leaves() >= 1
    # minbucket = round(minsplit/3) = 3
    assert big.min_samples_leaf == 3


def test_embed_stats_includes_target_like_R():
    df = pd.DataFrame({"V10": [1.0, 2.0], "V9": [3.0, 4.0]}, index=pd.date_range("2020-01-01", periods=2))
    out = embed_stats(df)
    assert list(out.columns) == ["V10", "V9", "mean", "var"]
    np.testing.assert_allclose(out["mean"], [2.0, 3.0])
    np.testing.assert_allclose(out["var"], [2.0, 2.0])  # sample variance


def test_bdes_number_of_trees(split):
    from tsresamp.models import bagged_trees
    train, test = split
    tr = embed_stats(train[list(train.columns)[::-1]])
    models = bagged_trees(tr, embedding_dimension=10, nstats=2, ntrees=500, rng=np.random.default_rng(0))
    assert len(models) == 6 * 83
    feats = {tuple(f) for _, f, _ in models}
    assert ("V9",) in feats  # K = 2.5 -> seq_len -> 1:2 -> V10 ~ V9
    assert ("V9", "mean", "var") in feats
    assert len([f for f in feats if len(f) == 9]) == 1


def test_resampled_workflows_change_training_size(split):
    train, test = split
    base = learner_workflow("lm", None, train, test, rng=0)
    under = learner_workflow("lm", "UNDERB", train, test, rng=0)
    over = learner_workflow("lm", "OVERB", train, test, rng=0)
    assert under["n_train"] < base["n_train"] < over["n_train"]


def test_un_ov_parameters(split):
    train, test = split
    r1 = learner_workflow("svm", "UNDERB", train, test, rng=0, cost=10, gamma=0.01, un=0.2)
    r2 = learner_workflow("svm", "OVERB", train, test, rng=0, cost=10, gamma=0.01, ov=3)
    r3 = learner_workflow("svm", "SMOTEB", train, test, rng=0, cost=10, gamma=0.01, un=0.2, ov=3)
    assert r1["n_train"] < r2["n_train"]
    assert r3["n_train"] > 0


def test_rf_nodesize_semantics_like_randomForest():
    from tsresamp.models import RFLearner
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 200)
    m = RFLearner(mtry=2, ntree=5, seed=1).fit(X, y).m
    # regTree.c: nodes with <= nodesize cases are terminal, children may have 1 case
    assert m.min_samples_split == 6 and m.min_samples_leaf == 1
    m2 = RFLearner(mtry=2, ntree=5, seed=1, nodesize_mode="leaf").fit(X, y).m
    assert m2.min_samples_leaf == 5
    # with the R semantics the trees can isolate single cases (deeper trees)
    assert max(t.get_n_leaves() for t in m.estimators_) > max(t.get_n_leaves() for t in m2.estimators_)


def test_rpart_cp_is_cost_complexity_pruning():
    from tsresamp.models import rpart_regressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] ** 2 + rng.normal(0, 0.1, 300)
    # default: rpart's cp = minimal cost-complexity pruning with alpha = cp * SS_root (sklearn units: cp * var(y))
    t = rpart_regressor(y, minsplit=10, cp=0.01).fit(X, y)
    assert t.ccp_alpha == pytest.approx(0.01 * np.var(y)) and t.min_impurity_decrease == 0.0
    assert t.min_samples_split == 10 and t.min_samples_leaf == 3 and t.max_depth == 30
    # optional pre-pruning rule (the one the rpart documentation seems to describe)
    t2 = rpart_regressor(y, minsplit=10, cp=0.01, pruning="pre").fit(X, y)
    assert t2.min_impurity_decrease == pytest.approx(0.01 * np.var(y)) and t2.ccp_alpha == 0.0
    # pre-pruning is greedier: never more leaves than the post-pruned tree at the same threshold
    assert t2.get_n_leaves() <= t.get_n_leaves()


def test_tree_ranks_is_monotone_and_midpoint_consistent():
    """The rank transform used before the sklearn trees keeps the order of the
    values (same split positions as R), separates values that differ by one ulp
    (which sklearn's 1e-7 tolerance would merge) and maps a new value to the same
    side of a split as the R midpoint threshold."""
    from tsresamp.models import TreeRanks
    a = 0.063334
    x = np.array([[0.01], [0.02], [np.nextafter(a, 0)], [a], [0.07], [0.02]])
    r = TreeRanks().fit(x)
    t = r.transform(x)[:, 0]
    assert np.all(np.diff(t[np.argsort(x[:, 0], kind="stable")]) >= 0)
    assert t[2] + 1 == t[3]  # one ulp apart -> consecutive ranks
    assert t[1] == t[5]      # equal values -> equal ranks
    # a value below/above the R midpoint between 0.02 and 0.0633.. lands on the same side of rank midpoint
    mid = (0.02 + np.nextafter(a, 0)) / 2
    lo, hi = r.transform(np.array([[mid - 1e-3], [mid + 1e-3]]))[:, 0]
    assert lo < 1.5 < hi
    # outside the training range: clamped to the extreme ranks
    assert r.transform(np.array([[-1.0], [1.0]]))[:, 0].tolist() == [0.0, 4.0]
