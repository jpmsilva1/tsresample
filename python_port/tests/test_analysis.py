import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tsresamp.analysis import (MODELS, WLdef, get_results, get_results_runtime, paired_comparisons, wilcox_test_paired)
from tsresamp.estimation import ComparisonResults, EstimationTask, MonteCarlo, Split, Workflow, WorkflowResults


def build_results(n_tasks=3, n_reps=20, seed=0, names=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    et = EstimationTask(method=MonteCarlo(nReps=n_reps))
    res = ComparisonResults(et)
    shift = {"A": 0.0, "B": 0.15, "C": -0.1}
    for t in range(n_tasks):
        for nm in names:
            its = []
            for i in range(n_reps):
                base = 0.5 + shift[nm] + rng.normal(0, 0.05)
                its.append({"evaluation": {"prec": base, "rec": base - 0.1, "F1": base - 0.05},
                            "traintime": 1.0 + t, "trainpredtime": 1.5 + t})
            res.add(f"ds{t + 1}", WorkflowResults(Workflow(nm), its, [Split(i + 1, 0, range(0), range(0)) for i in range(n_reps)]))
    return res


def test_models_vector():
    assert len(MODELS) == 52 and MODELS[0] == "mc.lm" and MODELS[-1] == "mc.BDES" and MODELS[10] == "mc.svm"


def test_get_results_table():
    res = build_results()
    tab = get_results(res)
    assert list(tab.columns) == ["model", "prec", "rec", "F1"]
    assert list(tab.model) == ["A", "B", "C"]
    assert tab.set_index("model").loc["B", "F1"] > tab.set_index("model").loc["A", "F1"] > tab.set_index("model").loc["C", "F1"]
    med = get_results(res, F1_summary="median")
    assert np.isfinite(med.F1).all()


def test_get_results_runtime():
    res = build_results()
    tab = get_results_runtime(res, n_reps=10)
    assert list(tab.columns) == ["model", "tr.time", "pr.time", "tot.time"]
    assert np.allclose(tab["tr.time"], 1.0) and np.allclose(tab["pr.time"], 0.5) and np.allclose(tab["tot.time"], 1.5)


def test_wilcox_matches_R_conventions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20)
    y = x + rng.normal(0.5, 0.3, size=20)
    p_exact = wilcox_test_paired(x, y)
    p_scipy = stats.wilcoxon(x - y, method="exact").pvalue
    assert p_exact == pytest.approx(p_scipy)
    # n >= 50 -> normal approximation with continuity correction
    x50 = rng.normal(size=50)
    y50 = x50 + rng.normal(0.2, 0.5, size=50)
    p_apx = wilcox_test_paired(x50, y50)
    assert p_apx == pytest.approx(stats.wilcoxon(x50 - y50, method="approx", correction=True).pvalue)
    # all zeros -> NaN (R raises an error, caught by pairedComparisons)
    assert np.isnan(wilcox_test_paired(np.ones(5), np.ones(5)))


def test_paired_comparisons_structure():
    res = build_results()
    pres = paired_comparisons(res, "A", maxs=[True, True, True], p_value=0.05)
    assert set(pres) == {"prec", "rec", "F1"}
    W = pres["F1"]["WilcoxonSignedRank.test"]["ds1"]
    assert list(W.index) == ["A", "B", "C"]  # baseline first
    assert np.isnan(W.loc["A", "p.value"])
    assert W.loc["B", "DiffMedScores"] < 0 and W.loc["C", "DiffMedScores"] > 0
    assert W.loc["B", "p.value"] < 0.05
    assert pres["F1"]["baseline"] == "A"
    assert pres["F1"]["avgRksWFs"].idxmin() == "B"  # with maxs=TRUE the best has rank 1
    assert pres["F1"]["F.test"] is not None
    # baseline chosen automatically = lowest average rank
    auto = paired_comparisons(res, maxs=[True, True, True])
    assert auto["F1"]["baseline"] == "B"


def test_WLdef_counts():
    res = build_results(n_tasks=4)
    WL = WLdef(res, "A", "F1", 0.05)
    assert list(WL.index) == ["B", "C"]
    assert list(WL.columns) == ["Win", "sigWin", "Loss", "SigLoss", "Tie"]
    assert WL.loc["B", "Win"] == 4 and WL.loc["B", "sigWin"] == 4 and WL.loc["B", "Loss"] == 0
    assert WL.loc["C", "Loss"] == 4 and WL.loc["C", "SigLoss"] == 4 and WL.loc["C", "Win"] == 0
    assert (WL.sum(axis=1) - WL["sigWin"] - WL["SigLoss"] == 4).all()


def test_paired_comparisons_uses_common_workflows():
    from tsresamp.analysis import common_workflows
    res = build_results(n_tasks=2)
    # drop workflow C from the second task
    del res.tasks["ds2"]["C"]
    assert common_workflows(res) == ["A", "B"]
    pres = paired_comparisons(res, "A", maxs=[True] * 3)
    assert list(pres["F1"]["WilcoxonSignedRank.test"]["ds2"].index) == ["A", "B"]
    WL = WLdef(res, "A", "F1")
    assert list(WL.index) == ["B"] and WL.loc["B", "Win"] == 2
    with pytest.raises(ValueError):
        paired_comparisons(res, "C")
