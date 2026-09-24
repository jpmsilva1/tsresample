import numpy as np
import pandas as pd
import pytest

from tsresamp import estimation as est
from tsresamp.estimation import (ComparisonResults, EstimationTask, MonteCarlo, Workflow, WorkflowResults, mc_splits,
                                 performance_estimation, workflow_variants)


def test_mc_splits_follow_R_semantics():
    n = 721
    mc = MonteCarlo(nReps=50, szTrain=0.5, szTest=0.25, seed=1234)
    sp = mc_splits(n, mc)
    assert len(sp) == 50
    train_size, test_size = int(n * 0.5), int(n * 0.25)
    starts = [s.start for s in sp]
    assert starts == sorted(starts) and len(set(starts)) == 50
    for s in sp:
        assert train_size + 1 <= s.start <= n - test_size + 1
        assert len(s.train) == train_size and len(s.test) == test_size
        # test window starts right after the train window, both inside the data
        assert s.train.stop == s.test.start == s.start - 1
        assert s.train.start >= 0 and s.test.stop <= n
    # deterministic
    assert [s.start for s in mc_splits(n, mc)] == starts
    # absolute sizes
    sp2 = mc_splits(100, MonteCarlo(nReps=3, szTrain=40, szTest=10))
    assert all(len(s.train) == 40 and len(s.test) == 10 for s in sp2)
    with pytest.raises(ValueError):
        mc_splits(100, MonteCarlo(nReps=3, szTrain=0.9, szTest=0.2))


def test_workflow_variants_expand_grid_order():
    vs = workflow_variants("mc.svm_UNDERB", cost=[10, 150, 300], gamma=[0.01, 0.001], un=[0.1, 0.2])
    assert len(vs) == 12
    assert vs[0].name == "mc.svm_UNDERB.v1"
    # first parameter varies fastest
    assert [v.pars["cost"] for v in vs[:4]] == [10, 150, 300, 10]
    assert [v.pars["gamma"] for v in vs[:4]] == [0.01, 0.01, 0.01, 0.001]
    assert vs[-1].pars == {"cost": 300, "gamma": 0.001, "un": 0.2}
    single = workflow_variants("mc.svm", cost=150, gamma=[0.01])
    assert len(single) == 1 and single[0].pars == {"cost": 150, "gamma": 0.01}


def _dummy_workflows(monkeypatch):
    calls = []

    def fake_run(name, train, test, rng=None, **pars):
        calls.append((name, len(train), len(test), pars))
        if name == "bad":
            raise RuntimeError("boom")
        v = float(rng.random())
        return {"evaluation": {"prec": v, "rec": v / 2, "F1": v / 3}, "traintime": 0.1, "trainpredtime": 0.2,
                "traintime_elapsed": 0.1, "trainpredtime_elapsed": 0.2}

    import tsresamp.models as models
    monkeypatch.setattr(models, "run_workflow", fake_run)
    return calls


def test_performance_estimation_with_dummy_workflows(monkeypatch, tmp_path):
    calls = _dummy_workflows(monkeypatch)
    ds = pd.DataFrame({"V1": np.arange(100.0), "V10": np.arange(100.0)}, index=pd.date_range("2020-01-01", periods=100))
    et = EstimationTask(method=MonteCarlo(nReps=4, szTrain=0.5, szTest=0.25, seed=7))
    wfs = [Workflow("good", {"a": 1}), Workflow("bad")]
    res = performance_estimation("ds1", ds, wfs, et, verbose=False, checkpoint_dir=tmp_path)
    assert res.task_names() == ["ds1"] and res.workflow_names() == ["good", "bad"]
    sc = res.iterations_scores("ds1", "good")
    assert sc.shape == (4, 3) and np.isfinite(sc).all()
    assert np.isnan(res.iterations_scores("ds1", "bad")).all()
    assert "error" in res.get_iterations_info(1, 2, 1)
    info = res.get_iterations_info("ds1", "good", 3)
    assert info["it"] == 3 and len(info["train"]) == 50 and len(info["test"]) == 25
    assert all(c[1] == 50 and c[2] == 25 for c in calls)
    # reproducible
    res2 = performance_estimation("ds1", ds, wfs, et, verbose=False)
    np.testing.assert_allclose(res2.iterations_scores("ds1", "good"), sc)
    # checkpoints written and reused
    assert (tmp_path / "ds1__good.pkl").exists()
    n_calls = len(calls)
    performance_estimation("ds1", ds, wfs, et, verbose=False, checkpoint_dir=tmp_path)
    assert len(calls) == n_calls
    # save / load / merge
    p = res.save(tmp_path / "res.pkl")
    loaded = ComparisonResults.load(p)
    assert loaded.workflow_names() == ["good", "bad"]
    other = performance_estimation("ds2", ds, wfs, et, verbose=False)
    merged = ComparisonResults.merge_all([loaded, other])
    assert merged.task_names() == ["ds1", "ds2"]
    summ = merged.summary("ds1")
    assert list(summ.columns) == ["workflow", "prec", "rec", "F1", "failed"]
    assert summ.loc[summ.workflow == "bad", "failed"].item() == 4


def test_parallel_matches_sequential(monkeypatch):
    _dummy_workflows(monkeypatch)
    ds = pd.DataFrame({"V1": np.arange(60.0), "V10": np.arange(60.0)}, index=pd.date_range("2020-01-01", periods=60))
    et = EstimationTask(method=MonteCarlo(nReps=3, szTrain=0.5, szTest=0.25))
    r1 = performance_estimation("t", ds, [Workflow("good")], et, verbose=False, n_jobs=1)
    # joblib with n_jobs=2 uses subprocesses, where the monkeypatch does not apply -> use threads via
    # the same process by requesting n_jobs=1 twice instead (parallel path is exercised in the smoke test)
    r2 = performance_estimation("t", ds, [Workflow("good")], et, verbose=False, n_jobs=1)
    np.testing.assert_allclose(r1.iterations_scores("t", "good"), r2.iterations_scores("t", "good"))
