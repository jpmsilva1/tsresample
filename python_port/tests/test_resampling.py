import numpy as np
import pandas as pd
import pytest

from tsresamp.data import create_data
from tsresamp.resampling import (RESAMPLERS, _bumps, _smote_exs, neighbours, rand_over_regress_B,
                                 rand_over_regress_T, rand_over_regress_TPhi, rand_under_regress_B,
                                 rand_under_regress_T, rand_under_regress_TPhi, smote_regress_B,
                                 smote_regress_T, smote_regress_TPhi)
from tsresamp.uba import phi, phi_control


def make_ds(n=400, seed=0, embed=10):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-1, 1, n)  # no low outliers -> rare cases are only the high tail
    spikes = rng.choice(n, 25, replace=False)
    y[spikes] += rng.uniform(5, 9, 25)  # rare high values
    ts = pd.Series(y, index=pd.date_range("2015-01-01", periods=n, freq="D"))
    return create_data(ts, embed)


@pytest.fixture(scope="module")
def ds():
    return make_ds()


def rare_mask(df, thr=0.9):
    y = df["V10"].to_numpy()
    pc = phi_control(y)
    return phi(y, pc) > thr


# ---------------------------------------------------------------- bumps
def test_bumps_partition_is_complete_and_sorted(ds):
    y = ds["V10"].to_numpy()
    pc = phi_control(y)
    obs, imp = _bumps(y, pc, 0.9, "sign")
    allpos = np.concatenate(obs)
    assert sorted(allpos.tolist()) == list(range(len(y)))
    assert np.all(np.diff(y[allpos]) >= 0)  # increasing y across the concatenated bumps
    assert len(obs) == 2 and imp[0] < 0.9 < imp[1]
    obs2, imp2 = _bumps(y, pc, 0.9, "thr")
    assert len(obs2) == 2


# ---------------------------------------------------------------- under
def test_under_B_balance(ds):
    rare = rare_mask(ds)
    out = rand_under_regress_B(ds, thr_rel=0.9, C_perc="balance", repl=False, rng=1)
    n_rare = int(rare.sum())
    assert out.shape[1] == ds.shape[1]
    # all rare cases kept, normal cases reduced to ~ the number of rare cases
    assert (out["V10"].to_numpy()[:n_rare] > 0).all()
    assert len(out) == n_rare + int(round(n_rare / (len(ds) - n_rare), 5) * (len(ds) - n_rare))
    assert len(out) <= 2 * n_rare + 1
    assert not out.index.duplicated().any()  # no replacement


def test_under_B_explicit_percentage(ds):
    rare = rare_mask(ds)
    n_norm = int((~rare).sum())
    out = rand_under_regress_B(ds, thr_rel=0.9, C_perc=[0.5], rng=2)
    assert len(out) == int(rare.sum()) + int(0.5 * n_norm)


def test_under_T_sorted_and_recency_bias(ds):
    rare = rare_mask(ds)
    out = rand_under_regress_T(ds, thr_rel=0.9, C_perc="balance", rng=3)
    assert out.index.is_monotonic_increasing
    assert len(out) == len(rand_under_regress_B(ds, thr_rel=0.9, C_perc="balance", rng=3))
    # recency bias: repeat many times, the mean position of kept normal cases is above the middle
    pos = ds.index.get_indexer(ds.index)
    normal_pos = pos[~rare]
    mid = normal_pos.mean()
    means = []
    for s in range(30):
        o = rand_under_regress_T(ds, thr_rel=0.9, C_perc=[0.3], rng=s)
        kept = ds.index.get_indexer(o.index)
        kept_norm = [p for p in kept if not rare[p]]
        means.append(np.mean(kept_norm))
    assert np.mean(means) > mid + 20


def test_under_TPhi_runs_and_prefers_relevant(ds):
    out = rand_under_regress_TPhi(ds, thr_rel=0.9, C_perc="balance", rng=4)
    assert out.index.is_monotonic_increasing
    # the kept normal cases must have phi > 0 (zero-probability cases are never chosen)
    y = ds["V10"].to_numpy()
    pc = phi_control(y)
    kept = ds.index.get_indexer(out.index)
    rel = phi(y[kept], pc)
    assert (rel > 0).all()


# ---------------------------------------------------------------- over
def test_over_B_balance(ds):
    rare = rare_mask(ds)
    n_rare, n_norm = int(rare.sum()), int((~rare).sum())
    out = rand_over_regress_B(ds, thr_rel=0.9, C_perc="balance", repl=True, rng=5)
    C = round((n_norm / 1) / n_rare, 5)
    assert len(out) == len(ds) + int(C * n_rare)
    # the first rows are the original data
    pd.testing.assert_frame_equal(out.iloc[:len(ds)], ds, check_freq=False)
    # replicas are rare cases (relevance >= 0.9)
    pc = phi_control(ds["V10"].to_numpy())
    assert (phi(out["V10"].to_numpy()[len(ds):], pc) >= 0.9).all()


def test_over_rejects_percentage_below_one(ds):
    with pytest.raises(ValueError):
        rand_over_regress_B(ds, thr_rel=0.9, C_perc=[0.5])


def test_over_T_and_TPhi(ds):
    rare = rare_mask(ds)
    n_rare = int(rare.sum())
    for f in (rand_over_regress_T, rand_over_regress_TPhi):
        out = f(ds, thr_rel=0.9, C_perc=[3], repl=True, rng=6)
        assert len(out) == len(ds) + 3 * n_rare
        pc = phi_control(ds["V10"].to_numpy())
        assert (phi(out["V10"].to_numpy()[len(ds):], pc) >= 0.9).all()
    # recency bias on the replicas
    rare_pos = np.where(rare)[0]
    mid = rare_pos.mean()
    means = []
    for s in range(30):
        o = rand_over_regress_T(ds, thr_rel=0.9, C_perc=[3], rng=s)
        rep = ds.index.get_indexer(o.index[len(ds):])
        means.append(rep.mean())
    assert np.mean(means) > mid


# ---------------------------------------------------------------- neighbours
def test_neighbours_excludes_self_and_breaks_ties_by_position():
    T = np.array([[0, 0, 9], [1, 0, 9], [0, 1, 9], [5, 5, 9], [1, 0, 9]], float)
    nn = neighbours(tgt=2, T=T, dist="Euclidean", p=2, k=2)
    assert nn.shape == (5, 2)
    assert 0 not in nn[0]
    # row 0: rows 1, 2 and 4 are at distance 1 -> lowest positions first
    assert nn[0].tolist() == [1, 2]
    assert nn[1].tolist() == [4, 0]  # row 4 is a duplicate of row 1 (distance 0)
    with pytest.raises(ValueError):
        neighbours(2, T, "Euclidean", 2, k=5)


# ---------------------------------------------------------------- smote
def test_smote_exs_generates_on_segments():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    bump = pd.DataFrame({"a": [0.0, 1.0, 0.0, 10.0], "b": [0.0, 0.0, 1.0, 10.0], "y": [5.0, 6.0, 7.0, 8.0]}, index=idx)
    pc = phi_control(np.concatenate([np.zeros(50), [5, 6, 7, 8]]))
    rng = np.random.default_rng(0)
    new = _smote_exs(bump, tgt=2, N=3, k=2, dist="Euclidean", p=2, variant="B", pc=pc, rng=rng)
    assert new.shape == (8, 3)
    assert new.index.isna().all()
    T = bump.to_numpy()
    for r in range(8):
        i = r // 2
        seed = T[i]
        # synthetic case lies on the segment between the seed and one of its neighbours
        found = False
        for j in range(4):
            if j == i:
                continue
            d = T[j, :2] - seed[:2]
            v = new.iloc[r, :2].to_numpy() - seed[:2]
            if np.linalg.norm(d) > 0:
                u = np.dot(v, d) / np.dot(d, d)
                if np.allclose(v, u * d, atol=1e-9) and -1e-9 <= u <= 1 + 1e-9:
                    # target is the matching weighted average
                    expect = (1 - u) * seed[2] + u * T[j, 2] if not np.isclose(d[-1], 0) else (seed[2] + T[j, 2]) / 2
                    assert np.isclose(new.iloc[r, 2], expect, atol=1e-9)
                    found = True
                    break
        assert found


def test_smote_exs_T_uses_most_recent_neighbour():
    # three cases: seed at origin, one neighbour old and near, one recent and far-ish (still within k=2)
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    bump = pd.DataFrame({"a": [0.0, 0.1, 1.0], "y": [5.0, 6.0, 7.0]}, index=idx)
    pc = phi_control(np.concatenate([np.zeros(50), [5, 6, 7]]))
    new = _smote_exs(bump, tgt=1, N=2, k=2, dist="Euclidean", p=2, variant="T", pc=pc, rng=np.random.default_rng(1),
                     r_index_quirk=False)
    # after ordering by time: rows = (01-01: 0.1), (01-02: 1.0), (01-03: 0.0)
    # the most recent neighbour of every row is the row with the highest time position among its k=2 NNs
    # for the 01-01 case (a=0.1) its NNs are 01-03 (a=0) and 01-02 (a=1) -> most recent = 01-03 (a=0)
    seed = 0.1
    gen = new.iloc[0, 0]
    assert 0.0 <= gen <= 0.1 + 1e-12


def test_smote_B_balance_sizes(ds):
    rare = rare_mask(ds)
    n = len(ds)
    n_rare, n_norm = int(rare.sum()), int((~rare).sum())
    out = smote_regress_B(ds, thr_rel=0.9, C_perc="balance", k=5, repl=True, rng=7)
    B = round(n / 2)
    C_norm, C_rare = B / n_norm, B / n_rare
    expect = int(C_norm * n_norm) + (int(C_rare - 1) * n_rare + int(n_rare * (C_rare - 1 - int(C_rare - 1)))) + n_rare
    assert len(out) == expect
    assert out.shape[1] == ds.shape[1] and list(out.columns) == list(ds.columns)
    assert not out.isna().any().any()
    # synthetic cases are inside the bounding box of the rare cases
    syn = out[out.index.isna()]
    rare_df = ds[rare]
    assert len(syn) > 0
    assert (syn.to_numpy() >= rare_df.to_numpy().min(axis=0) - 1e-9).all()
    assert (syn.to_numpy() <= rare_df.to_numpy().max(axis=0) + 1e-9).all()


def test_smote_T_TPhi_run(ds):
    for f in (smote_regress_T, smote_regress_TPhi):
        out = f(ds, thr_rel=0.9, C_perc="balance", k=5, repl=True, rng=8)
        assert not out.isna().any().any()
        assert len(out) == len(smote_regress_B(ds, thr_rel=0.9, C_perc="balance", k=5, repl=True, rng=8))


def test_smote_explicit_percentages_and_target_not_last(ds):
    out = smote_regress_B(ds, thr_rel=0.9, C_perc=[0.5, 2], k=5, rng=9)
    rare = rare_mask(ds)
    assert len(out) == int(0.5 * (~rare).sum()) + 2 * int(rare.sum())
    # target given by name in a different position
    ds2 = ds[["V10"] + [c for c in ds.columns if c != "V10"]]
    out2 = smote_regress_B(ds2, tgt="V10", thr_rel=0.9, C_perc=[0.5, 2], k=5, rng=9)
    assert list(out2.columns) == list(ds2.columns)
    assert len(out2) == len(out)


def test_smote_rejects_na(ds):
    bad = ds.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        smote_regress_B(bad)


def test_registry_has_nine():
    assert len(RESAMPLERS) == 9


def test_smote_un_ov_dict_applies_per_bump_side():
    # two-sided extremes: rare-low, normal, rare-high (3 bumps)
    rng = np.random.default_rng(5)
    n = 400
    y = rng.uniform(-1, 1, n)
    lo = rng.choice(n, 12, replace=False)
    hi = rng.choice(np.setdiff1d(np.arange(n), lo), 12, replace=False)
    y[lo] -= rng.uniform(5, 9, 12)
    y[hi] += rng.uniform(5, 9, 12)
    ds3 = create_data(pd.Series(y, index=pd.date_range("2015-01-01", periods=n, freq="D")), 10)
    yy = ds3["V10"].to_numpy()
    pc = phi_control(yy)
    obs, imp = _bumps(yy, pc, 0.9, "sign")
    assert len(obs) == 3
    with pytest.raises(ValueError):
        smote_regress_B(ds3, thr_rel=0.9, C_perc=[0.5, 2], k=5, rng=1)  # R semantics: wrong length
    out = smote_regress_B(ds3, thr_rel=0.9, C_perc={"un": 0.5, "ov": 2}, k=5, rng=1)
    n_norm = len(obs[1])
    n_rare = len(obs[0]) + len(obs[2])
    assert len(out) == int(0.5 * n_norm) + 2 * n_rare
    for f in (smote_regress_T, smote_regress_TPhi):
        o = f(ds3, thr_rel=0.9, C_perc={"un": 0.5, "ov": 2}, k=5, repl=True, rng=2)
        assert len(o) == len(out) and not o.isna().any().any()


def test_smote_T_r_index_quirk_reproduces_R_pairing():
    # R: T is filled from the bump in target order, neighbours computed on the time-ordered copy,
    # and the time-order indices are applied to T. With the quirk the generated case lies on a
    # segment between two rows of the *target-ordered* matrix.
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-04"])
    bump = pd.DataFrame({"a": [0.0, 0.1, 1.0, 2.0], "y": [5.0, 6.0, 7.0, 8.0]}, index=idx)
    pc = phi_control(np.concatenate([np.zeros(50), [5, 6, 7, 8]]))
    q = _smote_exs(bump, tgt=1, N=2, k=2, dist="Euclidean", p=2, variant="T", pc=pc, rng=np.random.default_rng(3), r_index_quirk=True)
    f = _smote_exs(bump, tgt=1, N=2, k=2, dist="Euclidean", p=2, variant="T", pc=pc, rng=np.random.default_rng(3), r_index_quirk=False)
    assert q.shape == f.shape == (4, 2)
    T = bump.to_numpy()
    for r in range(4):
        # quirk: segment between T[r] (target order) and some other row of T
        seed = T[r]
        assert any(min(seed[0], T[j, 0]) - 1e-9 <= q.iloc[r, 0] <= max(seed[0], T[j, 0]) + 1e-9 for j in range(4) if j != r)
    assert not np.allclose(q.to_numpy(), f.to_numpy())
