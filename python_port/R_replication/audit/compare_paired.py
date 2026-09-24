"""Paired check of rpart / randomForest / auto.arima: R predictions (fit_others.R) vs the
Python learners on the SAME train/test windows.  F1 = utility-based F1 of the experiments."""
import sys, glob, os, re
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'Py_Code'))
import numpy as np, pandas as pd
from tsresamp.models import eval_stats, RPartLearner, RFLearner, AutoARIMA
from tsresamp.uba import phi_control, loss_control
W = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1] in ('rpart','rf','arima') else 'win'
PARS = {1: dict(mtry=5, ntree=1500, minsplit=10, cp=0.01), 2: dict(mtry=7, ntree=750, minsplit=10, cp=0.001),
        3: dict(mtry=7, ntree=500, minsplit=10, cp=0.001), 4: dict(mtry=7, ntree=750, minsplit=10, cp=0.1),
        10: dict(mtry=7, ntree=500, minsplit=10, cp=0.001)}
which = [a for a in sys.argv[1:] if a in ('rpart', 'rf', 'arima')]  # e.g. rpart rf arima
for learner in which:
    groups = sorted(set(re.sub(r"_it\d+_%s_pred.csv" % learner, "", os.path.basename(f)) for f in glob.glob(f"{W}/*_{learner}_pred.csv")))
    for g in groups:
        ds = int(g.split("_")[0][2:]); pars = PARS[ds]
        rows = []
        for f in sorted(glob.glob(f"{W}/{g}_it*_{learner}_pred.csv")):
            name = os.path.basename(f).replace(f"_{learner}_pred.csv", "")
            tr = pd.read_csv(f"{W}/{name}_train.csv"); te = pd.read_csv(f"{W}/{name}_test.csv")
            X, y = tr.iloc[:, :-1].to_numpy(), tr.iloc[:, -1].to_numpy(); Xt = te.iloc[:, :-1].to_numpy()
            yo = pd.read_csv(f"{W}/{name}_orig.csv").iloc[:, -1].to_numpy() if os.path.exists(f"{W}/{name}_orig.csv") else y
            ph = phi_control(yo, method="extremes"); ls = loss_control(yo)
            pr = pd.read_csv(f).pred.to_numpy()
            extra = ""
            if learner == "rpart":
                m = RPartLearner(minsplit=pars["minsplit"], cp=pars["cp"], seed=0).fit(X, y); pp = m.predict(Xt)
                extra = f"leaves py={m.m.get_n_leaves()}"
            elif learner == "rf":
                m = RFLearner(mtry=pars["mtry"], ntree=pars["ntree"], seed=int(name[-2:]), n_jobs=4).fit(X, y); pp = m.predict(Xt)
            else:
                m = AutoARIMA().fit(y); pp = m.fitted_on(np.concatenate([y, te.iloc[:, -1].to_numpy()]))[len(y):]
                extra = f"order py={m.m.order}"
            f1r = eval_stats(tr, te, pr, ph, ls)["F1"]; f1p = eval_stats(tr, te, pp, ph, ls)["F1"]
            rows.append((name, f1r, f1p, float(np.corrcoef(pr, pp)[0, 1]) if pr.std() > 0 and pp.std() > 0 else np.nan,
                         float(np.max(np.abs(pr - pp)) / (np.std(te.iloc[:, -1]) + 1e-12)), extra))
        d = pd.DataFrame(rows, columns=["window", "F1_R", "F1_py", "corr_pred", "maxdiff/sd(y)", "extra"])
        diff = d.F1_py - d.F1_R
        print(f"{learner:6s} {g:12s} n={len(d)}  mean F1 R {d.F1_R.mean():.3f}  py {d.F1_py.mean():.3f}  "
              f"diff {diff.mean():+.3f} (sd {diff.std(ddof=1):.3f}, SE {diff.std(ddof=1)/np.sqrt(len(d)):.3f})  "
              f"median corr(pred) {d.corr_pred.median():.3f}  median maxdiff/sd {d['maxdiff/sd(y)'].median():.2f}")
        if "-v" in os.environ.get("VERB", ""):
            print(d.to_string(index=False))
