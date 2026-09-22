"""M0 oracle probe: utility surface U(yhat, y) of Ribeiro (2011) §3.3-3.4, checked against R.

Research tooling, not library code. Derived from the thesis text and recorded numerical
output only (Blueprint/docs/PROVENANCE.md); see Blueprint/docs/adr/0014-utility-surface.md.

    python m0_utility.py sweep [DS01 DS02 ...]   # prec/rec/F1 vs raw_iterations_by_dataset_v2
    python m0_utility.py sera  [DS01 ...]        # SERA (step=0.01) vs raw_sera_by_dataset, two-sided
    python m0_utility.py fixture OUT.json        # freeze Blueprint/tests/fixtures/metric_oracle.json
"""
import csv, glob, json, math, os, sys
from collections import defaultdict

import numpy as np
from scipy.interpolate import CubicHermiteSpline

# Clone of github.com/jpmsilva1/ts-resampling-replication at the commit pinned in Blueprint/docs/REPLICATION.md §1.
ROOT = os.path.join(os.environ["TSRESAMPLE_REPLICATION_ROOT"], "Results (Clean)", "Results Data")
INF = math.inf
T_E, P, EMPTY = 0.9, 0.5, 1e-5


def phi_fn(ctrl):
    x, f, d = (np.array(c) for c in zip(*ctrl))
    s = CubicHermiteSpline(x, f, d)
    # Exact endpoint phi at/beyond the knots and clip to [0, 1]: spline evaluation can give 1 - 4e-16 at x[-1]
    # and -2e-16 near the median, which move cases across SERA's t = 1 and t = 0 grid points.
    return lambda v: np.where(v <= x[0], f[0], np.where(v >= x[-1], f[-1], np.clip(s(np.clip(v, x[0], x[-1])), 0, 1)))


def bumps(ctrl):
    """Alg 3.2 over the critical points, with the initialisation resolved by ADR-0014.

    Returns [(b_minus, b_star), ...]. If phi first moves down, bump 1 is open from -inf;
    otherwise the leading constant-phi run is averaged into b_minus of bump 1 and a
    <-inf,-inf> bump precedes it.
    """
    S = [(c[0], c[1]) for c in ctrl if c[2] == 0]
    ys, ph = [s[0] for s in S], [s[1] for s in S]
    n = len(S)
    first = next((j for j in range(n - 1) if ph[j] != ph[j + 1]), None)
    if first is None:
        raise ValueError("phi is constant on the control points: no bumps")
    if ph[first] > ph[first + 1]:
        P_, inb = [[-INF, None]], True
    else:
        P_, inb = [[-INF, -INF]], False
    l = 0
    for i in range(n - 1):
        if ph[i] < ph[i + 1] and not inb:
            P_.append([sum(ys[l:i + 1]) / (i - l + 1), None]); inb = True
        elif ph[i] > ph[i + 1] and inb:
            P_[-1][1] = sum(ys[l:i + 1]) / (i - l + 1); inb = False
        if ph[i] != ph[i + 1]:
            l = i + 1
    if inb:
        P_[-1][1] = sum(ys[l:]) / (n - l)
    else:
        P_.append([sum(ys[l:]) / (n - l), INF])
    return [tuple(b) for b in P_]


def max_admissible_loss(P_):
    """Def 3.12; a non-finite value takes the adjacent bump's (thesis p. 85)."""
    edges = [b[0] for b in P_] + [INF]
    d = []
    for i, (bm, bs) in enumerate(P_):
        a, b = abs(bm - bs), abs(bs - edges[i + 1])
        d.append(2 * min(a, b) if not (math.isnan(a) or math.isnan(b)) else INF)
    for i in range(len(d)):
        if not math.isfinite(d[i]):
            nb = [d[j] for j in (i + 1, i - 1) if 0 <= j < len(d) and math.isfinite(d[j])]
            d[i] = nb[0]
    return np.array(d)


def utility(yh, y, phi, P_, p=P):
    """Eq 3.19: U = phi(y)(1 - Gamma_B) - phi_p(yh, y) Gamma_C, L = |yh - y|."""
    D = max_admissible_loss(P_)
    edges = np.array([b[0] for b in P_] + [INF])
    stars = np.array([-INF] + [b[1] for b in P_] + [INF])  # stars[g+1] is bump g's max
    g = np.searchsorted(edges, y, side="right") - 1
    below = yh < y
    lb = np.abs(y - np.where(below, edges[g], edges[g + 1]))          # Eq 3.15
    lc = np.abs(y - np.where(below, stars[g], stars[g + 2]))          # Eq 3.18
    LB, LC, L = np.minimum(D[g], lb), np.minimum(D[g], lc), np.abs(yh - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        gb = np.where(L < LB, L / LB, 1.0)                           # Eq 3.13
        gc = np.where(L < LC, L / LC, 1.0)
    fy, fyh = phi(y), phi(yh)
    return fy * (1 - gb) - ((1 - p) * fyh + p * fy) * gc


def prec_rec_f1(y, yh, ctrl):
    """SPEC §4.7 conventions: >= t_E, |1+u|, empty selection -> 1e-5."""
    phi = phi_fn(ctrl); u = utility(yh, y, phi, bumps(ctrl)); fy, fyh = phi(y), phi(yh)
    sr, sp = fy >= T_E, fyh >= T_E
    rec = np.abs(1 + u[sr]).sum() / np.abs(1 + fy[sr]).sum() if sr.any() else EMPTY
    prec = np.abs(1 + u[sp]).sum() / np.abs(1 + fyh[sp]).sum() if sp.any() else EMPTY
    return float(prec), float(rec), 0.0 if prec == 0 or rec == 0 else float(2 * prec * rec / (prec + rec))


def sera(y, yh, ctrl, step=0.01):
    """Harness SERA (ADR-0009 amendment 2): uniform grid, trapezoid; two-sided splits only."""
    fy = phi_fn(ctrl)(y); t = np.arange(0, 1 + step / 2, step); e2 = (y - yh) ** 2
    ser = np.array([e2[fy >= ti].sum() for ti in t])
    return float(np.trapezoid(ser, t))


def two_sided(ctrl):
    return ctrl[0][1] == 1 and ctrl[-1][1] == 1


def load_ctrl(ds, wf):
    out = defaultdict(list)
    with open(f"{ROOT}/raw_predictions_by_dataset/{ds}/{wf}_phi_ctrl.csv") as f:
        for r in csv.DictReader(f):
            out[int(r["iteration"])].append((float(r["ctrl_x"]), float(r["ctrl_phi"]), float(r["ctrl_deriv"])))
    return out


def load_preds(ds, wf):
    out = defaultdict(lambda: ([], []))
    with open(f"{ROOT}/raw_predictions_by_dataset/{ds}/{wf}_predictions.csv") as f:
        r = csv.reader(f); next(r)
        for it, _, a, b in r:
            out[int(it)][0].append(float(a)); out[int(it)][1].append(float(b))
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def load_prf(ds):
    out = {}
    with open(glob.glob(f"{ROOT}/raw_iterations_by_dataset_v2/{ds}_*.csv")[0]) as f:
        for r in csv.DictReader(f):
            out[(r["workflow"], int(r["iteration"]))] = (float(r["prec"]), float(r["rec"]), float(r["F1"]))
    return out


def load_sera(ds):
    with open(f"{ROOT}/raw_sera_by_dataset/{ds}.csv") as f:
        return {(r["workflow"], int(r["iteration"])): float(r["sera"]) for r in csv.DictReader(f)}


def workflows(ds):
    return sorted(os.path.basename(p)[:-len("_predictions.csv")]
                  for p in glob.glob(f"{ROOT}/raw_predictions_by_dataset/{ds}/*_predictions.csv"))


def shape(ctrl):
    return tuple(c[1] for c in ctrl)


def cmd_sweep(dss):
    for ds in dss:
        rec = load_prf(ds); by_shape = defaultdict(lambda: [0, 0, 0.0])
        for wf in workflows(ds):
            ctrl, pr = load_ctrl(ds, wf), load_preds(ds, wf)
            for it, (y, yh) in pr.items():
                d = max(abs(a - b) for a, b in zip(prec_rec_f1(y, yh, ctrl[it]), rec[(wf, it)]))
                s = by_shape[shape(ctrl[it])]; s[0] += 1; s[1] += not d <= 1e-6; s[2] = max(s[2], d)
        print(ds, "  ".join(f"phi={k}: splits={v[0]} over_1e-6={v[1]} worst={v[2]:.2e}" for k, v in sorted(by_shape.items())), flush=True)


def cmd_sera(dss):
    for ds in dss:
        rec = load_sera(ds); n = bad = 0; worst = 0.0
        for wf in workflows(ds):
            ctrl, pr = load_ctrl(ds, wf), load_preds(ds, wf)
            for it, (y, yh) in pr.items():
                if not two_sided(ctrl[it]) or (wf, it) not in rec:
                    continue
                r = abs(sera(y, yh, ctrl[it]) / rec[(wf, it)] - 1); n += 1; bad += not r <= 1e-6; worst = max(worst, r)
        print(f"{ds} SERA two-sided splits={n} over_1e-6_rel={bad} worst_rel={worst:.2e}", flush=True)


# (dataset, %Rare per paper Table 1, [(workflow, iteration), ...]); DS05 (3.5%) is too large to embed.
FIXTURE = [
    ("DS10", 4.8, [("mc.lm_OVERB", 1)]),
    ("DS01", 9.9, [("mc.lm_OVERB", 1), ("mc.svm", 1), ("mc.rf_SMOTET", 2)]),
    ("DS04", 13.3, [("mc.rpart_UNDERTPhi", 1), ("mc.lm", 3)]),
    ("DS09", 21.1, [("mc.lm_OVERB", 1), ("mc.rf_SMOTET", 1)]),
]


def cmd_fixture(out):
    cases = []
    for ds, pct, picks in FIXTURE:
        prf, sr = load_prf(ds), load_sera(ds)
        for wf, it in picks:
            ctrl = load_ctrl(ds, wf)[it]; y, yh = load_preds(ds, wf)[it]
            R = prf[(wf, it)]
            assert max(abs(a - b) for a, b in zip(prec_rec_f1(y, yh, ctrl), R)) <= 1e-6, (ds, wf, it)
            c = {"dataset": ds, "paper_pct_rare": pct, "workflow": wf, "iteration": it,
                 "ctrl": [list(p) for p in ctrl], "y_true": y.tolist(), "y_pred": yh.tolist(),
                 "prec": R[0], "rec": R[1], "F1": R[2]}
            if two_sided(ctrl):
                assert abs(sera(y, yh, ctrl) / sr[(wf, it)] - 1) <= 1e-6, (ds, wf, it)
                c["sera_step_0_01"] = sr[(wf, it)]
            cases.append(c)
    meta = {"source": "Results (Clean)/Results Data/raw_iterations_by_dataset_v2 (prec/rec/F1, R uba::util) and "
                      "raw_sera_by_dataset (SERA, harness sera_metric.py, step=0.01); y_true/y_pred/ctrl from "
                      "raw_predictions_by_dataset. Recorded values, not computed by our code.",
            "adr": "Blueprint/docs/adr/0014-utility-surface.md", "t_E": T_E, "p": P,
            "tolerance": {"prec_rec_F1_abs": 1e-6, "sera_rel": 1e-6},
            "sera_note": "sera_step_0_01 present only on two-sided splits (ctrl phi = 1 at both ends)",
            "excluded": "DS19 phi=(1,0,0) splits: known residual, ADR-0014"}
    with open(out, "w") as f:
        json.dump({"_meta": meta, "cases": cases}, f, separators=(",", ":"))
    print(out, os.path.getsize(out), "bytes,", len(cases), "cases")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    default = [f"DS{i:02d}" for i in range(1, 25)]
    {"sweep": lambda: cmd_sweep(args or default), "sera": lambda: cmd_sera(args or default),
     "fixture": lambda: cmd_fixture(args[0])}[cmd]()
