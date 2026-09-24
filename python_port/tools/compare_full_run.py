"""Full cluster run (24 data sets, 52 workflows, 50 reps) vs the article's Tables 3, 4, 5
(wins / losses by Wilcoxon paired test) and vs the independent R replication report
(J. P. Miranda, F1 per data set and workflow).

    .venv/bin/python tools/compare_full_run.py results/apuana [--override results/mars_v2] [--out results_cluster]

--override DIR: replace, for every data set, the workflows found in DIR/exp_ds<i>_*.pkl
(e.g. the mc.mars* workflows re-run locally with the earth port) before comparing.
--out DIR: where to write the summaries (default: the results directory).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(PY, "Py_Code"))
from tsresamp.estimation import ComparisonResults  # noqa: E402
from tsresamp.analysis import WLdef, get_results  # noqa: E402

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
def opt(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
RES = argv[0] if argv else os.path.join(PY, "results", "apuana")
OVERRIDE = opt("--override"); OUT = opt("--out") or RES
L = ["lm", "svm", "mars", "rf", "rpart"]
STRAT = {"U_B": "UNDERB", "U_T": "UNDERT", "U_TPhi": "UNDERTPhi", "O_B": "OVERB", "O_T": "OVERT",
         "O_TPhi": "OVERTPhi", "SM_B": "SMOTEB", "SM_T": "SMOTET", "SM_TPhi": "SMOTETPhi"}

def cell(s):
    w, sw, l, sl = map(int, re.findall(r"\d+", s)); return (w, sw, l, sl)

# ---- article, transcribed from the PDF text layer (pages 14-15)
T3 = {"U_B": ["19 (18) / 5 (4)", "8 (6) / 16 (8)", "15 (12) / 9 (7)", "18 (17) / 6 (2)", "12 (8) / 12 (3)"],
      "O_B": ["18 (17) / 6 (3)", "7 (6) / 17 (10)", "17 (17) / 7 (4)", "20 (15) / 4 (1)", "11 (9) / 13 (8)"],
      "SM_B": ["19 (18) / 5 (3)", "7 (6) / 17 (10)", "18 (17) / 6 (4)", "20 (20) / 4 (1)", "10 (10) / 14 (7)"]}
T4 = {"U_T": ["14 (2) / 10 (0)", "10 (0) / 14 (0)", "11 (0) / 13 (2)", "12 (1) / 12 (2)", "14 (4) / 10 (0)"],
      "U_TPhi": ["15 (10) / 9 (3)", "11 (5) / 13 (4)", "17 (6) / 7 (1)", "16 (6) / 8 (3)", "16 (7) / 8 (5)"],
      "O_T": ["14 (8) / 10 (9)", "12 (5) / 12 (6)", "11 (3) / 13 (4)", "8 (4) / 16 (4)", "12 (3) / 12 (2)"],
      "O_TPhi": ["14 (9) / 10 (7)", "12 (4) / 12 (7)", "11 (3) / 13 (2)", "8 (2) / 16 (5)", "14 (3) / 10 (2)"],
      "SM_T": ["6 (5) / 18 (13)", "10 (5) / 14 (10)", "9 (6) / 15 (10)", "9 (3) / 15 (10)", "8 (1) / 15 (11)"],
      "SM_TPhi": ["6 (4) / 18 (11)", "9 (6) / 15 (12)", "12 (4) / 12 (6)", "12 (5) / 12 (10)", "6 (4) / 17 (9)"]}
T5 = {  # learner -> strategy -> (vs ARIMA, vs BDES)
 "lm": {"U_B": ("18 (18) / 6 (3)", "22 (22) / 2 (2)"), "U_T": ("18 (18) / 6 (3)", "22 (22) / 2 (2)"), "U_TPhi": ("18 (18) / 6 (5)", "22 (22) / 2 (2)"),
        "O_B": ("21 (18) / 3 (2)", "22 (22) / 2 (2)"), "O_T": ("18 (18) / 6 (3)", "22 (22) / 2 (2)"), "O_TPhi": ("18 (18) / 6 (3)", "22 (22) / 2 (2)"),
        "SM_B": ("20 (18) / 4 (3)", "22 (22) / 2 (2)"), "SM_T": ("18 (17) / 6 (5)", "22 (20) / 2 (2)"), "SM_TPhi": ("18 (18) / 6 (5)", "22 (20) / 2 (2)")},
 "svm": {"U_B": ("21 (21) / 3 (3)", "22 (22) / 2 (1)"), "U_T": ("21 (21) / 3 (3)", "22 (22) / 2 (1)"), "U_TPhi": ("20 (20) / 4 (4)", "22 (22) / 2 (2)"),
         "O_B": ("21 (21) / 3 (1)", "22 (22) / 2 (2)"), "O_T": ("21 (21) / 3 (3)", "22 (22) / 2 (2)"), "O_TPhi": ("21 (21) / 3 (3)", "22 (22) / 2 (2)"),
         "SM_B": ("19 (19) / 5 (1)", "22 (22) / 2 (2)"), "SM_T": ("20 (20) / 4 (3)", "20 (20) / 4 (2)"), "SM_TPhi": ("19 (19) / 5 (4)", "22 (20) / 2 (2)")},
 "mars": {"U_B": ("23 (18) / 1 (1)", "21 (20) / 3 (3)"), "U_T": ("20 (18) / 4 (2)", "21 (19) / 3 (2)"), "U_TPhi": ("22 (19) / 2 (2)", "21 (21) / 3 (3)"),
          "O_B": ("19 (18) / 5 (1)", "22 (22) / 2 (2)"), "O_T": ("18 (18) / 6 (2)", "22 (22) / 2 (2)"), "O_TPhi": ("18 (18) / 6 (2)", "22 (22) / 2 (2)"),
          "SM_B": ("19 (19) / 5 (1)", "22 (22) / 2 (2)"), "SM_T": ("19 (19) / 5 (4)", "22 (22) / 2 (2)"), "SM_TPhi": ("19 (19) / 5 (4)", "22 (22) / 2 (2)")},
 "rf": {"U_B": ("19 (18) / 5 (1)", "19 (18) / 5 (2)"), "U_T": ("21 (18) / 3 (2)", "19 (18) / 5 (2)"), "U_TPhi": ("21 (17) / 3 (2)", "18 (18) / 6 (2)"),
        "O_B": ("20 (17) / 4 (2)", "18 (16) / 6 (2)"), "O_T": ("19 (17) / 5 (2)", "15 (15) / 9 (3)"), "O_TPhi": ("19 (16) / 5 (2)", "15 (15) / 9 (3)"),
        "SM_B": ("22 (22) / 2 (1)", "22 (22) / 2 (2)"), "SM_T": ("20 (20) / 4 (2)", "22 (22) / 2 (2)"), "SM_TPhi": ("20 (20) / 4 (2)", "22 (22) / 2 (2)")},
 "rpart": {"U_B": ("22 (20) / 2 (2)", "22 (22) / 2 (1)"), "U_T": ("22 (20) / 2 (2)", "22 (22) / 2 (1)"), "U_TPhi": ("20 (18) / 4 (1)", "23 (22) / 1 (1)"),
           "O_B": ("20 (20) / 4 (1)", "22 (22) / 2 (2)"), "O_T": ("20 (20) / 4 (1)", "22 (22) / 2 (2)"), "O_TPhi": ("21 (20) / 3 (1)", "22 (22) / 2 (2)"),
           "SM_B": ("22 (18) / 2 (1)", "22 (22) / 2 (1)"), "SM_T": ("17 (17) / 7 (4)", "22 (22) / 2 (2)"), "SM_TPhi": ("19 (18) / 5 (3)", "22 (22) / 2 (2)")}}

files = sorted(glob.glob(os.path.join(RES, "exp_ds*.pkl")), key=lambda f: int(re.search(r"ds(\d+)", f).group(1)))
exp = ComparisonResults.merge_all([ComparisonResults.load(f) for f in files])
replaced = []
if OVERRIDE:
    for f in sorted(glob.glob(os.path.join(OVERRIDE, "exp_ds*_*.pkl"))):
        o = ComparisonResults.load(f)
        for t in o.task_names():
            if t in exp.tasks:
                for wf, wres in o.tasks[t].items():
                    exp.tasks[t][wf] = wres; replaced.append((t, wf))
    print(f"override: {len(replaced)} workflow results replaced from {OVERRIDE} "
          f"({len(set(t for t, _ in replaced))} data sets: {sorted(set(w for _, w in replaced))})\n")
print(f"{len(exp.task_names())} data sets, {len(exp.workflow_names())} workflows, "
      f"{len(exp.iterations_scores(exp.task_names()[0], 'mc.lm'))} repetitions\n")

def fmt(t): return f"{t[0]} ({t[1]}) / {t[2]} ({t[3]})"
def ours(base, wf):
    r = WL[base].loc[wf]; return (int(r.Win), int(r.sigWin), int(r.Loss), int(r.SigLoss))
WL = {}
for b in [f"mc.{l}" for l in L] + [f"mc.{l}_{s}" for l in L for s in ("UNDERB", "OVERB", "SMOTEB")] + ["mc.arima", "mc.BDES"]:
    WL[b] = WLdef(exp, b, "F1", 0.05)

lines = []; diffs = []
def emit(s=""): lines.append(s); print(s)
def row(label, pairs):
    """pairs: list of (ours, article) tuples per learner column"""
    emit(f"{label:9s}" + "".join(f"{fmt(o):>18s} | {fmt(a):<18s}" for o, a in pairs))
    for o, a in pairs: diffs.append((label, o, a))

emit("Formato de cada célula:  nosso | artigo   =  Vitórias (significativas) / Derrotas (significativas), sobre os 24 data sets\n")
emit("=== Tabela 3: estratégia base vs modelo sem resampling")
emit(f"{'':9s}" + "".join(f"{l.upper():>18s} | {'artigo':<18s}" for l in L))
for s in ("U_B", "O_B", "SM_B"):
    row(s, [(ours(f"mc.{l}", f"mc.{l}_{STRAT[s]}"), cell(T3[s][i])) for i, l in enumerate(L)])
emit("\n=== Tabela 4: estratégia com viés temporal / relevância vs a sua versão base (B)")
emit(f"{'':9s}" + "".join(f"{l.upper():>18s} | {'artigo':<18s}" for l in L))
for s, base in (("U_T", "UNDERB"), ("U_TPhi", "UNDERB"), ("O_T", "OVERB"), ("O_TPhi", "OVERB"), ("SM_T", "SMOTEB"), ("SM_TPhi", "SMOTEB")):
    row(s, [(ours(f"mc.{l}_{base}", f"mc.{l}_{STRAT[s]}"), cell(T4[s][i])) for i, l in enumerate(L)])
emit("\n=== Tabela 5: cada modelo + estratégia vs ARIMA e vs BDES")
emit(f"{'':15s}{'vs ARIMA (nosso)':>18s} | {'artigo':<18s}{'vs BDES (nosso)':>18s} | {'artigo':<18s}")
for l in L:
    for s in STRAT:
        wf = f"mc.{l}_{STRAT[s]}"
        oa, ob = ours("mc.arima", wf), ours("mc.BDES", wf); aa, ab = cell(T5[l][s][0]), cell(T5[l][s][1])
        emit(f"{l.upper()+' '+s:15s}{fmt(oa):>18s} | {fmt(aa):<18s}{fmt(ob):>18s} | {fmt(ab):<18s}")
        diffs.append((f"{l} {s} vs ARIMA", oa, aa)); diffs.append((f"{l} {s} vs BDES", ob, ab))

# ---- agreement summary
emit("\n=== Resumo da concordância (nosso - artigo), por célula")
D = pd.DataFrame([(lab, o[0] - a[0], o[1] - a[1], o[2] - a[2], o[3] - a[3], np.sign(o[0] - o[2]) == np.sign(a[0] - a[2]) or (o[0] == o[2]) or (a[0] == a[2]))
                  for lab, o, a in diffs], columns=["cell", "dWin", "dSigWin", "dLoss", "dSigLoss", "same_direction"])
for tbl, sel in (("Tabela 3", D.cell.isin(["U_B", "O_B", "SM_B"])), ("Tabela 4", D.cell.isin(list(T4))), ("Tabela 5", D.cell.str.contains(" vs "))):
    d = D[sel]
    emit(f"{tbl}: {len(d)} células | |Δvitórias| média {d.dWin.abs().mean():.2f}, máx {d.dWin.abs().max()} | "
         f"|Δvit. signif.| média {d.dSigWin.abs().mean():.2f}, máx {d.dSigWin.abs().max()} | "
         f"células com |Δvitórias| <= 2: {(d.dWin.abs() <= 2).mean()*100:.0f}% | mesma direção (quem ganha): {d.same_direction.mean()*100:.0f}%")
big = D[D.dWin.abs() > 3]
if len(big):
    emit("Células com |Δvitórias| > 3:")
    for _, r in big.iterrows(): emit(f"  {r.cell}: Δwin={r.dWin:+d}, Δloss={r.dLoss:+d}")

# ---- F1 per data set vs the independent R report (J. P. Miranda)
jf = os.path.join(PY, "R_replication", "results", "joao_report_F1.json")
f1 = pd.concat([get_results(exp, task=t).assign(task=t) for t in exp.task_names()], ignore_index=True)
f1["ds"] = f1["task"].str.extract(r"(\d+)")[0].astype(int)
wide = f1.pivot(index="ds", columns="model", values="F1").sort_index()
wide.to_csv(os.path.join(OUT, "F1_by_dataset.csv"), float_format="%.4f")
if os.path.exists(jf):
    J = json.load(open(jf)); rows = []
    for k, v in J.items():
        ds, wf = k.split("|"); ds = int(ds)
        if wf in wide.columns and ds in wide.index:
            rows.append((ds, wf, wf.split("_")[0].replace("mc.", ""), wide.loc[ds, wf], v))
    Dj = pd.DataFrame(rows, columns=["ds", "workflow", "model", "python_cluster", "relatorio"])
    Dj["desvio"] = Dj.python_cluster - Dj.relatorio
    Dj.to_csv(os.path.join(OUT, "desvios_python_cluster_vs_relatorio.csv"), index=False, float_format="%.4f")
    emit(f"\n=== F1 por data set: Python (cluster, 24 data sets) vs relatório R independente ({len(Dj)} células)")
    emit(f"|desvio| médio {Dj.desvio.abs().mean():.4f}, mediana {Dj.desvio.abs().median():.4f}, "
         f"dentro de 0.02: {(Dj.desvio.abs() <= 0.02).mean()*100:.0f}%, dentro de 0.05: {(Dj.desvio.abs() <= 0.05).mean()*100:.0f}%, "
         f"desvio médio (viés) {Dj.desvio.mean():+.4f}")
    g = Dj.groupby("model").desvio.agg(n="size", abs_mean=lambda x: x.abs().mean(), bias="mean", abs_max=lambda x: x.abs().max())
    emit(g.round(4).to_string())
    emit("\nPor data set (|desvio| médio sobre os 50 workflows):")
    gd = Dj.groupby("ds").desvio.apply(lambda x: x.abs().mean()).round(3)
    emit("  " + "  ".join(f"DS{d}:{v:.3f}" for d, v in gd.items()))
    worst = Dj.reindex(Dj.desvio.abs().sort_values(ascending=False).index).head(12)
    emit("\nMaiores desvios:"); emit(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
if replaced:
    lines.insert(0, f"(mc.mars* workflows re-run locally with the earth port, {OVERRIDE}; the other workflows from the cluster run)\n")
open(os.path.join(OUT, "comparison_tables_3_4_5.txt"), "w").write("\n".join(lines) + "\n")
print(f"\nsaved: {OUT}/comparison_tables_3_4_5.txt, F1_by_dataset.csv, desvios_python_cluster_vs_relatorio.csv")
