"""Read Fig. 7 of the article (F1 per data set, 24 data sets x 52 workflows) from the PDF and
compare with the full cluster run (results_cluster/F1_by_dataset.csv).
    .venv/bin/python tools/compare_fig7_all.py
Reading accuracy is about +-0.03 (pixel position on a 0..1 axis); points hidden behind another
point of the same panel are not read (NaN).
"""
import sys, pathlib, numpy as np, pandas as pd, pymupdf
from scipy import ndimage
HERE = pathlib.Path(__file__).resolve().parent; PY = HERE.parent
PDF = PY.parent / "s41060-017-0044-3.pdf"
doc = pymupdf.open(str(PDF)); page = doc[12]; Z = 8
pix = page.get_pixmap(matrix=pymupdf.Matrix(Z, Z), clip=pymupdf.Rect(0, 60, page.rect.width, 0.63 * page.rect.height))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
grey = (np.abs(img - 240).max(axis=2) <= 3)
merged = ndimage.binary_closing(grey, structure=np.ones((15, 15)))
lab, n = ndimage.label(merged); objs = ndimage.find_objects(lab)
panels = []
for i, sl in enumerate(objs):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if h > 400 and w > 400 and (lab[sl] == i + 1).mean() > 0.8:
        panels.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
panels.sort()
def cluster(vals, gap=150):
    vals = sorted(set(vals)); groups = [[vals[0]]]
    for v in vals[1:]:
        (groups[-1].append(v) if v - groups[-1][-1] < gap else groups.append([v]))
    return groups
rowg = cluster([p[0] for p in panels]); colg = cluster([p[1] for p in panels])
assert len(rowg) == 4 and len(colg) == 5, (len(rowg), len(colg))
grid = {}
for p in panels:
    r = next(i for i, g in enumerate(rowg) if p[0] in g); c = next(i for i, g in enumerate(colg) if p[1] in g); grid[(r, c)] = p
learners = ["lm", "svm", "mars", "rf", "rpart"]; strat = ["None", "UNDER", "OVER", "SmoteR"]
colors = {"Original": (246, 118, 108), "B": (183, 159, 0), "T": (26, 180, 63), "TPhi": (37, 189, 192), "ARIMA": (97, 156, 255), "BDES": (245, 100, 227)}
def lines(mask_1d, min_len=1):
    idx = np.where(mask_1d)[0]
    if len(idx) == 0: return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(g[0], g[-1], len(g)) for g in groups if len(g) >= min_len]
# Geometry: 9 horizontal gridlines (F1 = 1 at the top one, 0 at the bottom one); vertical gridlines
# at data sets 1, 5, 10, 15, 20 and 24, so the first and the last vertical line are DS1 and DS24.
def geometry(p):
    y0, x0, y1, x1 = p; sub = img[y0:y1, x0:x1]; white = (sub.min(axis=2) >= 250)
    hl = lines(white.mean(axis=1) > 0.7); vl = lines(white.mean(axis=0) > 0.7)
    hy = np.array([(l[0] + l[1]) / 2 for l in hl]); vx = np.array([(l[0] + l[1]) / 2 for l in vl])
    if len(hy) < 5 or len(vx) < 6: return None
    hs = np.median(np.diff(hy))
    return hy.min(), hy.min() + 8 * hs, vx.min(), vx.max(), (y1 - y0), (x1 - x0)
def _fit_at(xs, ys, xc):
    if len(xs) < 4 or xs.max() - xs.min() < 4: return None
    b, a = np.polyfit(xs, ys, 1); return a + b * xc
def read_panel(p, geom, datasets):
    """The series are drawn as lines whose vertices are the data sets: the value at data set d is the
    line's y at x(d). Pixels of the colour within +-1 px of x(d) give it directly; when the line is
    dotted/dashed and has a gap there, the segments on both sides are fitted and extrapolated to x(d)
    (so a vertex is not flattened)."""
    y0, x0, y1, x1 = p; sub = img[y0:y1, x0:x1]
    y_top, y_bot, x_left, x_right, gh, gw = geom
    sy = (y1 - y0) / gh; sx = (x1 - x0) / gw
    y_top, y_bot, x_left, x_right = y_top * sy, y_bot * sy, x_left * sx, x_right * sx
    cols = np.array(list(colors.values())); flat = sub.reshape(-1, 3)
    dist = np.abs(flat[:, None, :] - cols[None, :, :]).max(axis=2); best = dist.argmin(axis=1)
    ok = (dist.min(axis=1) <= 55) & (flat.max(axis=1) - flat.min(axis=1) >= 60)
    yy_all, xx_all = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]; yy_all = yy_all.ravel(); xx_all = xx_all.ravel()
    out = {d: {} for d in datasets}
    for k, name in enumerate(colors):
        sel = ok & (best == k)
        if sel.sum() < 4: continue
        px, py = xx_all[sel], yy_all[sel]
        for d in datasets:
            xc = x_left + (d - 1) / 23 * (x_right - x_left)
            c = np.abs(px - xc) <= 1.5
            if c.sum() >= 2:
                y = np.median(py[c])
            else:
                l = (px >= xc - 18) & (px < xc - 1.5); r = (px > xc + 1.5) & (px <= xc + 18)
                est = [v for v in (_fit_at(px[l], py[l], xc), _fit_at(px[r], py[r], xc)) if v is not None]
                if not est: continue
                y = float(np.mean(est))
            out[d][name] = float(np.clip(1 - (y - y_top) / (y_bot - y_top), -0.05, 1.05))
    return out
suffix = {"UNDER": ["_UNDERB", "_UNDERT", "_UNDERTPhi"], "OVER": ["_OVERB", "_OVERT", "_OVERTPhi"], "SmoteR": ["_SMOTEB", "_SMOTET", "_SMOTETPhi"]}
art = {}
for r, srow in enumerate(strat):
    for c, L in enumerate(learners):
        p = grid[(r, c)]; g = geometry(p)
        if g is None:
            g = geometry(grid[(r, 1)])
        vals = read_panel(p, g, range(1, 25))
        for d, v in vals.items():
            row = art.setdefault(d, {})
            if "Original" in v: row.setdefault(f"mc.{L}", v["Original"])
            if srow == "None":
                if "ARIMA" in v: row.setdefault("mc.arima", v["ARIMA"])
                if "BDES" in v: row.setdefault("mc.BDES", v["BDES"])
            else:
                for key, suf in zip(("B", "T", "TPhi"), suffix[srow]):
                    if key in v: row[f"mc.{L}{suf}"] = v[key]
A = pd.DataFrame(art).T.sort_index(); A.index.name = "ds"
A.to_csv(PY / "results_cluster" / "fig7_article_readings.csv", float_format="%.3f")
W = pd.read_csv(PY / "results_cluster" / "F1_by_dataset.csv", index_col=0)
cols = [c for c in W.columns if c in A.columns]
D = (W[cols] - A[cols]); absD = D.abs()
print(f"Fig. 7 read: {A.notna().sum().sum()} of {24*52} points ({A.notna().sum().sum()/(24*52)*100:.0f} %); the rest are hidden behind another point or off the panel\n")
print(f"Overall: mean |diff| {absD.stack().mean():.3f}, median {absD.stack().median():.3f}, "
      f"within 0.05: {(absD.stack() <= 0.05).mean()*100:.0f} %, within 0.10: {(absD.stack() <= 0.10).mean()*100:.0f} %, bias (ours - article) {D.stack().mean():+.3f}")
def fam(c):
    return "baseline" if c.count("_") == 0 else c.split("_")[1].replace("UNDER", "U_").replace("OVER", "O_").replace("SMOTE", "SM_")
def model(c): return c.split("_")[0].replace("mc.", "")
S = pd.DataFrame({"model": [model(c) for c in cols], "strategy": [fam(c) for c in cols],
                  "n": absD[cols].notna().sum().values, "abs_mean": absD[cols].mean().values, "bias": D[cols].mean().values,
                  "within_0.05": (absD[cols] <= 0.05).sum().values / absD[cols].notna().sum().values})
print("\nPer model:"); print(S.groupby("model").apply(lambda g: pd.Series({"n": g.n.sum(), "abs_mean": np.average(g.abs_mean, weights=g.n), "bias": np.average(g.bias, weights=g.n), "within_0.05": np.average(g["within_0.05"], weights=g.n)}), include_groups=False).round(3).to_string())
print("\nPer strategy:"); print(S.groupby("strategy").apply(lambda g: pd.Series({"n": g.n.sum(), "abs_mean": np.average(g.abs_mean, weights=g.n), "bias": np.average(g.bias, weights=g.n), "within_0.05": np.average(g["within_0.05"], weights=g.n)}), include_groups=False).round(3).to_string())
pds = absD.mean(axis=1).round(3)
print("\nPer data set (mean |diff| over the workflows read):"); print("  " + "  ".join(f"DS{d}:{v:.3f}" for d, v in pds.items()))
big = D.stack().abs().sort_values(ascending=False).head(15)
print("\nLargest |diff| (ours vs article):")
for (d, c), v in big.items(): print(f"  DS{d:<3d} {c:20s} ours={W.loc[d, c]:.3f} article={A.loc[d, c]:.3f} diff={D.loc[d, c]:+.3f}")
# side by side table for the user: mean over the 24 data sets, ours | article, per model x strategy
print("\nMean F1 over the data sets read, ours | article:")
L = ["lm", "svm", "mars", "rf", "rpart"]; ST = ["", "_UNDERB", "_UNDERT", "_UNDERTPhi", "_OVERB", "_OVERT", "_OVERTPhi", "_SMOTEB", "_SMOTET", "_SMOTETPhi"]
hdr = ["base"] + [s[1:] for s in ST[1:]]
print(f"{'':6s}" + "".join(f"{h:>15s}" for h in hdr))
for l in L:
    cells = []
    for s in ST:
        c = f"mc.{l}{s}"; m = A[c].notna()
        cells.append(f"{W.loc[m, c].mean():.2f}|{A.loc[m, c].mean():.2f}" if c in A else "-")
    print(f"{l:6s}" + "".join(f"{x:>15s}" for x in cells))
for c in ("mc.arima", "mc.BDES"):
    m = A[c].notna(); print(f"{c:10s} ours {W.loc[m, c].mean():.3f} | article {A.loc[m, c].mean():.3f}  (n={m.sum()})")
# long table of the deviations (one row per point read from the figure)
rows = [(d, c, W.loc[d, c], A.loc[d, c], D.loc[d, c]) for d in D.index for c in cols if pd.notna(D.loc[d, c])]
pd.DataFrame(rows, columns=["ds", "workflow", "python_cluster", "artigo_fig7", "desvio"]).to_csv(
    PY / "results_cluster" / "desvios_python_cluster_vs_artigo_fig7.csv", index=False, float_format="%.3f")
