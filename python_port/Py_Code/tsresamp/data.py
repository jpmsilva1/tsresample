"""
Data loading and embedding.

R equivalents (Exps.R):

    load("Data/data_NM_PB_LT_DSAA2016.Rdata")   -> load_data()
    create.data(ts, embed)                       -> create_data(ts, embed)

The .Rdata file holds a list `data` with 24 xts objects. Here each series is
returned as a pandas Series with a DatetimeIndex (the xts index), so that the
row "names" used by the R code (timestamps) are available for time ordering.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_RDATA = PROJECT_ROOT / "Data" / "data_NM_PB_LT_DSAA2016.Rdata"
DEFAULT_PICKLE = PROJECT_ROOT / "Data" / "data_NM_PB_LT_DSAA2016.pkl"

DATASET_NAMES = {
    1: "Bike Sharing - Temperature (Daily)",
    2: "Bike Sharing - Humidity (Daily)",
    3: "Bike Sharing - Windspeed (Daily)",
    4: "Bike Sharing - Count of Bike Rentals (Daily)",
    5: "Bike Sharing - Temperature (Hourly)",
    6: "Bike Sharing - Humidity (Hourly)",
    7: "Bike Sharing - Windspeed (Hourly)",
    8: "Bike Sharing - Count of Bike Rentals (Hourly)",
    9: "Icelandic River - Flow of Vatnsdalsa River (Daily)",
    10: "Porto Weather - Minimum Temperature (Daily)",
    11: "Porto Weather - Maximum Temperature (Daily)",
    12: "Porto Weather - Maximum Steady Wind (Daily)",
    13: "Porto Weather - Maximum Wind Gust (Daily)",
    14: "Istanbul Stock Exchange - SP (Daily)",
    15: "Istanbul Stock Exchange - DAX (Daily)",
    16: "Istanbul Stock Exchange - FTSE (Daily)",
    17: "Istanbul Stock Exchange - NIKKEI (Daily)",
    18: "Istanbul Stock Exchange - BOVESPA (Daily)",
    19: "Istanbul Stock Exchange - EU (Daily)",
    20: "Istanbul Stock Exchange - Emerging Markets (Daily)",
    21: "Australian Electricity Load - Total Demand (Half-Hourly)",
    22: "Australian Electricity Load - Recommended Retail Price (Half-Hourly)",
    23: "Water Consumption of Oporto - Pedroucos (Half-Hourly)",
    24: "Water Consumption of Oporto - Rotunda AEP (Half-Hourly)",
}

# Train/test sizes used in the article (see R_Code/README.md):
# 50%/25% by default, 10%/5% for data sets 21 and 22, 20%/10% for 23 and 24.
def default_sizes(dataset: int) -> tuple[float, float]:
    if dataset in (21, 22):
        return 0.10, 0.05
    if dataset in (23, 24):
        return 0.20, 0.10
    return 0.50, 0.25


# --------------------------------------------------------------------------
# Low level .Rdata parsing (pure python, via the `rdata` package)
# --------------------------------------------------------------------------
def _tag_name(tag):
    if tag is None:
        return None
    from rdata.parser import RObjectType
    obj = tag
    if obj.info.type == RObjectType.REF:
        obj = obj.referenced_object
    v = obj.value
    if isinstance(v, bytes):
        return v.decode()
    if hasattr(v, "value"):
        vv = v.value
        return vv.decode() if isinstance(vv, bytes) else str(vv)
    return str(v)


def _walk_attrs(node):
    from rdata.parser import RObjectType
    out = {}
    while node is not None and node.info.type == RObjectType.LIST:
        car, cdr = node.value
        out[_tag_name(node.tag)] = car
        node = cdr
    return out


def _str_vec(obj):
    from rdata.parser import RObjectType
    if obj is None:
        return None
    if obj.info.type == RObjectType.STR:
        return [c.value.decode() if isinstance(c.value, bytes) else c.value for c in obj.value]
    return list(np.atleast_1d(obj.value))


def load_rdata(path: os.PathLike | str = DEFAULT_RDATA) -> List[pd.Series]:
    """Parse the original .Rdata file and return the 24 series (1-based order
    preserved as a python list, i.e. series i of the article is result[i-1])."""
    import rdata
    from rdata.parser import RObjectType

    parsed = rdata.parser.parse_file(str(path))
    top = parsed.object
    # top level is a pairlist  data -> VEC(24)
    assert top.info.type == RObjectType.LIST
    assert _tag_name(top.tag) == "data"
    lst = top.value[0]
    series = []
    for k, el in enumerate(lst.value):
        attrs = _walk_attrs(el.attributes)
        values = np.asarray(el.value, dtype=float).reshape(-1)
        idx_obj = attrs.get("index")
        if idx_obj is None:
            raise ValueError(f"element {k + 1} has no xts index")
        secs = np.asarray(idx_obj.value, dtype=float)
        idx_attrs = _walk_attrs(idx_obj.attributes) if idx_obj.attributes is not None else {}
        tclass = _str_vec(attrs.get("tclass")) or _str_vec(idx_attrs.get("tclass")) or ["POSIXct"]
        tz = (_str_vec(attrs.get("tzone")) or _str_vec(idx_attrs.get("tzone")) or ["UTC"])[0]
        if tz == "":
            tz = "UTC"
        index = pd.to_datetime(secs, unit="s", utc=True)
        # present timestamps as they would be printed by R (in the series tzone)
        try:
            index = index.tz_convert(tz).tz_localize(None)
        except Exception:
            index = index.tz_localize(None)
        if "Date" in tclass:
            index = index.normalize()
        s = pd.Series(values, index=pd.DatetimeIndex(index, name="time"), name=f"ds{k + 1}")
        s.attrs["tclass"] = tclass
        s.attrs["tzone"] = tz
        s.attrs["dataset"] = k + 1
        s.attrs["description"] = DATASET_NAMES.get(k + 1, "")
        series.append(s)
    if len(series) != 24:
        raise ValueError(f"expected 24 series, found {len(series)}")
    return series


def convert_rdata(path=DEFAULT_RDATA, out=DEFAULT_PICKLE) -> Path:
    """Convert the .Rdata file into a pickle with the list of 24 pandas Series."""
    series = load_rdata(path)
    out = Path(out)
    with open(out, "wb") as fh:
        pickle.dump(series, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_data(path=None) -> List[pd.Series]:
    """Equivalent of  load("Data/data_NM_PB_LT_DSAA2016.Rdata")  -> `data`.

    Loads the converted pickle when it exists, otherwise parses the .Rdata.
    """
    if path is not None:
        p = Path(path)
        if p.suffix.lower() == ".pkl":
            with open(p, "rb") as fh:
                return pickle.load(fh)
        return load_rdata(p)
    if DEFAULT_PICKLE.exists():
        with open(DEFAULT_PICKLE, "rb") as fh:
            return pickle.load(fh)
    return load_rdata(DEFAULT_RDATA)


def get_dataset(i: int, data=None) -> pd.Series:
    """1-based access, like data[[i]] in R."""
    if data is None:
        data = load_data()
    if not 1 <= i <= len(data):
        raise IndexError(f"dataset index must be in 1..{len(data)}")
    return data[i - 1]


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def create_data(ts: pd.Series, embed: int) -> pd.DataFrame:
    """Port of create.data(ts, embed).

    R:
        t <- index(ts)[-(1:(embed-1))]
        e <- embed(ts, embed)[, embed:1]
        colnames(e) <- paste('V', 1:embed, sep='')
        as.data.frame(xts(e, t))

    Row t holds (y[t-embed+1], ..., y[t]) as V1..V<embed>; the target is the
    LAST column (V<embed>), which is mandatory for the SMOTE variants.
    """
    y = np.asarray(ts.values, dtype=float).reshape(-1)
    n = len(y)
    if embed < 1 or embed > n:
        raise ValueError("invalid embedding dimension")
    m = n - embed + 1
    # sliding windows: row i = y[i : i+embed]
    e = np.lib.stride_tricks.sliding_window_view(y, embed)[:m].copy()
    cols = [f"V{j}" for j in range(1, embed + 1)]
    idx = ts.index[embed - 1:]
    return pd.DataFrame(e, columns=cols, index=idx)


def target_name(df: pd.DataFrame) -> str:
    """The target is always the last column (V10 with embed=10)."""
    return df.columns[-1]


if __name__ == "__main__":  # pragma: no cover
    out = convert_rdata()
    print("written", out)


# --------------------------------------------------------------------------
# Missing values (Exps.R has these two options commented out:
#     #ds <- knnImputation(ds)
#     #ds <- ds[complete.cases(ds),]
# Data sets 12, 13, 23 and 24 contain NA values, so one of them is needed.)
# --------------------------------------------------------------------------
def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """R: ds[complete.cases(ds), ]"""
    return df.dropna(axis=0, how="any")


def knn_imputation(df: pd.DataFrame, k: int = 10, scale: bool = True, meth: str = "weighAvg") -> pd.DataFrame:
    """Port of DMwR::knnImputation for numeric data frames.

    For every row with NAs the k nearest complete rows (Euclidean distance on
    the standardised non-missing columns) are found and each missing value is
    replaced by the weighted average (weights exp(-dist)) or the median of the
    neighbours' values.
    """
    data = df.copy()
    X = data.to_numpy(dtype=float)
    n, ncol = X.shape
    dm = X.copy()
    if scale:
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(dm, axis=0)
            sd = np.nanstd(dm, axis=0, ddof=1)
            sd[sd == 0] = 1.0
            dm = (dm - mu) / sd
    nas = np.where(np.isnan(dm).any(axis=1))[0]
    if len(nas) == 0:
        return data
    complete = np.setdiff1d(np.arange(n), nas)
    if len(complete) < k:
        raise ValueError("Not sufficient complete cases for computing neighbors.")
    xcomplete = dm[complete]
    out = X.copy()
    for i in nas:
        tgt_as = np.where(np.isnan(dm[i]))[0]
        keep = np.setdiff1d(np.arange(ncol), tgt_as)
        diff = xcomplete[:, keep] - dm[i, keep]
        dist = np.sqrt((diff ** 2).sum(axis=1))
        ks = np.argsort(dist, kind="stable")[:k]
        for j in tgt_as:
            vals = X[complete[ks], j]
            if meth == "median":
                out[i, j] = np.median(vals)
            else:
                w = np.exp(-dist[ks])
                out[i, j] = np.sum(vals * w) / np.sum(w)
    return pd.DataFrame(out, columns=df.columns, index=df.index)


def handle_na(df: pd.DataFrame, how: str = "knn") -> pd.DataFrame:
    """how: 'none' (leave as is), 'complete' (complete.cases) or 'knn' (knnImputation)."""
    if how == "none":
        return df
    if how == "complete":
        return complete_cases(df)
    if how == "knn":
        return knn_imputation(df)
    raise ValueError(f"unknown NA handling: {how}")
