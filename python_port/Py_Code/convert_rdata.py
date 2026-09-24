"""Convert Data/data_NM_PB_LT_DSAA2016.Rdata into a pickle of 24 pandas Series.

Usage:  python Py_Code/convert_rdata.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsresamp.data import convert_rdata, load_data  # noqa: E402

if __name__ == "__main__":
    out = convert_rdata()
    data = load_data(out)
    print(f"written {out}")
    for s in data:
        print(f"[{s.attrs['dataset']:2d}] n={len(s):6d}  {s.index[0]} .. {s.index[-1]}  {s.attrs['description']}")
