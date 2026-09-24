"""Console script ``tsresample``: the paper's Table 1 imbalance summary.

Mirrors ``imbalance_eval``'s command line so existing invocations keep working;
intentional differences are listed in docs/MIGRATION.md.

    tsresample day.csv --target temp --date-col dteday --diff
    tsresample --config manifest.csv --output table1.csv
"""

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from tsresample.pipeline import imbalance_summary, load_series

REQUIRED_MANIFEST_COLUMNS = {"id", "name", "path", "target"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tsresample",
        description="Summarise how imbalanced a time series is (Moniz et al. 2017, "
        "Table 1): N, n_normal, n_rare, IR = n_rare / n_normal, %Rare.",
    )
    p.add_argument(
        "csv_path", nargs="?", help="CSV file to evaluate (single-file mode)"
    )
    p.add_argument("--target", help="target column (required in single-file mode)")
    p.add_argument("--id", default="DS1", help="dataset ID (default: DS1)")
    p.add_argument("--name", default=None, help="dataset name (default: --target)")
    p.add_argument("--granularity", default="", help="granularity label")
    p.add_argument("--date-col", default=None, help="date column to sort by")
    p.add_argument(
        "--config",
        default=None,
        help="batch manifest CSV: id,name,granularity,path,target[,threshold,k]",
    )
    p.add_argument("--threshold", type=float, default=0.9, help="phi threshold (0.9)")
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="embedding lags: rare cases are counted on series[k:] (default: 10)",
    )
    p.add_argument("--no-embed", action="store_true", help="count on the whole series")
    p.add_argument("--diff", action="store_true", help="first-difference the series")
    p.add_argument("--xtrm-type", default="both", help="only 'both' is supported")
    p.add_argument("--coef", type=float, default=1.5, help="only 1.5 is supported")
    p.add_argument(
        "--impute",
        default="knn",
        choices=["knn", "drop", "none"],
        help="missing values: lag-window kNN (default), drop, or fail",
    )
    p.add_argument("--output", default=None, help="also write the table to this CSV")
    return p


def _row(
    id_: str,
    name: str,
    gran: str,
    s: np.ndarray,
    k: int,
    thr: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    y = s if args.no_embed else s[k:]  # imbalance_eval's k lags -> target s[k:]
    st = imbalance_summary(y, rel_threshold=thr)
    return {
        "ID": id_,
        "Dataset": name,
        "N": st["N"],
        "Granularity": gran,
        "n_normal": st["n_normal"],
        "n_rare": st["n_rare"],
        "IR": round(st["IR"], 2),
        "%Rare": round(st["pct_rare"], 2),
    }


def _load(
    path: str, target: str, date_col: str | None, args: argparse.Namespace
) -> np.ndarray:
    impute = None if args.impute == "none" else args.impute
    return load_series(
        path, target=target, date_col=date_col, diff=args.diff, impute=impute
    )


def _batch(args: argparse.Namespace) -> list[dict[str, object]]:
    with open(args.config, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"manifest {args.config} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )
        rows = list(reader)
    out = []
    for r in rows:
        path = r["path"]  # as given, relative to the working directory (as before)
        thr = float(r["threshold"]) if r.get("threshold") else args.threshold
        k = int(r["k"]) if r.get("k") else args.k
        s = _load(path, r["target"], None, args)
        out.append(
            _row(r["id"], r["name"], r.get("granularity") or "", s, k, thr, args)
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.xtrm_type != "both":
            raise ValueError(
                "--xtrm-type: only 'both' is supported (phi is the "
                "extremes method of SPEC §4.1; see docs/MIGRATION.md)"
            )
        if args.coef != 1.5:
            raise ValueError("--coef: only 1.5 is supported (see docs/MIGRATION.md)")
        if args.k < 0:
            raise ValueError(f"--k: expected k >= 0; got {args.k}")
        if args.config and args.csv_path:
            raise ValueError("provide either a CSV path or --config, not both")
        if args.config:
            rows = _batch(args)
        elif args.csv_path:
            if not args.target:
                raise ValueError("--target is required in single-file mode")
            s = _load(args.csv_path, args.target, args.date_col, args)
            rows = [
                _row(
                    args.id,
                    args.name or args.target,
                    args.granularity,
                    s,
                    args.k,
                    args.threshold,
                    args,
                )
            ]
        else:
            raise ValueError("provide a CSV path or --config")
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.output:
        table.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
