"""CLI seam: subprocess, exit codes and stdout (QUALITY_GATES §2)."""

import contextlib
import csv
import io
import subprocess
import sys
from pathlib import Path

from tsresample.pipeline.cli import main

DATA = Path(__file__).resolve().parents[1] / "Blueprint" / "replication" / "datasets"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    # main(argv) in-process so coverage sees it; one real subprocess test below.
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in args])
    return subprocess.CompletedProcess(args, code, out.getvalue(), err.getvalue())


def test_installed_console_script_runs(tmp_path: Path) -> None:
    exe = Path(sys.executable).with_name("tsresample")
    p = _csv(tmp_path, ["4", "1", "9", "2", "100", "3", "8", "5", "7", "6"])
    res = subprocess.run(
        [exe, p, "--target", "value", "--no-embed"], capture_output=True, text=True
    )
    assert res.returncode == 0 and "value" in res.stdout, res.stderr


def _csv(tmp_path: Path, values: list[str]) -> Path:
    p = tmp_path / "s.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "value"])
        w.writerows([f"2020-01-{i + 1:02d}", v] for i, v in enumerate(values))
    return p


def test_single_file_prints_the_table_1_row(tmp_path: Path) -> None:
    # --no-embed on y = 4, 1, 9, 2, 100, 3, 8, 5, 7, 6: rare = {9, 100} ->
    # N 10, n_normal 8, n_rare 2, IR 0.25, %Rare 20.0.
    p = _csv(tmp_path, ["4", "1", "9", "2", "100", "3", "8", "5", "7", "6"])
    out = _run(p, "--target", "value", "--date-col", "day", "--no-embed")
    assert out.returncode == 0, out.stderr
    header, row = out.stdout.strip().splitlines()
    assert header.split() == [
        "ID",
        "Dataset",
        "N",
        "Granularity",
        "n_normal",
        "n_rare",
        "IR",
        "%Rare",
    ]
    assert row.split() == ["DS1", "value", "10", "8", "2", "0.25", "20.0"]


def test_k_translates_one_to_one_from_imbalance_eval() -> None:
    # imbalance_eval --k 10 keeps 10 lags, so N = 730 - 10 = 720 on DS01.
    p = next(DATA.glob("DS01_*.csv"))
    out = _run(p, "--target", "target", "--date-col", "time_index", "--k", "10")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[1].split()[2] == "720"


def test_batch_manifest_and_output_file(tmp_path: Path) -> None:
    p = _csv(tmp_path, ["4", "1", "9", "2", "100", "3", "8", "5", "7", "6"])
    m = tmp_path / "m.csv"
    m.write_text(
        f"id,name,granularity,path,target,k\nA,first,Daily,{p},value,\nB,second,,{p},value,3\n"
    )
    out_csv = tmp_path / "out.csv"
    res = _run("--config", m, "--no-embed", "--output", out_csv)
    assert res.returncode == 0, res.stderr
    rows = list(csv.DictReader(out_csv.open()))
    assert [r["ID"] for r in rows] == ["A", "B"] and rows[0]["n_rare"] == "2"


def test_manifest_missing_columns_are_named(tmp_path: Path) -> None:
    m = tmp_path / "m.csv"
    m.write_text("id,path\nA,x.csv\n")
    res = _run("--config", m)
    assert res.returncode == 2
    assert "missing required column(s): name, target" in res.stderr


def test_usage_errors_exit_2_with_a_message(tmp_path: Path) -> None:
    p = _csv(tmp_path, ["1", "2", "3"])
    cases = [
        ((), "provide a CSV path or --config"),
        ((p,), "--target is required"),
        ((p, "--config", p), "not both"),
        ((p, "--target", "temp"), "available: ['day', 'value']"),
        ((p, "--target", "value", "--xtrm-type", "high"), "--xtrm-type"),
        ((p, "--target", "value", "--coef", "3"), "--coef"),
    ]
    for args, msg in cases:
        res = _run(*args)
        assert res.returncode == 2 and msg in res.stderr, (args, res.stderr)


def test_impute_modes_are_exposed(tmp_path: Path) -> None:
    p = _csv(tmp_path, ["4", "", "9", "2", "100", "3", "8", "5", "7", "6"])
    assert (
        _run(p, "--target", "value", "--no-embed", "--impute", "none").returncode == 2
    )
    res = _run(p, "--target", "value", "--no-embed", "--impute", "drop")
    assert res.returncode == 0 and res.stdout.splitlines()[1].split()[2] == "9"
