"""The user-facing docs stay true: examples run, API page matches docstrings."""

import inspect
import os
import re
from pathlib import Path

import pytest

import tsresample
from tsresample import metrics, pipeline

REPO = Path(__file__).resolve().parents[1]
API_PAGE = REPO / "docs" / "api.md"


def test_readme_python_examples_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = re.findall(
        r"```python\n(.*?)```", (REPO / "README.md").read_text(encoding="utf-8"), re.S
    )
    assert blocks, "README has no python examples"
    monkeypatch.chdir(tmp_path)  # examples write their own CSV
    for code in blocks:
        exec(compile(code, "README.md", "exec"), {})


def _sig(obj: object) -> str:
    # Without annotations: their repr differs between Python versions.
    sig = inspect.signature(obj)  # type: ignore[arg-type]
    params = [
        p.replace(annotation=p.empty)
        for p in sig.parameters.values()
        if p.name != "self"
    ]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def _api_markdown() -> str:
    out = [
        "# API reference",
        "",
        "Generated from the docstrings by "
        "`tests/test_docs.py` (`TSRESAMPLE_UPDATE_DOCS=1 pytest tests/test_docs.py`).",
        "",
    ]
    for mod, names in (
        (tsresample, ["embed", "TimeSeriesResampler"]),
        (metrics, ["precision_phi", "recall_phi", "f1_phi", "sera", "control_points"]),
        (pipeline, ["load_series", "imbalance_summary", "temporal_split", "evaluate"]),
    ):
        out += [f"## `{mod.__name__}`", ""]
        for name in names:
            obj = getattr(mod, name)
            target = obj.fit_resample if inspect.isclass(obj) else obj
            sig = _sig(obj) if inspect.isclass(obj) else _sig(target)
            out += [
                f"### `{name}{sig}`",
                "",
                "```text",
                inspect.getdoc(obj) or "",
                "```",
                "",
            ]
            if inspect.isclass(obj):
                out += [
                    f"#### `fit_resample{_sig(target)}`",
                    "",
                    "```text",
                    inspect.getdoc(target) or "",
                    "```",
                    "",
                ]
    return "\n".join(out)


def test_api_reference_matches_docstrings() -> None:
    text = _api_markdown()
    if os.environ.get("TSRESAMPLE_UPDATE_DOCS"):
        API_PAGE.write_text(text, encoding="utf-8")
    assert API_PAGE.read_text(encoding="utf-8") == text


def test_citation_file_names_the_paper_and_licence() -> None:
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    for needle in (
        "cff-version:",
        "title: tsresample",
        "license: MIT",
        "Resampling strategies for imbalanced time series forecasting",
        "10.1007/s41060-017-0044-3",
    ):
        assert needle in cff, needle


def test_deviations_page_lists_every_r_over_paper_choice() -> None:
    page = (REPO / "docs" / "deviations.md").read_text(encoding="utf-8")
    for adr in ("0004", "0006", "0007", "0011", "0012", "0013", "0015", "0016"):
        assert f"adr/{adr}" in page, adr
    assert "r_quirks=False" in page
