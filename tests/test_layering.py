"""Gate A: Layer 1 never imports ``pipeline`` or ``pandas`` (SPEC §3)."""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "tsresample"
LAYER1 = sorted(p for p in PKG.glob("*.py"))


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def _forbidden(names: set[str]) -> set[str]:
    return {
        n
        for n in names
        if n.split(".")[0] == "pandas" or "pipeline" in n.lstrip(".").split(".")
    }


def test_layer1_modules_exist() -> None:
    assert {p.name for p in LAYER1} >= {"__init__.py", "embed.py", "resampler.py"}


@pytest.mark.parametrize("module", LAYER1, ids=lambda p: p.name)
def test_layer1_module_imports_neither_pandas_nor_pipeline(module: Path) -> None:
    assert _forbidden(_imports(module)) == set()


@pytest.mark.parametrize(
    "src",
    [
        "import pandas",
        "import pandas as pd",
        "from pandas import DataFrame",
        "from .pipeline import io",
        "from tsresample.pipeline.io import load_csv",
        "from . import pipeline",
    ],
)
def test_checker_flags_forbidden_imports(src: str, tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text(src)
    assert _forbidden(_imports(f))
