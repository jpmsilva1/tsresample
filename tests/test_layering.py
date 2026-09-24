"""Gate A: Layer 1 never imports ``pipeline`` or ``pandas`` (SPEC §3)."""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "tsresample"


def _layer1(root: Path) -> list[Path]:
    # Every module under the package, subpackages included, except pipeline/.
    return sorted(
        p for p in root.rglob("*.py") if "pipeline" not in p.relative_to(root).parts
    )


LAYER1 = _layer1(PKG)


def _imports(path: Path) -> set[str]:
    # Relative imports resolve into the tsresample package (Layer 1 is one level).
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(filter(None, ["tsresample", base]))
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def _forbidden(names: set[str]) -> set[str]:
    return {
        n
        for n in names
        if n.split(".")[0] == "pandas" or n.split(".")[:2] == ["tsresample", "pipeline"]
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


@pytest.mark.parametrize(
    "src",
    [
        "import numpy",
        "from sklearn.pipeline import Pipeline",
        "import imblearn.pipeline",
        "from scipy.interpolate import CubicHermiteSpline",
        "from ._relevance import phi",
    ],
)
def test_checker_allows_other_libraries_pipelines(src: str, tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text(src)
    assert _forbidden(_imports(f)) == set()


def test_layer1_subpackages_are_scanned_too(tmp_path: Path) -> None:
    for rel in ("__init__.py", "_core/__init__.py", "pipeline/io.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    assert _layer1(tmp_path) == [
        tmp_path / "__init__.py",
        tmp_path / "_core/__init__.py",
    ]
