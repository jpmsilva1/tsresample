"""The package installs, imports, and carries a single-sourced version."""

import importlib
import pkgutil
from importlib.metadata import version

import tsresample


def test_version_is_single_sourced_from_init() -> None:
    assert version("tsresample") == tsresample.__version__


def test_every_module_imports() -> None:
    names = [m.name for m in pkgutil.walk_packages(tsresample.__path__, "tsresample.")]
    assert "tsresample.pipeline.cli" in names
    for name in names:
        importlib.import_module(name)
