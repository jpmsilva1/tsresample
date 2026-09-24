"""Layer 2 pipeline: load_series, imbalance_summary, temporal_split, evaluate.

SPEC §2.4; ADR-0010. Needs the ``io`` extra: ``pip install tsresample[io]``.
"""

from tsresample.pipeline.io import load_series
from tsresample.pipeline.splits import temporal_split

__all__ = ["load_series", "temporal_split"]
