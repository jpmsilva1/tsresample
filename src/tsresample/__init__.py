"""Resampling strategies for imbalanced time series forecasting.

Implements Moniz, Branco & Torgo (2017). The public API (``embed``,
``TimeSeriesResampler``) is re-exported here as the nodes that build it land.
"""

from tsresample.embed import embed
from tsresample.resampler import TimeSeriesResampler

__all__ = ["TimeSeriesResampler", "__version__", "embed"]
__version__ = "0.0.1"
