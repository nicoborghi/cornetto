import logging as _logging
_logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
del _logging

"""
cornetto — ultrafast, beautiful corner plots powered by KDExpress FFT-KDE.

Quick start
-----------
>>> from cornetto import corner
>>> fig, axes = corner({"mass": samples_1d, "spin": samples_1d},
...                    truths={"mass": 30.0})

Fast iteration
--------------
>>> from cornetto import quick_corner
>>> fig, axes = quick_corner(data)   # histograms-only, any N_params

Inspect setup
-------------
>>> c = Cornetto(data)
>>> fig, axes = c.plot()
>>> c.info()                          # prints the configuration used

Multiple chains
---------------
>>> fig, axes = corner(
...     {"mass": shape_(N_chains, N_post), "spin": ...},
...     chain_labels=["GW150914", "GW190521"],
...     truths={"mass": np.array([30., 50.])},   # one per chain
... )

With summary table
------------------
>>> c = Cornetto(data)
>>> tbl = c.summary()
>>> tbl   # renders as HTML in Jupyter, plain text in terminal
"""

from .core import corner, quick_corner, marginal, trace, trace_marginal, Cornetto
from .styles import (
    PALETTES, CORNETTO_PALETTE, MAX_CHAINS, DEFAULT_PALETTE,
    ColorManager, make_contour_colors,
)
from .stats import (
    SummaryTable, hdi, compute_stats, sigmas_to_levels, DEFAULT_LEVELS,
    stat_median, stat_median_mad, stat_median_hdi, stat_mean, STAT_REGISTRY,
)

__all__ = [
    "corner", "quick_corner", "marginal", "trace", "trace_marginal", "Cornetto",
    "PALETTES", "CORNETTO_PALETTE", "MAX_CHAINS", "DEFAULT_PALETTE",
    "ColorManager", "make_contour_colors",
    "SummaryTable", "hdi", "compute_stats", "sigmas_to_levels", "DEFAULT_LEVELS",
    "stat_median", "stat_median_mad", "stat_median_hdi", "stat_mean", "STAT_REGISTRY",
]
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("cornetto")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
