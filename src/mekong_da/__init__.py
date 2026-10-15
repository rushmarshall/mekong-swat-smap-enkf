"""Mekong SWAT + SMAP data assimilation.

A reproducible implementation of the two assimilation experiments behind Marshall et al.,
"Relative roles of spatial resolution and temporal coverage in basin-average SMAP soil
moisture assimilation over the Mekong River Basin": a Python Ensemble Kalman Filter for
basin soil moisture, and a Python-driven Fortran SWAT2012 coupling for streamflow.

Start at :func:`mekong_da.pipeline.reproduce_soil_moisture`, or run ``mekong-da reproduce``.

Author: Sebastian R.O. Marshall.
"""

from .config import (  # noqa: F401
    DEFAULT_CONFIG, EnKFConfig, MEKONG_2024, OPTIMAL_CONFIG, RESOLUTIONS, STATIONS,
    SwatRunSpec,
)
from .enkf import EnKFResult, run_enkf, run_enkf_frame  # noqa: F401
from .inputda import InputDA  # noqa: F401
from .metrics import metrics, metrics_table  # noqa: F401

__version__ = "1.1.0"
__author__ = "Sebastian R.O. Marshall"

__all__ = [
    "EnKFConfig", "EnKFResult", "InputDA", "SwatRunSpec",
    "DEFAULT_CONFIG", "OPTIMAL_CONFIG", "MEKONG_2024", "RESOLUTIONS", "STATIONS",
    "run_enkf", "run_enkf_frame", "metrics", "metrics_table",
    "__version__",
]
