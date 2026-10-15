"""Configuration objects and canonical constants for the Mekong SWAT-DA pipeline.

Everything that the published run treated as a fixed number lives here, so that a
reader can find it in one place and a re-run can change it in one place. Nothing in
this module reads the filesystem beyond resolving repository-relative paths.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# --------------------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------------------

#: Repository root, resolved from this file rather than hard-coded to any machine.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("MEKONG_DA_DATA", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("MEKONG_DA_OUTPUT", REPO_ROOT / "outputs"))
FORTRAN_DIR = REPO_ROOT / "fortran"

#: The three SMAP soil-moisture products assimilated in the study, coarse to fine.
RESOLUTIONS: Tuple[str, ...] = ("9km", "1km", "400m")


def output_path(*parts: str) -> Path:
    """Return a path under ``outputs/`` and create the parent directory."""
    p = OUTPUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# Soil-moisture EnKF (the Python filter)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnKFConfig:
    """Scalar Ensemble Kalman Filter settings.

    Defaults define the production scalar configuration. Sensitivity experiments
    vary individual settings, including a 50-member comparison with the 200-member
    production ensemble. The name OPTIMAL_CONFIG is retained as an API identifier;
    it does not claim a universal optimum or apply to the coupled Fortran model.

    Observation and process-error standard deviations use m3/m3. The observation
    variance enters the gain and member perturbations. Inflation multiplies forecast
    variance before the gain calculation. The autoregressive forecast combines the
    previous analysis with the fixed baseline, then adds process noise. Persistence
    zero anchors that forecast to the baseline but does not remove stochastic noise.

    The scalar bounds apply after forecast and analysis steps. The fixed NumPy
    seed reproduces trajectories when random-number order and inputs are preserved.
    Current innovation diagnostics use the explicitly recorded forecast innovations.
    """

    ensemble_size: int = 200
    obs_error_std: float = 0.04
    model_error_std: float = 0.04
    inflation_factor: float = 1.02
    persistence_phi: float = 0.50
    sm_bounds: Tuple[float, float] = (0.01, 0.60)
    seed: int = 42

    @classmethod
    def from_json(cls, path: os.PathLike | str) -> "EnKFConfig":
        """Load a configuration from a JSON file such as ``optimal_config.json``."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "sm_bounds" in kwargs:
            kwargs["sm_bounds"] = tuple(kwargs["sm_bounds"])
        return cls(**kwargs)

    def to_dict(self) -> Dict:
        return asdict(self)


#: The tuned configuration shipped with the repository.
OPTIMAL_CONFIG = EnKFConfig()

#: The pre-tuning configuration, kept so the tuning experiment can be reproduced.
DEFAULT_CONFIG = EnKFConfig(inflation_factor=1.05, persistence_phi=0.70)

#: Evaluation windows for the soil-moisture table (T1).
SM_PERIODS: Dict[str, Tuple[str, str]] = {
    "Full (2015-2024)": ("2015-01-01", "2024-12-31"),
    "Training (2015-2019)": ("2015-01-01", "2019-12-31"),
    "Validation (2020-2024)": ("2020-01-01", "2024-12-31"),
}


# --------------------------------------------------------------------------------------
# Streamflow SWAT-DA (the Fortran model)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Station:
    """A Mekong River Commission discharge gauge and the SWAT reach that carries it."""

    key: str
    label: str
    reach: int


#: The five MRC gauges used for streamflow validation, upstream to downstream.
STATIONS: Tuple[Station, ...] = (
    Station("Chiang_Saen", "Chiang Saen", 70),
    Station("Nong_Khai", "Nong Khai", 43),
    Station("Mukdahan", "Mukdahan", 524),
    Station("Pakse", "Pakse", 18),
    Station("Kratie", "Kratie", 9),
)

STATION_BY_KEY: Dict[str, Station] = {s.key: s for s in STATIONS}


@dataclass(frozen=True)
class SwatRunSpec:
    """Geometry and calendar of a SWAT2012 application.

    The defaults describe the calibrated Mekong model used in the paper. For another basin,
    build your own instance from your project's ``file.cio`` and HRU count; the DA
    observation files must be written with exactly ``n_days`` rows and ``n_hru`` columns or
    the Fortran read fails. See docs/08_your_own_basin.md.
    """

    n_hru: int = 5605
    n_subbasins: int = 540
    start_year: int = 1998
    n_years: int = 27
    nyskip: int = 3  #: warm-up years excluded from output.rch
    n_days: int = 9862  #: 1998-01-01 through 2024-12-31 inclusive, leap years counted

    @property
    def output_start_year(self) -> int:
        """First year present in ``output.rch``."""
        return self.start_year + self.nyskip


MEKONG_2024 = SwatRunSpec()

#: Evaluation windows for the streamflow table (T2).
STREAMFLOW_PERIODS: Dict[str, Tuple[str, str]] = {
    "Full (2005-2020)": ("2005-01-01", "2020-12-31"),
    "Pre-SMAP (2005-2014)": ("2005-01-01", "2014-12-31"),
    "SMAP (2015-2020)": ("2015-01-01", "2020-12-31"),
    "SMAP-All (2015-2024)": ("2015-04-01", "2024-12-31"),
    "Calibration (2005-2009)": ("2005-01-01", "2009-12-31"),
    "Validation (2010-2020)": ("2010-01-01", "2020-12-31"),
}

#: Column in the merged streamflow CSVs for each model arm.
STREAMFLOW_MODELS: Dict[str, str] = {
    "Control": "swat_baseline",
    "EnKF-9km": "swat_da_9km",
    "EnKF-1km": "swat_da_1km",
    "EnKF-400m": "swat_da_400m",
}
