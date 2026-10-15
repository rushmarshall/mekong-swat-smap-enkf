"""Reader and writer for ``input.da``, the Fortran SWAT-DA control file.

``input.da`` is a bare list of 22 values, one per line, with no keys and no comments. The
Fortran reader (``fortran/src/readinput.f90``) consumes them strictly in order, and lines
2 through 11 are read under a fixed ``f5.2`` format, so a value written at a different
width is silently misread rather than rejected. This module is the safe way to produce
one.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

__all__ = ["InputDA", "SAMPLING_METHODS", "DA_ALGORITHMS", "SM_SENSORS"]

#: ``n_sampling_method`` codes accepted by the Fortran ensemble generator.
SAMPLING_METHODS = {1: "Sobol sequence", 2: "Halton sequence", 3: "Latin hypercube", 4: "random"}

#: ``da_alg`` codes.
DA_ALGORITHMS = {1: "EnKF (covariance update)", 2: "BLUE"}

#: ``smsat`` codes. The Mekong application assimilates SMAP through the SMOS branch,
#: which is why the observation files carry SMOS-style names and quality flags.
SM_SENSORS = {1: "SMOS branch (used here for SMAP)", 2: "ASCAT branch"}


@dataclass
class InputDA:
    """The 22 values of ``input.da``, in file order.

    The defaults are the published Mekong configuration. Field order in this dataclass is
    the file order; do not reorder it.

    Attributes
    ----------
    da:
        0 selects the deterministic branch; 1 retains ensemble dynamics and permits
        observation updates. The corrected matched control uses ``da=1`` with
        all-missing observation files, as described in docs/04_coupled_model.md.
    pcp_std_par:
        Precipitation perturbation coefficient (fraction of the daily depth).
    fcpercent, fcpercent2, fcpercent3:
        Legacy field-capacity controls retained in the input format. The corrected
        update bounds each member using available-water saturation storage.
    sm_dqx_co, sm_rfi_co:
        Legacy quality-index multipliers retained for input compatibility. The
        corrected operator uses ``0.04 * layer_depth_mm`` as its observation-error
        standard deviation; quality flags still determine observation eligibility.
    sm_std1, sm_std2, sm_std3:
        Soil-moisture forecast error standard deviations by layer, mm/mm.
    cn_std:
        Curve-number perturbation standard deviation.
    n_en_member:
        Ensemble size. Must match the row count of ``sample.txt``.
    n_ensembles:
        Number of ensembles (1 for a single filter pass).
    n_parameters:
        Perturbed parameter count, and therefore the column count of ``sample.txt``.
    n_sampling_method:
        See :data:`SAMPLING_METHODS`.
    sol_st_std:
        Standard deviation used to spread the initial soil-storage ensemble.
    no_sta_var:
        Length of the state vector (2 soil layers in this application).
    no_obs_var:
        Length of the observation vector (1 surface soil-moisture value).
    wrdtf:
        1 writes the per-step diagnostic file ``KalmanGain.dat`` for HRU 1, 0 suppresses
        it. Leave at 1 only for short diagnostic runs; it is written every step.
    da_stopday:
        Simulation day after which assimilation stops, which is how a forecast-skill
        experiment is set up. 99999 assimilates throughout.
    da_alg:
        See :data:`DA_ALGORITHMS`.
    smsat:
        See :data:`SM_SENSORS`.
    """

    da: int = 1
    pcp_std_par: float = 0.50
    fcpercent: float = 0.50
    fcpercent2: float = 0.50
    fcpercent3: float = 0.50
    sm_dqx_co: float = 1.00
    sm_rfi_co: float = 1.00
    sm_std1: float = 0.04
    sm_std2: float = 0.04
    sm_std3: float = 0.04
    cn_std: float = 0.10
    n_en_member: int = 200
    n_ensembles: int = 1
    n_parameters: int = 6
    n_sampling_method: int = 3
    sol_st_std: float = 0.1
    no_sta_var: int = 2
    no_obs_var: int = 1
    wrdtf: int = 0
    da_stopday: int = 99999
    da_alg: int = 1
    smsat: int = 1

    #: Field names that the Fortran reads under the fixed ``f5.2`` format.
    F52_FIELDS = (
        "pcp_std_par", "fcpercent", "fcpercent2", "fcpercent3",
        "sm_dqx_co", "sm_rfi_co", "sm_std1", "sm_std2", "sm_std3", "cn_std",
    )

    def __post_init__(self) -> None:
        if self.da not in (0, 1):
            raise ValueError(f"da must be 0 (open loop) or 1 (assimilate), got {self.da}")
        if self.n_sampling_method not in SAMPLING_METHODS:
            raise ValueError(f"n_sampling_method must be one of {sorted(SAMPLING_METHODS)}")
        if self.da_alg not in DA_ALGORITHMS:
            raise ValueError(f"da_alg must be one of {sorted(DA_ALGORITHMS)}")
        if self.smsat not in SM_SENSORS:
            raise ValueError(f"smsat must be one of {sorted(SM_SENSORS)}")
        if self.n_en_member < 1:
            raise ValueError("n_en_member must be at least 1")
        for name in self.F52_FIELDS:
            value = getattr(self, name)
            if not -9.99 <= value <= 99.99:
                raise ValueError(
                    f"{name}={value} does not fit the f5.2 field the Fortran reader uses; "
                    "a wider value is silently misread, not rejected"
                )

    def to_text(self) -> str:
        """Render the file exactly as the Fortran reader expects it."""
        lines = []
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"{value:5.2f}" if f.name in self.F52_FIELDS else str(value))
        return "\n".join(lines) + "\n"

    def write(self, path: Union[str, Path]) -> Path:
        """Write the file to ``path`` and return the path."""
        path = Path(path)
        path.write_text(self.to_text(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Union[str, Path]) -> "InputDA":
        """Parse an existing ``input.da``."""
        raw = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
               if ln.strip() != ""]
        expected = len(fields(cls))
        if len(raw) < expected:
            raise ValueError(
                f"{path} has {len(raw)} value lines but the Fortran reader consumes {expected}"
            )
        kwargs = {}
        for f, token in zip(fields(cls), raw):
            kwargs[f.name] = int(token) if f.type == "int" else float(token)
        return cls(**kwargs)

    def describe(self) -> str:
        """Human-readable summary, for logs and run manifests."""
        mode = "assimilating" if self.da == 1 else "OPEN LOOP (no assimilation)"
        return (
            f"input.da: {mode}; {self.n_en_member} members x {self.n_parameters} perturbed "
            f"parameters; sampling={SAMPLING_METHODS[self.n_sampling_method]}; "
            f"algorithm={DA_ALGORITHMS[self.da_alg]}; sensor branch={SM_SENSORS[self.smsat]}; "
            f"state vector length {self.no_sta_var}, observation vector length {self.no_obs_var}; "
            f"assimilation stops after day {self.da_stopday}; "
            f"diagnostics {'on' if self.wrdtf else 'off'}"
        )
