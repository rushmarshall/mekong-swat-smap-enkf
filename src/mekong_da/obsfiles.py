"""Writers for the five plain-text files the Fortran SWAT-DA build reads.

Beyond a standard SWAT project, the assimilating build needs:

===========================  ===============================  =====================================
file                          shape                            content
===========================  ===============================  =====================================
``sm_obs`` file               n_days x n_hru                   quality-controlled soil moisture
``dqx`` file                  n_days x n_hru                   retrieval data-quality index
``rfi`` file                  n_days x n_hru                   radio-frequency interference index
``sample.txt``                (n_days * n_members) x n_params   standard normal perturbation draws
``input.da``                  22 lines                         see :mod:`mekong_da.inputda`
===========================  ===============================  =====================================

The Mekong application inherits the SMOS branch of the upstream code, so the three
observation files keep the upstream SMOS names. Absent retrievals are not blank and not
NaN: they are sentinel values that make the Fortran skip the update for that HRU-day.

The screening test lives in ``fortran/src/simulate.f`` and reads, per HRU and day::

    if (d5 > 0. .and. d2 < 0.4 .and. d3 < 0.1)   ! soil moisture, RFI, DQX

Read that carefully, because the missing soil-moisture sentinel is 0.001, which is
**greater than zero and therefore passes the first test on its own**. What actually
suppresses an update is the DQX and RFI sentinels of 1.0 failing the second and third
tests. A run given a soil-moisture file full of sentinels but a quality file full of
valid flags will happily assimilate 0.001 m3/m3 into every HRU on every day, and it will
not warn. Always write all three files together.

All three files, and ``sample.txt``, are read list-directed
(``read (unit,*) ...``), so column alignment is free but the row and column counts are
not: they must equal the simulated day count and the HRU count exactly.

These files are large. At the published geometry (9862 days, 5605 HRUs) each observation
file is roughly 400 MB and ``sample.txt`` roughly 165 MB, so they are generated into a run
directory rather than stored.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import MEKONG_2024, SwatRunSpec

__all__ = [
    "SM_FILE", "DQX_FILE", "RFI_FILE",
    "MISSING_SM", "MISSING_DQX", "MISSING_RFI", "VALID_DQX", "VALID_RFI",
    "load_hru_subbasin_map", "build_observation_arrays",
    "write_observation_files", "write_open_loop_observation_files", "write_sample_file",
]

#: Filenames the Fortran opens by literal name. Renaming them breaks the run.
SM_FILE = "sm_ad_mn_std_monthly_marol.txt"
DQX_FILE = "smdqx_ad_marol.txt"
RFI_FILE = "smrfi_ad_marol.txt"

#: Sentinels marking "no usable retrieval on this HRU-day".
MISSING_SM = 0.001   #: the Fortran skips the update when soil moisture equals this
MISSING_DQX = 1.0    #: the Fortran skips the update when DQX exceeds its threshold
MISSING_RFI = 1.0    #: the Fortran skips the update when RFI exceeds its threshold

#: Flags marking a retrieval that passed screening.
VALID_DQX = 0.01
VALID_RFI = 0.01

#: Physical acceptance window applied to every retrieval before it is written, m3/m3.
SM_VALID_RANGE = (0.01, 0.60)


def load_hru_subbasin_map(path: Union[str, Path]) -> np.ndarray:
    """Return the subbasin identifier of every HRU, in HRU order.

    Parameters
    ----------
    path:
        CSV with columns ``hru_index``, ``subbasin``, ``hru_in_sub``.

    Returns
    -------
    numpy.ndarray
        Integer array of length n_hru. Position i holds the subbasin of HRU i+1.
    """
    df = pd.read_csv(path)
    for col in ("hru_index", "subbasin"):
        if col not in df.columns:
            raise ValueError(f"{path} is missing the '{col}' column")
    df = df.sort_values("hru_index")
    return df["subbasin"].to_numpy(int)


def build_observation_arrays(
    smap: pd.DataFrame,
    hru_subbasins: Sequence[int],
    spec: SwatRunSpec = MEKONG_2024,
) -> Dict[str, np.ndarray]:
    """Expand a daily subbasin soil-moisture table onto the HRU-by-day model grid.

    Every HRU inherits the retrieval of the subbasin it sits in, which is the resolution
    at which SMAP was extracted. Days outside the simulation calendar are dropped; days
    inside it with no retrieval keep the missing sentinels.

    Parameters
    ----------
    smap:
        Daily frame indexed by date with one column per subbasin, named ``sub_<id>``.
    hru_subbasins:
        Subbasin identifier per HRU, from :func:`load_hru_subbasin_map`.
    spec:
        Model geometry and calendar.

    Returns
    -------
    dict
        ``{"sm": ..., "dqx": ..., "rfi": ..., "n_filled": ...}`` where the three arrays
        have shape ``(spec.n_days, spec.n_hru)`` and ``n_filled`` counts assimilable cells.
    """
    hru_subbasins = np.asarray(hru_subbasins, dtype=int)
    n_hru = len(hru_subbasins)
    if n_hru != spec.n_hru:
        raise ValueError(
            f"HRU map has {n_hru} rows but the run spec declares {spec.n_hru}; "
            "the observation files must match the model geometry exactly"
        )

    sub_col = {int(c.replace("sub_", "")): c for c in smap.columns if c.startswith("sub_")}
    if not sub_col:
        raise ValueError("no 'sub_<id>' columns found in the SMAP frame")
    # Column position in the SMAP frame for each HRU, or -1 where the subbasin is absent.
    col_names = list(smap.columns)
    hru_col_pos = np.array(
        [col_names.index(sub_col[s]) if s in sub_col else -1 for s in hru_subbasins]
    )

    sm = np.full((spec.n_days, n_hru), MISSING_SM, dtype=np.float32)
    dqx = np.full((spec.n_days, n_hru), MISSING_DQX, dtype=np.float32)
    rfi = np.full((spec.n_days, n_hru), MISSING_RFI, dtype=np.float32)

    calendar = pd.date_range(f"{spec.start_year}-01-01", periods=spec.n_days, freq="D")
    day_of = {d: i for i, d in enumerate(calendar)}

    values = smap.to_numpy(dtype=float)
    lo, hi = SM_VALID_RANGE
    n_filled = 0
    for row_idx, date in enumerate(smap.index):
        day = day_of.get(pd.Timestamp(date))
        if day is None:
            continue
        row = values[row_idx]
        # -1 positions read the last column; the validity mask below discards them.
        per_hru = row[hru_col_pos]
        good = (hru_col_pos >= 0) & np.isfinite(per_hru) & (per_hru > lo) & (per_hru < hi)
        sm[day, good] = per_hru[good]
        dqx[day, good] = VALID_DQX
        rfi[day, good] = VALID_RFI
        n_filled += int(good.sum())

    return {"sm": sm, "dqx": dqx, "rfi": rfi, "n_filled": n_filled}


def _write_matrix(path: Path, array: np.ndarray) -> Path:
    """Write a float matrix as space-separated text, six decimals, one row per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in array:
            fh.write(" ".join(f"{v:.6f}" for v in row))
            fh.write("\n")
    return path


def write_observation_files(
    run_dir: Union[str, Path],
    arrays: Dict[str, np.ndarray],
) -> Dict[str, Path]:
    """Write the soil-moisture, DQX and RFI files into a run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        SM_FILE: _write_matrix(run_dir / SM_FILE, arrays["sm"]),
        DQX_FILE: _write_matrix(run_dir / DQX_FILE, arrays["dqx"]),
        RFI_FILE: _write_matrix(run_dir / RFI_FILE, arrays["rfi"]),
    }


def write_open_loop_observation_files(
    run_dir: Union[str, Path],
    spec: SwatRunSpec = MEKONG_2024,
) -> Dict[str, Path]:
    """Write all-missing observation files for the matched ensemble control.

    The control runs the ensemble branch (``da=1``) like every assimilating arm, and these
    sentinel-only files are what suppress every update, so it shares the arms' initial
    states, forcing and random draws. See ``docs/04_coupled_model.md``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, value in ((SM_FILE, MISSING_SM), (DQX_FILE, MISSING_DQX), (RFI_FILE, MISSING_RFI)):
        line = " ".join([f"{value:.6f}"] * spec.n_hru) + "\n"
        with open(run_dir / name, "w", encoding="utf-8") as fh:
            for _ in range(spec.n_days):
                fh.write(line)
        written[name] = run_dir / name
    return written


def write_sample_file(
    run_dir: Union[str, Path],
    n_members: int = 200,
    n_parameters: int = 6,
    spec: SwatRunSpec = MEKONG_2024,
    seed: int = 42,
    filename: str = "sample.txt",
) -> Path:
    """Write ``sample.txt``, the standard normal draws the Fortran ensemble consumes.

    ``fortran/src/rngen.f90`` allocates ``n_days_sim * n_ensembles * n_en_member`` rows of
    ``n_parameters`` columns and reads them list-directed, then walks them sequentially as
    the simulation advances. With ``n_ensembles = 1`` that is ``n_days * n_members`` rows.
    Too few rows ends the run with an end-of-file error; too many are ignored. Values are
    written in ``14.8f`` fields for legibility, not because the reader requires it.

    The seed is fixed for the same reason it is fixed in the Python filter: without it,
    two runs of an identical configuration are not comparable.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    n_rows = spec.n_days * n_members

    np.random.seed(seed)
    samples = np.random.randn(n_rows, n_parameters)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n_rows):
            fh.write("".join(f"{samples[i, j]:14.8f}" for j in range(n_parameters)))
            fh.write("\n")
    return path
