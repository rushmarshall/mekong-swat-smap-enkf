"""End-to-end orchestration: the steps that turn inputs into the published tables.

The study has two assimilation experiments, and this module keeps them apart because
conflating them is the single easiest mistake to make with this code base:

**Soil-moisture experiment (Python).** A scalar Ensemble Kalman Filter assimilates
CDF-matched SMAP into basin-average SWAT soil moisture. It runs in seconds from the data
shipped as the worked example in ``data/enkf/`` and reproduces the published series bit for bit. Entry point:
:func:`reproduce_soil_moisture`.

**Streamflow experiment (Fortran).** An ensemble Kalman filter inside SWAT2012 updates
soil-water storage per HRU inside the hydrological time loop, so the analysis propagates
into routed discharge. It needs a calibrated SWAT project, which the user supplies.
Entry points: :func:`prepare_fortran_run`, :func:`harvest_fortran_run`,
:func:`build_merged_streamflow` and :func:`streamflow_table`, which scores the merged
daily series a user builds from their own runs and gauge records.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR, MEKONG_2024, OPTIMAL_CONFIG, OUTPUT_DIR, RESOLUTIONS,
    SM_PERIODS, STATIONS, STREAMFLOW_MODELS, STREAMFLOW_PERIODS,
    EnKFConfig, SwatRunSpec,
)
from .enkf import run_enkf_frame
from .inputda import InputDA
from .metrics import metrics_table
from .obsfiles import (
    build_observation_arrays, load_hru_subbasin_map, write_observation_files,
    write_open_loop_observation_files, write_sample_file,
)
from .rch import station_frame

__all__ = [
    "reproduce_soil_moisture", "soil_moisture_table", "streamflow_table",
    "prepare_fortran_run", "harvest_fortran_run", "build_merged_streamflow",
]

#: Tolerance for calling a re-run a faithful reproduction of the archived series.
REPRODUCTION_TOLERANCE = 1e-6


# --------------------------------------------------------------------------------------
# Soil-moisture experiment
# --------------------------------------------------------------------------------------


def reproduce_soil_moisture(
    data_dir: Union[str, Path] = DATA_DIR,
    out_dir: Union[str, Path] = OUTPUT_DIR,
    config: EnKFConfig = OPTIMAL_CONFIG,
    resolutions: Sequence[str] = RESOLUTIONS,
    verbose: bool = True,
) -> Dict[str, object]:
    """Re-run the filter at every resolution and check it against the archived series.

    Reads ``data/enkf/inputs_<res>.csv``, writes ``outputs/enkf/da_results_<res>.csv``,
    and compares the regenerated analysis with ``data/enkf/reference_<res>.csv``. The
    archived reference is never overwritten, so a re-run cannot quietly redefine the thing
    it is being checked against.

    Returns
    -------
    dict
        ``{"frames": {res: DataFrame}, "summary": DataFrame, "max_abs_diff": float,
        "reproduced": bool}`` where ``summary`` carries, per resolution, the open-loop and
        analysis root-mean-square error against the assimilated observation, the reduction
        in per cent, and the reproduction difference.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    frames: Dict[str, pd.DataFrame] = {}
    rows = []
    worst = 0.0

    for res in resolutions:
        inputs = pd.read_csv(data_dir / "enkf" / f"inputs_{res}.csv")
        frame = run_enkf_frame(inputs, config=config)
        frames[res] = frame

        target = out_dir / "enkf" / f"da_results_{res}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)

        reference_path = data_dir / "enkf" / f"reference_{res}.csv"
        diff = np.nan
        if reference_path.exists():
            reference = pd.read_csv(reference_path)
            diff = float(np.nanmax(np.abs(frame["analysis"].to_numpy() - reference["analysis"].to_numpy())))
            worst = max(worst, diff)

        available = frame["obs_available"].to_numpy(bool)
        obs = frame["observation_corrected"].to_numpy(float)[available]
        rmse_open_loop = float(np.sqrt(np.mean((frame["baseline"].to_numpy(float)[available] - obs) ** 2)))
        rmse_analysis = float(np.sqrt(np.mean((frame["analysis"].to_numpy(float)[available] - obs) ** 2)))
        rows.append(
            dict(
                Resolution=res,
                N_assimilated=int(available.sum()),
                RMSE_open_loop=rmse_open_loop,
                RMSE_EnKF=rmse_analysis,
                RMSE_reduction_pct=100 * (rmse_open_loop - rmse_analysis) / rmse_open_loop,
                Max_abs_diff_vs_archive=diff,
            )
        )
        if verbose:
            print(
                f"  {res:>5s}  n={int(available.sum()):5d}  RMSE {rmse_open_loop:.4f} -> "
                f"{rmse_analysis:.4f}  ({100*(rmse_open_loop-rmse_analysis)/rmse_open_loop:.1f}% lower)"
                f"  |archive diff| = {diff:.2e}"
            )

    summary = pd.DataFrame(rows)
    return {
        "frames": frames,
        "summary": summary,
        "max_abs_diff": worst,
        "reproduced": worst < REPRODUCTION_TOLERANCE,
    }


def soil_moisture_table(
    frames: Dict[str, pd.DataFrame],
    periods: Dict[str, tuple] = SM_PERIODS,
) -> pd.DataFrame:
    """Build the soil-moisture skill table (T1) from re-run filter output.

    Skill is scored against the assimilated observation on assimilation days only, for the
    open loop and for the analysis, over each evaluation window.
    """
    parts = []
    for res, frame in frames.items():
        parts.append(
            metrics_table(
                frame,
                obs_col="observation_corrected",
                model_cols={"Open-Loop": "baseline", "EnKF": "analysis"},
                periods=periods,
                row_filter=frame["obs_available"],
                Resolution=res,
            )
        )
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------
# Streamflow experiment: analysis of archived model output
# --------------------------------------------------------------------------------------


def streamflow_table(
    data_dir: Union[str, Path] = DATA_DIR,
    periods: Dict[str, tuple] = STREAMFLOW_PERIODS,
    models: Dict[str, str] = STREAMFLOW_MODELS,
) -> pd.DataFrame:
    """Build the streamflow skill table (T2) from the merged daily series.

    Reads ``<data_dir>/streamflow/<Station>_merged_2024.csv``, written by
    :func:`build_merged_streamflow` from a user's own runs and gauge observations (the
    control in ``swat_baseline``, each assimilating arm in ``swat_da_<res>``, the gauge
    record in ``mrc_obs``), and scores every arm at every gauge over every window. Gauge
    observations are not distributed with this repository.
    """
    data_dir = Path(data_dir)
    parts = []
    for station in STATIONS:
        path = data_dir / "streamflow" / f"{station.key}_merged_2024.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        parts.append(
            metrics_table(
                frame, obs_col="mrc_obs", model_cols=models, periods=periods,
                Station=station.label,
            )
        )
    if not parts:
        raise FileNotFoundError(f"no merged streamflow series found under {data_dir/'streamflow'}")
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------
# Streamflow experiment: driving the Fortran model
# --------------------------------------------------------------------------------------


def prepare_fortran_run(
    run_dir: Union[str, Path],
    resolution: Optional[str],
    template_dir: Optional[Union[str, Path]] = None,
    data_dir: Union[str, Path] = DATA_DIR,
    spec: SwatRunSpec = MEKONG_2024,
    input_da: Optional[InputDA] = None,
    n_members: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Stage a SWAT project directory for one assimilation arm.

    Parameters
    ----------
    run_dir:
        Directory the model will run in. Created if absent.
    resolution:
        SMAP product to assimilate, or ``None`` for the matched ensemble control, which
        writes all-missing observation files. Every arm, the control included, runs with
        ``da=1`` so that all four share the same ensemble dynamics, initial states,
        forcing and random draws, and differ only in whether observations are present.
    template_dir:
        A calibrated SWAT ``TxtInOut`` to copy in first. Model output files are skipped so
        a stale ``output.rch`` cannot be mistaken for a fresh one. Pass ``None`` when
        ``run_dir`` already holds the project.
    data_dir:
        Repository data directory, holding the SMAP subbasin series and the HRU map.
    spec:
        Model geometry and calendar. Must match the project in ``template_dir``.
    input_da:
        Control-file settings. Defaults to the published configuration with ``da`` and
        ``n_en_member`` set from the other arguments.
    n_members:
        Ensemble size. Also determines the row count of ``sample.txt``.
    seed:
        Seed for the perturbation draws.

    Returns
    -------
    dict
        ``{"run_dir": Path, "n_filled": int, "input_da": InputDA}`` where ``n_filled``
        counts the HRU-days carrying an assimilable retrieval (zero for the control).
    """
    if template_dir is not None and not Path(template_dir).is_dir():
        raise NotADirectoryError(
            f"--template is not a directory: {template_dir}. It must point at the "
            "calibrated SWAT project's TxtInOut folder, which this repository "
            "does not ship. See docs/04_coupled_model.md."
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    if template_dir is not None:
        template_dir = Path(template_dir)
        skip_prefixes = (
            "output.", "result_", "watout.", "hyd.out", "chan.deg", "input.std",
            "sm_obs1.", "sm_obs2.", "da_flag.", "correlation.", "inflow_flag.",
        )
        copied = 0
        for item in template_dir.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith(skip_prefixes) or item.name.endswith(".db3"):
                continue
            shutil.copy2(item, run_dir / item.name)
            copied += 1
        if verbose:
            print(f"  staged {copied} project files from {template_dir}")

    if resolution is None:
        write_open_loop_observation_files(run_dir, spec=spec)
        n_filled = 0
    else:
        smap_path = data_dir / "smap_subbasin_daily" / f"smap_{resolution}_subbasin_daily.csv"
        if not smap_path.exists():
            raise FileNotFoundError(f"SMAP series not found: {smap_path}")
        smap = pd.read_csv(smap_path, index_col="date", parse_dates=True)
        hru_map = load_hru_subbasin_map(data_dir / "hru_subbasin_mapping.csv")
        arrays = build_observation_arrays(smap, hru_map, spec=spec)
        write_observation_files(run_dir, arrays)
        n_filled = int(arrays["n_filled"])
        if verbose:
            total = spec.n_days * spec.n_hru
            print(f"  {resolution}: {n_filled:,} of {total:,} HRU-days assimilable ({100*n_filled/total:.2f}%)")

    write_sample_file(run_dir, n_members=n_members, spec=spec, seed=seed)

    settings = input_da or InputDA()
    settings = InputDA(**{**settings.__dict__, "da": 1, "n_en_member": n_members})
    settings.write(run_dir / "input.da")
    if verbose:
        print(f"  {settings.describe()}")

    return {"run_dir": run_dir, "n_filled": n_filled, "input_da": settings}


def harvest_fortran_run(
    rch_path: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    spec: SwatRunSpec = MEKONG_2024,
) -> pd.DataFrame:
    """Extract the five gauge hydrographs from a completed run's ``output.rch``."""
    frame = station_frame(rch_path, spec=spec)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_csv)
    return frame


def build_merged_streamflow(
    station_key: str,
    open_loop: pd.Series,
    da_by_resolution: Dict[str, pd.Series],
    observed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Assemble one gauge's comparison frame from harvested model output.

    Produces the column layout :func:`streamflow_table` reads, so harvested runs drop
    straight into it. ``open_loop`` is the matched-control series.
    """
    frame = pd.DataFrame({"swat_baseline": open_loop})
    frame.index.name = "date"
    if observed is not None:
        frame["mrc_obs"] = observed.reindex(frame.index)
    for res, series in da_by_resolution.items():
        frame[f"swat_da_{res}"] = series.reindex(frame.index)
    ordered = ["swat_baseline", "mrc_obs"] + [f"swat_da_{r}" for r in da_by_resolution]
    return frame[[c for c in ordered if c in frame.columns]]
