#!/usr/bin/env python3
"""Reduce raw SMAP NetCDF archives to the basin-mean daily series the tuning consumes.

The Mekong series it produced ship as the worked example in
``data/enkf_tuning/inputs/``. This script records how they were produced, and lets anyone
holding the raw archives re-derive them.

The archives are large (84 GB, 9.5 GB and 26 GB for the 9 km, 1 km and 400 m products), so
a full pass takes minutes to tens of minutes per product.

    pip install -e ".[raw]"
    python scripts/extract_basin_mean.py --archive 9km=/path/to/SPL3SMP_E_9km.nc

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Mekong window used for the basin mean, degrees.
LAT_MIN, LAT_MAX = 8.0, 34.0
LON_MIN, LON_MAX = 94.0, 110.0

#: Values outside this range are treated as fill or as physically impossible.
VALID_RANGE = (0.0, 1.0)
FILL_VALUE = -9999


def reduce_archive(path: Path, label: str) -> pd.Series:
    """Return the daily basin-mean soil-moisture series from one NetCDF archive."""
    import xarray as xr  # imported here so the module loads without the optional extras

    ds = xr.open_dataset(path, chunks={"time": 50})
    try:
        lat = "latitude" if "latitude" in ds.coords else "lat"
        lon = "longitude" if "longitude" in ds.coords else "lon"
        variable = next(
            (v for v in ("soil_moisture", "sm", "SM", "ssm") if v in ds.data_vars),
            list(ds.data_vars)[0],
        )
        ascending = ds[lat].values[0] < ds[lat].values[-1]
        window = ds.sel(
            **{
                lat: slice(LAT_MIN, LAT_MAX) if ascending else slice(LAT_MAX, LAT_MIN),
                lon: slice(LON_MIN, LON_MAX),
            }
        )
        print(f"[{label}] variable={variable} shape={window[variable].shape}", flush=True)
        reduced = window[variable].mean(dim=[lat, lon]).compute()
    finally:
        ds.close()

    series = pd.Series(reduced.values, index=pd.DatetimeIndex(reduced.time.values))
    series = series.replace(FILL_VALUE, np.nan).dropna()
    lo, hi = VALID_RANGE
    series = series[(series > lo) & (series < hi)]
    series.index.name = "date"
    series.name = f"smap_{label}_basin_mean"
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive", action="append", required=True, metavar="LABEL=PATH",
        help="resolution label and NetCDF path, for example 9km=/data/SPL3SMP_E.nc. Repeatable.",
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "enkf_tuning" / "inputs"),
    )
    args = parser.parse_args()

    # HDF5 file locking makes a read-only open fail with a permission error on many systems.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.archive:
        if "=" not in spec:
            parser.error(f"--archive expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        series = reduce_archive(Path(path), label)
        target = out_dir / f"smap_{label}_basin_mean_daily.csv"
        series.to_csv(target)
        print(f"[{label}] {len(series)} days, mean {series.mean():.4f} m3/m3 -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
