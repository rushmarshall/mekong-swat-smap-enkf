"""End-to-end checks on the orchestration layer."""

import numpy as np
import pandas as pd
import pytest

from mekong_da.config import MEKONG_2024
from mekong_da.pipeline import (
    REPRODUCTION_TOLERANCE, build_merged_streamflow, prepare_fortran_run,
    reproduce_soil_moisture, streamflow_table,
)


def test_full_soil_moisture_reproduction(data_dir, tmp_path):
    result = reproduce_soil_moisture(data_dir=data_dir, out_dir=tmp_path, verbose=False)
    assert result["reproduced"], f"max difference {result['max_abs_diff']:.2e}"
    assert result["max_abs_diff"] < REPRODUCTION_TOLERANCE
    assert len(result["summary"]) == 3
    for res in ("9km", "1km", "400m"):
        assert (tmp_path / "enkf" / f"da_results_{res}.csv").exists()


def test_reproduction_does_not_touch_the_archive(data_dir, tmp_path):
    """A re-run must never overwrite the series it is checked against."""
    reference = data_dir / "enkf" / "reference_9km.csv"
    before = reference.read_bytes()
    reproduce_soil_moisture(data_dir=data_dir, out_dir=tmp_path, verbose=False)
    assert reference.read_bytes() == before


def test_streamflow_table_covers_every_gauge(data_dir):
    if not sorted((data_dir / "streamflow").glob("*_merged_2024.csv")):
        pytest.skip("merged streamflow series are built from the user's own runs and gauges")
    table = streamflow_table(data_dir=data_dir)
    assert set(table["Station"]) == {"Chiang Saen", "Nong Khai", "Mukdahan", "Pakse", "Kratie"}
    assert set(table["Model"]) == {"Control", "EnKF-9km", "EnKF-1km", "EnKF-400m"}


def test_prepare_control_run_writes_five_files(tmp_path):
    small = MEKONG_2024.__class__(n_hru=4, n_subbasins=2, start_year=2015, n_years=1, nyskip=0, n_days=3)
    out = prepare_fortran_run(tmp_path, resolution=None, spec=small, n_members=2, verbose=False)
    names = {p.name for p in tmp_path.iterdir()}
    assert "input.da" in names and "sample.txt" in names
    assert len([n for n in names if n.endswith(".txt")]) >= 4
    assert out["n_filled"] == 0
    assert out["input_da"].da == 1, "the matched control keeps the ensemble branch"
    assert out["input_da"].wrdtf == 0, "the per-step diagnostic writer is off by default"


def test_merged_streamflow_layout():
    dates = pd.date_range("2005-01-01", periods=10, freq="D")
    frame = build_merged_streamflow(
        "Kratie",
        open_loop=pd.Series(np.arange(10.0), index=dates),
        da_by_resolution={"9km": pd.Series(np.arange(10.0) + 1, index=dates)},
        observed=pd.Series(np.arange(10.0) - 1, index=dates),
    )
    assert list(frame.columns) == ["swat_baseline", "mrc_obs", "swat_da_9km"]
