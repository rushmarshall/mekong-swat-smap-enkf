"""``output.rch`` parsing, against a synthetic file in the real SWAT2012 layout."""

import numpy as np
import pytest

from mekong_da.config import SwatRunSpec
from mekong_da.rch import FLOW_IN_INDEX, FLOW_OUT_INDEX, parse_output_rch

HEADER = "\n".join(
    [
        "1",
        "    SWAT Aug 25 2014    VER 2012/Rev 629",
        "",
        "    General Input/Output section (file.cio):",
        "    2/8/2026 12:00:00 AM ARCGIS-SWAT interface AV",
        "",
        "",
        "",
        "       RCH      GIS   MON     AREAkm2  FLOW_INcms FLOW_OUTcms",
    ]
)


def _write(path, n_days=5, n_reaches=4):
    lines = [HEADER]
    for day in range(n_days):
        for reach in range(1, n_reaches + 1):
            inflow = 1000 * reach + day
            outflow = inflow + 0.5
            lines.append(
                f"REACH {reach:8d} {0:8d} {day % 12 + 1:5d}  0.7556E+06  "
                f"{inflow:.4E}  {outflow:.4E}"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reads_outflow_not_inflow(tmp_path):
    path = _write(tmp_path / "output.rch")
    spec = SwatRunSpec(start_year=2000, nyskip=0)
    out = parse_output_rch(path, [2], spec=spec)
    assert out["reach_2"].iloc[0] == pytest.approx(2000.5)
    inflow = parse_output_rch(path, [2], spec=spec, column=FLOW_IN_INDEX)
    assert inflow["reach_2"].iloc[0] == pytest.approx(2000.0)
    assert FLOW_OUT_INDEX != FLOW_IN_INDEX


def test_day_count_and_dates(tmp_path):
    path = _write(tmp_path / "output.rch", n_days=7)
    out = parse_output_rch(path, [1, 3], spec=SwatRunSpec(start_year=2000, nyskip=1))
    assert len(out) == 7
    assert str(out.index[0].date()) == "2001-01-01"


def test_absent_reach_becomes_a_nan_column(tmp_path):
    """A missing gauge must be visible, not silently dropped."""
    path = _write(tmp_path / "output.rch", n_reaches=3)
    out = parse_output_rch(path, [1, 99], spec=SwatRunSpec(start_year=2000, nyskip=0))
    assert "reach_99" in out.columns
    assert out["reach_99"].isna().all()


def test_empty_file_raises(tmp_path):
    path = tmp_path / "output.rch"
    path.write_text(HEADER + "\n")
    with pytest.raises(ValueError, match="no REACH records"):
        parse_output_rch(path, [1])
