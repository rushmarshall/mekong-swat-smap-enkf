"""The Fortran-facing observation files: shape, sentinels, and field widths."""

import numpy as np
import pandas as pd
import pytest

from mekong_da.config import SwatRunSpec
from mekong_da.obsfiles import (
    MISSING_DQX, MISSING_RFI, MISSING_SM, VALID_DQX,
    build_observation_arrays, load_hru_subbasin_map, write_observation_files,
    write_open_loop_observation_files, write_sample_file,
)

SPEC = SwatRunSpec(n_hru=6, n_subbasins=3, start_year=2015, n_years=1, nyskip=0, n_days=5)


def _smap():
    dates = pd.date_range("2015-01-02", periods=3, freq="D")
    return pd.DataFrame(
        {"sub_1": [0.25, np.nan, 0.30], "sub_2": [0.9, 0.20, 0.20], "sub_3": [0.15, 0.15, 0.15]},
        index=dates,
    )


HRU_SUBS = [1, 1, 2, 2, 3, 3]


def test_shape_matches_the_model_geometry():
    arrays = build_observation_arrays(_smap(), HRU_SUBS, spec=SPEC)
    assert arrays["sm"].shape == (SPEC.n_days, SPEC.n_hru)


def test_absent_days_carry_the_missing_sentinels():
    arrays = build_observation_arrays(_smap(), HRU_SUBS, spec=SPEC)
    assert np.all(arrays["sm"][0] == MISSING_SM)      # 2015-01-01 has no retrieval
    assert np.all(arrays["dqx"][0] == MISSING_DQX)
    assert np.all(arrays["rfi"][0] == MISSING_RFI)


def test_out_of_range_retrievals_are_rejected():
    """0.9 m3/m3 is outside the physical window and must not reach the filter."""
    arrays = build_observation_arrays(_smap(), HRU_SUBS, spec=SPEC)
    assert arrays["sm"][1, 2] == MISSING_SM           # sub_2 on day 2 held 0.9
    assert arrays["sm"][1, 0] == pytest.approx(0.25, abs=1e-6)
    assert arrays["dqx"][1, 0] == pytest.approx(VALID_DQX)


def test_every_hru_inherits_its_subbasin():
    arrays = build_observation_arrays(_smap(), HRU_SUBS, spec=SPEC)
    assert arrays["sm"][1, 0] == arrays["sm"][1, 1]   # both HRUs sit in subbasin 1


def test_geometry_mismatch_raises():
    with pytest.raises(ValueError, match="model geometry"):
        build_observation_arrays(_smap(), [1, 2], spec=SPEC)


def test_written_files_have_one_line_per_day(tmp_path):
    arrays = build_observation_arrays(_smap(), HRU_SUBS, spec=SPEC)
    written = write_observation_files(tmp_path, arrays)
    for path in written.values():
        lines = path.read_text().strip().splitlines()
        assert len(lines) == SPEC.n_days
        assert all(len(line.split()) == SPEC.n_hru for line in lines)


def test_open_loop_files_are_all_sentinel(tmp_path):
    written = write_open_loop_observation_files(tmp_path, spec=SPEC)
    values = {float(v) for v in list(written.values())[0].read_text().split()}
    assert values == {MISSING_SM}


def test_sample_file_shape(tmp_path):
    """Row count is what the Fortran allocator demands; column alignment is cosmetic."""
    path = write_sample_file(tmp_path, n_members=3, n_parameters=2, spec=SPEC)
    lines = path.read_text().splitlines()
    assert len(lines) == SPEC.n_days * 3
    assert all(len(line.split()) == 2 for line in lines)


def test_sample_file_is_seeded(tmp_path):
    a = write_sample_file(tmp_path / "a", n_members=2, spec=SPEC).read_text()
    b = write_sample_file(tmp_path / "b", n_members=2, spec=SPEC).read_text()
    assert a == b


def test_hru_map_is_read_in_hru_order(tmp_path):
    path = tmp_path / "map.csv"
    path.write_text("hru_index,subbasin,hru_in_sub\n3,7,1\n1,5,1\n2,5,2\n")
    np.testing.assert_array_equal(load_hru_subbasin_map(path), [5, 5, 7])


def test_real_hru_map_matches_the_published_geometry(data_dir):
    from mekong_da.config import MEKONG_2024

    path = data_dir / "hru_subbasin_mapping.csv"
    if not path.exists():
        pytest.skip("the HRU-to-subbasin map comes from the user's own SWAT project")
    hru = load_hru_subbasin_map(path)
    assert len(hru) == MEKONG_2024.n_hru
    assert len(set(hru)) == MEKONG_2024.n_subbasins
