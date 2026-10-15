"""The parameter search must still reproduce the archived grids and pick the same winner."""

import numpy as np
import pandas as pd
import pytest

from mekong_da.biascorrect import cdf_match_rank
from mekong_da.config import DEFAULT_CONFIG, OPTIMAL_CONFIG
from mekong_da.tuning import (
    MODEL_ERROR_GRID, OBS_ERROR_GRID, TRAIN_END, score, sweep,
)


@pytest.fixture(scope="module")
def tuning_inputs(request):
    from mekong_da.config import DATA_DIR

    folder = DATA_DIR / "enkf_tuning" / "inputs"
    baseline = folder / "swat_baseline_smap_2015_2020.csv"
    smap = folder / "smap_9km_basin_mean_daily.csv"
    if not (baseline.exists() and smap.exists()):
        pytest.skip("tuning inputs not vendored")
    open_loop = pd.read_csv(baseline, parse_dates=["date"], index_col="date")["sm_baseline"]
    raw = pd.read_csv(smap, parse_dates=["date"], index_col="date").iloc[:, 0]
    train = np.asarray(open_loop.index <= TRAIN_END)
    corrected = pd.Series(
        cdf_match_rank(raw.reindex(open_loop.index).to_numpy(float), open_loop.to_numpy(float), train_mask=train),
        index=open_loop.index,
    )
    return open_loop, corrected


def test_stage1a_reproduces_the_archived_grid(tuning_inputs, data_dir):
    open_loop, observation = tuning_inputs
    archived = pd.read_csv(data_dir / "enkf_tuning" / "stage1a_obs_model_error.csv")
    got = sweep(
        open_loop, observation, DEFAULT_CONFIG,
        {"obs_error_std": OBS_ERROR_GRID, "model_error_std": MODEL_ERROR_GRID}, verbose=False,
    )
    key = ["obs_error_std", "model_error_std"]
    a = got.set_index(key).sort_index()
    b = archived.set_index(key).sort_index()
    columns = [c for c in b.columns if c in a.columns and c not in ("inflation_factor", "persistence_phi")]
    assert len(a) == len(b) == 16
    assert np.nanmax(np.abs((a[columns] - b[columns]).to_numpy())) < 1e-9


def test_selection_lands_on_the_published_configuration(data_dir):
    stage1b = pd.read_csv(data_dir / "enkf_tuning" / "stage1b_inflation_phi.csv")
    winner = stage1b.loc[stage1b.apply(score, axis=1).idxmax()]
    assert winner["inflation_factor"] == pytest.approx(OPTIMAL_CONFIG.inflation_factor)
    assert winner["persistence_phi"] == pytest.approx(OPTIMAL_CONFIG.persistence_phi)


def test_tuning_improved_innovation_whiteness(data_dir):
    """The point of the tuning: a filter whose innovations stopped ringing."""
    validation = pd.read_csv(data_dir / "enkf_tuning" / "validation_default_vs_optimal.csv")
    for res in validation["resolution"].unique():
        block = validation[validation["resolution"] == res].set_index("config")
        assert abs(block.loc["OPTIMAL", "innov_autocorr_1"]) < abs(block.loc["DEFAULT", "innov_autocorr_1"])


def test_stage2_reproduces_the_archived_validation(data_dir):
    """The multi-resolution validation must reproduce for every product whose reduced
    basin-mean series is vendored."""
    from mekong_da.config import DEFAULT_CONFIG, OPTIMAL_CONFIG, RESOLUTIONS
    from mekong_da.tuning import evaluate_config

    folder = data_dir / "enkf_tuning" / "inputs"
    baseline = folder / "swat_baseline_smap_2015_2020.csv"
    if not baseline.exists():
        pytest.skip("tuning inputs not vendored")
    open_loop = pd.read_csv(baseline, parse_dates=["date"], index_col="date")["sm_baseline"]
    train = np.asarray(open_loop.index <= TRAIN_END)
    archived = pd.read_csv(data_dir / "enkf_tuning" / "validation_default_vs_optimal.csv")
    columns = ["innov_autocorr_1", "mean_K", "innov_var_ratio", "nse_da", "kge_da",
               "rmse_da", "rmse_da_val", "nse_da_val"]

    checked = 0
    for res in RESOLUTIONS:
        path = folder / f"smap_{res}_basin_mean_daily.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["date"], index_col="date").iloc[:, 0]
        observation = pd.Series(
            cdf_match_rank(raw.reindex(open_loop.index).to_numpy(float),
                           open_loop.to_numpy(float), train_mask=train),
            index=open_loop.index,
        )
        for label, config in (("DEFAULT", DEFAULT_CONFIG), ("OPTIMAL", OPTIMAL_CONFIG)):
            got = evaluate_config(open_loop, observation, config)
            expected = archived[(archived.resolution == res) & (archived.config == label)].iloc[0]
            for column in columns:
                assert got[column] == pytest.approx(expected[column], abs=1e-9), f"{res} {label} {column}"
            checked += 1
    assert checked >= 2, "at least the 9 km and 1 km validations should be reproducible"
