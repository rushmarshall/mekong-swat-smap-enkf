"""Metric definitions, checked against hand-computable cases and the published tables."""

import numpy as np
import pandas as pd
import pytest

from mekong_da.metrics import metrics, metrics_table


def test_perfect_prediction():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = metrics(obs, obs)
    assert m["RMSE"] == 0.0
    assert m["NSE"] == pytest.approx(1.0)
    assert m["KGE"] == pytest.approx(1.0)
    assert m["R"] == pytest.approx(1.0)
    assert m["PBIAS"] == pytest.approx(0.0)


def test_constant_offset_is_pure_bias():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = metrics(obs + 2.0, obs)
    assert m["Bias"] == pytest.approx(2.0)
    assert m["ubRMSE"] == pytest.approx(0.0, abs=1e-12)
    assert m["R"] == pytest.approx(1.0)


def test_pairwise_complete_only():
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    sim = np.array([1.0, np.nan, 3.0, 4.0])
    assert metrics(sim, obs)["N"] == 2


def test_too_few_points_returns_nan():
    m = metrics([1.0, 2.0], [1.0, 2.0])
    assert m["N"] == 2 and np.isnan(m["RMSE"])


def test_kge_uses_the_standard_deviation_ratio():
    """The variability term is sd(sim)/sd(obs), not the coefficient-of-variation ratio."""
    rng = np.random.default_rng(0)
    obs = rng.normal(10, 2, 500)
    sim = obs * 1.3
    m = metrics(sim, obs)
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std(ddof=0) / obs.std(ddof=0)
    beta = sim.mean() / obs.mean()
    assert m["KGE"] == pytest.approx(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def test_metrics_table_windows_and_labels():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    frame = pd.DataFrame({"date": dates, "obs": np.arange(100.0), "a": np.arange(100.0)})
    out = metrics_table(
        frame, obs_col="obs", model_cols={"A": "a"},
        periods={"first50": ("2020-01-01", "2020-02-19")}, Station="X",
    )
    assert list(out.columns[:3]) == ["Station", "Period", "Model"]
    assert out.iloc[0]["N"] == 50


def test_t1_matches_the_published_table(data_dir):
    """Regenerating the soil-moisture table must land on the published numbers."""
    from mekong_da.pipeline import reproduce_soil_moisture, soil_moisture_table

    reference_path = data_dir / "reference_metrics" / "T1_soil_moisture_metrics.csv"
    if not reference_path.exists():
        pytest.skip("published T1 not vendored")
    result = reproduce_soil_moisture(data_dir=data_dir, out_dir=data_dir.parent / "outputs", verbose=False)
    got = soil_moisture_table(result["frames"])
    expected = pd.read_csv(reference_path)
    merged = got.merge(expected, on=["Resolution", "Period", "Model"], suffixes=("", "_ref"))
    assert len(merged) == len(expected)
    for col in ("RMSE", "MAE", "Bias", "R", "NSE", "KGE", "PBIAS", "ubRMSE"):
        assert np.nanmax(np.abs(merged[col] - merged[f"{col}_ref"])) < 1e-9, col


def test_t2_matches_the_published_table(data_dir):
    """Regenerating the streamflow table from the archived series must do the same."""
    from mekong_da.pipeline import streamflow_table

    reference_path = data_dir / "reference_metrics" / "T2_streamflow_metrics_from_merged2024.csv"
    if not reference_path.exists():
        pytest.skip("published T2 not vendored")
    got = streamflow_table(data_dir=data_dir)
    expected = pd.read_csv(reference_path)
    merged = got.merge(expected, on=["Station", "Period", "Model"], suffixes=("", "_ref"))
    assert len(merged) == len(expected)
    for col in ("RMSE", "MAE", "Bias", "R", "NSE", "KGE", "PBIAS"):
        assert np.nanmax(np.abs(merged[col] - merged[f"{col}_ref"])) < 1e-9, col


def test_metrics_argument_order_is_prediction_then_reference():
    """`metrics(pred, obs)`, and getting it backwards silently changes NSE and PBIAS.

    R and RMSE are symmetric in their arguments, so a reversed call looks right in the two
    columns people check first and is wrong in the two that carry the sign. A validation
    script shipped with the arguments reversed at both call sites for exactly that reason.
    """
    import numpy as np

    from mekong_da.metrics import metrics

    reference = np.array([10.0, 20.0, 30.0, 40.0])
    prediction = reference * 1.10                      # ten per cent high

    forward = metrics(prediction, reference)
    reversed_ = metrics(reference, prediction)

    # the symmetric pair cannot catch the mistake
    assert forward["R"] == pytest.approx(reversed_["R"])
    assert forward["RMSE"] == pytest.approx(reversed_["RMSE"])

    # the asymmetric pair can, and the sign follows the prediction
    assert forward["PBIAS"] == pytest.approx(10.0), "a high prediction is a positive bias"
    assert reversed_["PBIAS"] != pytest.approx(forward["PBIAS"])
    assert forward["NSE"] != pytest.approx(reversed_["NSE"])
