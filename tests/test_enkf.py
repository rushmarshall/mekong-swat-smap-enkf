"""The reproducibility contract: the filter must still produce the published series.

If any test in this file fails after a change to :mod:`mekong_da.enkf`, the change altered
the sequence of random draws. Read ``docs/07_pitfalls.md`` before deciding the test is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG
from mekong_da.enkf import run_enkf, run_enkf_frame

#: Machine epsilon on a value of order 0.1. Anything larger is a real change.
TOLERANCE = 1e-12

#: Root-mean-square-error reduction per resolution, as published, in per cent.
PUBLISHED_REDUCTION = {"9km": 58.9, "1km": 60.5, "400m": 60.6}

#: Number of assimilated days per resolution, as reported in Table 1.
PUBLISHED_N = {"9km": 3447, "1km": 3182, "400m": 3065}

#: Lag-1 autocorrelation of the true innovation, measured independently.
PUBLISHED_INNOVATION_AC1 = {"9km": 0.0450, "1km": 0.0666, "400m": 0.1319}


def test_reproduces_archived_analysis(filter_inputs, filter_reference, resolution):
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    diff = np.nanmax(np.abs(out["analysis"].to_numpy() - filter_reference["analysis"].to_numpy()))
    assert diff < TOLERANCE, f"{resolution}: analysis drifted by {diff:.2e}"


def test_reproduces_archived_spread(filter_inputs, filter_reference, resolution):
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    diff = np.nanmax(np.abs(out["uncertainty"].to_numpy() - filter_reference["uncertainty"].to_numpy()))
    assert diff < TOLERANCE, f"{resolution}: ensemble spread drifted by {diff:.2e}"


def test_published_rmse_reduction(filter_inputs, resolution):
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    keep = out["obs_available"].to_numpy(bool)
    obs = out["observation_corrected"].to_numpy()[keep]
    before = np.sqrt(np.mean((out["baseline"].to_numpy()[keep] - obs) ** 2))
    after = np.sqrt(np.mean((out["analysis"].to_numpy()[keep] - obs) ** 2))
    reduction = 100 * (before - after) / before
    assert reduction == pytest.approx(PUBLISHED_REDUCTION[resolution], abs=0.05)


def test_assimilated_day_count(filter_inputs, resolution):
    assert int(filter_inputs["obs_available"].sum()) == PUBLISHED_N[resolution]


def test_true_innovation_diagnostics(filter_inputs, resolution):
    """The recorded innovation is the filter's own, not observation minus open loop."""
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    innovation = out["innovation"].to_numpy()
    innovation = innovation[np.isfinite(innovation)]
    ac1 = np.corrcoef(innovation[:-1], innovation[1:])[0, 1]
    assert ac1 == pytest.approx(PUBLISHED_INNOVATION_AC1[resolution], abs=1e-3)


def test_gain_is_a_probability(filter_inputs):
    """A Kalman gain in this scalar problem cannot leave the unit interval."""
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    gain = out["kalman_gain"].to_numpy()
    gain = gain[np.isfinite(gain)]
    assert gain.min() > 0.0 and gain.max() < 1.0


def test_analysis_respects_physical_bounds(filter_inputs):
    out = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)
    lo, hi = OPTIMAL_CONFIG.sm_bounds
    assert out["analysis"].min() >= lo - 1e-12
    assert out["analysis"].max() <= hi + 1e-12


def test_same_seed_gives_identical_output(filter_inputs):
    a = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)["analysis"].to_numpy()
    b = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)["analysis"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_different_seed_gives_different_output(filter_inputs):
    """Guards against the seed being ignored, which would make the filter look stable."""
    a = run_enkf_frame(filter_inputs, config=OPTIMAL_CONFIG)["analysis"].to_numpy()
    other = EnKFConfig(**{**OPTIMAL_CONFIG.to_dict(), "seed": 7})
    b = run_enkf_frame(filter_inputs, config=other)["analysis"].to_numpy()
    assert not np.allclose(a, b)


def test_no_observations_leaves_the_open_loop_alone():
    """With nothing to assimilate the analysis must track the open loop, not wander."""
    open_loop = np.linspace(0.10, 0.30, 400)
    observation = np.full(400, np.nan)
    result = run_enkf(open_loop, observation, config=OPTIMAL_CONFIG)
    assert result.n_updates == 0
    # AR(1) nudging leaves a small lag, bounded by the process noise.
    assert np.abs(result.analysis[50:] - open_loop[50:]).max() < 0.02


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="aligned"):
        run_enkf(np.zeros(10), np.zeros(9))


def test_open_loop_gaps_raise():
    open_loop = np.array([0.2, np.nan, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        run_enkf(open_loop, np.full(3, np.nan))


def test_vectorised_draw_reproduces_the_loop(data_dir):
    """The protected property is the draw count and order, not the per-member loop.

    NumPy's legacy ``RandomState`` gaussian sampler consumes the stream identically whether
    asked for N values at once or N times for one, so a vectorised update reproduces the
    loop exactly. The loop stays in :mod:`mekong_da.enkf` as the reference form.
    """
    frame = pd.read_csv(data_dir / "enkf" / "inputs_9km.csv")
    open_loop = frame["baseline"].to_numpy(float)
    observation = frame["observation_corrected"].to_numpy(float)
    avail = frame["obs_available"].to_numpy(bool)
    cfg = OPTIMAL_CONFIG
    N, lo, hi = cfg.ensemble_size, cfg.sm_bounds[0], cfg.sm_bounds[1]

    np.random.seed(cfg.seed)
    x = np.clip(open_loop[0] + np.random.normal(0, cfg.model_error_std, N), lo, hi)
    analysis = np.zeros(len(open_loop))
    for t in range(len(open_loop)):
        if t > 0:
            x = cfg.persistence_phi * x + (1 - cfg.persistence_phi) * open_loop[t]
            x = x + np.random.normal(0, cfg.model_error_std, N)
        if avail[t] and np.isfinite(observation[t]):
            x_var = np.var(x) * cfg.inflation_factor
            K = x_var / (x_var + cfg.obs_error_std**2)
            x = x + K * (observation[t] + np.random.normal(0, cfg.obs_error_std, N) - x)
        x = np.clip(x, lo, hi)
        analysis[t] = x.mean()

    loop = run_enkf_frame(frame, config=cfg)["analysis"].to_numpy()
    diff = np.nanmax(np.abs(analysis - loop))
    assert diff < TOLERANCE, f"vectorised update drifted from the loop by {diff:.2e}"
