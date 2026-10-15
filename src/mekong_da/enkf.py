"""Offline scalar Ensemble Kalman Filter for extraction-box SMAP diagnostics.

The scalar state and corrected observation are volumetric soil moisture in m3/m3.
The fixed SWAT baseline anchors the stochastic forecast. These analysis states
do not feed back into the coupled SWAT simulation. The identity operator gives
K = inflation * var(forecast) / (inflation * var(forecast) + obs_error**2).

Random-number order is preserved because it determines the reference trajectories.
Posterior spread and input-agreement scores remain conditional on these settings.
The corrected coupled Fortran model uses a separate state, operator, and control.

Author: Sebastian R.O. Marshall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import EnKFConfig, OPTIMAL_CONFIG

__all__ = ["EnKFResult", "run_enkf", "run_enkf_frame"]


@dataclass
class EnKFResult:
    """Everything the filter produced, per time step.

    Attributes
    ----------
    analysis:
        Posterior ensemble mean, m3/m3. This is the assimilated soil-moisture series.
    spread:
        Posterior ensemble standard deviation (``ddof=0``), m3/m3. Used as the filter's
        own uncertainty estimate.
    forecast_mean:
        Prior (forecast) ensemble mean recorded immediately before each update. NaN on
        days with no observation.
    forecast_spread:
        Prior (forecast) ensemble standard deviation (``ddof=0``) recorded immediately
        before each update, so it is the spread the gain was formed from rather than the
        narrowed posterior spread. NaN on days with no observation.
    innovation:
        The true innovation, ``observation - forecast_mean``. NaN when no update ran.
        This is the quantity a filter diagnostic needs; see the warning in
        ``docs/07_pitfalls.md`` about the differently-defined column of the same name that
        appears in the archived project CSVs.
    kalman_gain:
        The gain actually applied on each update day, in [0, 1]. NaN when no update ran.
    n_updates:
        Count of days on which an update was applied.
    """

    analysis: np.ndarray
    spread: np.ndarray
    forecast_mean: np.ndarray
    forecast_spread: np.ndarray
    innovation: np.ndarray
    kalman_gain: np.ndarray
    n_updates: int


def run_enkf(
    open_loop: np.ndarray,
    observation: np.ndarray,
    obs_available: Optional[np.ndarray] = None,
    config: EnKFConfig = OPTIMAL_CONFIG,
) -> EnKFResult:
    """Run the filter over a daily time series.

    Parameters
    ----------
    open_loop:
        SWAT open-loop (no assimilation) soil moisture, m3/m3, one value per day, no gaps.
        This series both initialises the ensemble and nudges the forecast at every step.
    observation:
        CDF-matched SMAP soil moisture, m3/m3, aligned day-for-day with ``open_loop``.
        Days without a retrieval hold NaN.
    obs_available:
        Optional boolean mask of assimilation days. When omitted, every finite entry of
        ``observation`` is assimilated. Supplying the archived mask reproduces the
        published run exactly, including any day the original screening rejected.
    config:
        Filter settings. Defaults to the tuned configuration.

    Returns
    -------
    EnKFResult
        Posterior mean and spread plus the filter diagnostics.

    Notes
    -----
    The forecast step is an AR(1) blend toward the open loop rather than a re-integration
    of SWAT. That is a deliberate simplification of the soil-moisture experiment: the
    coupled experiment, where the analysis feeds back into the hydrology, is the Fortran
    system in :mod:`mekong_da.fortran`.
    """
    open_loop = np.asarray(open_loop, dtype=float)
    observation = np.asarray(observation, dtype=float)
    n = len(open_loop)
    if len(observation) != n:
        raise ValueError(
            f"observation has {len(observation)} rows but open_loop has {n}; "
            "the two series must be aligned day-for-day before the filter runs"
        )
    if obs_available is None:
        avail = np.isfinite(observation)
    else:
        avail = np.asarray(obs_available, dtype=bool)
        if len(avail) != n:
            raise ValueError("obs_available must be the same length as open_loop")
    if n == 0:
        raise ValueError("cannot run the filter on an empty series")
    if not np.isfinite(open_loop).all():
        raise ValueError(
            "open_loop contains NaN; the forecast step nudges toward it every day, so "
            "gaps must be filled before the filter runs"
        )

    N = config.ensemble_size
    model_error = config.model_error_std
    obs_error = config.obs_error_std
    inflation = config.inflation_factor
    phi = config.persistence_phi
    lo, hi = config.sm_bounds

    analysis = np.zeros(n)
    spread = np.zeros(n)
    forecast_mean = np.full(n, np.nan)
    forecast_spread = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    gain = np.full(n, np.nan)
    n_updates = 0

    # -- DRAW ORDER BEGINS. Every np.random call below is part of the reproducibility
    # -- contract. Do not add, remove or reorder them; a vectorised draw reproduces the
    # -- archive (tests/test_enkf.py), but the loop stays as the reference form.
    np.random.seed(config.seed)
    x = open_loop[0] + np.random.normal(0, model_error, N)
    x = np.clip(x, lo, hi)

    for t in range(n):
        if t > 0:
            # Forecast: AR(1) persistence blended toward the open loop, plus process noise.
            x = phi * x + (1 - phi) * open_loop[t]
            x = x + np.random.normal(0, model_error, N)

        if avail[t] and np.isfinite(observation[t]):
            x_var = np.var(x) * inflation          # ddof=0, matching the published run
            K = x_var / (x_var + obs_error**2)
            forecast_mean[t] = x.mean()            # recorded before the update; draws no RNG
            forecast_spread[t] = x.std()
            innovation[t] = observation[t] - forecast_mean[t]
            gain[t] = K
            n_updates += 1
            for i in range(N):                     # per-member perturbed observation
                x[i] = x[i] + K * (observation[t] + np.random.normal(0, obs_error) - x[i])

        x = np.clip(x, lo, hi)
        analysis[t] = x.mean()
        spread[t] = x.std()
    # -- DRAW ORDER ENDS.

    return EnKFResult(
        analysis=analysis,
        spread=spread,
        forecast_mean=forecast_mean,
        forecast_spread=forecast_spread,
        innovation=innovation,
        kalman_gain=gain,
        n_updates=n_updates,
    )


def run_enkf_frame(
    inputs: pd.DataFrame,
    config: EnKFConfig = OPTIMAL_CONFIG,
) -> pd.DataFrame:
    """Run the filter over an inputs table and return a tidy result frame.

    Parameters
    ----------
    inputs:
        Frame with columns ``date``, ``baseline`` (the SWAT open loop),
        ``observation_corrected`` (CDF-matched SMAP) and ``obs_available``. The optional
        ``observation`` column (raw SMAP, before bias correction) is carried through.
    config:
        Filter settings.

    Returns
    -------
    pandas.DataFrame
        One row per day, carrying the inputs alongside ``analysis``, ``uncertainty``,
        ``forecast_mean``, ``forecast_spread``, ``innovation``, ``kalman_gain`` and
        ``update``.
    """
    required = {"date", "baseline", "observation_corrected", "obs_available"}
    missing = required - set(inputs.columns)
    if missing:
        raise ValueError(f"inputs is missing required column(s): {sorted(missing)}")

    res = run_enkf(
        inputs["baseline"].to_numpy(float),
        inputs["observation_corrected"].to_numpy(float),
        inputs["obs_available"].to_numpy(bool),
        config=config,
    )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(inputs["date"]).dt.strftime("%Y-%m-%d"),
            "baseline": inputs["baseline"].to_numpy(float),
            "observation_corrected": inputs["observation_corrected"].to_numpy(float),
            "obs_available": inputs["obs_available"].to_numpy(bool),
            "analysis": res.analysis,
            "uncertainty": res.spread,
            "forecast_mean": res.forecast_mean,
            "forecast_spread": res.forecast_spread,
            "innovation": res.innovation,
            "kalman_gain": res.kalman_gain,
            "update": res.analysis - inputs["baseline"].to_numpy(float),
        }
    )
    if "observation" in inputs.columns:
        out.insert(2, "observation", inputs["observation"].to_numpy(float))
    return out
