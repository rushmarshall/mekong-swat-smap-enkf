"""Empirical distribution matching for the offline scalar diagnostic.

The rank mapping transfers retrievals to the baseline distribution on the selected
training sample. This aligns distributions but does not establish independent
state accuracy or remove every time-varying bias. The corrected columns of the
worked example closely match a mapping fitted on the pre-2020 training window, with
small residual differences.

The rank and quantile functions remain available for numerical provenance. The
corrected coupled Fortran experiment uses raw screened retrievals with physical
storage-unit conversion; it applies neither of these distribution mappings.

Author: Sebastian R.O. Marshall."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["cdf_match_rank", "cdf_match_quantile"]


def cdf_match_rank(
    obs: np.ndarray,
    model: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    min_samples: int = 10,
) -> np.ndarray:
    """Map observations onto the model distribution by empirical rank.

    Each observation is located in the sorted observation sample, and the resulting
    non-exceedance probability is read back through the sorted model sample. Missing
    observations pass through as NaN.

    Parameters
    ----------
    obs:
        Satellite observations, m3/m3, NaN where absent.
    model:
        Model values over the same period, m3/m3.
    train_mask:
        Optional boolean mask restricting which days build the transfer function. The
        published run fitted the mapping on the training window that ends on
        :data:`mekong_da.tuning.TRAIN_END` (2019-12-31, the close of
        ``config.SM_PERIODS["Training (2015-2019)"]``) and applied it forward through
        2024 without refitting; the example's ``observation_corrected`` column agrees with
        that construction to 2.5e-4 m3/m3 and disagrees with a full-overlap fit by 3e-2.
        Omitting the mask fits on the full overlap, which is not the published
        construction.
    min_samples:
        Below this many valid values in either sample the mapping is not identifiable and
        ``obs`` is returned unchanged.

    Returns
    -------
    numpy.ndarray
        Bias-corrected observations, same length as ``obs``.
    """
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)

    if train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool)
        valid_obs = obs[train_mask & ~np.isnan(obs)]
        valid_model = model[train_mask & ~np.isnan(model)]
    else:
        valid_obs = obs[~np.isnan(obs)]
        valid_model = model[~np.isnan(model)]

    if len(valid_obs) < min_samples or len(valid_model) < min_samples:
        return obs

    obs_sorted = np.sort(valid_obs)
    model_sorted = np.sort(valid_model)
    model_cdf = np.linspace(0, 1, len(model_sorted))

    corrected = np.full_like(obs, np.nan)
    for i, val in enumerate(obs):
        if np.isnan(val):
            continue
        cdf_pos = np.searchsorted(obs_sorted, val) / len(obs_sorted)
        cdf_pos = np.clip(cdf_pos, 0, 1)
        corrected[i] = np.interp(cdf_pos, model_cdf, model_sorted)
    return corrected


def cdf_match_quantile(
    obs: np.ndarray,
    model: np.ndarray,
    n_quantiles: int = 100,
    valid_floor: float = 0.01,
    min_samples: int = 30,
) -> np.ndarray:
    """Map observations onto the model distribution through matched percentiles.

    Both samples are reduced to ``n_quantiles`` percentiles and observations are
    interpolated from the observation percentiles onto the model percentiles. Values at
    or below ``valid_floor`` are treated as missing, because the Fortran model uses a
    small positive sentinel rather than NaN for absent retrievals.

    Parameters
    ----------
    obs, model:
        Observation and model samples, m3/m3.
    n_quantiles:
        Number of percentile knots in the transfer function.
    valid_floor:
        Values at or below this are left untouched and excluded from the fit.
    min_samples:
        Below this many valid values the mapping is not identifiable and ``obs`` is
        returned unchanged.

    Returns
    -------
    numpy.ndarray
        Bias-corrected observations, same length as ``obs``.
    """
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)

    valid_obs = obs[~np.isnan(obs) & (obs > valid_floor)]
    valid_model = model[~np.isnan(model) & (model > valid_floor)]
    if len(valid_obs) < min_samples or len(valid_model) < min_samples:
        return obs

    quantiles = np.linspace(0, 100, n_quantiles)
    obs_q = np.percentile(valid_obs, quantiles)
    model_q = np.percentile(valid_model, quantiles)

    corrected = obs.copy()
    mask = ~np.isnan(obs) & (obs > valid_floor)
    corrected[mask] = np.interp(obs[mask], obs_q, model_q)
    return corrected
