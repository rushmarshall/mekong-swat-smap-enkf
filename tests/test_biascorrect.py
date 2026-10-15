"""CDF matching: the mapping must move the observation onto the model distribution."""

import numpy as np
import pytest

from mekong_da.biascorrect import cdf_match_quantile, cdf_match_rank


def test_rank_matching_removes_the_offset():
    rng = np.random.default_rng(1)
    model = rng.normal(0.25, 0.05, 2000)
    obs = rng.normal(0.35, 0.05, 2000)          # 0.10 m3/m3 too wet
    corrected = cdf_match_rank(obs, model)
    assert abs(corrected.mean() - model.mean()) < abs(obs.mean() - model.mean()) / 10


def test_rank_matching_preserves_missing_values():
    obs = np.array([0.2, np.nan, 0.3] * 20)
    model = np.linspace(0.1, 0.4, 60)
    corrected = cdf_match_rank(obs, model)
    assert np.isnan(corrected[1])
    assert np.isfinite(corrected[0])


def test_rank_matching_is_monotone():
    """Order must survive the mapping, otherwise wet and dry days would swap."""
    rng = np.random.default_rng(2)
    obs = rng.uniform(0.05, 0.5, 500)
    model = rng.uniform(0.1, 0.4, 500)
    corrected = cdf_match_rank(obs, model)
    order = np.argsort(obs)
    assert np.all(np.diff(corrected[order]) >= -1e-12)


def test_too_few_samples_passes_through_unchanged():
    obs = np.array([0.2, 0.3, 0.4])
    out = cdf_match_rank(obs, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(out, obs)


def test_quantile_matching_ignores_the_missing_sentinel():
    """The Fortran path marks absent retrievals with 0.001, which must not enter the fit."""
    rng = np.random.default_rng(3)
    obs = np.concatenate([rng.normal(0.35, 0.05, 300), np.full(300, 0.001)])
    model = rng.normal(0.25, 0.05, 300)
    corrected = cdf_match_quantile(obs, model)
    assert np.all(corrected[300:] == 0.001)
    assert abs(corrected[:300].mean() - model.mean()) < 0.02
