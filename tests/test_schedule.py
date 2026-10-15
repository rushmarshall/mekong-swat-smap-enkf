"""The observation schedule the filter ran on, checked against the numbers the paper states.

Coverage is the share of the 3,563 record days (2015-04-01 to 2024-12-31) that carry a
screened retrieval, and a gap is the difference in days between consecutive observed dates,
so two adjacent observed days count as a gap of 1. Under that definition the vendored
``obs_available`` masks reproduce the published coverage, mean gap and retained day count at
all three resolutions. On the vendored files ``obs_available`` coincides with
``isfinite(observation)``; the column is carried anyway so the scored day set is stated in
the data rather than inferred from it, and the coincidence is checked here.
"""

import pandas as pd
import pytest

#: (coverage in per cent of record days, mean gap in days, retained days), as published.
PUBLISHED = {"9km": (96.7, 1.034, 3447), "1km": (89.3, 1.119, 3182), "400m": (86.0, 1.162, 3065)}


def test_observation_schedule_matches_the_published_numbers(filter_inputs, resolution):
    available = filter_inputs["obs_available"].to_numpy(bool)
    finite = filter_inputs["observation"].notna().to_numpy(bool)
    disagree = int((available != finite).sum())
    assert disagree == 0, f"{resolution}: obs_available and isfinite(observation) differ on {disagree} days"

    observed = pd.to_datetime(filter_inputs["date"])[available]
    n = len(observed)
    coverage = 100.0 * n / len(filter_inputs)
    mean_gap = float(observed.diff().dt.days.dropna().mean())
    computed = f"{resolution}: coverage={coverage:.3f} per cent, mean gap={mean_gap:.4f} d, n={n}"
    print(computed)
    want_coverage, want_gap, want_n = PUBLISHED[resolution]
    assert n == want_n, computed
    assert coverage == pytest.approx(want_coverage, abs=0.05), computed
    assert mean_gap == pytest.approx(want_gap, abs=0.001), computed
