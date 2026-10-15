"""Shared fixtures. The test suite runs against the data vendored in the repository."""

from pathlib import Path

import pandas as pd
import pytest

from mekong_da.config import DATA_DIR, RESOLUTIONS


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "enkf").is_dir():
        pytest.skip(f"vendored data not found at {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session", params=list(RESOLUTIONS))
def resolution(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def filter_inputs(data_dir, resolution) -> pd.DataFrame:
    return pd.read_csv(data_dir / "enkf" / f"inputs_{resolution}.csv")


@pytest.fixture(scope="session")
def filter_reference(data_dir, resolution) -> pd.DataFrame:
    return pd.read_csv(data_dir / "enkf" / f"reference_{resolution}.csv")
