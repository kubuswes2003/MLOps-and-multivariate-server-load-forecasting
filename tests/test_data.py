"""Tests for the data-loading pipeline."""

import numpy as np
import pandas as pd
import pytest

from server_load.data import RAW_COLUMNS, clean_usage, read_machine_csv, resample_usage


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Five raw rows, unsorted, with invalid disk values and a duplicate timestamp."""
    rows = [
        ["m_1", 20, 50, 60, 10, 3, 30, 30, 101],
        ["m_1", 0, 40, 60, 10, 3, 30, 30, -1],
        ["m_1", 10, 45, 60, 10, 3, 30, 30, 20],
        ["m_1", 10, 47, 60, 10, 3, 30, 30, 22],
        ["m_1", 70, 55, 60, 10, 3, 30, 30, 25],
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_clean_usage_marks_invalid_as_nan(raw_df: pd.DataFrame) -> None:
    cleaned = clean_usage(raw_df)
    assert np.isnan(cleaned.loc[pd.Timedelta(seconds=0), "disk"])
    assert np.isnan(cleaned.loc[pd.Timedelta(seconds=20), "disk"])


def test_clean_usage_sorts_and_deduplicates(raw_df: pd.DataFrame) -> None:
    cleaned = clean_usage(raw_df)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    assert cleaned.loc[pd.Timedelta(seconds=10), "cpu"] == 47  # last duplicate kept


def test_resample_keeps_gaps(raw_df: pd.DataFrame) -> None:
    resampled = resample_usage(clean_usage(raw_df), "30s")
    assert len(resampled) == 3  # bins at 0 s, 30 s, 60 s
    assert resampled["cpu"].isna().sum() == 1  # nothing between 30 s and 60 s


def test_read_machine_csv_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_machine_csv(tmp_path / "missing.csv")
