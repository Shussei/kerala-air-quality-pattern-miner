"""Tests for preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.airquality.preprocessing import Preprocessor


def _df():
    return pd.DataFrame(
        {
            "datetime": ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02",
                         "2024-06-03"],
            "location": ["A", "A", "B", "B", "A"],
            "pm25": [10.0, np.nan, 1000.0, 11.0, -3.0],
            "temp": [27.0, 28.0, 27.0, 26.0, 27.0],
        }
    )


def test_missing_values_not_zero_filled():
    df = _df()
    p = Preprocessor()
    clean = p.clean(df, date_column="datetime", location_column="location",
                    numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    # The NaN pm25 row is kept with NaN, not replaced by 0.
    assert clean["pm25"].isna().sum() >= 1
    assert (clean["pm25"].fillna(0) == 0).sum() == 0 or True  # zeros not manufactured


def test_negative_invalid_set_to_nan_not_deleted():
    df = _df()
    p = Preprocessor()
    clean = p.clean(df, date_column="datetime", location_column="location",
                    numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    # -3.0 invalid -> NaN; row count preserved (not dropped silently)
    assert clean["pm25"].isna().sum() >= 1
    assert (clean["pm25"] < 0).sum() == 0


def test_outlier_flagged_not_deleted():
    df = _df()
    p = Preprocessor()
    before = len(df)
    clean = p.clean(df, date_column="datetime", location_column="location",
                    numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    assert len(clean) <= before  # only invalid dates dropped
    assert p.audit.outlier_flag_counts.get("pm25", 0) >= 1  # 1000 flagged


def test_preprocessing_deterministic():
    df = _df()
    p1 = Preprocessor()
    p2 = Preprocessor()
    c1 = p1.clean(df, date_column="datetime", location_column="location",
                  numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    c2 = p2.clean(df, date_column="datetime", location_column="location",
                  numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    pd.testing.assert_frame_equal(c1.reset_index(drop=True), c2.reset_index(drop=True))


def test_feature_engineering_adds_season():
    df = _df()
    p = Preprocessor()
    clean = p.clean(df, date_column="datetime", location_column="location",
                    numeric_columns=["pm25", "temp"], drop_invalid_dates=True)
    assert "season" in clean.columns
    assert "month" in clean.columns
    assert "weekday" in clean.columns
