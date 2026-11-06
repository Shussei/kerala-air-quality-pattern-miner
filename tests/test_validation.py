"""Tests for schema validation and data-quality profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.airquality.validation import (
    data_quality_report,
    data_dictionary,
    missing_value_profile,
    duplicate_profile,
    parse_dates,
    validate_schema,
)


def test_validate_schema_valid():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    res = validate_schema(df, ["a", "b"], name="t")
    assert res.valid is True
    assert res.missing_columns == []


def test_validate_schema_missing_required():
    df = pd.DataFrame({"a": [1, 2]})
    res = validate_schema(df, ["a", "b"], name="t")
    assert res.valid is False
    assert res.missing_columns == ["b"]
    assert any("missing required columns" in m for m in res.messages)


def test_data_dictionary_shape():
    df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    d = data_dictionary(df)
    assert len(d) == 2
    assert d[0]["column"] == "a"
    assert d[0]["n_missing"] == 1


def test_missing_values_distinguished_from_zeros():
    df = pd.DataFrame({"pm25": [np.nan, 0.0, 10.0, np.nan, 0.0]})
    prof = missing_value_profile(df)[0]
    assert prof.n_missing == 2
    assert prof.n_measured_zeros == 2  # zeros are NOT counted as missing


def test_invalid_date_values():
    df = pd.DataFrame({"dt": ["2024-01-01", "not-a-date", "2024-01-02"]})
    prof = parse_dates(df["dt"])
    assert prof.n_parsed == 2
    assert prof.n_invalid == 1
    assert "not-a-date" in prof.invalid_examples


def test_duplicate_records():
    df = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    dp = duplicate_profile(df)
    assert dp.n_duplicate_rows == 1  # the repeated (1,'a')


def test_duplicate_join_keys():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                       "loc": ["A", "A", "A"]})
    dp = duplicate_profile(df, key_columns=["date", "loc"])
    assert dp.n_duplicate_key_rows == 2


def test_quality_report_outliers_and_negatives():
    df = pd.DataFrame({"pm25": [-1.0, 10.0, 1000.0, 11.0, 12.0]})
    report = data_quality_report(df, numeric_columns=["pm25"])
    assert report.invalid_value_flags["pm25"] == 1
    assert report.outlier_flags["pm25"] >= 1  # 1000 flagged as outlier


def test_empty_dataset_report():
    df = pd.DataFrame({"a": [], "b": []})
    report = data_quality_report(df)
    assert report.n_rows == 0
    assert report.n_columns == 2
