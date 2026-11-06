"""Tests for the data-integration pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.airquality.integration import (
    DataIntegrator,
    ColumnMapping,
    haversine_km,
    _infer_unit_conversion,
)
from src.airquality.preprocessing import Preprocessor


def test_matched_and_unmatched_counts(integrable_pair):
    aq, wx = integrable_pair
    mapping = ColumnMapping(
        air_quality={"station": "location", "datetime": "datetime",
                     "latitude": "latitude", "longitude": "longitude"},
        weather={"grid": "location", "datetime": "datetime",
                 "latitude": "latitude", "longitude": "longitude"},
    )
    prep = {}
    for name, raw in (("aq", aq), ("wx", wx)):
        p = Preprocessor()
        clean = p.clean(raw, date_column="datetime", location_column="location",
                        numeric_columns=[c for c in raw.columns
                                         if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("datetime", "location")],
                        drop_invalid_dates=True)
        prep[name] = clean
    integrator = DataIntegrator(max_distance_km=20.0)
    result = integrator.integrate(prep["aq"], prep["wx"], mapping, left_name="aq", right_name="wx")
    assert result.audit.matched_rows == 2
    assert result.audit.output_rows == 2
    assert result.audit.left_unmatched == 0
    assert result.audit.right_unmatched == 0


def test_spatial_method_and_distance_recorded(integrable_pair):
    aq, wx = integrable_pair
    mapping = ColumnMapping(
        air_quality={"station": "location", "datetime": "datetime",
                     "latitude": "latitude", "longitude": "longitude"},
        weather={"grid": "location", "datetime": "datetime",
                 "latitude": "latitude", "longitude": "longitude"},
    )
    p = Preprocessor()
    clean_aq = p.clean(aq, date_column="datetime", location_column="location",
                       numeric_columns=["pm25"], drop_invalid_dates=True)
    clean_wx = p.clean(wx, date_column="datetime", location_column="location",
                       numeric_columns=["temperature"], drop_invalid_dates=True)
    integrator = DataIntegrator(max_distance_km=20.0)
    result = integrator.integrate(clean_aq, clean_wx, mapping)
    assert result.audit.spatial_method == "nearest_neighbour_haversine"
    assert result.audit.max_distance_km is not None
    assert result.audit.max_distance_km > 0


def test_haversine_scalar():
    d = haversine_km(10.0, 76.0, 10.05, 76.05)
    assert 5.0 < d < 12.0  # approx 7-8 km


def test_row_multiplication_prevention(many_to_many_pair):
    aq, wx = many_to_many_pair
    mapping = ColumnMapping(
        air_quality={"station": "location", "datetime": "datetime",
                     "latitude": "latitude", "longitude": "longitude"},
        weather={"grid": "location", "datetime": "datetime",
                 "latitude": "latitude", "longitude": "longitude"},
    )
    p = Preprocessor()
    clean_aq = p.clean(aq, date_column="datetime", location_column="location",
                       numeric_columns=["pm25"], drop_invalid_dates=True)
    clean_wx = p.clean(wx, date_column="datetime", location_column="location",
                       numeric_columns=["temperature"], drop_invalid_dates=True)
    integrator = DataIntegrator(max_distance_km=20.0)
    result = integrator.integrate(clean_aq, clean_wx, mapping)
    # Duplicates aggregated first; output must not explode to 4 (2x2).
    assert result.audit.output_rows <= max(len(aq), len(wx))
    assert result.audit.row_multiplication is False


def test_input_data_preserved(integrable_pair):
    aq, wx = integrable_pair
    aq_before = aq.copy()
    wx_before = wx.copy()
    mapping = ColumnMapping(
        air_quality={"station": "location", "datetime": "datetime",
                     "latitude": "latitude", "longitude": "longitude"},
        weather={"grid": "location", "datetime": "datetime",
                 "latitude": "latitude", "longitude": "longitude"},
    )
    p = Preprocessor()
    clean_aq = p.clean(aq, date_column="datetime", location_column="location",
                       numeric_columns=["pm25"], drop_invalid_dates=True)
    clean_wx = p.clean(wx, date_column="datetime", location_column="location",
                       numeric_columns=["temperature"], drop_invalid_dates=True)
    integrator = DataIntegrator(max_distance_km=20.0)
    integrator.integrate(clean_aq, clean_wx, mapping)
    pd.testing.assert_frame_equal(aq, aq_before)
    pd.testing.assert_frame_equal(wx, wx_before)


def test_empty_dataset_integration():
    aq = pd.DataFrame(columns=["station", "datetime", "latitude", "longitude", "pm25"])
    wx = pd.DataFrame(columns=["grid", "datetime", "latitude", "longitude", "temperature"])
    mapping = ColumnMapping(
        air_quality={"station": "location", "datetime": "datetime",
                     "latitude": "latitude", "longitude": "longitude"},
        weather={"grid": "location", "datetime": "datetime",
                 "latitude": "latitude", "longitude": "longitude"},
    )
    p = Preprocessor()
    clean_aq = p.clean(aq, date_column="datetime", location_column="location",
                       numeric_columns=["pm25"], drop_invalid_dates=True)
    clean_wx = p.clean(wx, date_column="datetime", location_column="location",
                       numeric_columns=["temperature"], drop_invalid_dates=True)
    integrator = DataIntegrator(max_distance_km=20.0)
    result = integrator.integrate(clean_aq, clean_wx, mapping)
    assert len(result.frame) == 0


def test_unit_conversion_only_supported_pairs():
    assert _infer_unit_conversion("mg/m^3", "ug/m^3") == "multiply by 1000 (mg/m3 -> ug/m3)"
    assert _infer_unit_conversion("ug/m^3", "mg/m^3") == "divide by 1000 (ug/m3 -> mg/m3)"
    assert _infer_unit_conversion("ug/m^3", "ug/m^3") == "none (units already match)"
    assert _infer_unit_conversion("mg/m^3", "ppb") is None  # unsupported guess
    assert _infer_unit_conversion(None, "ug/m^3") is None
