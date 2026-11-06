"""Pytest configuration and synthetic test fixtures.

All fixtures are explicitly synthetic and labelled as test data. They are never
used as real project results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Synthetic air-quality fixture (Kerala-style stations)
# --------------------------------------------------------------------------- #
@pytest.fixture
def aq_df():
    return pd.DataFrame(
        {
            "station": ["Eloor", "Eloor", "Plamood", "Plamood", "Kannur", "Kannur"],
            "datetime": [
                "2024-06-01 10:00",
                "2024-06-01 10:00",  # duplicate key (date+station)
                "2024-06-01 10:00",
                "2024-06-02 10:00",
                "2024-06-01 10:00",
                "2024-06-01 10:00",
            ],
            "latitude": [10.0, 10.0, 10.0, 10.0, 11.9, 11.9],
            "longitude": [76.3, 76.3, 76.3, 76.3, 75.4, 75.4],
            "pm25": [20.0, 22.0, 15.0, np.nan, 30.0, -5.0],  # missing + negative invalid
            "no2": [10.0, 12.0, 8.0, 9.0, 14.0, 11.0],
        }
    )


# --------------------------------------------------------------------------- #
# Synthetic weather fixture
# --------------------------------------------------------------------------- #
@pytest.fixture
def wx_df():
    return pd.DataFrame(
        {
            "grid": ["G1", "G1", "G2", "G3"],
            "datetime": [
                "2024-06-01 10:00",
                "2024-06-02 10:00",
                "2024-06-01 10:00",
                "2024-06-01 10:00",
            ],
            "latitude": [10.05, 10.05, 10.1, 12.0],
            "longitude": [76.35, 76.35, 76.3, 75.5],
            "temperature": [27.0, 28.0, 26.5, 29.0],
            "humidity": [80.0, 82.0, 78.0, 70.0],
        }
    )


# --------------------------------------------------------------------------- #
# Clean one-to-one integrable fixture
# --------------------------------------------------------------------------- #
@pytest.fixture
def integrable_pair():
    aq = pd.DataFrame(
        {
            "station": ["A", "B"],
            "datetime": ["2024-01-01", "2024-01-01"],
            "latitude": [10.0, 11.0],
            "longitude": [76.0, 76.5],
            "pm25": [10.0, 20.0],
        }
    )
    wx = pd.DataFrame(
        {
            "grid": ["G1", "G2"],
            "datetime": ["2024-01-01", "2024-01-01"],
            "latitude": [10.05, 11.05],
            "longitude": [76.05, 76.55],
            "temperature": [27.0, 26.0],
        }
    )
    return aq, wx


# --------------------------------------------------------------------------- #
# Fixture with many-to-many keys (row multiplication risk)
# --------------------------------------------------------------------------- #
@pytest.fixture
def many_to_many_pair():
    aq = pd.DataFrame(
        {
            "station": ["A", "A"],
            "datetime": ["2024-01-01", "2024-01-01"],
            "latitude": [10.0, 10.0],
            "longitude": [76.0, 76.0],
            "pm25": [10.0, 12.0],
        }
    )
    wx = pd.DataFrame(
        {
            "grid": ["G1", "G1"],
            "datetime": ["2024-01-01", "2024-01-01"],
            "latitude": [10.05, 10.05],
            "longitude": [76.05, 76.05],
            "temperature": [27.0, 28.0],
        }
    )
    return aq, wx
