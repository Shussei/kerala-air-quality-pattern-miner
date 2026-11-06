"""Configuration for the Kerala Air Quality Pattern Miner.

All paths use pathlib. Data directories default to ./data relative to this
project but can be overridden through the DATA_DIR environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
"""Project root (the folder containing this src/ package)."""

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time / timezone policy
# ---------------------------------------------------------------------------

# All timestamps parsed by the pipeline are localized to this IANA zone when no
# timezone is present in the source. Kerala, India uses IST (Asia/Kolkata,
# UTC+05:30) year-round with no daylight saving time.
DEFAULT_TIMEZONE = os.environ.get("DEFAULT_TIMEZONE", "Asia/Kolkata")

# Default common time grain used for integration. Only applied when the source
# data actually support it (see integration audit).
DEFAULT_GRAIN = "daily"

# ---------------------------------------------------------------------------
# Kerala season / monsoon classification (documented definition)
# ---------------------------------------------------------------------------
# Kerala, India (IMD-style):
#   Winter        : December - February
#   Pre-monsoon   : March - May
#   Southwest monsoon (rainy): June - September
#   Post-monsoon  : October - November
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",
    6: "Southwest monsoon",
    7: "Southwest monsoon",
    8: "Southwest monsoon",
    9: "Southwest monsoon",
    10: "Post-monsoon",
    11: "Post-monsoon",
}

SEASON_ORDER = ["Winter", "Pre-monsoon", "Southwest monsoon", "Post-monsoon"]

# ---------------------------------------------------------------------------
# Plausibility / outlier thresholds (documented defaults; user-overridable)
# ---------------------------------------------------------------------------

# Physically implausible negative concentrations are flagged as invalid.
NEGATIVE_INVALID = True

# IQR multiplier used for outlier *flagging* (outliers are flagged, never
# silently removed).
IQR_MULTIPLIER = 1.5

# Z-score threshold used for outlier flagging.
Z_SCORE_THRESHOLD = 3.5

# ---------------------------------------------------------------------------
# Integration defaults
# ---------------------------------------------------------------------------

# Maximum distance (km) between an air-quality station and a weather grid point
# for them to be considered co-located when no explicit mapping is provided.
MAX_COLOCATION_DISTANCE_KM = 25.0

# Supported merge cardinality directions.
CARDINALITY_ONE_TO_ONE = "one_to_one"
CARDINALITY_ONE_TO_MANY = "one_to_many"
CARDINALITY_MANY_TO_ONE = "many_to_one"
CARDINALITY_MANY_TO_MANY = "many_to_many"

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

MAX_UPLOAD_ROWS = 2_000_000
MAX_UPLOAD_MB = 200
ALLOWED_CSV_EXTENSIONS = {".csv"}

# ---------------------------------------------------------------------------
# Environment-backed secrets
# ---------------------------------------------------------------------------

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
NASA_POWER_API_KEY = os.environ.get("NASA_POWER_API_KEY", "")
