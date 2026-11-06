"""Exploratory Analysis page: interactive charts from the loaded data."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, plotly_figure, status_box, metric_card
from src.airquality import exploration

st.title("\U0001F4C8 Exploratory Analysis")
st.caption("Every chart is computed from the currently loaded data. Change the dataset/columns below and the charts update.")

# --------------------------------------------------------------------------- #
# Dataset selection
# --------------------------------------------------------------------------- #
integ = st.session_state.get("df_integrated")
aq = st.session_state.get("df_aq_clean")
if aq is None or (hasattr(aq, "__len__") and len(aq) == 0):
    aq = st.session_state.get("df_aq_raw")
wx = st.session_state.get("df_wx_clean")
if wx is None or (hasattr(wx, "__len__") and len(wx) == 0):
    wx = st.session_state.get("df_wx_raw")

# Determine which datasets are available
available = []
if integ is not None and hasattr(integ, "__len__") and len(integ) > 0:
    available.append("Integrated (if available)")
if aq is not None and hasattr(aq, "__len__") and len(aq) > 0:
    available.append("Air-quality")
if wx is not None and hasattr(wx, "__len__") and len(wx) > 0:
    available.append("Weather")

if not available:
    st.warning("No dataset available. Please load data on the Data Import page first.")
    st.stop()

choice = st.selectbox("Dataset to analyse",
                      available,
                      key="eda_dataset")
df = None
if choice == "Integrated (if available)" and integ is not None and len(integ):
    df = integ
elif choice == "Air-quality" and aq is not None and len(aq):
    df = aq
elif choice == "Weather" and wx is not None and len(wx):
    df = wx

# Fallback: if the selected choice somehow has no data, use the first available
if df is None or (hasattr(df, "__len__") and len(df) == 0):
    if "Integrated (if available)" in available and integ is not None and len(integ):
        choice = "Integrated (if available)"
        df = integ
    elif "Air-quality" in available and aq is not None and len(aq):
        choice = "Air-quality"
        df = aq
    elif "Weather" in available and wx is not None and len(wx):
        choice = "Weather"
        df = wx

if df is None or (hasattr(df, "__len__") and len(df) == 0):
    st.warning("No dataset available for the selected choice. Load and integrate data first.")
    st.stop()

st.write(f"Analysing **{choice}** — {len(df)} rows x {len(df.columns)} columns")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

section_header("1. Distribution")
dist_col = st.selectbox("Numeric column", numeric_cols, key="eda_dist_col",
                        help="Histogram of the selected variable.")
res = exploration.distribution_histogram(df, dist_col, title=f"Distribution of {dist_col}")
plotly_figure(res.figure, key="eda_dist_fig", caption=res.message)

# AQI
st.divider()
section_header("2. AQI distribution")
aqi_cols = [c for c in df.columns if c.lower() == "aqi"]
if aqi_cols:
    res = exploration.aqi_distribution(df, aqi_cols[0])
    plotly_figure(res.figure, key="eda_aqi_fig", caption=res.message)
else:
    st.info("No AQI column present in this dataset.")

# Time series
st.divider()
section_header("3. Trend over time")
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
if date_cols and numeric_cols:
    t_date = st.selectbox("Date column", date_cols, key="eda_ts_date")
    t_val = st.selectbox("Value column", numeric_cols, key="eda_ts_val")
    t_loc = st.selectbox("Location column (optional)", ["-- none --"] + [c for c in df.columns if c not in (t_date, t_val)],
                         key="eda_ts_loc")
    loc = None if t_loc == "-- none --" else t_loc
    res = exploration.time_series(df, t_date, t_val, location_column=loc, title=f"{t_val} over time")
    plotly_figure(res.figure, key="eda_ts_fig", caption=res.message)
else:
    st.info("Need a date column and at least one numeric column for a time series.")

# Location comparison
st.divider()
section_header("4. Comparison by location")
loc_cols = [c for c in df.columns if c.lower() in ("location", "station", "grid") or df[c].dtype == object]
if loc_cols and numeric_cols:
    l_loc = st.selectbox("Location column", loc_cols, key="eda_box_loc")
    l_val = st.selectbox("Value column", numeric_cols, key="eda_box_val")
    res = exploration.box_by_location(df, l_val, l_loc, title=f"{l_val} by {l_loc}")
    plotly_figure(res.figure, key="eda_box_fig", caption=res.message)
else:
    st.info("Need a location column and a numeric column.")

# Monthly / seasonal
st.divider()
section_header("5. Monthly & seasonal profile")
if date_cols and numeric_cols:
    m_date = st.selectbox("Date column", date_cols, key="eda_prof_date")
    m_val = st.selectbox("Value column", numeric_cols, key="eda_prof_val")
    res = exploration.monthly_profile(df, m_date, m_val, title=f"Monthly mean {m_val}")
    plotly_figure(res.figure, key="eda_month_fig", caption=res.message)
    res = exploration.seasonal_profile(df, m_date, m_val, title=f"Seasonal mean {m_val}")
    plotly_figure(res.figure, key="eda_season_fig", caption=res.message)
else:
    st.info("Need a date column and a numeric column.")

# Missing values
st.divider()
section_header("6. Missing values")
res = exploration.missing_value_bar(df)
plotly_figure(res.figure, key="eda_missing_fig", caption=res.message)

# Correlation
st.divider()
section_header("7. Correlation analysis")
if len(numeric_cols) >= 2:
    corr_cols = st.multiselect("Numeric variables", numeric_cols, default=numeric_cols[:6], key="eda_corr_cols")
    if corr_cols:
        res = exploration.correlation_heatmap(df, columns=corr_cols)
        plotly_figure(res.figure, key="eda_corr_fig", caption=res.message)
else:
    st.info("Need at least two numeric columns.")

# AQ vs Weather scatter (only meaningful on integrated data)
st.divider()
section_header("8. Air-quality vs weather (paired)")
if choice.startswith("Integrated") and numeric_cols:
    pairs = [(c, c) for c in numeric_cols]
    aq_candidates = [c for c in numeric_cols if any(k in c.lower() for k in ("pm", "no2", "so2", "o3", "co", "nh3", "nox"))]
    wx_candidates = [c for c in numeric_cols if c not in aq_candidates]
    default_aq = aq_candidates[0] if aq_candidates else (numeric_cols[0] if numeric_cols else None)
    default_wx = wx_candidates[0] if wx_candidates else (numeric_cols[1] if len(numeric_cols) > 1 else None)
    s1 = st.selectbox("Air-quality variable", numeric_cols,
                      index=numeric_cols.index(default_aq) if default_aq in numeric_cols else 0, key="eda_sc_aq")
    s2 = st.selectbox("Weather variable", numeric_cols,
                      index=numeric_cols.index(default_wx) if default_wx in numeric_cols else 1, key="eda_sc_wx")
    loc_col = "location" if "location" in df.columns else None
    res = exploration.aq_weather_scatter(df, s1, s2, location_column=loc_col)
    plotly_figure(res.figure, key="eda_sc_fig", caption=res.message)
else:
    st.info("Air-quality vs weather comparison requires the integrated dataset.")
