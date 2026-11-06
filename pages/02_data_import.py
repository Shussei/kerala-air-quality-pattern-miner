"""Data Import page: upload, column mapping, schema validation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, mapping_select, show_dataframe, status_box
from src.airquality.ingestion import read_csv

st.title("\U0001F4E4 Data Import")
st.caption("Upload air-quality and weather CSVs, map their columns, and validate schemas.")

st.info(
    "Column mapping lets you use CSVs with different schemas. Map each dataset's **datetime** "
    "and **location** columns (required) and optionally **latitude**/**longitude**. "
    "Validation messages appear below each dataset."
)

# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
aq_file = st.file_uploader("Air-quality CSV", type=["csv"], key="import_aq_file")
wx_file = st.file_uploader("Weather CSV", type=["csv"], key="import_wx_file")

# --------------------------------------------------------------------------- #
# Air-quality mapping
# --------------------------------------------------------------------------- #
st.subheader("Air-quality dataset")
aq_df_preview = None
if aq_file is not None:
    try:
        aq_file.seek(0)
        skip_aq_preview = st.session_state.get("skip_aq", 0)
        aq_df_preview = pd.read_csv(aq_file, skiprows=skip_aq_preview)
        aq_file.seek(0)
    except Exception as exc:
        st.error(f"Could not read air-quality CSV: {exc}")

if aq_df_preview is not None:
    cols = list(aq_df_preview.columns)
    st.write(f"**{aq_file.name}**: {len(aq_df_preview)} rows x {len(aq_df_preview)} columns")
    c1, c2, c3, c4 = st.columns(4)
    map_aq = {}
    map_aq["datetime"] = mapping_select("datetime (required) *", cols,
                                        st.session_state.get("map_aq_datetime"), key="map_aq_datetime_box", required=True)
    map_aq["location"] = mapping_select("location (required) *", cols,
                                        st.session_state.get("map_aq_location"), key="map_aq_location_box", required=True)
    map_aq["latitude"] = mapping_select("latitude (optional)", cols,
                                        st.session_state.get("map_aq_latitude"), key="map_aq_lat_box")
    map_aq["longitude"] = mapping_select("longitude (optional)", cols,
                                         st.session_state.get("map_aq_lon_box"), key="map_aq_lon_box")

    with st.expander("Advanced options (header rows, combined date columns)", expanded=False):
        skip_aq = st.number_input("Skip leading rows (e.g. NASA POWER header)", min_value=0, value=0, step=1,
                                  key="skip_aq", help="Set to 14 for NASA POWER CSV files with header blocks.")
        st.caption("If your file has separate year/month/day columns instead of one datetime column:")
        dc1, dc2, dc3 = st.columns(3)
        map_aq["date_year"] = dc1.selectbox("Year column", ["-- none --"] + cols,
                                            key="aq_date_year")
        map_aq["date_month"] = dc2.selectbox("Month column", ["-- none --"] + cols,
                                             key="aq_date_month")
        map_aq["date_day"] = dc3.selectbox("Day column", ["-- none --"] + cols,
                                           key="aq_date_day")

    map_aq = {k: v for k, v in map_aq.items() if v and v != "-- none --"}
    st.session_state.map_aq = map_aq
    st.session_state.skip_aq = skip_aq

    # Collect date columns for combination
    date_cols_aq = []
    for key in ("date_year", "date_month", "date_day"):
        v = map_aq.get(key)
        if v:
            date_cols_aq.append(v)
            # Remove from column mapping so it's not treated as a data column
            map_aq.pop(key, None)

    if st.button("Validate & load air-quality", key="btn_validate_aq"):
        req = [map_aq.get("datetime"), map_aq.get("location")]
        req = [r for r in req if r]
        res = read_csv(aq_file, required_columns=req, name=aq_file.name,
                       skiprows=skip_aq, date_columns=date_cols_aq if date_cols_aq else None)
        for m in res.messages:
            if m.lower().startswith("warning"):
                st.warning(m)
            else:
                st.info(m)
        if res.error:
            st.error(res.messages[0] if res.messages else "Invalid air-quality file.")
        else:
            st.session_state.df_aq_raw = res.dataframe
            st.session_state.mapping_aq = map_aq
            st.session_state.df_aq_clean = None
            st.success(f"Air-quality loaded: {len(res.dataframe)} rows x {len(res.dataframe.columns)} columns.")
            if res.source_audit is not None:
                from src.airquality.source_audit import SourceAuditLog
                log = st.session_state.source_log or SourceAuditLog()
                log.add(res.source_audit)
                st.session_state.source_log = log
    st.subheader("Preview (air-quality)")
    st.dataframe(aq_df_preview.head(12), use_container_width=True, hide_index=True)
else:
    st.info("Upload an air-quality CSV to begin mapping.")

# --------------------------------------------------------------------------- #
# Weather mapping
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Weather dataset")
wx_df_preview = None
if wx_file is not None:
    try:
        wx_file.seek(0)
        skip_wx_preview = st.session_state.get("skip_wx", 0)
        wx_df_preview = pd.read_csv(wx_file, skiprows=skip_wx_preview)
        wx_file.seek(0)
    except Exception as exc:
        st.error(f"Could not read weather CSV: {exc}")

if wx_df_preview is not None:
    cols = list(wx_df_preview.columns)
    st.write(f"**{wx_file.name}**: {len(wx_df_preview)} rows x {len(wx_df_preview)} columns")
    c1, c2, c3, c4 = st.columns(4)
    map_wx = {}
    map_wx["datetime"] = mapping_select("datetime (required) *", cols,
                                        st.session_state.get("map_wx_datetime"), key="map_wx_datetime_box", required=True)
    map_wx["location"] = mapping_select("location (required) *", cols,
                                        st.session_state.get("map_wx_location"), key="map_wx_location_box", required=True)
    map_wx["latitude"] = mapping_select("latitude (optional)", cols,
                                        st.session_state.get("map_wx_latitude"), key="map_wx_lat_box")
    map_wx["longitude"] = mapping_select("longitude (optional)", cols,
                                         st.session_state.get("map_wx_lon_box"), key="map_wx_lon_box")

    with st.expander("Advanced options (header rows, combined date columns)", expanded=False):
        skip_wx = st.number_input("Skip leading rows (e.g. NASA POWER header)", min_value=0, value=0, step=1,
                                  key="skip_wx", help="Set to 14 for NASA POWER CSV files with header blocks.")
        st.caption("If your file has separate year/month/day columns instead of one datetime column:")
        dc1, dc2, dc3 = st.columns(3)
        map_wx["date_year"] = dc1.selectbox("Year column", ["-- none --"] + cols,
                                            key="wx_date_year")
        map_wx["date_month"] = dc2.selectbox("Month column", ["-- none --"] + cols,
                                             key="wx_date_month")
        map_wx["date_day"] = dc3.selectbox("Day column", ["-- none --"] + cols,
                                           key="wx_date_day")

    map_wx = {k: v for k, v in map_wx.items() if v and v != "-- none --"}
    st.session_state.map_wx = map_wx
    st.session_state.skip_wx = skip_wx

    date_cols_wx = []
    for key in ("date_year", "date_month", "date_day"):
        v = map_wx.get(key)
        if v:
            date_cols_wx.append(v)
            map_wx.pop(key, None)

    if st.button("Validate & load weather", key="btn_validate_wx"):
        req = [map_wx.get("datetime"), map_wx.get("location")]
        req = [r for r in req if r]
        res = read_csv(wx_file, required_columns=req, name=wx_file.name,
                       skiprows=skip_wx, date_columns=date_cols_wx if date_cols_wx else None)
        for m in res.messages:
            if m.lower().startswith("warning"):
                st.warning(m)
            else:
                st.info(m)
        if res.error:
            st.error(res.messages[0] if res.messages else "Invalid weather file.")
        else:
            st.session_state.df_wx_raw = res.dataframe
            st.session_state.mapping_wx = map_wx
            st.session_state.df_wx_clean = None
            st.success(f"Weather loaded: {len(res.dataframe)} rows x {len(res.dataframe.columns)} columns.")
            if res.source_audit is not None:
                from src.airquality.source_audit import SourceAuditLog
                log = st.session_state.source_log or SourceAuditLog()
                log.add(res.source_audit)
                st.session_state.source_log = log
    st.subheader("Preview (weather)")
    st.dataframe(wx_df_preview.head(12), use_container_width=True, hide_index=True)
else:
    st.info("Upload a weather CSV to begin mapping.")

# --------------------------------------------------------------------------- #
# Currently loaded
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Currently loaded in session")
col1, col2 = st.columns(2)
with col1:
    if st.session_state.df_aq_raw is not None:
        st.success(f"Air-quality: {len(st.session_state.df_aq_raw)} rows")
        show_dataframe(st.session_state.df_aq_raw.head(8), key="dl_aq_raw", download_name="air_quality_raw_preview.csv")
    else:
        st.info("No air-quality dataset loaded.")
with col2:
    if st.session_state.df_wx_raw is not None:
        st.success(f"Weather: {len(st.session_state.df_wx_raw)} rows")
        show_dataframe(st.session_state.df_wx_raw.head(8), key="dl_wx_raw", download_name="weather_raw_preview.csv")
    else:
        st.info("No weather dataset loaded.")
