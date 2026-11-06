"""Integration & Preprocessing page: run the cleaning + integration pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, show_dataframe, status_box, metric_card
from src.airquality.preprocessing import Preprocessor
from src.airquality.integration import (
    DataIntegrator, ColumnMapping, integration_audit_text,
)
from src.airquality.config import DEFAULT_GRAIN, DEFAULT_TIMEZONE, MAX_COLOCATION_DISTANCE_KM

st.title("\U0001F50D Integration & Preprocessing")
st.caption("Clean each dataset, then integrate air-quality and weather observations on date + location.")

aq_raw = st.session_state.get("df_aq_raw")
wx_raw = st.session_state.get("df_wx_raw")
map_aq = st.session_state.get("mapping_aq")
map_wx = st.session_state.get("mapping_wx")

if aq_raw is None or wx_raw is None:
    st.warning("Load both air-quality and weather datasets on the Data Import page first.")
    st.stop()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.subheader("1. Configure the integration")
c1, c2, c3 = st.columns(3)
grain = c1.selectbox("Common time grain", ["daily", "monthly"], index=0,
                     help="Only 'daily' is defensible if both sources are daily; otherwise use the coarser supported grain.")
tz = c2.text_input("Timezone", value=DEFAULT_TIMEZONE, help="IANA zone; Kerala = Asia/Kolkata (UTC+05:30, no DST).")
max_dist = c3.number_input("Max colocation distance (km)", min_value=1.0, max_value=500.0,
                           value=float(MAX_COLOCATION_DISTANCE_KM), step=1.0)

st.write("**Column mappings (from Data Import page):**")
st.write("Air-quality:", map_aq or "not set")
st.write("Weather:", map_wx or "not set")

use_explicit = st.checkbox("Use explicit station→weather-grid mapping (instead of nearest-neighbour by coordinates)",
                           value=False)
explicit_map_text = ""
if use_explicit:
    explicit_map_text = st.text_area(
        "Mapping (one per line: aq_station = wx_grid)",
        placeholder="Eloor = G1\nPlamood = G2",
        help="When provided, air-quality stations are mapped to named weather grids by this table. "
             "Coordinates are still recorded where available.",
    )

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("2. Run the pipeline")

if st.button("Clean + integrate datasets", type="primary", key="btn_integrate"):
    # Build mapping objects (canonical target -> source column resolved later)
    cm_aq = ColumnMapping(air_quality=map_aq, weather={})
    cm_wx = ColumnMapping(air_quality={}, weather=map_wx)
    mapping = ColumnMapping(air_quality=map_aq, weather=map_wx)

    # Preprocess each dataset
    prep_results = {}
    for name, raw, mp in (("air_quality", aq_raw, map_aq), ("weather", wx_raw, map_wx)):
        date_col = mp.get("datetime") if mp else None
        loc_col = mp.get("location") if mp else None
        if not date_col or not loc_col:
            st.error(f"{name}: datetime and location must be mapped.")
            st.stop()
        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c not in (date_col, loc_col)]
        p = Preprocessor(tz=tz, grain=grain)
        try:
            clean = p.clean(raw, date_column=date_col, location_column=loc_col,
                            numeric_columns=numeric_cols, drop_invalid_dates=True, drop_duplicates=False)
        except Exception as exc:
            st.error(f"{name} preprocessing failed: {exc}")
            st.stop()
        prep_results[name] = (clean, p.audit)

    st.session_state.df_aq_clean = prep_results["air_quality"][0]
    st.session_state.df_wx_clean = prep_results["weather"][0]
    st.session_state.audit_prep_aq = prep_results["air_quality"][1]
    st.session_state.audit_prep_wx = prep_results["weather"][1]

    # Explicit station->grid map
    explicit_map = None
    if use_explicit and explicit_map_text.strip():
        explicit_map = {}
        for line in explicit_map_text.strip().splitlines():
            if "=" in line:
                a, b = line.split("=", 1)
                explicit_map[a.strip()] = b.strip()

    # Integrate
    integrator = DataIntegrator(tz=tz, grain=grain, max_distance_km=max_dist)
    try:
        result = integrator.integrate(
            prep_results["air_quality"][0],
            prep_results["weather"][0],
            mapping,
            left_name="air_quality",
            right_name="weather",
            explicit_location_map=explicit_map,
        )
    except Exception as exc:
        st.error(f"Integration failed: {exc}")
        st.stop()

    st.session_state.df_integrated = result.frame
    st.session_state.audit_integration = result.audit
    st.session_state.integrated_left_clean = result.left_clean
    st.session_state.integrated_right_clean = result.right_clean
    st.session_state.integrated_left_raw = result.left_raw
    st.session_state.integrated_right_raw = result.right_raw

    st.success("Integration complete.")

# --------------------------------------------------------------------------- #
# Audit display
# --------------------------------------------------------------------------- #
audit = st.session_state.get("audit_integration")
if audit is not None:
    st.divider()
    st.subheader("3. Integration audit")
    st.code(integration_audit_text(audit))
    st.dataframe(audit.to_dataframe(), use_container_width=True, hide_index=True,
                 key="dl_integration_audit_table")
    csv_audit = audit.to_dataframe().to_csv(index=False).encode("utf-8")
    st.download_button("Download integration audit (CSV)", data=csv_audit,
                       file_name="integration_audit.csv", mime="text/csv", key="dl_integration_audit")

    for w in audit.warnings:
        st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Left input rows", audit.left_rows)
    c2.metric("Right input rows", audit.right_rows)
    c3.metric("Matched rows", audit.matched_rows)
    c4.metric("Output rows", audit.output_rows)
    c1.metric("Left unmatched", audit.left_unmatched)
    c2.metric("Right unmatched", audit.right_unmatched)
    c3.metric("Left duplicate keys", audit.left_duplicate_keys)
    c4.metric("Row multiplication", "YES" if audit.row_multiplication else "no")

# --------------------------------------------------------------------------- #
# Preprocessing audits
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("4. Preprocessing decisions")
paq = st.session_state.get("audit_prep_aq")
pwx = st.session_state.get("audit_prep_wx")
if paq:
    st.write("**Air-quality preprocessing**")
    st.dataframe(paq.to_dataframe(), use_container_width=True, hide_index=True, key="dl_prep_aq")
    st.download_button("Download AQ preprocessing audit", data=paq.to_dataframe().to_csv(index=False).encode(),
                       file_name="preprocessing_audit_air_quality.csv", mime="text/csv", key="dl_prep_aq_csv")
if pwx:
    st.write("**Weather preprocessing**")
    st.dataframe(pwx.to_dataframe(), use_container_width=True, hide_index=True, key="dl_prep_wx")
    st.download_button("Download WX preprocessing audit", data=pwx.to_dataframe().to_csv(index=False).encode(),
                       file_name="preprocessing_audit_weather.csv", mime="text/csv", key="dl_prep_wx_csv")

# --------------------------------------------------------------------------- #
# Integrated preview
# --------------------------------------------------------------------------- #
integ = st.session_state.get("df_integrated")
if integ is not None and len(integ):
    st.divider()
    st.subheader("5. Integrated dataset preview")
    st.write(f"{len(integ)} matched rows x {len(integ.columns)} columns")
    show_dataframe(integ.head(20), key="dl_integrated_prev", download_name="integrated_preview.csv")
    csv = integ.to_csv(index=False).encode("utf-8")
    st.download_button("Download integrated dataset (CSV)", data=csv, file_name="integrated_data.csv",
                       mime="text/csv", key="dl_integrated")
