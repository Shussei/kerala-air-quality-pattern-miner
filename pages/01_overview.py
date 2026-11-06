"""Overview page."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, status_box, metric_card

st.title("\U0001F527 Kerala Air Quality Pattern Miner")
st.caption("Multi-source data integration and pattern discovery in air quality and weather data")

section_header("About the project", "Academic data-mining project demonstrating end-to-end "
    "integration, preprocessing, exploratory analysis, K-Means clustering and Apriori "
    "association-rule mining on real (or user-uploaded) data.")

st.markdown(
    "- **Objective:** Integrate independent air-quality and weather observations for Kerala, "
    "clean and preprocess them, then discover patterns via clustering and association rules.\n"
    "- **Workflow:** Import → Validate → Clean/Preprocess → Integrate (date + location) → EDA → "
    "K-Means → Apriori → Export.\n"
    "- **Stack:** Python, pandas, NumPy, scikit-learn, mlxtend, Streamlit, Plotly, pytest.\n"
    "- **No synthetic results:** every chart, metric, cluster and rule is computed from the data "
    "you load. If no data are loaded, empty states explain what is needed."
)

section_header("Verified data sources", "Documented in DATA_SOURCES.md. Automated fetchers are "
    "optional and credential/network guarded; CSV upload always works.")
src_col1, src_col2, src_col3 = st.columns(3)
with src_col1:
    st.info(
        "**OpenAQ Platform** (air quality)\n"
        "API v3, requires `OPENAQ_API_KEY`. Kerala coverage must be verified per query.\n"
        "Confirmed: requests without a key return HTTP 401."
    )
with src_col2:
    st.info(
        "**NASA POWER** (weather)\n"
        "Global point-based meteorology, no key required public API. Endpoint path re-verification "
        "recommended; CSV upload is the reliable fallback."
    )
with src_col3:
    st.info(
        "**KSPCB** (Kerala air quality)\n"
        "Dashboard / dated AQI PDF widget; no confirmed public bulk API. CSV upload recommended."
    )

section_header("Current dataset coverage", "Metrics below reflect the datasets currently loaded "
    "in the session. Upload CSVs on the Data Import page to populate them.")

aq = st.session_state.get("df_aq_raw")
wx = st.session_state.get("df_wx_raw")
integ = st.session_state.get("df_integrated")

m1, m2, m3, m4 = st.columns(4)
metric_card("Air-quality rows", len(aq) if aq is not None else "—", "raw upload")
metric_card("Weather rows", len(wx) if wx is not None else "—", "raw upload")
metric_card("Integrated rows", len(integ) if integ is not None else "—", "after join")
metric_card("Integrated cols", len(integ.columns) if integ is not None else "—", "")

st.divider()
section_header("Processing status")
status_items = [
    ("Air-quality imported", aq is not None),
    ("Weather imported", wx is not None),
    ("Air-quality cleaned", st.session_state.get("df_aq_clean") is not None),
    ("Weather cleaned", st.session_state.get("df_wx_clean") is not None),
    ("Datasets integrated", integ is not None),
    ("K-Means computed", st.session_state.get("kmeans_result") is not None),
    ("Apriori computed", st.session_state.get("apriori_result") is not None),
]
for label, ok in status_items:
    status_box("success" if ok else "info", f"{'✅' if ok else '⏳'} {label}")

st.divider()
st.subheader("Preview: air-quality")
if aq is not None:
    st.dataframe(aq.head(10), use_container_width=True, hide_index=True)
else:
    st.info("Upload an air-quality CSV on the Data Import page.")
st.subheader("Preview: weather")
if wx is not None:
    st.dataframe(wx.head(10), use_container_width=True, hide_index=True)
else:
    st.info("Upload a weather CSV on the Data Import page.")
