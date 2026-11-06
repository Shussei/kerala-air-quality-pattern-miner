"""Exports & Methodology page: download results and view methodology."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, show_dataframe, status_box
from src.airquality.reporting import combine_exports, build_methodology_report
from src.airquality.integration import integration_audit_text

st.title("\U0001F4C4 Exports & Methodology")
st.caption("Download computed results and review the methodology, source provenance and limitations.")

# --------------------------------------------------------------------------- #
# Gather available results from session
# --------------------------------------------------------------------------- ---
df_integrated = st.session_state.get("df_integrated")
df_aq_clean = st.session_state.get("df_aq_clean")
df_wx_clean = st.session_state.get("df_wx_clean")
audit_int = st.session_state.get("audit_integration")
audit_prep_aq = st.session_state.get("audit_prep_aq")
audit_prep_wx = st.session_state.get("audit_prep_wx")
km = st.session_state.get("kmeans_result")
ap = st.session_state.get("apriori_result")
source_log = st.session_state.get("source_log")

exports = combine_exports(
    cleaned_aq=df_aq_clean,
    cleaned_wx=df_wx_clean,
    integrated=df_integrated,
    integration_audit_df=audit_int.to_dataframe() if audit_int is not None else None,
    preprocessing_audit_df=None,
    cluster_assignments=km.assignments_dataframe(
        st.session_state.get("df_integrated", pd.DataFrame()).index
    ) if km is not None else None,
    cluster_summary=km.cluster_profiles.reset_index() if km is not None else None,
    itemsets=ap.itemsets if ap is not None else None,
    rules=ap.rules if ap is not None else None,
)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
st.subheader("1. Download results")
if not exports:
    st.info("No results available yet. Run integration, clustering or Apriori first.")
else:
    for fname, dframe in exports.items():
        csv = dframe.to_csv(index=False).encode("utf-8")
        st.download_button(label=f"Download {fname}", data=csv, file_name=fname,
                           mime="text/csv", key=f"exp_{fname}")

# --------------------------------------------------------------------------- #
# Source audit
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("2. Data sources (provenance)")
if source_log is not None:
    show_dataframe(source_log.to_dataframe(), key="src_log_tbl",
                   download_name="source_audit_log.csv")
else:
    from src.airquality.ingestion import build_audit_log
    source_log = build_audit_log()
    st.session_state.source_log = source_log
    show_dataframe(source_log.to_dataframe(), key="src_log_tbl_default",
                   download_name="source_audit_log.csv")
    st.info("Default documented source audit (CSV upload sources are added when you load files).")

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("3. Methodology report")
report = build_methodology_report(
    preprocessing_audit=audit_prep_aq or audit_prep_wx,
    integration_audit=audit_int,
    source_log=source_log,
)
st.markdown(report.to_markdown())
md_bytes = report.to_markdown().encode("utf-8")
st.download_button("Download methodology report (Markdown)", data=md_bytes,
                   file_name="methodology_report.md", mime="text/markdown", key="exp_md")

st.divider()
st.caption(
    "Kerala Air Quality Pattern Miner — academic project. All results are computed from the "
    "loaded data; nothing is fabricated. See README.md and DATA_SOURCES.md for full details."
)
