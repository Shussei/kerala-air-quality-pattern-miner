"""Data Quality page: profiling and quality audit for loaded datasets."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from ui_helpers import section_header, show_dataframe, status_box, metric_card, plotly_figure
from src.airquality.validation import data_quality_report

st.title("\U0001F9CA Data Quality")
st.caption("Profile missing values, duplicates, invalid dates/values and outlier flags. "
           "Raw inputs are never modified.")

aq = st.session_state.get("df_aq_raw")
wx = st.session_state.get("df_wx_raw")

def render_quality(label, df, key_prefix):
    if df is None:
        st.info(f"{label} not loaded yet (Data Import page).")
        return
    section_header(f"{label} quality profile", f"{len(df)} rows x {len(df.columns)} columns")
    m1, m2, m3, m4 = st.columns(4)
    metric_card("Rows", len(df), key_prefix)
    metric_card("Columns", len(df), key_prefix)
    metric_card("Memory (MB)", df.memory_usage(deep=True).sum() / (1024 * 1024), "")
    total_missing = int(df.isna().sum().sum())
    metric_card("Missing cells", total_missing, "")

    report = data_quality_report(df)

    # Missing values
    st.write("#### Missing values by column (missing ≠ measured zero)")
    miss = pd.DataFrame([{
        "column": m.column, "missing": m.n_missing, "pct_missing": m.pct_missing,
        "measured_zeros": m.n_measured_zeros,
    } for m in report.missing_summary])
    st.dataframe(miss, use_container_width=True, hide_index=True, key=f"{key_prefix}_miss")
    fig_res = None
    # bar via plotly express
    import plotly.express as px
    miss_plot = miss[miss["missing"] > 0]
    if not miss_plot.empty:
        fig = px.bar(miss_plot, x="column", y="missing", title=f"Missing values — {label}",
                     color_discrete_sequence=["#d62728"])
        fig.update_xaxes(tickangle=45)
        plotly_figure(fig, key=f"{key_prefix}_miss_fig", caption="Columns with missing values.")

    # Duplicates
    st.write("#### Duplicates")
    dp = report.duplicate_profile
    c1, c2 = st.columns(2)
    c1.metric("Exact duplicate rows", dp.n_duplicate_rows)
    if dp.key_columns:
        c2.metric(f"Duplicate rows on {', '.join(dp.key_columns)}", dp.n_duplicate_key_rows)
        if dp.example_keys:
            st.dataframe(pd.DataFrame(dp.example_keys), use_container_width=True, hide_index=True,
                         key=f"{key_prefix}_dup_ex")
    status_box("info", "Duplicates are flagged, not silently removed. Handling is decided in Integration & Preprocessing.")

    # Dates
    st.write("#### Date parsing")
    # Find a datetime-like column
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_col = c
            break
    if date_col:
        from src.airquality.validation import parse_dates
        dprof = parse_dates(df[date_col])
        c1, c2, c3 = st.columns(3)
        c1.metric("Parsed", dprof.n_parsed)
        c2.metric("Invalid", dprof.n_invalid)
        c3.metric("Timezone policy", dprof.tz_policy)
        if dprof.invalid_examples:
            st.write("Invalid examples:", dprof.invalid_examples[:5])
    else:
        st.info("No date/time column detected.")

    # Outliers / invalid
    st.write("#### Outlier flags (IQR) and invalid negatives")
    out_rows = []
    for col, n in report.outlier_flags.items():
        out_rows.append({"column": col, "outlier_flags": n, "invalid_negatives": report.invalid_value_flags.get(col, 0)})
    if out_rows:
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True, key=f"{key_prefix}_out")
        status_box("info", "Outliers are flagged only — never automatically deleted. Negative physically-implausible values are treated as invalid.")
    else:
        st.info("No numeric columns to profile for outliers.")

render_quality("Air-quality", aq, "aq")
st.divider()
render_quality("Weather", wx, "wx")
