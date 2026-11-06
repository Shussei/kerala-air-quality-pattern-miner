"""Shared Streamlit UI helpers for the Kerala Air Quality Pattern Miner."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the project root is importable when page modules are run by Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st


def section_header(text: str, help_text: Optional[str] = None) -> None:
    st.markdown(f"### {text}")
    if help_text:
        st.caption(help_text)


def show_dataframe(
    df: pd.DataFrame,
    *,
    key: str,
    download_name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> None:
    """Display a dataframe with an optional CSV download button."""
    if df is None or df.empty:
        st.info("No data to display.")
        return
    display_df = df.head(max_rows) if max_rows else df
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    if download_name:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {download_name} (CSV)",
            data=csv,
            file_name=download_name,
            mime="text/csv",
            key=key,
        )


def status_box(kind: str, message: str) -> None:
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "success":
        st.success(message)
    else:
        st.info(message)


def metric_card(label: str, value, help_text: Optional[str] = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def plotly_figure(fig, *, key: str, caption: Optional[str] = None) -> None:
    if fig is None:
        st.info(caption or "No chart to display.")
        return
    st.plotly_chart(fig, use_container_width=True, key=key)
    if caption:
        st.caption(canonical_caption(caption))


def canonical_caption(text: str) -> str:
    return text


def canonical_options(df: pd.DataFrame) -> list[str]:
    return ["-- none --"] + [c for c in df.columns]


def mapping_select(
    label: str,
    columns: list[str],
    current: Optional[str] = None,
    key: str = "mapping",
    help_text: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Select a source column for a canonical field. Returns None if '-- none --'."""
    options = ["-- none --"] + columns
    default_index = 0
    if current and current in columns:
        default_index = options.index(current)
    choice = st.selectbox(
        label,
        options,
        index=default_index,
        key=key,
        help=help_text,
    )
    if choice == "-- none --":
        if required:
            st.error(f"{label} is required.")
        return None
    return choice
