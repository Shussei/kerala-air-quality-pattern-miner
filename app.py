"""Kerala Air Quality Pattern Miner — Streamlit application entry point.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.airquality import config  # noqa: F401  (ensures dirs exist)

st.set_page_config(
    page_title="Kerala Air Quality Pattern Miner",
    page_icon="\U0001F527",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

DEFAULTS = {
    "df_aq_raw": None,
    "df_wx_raw": None,
    "df_aq_clean": None,
    "df_wx_clean": None,
    "df_integrated": None,
    "mapping_aq": None,
    "mapping_wx": None,
    "audit_prep_aq": None,
    "audit_prep_wx": None,
    "audit_integration": None,
    "integrated_left_clean": None,
    "integrated_right_clean": None,
    "integrated_left_raw": None,
    "integrated_right_raw": None,
    "kmeans_result": None,
    "apriori_result": None,
    "apriori_matrix": None,
    "source_log": None,
    "status_messages": [],
    "auto_loaded": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Auto-load datasets from data/raw/ on startup (no manual upload required)
# ---------------------------------------------------------------------------

def _auto_load_datasets() -> None:
    """Automatically load CPCB air-quality and NASA POWER weather CSVs from
    data/raw/ with correct column mappings. Runs once per session."""
    from pathlib import Path
    from datetime import datetime, timezone
    import pandas as pd

    from src.airquality.ingestion import read_csv
    from src.airquality.source_audit import SourceAudit, SourceAuditLog

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return

    # --- Load CPCB air-quality files ---------------------------------------
    cpcb_files = sorted(raw_dir.glob("cpcb_dly_aq_kerala-*.csv"))
    aq_frames = []
    for f in cpcb_files:
        try:
            res = read_csv(
                f,
                required_columns=["Sampling Date", "Location of Monitoring Station"],
                name=f.name,
            )
            if res.error or res.dataframe is None:
                st.session_state.status_messages.append(
                    ("warning", f"Auto-load skipped {f.name}: "
                     f"{res.messages[0] if res.messages else 'unknown error'}")
                )
                continue
            df = res.dataframe.copy()
            df = df.rename(columns={
                "Sampling Date": "datetime",
                "Location of Monitoring Station": "location",
            })
            # Parse dates (CPCB uses mixed formats: M/D/YYYY and DD-MM-YY)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=False)
            bad = df["datetime"].isna()
            if bad.any():
                df.loc[bad, "datetime"] = pd.to_datetime(
                    df.loc[bad, "datetime"], errors="coerce", dayfirst=True
                )
            df = df.dropna(subset=["datetime"])
            aq_frames.append(df)
            st.session_state.status_messages.append(
                ("success", f"Auto-loaded {f.name}: {len(df)} row(s)")
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state.status_messages.append(
                ("error", f"Auto-load failed for {f.name}: {exc}")
            )

    if aq_frames:
        aq_combined = pd.concat(aq_frames, ignore_index=True)
        keep_cols = [c for c in ("datetime", "location", "SO2", "NO2", "RSPM/PM10", "SPM")
                     if c in aq_combined.columns]
        aq_combined = aq_combined[keep_cols]
        st.session_state.df_aq_raw = aq_combined
        st.session_state.mapping_aq = {"datetime": "datetime", "location": "location"}

        log = st.session_state.source_log or SourceAuditLog()
        log.add(SourceAudit(
            name="CPCB Kerala Air Quality (auto-loaded)",
            url="data/raw/cpcb_dly_aq_kerala-*.csv",
            retrieval_date_utc=datetime.now(timezone.utc).isoformat(),
            access_method="Local CSV file auto-load",
            schema=", ".join(aq_combined.columns.astype(str).tolist()),
            units="µg/m³ (SO2, NO2, RSPM/PM10, SPM)",
            location_coverage="Kerala stations (Eloor, Thiruvananthapuram, Alappuzha, etc.)",
            temporal_coverage="2008, 2009, 2010, 2011, 2015 (intermittent sampling dates)",
            license_or_terms="CPCB / KSPCB public data",
            api_requirements="none",
            limitations="Date format varies; no lat/lon in files; intermittent sampling",
            verified=True,
            notes="Auto-loaded from data/raw/ on startup.",
        ))
        st.session_state.source_log = log

    # --- Load NASA POWER weather file --------------------------------------
    power_files = sorted(raw_dir.glob("POWER_*.csv"))
    for f in power_files:
        try:
            res = read_csv(
                f,
                skiprows=14,
                date_columns=["YEAR", "MO", "DY"],
                name=f.name,
            )
            if res.error or res.dataframe is None:
                st.session_state.status_messages.append(
                    ("warning", f"Auto-load skipped {f.name}: "
                     f"{res.messages[0] if res.messages else 'unknown error'}")
                )
                continue
            df = res.dataframe.copy()
            df["location"] = "Kochi (POWER grid)"
            st.session_state.df_wx_raw = df
            st.session_state.mapping_wx = {"datetime": "datetime", "location": "location"}
            st.session_state.status_messages.append(
                ("success", f"Auto-loaded {f.name}: {len(df)} row(s)")
            )

            log = st.session_state.source_log or SourceAuditLog()
            log.add(SourceAudit(
                name="NASA POWER Weather (auto-loaded)",
                url=str(f),
                retrieval_date_utc=datetime.now(timezone.utc).isoformat(),
                access_method="Local CSV file auto-load",
                schema=", ".join(df.columns.astype(str).tolist()),
                units="T2M °C, RH2M %, PRECTOTCORR mm/day, WS10M m/s",
                location_coverage="Kerala point: 9.9312°N, 76.2673°E",
                temporal_coverage="2019-01-01 to 2025-12-31 daily",
                license_or_terms="NASA POWER terms of use (public, free)",
                api_requirements="none",
                limitations="No temporal overlap with CPCB data (2008-2015)",
                verified=True,
                notes="Auto-loaded from data/raw/ on startup.",
            ))
            st.session_state.source_log = log
        except Exception as exc:  # noqa: BLE001
            st.session_state.status_messages.append(
                ("error", f"Auto-load failed for {f.name}: {exc}")
            )


# Run auto-load once per session (only if nothing is loaded yet)
if (not st.session_state.auto_loaded
        and st.session_state.df_aq_raw is None
        and st.session_state.df_wx_raw is None):
    _auto_load_datasets()
    st.session_state.auto_loaded = True

# ---------------------------------------------------------------------------
# Sidebar: data import + status (available from every page)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("\U0001F527 Kerala AQ Pattern Miner")
    st.caption("Multi-source air quality & weather pattern mining")

    st.divider()
    st.subheader("1. Data import")
    aq_file = st.file_uploader(
        "Upload air-quality CSV",
        type=["csv"],
        key="upl_aq",
        help="CSV with air-quality observations (e.g., station, datetime, PM2.5, ...).",
    )
    wx_file = st.file_uploader(
        "Upload weather CSV",
        type=["csv"],
        key="upl_wx",
        help="CSV with weather observations (e.g., grid/station, datetime, temperature, ...).",
    )

    col_load_aq, col_load_wx = st.columns(2)
    if col_load_aq.button("Load air-quality", key="btn_load_aq"):
        if aq_file is None:
            st.session_state.status_messages.append(("error", "No air-quality file selected."))
        else:
            _load_uploaded(aq_file, "air_quality")
    if col_load_wx.button("Load weather", key="btn_load_wx"):
        if wx_file is None:
            st.session_state.status_messages.append(("error", "No weather file selected."))
        else:
            _load_uploaded(wx_file, "weather")

    st.divider()
    st.subheader("2. Processing status")
    def _flag(label, cond):
        st.write(("✅ " if cond else "⏳ ") + label)

    _flag("Air-quality imported", st.session_state.df_aq_raw is not None)
    _flag("Weather imported", st.session_state.df_wx_raw is not None)
    _flag("Air-quality cleaned", st.session_state.df_aq_clean is not None)
    _flag("Weather cleaned", st.session_state.df_wx_clean is not None)
    _flag("Datasets integrated", st.session_state.df_integrated is not None)

    if st.session_state.status_messages:
        st.divider()
        st.subheader("Messages")
        for kind, msg in st.session_state.status_messages[-8:]:
            if kind == "error":
                st.error(msg)
            elif kind == "warning":
                st.warning(msg)
            else:
                st.info(msg)

    st.divider()
    st.caption("Sources: see DATA_SOURCES.md. OpenAQ requires OPENAQ_API_KEY; CSV upload works offline.")


def _load_uploaded(file_obj, kind: str) -> None:
    from src.airquality.ingestion import read_csv

    result = read_csv(file_obj, name=file_obj.name)
    st.session_state.status_messages.extend(
        [( "warning" if m.lower().startswith("warning") else "info", m) for m in result.messages]
    )
    if result.error:
        st.session_state.status_messages.append(("error", result.messages[0] if result.messages else "Load failed."))
        return
    df = result.dataframe
    if kind == "air_quality":
        st.session_state.df_aq_raw = df
        st.session_state.df_aq_clean = None
        st.session_state.mapping_aq = None
    else:
        st.session_state.df_wx_raw = df
        st.session_state.df_wx_clean = None
        st.session_state.mapping_wx = None
    if result.source_audit is not None:
        from src.airquality.source_audit import SourceAuditLog

        log = st.session_state.source_log or SourceAuditLog()
        log.add(result.source_audit)
        st.session_state.source_log = log
    st.session_state.status_messages.append(
        ("success", f"{kind}: loaded {len(df)} rows x {len(df.columns)} columns.")
    )


# ---------------------------------------------------------------------------
# Page routing (sidebar-based navigation, compatible with all Streamlit versions)
# ---------------------------------------------------------------------------

PAGES = [
    ("\U0001F4CA Overview", "pages/01_overview.py"),
    ("\U0001F4E4 Data Import", "pages/02_data_import.py"),
    ("\U0001F9CA Data Quality", "pages/03_data_quality.py"),
    ("\U0001F50D Integration & Preprocessing", "pages/04_integration.py"),
    ("\U0001F4C8 Exploratory Analysis", "pages/05_eda.py"),
    ("\U0001F4BB K-Means Clustering", "pages/06_kmeans.py"),
    ("\U0001F6E0 Apriori Rules", "pages/07_apriori.py"),
    ("\U0001F4C4 Exports & Methodology", "pages/08_exports.py"),
]

with st.sidebar:
    st.divider()
    st.subheader("Navigation")
    page_labels = [label for label, _ in PAGES]
    selected_label = st.radio("Go to", page_labels, index=0, key="nav_radio")

# Run the selected page module
selected_path = dict(PAGES)[selected_label]
import importlib.util

spec = importlib.util.spec_from_file_location(selected_label, selected_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
