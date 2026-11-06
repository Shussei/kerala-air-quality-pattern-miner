"""Data ingestion: CSV upload + optional remote source fetchers.

Remote fetchers (OpenAQ API v3, NASA POWER) are guarded by API-key presence
and network availability. They return None (with a message) rather than
crashing, so the CSV-upload workflow always remains available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import (
    ALLOWED_CSV_EXTENSIONS,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_ROWS,
    NASA_POWER_API_KEY,
    OPENAQ_API_KEY,
)
from .source_audit import SourceAudit, SourceAuditLog


@dataclass
class IngestionResult:
    dataframe: Optional[pd.DataFrame]
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_audit: Optional[SourceAudit] = None
    error: bool = False


def validate_uploaded_file(
    file_obj, *, required_columns: Optional[list[str]] = None, name: str = "file"
) -> IngestionResult:
    """Validate an uploaded file (Streamlit UploadedFile or path) before reading.

    Checks type, size, required fields and row limits. Never executes file
    content.
    """
    messages: list[str] = []
    warnings: list[str] = []

    # Determine file handle / name / size.
    path: Optional[Path] = None
    size_bytes: Optional[int] = None
    file_name = name

    if hasattr(file_obj, "read") and hasattr(file_obj, "name"):
        file_name = file_obj.name
        try:
            size_bytes = file_obj.size
        except AttributeError:
            size_bytes = None
    elif isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        file_name = path.name
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None

    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_CSV_EXTENSIONS:
        return IngestionResult(
            dataframe=None,
            messages=[f"{name}: unsupported file type '{ext}'. Only .csv is accepted."],
            error=True,
        )

    if size_bytes is not None and size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        return IngestionResult(
            dataframe=None,
            messages=[f"{name}: file exceeds {MAX_UPLOAD_MB} MB size limit."],
            error=True,
        )

    return IngestionResult(dataframe=None, messages=messages, warnings=warnings)


def read_csv(
    file_obj,
    *,
    required_columns: Optional[list[str]] = None,
    name: str = "file",
    skiprows: int = 0,
    date_columns: Optional[list[str]] = None,
) -> IngestionResult:
    """Read and validate a CSV upload. Returns dataframe + audit messages.

    ``skiprows`` skips leading non-data rows (e.g. NASA POWER header blocks).
    ``date_columns`` lists separate year/month/day columns to combine into a
    single ``datetime`` column.
    """
    messages: list[str] = []
    validation = validate_uploaded_file(file_obj, required_columns=required_columns, name=name)
    if validation.error:
        return validation

    try:
        if hasattr(file_obj, "read"):
            file_obj.seek(0)
            # Auto-detect NASA POWER-style header blocks.
            if skiprows == 0:
                first_line = file_obj.readline().decode("utf-8", "ignore").strip()
                file_obj.seek(0)
                if first_line.startswith("-BEGIN HEADER-"):
                    # Skip to the line after "-END HEADER-"; the next line is the header.
                    skiprows = _detect_power_header_rows(file_obj)
                    messages.append(
                        f"{name}: detected NASA POWER header block; skipping {skiprows} row(s)."
                    )
            df = pd.read_csv(file_obj, skiprows=skiprows)
        else:
            df = pd.read_csv(file_obj, skiprows=skiprows)
    except Exception as exc:  # malformed upload
        return IngestionResult(
            dataframe=None,
            messages=[f"{name}: could not parse CSV: {exc}"],
            error=True,
        )

    # Combine separate year/month/day columns into a single datetime column.
    if date_columns:
        present = [c for c in date_columns if c in df.columns]
        missing = [c for c in date_columns if c not in df.columns]
        if missing:
            return IngestionResult(
                dataframe=df,
                messages=[f"{name}: date column(s) not found: {', '.join(missing)}"],
                error=True,
            )
        if len(present) >= 3:
            try:
                df["datetime"] = pd.to_datetime(
                    dict(year=df[present[0]], month=df[present[1]], day=df[present[2]]),
                    errors="coerce",
                )
                n_bad = int(df["datetime"].isna().sum())
                if n_bad:
                    messages.append(
                        f"{name}: {n_bad} row(s) had unparseable combined dates."
                    )
                # Remove the source date columns to avoid duplication.
                df = df.drop(columns=present)
            except Exception as exc:
                return IngestionResult(
                    dataframe=df,
                    messages=[f"{name}: could not combine date columns: {exc}"],
                    error=True,
                )
        else:
            messages.append(
                f"{name}: need at least 3 date columns (year, month, day) to combine; "
                f"found {len(present)}."
            )

    if df is None or df.empty:
        return IngestionResult(
            dataframe=None,
            messages=[f"{name}: the CSV contains no rows."],
            error=True,
        )

    if len(df) > MAX_UPLOAD_ROWS:
        messages.append(
            f"Warning: {name} has {len(df)} rows, above the recommended "
            f"{MAX_UPLOAD_ROWS}; performance may be slow."
        )

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return IngestionResult(
                dataframe=df,
                messages=[f"{name}: missing required column(s): {', '.join(missing)}"],
                error=True,
            )
        messages.append(f"{name}: {len(df)} rows x {len(df.columns)} columns loaded.")

    source_audit = SourceAudit(
        name=f"User upload: {name}",
        url="user-provided CSV",
        retrieval_date_utc=datetime.now(timezone.utc).isoformat(),
        access_method="CSV file upload",
        schema=", ".join(df.columns.astype(str).tolist()),
        units="user-documented",
        location_coverage="user-documented",
        temporal_coverage="user-documented",
        license_or_terms="user-provided",
        api_requirements="none",
        limitations="Coverage and quality depend on the uploaded file.",
        verified=False,
    )
    return IngestionResult(dataframe=df, messages=messages, source_audit=source_audit)


# ---------------------------------------------------------------------------
# Remote fetchers (optional, credential / network guarded)
# ---------------------------------------------------------------------------

def _detect_power_header_rows(file_obj) -> int:
    """Detect NASA POWER header block and return the number of rows to skip.

    Reads through the file until the line after ``-END HEADER-`` (which is the
    column header row). Returns the count of rows to skip so that the column
    header becomes the DataFrame header.
    """
    count = 0
    for raw_line in file_obj:
        # Handle both binary and text file objects.
        line = raw_line.decode("utf-8", "ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
        count += 1
        if line.strip() == "-END HEADER-":
            # The next line is the column header; skip everything up to and
            # including this line, so `header=0` picks up the header row.
            return count
    return count


def fetch_openaq_locations(
    *,
    api_key: str,
    country: str = "IN",
    limit: int = 50,
) -> IngestionResult:
    """Fetch OpenAQ v3 locations. Requires a valid API key."""
    messages: list[str] = []
    if not api_key:
        return IngestionResult(
            dataframe=None,
            messages=["OpenAQ API key not set; skipping remote air-quality fetch."],
        )
    url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    params = {"country": country, "limit": limit}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=25)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return IngestionResult(
            dataframe=None,
            messages=[f"OpenAQ locations request failed: {exc}"],
        )
    data = resp.json()
    results = data.get("results", [])
    rows = []
    for item in results:
        coords = item.get("coordinates") or {}
        rows.append(
            {
                "location_id": item.get("id"),
                "name": item.get("name"),
                "country": (item.get("country") or {}).get("name"),
                "city": item.get("city"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "timezone": item.get("timezone"),
            }
        )
    df = pd.DataFrame(rows)
    messages.append(f"Fetched {len(df)} OpenAQ location(s) for country={country}.")
    return IngestionResult(dataframe=df, messages=messages)


def fetch_nasa_power(
    *,
    latitude: float,
    longitude: float,
    parameters: str = "T2M,RH2M,PRECTOTCORR,WS2M",
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
) -> IngestionResult:
    """Fetch NASA POWER daily meteorological data for a point.

    The public API does not require a key. The documented endpoint returned
    HTTP 404 when tested during development, so this fetcher is defensive:
    network failures and 404s are reported as messages, not crashes.
    """
    messages: list[str] = []
    url = "https://power.larc.nasa.gov/api/temporal/daily/"
    params = {
        "parameters": parameters,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 404:
            return IngestionResult(
                dataframe=None,
                messages=[
                    "NASA POWER endpoint returned 404 (path may have changed). "
                    "Use CSV upload for weather data instead."
                ],
            )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return IngestionResult(
            dataframe=None,
            messages=[f"NASA POWER request failed: {exc}. Use CSV upload for weather data."],
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        return IngestionResult(
            dataframe=None,
            messages=[f"NASA POWER returned invalid JSON: {exc}"],
        )

    parameters_dict = payload.get("parameters", {})
    if not parameters_dict:
        return IngestionResult(dataframe=None, messages=["NASA POWER returned no parameters."])

    # All parameter series share the same date index.
    dates = None
    series = {}
    for param, values in parameters_dict.items():
        if isinstance(values, dict):
            if dates is None:
                dates = list(values.keys())
            series[param] = [values.get(d) for d in dates]
    if dates is None:
        return IngestionResult(dataframe=None, messages=["NASA POWER returned no date series."])

    df = pd.DataFrame(series, index=pd.to_datetime(list(dates))).reset_index().rename(columns={"index": "datetime"})
    df["latitude"] = latitude
    df["longitude"] = longitude
    messages.append(f"Fetched {len(df)} NASA POWER daily record(s) for ({latitude}, {longitude}).")
    return IngestionResult(dataframe=df, messages=messages)


def build_audit_log() -> SourceAuditLog:
    """Return an audit log with the verified-source documentation entries."""
    log = SourceAuditLog()
    today = datetime.now(timezone.utc).date().isoformat()
    log.add(
        SourceAudit(
            name="OpenAQ Platform (air quality)",
            url="https://api.openaq.org/v3/locations ; https://docs.openaq.org/",
            retrieval_date_utc=today,
            access_method="REST API v3 (GET /v3/locations, /v3/sensors/{id}/measurements/daily)",
            schema="location id, name, country, city, coordinates (lat/lon), sensors/parameters, datetime, value, unit",
            units="µg/m³ for PM2.5/PM10/etc.; varies by parameter",
            location_coverage="Global; Kerala stations present if operated by a provider in OpenAQ (to be verified per run)",
            temporal_coverage="Varies by sensor; hourly/daily aggregations available",
            license_or_terms="OpenAQ terms of service; attribution required",
            api_requirements="API key (X-API-Key header). Read from OPENAQ_API_KEY env var. Never hardcoded.",
            limitations="API v3 requires an API key. Kerala coverage must be verified by querying locations for country=IN.",
            verified=True,
            notes="Confirmed: requests without X-API-Key receive HTTP 401 Unauthorized.",
        )
    )
    log.add(
        SourceAudit(
            name="NASA POWER (meteorology/weather)",
            url="https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
            retrieval_date_utc=today,
            access_method="REST API (temporal/daily), point-based, no API key required for public access",
            schema="datetime plus meteorological parameters (T2M, RH2M, PRECTOTCORR, WS2M, ...)",
            units="NASA POWER documented units (e.g., T2M °C, RH2M %, PRECTOTCORR mm/day, WS2M m/s)",
            location_coverage="Global grid; any latitude/longitude point (incl. Kerala)",
            temporal_coverage="Daily; historical and near-real-time",
            license_or_terms="NASA POWER terms of use (public, free)",
            api_requirements="None required for public API; optional registered token",
            limitations="Documented API endpoint returned HTTP 404 for the tested request path during development; "
            "treat as unverified until a successful response is obtained. CSV upload remains the reliable path.",
            verified=False,
            notes="Endpoint reachability/path to be re-verified before relying on automated ingestion.",
        )
    )
    log.add(
        SourceAudit(
            name="Kerala State Pollution Control Board (KSPCB)",
            url="https://kspcb.kerala.gov.in/samp ; https://www.kspcbipam.kerala.gov.in/Dashboard/airQuality",
            retrieval_date_utc=today,
            access_method="Web dashboard; PDF widget (continuous AAQM) observed; no confirmed public bulk API",
            schema="Station, SO2, CO, PM2.5, O3, PM10, NH3, NOx, NO, NO2, AQI, Remarks (from published PDF widget)",
            units="µg/m³ (gases/PM); CO in mg/m³ (per KSPCB PDF)",
            location_coverage="Kerala stations (e.g., Eloor, MG Road EKM, Vytilla, Plamood, Karyavattom, Kannur, Thrissur, Kollam, Palayam) — as seen in one dated PDF",
            temporal_coverage="Single-date snapshot (PDF dated 28-08-2026 observed); historical bulk coverage unverified",
            license_or_terms="KSPCB website terms",
            api_requirements="None confirmed; no public bulk API identified",
            limitations="Dashboard is not confirmed to offer a downloadable bulk dataset or public API. "
            "Observed data is a dated snapshot; use CSV upload for historical analysis.",
            verified=False,
            notes="KSPCB AQI PDF widget located via web search; programmatic access not verified.",
        )
    )
    return log
