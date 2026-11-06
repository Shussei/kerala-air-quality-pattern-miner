"""Multi-source data integration pipeline.

Integrates an air-quality dataset with a weather dataset on a defensible
combination of time and location. Never joins on date alone when locations
differ.

All functions are pure: raw inputs are never mutated. Results are captured in
audit objects suitable for display and download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    CARDINALITY_MANY_TO_MANY,
    CARDINALITY_MANY_TO_ONE,
    CARDINALITY_ONE_TO_MANY,
    CARDINALITY_ONE_TO_ONE,
    DEFAULT_GRAIN,
    DEFAULT_TIMEZONE,
    MAX_COLOCATION_DISTANCE_KM,
)
from .preprocessing import Preprocessor
from .validation import (
    DataQualityReport,
    SchemaValidationResult,
    data_quality_report,
    parse_dates,
    validate_schema,
)


@dataclass
class ColumnMapping:
    """User-supplied mapping from source column names to canonical names.

    ``air_quality`` / ``weather`` map ``source_column -> canonical_column``
    (e.g. ``{"station": "location", "datetime_utc": "datetime"}``).
    """

    air_quality: dict[str, str] = field(default_factory=dict)
    weather: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _source_for(mapping: dict[str, str], canonical: str) -> Optional[str]:
        for src, canon in mapping.items():
            if canon == canonical:
                return src
        return None

    # Source columns that map to canonical 'datetime' / 'location'.
    @property
    def aq_datetime(self) -> Optional[str]:
        return self._source_for(self.air_quality, "datetime")

    @property
    def aq_location(self) -> Optional[str]:
        return self._source_for(self.air_quality, "location")

    @property
    def w_datetime(self) -> Optional[str]:
        return self._source_for(self.weather, "datetime")

    @property
    def w_location(self) -> Optional[str]:
        return self._source_for(self.weather, "location")


@dataclass
class JoinAudit:
    """Full record of the integration join."""

    left_name: str = "air_quality"
    right_name: str = "weather"
    left_rows: int = 0
    right_rows: int = 0
    left_valid_keys: int = 0
    right_valid_keys: int = 0
    left_duplicate_keys: int = 0
    right_duplicate_keys: int = 0
    left_invalid_keys: int = 0
    right_invalid_keys: int = 0
    matched_rows: int = 0
    output_rows: int = 0
    left_unmatched: int = 0
    right_unmatched: int = 0
    cardinality: str = CARDINALITY_MANY_TO_ONE
    row_multiplication: bool = False
    join_keys: list[str] = field(default_factory=list)
    spatial_method: str = "none"
    max_distance_km: Optional[float] = None
    date_coverage_left: Optional[dict] = None
    date_coverage_right: Optional[dict] = None
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"metric": "left_input_rows", "value": self.left_rows},
            {"metric": "right_input_rows", "value": self.right_rows},
            {"metric": "left_valid_keys", "value": self.left_valid_keys},
            {"metric": "right_valid_keys", "value": self.right_valid_keys},
            {"metric": "left_duplicate_keys", "value": self.left_duplicate_keys},
            {"metric": "right_duplicate_keys", "value": self.right_duplicate_keys},
            {"metric": "left_invalid_keys", "value": self.left_invalid_keys},
            {"metric": "right_invalid_keys", "value": self.right_invalid_keys},
            {"metric": "matched_rows", "value": self.matched_rows},
            {"metric": "output_rows", "value": self.output_rows},
            {"metric": "left_unmatched", "value": self.left_unmatched},
            {"metric": "right_unmatched", "value": self.right_unmatched},
            {"metric": "cardinality", "value": self.cardinality},
            {"metric": "row_multiplication", "value": int(self.row_multiplication)},
            {"metric": "spatial_method", "value": self.spatial_method},
        ]
        if self.max_distance_km is not None:
            rows.append({"metric": "max_distance_km", "value": round(self.max_distance_km, 3)})
        return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres.

    Accepts scalars or (same-shaped) NumPy arrays; returns a scalar for scalar
    input and an array for array input.
    """
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a))
    return float(dist) if dist.ndim == 0 else dist


def standardize_frame(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    tz: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Rename columns per mapping and parse the datetime column.

    Returns a copy with canonical columns: datetime (tz-aware), location,
    latitude, longitude where available. Unmapped source columns are kept.
    """
    out = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}).copy()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
        d = out["datetime"].dropna()
        if len(d) and getattr(d.iloc[0], "tzinfo", None) is None:
            out["datetime"] = out["datetime"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        else:
            try:
                out["datetime"] = out["datetime"].dt.tz_convert(tz)
            except TypeError:
                out["datetime"] = out["datetime"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        out["__date__"] = out["datetime"].dt.floor("D")

    for c in ("latitude", "longitude", "location"):
        if c in out.columns:
            out[c] = out[c].astype("string") if c == "location" else pd.to_numeric(out[c], errors="coerce")

    return out


def _temporal_resolution(series: pd.Series) -> str:
    """Heuristic temporal resolution label from a datetime series."""
    s = series.dropna().sort_values()
    if len(s) < 2:
        return "unknown"
    diffs = s.diff().dropna()
    if diffs.empty:
        return "unknown"
    median = diffs.median()
    total_seconds = median.total_seconds()
    if total_seconds <= 60:
        return "sub-hourly"
    if total_seconds <= 3600:
        return "hourly"
    if total_seconds <= 86400 * 1.5:
        return "daily"
    if total_seconds <= 86400 * 35:
        return "weekly"
    return "monthly_or_coarser"


def _detect_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> int:
    if not keys or not all(k in df.columns for k in keys):
        return 0
    return int(df.duplicated(subset=keys, keep=False).sum())


def _infer_unit_conversion(
    source_unit: Optional[str], target_unit: Optional[str]
) -> Optional[str]:
    """Return a conversion description only when source metadata supports it.

    We do not guess conversions from values alone. Known supported pairs:
      mg/m3 -> ug/m3 (x1000) for CO-like gases.
    """
    if not source_unit or not target_unit:
        return None
    su = source_unit.strip().lower()
    tu = target_unit.strip().lower()
    if su == "mg/m^3" and tu == "ug/m^3":
        return "multiply by 1000 (mg/m3 -> ug/m3)"
    if su == "ug/m^3" and tu == "mg/m^3":
        return "divide by 1000 (ug/m3 -> mg/m3)"
    if su == tu:
        return "none (units already match)"
    return None


@dataclass
class IntegrationResult:
    frame: pd.DataFrame
    audit: JoinAudit
    left_clean: pd.DataFrame
    right_clean: pd.DataFrame
    left_raw: pd.DataFrame
    right_raw: pd.DataFrame


class DataIntegrator:
    """Integrates air-quality and weather observations on time + location."""

    def __init__(
        self,
        *,
        tz: str = DEFAULT_TIMEZONE,
        grain: str = DEFAULT_GRAIN,
        max_distance_km: float = MAX_COLOCATION_DISTANCE_KM,
    ):
        self.tz = tz
        self.grain = grain
        self.max_distance_km = max_distance_km

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def integrate(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        mapping: ColumnMapping,
        *,
        left_name: str = "air_quality",
        right_name: str = "weather",
        explicit_location_map: Optional[dict[str, str]] = None,
        left_unit: Optional[str] = None,
        right_unit: Optional[str] = None,
    ) -> IntegrationResult:
        left_raw = left_df.copy()
        right_raw = right_df.copy()

        audit = JoinAudit(left_name=left_name, right_name=right_name)
        audit.left_rows = len(left_raw)
        audit.right_rows = len(right_raw)
        audit.join_keys = ["__date__", "__location_key__"]
        audit.spatial_method = "none"

        messages: list[str] = []
        warnings: list[str] = []

        # --- Validate schemas against mapped canonical columns -------------
        left_req = [mapping.aq_datetime, mapping.aq_location]
        right_req = [mapping.w_datetime, mapping.w_location]
        left_req = [c for c in left_req if c]
        right_req = [c for c in right_req if c]

        left_val = validate_schema(left_df, left_req, name=left_name)
        right_val = validate_schema(right_df, right_req, name=right_name)
        messages.extend(left_val.messages)
        messages.extend(right_val.messages)
        if not left_val.valid or not right_val.valid:
            audit.messages = messages
            audit.warnings = warnings
            return self._empty_result(left_raw, right_raw, audit)

        # --- Standardize + parse ------------------------------------------
        left_clean = standardize_frame(left_df, mapping.air_quality, tz=self.tz)
        right_clean = standardize_frame(right_df, mapping.weather, tz=self.tz)

        # --- Unit conversion (only when source metadata supports it) -------
        conversion = _infer_unit_conversion(left_unit, right_unit)
        if conversion and conversion != "none (units already match)":
            messages.append(f"unit conversion: {conversion}")
        audit.messages = messages

        # --- Temporal resolution -------------------------------------------
        l_res = _temporal_resolution(left_clean["datetime"]) if "datetime" in left_clean.columns else "unknown"
        r_res = _temporal_resolution(right_clean["datetime"]) if "datetime" in right_clean.columns else "unknown"
        messages.append(f"temporal resolution: {left_name}={l_res}, {right_name}={r_res}")
        audit.date_coverage_left = self._coverage(left_clean)
        audit.date_coverage_right = self._coverage(right_clean)

        # --- Build location keys -------------------------------------------
        left_clean = self._build_location_key(left_clean, "location")
        right_clean = self._build_location_key(right_clean, "location")

        # --- Spatial matching (never date-only join) -----------------------
        lat_lon_left = {"latitude": "latitude", "longitude": "longitude"}
        lat_lon_right = {"latitude": "latitude", "longitude": "longitude"}
        have_coords = all(
            c in left_clean.columns for c in lat_lon_left
        ) and all(c in right_clean.columns for c in lat_lon_right)

        if explicit_location_map:
            # User-provided station -> weather-grid mapping.
            audit.spatial_method = "explicit_station_to_grid_mapping"
            left_clean["__grid_key__"] = left_clean[mapping.aq_location].map(explicit_location_map)
            right_clean["__grid_key__"] = right_clean[mapping.w_location].map(
                lambda v: v  # identity; grid key is the location itself
            )
            # Align: left uses mapped grid, right uses its own location id.
            left_clean["__location_key__"] = left_clean["__grid_key__"]
            right_clean["__location_key__"] = right_clean[mapping.w_location]
        elif have_coords:
            audit.spatial_method = "nearest_neighbour_haversine"
            left_clean, right_clean, max_dist = self._spatial_colocate(
                left_clean, right_clean, mapping.aq_location, mapping.w_location
            )
            audit.max_distance_km = max_dist
            messages.append(
                f"spatial colocation: nearest weather grid within {MAX_COLOCATION_DISTANCE_KM} km; "
                f"max observed distance {max_dist:.2f} km" if max_dist is not None else "spatial colocation attempted"
            )
        else:
            warnings.append(
                "No coordinates in either dataset; falling back to exact location-name match on date."
            )
            audit.spatial_method = "exact_location_name_match"

        # --- Duplicate key detection (pre-merge) ---------------------------
        audit.left_duplicate_keys = _detect_duplicate_keys(left_clean, ["__date__", "__location_key__"])
        audit.right_duplicate_keys = _detect_duplicate_keys(right_clean, ["__date__", "__location_key__"])
        if audit.left_duplicate_keys:
            warnings.append(f"{left_name}: {audit.left_duplicate_keys} row(s) have duplicate (date, location) keys")
        if audit.right_duplicate_keys:
            warnings.append(f"{right_name}: {audit.right_duplicate_keys} row(s) have duplicate (date, location) keys")

        audit.left_valid_keys = int(left_clean["__location_key__"].notna().sum())
        audit.right_valid_keys = int(right_clean["__location_key__"].notna().sum())
        audit.left_invalid_keys = int(left_clean["__location_key__"].isna().sum())
        audit.right_invalid_keys = int(right_clean["__location_key__"].isna().sum())

        # --- De-duplicate join keys before merging -------------------------
        # Duplicate (date, location) keys would cause row multiplication.
        # Instead of failing silently, aggregate duplicates to one row per key
        # (mean for numerics, first for categoricals) and record the decision.
        left_clean, left_agg_msg = self._dedupe_on_keys(left_clean)
        right_clean, right_agg_msg = self._dedupe_on_keys(right_clean)
        if left_agg_msg:
            messages.append(left_agg_msg)
        if right_agg_msg:
            messages.append(right_agg_msg)

        # --- Merge ----------------------------------------------------------
        on_cols = ["__date__", "__location_key__"]
        merged = pd.merge(
            left_clean,
            right_clean,
            on=on_cols,
            how="inner",
            suffixes=("_aq", "_wx"),
            validate="1:1",
        )
        after = len(merged)
        audit.output_rows = after
        audit.matched_rows = after
        audit.left_unmatched = len(left_clean) - after
        audit.right_unmatched = len(right_clean) - after

        if audit.left_duplicate_keys or audit.right_duplicate_keys:
            audit.cardinality = CARDINALITY_MANY_TO_ONE
        else:
            audit.cardinality = CARDINALITY_ONE_TO_ONE

        # Row-multiplication check: for a genuine enrichment the output should
        # not exceed the larger input after de-duplication.
        audit.row_multiplication = after > max(audit.left_rows, audit.right_rows) and after > 0
        if audit.row_multiplication:
            warnings.append(
                "Row multiplication detected: output rows exceed both inputs; "
                "review join keys (possible many-to-many match)."
            )

        audit.messages = messages
        audit.warnings = warnings

        # --- Tidy output ----------------------------------------------------
        merged = merged.drop(columns=[c for c in ("__grid_key__",) if c in merged.columns], errors="ignore")
        return IntegrationResult(
            frame=merged,
            audit=audit,
            left_clean=left_clean,
            right_clean=right_clean,
            left_raw=left_raw,
            right_raw=right_raw,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _dedupe_on_keys(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
        """Aggregate duplicate (date, location) keys to one row per key.

        Numeric columns are averaged; other columns take the first value.
        Returns the de-duplicated frame and a message (None if no duplicates).
        """
        keys = ["__date__", "__location_key__"]
        dup_mask = df.duplicated(subset=keys, keep=False)
        n_dups = int(dup_mask.sum())
        if n_dups == 0:
            return df, None
        out = df.copy()
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in keys]
        other_cols = [c for c in out.columns if c not in keys and c not in numeric_cols]
        agg = {c: "mean" for c in numeric_cols}
        agg.update({c: "first" for c in other_cols})
        out = out.groupby(keys, as_index=False).agg(agg)
        msg = (
            f"{n_dups} row(s) had duplicate (date, location) key(s); aggregated "
            f"to {len(out)} unique key(s) to prevent row multiplication."
        )
        return out, msg

    def _coverage(self, df: pd.DataFrame) -> Optional[dict]:
        if "datetime" not in df.columns or df["datetime"].notna().sum() == 0:
            return None
        d = df["datetime"].dropna()
        return {
            "min": str(d.min()),
            "max": str(d.max()),
            "n_observations": int(d.notna().sum()),
            "resolution": _temporal_resolution(d),
        }

    def _build_location_key(self, df: pd.DataFrame, location_col: Optional[str]) -> pd.DataFrame:
        out = df.copy()
        if location_col and location_col in out.columns:
            out["__location_key__"] = out[location_col].astype("string").str.strip()
        else:
            out["__location_key__"] = pd.NA
        return out

    def _spatial_colocate(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_loc_col: Optional[str],
        right_loc_col: Optional[str],
    ):
        """Map each left location to the nearest right location within max distance."""
        left = left.copy()
        right = right.copy()
        left["__grid_key__"] = pd.NA
        max_dist = None

        left_coords = left.dropna(subset=["latitude", "longitude"])
        right_coords = right.dropna(subset=["latitude", "longitude"])

        if left_coords.empty or right_coords.empty:
            return left, right, max_dist

        # Unique locations on each side (avoid O(n*m) over raw rows).
        l_uniq = left_coords.groupby("__location_key__")[["latitude", "longitude"]].first().reset_index()
        r_uniq = right_coords.groupby("__location_key__")[["latitude", "longitude"]].first().reset_index()

        best_right = {}
        best_dist = {}
        for _, lr in l_uniq.iterrows():
            distances = haversine_km(
                lr["latitude"], lr["longitude"],
                r_uniq["latitude"].values, r_uniq["longitude"].values,
            )
            idx = int(np.argmin(distances))
            dist = float(distances[idx])
            if max_dist is None or dist > max_dist:
                max_dist = dist
            if dist <= self.max_distance_km:
                best_right[lr["__location_key__"]] = r_uniq.iloc[idx]["__location_key__"]
                best_dist[lr["__location_key__"]] = dist
            else:
                best_right[lr["__location_key__"]] = pd.NA
                best_dist[lr["__location_key__"]] = dist

        left["__grid_key__"] = left["__location_key__"].map(best_right)
        left["__location_key__"] = left["__grid_key__"]
        # Store distances for audit/inspection.
        left["__distance_km__"] = left["__location_key__"].map(best_dist)
        return left, right, max_dist

    def _empty_result(self, left_raw, right_raw, audit) -> IntegrationResult:
        empty = pd.DataFrame()
        return IntegrationResult(
            frame=empty,
            audit=audit,
            left_clean=left_raw.copy(),
            right_clean=right_raw.copy(),
            left_raw=left_raw.copy(),
            right_raw=right_raw.copy(),
        )


def integration_audit_text(audit: JoinAudit) -> str:
    """Human-readable integration audit summary."""
    lines = [
        "Integration Audit",
        "=================",
        f"Left dataset ({audit.left_name}): {audit.left_rows} rows",
        f"Right dataset ({audit.right_name}): {audit.right_rows} rows",
        f"Join keys: {', '.join(audit.join_keys)}",
        f"Spatial matching method: {audit.spatial_method}",
    ]
    if audit.max_distance_km is not None:
        lines.append(f"Max colocation distance (km): {audit.max_distance_km:.3f}")
    lines += [
        f"Left valid keys: {audit.left_valid_keys}",
        f"Right valid keys: {audit.right_valid_keys}",
        f"Left duplicate keys: {audit.left_duplicate_keys}",
        f"Right duplicate keys: {audit.right_duplicate_keys}",
        f"Left invalid keys: {audit.left_invalid_keys}",
        f"Right invalid keys: {audit.right_invalid_keys}",
        f"Matched rows: {audit.matched_rows}",
        f"Output rows: {audit.output_rows}",
        f"Left unmatched: {audit.left_unmatched}",
        f"Right unmatched: {audit.right_unmatched}",
        f"Cardinality: {audit.cardinality}",
        f"Row multiplication: {audit.row_multiplication}",
    ]
    if audit.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in audit.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
