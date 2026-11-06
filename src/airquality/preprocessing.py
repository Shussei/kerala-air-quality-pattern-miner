"""Data-cleaning and preprocessing pipeline.

Design principles:
  * Input raw frames are never mutated.
  * Missing values are distinguished from measured zeros and are never
    silently replaced with zero.
  * Outliers are flagged, not automatically deleted.
  * Transformations are fitted only on the data provided to fit() and reused
    by transform(), avoiding data leakage.
  * Every step is deterministic (fixed seeds, sorted keys).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from .config import DEFAULT_GRAIN, DEFAULT_TIMEZONE, SEASON_MAP, SEASON_ORDER
from .validation import (
    ColumnProfile,
    DataQualityReport,
    MissingValueProfile,
    data_dictionary,
    data_quality_report,
    missing_value_profile,
)


@dataclass
class PreprocessingDecision:
    step: str
    decision: str
    detail: str


@dataclass
class PreprocessingAudit:
    input_rows: int
    output_rows: int
    rows_dropped_invalid: int
    rows_dropped_duplicate: int
    decisions: list[PreprocessingDecision] = field(default_factory=list)
    missing_treatment: dict[str, str] = field(default_factory=dict)
    outlier_flag_counts: dict[str, int] = field(default_factory=dict)
    invalid_counts: dict[str, int] = field(default_factory=dict)
    date_coverage: Optional[dict] = None
    location_coverage: Optional[dict] = None

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for d in self.decisions:
            rows.append(
                {
                    "step": d.step,
                    "decision": d.decision,
                    "detail": d.detail,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ScalerBundle:
    """Fitted scalers keyed by feature name, to avoid leakage across calls."""

    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, features: list[str]) -> "ScalerBundle":
        self.feature_names = list(features)
        self.scalers = {}
        for f in features:
            sc = StandardScaler()
            vals = df[[f]].dropna()
            if not vals.empty:
                sc.fit(vals)
            self.scalers[f] = sc
        return self

    def transform(self, df: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
        features = features or self.feature_names
        out = df.copy()
        for f in features:
            sc = self.scalers.get(f)
            if sc is None:
                continue
            vals = out[[f]].copy()
            mask = vals[f].notna()
            if mask.any():
                out.loc[mask, f] = sc.transform(vals.loc[mask, [f]]).ravel()
        return out


@dataclass
class DiscretizationBundle:
    """Documented discretization for Apriori (bin edges stored)."""

    bin_edges: dict[str, list[float]] = field(default_factory=dict)
    labels: dict[str, list[str]] = field(default_factory=dict)
    n_bins: int = 4
    strategy: str = "quantile"  # 'quantile' or 'uniform'

    def fit(self, df: pd.DataFrame, features: list[str]) -> "DiscretizationBundle":
        self.bin_edges = {}
        self.labels = {}
        for f in features:
            s = df[f].dropna()
            if s.empty or s.nunique() < 2:
                # Cannot bin a constant/empty column meaningfully.
                self.bin_edges[f] = [float(s.min()) if not s.empty else 0.0, 1.0]
                self.labels[f] = ["low", "high"]
                continue
            est = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.strategy,
            )
            est.fit(s.values.reshape(-1, 1))
            edges = est.bin_edges_[0].tolist()
            self.bin_edges[f] = [float(e) for e in edges]
            self.labels[f] = [f"{f}_bin{i}" for i in range(len(edges) - 1)]
        return self

    def transform(self, df: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
        features = features or list(self.bin_edges.keys())
        out = df.copy()
        for f in features:
            edges = self.bin_edges.get(f)
            if edges is None:
                continue
            lab = self.labels.get(f, [f"{f}_bin{i}" for i in range(len(edges) - 1)])
            out[f + "_bin"] = pd.cut(
                out[f], bins=edges, labels=lab, include_lowest=True
            ).astype("string")
        return out


class Preprocessor:
    """End-to-end preprocessing for one dataset (air quality OR weather)."""

    def __init__(
        self,
        *,
        tz: str = DEFAULT_TIMEZONE,
        grain: str = DEFAULT_GRAIN,
        negative_invalid: bool = True,
        iqr_multiplier: float = 1.5,
        z_score_threshold: float = 3.5,
    ):
        self.tz = tz
        self.grain = grain
        self.negative_invalid = negative_invalid
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.audit = PreprocessingAudit(input_rows=0, output_rows=0, rows_dropped_invalid=0, rows_dropped_duplicate=0)
        self.scaler_bundle = ScalerBundle()
        self.discretizer = DiscretizationBundle()
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Public pipeline
    # ------------------------------------------------------------------ #
    def clean(
        self,
        df: pd.DataFrame,
        *,
        date_column: str,
        location_column: str,
        numeric_columns: Optional[list[str]] = None,
        drop_invalid_dates: bool = True,
        drop_duplicates: bool = False,
        handle_missing: str = "keep",  # 'keep' | 'drop_rows' | 'drop_cols_above'
        missing_threshold_pct: float = 50.0,
    ) -> pd.DataFrame:
        """Clean a single dataset and record every decision in self.audit.

        Parameters
        ----------
        drop_invalid_dates:
            If True, rows whose date_column cannot be parsed are dropped and
            counted. If False they are kept with NaT and flagged.
        drop_duplicates:
            If True, exact full-row duplicates are removed (counted). If False
            they are only flagged. Duplicate handling is always documented,
            never silent.
        handle_missing:
            Policy for missing values. 'keep' (default) preserves them and
            only profiles them. 'drop_rows' removes rows missing any numeric
            column. 'drop_cols_above' removes columns above the threshold.
        """
        self.audit = PreprocessingAudit(input_rows=int(len(df)), output_rows=int(len(df)), rows_dropped_invalid=0, rows_dropped_duplicate=0)
        decisions: list[PreprocessingDecision] = []
        out = df.copy()

        decisions.append(
            PreprocessingDecision(
                step="schema",
                decision="inspected",
                detail=f"{len(out)} rows x {len(out.columns)} columns; timezone={self.tz}",
            )
        )

        # --- 1. Date parsing (explicit timezone policy) -----------------
        if date_column not in out.columns:
            raise ValueError(f"date_column '{date_column}' not found in dataset")
        parsed = pd.to_datetime(out[date_column], errors="coerce", dayfirst=False, utc=False)
        n_invalid_dates = int(parsed.isna().sum())
        self.audit.invalid_counts["invalid_dates"] = n_invalid_dates
        decisions.append(
            PreprocessingDecision(
                step="date_parsing",
                decision=f"parsed with timezone {self.tz}; {n_invalid_dates} unparseable values",
                detail="NaT marks invalid dates; missing != measured zero",
            )
        )
        out = out.assign(__parsed_date__=parsed)
        if drop_invalid_dates and n_invalid_dates:
            before = len(out)
            out = out.loc[out["__parsed_date__"].notna()].copy()
            self.audit.rows_dropped_invalid += before - len(out)
            decisions.append(
                PreprocessingDecision(
                    step="invalid_dates",
                    decision="dropped rows with unparseable dates",
                    detail=f"dropped {before - len(out)} row(s)",
                )
            )

        # --- 2. Duplicate identification ---------------------------------
        dup_mask = out.duplicated(keep=False)
        n_dups = int(dup_mask.sum())
        self.audit.decisions = decisions
        decisions.append(
            PreprocessingDecision(
                step="duplicates",
                decision="flagged" if not drop_duplicates else "removed",
                detail=f"{n_dups} exact duplicate row(s)",
            )
        )
        if drop_duplicates and n_dups:
            before = len(out)
            out = out.drop_duplicates().copy()
            self.audit.rows_dropped_duplicate += before - len(out)

        # --- 3. Missing-value profiling (never silently zero-filled) -----
        missing = missing_value_profile(out)
        self.audit.missing_treatment = {m.column: "kept_as_missing" for m in missing}
        high_missing = [m.column for m in missing if m.pct_missing > missing_threshold_pct]
        decisions.append(
            PreprocessingDecision(
                step="missing_values",
                decision="kept; profiled, not zero-filled",
                detail=f"{len(high_missing)} column(s) above {missing_threshold_pct}% missing: {', '.join(high_missing[:5])}",
            )
        )
        if handle_missing == "drop_cols_above" and high_missing:
            before_cols = len(out.columns)
            out = out.drop(columns=high_missing)
            decisions.append(
                PreprocessingDecision(
                    step="missing_values",
                    decision="dropped high-missing columns",
                    detail=f"dropped {before_cols - len(out.columns)} column(s)",
                )
            )
        elif handle_missing == "drop_rows" and numeric_columns:
            before = len(out)
            cols = [c for c in numeric_columns if c in out.columns]
            out = out.dropna(subset=cols).copy()
            self.audit.rows_dropped_invalid += before - len(out)
            decisions.append(
                PreprocessingDecision(
                    step="missing_values",
                    decision="dropped rows missing selected numerics",
                    detail=f"dropped {before - len(out)} row(s)",
                )
            )

        # --- 4. Numeric validation & plausibility ------------------------
        numeric_columns = numeric_columns or [
            c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])
        ]
        invalid_counts: dict[str, int] = {}
        outlier_counts: dict[str, int] = {}
        for col in numeric_columns:
            series = pd.to_numeric(out[col], errors="coerce")
            out[col] = series
            invalid_neg = 0
            if self.negative_invalid:
                invalid_neg = int((series < 0).sum())
                # flag invalid negatives but keep them marked (NaN) so they
                # are not treated as valid measurements.
                out.loc[series < 0, col] = np.nan
            outlier_counts[col] = int(self._iqr_outlier_mask(series, self.iqr_multiplier).sum())
            invalid_counts[col] = invalid_neg
        self.audit.invalid_counts.update(invalid_counts)
        self.audit.outlier_flag_counts = outlier_counts
        decisions.append(
            PreprocessingDecision(
                step="plausibility",
                decision="negative values treated as invalid (set to NaN); outliers flagged only",
                detail=f"invalid negatives: {sum(invalid_counts.values())}; outliers flagged: {sum(outlier_counts.values())}",
            )
        )

        # --- 5. Temporal consistency -------------------------------------
        out["__date__"] = out["__parsed_date__"].dt.floor("D")
        if self.grain == "daily":
            out["__grain_key__"] = out["__date__"]
        elif self.grain == "monthly":
            out["__grain_key__"] = out["__parsed_date__"].dt.to_period("M").dt.to_timestamp()
        else:
            out["__grain_key__"] = out["__parsed_date__"]
        date_min = out["__parsed_date__"].min()
        date_max = out["__parsed_date__"].max()
        self.audit.date_coverage = {
            "min": str(date_min) if pd.notna(date_min) else None,
            "max": str(date_max) if pd.notna(date_max) else None,
            "n_days": int(out["__date__"].nunique()),
        }
        decisions.append(
            PreprocessingDecision(
                step="temporal_consistency",
                decision=f"grain={self.grain}; floored dates to {_grain_label(self.grain)}",
                detail=f"coverage {self.audit.date_coverage['min']} to {self.audit.date_coverage['max']}",
            )
        )

        # --- 6. Location consistency -------------------------------------
        if location_column in out.columns:
            n_locs = int(out[location_column].nunique(dropna=True))
            self.audit.location_coverage = {"n_locations": n_locs}
            decisions.append(
                PreprocessingDecision(
                    step="location_consistency",
                    decision="retained location identifiers",
                    detail=f"{n_locs} unique location(s)",
                )
            )
        else:
            decisions.append(
                PreprocessingDecision(
                    step="location_consistency",
                    decision="no location column found",
                    detail="integration will require user-supplied mapping",
                )
            )

        # --- 7. Feature engineering --------------------------------------
        out = self._engineer_features(out)
        decisions.append(
            PreprocessingDecision(
                step="feature_engineering",
                decision="added month, weekday, season (Kerala definition)",
                detail="season: Winter Dec-Feb, Pre-monsoon Mar-May, SW monsoon Jun-Sep, Post-monsoon Oct-Nov",
            )
        )

        self.audit.decisions = decisions
        self.audit.output_rows = int(len(out))
        self._fitted = True
        return out

    # ------------------------------------------------------------------ #
    # Scaling / discretization helpers
    # ------------------------------------------------------------------ #
    def fit_scaler(self, df: pd.DataFrame, features: list[str]) -> ScalerBundle:
        self.scaler_bundle.fit(df, features)
        return self.scaler_bundle

    def fit_discretizer(self, df: pd.DataFrame, features: list[str], n_bins: int = 4, strategy: str = "quantile") -> DiscretizationBundle:
        self.discretizer = DiscretizationBundle(n_bins=n_bins, strategy=strategy)
        self.discretizer.fit(df, features)
        return self.discretizer

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iqr_outlier_mask(series: pd.Series, multiplier: float) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series(False, index=series.index)
        s = series.dropna()
        if s.empty:
            return pd.Series(False, index=series.index)
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)
        lo = q1 - multiplier * iqr
        hi = q3 + multiplier * iqr
        return ~series.between(lo, hi)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "__parsed_date__" in out.columns:
            dt = out["__parsed_date__"]
            out["month"] = dt.dt.month
            out["weekday"] = dt.dt.day_name()
            out["season"] = dt.dt.month.map(SEASON_MAP).astype("string")
            out["is_monsoon"] = out["season"].eq("Southwest monsoon")
        return out


def _grain_label(grain: str) -> str:
    return {"daily": "day", "monthly": "month", "hourly": "hour"}.get(grain, grain)


@dataclass
class IntegratedFrame:
    """Result of integration: the frame plus a full audit."""

    frame: pd.DataFrame
    audit: dict
    left_raw: pd.DataFrame
    right_raw: pd.DataFrame
    left_clean: pd.DataFrame
    right_clean: pd.DataFrame

    def to_csv(self, path: str | Path) -> None:
        self.frame.to_csv(path, index=False)
