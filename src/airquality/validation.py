"""Schema validation, data-dictionary generation, and data-quality profiling.

All functions are pure (input frames are never mutated) and return small
dataclass-style results so they can be unit-tested and reused by the Streamlit
UI without depending on Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_TIMEZONE, IQR_MULTIPLIER, NEGATIVE_INVALID


@dataclass
class SchemaValidationResult:
    valid: bool
    required_columns: list[str]
    missing_columns: list[str]
    present_columns: list[str]
    extra_columns: list[str]
    n_rows: int
    n_columns: int
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "required_columns": self.required_columns,
            "missing_columns": self.missing_columns,
            "present_columns": self.present_columns,
            "extra_columns": self.extra_columns,
            "messages": list(self.messages),
        }


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    n_missing_pct: float
    n_unique: int
    measured_zeros: int
    sample: str


@dataclass
class MissingValueProfile:
    column: str
    n_missing: int
    pct_missing: float
    n_measured_zeros: int


@dataclass
class DuplicateProfile:
    n_duplicate_rows: int
    n_duplicate_key_rows: int
    key_columns: list[str]
    example_keys: list[dict]


@dataclass
class DateParseProfile:
    n_total: int
    n_parsed: int
    n_invalid: int
    invalid_examples: list[str]
    tz_policy: str


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    memory_mb: float
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    missing_summary: list[MissingValueProfile] = field(default_factory=list)
    duplicate_profile: Optional[DuplicateProfile] = None
    date_profile: Optional[DateParseProfile] = None
    outlier_flags: dict[str, int] = field(default_factory=dict)
    invalid_value_flags: dict[str, int] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    name: str = "dataset",
) -> SchemaValidationResult:
    """Validate that ``df`` contains every column in ``required_columns``.

    Returns a result object (never raises). The caller decides how to surface
    messages. Input frame is not modified.
    """
    present = list(df.columns)
    missing = [c for c in required_columns if c not in df.columns]
    extra = [c for c in present if c not in required_columns]
    messages: list[str] = []
    valid = len(missing) == 0

    if not valid:
        messages.append(
            f"{name}: missing required columns: {', '.join(missing)}"
        )
    if extra:
        messages.append(
            f"{name}: {len(extra)} extra column(s) not in required set: "
            f"{', '.join(extra[:10])}"
        )
    messages.append(f"{name}: shape = {df.shape[0]} rows x {df.shape[1]} columns")

    return SchemaValidationResult(
        valid=valid,
        required_columns=list(required_columns),
        missing_columns=missing,
        present_columns=present,
        extra_columns=extra,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        messages=messages,
    )


def data_dictionary(df: pd.DataFrame) -> list[dict]:
    """Return a data dictionary (one row per column) for display/export."""
    rows: list[dict] = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "n_missing": n_missing,
                "pct_missing": round(100.0 * n_missing / max(len(series), 1), 2),
                "n_unique": n_unique,
            }
        )
    return rows


def missing_value_profile(df: pd.DataFrame) -> list[MissingValueProfile]:
    """Profile missing values per column. Missing != measured zero."""
    out: list[MissingValueProfile] = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        if pd.api.types.is_numeric_dtype(series):
            n_zeros = int(((series == 0) & series.notna()).sum())
        else:
            n_zeros = 0
        out.append(
            MissingValueProfile(
                column=col,
                n_missing=n_missing,
                pct_missing=round(100.0 * n_missing / max(len(series), 1), 2),
                n_measured_zeros=n_zeros,
            )
        )
    return out


def duplicate_profile(
    df: pd.DataFrame, key_columns: Optional[list[str]] = None
) -> DuplicateProfile:
    """Profile full-row duplicates and duplicates on a candidate key."""
    n_dup_rows = int(df.duplicated().sum())
    key_columns = [c for c in (key_columns or []) if c in df.columns]
    n_dup_keys = 0
    example_keys: list[dict] = []
    if key_columns:
        dup_mask = df.duplicated(subset=key_columns, keep=False)
        n_dup_keys = int(dup_mask.sum())
        if n_dup_keys:
            example_keys = (
                df.loc[dup_mask, key_columns]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
    return DuplicateProfile(
        n_duplicate_rows=n_dup_rows,
        n_duplicate_key_rows=n_dup_keys,
        key_columns=key_columns,
        example_keys=example_keys,
    )


def parse_dates(
    series: pd.Series,
    *,
    tz: str | None = None,
    dayfirst: bool = False,
) -> DateParseProfile:
    """Parse a date/datetime column with an explicit timezone policy.

    Naive timestamps are localized to ``tz`` (default: config DEFAULT_TIMEZONE).
    Already-aware timestamps are converted to ``tz`` when provided. Invalid
    values become NaT and are counted, not dropped.

    Mixed date formats (e.g. CPCB files with both M/D/YYYY and DD-MM-YY) are
    handled by attempting dayfirst=False first, then dayfirst=True on the
    remaining unparsed values.
    """
    from .config import DEFAULT_TIMEZONE

    tz = tz or DEFAULT_TIMEZONE
    s = series.astype("string")

    # Primary parse with the requested dayfirst setting.
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=False)

    # Fallback: retry unparsed values with the opposite dayfirst setting.
    if parsed.isna().any():
        remaining = s[parsed.isna()]
        if len(remaining):
            parsed_fallback = pd.to_datetime(
                remaining, errors="coerce", dayfirst=not dayfirst, utc=False
            )
            # Align by index: fill NaT positions in parsed with fallback values.
            parsed = parsed.fillna(parsed_fallback)

    n_total = int(len(series))
    n_invalid = int(parsed.isna().sum())
    n_parsed = n_total - n_invalid
    invalid_examples: list[str] = []
    if n_invalid:
        invalid_examples = series[parsed.isna()].head(10).astype(str).tolist()

    tz_policy = "naive_localized"
    if parsed.notna().any():
        first = parsed.dropna().iloc[0]
        if getattr(first, "tzinfo", None) is not None:
            tz_policy = "aware_converted"
            try:
                parsed = parsed.dt.tz_convert(tz)
            except TypeError:
                # Mixed/naive values: localize the naive ones.
                parsed = parsed.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        else:
            parsed = parsed.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")

    return DateParseProfile(
        n_total=n_total,
        n_parsed=n_parsed,
        n_invalid=n_invalid,
        invalid_examples=invalid_examples,
        tz_policy=tz_policy,
    )


def _iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    s = series.dropna()
    if s.empty:
        return pd.Series(False, index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lo = q1 - multiplier * iqr
    hi = q3 + multiplier * iqr
    return series.between(lo, hi) == False  # noqa: E712


def data_quality_report(
    df: pd.DataFrame,
    *,
    key_columns: Optional[list[str]] = None,
    date_column: Optional[str] = None,
    tz: Optional[str] = None,
    numeric_columns: Optional[list[str]] = None,
) -> DataQualityReport:
    """Produce a full data-quality profile for a single dataset."""
    report = DataQualityReport(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        memory_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
    )

    for col in df.columns:
        series = df[col]
        report.column_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                n_missing=int(series.isna().sum()),
                n_missing_pct=round(
                    100.0 * int(series.isna().sum()) / max(len(series), 1), 2
                ),
                n_unique=int(series.nunique(dropna=True)),
                measured_zeros=(
                    int(((series == 0) & series.notna()).sum())
                    if pd.api.types.is_numeric_dtype(series)
                    else 0
                ),
                sample=_sample_string(series),
            )
        )

    report.missing_summary = missing_value_profile(df)
    report.duplicate_profile = duplicate_profile(df, key_columns=key_columns)

    if date_column and date_column in df.columns:
        report.date_profile = parse_dates(df[date_column], tz=tz)

    numeric_columns = numeric_columns or [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_columns:
        series = df[col]
        invalid_neg = 0
        outlier = 0
        if pd.api.types.is_numeric_dtype(series):
            if NEGATIVE_INVALID:
                invalid_neg = int((series < 0).sum())
            outlier = int(_iqr_outlier_mask(series, IQR_MULTIPLIER).sum())
        report.invalid_value_flags[col] = invalid_neg
        report.outlier_flags[col] = outlier

    return report


def _sample_string(series: pd.Series) -> str:
    try:
        nonnull = series.dropna().head(3)
        return ", ".join(str(v) for v in nonnull.tolist())
    except Exception:  # pragma: no cover - defensive
        return ""
