"""Reporting: build downloadable audit/methodology reports and combine exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .integration import JoinAudit, integration_audit_text
from .preprocessing import PreprocessingAudit
from .source_audit import SourceAuditLog


@dataclass
class MethodologyReport:
    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, text)

    def add_section(self, heading: str, text: str) -> None:
        self.sections.append((heading, text))

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for heading, text in self.sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)


def build_methodology_report(
    *,
    preprocessing_audit: Optional[PreprocessingAudit] = None,
    integration_audit: Optional[JoinAudit] = None,
    source_log=None,
) -> MethodologyReport:
    report = MethodologyReport(title="Kerala Air Quality Pattern Miner — Methodology")

    report.add_section(
        "Workflow",
        "1. Import air-quality and weather CSV files (column mapping supported).\n"
        "2. Validate schemas and profile data quality (missing values, duplicates, invalid dates).\n"
        "3. Preprocess each dataset: explicit timezone handling (Asia/Kolkata), plausibility "
        "checks, outlier flagging (never silent deletion), feature engineering.\n"
        "4. Integrate on date + location (never date alone). Spatial colocation via "
        "nearest-neighbour haversine or explicit station-to-grid mapping.\n"
        "5. Exploratory analysis (Plotly).\n"
        "6. K-Means clustering (scaled features, fixed random_state).\n"
        "7. Apriori frequent-itemset and association-rule mining (documented discretization).\n"
    )

    report.add_section(
        "Timezone policy",
        "All timestamps without an explicit timezone are localized to Asia/Kolkata "
        "(IST, UTC+05:30, no daylight saving). Already-aware timestamps are converted to Asia/Kolkata.",
    )

    report.add_section(
        "Missing-value policy",
        "Missing values are distinguished from measured zeros and are never silently "
        "replaced with zero. For K-Means, rows with missing selected features are excluded "
        "and counted. For Apriori, missing items are treated as absent (documented).",
    )

    report.add_section(
        "Outlier policy",
        "Outliers are flagged using the IQR rule (multiplier 1.5) but are not automatically "
        "deleted. Negative physically-implausible concentrations are treated as invalid.",
    )

    report.add_section(
        "Season definition (Kerala)",
        "Winter: Dec–Feb; Pre-monsoon: Mar–May; Southwest monsoon: Jun–Sep; "
        "Post-monsoon: Oct–Nov.",
    )

    report.add_section(
        "Integration",
        integration_audit_text(integration_audit) if integration_audit else "No integration run yet.",
    )

    report.add_section(
        "Preprocessing decisions",
        preprocessing_audit.to_dataframe().to_markdown(index=False)
        if preprocessing_audit else "No preprocessing run yet.",
    )

    if source_log is not None:
        report.add_section(
            "Data sources and provenance",
            source_log.to_dataframe().to_markdown(index=False),
        )

    report.add_section(
        "Interpretation guidance",
        "K-Means cluster numbers are arbitrary identifiers, not pollution rankings. "
        "Association rules describe co-occurrence, not causation. All results derive "
        "from the loaded data; no synthetic data are used.",
    )

    return report


def combine_exports(
    *,
    cleaned_aq: Optional[pd.DataFrame] = None,
    cleaned_wx: Optional[pd.DataFrame] = None,
    integrated: Optional[pd.DataFrame] = None,
    integration_audit_df: Optional[pd.DataFrame] = None,
    preprocessing_audit_df: Optional[pd.DataFrame] = None,
    cluster_assignments: Optional[pd.DataFrame] = None,
    cluster_summary: Optional[pd.DataFrame] = None,
    itemsets: Optional[pd.DataFrame] = None,
    rules: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """Bundle named dataframes for download. Only non-None frames are included."""
    exports: dict[str, pd.DataFrame] = {}
    if cleaned_aq is not None:
        exports["cleaned_air_quality.csv"] = cleaned_aq
    if cleaned_wx is not None:
        exports["cleaned_weather.csv"] = cleaned_wx
    if integrated is not None:
        exports["integrated_data.csv"] = integrated
    if integration_audit_df is not None:
        exports["integration_audit.csv"] = integration_audit_df
    if preprocessing_audit_df is not None:
        exports["preprocessing_audit.csv"] = preprocessing_audit_df
    if cluster_assignments is not None:
        exports["cluster_assignments.csv"] = cluster_assignments
    if cluster_summary is not None:
        exports["cluster_summary.csv"] = cluster_summary
    if itemsets is not None:
        exports["frequent_itemsets.csv"] = itemsets
    if rules is not None:
        exports["association_rules.csv"] = rules
    return exports
