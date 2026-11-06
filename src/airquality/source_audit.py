"""Source audit trail: records provenance and limitations for every data source.

Distinguishes verified facts from unresolved questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SourceAudit:
    name: str
    url: str
    retrieval_date_utc: str
    access_method: str
    schema: str = ""
    units: str = ""
    location_coverage: str = "unverified"
    temporal_coverage: str = "unverified"
    license_or_terms: str = "unverified"
    api_requirements: str = "none"
    limitations: str = ""
    verified: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "retrieval_date_utc": self.retrieval_date_utc,
            "access_method": self.access_method,
            "schema": self.schema,
            "units": self.units,
            "location_coverage": self.location_coverage,
            "temporal_coverage": self.temporal_coverage,
            "license_or_terms": self.license_or_terms,
            "api_requirements": self.api_requirements,
            "limitations": self.limitations,
            "verified": self.verified,
            "notes": self.notes,
        }


class SourceAuditLog:
    """Collects source audits and exports them as a DataFrame/CSV."""

    def __init__(self):
        self._sources: list[SourceAudit] = []

    def add(self, audit: SourceAudit) -> None:
        self._sources.append(audit)

    def to_dataframe(self) -> pd.DataFrame:  # type: ignore[name-defined]
        import pandas as pd

        return pd.DataFrame([s.to_dict() for s in self._sources])
