"""Tests for CSV export helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.airquality.reporting import combine_exports


def test_combine_exports_includes_only_non_none():
    exports = combine_exports(cleaned_aq=None, integrated=pd.DataFrame({"a": [1]}))
    assert "cleaned_air_quality.csv" not in exports
    assert "integrated_data.csv" in exports
    assert len(exports) == 1


def test_exports_serialise_to_csv():
    exports = combine_exports(
        rules=pd.DataFrame({"support": [0.5], "confidence": [0.8], "lift": [1.2]}),
        cluster_assignments=pd.DataFrame({"cluster": [0, 1]}),
    )
    assert len(exports) == 2
    for fname, df in exports.items():
        csv = df.to_csv(index=False)
        assert "support" in csv or "cluster" in csv
