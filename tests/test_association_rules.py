"""Tests for Apriori association-rule mining."""

from __future__ import annotations

import pandas as pd
import pytest

from src.airquality.association_rules import (
    mine_from_discretized,
    run_apriori,
)


def _df():
    return pd.DataFrame(
        {
            "location": ["A", "A", "B", "B", "A", "B"],
            "pm25": [10, 12, 80, 90, 11, 85],
            "temp": [27, 28, 30, 31, 27, 30],
        }
    )


def test_boolean_matrix_validation():
    # run_apriori rejects a genuinely non-boolean matrix directly.
    non_bool = pd.DataFrame({"x": [1.0, 2.0]})
    res = run_apriori(non_bool, min_support=0.5)
    assert res.rules is None or len(res.rules) == 0
    assert any("Boolean" in m for m in res.messages)


def test_no_rules_outcome():
    # Each item appears in only 20% of transactions, so nothing is frequent
    # at min_support=0.99.
    matrix = pd.DataFrame(
        {
            "item_a": [True, False, False, False, False],
            "item_b": [False, True, False, False, False],
            "item_c": [False, False, True, False, False],
        }
    )
    res = run_apriori(matrix, min_support=0.99, min_confidence=0.9)
    assert res.itemsets is None or len(res.itemsets) == 0
    assert res.rules is None or len(res.rules) == 0
    assert any("No frequent itemsets" in m or "No association rules" in m for m in res.messages)


def test_valid_rules_have_metrics():
    df = _df()
    res = mine_from_discretized(df, numeric_features=["pm25", "temp"],
                                categorical_features=["location"],
                                min_support=0.3, min_confidence=0.5, n_bins=3)
    if len(res.rules):
        for col in ("support", "confidence", "lift"):
            assert col in res.rules.columns
        assert (res.rules["support"] >= 0.3).all()
        assert (res.rules["confidence"] >= 0.5).all()


def test_discretization_documented():
    df = _df()
    res = mine_from_discretized(df, numeric_features=["pm25"],
                                categorical_features=["location"],
                                min_support=0.3, min_confidence=0.5, n_bins=3)
    assert res.discretization is not None
    assert "pm25" in res.discretization.bin_edges
    assert len(res.discretization.bin_edges["pm25"]) == 4  # 3 bins -> 4 edges
