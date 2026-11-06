"""Tests for K-Means clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.airquality.clustering import (
    compute_kmeans,
    compare_cluster_counts,
    eligible_numeric_features,
)


def _df():
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "pm25": np.concatenate([rng.normal(10, 2, 40), rng.normal(50, 5, 40)]),
            "temp": np.concatenate([rng.normal(27, 1, 40), rng.normal(29, 1, 40)]),
        }
    )


def test_invalid_feature_unknown():
    df = _df()
    res = compute_kmeans(df, ["nonexistent"], n_clusters=2)
    assert res.model is None
    assert any("Unknown feature" in m for m in res.messages)


def test_invalid_feature_non_numeric():
    df = pd.DataFrame({"pm25": [1.0, 2.0], "loc": ["a", "b"]})
    res = compute_kmeans(df, ["pm25", "loc"], n_clusters=2)
    assert res.model is None
    assert any("Non-numeric" in m for m in res.messages)


def test_invalid_cluster_count_too_few():
    df = _df()
    res = compute_kmeans(df, ["pm25", "temp"], n_clusters=1)
    assert res.model is None
    assert any("n_clusters must be" in m for m in res.messages)


def test_invalid_cluster_count_exceeds_samples():
    df = _df().head(3)
    res = compute_kmeans(df, ["pm25", "temp"], n_clusters=5)
    assert res.model is None
    assert any("exceeds" in m for m in res.messages)


def test_clustering_reproducibility():
    df = _df()
    r1 = compute_kmeans(df, ["pm25", "temp"], n_clusters=3, random_state=7)
    r2 = compute_kmeans(df, ["pm25", "temp"], n_clusters=3, random_state=7)
    assert np.array_equal(r1.labels, r2.labels)
    assert r1.inertia == r2.inertia


def test_silhouette_not_for_invalid_k():
    df = _df().head(3)
    res = compute_kmeans(df, ["pm25", "temp"], n_clusters=2)  # n_samples=3, k=2 -> silhouette valid actually
    # Use k == n_samples to force silhouette invalid
    res2 = compute_kmeans(df, ["pm25", "temp"], n_clusters=3)
    assert res2.silhouette is None
    assert any("Silhouette" in m for m in res2.messages)


def test_eligible_features_excludes_constant():
    df = pd.DataFrame({"pm25": [1.0, 1.0, 1.0], "temp": [27.0, 28.0, 27.0], "id": ["a", "b", "c"]})
    feats = eligible_numeric_features(df)
    assert "pm25" not in feats  # constant
    assert "temp" in feats
    assert "id" not in feats


def test_compare_cluster_counts_range():
    df = _df()
    out = compare_cluster_counts(df, ["pm25", "temp"], k_min=2, k_max=4)
    assert list(out["k"]) == [2, 3, 4]
    assert out["inertia"].notna().all()
    assert out["silhouette"].notna().all()
