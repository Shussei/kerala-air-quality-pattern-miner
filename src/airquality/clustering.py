"""K-Means clustering using scikit-learn.

Cluster numbers are arbitrary identifiers, not pollution rankings. Clustering
discovers similarity patterns; it does not establish causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_TIMEZONE


@dataclass
class KMeansResult:
    model: Optional[KMeans]
    scaler: Optional[StandardScaler]
    labels: Optional[np.ndarray]
    scaled_features: Optional[pd.DataFrame]
    inertia: Optional[float]
    silhouette: Optional[float]
    n_clusters: int
    n_samples_used: int
    n_samples_dropped_missing: int
    cluster_sizes: dict[int, int]
    cluster_profiles: pd.DataFrame
    feature_means_raw: pd.DataFrame
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def assignments_dataframe(self, original_index: pd.Index) -> pd.DataFrame:
        """Return a DataFrame aligned to the original input index with a
        'cluster' column (NaN for rows dropped due to missing values)."""
        if self.labels is None:
            return pd.DataFrame({"cluster": pd.NA}, index=original_index)
        used_index = self.scaled_features.index if self.scaled_features is not None else original_index
        assigned = pd.Series(self.labels, index=used_index, name="cluster")
        out = pd.Series(pd.NA, index=original_index, name="cluster", dtype="Int64")
        # Only assign labels for indices that exist in both used_index and
        # original_index (the clustering data may have had rows dropped).
        common = used_index.intersection(original_index)
        if len(common) > 0:
            out.loc[common] = assigned.loc[common]
        return out.to_frame()


def eligible_numeric_features(df: pd.DataFrame, exclude: Optional[list[str]] = None) -> list[str]:
    """Return numeric columns suitable for clustering, excluding identifiers."""
    exclude = set(exclude or {})
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            # Require some variation (non-constant) to be useful.
            if df[c].nunique(dropna=True) > 1:
                cols.append(c)
    return cols


def compute_kmeans(
    df: pd.DataFrame,
    features: list[str],
    *,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    missing_policy: str = "drop_rows",  # 'drop_rows' only; KMeans needs complete cases
    scale: bool = True,
) -> KMeansResult:
    """Run K-Means and return a fully populated result object.

    Missing policy: rows with any missing value in the selected features are
    excluded and counted (never imputed with zero). This is documented in the
    result messages.
    """
    messages: list[str] = []
    warnings: list[str] = []
    exclude = set()

    # --- Validate features -------------------------------------------------
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=0, n_samples_dropped_missing=0,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=[f"Unknown feature(s): {', '.join(missing_feats)}"],
            warnings=warnings,
        )

    non_numeric = [f for f in features if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]
    if non_numeric:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=0, n_samples_dropped_missing=0,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=[f"Non-numeric feature(s): {', '.join(non_numeric)}"],
            warnings=warnings,
        )

    constant = [f for f in features if f in df.columns and df[f].nunique(dropna=True) <= 1]
    if constant:
        warnings.append(f"Constant feature(s) excluded from clustering: {', '.join(constant)}")
        features = [f for f in features if f not in constant]

    if not features:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=0, n_samples_dropped_missing=0,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=["No valid numeric features with variation remain for clustering."],
            warnings=warnings,
        )

    # --- Handle missing values (documented policy: drop, never zero-fill) --
    work = df[features].copy()
    before = len(work)
    work = work.dropna()
    dropped = before - len(work)
    messages.append(
        f"Missing-value policy='{missing_policy}': {dropped} row(s) with missing "
        f"selected feature(s) excluded (not imputed, not zero-filled)."
    )

    n_samples = len(work)
    if n_samples == 0:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=0, n_samples_dropped_missing=dropped,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=["No complete cases remain after excluding missing values."],
            warnings=warnings,
        )

    # --- Validate cluster count -------------------------------------------
    if n_clusters < 2:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=n_samples, n_samples_dropped_missing=dropped,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=["n_clusters must be >= 2."],
            warnings=warnings,
        )
    if n_clusters > n_samples:
        return KMeansResult(
            model=None, scaler=None, labels=None, scaled_features=None,
            inertia=None, silhouette=None, n_clusters=n_clusters,
            n_samples_used=n_samples, n_samples_dropped_missing=dropped,
            cluster_sizes={}, cluster_profiles=pd.DataFrame(),
            feature_means_raw=pd.DataFrame(),
            messages=[f"n_clusters={n_clusters} exceeds n_samples={n_samples}."],
            warnings=warnings,
        )

    # --- Scale (fit only on the data used; no leakage) -------------------
    scaler = StandardScaler() if scale else None
    X = work.values
    if scale:
        X = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X, columns=features, index=work.index)

    # --- Fit K-Means -------------------------------------------------------
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    inertia = float(model.inertia_)

    # --- Silhouette only when mathematically valid -------------------------
    silhouette = None
    if 2 <= n_clusters <= n_samples - 1:
        try:
            silhouette = float(silhouette_score(X, labels))
        except ValueError:
            silhouette = None
            warnings.append("Silhouette score could not be computed.")
    else:
        messages.append(
            f"Silhouette score not reported (requires 2 <= n_clusters <= "
            f"n_samples-1; got {n_clusters} vs {n_samples})."
        )

    # --- Cluster sizes & profiles -----------------------------------------
    work2 = work.copy()
    work2["cluster"] = labels
    sizes = work2["cluster"].value_counts().sort_index().to_dict()
    cluster_sizes = {int(k): int(v) for k, v in sizes.items()}

    # Raw-feature profile per cluster (interpretable, in original units).
    profiles = work2.groupby("cluster")[features].mean().round(3)
    profiles.index = profiles.index.astype(int)
    feature_means_raw = work[features].mean().round(3).to_frame(name="overall_mean")

    messages.append(
        f"K-Means converged: n_clusters={n_clusters}, n_samples={n_samples}, "
        f"inertia={inertia:.4f}" + (f", silhouette={silhouette:.4f}" if silhouette is not None else "")
    )

    return KMeansResult(
        model=model,
        scaler=scaler,
        labels=labels,
        scaled_features=scaled_df,
        inertia=inertia,
        silhouette=silhouette,
        n_clusters=n_clusters,
        n_samples_used=n_samples,
        n_samples_dropped_missing=dropped,
        cluster_sizes=cluster_sizes,
        cluster_profiles=profiles,
        feature_means_raw=feature_means_raw,
        messages=messages,
        warnings=warnings,
    )


def compare_cluster_counts(
    df: pd.DataFrame,
    features: list[str],
    *,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Run K-Means for a range of k and return inertia/silhouette per k.

    Silhouette is NaN where mathematically invalid.
    """
    rows = []
    work = df[features].dropna()
    n_samples = len(work)
    if n_samples == 0:
        return pd.DataFrame(columns=["k", "n_samples", "inertia", "silhouette"])

    k_max = min(k_max, n_samples)
    k_min = max(k_min, 2)
    if k_min > k_max:
        return pd.DataFrame(columns=["k", "n_samples", "inertia", "silhouette"])

    scaler = StandardScaler()
    X = scaler.fit_transform(work.values)

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)
        sil = None
        if 2 <= k <= n_samples - 1:
            try:
                sil = float(silhouette_score(X, labels))
            except ValueError:
                sil = None
        rows.append({"k": int(k), "n_samples": int(n_samples), "inertia": float(model.inertia_), "silhouette": sil})
    return pd.DataFrame(rows)
