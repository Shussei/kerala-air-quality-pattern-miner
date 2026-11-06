"""K-Means clustering page."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui_helpers import section_header, plotly_figure, show_dataframe, status_box, metric_card
from src.airquality.clustering import compute_kmeans, compare_cluster_counts, eligible_numeric_features

st.title("\U0001F4BB K-Means Clustering")
st.caption("scikit-learn K-Means on scaled numeric features. Cluster numbers are arbitrary identifiers, "
           "not pollution rankings. Clustering finds similarity patterns; it does not establish causation.")

# Choose source dataset
df = st.session_state.get("df_integrated")
if df is None or (hasattr(df, '__len__') and len(df) == 0):
    df = st.session_state.get("df_aq_clean")
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        df = st.session_state.get("df_aq_raw")
if df is None or (hasattr(df, '__len__') and len(df) == 0):
    df = st.session_state.get("df_wx_clean")
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        df = st.session_state.get("df_wx_raw")

if df is None or (hasattr(df, "__len__") and len(df) == 0):
    st.warning("Load a dataset first (Data Import / Integration pages).")
    st.stop()

all_features = eligible_numeric_features(df)
if not all_features:
    st.error("No numeric features with variation available for clustering.")
    st.stop()

st.write(f"Clustering on: **{len(df)} rows** (source: current dataset). "
         f"Eligible numeric features: {', '.join(all_features)}")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.subheader("1. Configure K-Means")
c1, c2, c3 = st.columns(3)
features = c1.multiselect("Features", all_features, default=all_features[: min(3, len(all_features))],
                          key="km_features")
n_clusters = c2.number_input("Number of clusters (k)", min_value=2, max_value=50, value=3, step=1,
                             key="km_k")
random_state = c3.number_input("random_state", min_value=0, value=42, step=1, key="km_seed")
n_init = st.number_input("n_init (initializations)", min_value=1, value=10, step=1, key="km_n_init",
                         help="Number of times the algorithm runs with different centroid seeds.")

st.info(
    "Missing-value policy: rows with any missing value in the selected features are excluded "
    "(counted and reported). They are never imputed with zero. Features are standardised with "
    "StandardScaler fitted only on the clustering data (no data leakage)."
)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("2. Run clustering")

if st.button("Run K-Means", type="primary", key="btn_km"):
    if not features:
        st.error("Select at least one feature.")
        st.stop()
    res = compute_kmeans(df, features, n_clusters=int(n_clusters),
                         random_state=int(random_state), n_init=int(n_init))
    st.session_state.kmeans_result = res
    for m in res.messages:
        st.info(m)
    for w in res.warnings:
        st.warning(w)

res = st.session_state.get("kmeans_result")
if res is None:
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("3. Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples used", res.n_samples_used)
c2.metric("Samples dropped (missing)", res.n_samples_dropped_missing)
c3.metric("Inertia", f"{res.inertia:.4f}" if res.inertia is not None else "—")
c4.metric("Silhouette", f"{res.silhouette:.4f}" if res.silhouette is not None else "— (not valid)")

if res.silhouette is not None:
    st.caption("Silhouette is reported only when 2 ≤ k ≤ n_samples−1 (mathematically valid).")

st.write("#### Cluster sizes")
size_df = pd.DataFrame([{"cluster": int(k), "size": int(v)} for k, v in res.cluster_sizes.items()])
fig = px.bar(size_df, x="cluster", y="size", title="Cluster sizes",
             color_discrete_sequence=["#1f77b4"])
fig.update_xaxes(type="category")
plotly_figure(fig, key="km_sizes_fig", caption="Number of observations per cluster.")
show_dataframe(size_df, key="km_sizes_tbl", download_name="cluster_sizes.csv")

st.write("#### Cluster feature profiles (raw units, per cluster)")
show_dataframe(res.cluster_profiles.reset_index(), key="km_profiles_tbl",
               download_name="cluster_profiles.csv")

# Visualization: scatter of first two scaled features
st.write("#### Cluster visualization")
if res.scaled_features is not None and res.labels is not None:
    plot_df = res.scaled_features.copy()
    plot_df["cluster"] = res.labels.astype(str)
    feat_cols = list(res.scaled_features.columns)
    x_feat = st.selectbox("X feature", feat_cols, index=0, key="km_scatter_x")
    y_feat = st.selectbox("Y feature", feat_cols, index=min(1, len(feat_cols) - 1), key="km_scatter_y")
    fig = px.scatter(plot_df, x=x_feat, y=y_feat, color="cluster", title=f"K-Means clusters ({x_feat} vs {y_feat})",
                     template="plotly_white", height=500)
    plotly_figure(fig, key="km_scatter_fig", caption="Scaled feature space; points coloured by assigned cluster.")

# Cluster assignments
st.write("#### Cluster assignments (aligned to original input rows)")
assign = res.assignments_dataframe(df.index)
show_dataframe(assign.head(30), key="km_assign_tbl", download_name="cluster_assignments.csv")
csv_all = assign.to_csv(index=False).encode("utf-8")
st.download_button("Download all cluster assignments (CSV)", data=csv_all,
                   file_name="cluster_assignments.csv", mime="text/csv", key="km_dl_assign")

st.divider()
st.subheader("4. Compare cluster counts (elbow / silhouette)")
k_min = st.number_input("k min", min_value=2, value=2, key="km_cmp_min")
k_max = st.number_input("k max", min_value=2, value=10, key="km_cmp_max")
if st.button("Compare cluster counts", key="btn_km_cmp"):
    cmp_df = compare_cluster_counts(df, features, k_min=int(k_min), k_max=int(k_max))
    if cmp_df.empty:
        st.info("Not enough samples to compare across that range.")
    else:
        show_dataframe(cmp_df, key="km_cmp_tbl", download_name="cluster_comparison.csv")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cmp_df["k"], y=cmp_df["inertia"], mode="lines+markers", name="inertia"))
        fig.update_layout(title="Inertia vs k (elbow)", xaxis_title="k", yaxis_title="Inertia",
                          template="plotly_white", height=400)
        plotly_figure(fig, key="km_elbow_fig")
        valid_sil = cmp_df.dropna(subset=["silhouette"])
        if not valid_sil.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=valid_sil["k"], y=valid_sil["silhouette"], mode="lines+markers", name="silhouette"))
            fig2.update_layout(title="Silhouette vs k (where valid)", xaxis_title="k", yaxis_title="Silhouette",
                               template="plotly_white", height=400)
            plotly_figure(fig2, key="km_sil_fig")
        else:
            st.info("No valid silhouette scores in this k range.")
