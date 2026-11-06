"""Apriori association-rules page."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from ui_helpers import section_header, show_dataframe, status_box, metric_card
from src.airquality.association_rules import mine_from_discretized, run_apriori

st.title("\U0001F6E0 Apriori Association Rules")
st.caption("Frequent itemsets and association rules via mlxtend. Rules describe co-occurrence patterns, "
           "not causation or predictive certainty.")

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

st.write(f"Source dataset: **{len(df)} rows x {len(df.columns)} columns**")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.subheader("1. Select items and discretisation")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

default_cat = [c for c in ("location", "season") if c in df.columns]
cat_features = st.multiselect("Categorical items (used directly)", cat_cols,
                              default=[c for c in default_cat if c in cat_cols], key="ap_cat")
num_features = st.multiselect("Numeric items to discretise", numeric_cols, key="ap_num")

c1, c2, c3 = st.columns(3)
n_bins = c1.number_input("Bins per numeric feature", min_value=2, max_value=10, value=4, step=1, key="ap_bins")
strategy = c2.selectbox("Binning strategy", ["quantile", "uniform"], index=0, key="ap_strategy",
                        help="Quantile = equal-frequency bins; uniform = equal-width bins.")
min_support = c3.number_input("Minimum support", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                              format="%.2f", key="ap_support")
min_confidence = st.number_input("Minimum confidence", min_value=0.01, max_value=1.0, value=0.5, step=0.05,
                                 format="%.2f", key="ap_confidence")
min_lift = st.number_input("Minimum lift filter", min_value=1.0, max_value=10.0, value=1.0, step=0.1,
                           format="%.1f", key="ap_lift")

st.info(
    "Discretisation bin edges and category definitions are recorded and shown below. "
    "Missing items are treated as absent (documented), not imputed."
)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("2. Mine rules")

item_columns = cat_features + num_features
if st.button("Run Apriori", type="primary", key="btn_apriori"):
    if not item_columns:
        st.error("Select at least one categorical or numeric feature.")
        st.stop()
    result = mine_from_discretized(
        df, numeric_features=num_features, categorical_features=cat_features,
        min_support=float(min_support), min_confidence=float(min_confidence),
        n_bins=int(n_bins), strategy=strategy, min_lift=float(min_lift),
    )
    st.session_state.apriori_result = result
    for m in result.messages:
        st.info(m)
    for w in result.warnings:
        st.warning(w)

result = st.session_state.get("apriori_result")
if result is None:
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("3. Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", result.n_transactions)
c2.metric("Items", len(result.items))
c3.metric("Frequent itemsets", len(result.itemsets) if result.itemsets is not None else 0)
c4.metric("Association rules", len(result.rules) if result.rules is not None else 0)

if result.discretization is not None:
    st.write("**Discretisation (documented bin edges)**")
    disc_rows = []
    for f, edges in result.discretization.bin_edges.items():
        disc_rows.append({
            "feature": f,
            "n_bins": len(edges) - 1,
            "bin_edges": ", ".join(f"{e:.3g}" for e in edges),
            "labels": ", ".join(result.discretization.labels.get(f, [])),
        })
    show_dataframe(pd.DataFrame(disc_rows), key="ap_disc_tbl", download_name="discretisation.csv")

st.write("#### Frequent itemsets")
itemsets_disp = result.itemsets.copy() if result.itemsets is not None else pd.DataFrame()
if not itemsets_disp.empty:
    itemsets_disp = itemsets_disp[["support", "itemsets_str"]].rename(
        columns={"itemsets_str": "itemsets"})
    show_dataframe(itemsets_disp, key="ap_itemsets_tbl", download_name="frequent_itemsets.csv")
else:
    st.info("No frequent itemsets at the selected min_support. Lower min_support.")

st.write("#### Association rules")
rules_disp = result.rules.copy() if result.rules is not None else pd.DataFrame()
if not rules_disp.empty:
    show_cols = [c for c in ["antecedents_str", "consequents_str", "support", "confidence", "lift"] if c in rules_disp.columns]
    show_dataframe(rules_disp[show_cols].rename(columns={"antecedents_str": "antecedents", "consequents_str": "consequents"}),
                   key="ap_rules_tbl", download_name="association_rules.csv")
    csv = result.rules.to_csv(index=False).encode("utf-8")
    st.download_button("Download all association rules (CSV)", data=csv,
                       file_name="association_rules_full.csv", mime="text/csv", key="ap_dl_rules")
else:
    st.info("No association rules at the selected thresholds. Try lowering min_support or min_confidence.")

st.caption("Interpretation: a rule {A} ⇒ {B} with support s, confidence c and lift l means s of all "
           "transactions contain both A and B; c of transactions containing A also contain B; "
           "lift > 1 indicates positive co-occurrence beyond independence.")
