"""Apriori frequent-itemset and association-rule mining using mlxtend.

Rules are interpreted as co-occurrence patterns, not proof of causation or
predictive certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from warnings import catch_warnings, simplefilter

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .preprocessing import DiscretizationBundle


@dataclass
class AprioriResult:
    itemsets: pd.DataFrame
    rules: pd.DataFrame
    transaction_matrix: pd.DataFrame
    items: list[str]
    n_transactions: int
    min_support: float
    min_confidence: float
    discretization: Optional[DiscretizationBundle] = None
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has_rules(self) -> bool:
        return self.rules is not None and len(self.rules) > 0


def _format_itemset(itemset) -> str:
    return ", ".join(sorted(itemset))


def build_transaction_matrix(
    df: pd.DataFrame,
    item_columns: list[str],
) -> pd.DataFrame:
    """Convert selected (already Boolean or categorical) columns into a
    Boolean transaction matrix (one row per observation).

    Non-selected columns are ignored. Missing values become False (item
    absent) — this is an explicit, documented choice, not imputation.
    """
    if not item_columns:
        return pd.DataFrame()
    matrix = df[item_columns].copy()
    for c in item_columns:
        matrix[c] = matrix[c].notna() & (matrix[c].astype(str).str.lower() != "nan")
        matrix[c] = matrix[c].astype(bool)
    return matrix.reset_index(drop=True)


def run_apriori(
    transaction_matrix: pd.DataFrame,
    *,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    metric: str = "lift",
    min_lift: float = 1.0,
) -> AprioriResult:
    """Run Apriori and generate association rules.

    Validates the Boolean matrix. Returns empty itemsets/rules with clear
    messages when nothing qualifies.
    """
    messages: list[str] = []
    warnings: list[str] = []

    empty = AprioriResult(
        itemsets=pd.DataFrame(columns=["support", "itemsets"]),
        rules=pd.DataFrame(),
        transaction_matrix=transaction_matrix,
        items=list(transaction_matrix.columns),
        n_transactions=len(transaction_matrix),
        min_support=min_support,
        min_confidence=min_confidence,
    )

    if transaction_matrix is None or transaction_matrix.empty:
        messages.append("Transaction matrix is empty.")
        empty.messages = messages
        empty.warnings = warnings
        return empty

    if not all(transaction_matrix.dtypes == bool):
        non_bool = [c for c in transaction_matrix.columns if transaction_matrix[c].dtype != bool]
        messages.append(
            f"Transaction matrix must be Boolean. Non-Boolean column(s): {', '.join(non_bool)}"
        )
        empty.messages = messages
        empty.warnings = warnings
        return empty

    n_transactions = len(transaction_matrix)
    if n_transactions == 0:
        messages.append("No transactions to mine.")
        empty.messages = messages
        empty.warnings = warnings
        return empty

    items = list(transaction_matrix.columns)
    messages.append(f"{n_transactions} transactions, {len(items)} items.")

    # --- Frequent itemsets -------------------------------------------------
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        try:
            itemsets = apriori(transaction_matrix, min_support=min_support, use_colnames=True, max_len=None)
        except ValueError as exc:
            messages.append(f"Apriori failed: {exc}")
            empty.messages = messages
            empty.warnings = warnings
            return empty

    if itemsets is None or itemsets.empty:
        messages.append(
            f"No frequent itemsets at min_support={min_support} "
            f"({n_transactions} transactions). Lower min_support to find itemsets."
        )
        empty.messages = messages
        empty.warnings = warnings
        return empty

    itemsets = itemsets.copy()
    itemsets["itemsets_str"] = itemsets["itemsets"].map(_format_itemset)
    messages.append(f"{len(itemsets)} frequent itemset(s) found at min_support={min_support}.")

    # --- Association rules -------------------------------------------------
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        try:
            rules = association_rules(
                itemsets, metric=metric, min_threshold=min_confidence
            )
        except ValueError as exc:
            messages.append(f"Rule generation failed: {exc}")
            rules = pd.DataFrame()

    if rules is None or rules.empty:
        messages.append(
            f"No association rules at min_confidence={min_confidence}. "
            "Try lowering min_confidence or min_support."
        )
        return AprioriResult(
            itemsets=itemsets,
            rules=pd.DataFrame(),
            transaction_matrix=transaction_matrix,
            items=items,
            n_transactions=n_transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            messages=messages,
            warnings=warnings,
        )

    rules = rules.copy()
    rules["antecedents_str"] = rules["antecedents"].map(_format_itemset)
    rules["consequents_str"] = rules["consequents"].map(_format_itemset)
    if min_lift > 1.0:
        before = len(rules)
        rules = rules[rules["lift"] >= min_lift]
        if len(rules) < before:
            messages.append(f"Filtered to {len(rules)} rule(s) with lift >= {min_lift}.")

    # Sort by lift desc, then confidence desc.
    sort_cols = [c for c in ["lift", "confidence", "support"] if c in rules.columns]
    if sort_cols:
        rules = rules.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    messages.append(f"{len(rules)} association rule(s) generated.")
    return AprioriResult(
        itemsets=itemsets,
        rules=rules,
        transaction_matrix=transaction_matrix,
        items=items,
        n_transactions=n_transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        messages=messages,
        warnings=warnings,
    )


def mine_from_discretized(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    n_bins: int = 4,
    strategy: str = "quantile",
    min_lift: float = 1.0,
) -> AprioriResult:
    """Convenience: discretize numeric features, build Boolean matrix, mine.

    Bin edges and category definitions are documented via the returned
    discretization bundle.
    """
    messages: list[str] = []
    disc = DiscretizationBundle(n_bins=n_bins, strategy=strategy)
    disc.fit(df, numeric_features)
    binned = disc.transform(df, numeric_features)

    item_columns = list(categorical_features)
    for f in numeric_features:
        bin_col = f + "_bin"
        if bin_col in binned.columns:
            item_columns.append(bin_col)
            messages.append(
                f"Discretized '{f}' into bins {disc.bin_edges.get(f)} "
                f"(labels: {', '.join(disc.labels.get(f, []))})."
            )

    available = [c for c in item_columns if c in binned.columns]
    missing = [c for c in item_columns if c not in binned.columns]
    if missing:
        messages.append(f"Requested item column(s) not found: {', '.join(missing)}")

    matrix = build_transaction_matrix(binned, available)
    result = run_apriori(matrix, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
    result.discretization = disc
    result.messages = messages + result.messages
    return result
