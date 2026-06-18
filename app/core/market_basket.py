"""
Market basket analysis — Apriori frequent itemsets and association rules.

Follows the mlxtend workflow:
  1. Group items by transaction
  2. One-hot encode with TransactionEncoder
  3. apriori(min_support)
  4. association_rules(metric="confidence", min_threshold=min_confidence)

Pure Python/pandas/mlxtend — no Qt, no AppState.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from app.core.duckdb_sample import open_dataset_view, quote_identifier


MAX_TRANSACTIONS = 30_000      # cap baskets analysed in one run
MAX_UNIQUE_ITEMS = 2_000         # keep top-N products by line frequency
MAX_RULES_STORED = 5_000       # cap rules kept in memory / export
APRIORI_MAX_LEN = 4            # limit itemset size (reduces memory)
MAX_MATRIX_BYTES = 2 * 1024 ** 3   # 2 GB ceiling for the dense one-hot matrix


@dataclass
class MarketBasketResult:
    transaction_col: str
    item_col: str
    min_support: float
    min_confidence: float
    transaction_count: int
    unique_items: int
    rows_used: int
    sampled: bool
    items_filtered: bool
    transactions_sampled: bool
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame


class MarketBasketEngine:
    @staticmethod
    def run(
        df: pd.DataFrame | None,
        transaction_col: str,
        item_col: str,
        min_support: float,
        min_confidence: float,
        *,
        filepath: str | None = None,
        total_rows: int | None = None,
        encoding: str = "utf-8",
        is_large: bool = False,
        progress: Callable[[int], None] | None = None,
    ) -> MarketBasketResult:
        apriori, association_rules, TransactionEncoder = (
            MarketBasketEngine._import_dependencies()
        )

        if progress:
            progress(10)

        work = MarketBasketEngine._load_columns(
            df, transaction_col, item_col, filepath, encoding, is_large, total_rows,
        )
        if work.empty:
            raise ValueError("No rows available for market basket analysis.")

        work = work.dropna(subset=[transaction_col, item_col])
        work[item_col] = work[item_col].astype(str).str.strip()
        work = work[work[item_col] != ""]

        if work.empty:
            raise ValueError("No valid transaction / item pairs after cleaning.")

        rows_used = len(work)
        sampled = is_large
            
        # Cap item cardinality FIRST (shrinks basket sizes), then cap transaction
        # count. Order matters: filtering items before sampling transactions
        # keeps the resulting one-hot matrix bounded on both axes.
        work, items_filtered = MarketBasketEngine._keep_top_items(work, item_col)
        work, transactions_sampled = MarketBasketEngine._sample_transactions(
            work, transaction_col,
        )

        if progress:
            progress(25)

        transactions = MarketBasketEngine._build_transactions(work, transaction_col, item_col)
        transaction_count = len(transactions)

        if transaction_count < 2:
            raise ValueError("Need at least 2 transactions to mine association rules.")

        # Pre-flight check: the TransactionEncoder builds a DENSE
        # (transaction_count x unique_items) boolean matrix. Estimate its
        # size BEFORE building it so we fail with a clear message instead
        # of an unhandled OOM crash.
        unique_items_estimate = work[item_col].nunique()
        estimated_cells = transaction_count * unique_items_estimate
        estimated_bytes = estimated_cells  # 1 byte per bool cell in pandas
        if estimated_bytes > MAX_MATRIX_BYTES:
            raise ValueError(
                f"Estimated one-hot matrix is too large to fit in memory "
                f"({transaction_count:,} transactions x {unique_items_estimate:,} items "
                f"≈ {estimated_bytes / 1024**3:.2f} GB). "
                f"Lower MAX_TRANSACTIONS / MAX_UNIQUE_ITEMS, or increase min_support "
                f"to shrink the basket first."
            )

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        encoded = pd.DataFrame(te_ary, columns=te.columns_, dtype=bool)

        if progress:
            progress(45)

        frequent_itemsets = apriori(
            encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=APRIORI_MAX_LEN,
            low_memory=True,
        )

        if frequent_itemsets.empty:
            raise ValueError(
                f"No frequent itemsets found at min_support={min_support:.4f}. "
                "Try lowering minimum support."
            )

        if progress:
            progress(70)

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )

        if rules.empty:
            raise ValueError(
                f"No association rules met min_confidence={min_confidence:.4f}. "
                "Try lowering minimum confidence or support."
            )

        rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
        if len(rules) > MAX_RULES_STORED:
            rules = rules.head(MAX_RULES_STORED).reset_index(drop=True)

        frequent_itemsets = frequent_itemsets.sort_values(
            ["support", "itemsets"], ascending=[False, True],
        ).reset_index(drop=True)

        frequent_itemsets = MarketBasketEngine._format_itemsets(frequent_itemsets)
        rules = MarketBasketEngine._format_rules(rules)

        if progress:
            progress(90)

        return MarketBasketResult(
            transaction_col=transaction_col,
            item_col=item_col,
            min_support=min_support,
            min_confidence=min_confidence,
            transaction_count=transaction_count,
            unique_items=len(te.columns_),
            rows_used=rows_used,
            sampled=sampled,
            items_filtered=items_filtered,
            transactions_sampled=transactions_sampled,
            frequent_itemsets=frequent_itemsets,
            rules=rules,
        )

    @staticmethod
    def _build_transactions(
        work: pd.DataFrame,
        transaction_col: str,
        item_col: str,
    ) -> list[list[str]]:
        grouped = work.groupby(transaction_col, sort=False)[item_col]
        return grouped.apply(lambda items: [str(i) for i in items.tolist()]).tolist()

    @staticmethod
    def _keep_top_items(work: pd.DataFrame, item_col: str) -> tuple[pd.DataFrame, bool]:
        n_items = work[item_col].nunique()
        if n_items <= MAX_UNIQUE_ITEMS:
            return work, False
        top_items = (
            work[item_col].value_counts().head(MAX_UNIQUE_ITEMS).index.tolist()
        )
        filtered = work[work[item_col].isin(top_items)].copy()
        return filtered, True

    @staticmethod
    def _sample_transactions(
        work: pd.DataFrame,
        transaction_col: str,
    ) -> tuple[pd.DataFrame, bool]:
        n_tx = work[transaction_col].nunique()
        if n_tx <= MAX_TRANSACTIONS:
            return work, False
        tx_ids = work[transaction_col].drop_duplicates().sample(
            n=MAX_TRANSACTIONS, random_state=42,
        )
        sampled = work[work[transaction_col].isin(tx_ids)].copy()
        return sampled, True

    @staticmethod
    def _import_dependencies():
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        return apriori, association_rules, TransactionEncoder

    @staticmethod
    def _load_columns(
        df: pd.DataFrame | None,
        transaction_col: str,
        item_col: str,
        filepath: str | None,
        encoding: str,
        is_large: bool,
        total_rows: int | None,
    ) -> pd.DataFrame:
        cols = [transaction_col, item_col]

        if is_large:
            if not filepath:
                raise ValueError("Large dataset path not available.")
            tx = quote_identifier(transaction_col)
            item = quote_identifier(item_col)
            con = open_dataset_view(filepath, encoding)
            try:
                sql = f"SELECT {tx}, {item} FROM dataset WHERE {tx} IS NOT NULL AND {item} IS NOT NULL"
                return con.execute(sql).df()
            finally:
                con.close()

        if df is None:
            raise ValueError("No dataset loaded.")

        available = set(df.columns)
        missing = [c for c in cols if c not in available]
        if missing:
            raise ValueError(f"Column(s) not found: {', '.join(missing)}")

        work = df[cols].copy()
        return work

    @staticmethod
    def _format_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
        out = itemsets.copy()
        out["items"] = out["itemsets"].map(MarketBasketEngine._itemset_label)
        out["item_count"] = out["itemsets"].map(len)
        return out.drop(columns=["itemsets"])

    @staticmethod
    def _format_rules(rules: pd.DataFrame) -> pd.DataFrame:
        out = rules.copy()
        out["antecedents"] = out["antecedents"].map(MarketBasketEngine._itemset_label)
        out["consequents"] = out["consequents"].map(MarketBasketEngine._itemset_label)
        out["rule"] = out["antecedents"] + "  →  " + out["consequents"]
        ordered = ["rule", "antecedents", "consequents"]
        for col in [
            "antecedent support", "consequent support",
            "support", "confidence", "lift", "leverage", "conviction",
        ]:
            if col in out.columns:
                ordered.append(col)
        return out[ordered]

    @staticmethod
    def _itemset_label(value) -> str:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return ", ".join(sorted(str(v) for v in value))
        return str(value)