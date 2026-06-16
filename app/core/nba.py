"""
Rule-based Next Best Action engine.

The first NBA implementation deliberately avoids heavy modeling. It detects
common customer columns, scores a fixed action set, and returns explainable
recommendations that work with minimal user guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class NBAResult:
    recommendations: pd.DataFrame
    action_distribution: pd.Series
    detected_columns: dict[str, str | None]


class NextBestActionEngine:
    ACTIONS = [
        "Retain",
        "Win back",
        "Upsell",
        "Cross-sell",
        "Re-engage",
        "Loyalty nurture",
        "Monitor",
    ]

    COLUMN_PATTERNS = {
        "customer_id": ["customer_id", "customerid", "cust_id", "client_id", "account_id", "id"],
        "recency": ["recency", "days_since", "days inactive", "days_inactive", "last_purchase", "last_order"],
        "frequency": ["frequency", "orders", "order_count", "transactions", "txn_count", "purchases"],
        "monetary": ["monetary", "revenue", "spend", "sales", "amount", "ltv", "value"],
        "churn": ["churn", "attrition", "risk", "prob_churn", "churn_probability"],
        "engagement": ["engagement", "activity", "visits", "sessions", "opens", "clicks"],
        "products": ["products", "product_count", "num_products", "sku_count", "categories"],
    }

    @staticmethod
    def recommend(
        df: pd.DataFrame,
        customer_id_col: str | None = None,
        limit: int = 100,
        segmentation_result: Any | None = None,
    ) -> NBAResult:
        if df is None or df.empty:
            raise ValueError("No dataset loaded. Please import data first.")

        detected = NextBestActionEngine.detect_columns(df)
        if customer_id_col:
            detected["customer_id"] = customer_id_col

        id_col = detected.get("customer_id")
        feature_cols = [
            c for c in [
                detected.get("recency"),
                detected.get("frequency"),
                detected.get("monetary"),
                detected.get("churn"),
                detected.get("engagement"),
                detected.get("products"),
            ]
            if c is not None
        ]

        if not feature_cols:
            raise ValueError(
                "No usable NBA signals found. Add columns such as recency, frequency, "
                "monetary value, churn risk, engagement, or product count."
            )

        work_cols = ([id_col] if id_col else []) + feature_cols
        work = df[work_cols].copy()
        if id_col is None:
            id_col = "row_number"
            work[id_col] = df.index.astype(str)

        segment_col = None
        if segmentation_result is not None:
            work, segment_col = NextBestActionEngine._attach_segments(
                work, id_col, segmentation_result
            )

        scored_rows = []
        for _, row in work.iterrows():
            scores = NextBestActionEngine._score_row(row, df, detected)
            action, score = max(scores.items(), key=lambda item: item[1])
            confidence = NextBestActionEngine._confidence(score)
            reasons = NextBestActionEngine._reasons(row, df, detected, action)
            expected_value = NextBestActionEngine._expected_value(row, df, detected, action, score)

            out = {
                "customer_id": row[id_col],
                "recommended_action": action,
                "confidence": confidence,
                "score": round(score, 1),
                "expected_value": round(expected_value, 2),
                "reason": "; ".join(reasons) if reasons else "Best available action from detected signals",
            }
            if segment_col:
                out["segment"] = row.get(segment_col, "—")
            scored_rows.append(out)

        recs = pd.DataFrame(scored_rows)
        if recs.empty:
            raise ValueError("No recommendations could be generated from the current dataset.")

        recs = recs.sort_values(
            by=["score", "expected_value"], ascending=[False, False]
        ).head(limit).reset_index(drop=True)
        distribution = recs["recommended_action"].value_counts()
        return NBAResult(recs, distribution, detected)

    @staticmethod
    def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
        detected: dict[str, str | None] = {}
        normalized = {
            col: NextBestActionEngine._normalize_name(col)
            for col in df.columns
        }

        for role, patterns in NextBestActionEngine.COLUMN_PATTERNS.items():
            detected[role] = None
            for col, norm in normalized.items():
                if role != "customer_id" and not is_numeric_dtype(df[col]):
                    continue
                if any(pattern in norm for pattern in patterns):
                    detected[role] = col
                    break

        if detected.get("customer_id") is None:
            for col in df.columns:
                if not is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) == len(df):
                    detected["customer_id"] = col
                    break

        return detected

    @staticmethod
    def _score_row(row: pd.Series, df: pd.DataFrame, detected: dict[str, str | None]) -> dict[str, float]:
        inactivity = NextBestActionEngine._rank_signal(row, df, detected.get("recency"))
        recent_activity = 1 - inactivity if inactivity is not None else None
        frequency = NextBestActionEngine._rank_signal(row, df, detected.get("frequency"))
        monetary = NextBestActionEngine._rank_signal(row, df, detected.get("monetary"))
        churn = NextBestActionEngine._rank_signal(row, df, detected.get("churn"))
        engagement = NextBestActionEngine._rank_signal(row, df, detected.get("engagement"))
        products = NextBestActionEngine._rank_signal(row, df, detected.get("products"))

        low_engagement = 1 - engagement if engagement is not None else None
        low_products = 1 - products if products is not None else None

        scores = {action: 0.0 for action in NextBestActionEngine.ACTIONS}
        scores["Retain"] = NextBestActionEngine._weighted([
            (churn, 45), (monetary, 25), (frequency, 15), (low_engagement, 15)
        ])
        scores["Win back"] = NextBestActionEngine._weighted([
            (inactivity, 40), (monetary, 25), (frequency, 20), (low_engagement, 15)
        ])
        scores["Upsell"] = NextBestActionEngine._weighted([
            (monetary, 35), (frequency, 25), (engagement, 25), (1 - churn if churn is not None else None, 15)
        ])
        scores["Cross-sell"] = NextBestActionEngine._weighted([
            (engagement, 30), (frequency, 25), (monetary, 20), (low_products, 25)
        ])
        scores["Re-engage"] = NextBestActionEngine._weighted([
            (low_engagement, 45), (inactivity, 25), (frequency, 15), (monetary, 15)
        ])
        scores["Loyalty nurture"] = NextBestActionEngine._weighted([
            (frequency, 30), (monetary, 25), (engagement, 20),
            (recent_activity, 15), (1 - churn if churn is not None else None, 10)
        ])
        scores["Monitor"] = 35.0
        return scores

    @staticmethod
    def _weighted(parts: list[tuple[float | None, float]]) -> float:
        available = [(value, weight) for value, weight in parts if value is not None]
        if not available:
            return 0.0
        total_weight = sum(weight for _, weight in available)
        return sum(value * weight for value, weight in available) / total_weight * 100

    @staticmethod
    def _rank_signal(
        row: pd.Series,
        df: pd.DataFrame,
        col: str | None,
        high_is_good: bool = True,
    ) -> float | None:
        if col is None or pd.isna(row.get(col)):
            return None
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return None
        value = float(row[col])
        lo = float(series.min())
        hi = float(series.max())
        if hi == lo:
            score = 0.5
        else:
            score = (value - lo) / (hi - lo)
        score = max(0.0, min(1.0, score))
        return score if high_is_good else 1 - score

    @staticmethod
    def _reasons(
        row: pd.Series,
        df: pd.DataFrame,
        detected: dict[str, str | None],
        action: str,
    ) -> list[str]:
        reasons = []
        checks = [
            ("monetary", "high value", True),
            ("frequency", "frequent buyer", True),
            ("engagement", "high engagement", True),
            ("churn", "elevated churn risk", True),
            ("products", "limited product breadth", False),
            ("recency", "inactive recently", True),
        ]
        for role, text, high_is_good in checks:
            col = detected.get(role)
            signal = NextBestActionEngine._rank_signal(row, df, col, high_is_good)
            if signal is not None and signal >= 0.70:
                reasons.append(text)

        if action in {"Win back", "Re-engage"}:
            recency_col = detected.get("recency")
            if recency_col and pd.notna(row.get(recency_col)):
                reasons.append(f"{recency_col}={row[recency_col]}")
        return reasons[:3]

    @staticmethod
    def _expected_value(
        row: pd.Series,
        df: pd.DataFrame,
        detected: dict[str, str | None],
        action: str,
        score: float,
    ) -> float:
        monetary_col = detected.get("monetary")
        if monetary_col and pd.notna(row.get(monetary_col)):
            base_value = float(row[monetary_col])
        elif monetary_col:
            base_value = float(pd.to_numeric(df[monetary_col], errors="coerce").median())
        else:
            base_value = 100.0

        multipliers = {
            "Retain": 0.20,
            "Win back": 0.16,
            "Upsell": 0.28,
            "Cross-sell": 0.18,
            "Re-engage": 0.10,
            "Loyalty nurture": 0.08,
            "Monitor": 0.02,
        }
        return max(base_value, 0) * multipliers.get(action, 0.05) * (score / 100)

    @staticmethod
    def _confidence(score: float) -> str:
        if score >= 70:
            return "High"
        if score >= 50:
            return "Medium"
        return "Low"

    @staticmethod
    def _attach_segments(
        work: pd.DataFrame,
        customer_id_col: str,
        segmentation_result: Any,
    ) -> tuple[pd.DataFrame, str | None]:
        assignments = getattr(segmentation_result, "assignments", None)
        seg_id_col = getattr(segmentation_result, "customer_id_col", None)
        label_col = getattr(segmentation_result, "label_col", None)
        if assignments is None or seg_id_col is None or label_col is None:
            return work, None
        if seg_id_col not in assignments.columns or label_col not in assignments.columns:
            return work, None

        segs = assignments[[seg_id_col, label_col]].drop_duplicates(seg_id_col)
        merged = work.merge(
            segs,
            left_on=customer_id_col,
            right_on=seg_id_col,
            how="left",
        )
        if seg_id_col != customer_id_col and seg_id_col in merged.columns:
            merged = merged.drop(columns=[seg_id_col])
        return merged, label_col

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).lower().replace("-", "_").replace(" ", "_")
