"""
CatBoost churn prediction workflow.

This module is UI-free: it samples data when needed, fits a CatBoost classifier,
builds feature importance and SHAP beeswarm artifacts, and returns a scored
dataset that the UI can preview/export.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.core.duckdb_sample import fetch_sample


CHURN_SAMPLE_ROWS = 500_000
SHAP_SAMPLE_ROWS = 2_000


@dataclass
class ChurnModelResult:
    target_col: str
    predictor_cols: list[str]
    row_count: int
    train_rows: int
    scored_rows: int
    sampled: bool
    positive_class: str
    auc: float | None
    accuracy: float | None
    feature_importance: pd.DataFrame
    scored_data: pd.DataFrame
    shap_plot_path: str


class ChurnModelEngine:
    @staticmethod
    def fit(
        df: pd.DataFrame | None,
        target_col: str,
        predictor_cols: list[str],
        *,
        filepath: str | None = None,
        total_rows: int | None = None,
        encoding: str = "utf-8",
        all_columns: list[str] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> ChurnModelResult:
        CatBoostClassifier, Pool, shap, plt, train_test_split, metrics = (
            ChurnModelEngine._import_model_dependencies()
        )

        if not predictor_cols:
            raise ValueError("Select at least one predictor column.")
        if target_col in predictor_cols:
            raise ValueError("The target column cannot also be a predictor.")

        if progress:
            progress(10)

        raw = ChurnModelEngine._training_frame(
            df, target_col, predictor_cols, filepath, total_rows, encoding, all_columns
        )
        sampled = bool(total_rows and total_rows > CHURN_SAMPLE_ROWS) or len(raw) > CHURN_SAMPLE_ROWS
        row_count = int(total_rows or len(raw))

        work = raw[[target_col] + predictor_cols].dropna(subset=[target_col]).copy()
        if work.empty:
            raise ValueError("No rows remain after dropping missing target values.")

        y, positive_class = ChurnModelEngine._encode_target(work[target_col])
        if y.value_counts().min() < 2:
            raise ValueError("The target needs at least two rows in each class.")
        X = ChurnModelEngine._prepare_predictors(work[predictor_cols])
        cat_features = [
            i for i, col in enumerate(predictor_cols)
            if not is_numeric_dtype(X[col])
        ]

        if progress:
            progress(25)

        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        test_size = max(0.25, y.nunique() / len(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        eval_pool =Pool(X_test, y_test, cat_features=cat_features)

        model = CatBoostClassifier(
            iterations=300,
            #learning_rate=0.05,
            #depth=6,
            #loss_function="Logloss",
            #eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True,
        )

        if progress:
            progress(55)

        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        auc = None
        if y_test.nunique() == 2:
            auc = float(metrics.roc_auc_score(y_test, test_proba))
        accuracy = float(metrics.accuracy_score(y_test, test_pred))

        full_pool = Pool(X, y, cat_features=cat_features)
        importances = model.get_feature_importance(full_pool)
        feature_importance = (
            pd.DataFrame({
                "feature": predictor_cols,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        if progress:
            progress(70)

        scored = raw.copy()
        score_X = ChurnModelEngine._prepare_predictors(scored[predictor_cols])
        scored["churn_score"] = model.predict_proba(score_X)[:, 1]
        scored["churn_prediction"] = scored["churn_score"].ge(0.5).astype(int)

        shap_plot_path = ChurnModelEngine._build_shap_beeswarm(
            shap, plt, Pool, model, X, cat_features, max_rows=SHAP_SAMPLE_ROWS
        )

        if progress:
            progress(100)

        return ChurnModelResult(
            target_col=target_col,
            predictor_cols=predictor_cols,
            row_count=row_count,
            train_rows=len(work),
            scored_rows=len(scored),
            sampled=sampled,
            positive_class=positive_class,
            auc=auc,
            accuracy=accuracy,
            feature_importance=feature_importance,
            scored_data=scored,
            shap_plot_path=shap_plot_path,
        )

    @staticmethod
    def _training_frame(
        df: pd.DataFrame | None,
        target_col: str,
        predictor_cols: list[str],
        filepath: str | None,
        total_rows: int | None,
        encoding: str,
        all_columns: list[str] | None,
    ) -> pd.DataFrame:
        model_columns = [target_col] + predictor_cols
        columns = all_columns or model_columns
        if filepath and total_rows:
            return fetch_sample(
                filepath=filepath,
                columns=columns,
                n_rows=min(total_rows, CHURN_SAMPLE_ROWS),
                seed=42,
                encoding=encoding,
            )
        if df is None:
            raise ValueError("No dataset loaded. Please import data first.")
        if len(df) > CHURN_SAMPLE_ROWS:
            return df.sample(n=CHURN_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        return df.copy()

    @staticmethod
    def _encode_target(series: pd.Series) -> tuple[pd.Series, str]:
        values = series.dropna()
        unique = list(pd.Series(values.unique()).sort_values(key=lambda s: s.astype(str)))
        if len(unique) != 2:
            raise ValueError("CatBoost churn modeling expects a binary target variable.")

        lowered = {str(v).strip().lower(): v for v in unique}
        positive_markers = ["1", "true", "yes", "y", "churn", "churned", "high"]
        positive = None
        for marker in positive_markers:
            if marker in lowered:
                positive = lowered[marker]
                break
        if positive is None:
            positive = unique[-1]

        encoded = series.map(lambda v: 1 if v == positive else 0).astype(int)
        return encoded, str(positive)

    @staticmethod
    def _prepare_predictors(df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        for col in X.columns:
            if is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].astype("string").fillna("__missing__").astype(str)
        return X

    @staticmethod
    def _build_shap_beeswarm(
        shap,
        plt,
        Pool,
        model,
        X: pd.DataFrame,
        cat_features: list[int],
        max_rows: int,
    ) -> str:
        X_shap = X.sample(n=min(len(X), max_rows), random_state=42) if len(X) > max_rows else X
        pool = Pool(X_shap, cat_features=cat_features)
        raw_values = model.get_feature_importance(pool, type="ShapValues")
        shap_values = shap.Explanation(
            values=raw_values[:, :-1],
            data=ChurnModelEngine._plot_data(X_shap),
            feature_names=list(X_shap.columns),
        )

        plt.switch_backend("Agg")
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()

        out_dir = Path(tempfile.gettempdir()) / "customer_analytics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "churn_shap_beeswarm.png"
        plt.savefig(path, dpi=400, bbox_inches="tight")
        plt.close()
        return str(path)

    @staticmethod
    def _import_model_dependencies():
        try:
            from catboost import CatBoostClassifier, Pool
            import shap
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn import metrics
        except ImportError as exc:
            raise RuntimeError(
                "Churn modeling requires catboost, shap, matplotlib, and scikit-learn. "
                "Install them with: pip install -r requirements.txt"
            ) from exc
        return CatBoostClassifier, Pool, shap, plt, train_test_split, metrics

    @staticmethod
    def _plot_data(df: pd.DataFrame) -> pd.DataFrame:
        plot_df = df.copy()
        for col in plot_df.columns:
            if not is_numeric_dtype(plot_df[col]):
                plot_df[col] = pd.factorize(plot_df[col], sort=True)[0]
        return plot_df
