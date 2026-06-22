"""
Time series forecasting with Prophet.

Pure Python/pandas module used by the Qt page. It prepares a selected date
column and target column for Prophet, fits the model, computes fitted-value
statistics, renders a forecast plot, and returns export-ready predictions.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesForecastResult:
    date_col: str
    value_col: str
    periods: int
    frequency: str
    holidays_country: str | None
    row_count: int
    historical_rows: int
    forecast_rows: int
    stats: dict[str, float | int | str | None]
    forecast: pd.DataFrame
    plot_path: str


class TimeSeriesForecastEngine:
    @staticmethod
    def fit_forecast(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        periods: int,
        *,
        holidays_country: str | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> TimeSeriesForecastResult:
        Prophet, plt = TimeSeriesForecastEngine._import_dependencies()

        if df is None or df.empty:
            raise ValueError("No dataset loaded. Please import data first.")
        if date_col == value_col:
            raise ValueError("Date/time and forecast value columns must be different.")
        if periods < 1:
            raise ValueError("Forecast periods must be at least 1.")

        if progress:
            progress(10)

        history = TimeSeriesForecastEngine._prepare_history(df, date_col, value_col)
        if len(history) < 3:
            raise ValueError("Need at least 3 valid time points to fit a forecast.")

        freq = TimeSeriesForecastEngine._infer_frequency(history["ds"])
        if progress:
            progress(30)

        model = Prophet()
        if holidays_country:
            try:
                model.add_country_holidays(country_name=holidays_country)
            except Exception as exc:
                raise ValueError(
                    f"Prophet does not support public holidays for '{holidays_country}'."
                ) from exc
        model.fit(history)

        if progress:
            progress(55)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        merged = (
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .merge(history, on="ds", how="left")
            .rename(columns={"ds": date_col, "y": "actual"})
        )
        merged["is_forecast"] = merged["actual"].isna()

        stats = TimeSeriesForecastEngine._fit_stats(merged, date_col)
        plot_path = TimeSeriesForecastEngine._render_plot(plt, merged, date_col, value_col)

        if progress:
            progress(100)

        return TimeSeriesForecastResult(
            date_col=date_col,
            value_col=value_col,
            periods=periods,
            frequency=freq,
            holidays_country=holidays_country,
            row_count=len(df),
            historical_rows=int((~merged["is_forecast"]).sum()),
            forecast_rows=int(merged["is_forecast"].sum()),
            stats=stats,
            forecast=merged,
            plot_path=plot_path,
        )

    @staticmethod
    def _prepare_history(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            raise ValueError(f"Date/time column '{date_col}' was not found.")
        if value_col not in df.columns:
            raise ValueError(f"Forecast value column '{value_col}' was not found.")

        work = pd.DataFrame({
            "ds": pd.to_datetime(df[date_col], errors="coerce"),
            "y": pd.to_numeric(df[value_col], errors="coerce"),
        }).dropna(subset=["ds", "y"])

        if work.empty:
            raise ValueError("No valid date/value rows remain after parsing.")

        work = (
            work.groupby("ds", as_index=False)["y"]
            .sum()
            .sort_values("ds")
            .reset_index(drop=True)
        )
        return work

    @staticmethod
    def _infer_frequency(ds: pd.Series) -> str:
        inferred = pd.infer_freq(ds)
        if inferred:
            return inferred

        deltas = ds.sort_values().diff().dropna()
        if deltas.empty:
            return "D"
        median_delta = deltas.median()
        if median_delta <= pd.Timedelta(hours=1):
            return "h"
        if median_delta <= pd.Timedelta(days=1):
            return "D"
        if median_delta <= pd.Timedelta(days=8):
            return "W"
        if median_delta <= pd.Timedelta(days=32):
            return "MS"
        if median_delta <= pd.Timedelta(days=95):
            return "QS"
        return "YS"

    @staticmethod
    def _fit_stats(
        forecast: pd.DataFrame,
        date_col: str,
    ) -> dict[str, float | int | str | None]:
        fitted = forecast[forecast["actual"].notna()].copy()
        errors = fitted["actual"] - fitted["yhat"]
        abs_errors = errors.abs()
        mae = float(abs_errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        nonzero = fitted["actual"].abs() > 1e-12
        mape = None
        if nonzero.any():
            mape = float((abs_errors[nonzero] / fitted.loc[nonzero, "actual"].abs()).mean() * 100)

        actual_mean = float(fitted["actual"].mean())
        ss_res = float((errors ** 2).sum())
        ss_tot = float(((fitted["actual"] - actual_mean) ** 2).sum())
        r2 = None if ss_tot == 0 else float(1 - ss_res / ss_tot)

        return {
            "observations": int(len(fitted)),
            "forecast_periods": int(forecast["is_forecast"].sum()),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "last_actual_date": str(fitted.iloc[-1][date_col].date()),
        }

    @staticmethod
    def _render_plot(plt, forecast: pd.DataFrame, date_col: str, value_col: str) -> str:
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(11, 6))

        fitted = forecast[forecast["actual"].notna()]
        future = forecast[forecast["is_forecast"]]

        x_all = forecast[date_col].to_numpy()
        ax.plot(x_all, forecast["yhat"].to_numpy(), color="#4F7CFF", linewidth=2, label="Fitted / forecast")
        ax.fill_between(
            x_all,
            forecast["yhat_lower"].to_numpy(),
            forecast["yhat_upper"].to_numpy(),
            color="#4F7CFF",
            alpha=0.16,
            label="Uncertainty interval",
        )
        ax.scatter(
            fitted[date_col].to_numpy(),
            fitted["actual"].to_numpy(),
            color="#050505",
            s=18,
            alpha=0.75,
            label="Actual",
        )
        if not future.empty:
            ax.axvline(fitted[date_col].max(), color="#7A7F9A", linestyle="--", linewidth=1)

        ax.set_title(f"Forecast for {value_col}", fontsize=13)
        ax.set_xlabel(date_col)
        ax.set_ylabel(value_col)
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()

        out_dir = Path(tempfile.gettempdir()) / "customer_analytics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "time_series_forecast.png"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    @staticmethod
    def _import_dependencies():
        try:
            from prophet import Prophet
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Time series forecasting requires prophet and matplotlib. "
                "Install them with: pip install -r requirements.txt"
            ) from exc
        return Prophet, plt
