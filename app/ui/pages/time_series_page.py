"""Time series forecasting page — Prophet forecasts."""

from __future__ import annotations

import shutil

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QSplitter, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QProgressBar, QTabWidget, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from app.core.time_series import TimeSeriesForecastEngine, TimeSeriesForecastResult
from app.utils.app_state import AppState


DATE_ALIASES = ("date", "datetime", "timestamp", "time", "month", "week", "day")
VALUE_ALIASES = ("sales", "revenue", "amount", "value", "quantity", "orders", "count")
DISPLAY_ROW_LIMIT = 500
HOLIDAY_COUNTRIES = [
    ("US", "United States"),
    ("GB", "United Kingdom"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("ZA", "South Africa"),
    ("CA", "Canada"),
    ("AU", "Australia"),
    ("NL", "Netherlands"),
    ("ES", "Spain"),
    ("IT", "Italy"),
]


class ForecastWorker(QThread):
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = TimeSeriesForecastEngine.fit_forecast(
                df=self.config["df"],
                date_col=self.config["date_col"],
                value_col=self.config["value_col"],
                periods=self.config["periods"],
                holidays_country=self.config.get("holidays_country"),
                progress=self.progress.emit,
            )
            self.finished.emit(result, self.config["dataset_version"])
        except Exception as exc:
            self.error.emit(str(exc))


class TimeSeriesPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: ForecastWorker | None = None
        self._result: TimeSeriesForecastResult | None = None
        self._loaded_version: int | None = None
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        if self._loaded_version != AppState.get_version():
            self._refresh_columns()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([330, 900])
        root.addWidget(splitter)

    def _build_config_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("config_panel")
        panel.setFixedWidth(330)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 28, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Time Series Forecast")
        title.setObjectName("page_title")
        subtitle = QLabel("Fit a Prophet model and export fitted values plus forecasts.")
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._section("Date / Time Column"))
        self._date_combo = QComboBox()
        layout.addWidget(self._date_combo)

        layout.addWidget(self._section("Forecast Value Column"))
        self._value_combo = QComboBox()
        layout.addWidget(self._value_combo)

        layout.addWidget(self._section("Forecast Periods"))
        self._period_spin = QSpinBox()
        self._period_spin.setRange(1, 10_000)
        self._period_spin.setValue(12)
        self._period_spin.setSingleStep(1)
        self._period_spin.setFixedWidth(110)
        layout.addWidget(self._period_spin)

        self._holidays_chk = QCheckBox("Account for public holidays")
        self._holidays_chk.toggled.connect(self._holiday_country_combo_enabled)
        layout.addWidget(self._holidays_chk)

        self._holiday_country_combo = QComboBox()
        for code, label in HOLIDAY_COUNTRIES:
            self._holiday_country_combo.addItem(f"{label} ({code})", code)
        self._holiday_country_combo.setEnabled(False)
        layout.addWidget(self._holiday_country_combo)

        self._run_btn = QPushButton("Fit Forecast")
        self._run_btn.setObjectName("primary_btn")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setObjectName("load_progress")
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setObjectName("status_label")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()

        self._export_frame = QWidget()
        self._export_frame.setVisible(False)
        export_layout = QVBoxLayout(self._export_frame)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(8)
        export_layout.addWidget(self._section("Export"))

        self._export_predictions_btn = QPushButton("⬇  Predictions (.csv)")
        self._export_plot_btn = QPushButton("⬇  Forecast Plot (.png)")
        for btn in [self._export_predictions_btn, self._export_plot_btn]:
            btn.setObjectName("secondary_btn")
            export_layout.addWidget(btn)

        self._export_predictions_btn.clicked.connect(self._export_predictions)
        self._export_plot_btn.clicked.connect(self._export_plot)
        layout.addWidget(self._export_frame)

        scroll.setWidget(content)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)

        self._placeholder = QLabel("Select a date column, value column, and forecast horizon.")
        self._placeholder.setObjectName("page_subtitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder, alignment=Qt.AlignmentFlag.AlignCenter)

        self._results_widget = QWidget()
        self._results_widget.setVisible(False)
        res = QVBoxLayout(self._results_widget)
        res.setContentsMargins(0, 0, 0, 0)
        res.setSpacing(18)

        self._summary_strip = QFrame()
        self._summary_strip.setObjectName("summary_bar")
        strip = QHBoxLayout(self._summary_strip)
        strip.setContentsMargins(16, 8, 16, 8)
        self._lbl_rows = self._strip_lbl("Rows", "—")
        self._lbl_history = self._strip_lbl("History", "—")
        self._lbl_forecast = self._strip_lbl("Forecast", "—")
        self._lbl_freq = self._strip_lbl("Frequency", "—")
        for lbl in [self._lbl_rows, self._lbl_history, self._lbl_forecast, self._lbl_freq]:
            strip.addWidget(lbl)
        res.addWidget(self._summary_strip)

        self._tabs = QTabWidget()

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 8, 0, 0)
        self._plot_image = QLabel("")
        self._plot_image.setObjectName("status_label")
        self._plot_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plot_image.setMinimumHeight(420)
        plot_layout.addWidget(self._plot_image, 1)
        self._tabs.addTab(plot_widget, "Forecast Plot")

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 8, 0, 0)
        self._stats_table = self._make_table()
        self._stats_table.setFixedHeight(260)
        stats_layout.addWidget(self._stats_table)
        stats_layout.addStretch()
        self._tabs.addTab(stats_widget, "Fit Statistics")

        res.addWidget(self._tabs, 1)
        layout.addWidget(self._results_widget, 1)
        return panel

    def _make_table(self) -> QTableWidget:
        tbl = QTableWidget()
        tbl.setObjectName("preview_table")
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setAlternatingRowColors(True)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return tbl

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _strip_lbl(self, key: str, value: str) -> QLabel:
        lbl = QLabel(f"{key}: {value}")
        lbl.setObjectName("summary_item")
        return lbl

    def _refresh_columns(self):
        self._loaded_version = AppState.get_version()
        cols = AppState.get_column_names()
        for combo in [self._date_combo, self._value_combo]:
            combo.blockSignals(True)
            combo.clear()

        if not cols:
            self._run_btn.setEnabled(False)
            self._status.setText("Import a dataset to fit a forecast.")
            for combo in [self._date_combo, self._value_combo]:
                combo.blockSignals(False)
            return

        for combo in [self._date_combo, self._value_combo]:
            combo.addItems(cols)
            combo.blockSignals(False)

        self._select_default_column(self._date_combo, DATE_ALIASES)
        self._select_default_column(self._value_combo, VALUE_ALIASES, prefer_numeric=True)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Ready — {AppState.get_row_count():,} rows detected.")

    def _select_default_column(
        self,
        combo: QComboBox,
        aliases: tuple[str, ...],
        prefer_numeric: bool = False,
    ):
        df = AppState.get_dataframe()
        for i in range(combo.count()):
            name = combo.itemText(i).lower()
            if any(alias in name for alias in aliases):
                combo.setCurrentIndex(i)
                return
        if prefer_numeric and df is not None:
            for i in range(combo.count()):
                col = combo.itemText(i)
                if col in df.columns and df[col].dtype.kind in "fi":
                    combo.setCurrentIndex(i)
                    return

    def _run(self):
        if self._worker and self._worker.isRunning():
            return

        df = AppState.get_dataframe()
        if df is None:
            self._status.setText("❌  No dataset loaded. Please import data first.")
            return

        date_col = self._date_combo.currentText()
        value_col = self._value_combo.currentText()
        if date_col == value_col:
            self._status.setText("❌  Date/time and forecast value columns must be different.")
            return

        self._run_btn.setEnabled(False)
        self._export_frame.setVisible(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Fitting Prophet forecast...")
        self._placeholder.setVisible(True)
        self._results_widget.setVisible(False)

        self._worker = ForecastWorker({
            "df": df,
            "date_col": date_col,
            "value_col": value_col,
            "periods": self._period_spin.value(),
            "holidays_country": self._selected_holiday_country(),
            "dataset_version": AppState.get_version(),
        })
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: TimeSeriesForecastResult, dataset_version: int):
        if dataset_version != AppState.get_version():
            self._worker = None
            self._refresh_columns()
            return

        self._result = result
        self._worker = None
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._export_frame.setVisible(True)

        self._status.setText(
            f"Done — fitted {result.historical_rows:,} points and forecast "
            f"{result.forecast_rows:,} periods."
        )
        self._lbl_rows.setText(f"Rows: {result.row_count:,}")
        self._lbl_history.setText(f"History: {result.historical_rows:,}")
        self._lbl_forecast.setText(f"Forecast: {result.forecast_rows:,}")
        self._lbl_freq.setText(f"Frequency: {result.frequency}")

        self._populate_stats(result)
        self._show_plot(result.plot_path)

        self._placeholder.setVisible(False)
        self._results_widget.setVisible(True)
        self._tabs.setCurrentIndex(0)

    def _populate_stats(self, result: TimeSeriesForecastResult):
        rows = [
            ("Observations", result.stats.get("observations")),
            ("Forecast periods", result.stats.get("forecast_periods")),
            ("Public holidays", result.holidays_country or "Not used"),
            ("MAE", result.stats.get("mae")),
            ("RMSE", result.stats.get("rmse")),
            ("MAPE", result.stats.get("mape")),
            ("R²", result.stats.get("r2")),
            ("Last actual date", result.stats.get("last_actual_date")),
        ]
        tbl = self._stats_table
        tbl.clear()
        tbl.setRowCount(len(rows))
        tbl.setColumnCount(2)
        tbl.setHorizontalHeaderLabels(["Statistic", "Value"])
        for r, (name, value) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(name))
            if isinstance(value, float):
                text = f"{value:,.4f}"
            elif value is None:
                text = "N/A"
            else:
                text = str(value)
            val_item = QTableWidgetItem(text)
            val_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tbl.setItem(r, 1, val_item)

    def _holiday_country_combo_enabled(self, checked: bool):
        self._holiday_country_combo.setEnabled(checked)

    def _selected_holiday_country(self) -> str | None:
        if not self._holidays_chk.isChecked():
            return None
        return self._holiday_country_combo.currentData()

    def _show_plot(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._plot_image.setText("Forecast plot could not be rendered.")
            return
        scaled = pix.scaled(
            self._plot_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._plot_image.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._result:
            self._show_plot(self._result.plot_path)

    def _on_error(self, msg: str):
        self._worker = None
        self._progress.setVisible(False)
        self._run_btn.setEnabled(AppState.has_data())
        self._status.setText(f"❌  {msg}")
        QMessageBox.critical(self, "Time Series Forecast Error", msg)

    def _export_predictions(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Predictions",
            "time_series_predictions.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            self._result.forecast.to_csv(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_plot(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Forecast Plot",
            "time_series_forecast.png",
            "PNG Files (*.png)",
        )
        if not path:
            return
        try:
            shutil.copyfile(self._result.plot_path, path)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
