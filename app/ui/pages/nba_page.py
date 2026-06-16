"""Next Best Action page — rule-based recommendations."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from app.core.nba import NextBestActionEngine, NBAResult
from app.utils.app_state import AppState


class NBAWorker(QThread):
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = NextBestActionEngine.recommend(
                df=self.config["df"],
                customer_id_col=self.config.get("customer_id_col"),
                limit=self.config.get("limit", 100),
                segmentation_result=self.config.get("segmentation_result"),
            )
            self.finished.emit(result, self.config["dataset_version"])
        except Exception as exc:
            self.error.emit(str(exc))


class NextBestActionPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: NBAWorker | None = None
        self._result: NBAResult | None = None
        self._loaded_version: int | None = None
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        if self._loaded_version != AppState.get_version():
            self._refresh_columns()

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(16)

        title = QLabel("Next Best Action")
        title.setObjectName("page_title")
        subtitle = QLabel("Rule-based customer action recommendations.")
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("summary_bar")
        ctl = QHBoxLayout(controls)
        ctl.setContentsMargins(16, 10, 16, 10)
        ctl.setSpacing(12)

        ctl.addWidget(self._label("Customer ID"))
        self._id_combo = QComboBox()
        self._id_combo.setMinimumWidth(180)
        ctl.addWidget(self._id_combo)

        ctl.addWidget(self._label("Max rows"))
        self._limit_spin = QSpinBox()
        self._limit_spin.setRange(10, 1000)
        self._limit_spin.setValue(100)
        self._limit_spin.setSingleStep(25)
        self._limit_spin.setFixedWidth(90)
        ctl.addWidget(self._limit_spin)

        ctl.addStretch()
        self._run_btn = QPushButton("Generate Actions")
        self._run_btn.setObjectName("primary_btn")
        self._run_btn.clicked.connect(self._run_nba)
        ctl.addWidget(self._run_btn)
        layout.addWidget(controls)

        self._progress = QProgressBar()
        self._progress.setObjectName("load_progress")
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("Import a dataset to generate recommendations.")
        self._status.setObjectName("status_label")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        self._summary_frame = QFrame()
        self._summary_frame.setObjectName("summary_bar")
        self._summary_frame.setVisible(False)
        summary = QHBoxLayout(self._summary_frame)
        summary.setContentsMargins(16, 8, 16, 8)
        self._summary_labels = {}
        for key in ["Recommendations", "Top action", "High confidence", "Segments"]:
            lbl = QLabel(f"{key}: —")
            lbl.setObjectName("summary_item")
            self._summary_labels[key] = lbl
            summary.addWidget(lbl)
        layout.addWidget(self._summary_frame)

        self._signals_label = QLabel("")
        self._signals_label.setObjectName("field_label")
        self._signals_label.setWordWrap(True)
        self._signals_label.setVisible(False)
        layout.addWidget(self._signals_label)

        self._dist_label = QLabel("")
        self._dist_label.setObjectName("field_label")
        self._dist_label.setWordWrap(True)
        self._dist_label.setVisible(False)
        layout.addWidget(self._dist_label)

        self._table = QTableWidget()
        self._table.setObjectName("preview_table")
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setVisible(False)
        layout.addWidget(self._table, 1)

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("field_label")
        return lbl

    # ------------------------------------------------------------------ #
    #  Data / execution                                                    #
    # ------------------------------------------------------------------ #

    def _refresh_columns(self):
        self._loaded_version = AppState.get_version()
        self._id_combo.clear()

        df = AppState.get_dataframe()
        self._run_btn.setEnabled(df is not None)
        if df is None:
            self._status.setText("Import a dataset to generate recommendations.")
            self._summary_frame.setVisible(False)
            self._signals_label.setVisible(False)
            self._dist_label.setVisible(False)
            self._table.setVisible(False)
            return

        detected = NextBestActionEngine.detect_columns(df)
        columns = list(df.columns)
        preferred = detected.get("customer_id")
        if preferred and preferred in columns:
            columns.remove(preferred)
            columns.insert(0, preferred)

        self._id_combo.addItems(columns)
        self._status.setText(
            f"Ready — {len(df):,} rows available"
            + (" (preview sample in DuckDB mode)." if AppState.is_large() else ".")
        )
        self._show_detected_columns(detected)

    def _run_nba(self):
        df = AppState.get_dataframe()
        if df is None:
            self._on_error("No dataset loaded. Please import data first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status.setText("Generating recommendations...")
        self._summary_frame.setVisible(False)
        self._dist_label.setVisible(False)
        self._table.setVisible(False)

        self._worker = NBAWorker({
            "df": df,
            "customer_id_col": self._id_combo.currentText() or None,
            "limit": self._limit_spin.value(),
            "segmentation_result": AppState.get_segmentation_result(),
            "dataset_version": AppState.get_version(),
        })
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: NBAResult, dataset_version: int):
        if dataset_version != AppState.get_version():
            self._worker = None
            self._refresh_columns()
            return

        self._result = result
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._worker = None

        recs = result.recommendations
        self._status.setText(f"Done — generated {len(recs):,} recommendations.")
        self._update_summary(result)
        self._show_detected_columns(result.detected_columns)
        self._populate_table(recs)

    def _update_summary(self, result: NBAResult):
        recs = result.recommendations
        top_action = result.action_distribution.index[0] if not result.action_distribution.empty else "—"
        high_conf = int((recs["confidence"] == "High").sum()) if "confidence" in recs else 0
        has_segments = "segment" in recs.columns

        self._summary_labels["Recommendations"].setText(f"Recommendations: {len(recs):,}")
        self._summary_labels["Top action"].setText(f"Top action: {top_action}")
        self._summary_labels["High confidence"].setText(f"High confidence: {high_conf:,}")
        self._summary_labels["Segments"].setText("Segments: included" if has_segments else "Segments: not used")
        self._summary_frame.setVisible(True)

        dist_text = "Action mix: " + " | ".join(
            f"{action}: {count}" for action, count in result.action_distribution.items()
        )
        self._dist_label.setText(dist_text)
        self._dist_label.setVisible(True)

    def _show_detected_columns(self, detected: dict[str, str | None]):
        useful = [
            f"{role}: {col}"
            for role, col in detected.items()
            if col is not None and role != "customer_id"
        ]
        text = "Detected signals: " + (", ".join(useful) if useful else "none yet")
        self._signals_label.setText(text)
        self._signals_label.setVisible(True)

    def _populate_table(self, recs):
        tbl = self._table
        tbl.clear()
        columns = list(recs.columns)
        tbl.setRowCount(len(recs))
        tbl.setColumnCount(len(columns))
        tbl.setHorizontalHeaderLabels([self._title(col) for col in columns])

        for r, (_, row) in enumerate(recs.iterrows()):
            for c, col in enumerate(columns):
                value = row[col]
                text = f"{value:,.2f}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                if col in {"score", "expected_value"}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == "confidence":
                    item.setForeground(QColor(self._confidence_color(str(value))))
                tbl.setItem(r, c, item)

        tbl.resizeColumnsToContents()
        tbl.setVisible(True)

    def _on_error(self, msg: str):
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._run_btn.setEnabled(AppState.get_dataframe() is not None)
        self._worker = None
        self._status.setText(f"❌  {msg}")

    def _title(self, col: str) -> str:
        return col.replace("_", " ").title()

    def _confidence_color(self, confidence: str) -> str:
        if confidence == "High":
            return "#3DDC84"
        if confidence == "Medium":
            return "#FFD166"
        return "#FF7A45"
