"""Churn prediction page — CatBoost model with SHAP explainability."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QListWidget, QAbstractItemView, QSplitter,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from app.core.churn_model import ChurnModelEngine, ChurnModelResult
from app.utils.app_state import AppState


class ChurnWorker(QThread):
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = ChurnModelEngine.fit(
                df=self.config.get("df"),
                target_col=self.config["target_col"],
                predictor_cols=self.config["predictor_cols"],
                filepath=self.config.get("filepath"),
                total_rows=self.config.get("total_rows"),
                encoding=self.config.get("encoding", "utf-8"),
                all_columns=self.config.get("all_columns"),
                progress=self.progress.emit,
            )
            self.finished.emit(result, self.config["dataset_version"])
        except Exception as exc:
            self.error.emit(str(exc))


class ChurnPredictionPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: ChurnWorker | None = None
        self._result: ChurnModelResult | None = None
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

        title = QLabel("Churn Prediction")
        title.setObjectName("page_title")
        subtitle = QLabel("Fit a CatBoost classifier and explain churn drivers.")
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._large_file_banner = QLabel(
            "Datasets over 500,000 rows are modeled on a random sample."
        )
        self._large_file_banner.setObjectName("status_label")
        self._large_file_banner.setWordWrap(True)
        self._large_file_banner.setVisible(False)
        layout.addWidget(self._large_file_banner)

        layout.addWidget(self._section("Target Variable"))
        self._target_combo = QComboBox()
        self._target_combo.currentIndexChanged.connect(self._select_default_predictors)
        layout.addWidget(self._target_combo)

        layout.addWidget(self._section("Predictors"))
        self._predictor_list = QListWidget()
        self._predictor_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._predictor_list.setMinimumHeight(280)
        layout.addWidget(self._predictor_list)

        self._run_btn = QPushButton("Fit Churn Model")
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

        self._export_btn = QPushButton("Save Scored Dataset (.csv)")
        self._export_btn.setObjectName("secondary_btn")
        self._export_btn.setVisible(False)
        self._export_btn.clicked.connect(self._export_scores)
        layout.addWidget(self._export_btn)

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

        self._placeholder = QLabel("Select a binary target and predictors, then fit the model.")
        self._placeholder.setObjectName("page_subtitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self._lbl_sample = self._strip_lbl("Sample", "—")
        self._lbl_auc = self._strip_lbl("AUC", "—")
        self._lbl_acc = self._strip_lbl("Accuracy", "—")
        for lbl in [self._lbl_rows, self._lbl_sample, self._lbl_auc, self._lbl_acc]:
            strip.addWidget(lbl)
        res.addWidget(self._summary_strip)

        imp_lbl = QLabel("Feature Importance")
        imp_lbl.setObjectName("section_label")
        res.addWidget(imp_lbl)

        self._importance_table = QTableWidget()
        self._importance_table.setObjectName("preview_table")
        self._importance_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._importance_table.setAlternatingRowColors(True)
        self._importance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._importance_table.setFixedHeight(230)
        res.addWidget(self._importance_table)

        shap_lbl = QLabel("SHAP Beeswarm")
        shap_lbl.setObjectName("section_label")
        res.addWidget(shap_lbl)

        self._shap_image = QLabel("")
        self._shap_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._shap_image.setMinimumHeight(360)
        self._shap_image.setObjectName("status_label")
        res.addWidget(self._shap_image, 1)

        layout.addWidget(self._results_widget)
        return panel

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _strip_lbl(self, key: str, value: str) -> QLabel:
        lbl = QLabel(f"{key}: {value}")
        lbl.setObjectName("summary_item")
        return lbl

    # ------------------------------------------------------------------ #
    #  Data / execution                                                    #
    # ------------------------------------------------------------------ #

    def _refresh_columns(self):
        self._loaded_version = AppState.get_version()
        cols = AppState.get_column_names()
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._predictor_list.clear()

        if not cols:
            self._run_btn.setEnabled(False)
            self._status.setText("Import a dataset to fit a churn model.")
            self._target_combo.blockSignals(False)
            return

        self._target_combo.addItems(cols)
        for col in cols:
            self._predictor_list.addItem(col)

        self._target_combo.blockSignals(False)
        self._choose_likely_target(cols)
        self._select_default_predictors()
        self._run_btn.setEnabled(True)

        row_count = AppState.get_row_count()
        self._large_file_banner.setVisible(row_count > 500_000)
        self._status.setText(f"Ready — {row_count:,} rows detected.")

    def _choose_likely_target(self, cols: list[str]):
        lowered = {c.lower(): c for c in cols}
        for key in ["churn", "churned", "is_churn", "churn_flag", "target"]:
            if key in lowered:
                self._target_combo.setCurrentText(lowered[key])
                return
        for col in cols:
            if "churn" in col.lower():
                self._target_combo.setCurrentText(col)
                return

    def _select_default_predictors(self):
        target = self._target_combo.currentText()
        for i in range(self._predictor_list.count()):
            item = self._predictor_list.item(i)
            item.setSelected(item.text() != target)

    def _run(self):
        if self._worker and self._worker.isRunning():
            return

        target = self._target_combo.currentText()
        predictors = [
            self._predictor_list.item(i).text()
            for i in range(self._predictor_list.count())
            if self._predictor_list.item(i).isSelected()
        ]

        if not target:
            self._status.setText("❌  Select a target variable.")
            return
        if not predictors:
            self._status.setText("❌  Select at least one predictor.")
            return

        is_duckdb = AppState.is_large()
        filepath = AppState.get_filepath() if is_duckdb else None
        if is_duckdb and not filepath:
            self._status.setText("❌  Dataset path not available for sampling.")
            return

        self._run_btn.setEnabled(False)
        self._export_btn.setVisible(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Fitting CatBoost churn model...")
        self._placeholder.setVisible(True)
        self._results_widget.setVisible(False)

        self._worker = ChurnWorker({
            "df": AppState.get_dataframe(),
            "target_col": target,
            "predictor_cols": predictors,
            "filepath": filepath,
            "total_rows": AppState.get_row_count(),
            "encoding": AppState.get_load_encoding(),
            "all_columns": AppState.get_column_names(),
            "dataset_version": AppState.get_version(),
        })
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: ChurnModelResult, dataset_version: int):
        if dataset_version != AppState.get_version():
            self._worker = None
            self._refresh_columns()
            return

        self._result = result
        self._worker = None
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._export_btn.setVisible(True)

        self._status.setText(
            f"Done — fitted on {result.train_rows:,} rows and scored {result.scored_rows:,} rows."
        )
        self._lbl_rows.setText(f"Rows: {result.row_count:,}")
        sample_text = f"{result.train_rows:,}" + (" sampled" if result.sampled else "")
        self._lbl_sample.setText(f"Sample: {sample_text}")
        self._lbl_auc.setText(f"AUC: {result.auc:.3f}" if result.auc is not None else "AUC: N/A")
        self._lbl_acc.setText(
            f"Accuracy: {result.accuracy:.3f}" if result.accuracy is not None else "Accuracy: N/A"
        )
        self._populate_importance(result)
        self._show_shap_plot(result.shap_plot_path)

        self._placeholder.setVisible(False)
        self._results_widget.setVisible(True)

    def _populate_importance(self, result: ChurnModelResult):
        tbl = self._importance_table
        imp = result.feature_importance.head(50)
        tbl.clear()
        tbl.setRowCount(len(imp))
        tbl.setColumnCount(2)
        tbl.setHorizontalHeaderLabels(["Feature", "Importance"])
        for r, (_, row) in enumerate(imp.iterrows()):
            tbl.setItem(r, 0, QTableWidgetItem(str(row["feature"])))
            val = QTableWidgetItem(f"{row['importance']:,.4f}")
            val.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tbl.setItem(r, 1, val)

    def _show_shap_plot(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._shap_image.setText("SHAP plot could not be rendered.")
            return
        scaled = pix.scaled(
            self._shap_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._shap_image.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._result:
            self._show_shap_plot(self._result.shap_plot_path)

    def _on_error(self, msg: str):
        self._worker = None
        self._progress.setVisible(False)
        self._run_btn.setEnabled(AppState.has_data())
        self._status.setText(f"❌  {msg}")
        QMessageBox.critical(self, "Churn Model Error", msg)

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def _export_scores(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scored Dataset",
            "churn_scored_dataset.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            self._result.scored_data.to_csv(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
