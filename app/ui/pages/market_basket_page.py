"""Market basket analysis page — Apriori association rules via mlxtend."""

from __future__ import annotations

import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSplitter, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QProgressBar, QDoubleSpinBox, QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from app.core.market_basket import MarketBasketEngine, MarketBasketResult
from app.utils.app_state import AppState


DISPLAY_ROW_LIMIT = 500
CELL_TEXT_LIMIT = 120
INVOICE_ALIASES = ("invoice", "order_id", "order id", "transaction_id", "transaction id", "basket_id")
ITEM_ALIASES = ("description", "product", "item", "sku", "stockcode", "product_name")


class MarketBasketWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            cfg = self.config
            result = MarketBasketEngine.run(
                df=cfg.get("df"),
                transaction_col=cfg["transaction_col"],
                item_col=cfg["item_col"],
                min_support=cfg["min_support"],
                min_confidence=cfg["min_confidence"],
                filepath=cfg.get("filepath"),
                total_rows=cfg.get("total_rows"),
                encoding=cfg.get("encoding", "utf-8"),
                is_large=cfg.get("is_large", False),
                progress=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MarketBasketPage(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: MarketBasketWorker | None = None
        self._result: MarketBasketResult | None = None
        self._loaded_version: int | None = None
        self._build_ui()

    def showEvent(self, event):
        super().showEvent(event)
        if self._loaded_version != AppState.get_version():
            self._refresh_columns()
            self._update_run_state()

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

        title = QLabel("Market Basket Analysis")
        title.setObjectName("page_title")
        subtitle = QLabel(
            "Discover products frequently bought together using the Apriori algorithm. "
            "Each row should be one item in a transaction (invoice / basket)."
        )
        subtitle.setObjectName("page_subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._large_file_banner = QLabel(
            "Transaction and product limits are set automatically from your dataset size "
            "to stay within memory budget on typical 8 GB machines."
        )
        self._large_file_banner.setObjectName("status_label")
        self._large_file_banner.setWordWrap(True)
        self._large_file_banner.setVisible(False)
        layout.addWidget(self._large_file_banner)

        layout.addWidget(self._section("Transaction ID Column"))
        self._transaction_combo = QComboBox()
        self._transaction_combo.setToolTip("Invoice / order / basket identifier")
        layout.addWidget(self._transaction_combo)

        layout.addWidget(self._section("Item / Product Column"))
        self._item_combo = QComboBox()
        self._item_combo.setToolTip("Product name or SKU purchased in the transaction")
        layout.addWidget(self._item_combo)

        legend = QFrame()
        legend.setObjectName("rfm_legend")
        leg = QVBoxLayout(legend)
        leg.setContentsMargins(10, 8, 10, 8)
        leg.setSpacing(4)
        for line in [
            "Support — how often items appear together (e.g. 0.01 = 1% of baskets).",
            "Confidence — P(consequent | antecedent), e.g. 0.50 = 50%.",
            "Lift — how much more often items co-occur vs. chance (> 1 is positive).",
        ]:
            lbl = QLabel(line)
            lbl.setObjectName("rfm_legend_line")
            lbl.setWordWrap(True)
            leg.addWidget(lbl)
        layout.addWidget(legend)

        layout.addWidget(self._section("Minimum Support"))
        self._support_spin = QDoubleSpinBox()
        self._support_spin.setRange(0.001, 1.0)
        self._support_spin.setSingleStep(0.005)
        self._support_spin.setDecimals(4)
        self._support_spin.setValue(0.01)
        self._support_spin.setToolTip(
            "Fraction of transactions containing an itemset (e.g. 0.01 = 1%)"
        )
        layout.addWidget(self._support_spin)

        layout.addWidget(self._section("Minimum Rule Confidence"))
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.01, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(4)
        self._confidence_spin.setValue(0.50)
        self._confidence_spin.setToolTip(
            "P(consequent | antecedent) — e.g. 0.50 = 50% confidence"
        )
        layout.addWidget(self._confidence_spin)

        self._run_btn = QPushButton("▶  Run Analysis")
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
        exp = QVBoxLayout(self._export_frame)
        exp.setContentsMargins(0, 0, 0, 0)
        exp.setSpacing(8)
        exp.addWidget(self._section("Export"))

        self._export_rules_btn = QPushButton("⬇  Association Rules (.xlsx)")
        self._export_itemsets_btn = QPushButton("⬇  Frequent Itemsets (.xlsx)")
        for btn in [self._export_rules_btn, self._export_itemsets_btn]:
            btn.setObjectName("secondary_btn")
            exp.addWidget(btn)

        self._export_rules_btn.clicked.connect(lambda: self._export_table("rules"))
        self._export_itemsets_btn.clicked.connect(lambda: self._export_table("itemsets"))
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

        self._placeholder = QLabel(
            "Select transaction and item columns, set support/confidence thresholds, "
            "then click  ▶ Run Analysis."
        )
        self._placeholder.setObjectName("page_subtitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder, alignment=Qt.AlignmentFlag.AlignCenter)

        self._results_widget = QWidget()
        self._results_widget.setVisible(False)
        res = QVBoxLayout(self._results_widget)
        res.setContentsMargins(0, 0, 0, 0)
        res.setSpacing(16)

        self._summary_strip = QFrame()
        self._summary_strip.setObjectName("summary_bar")
        strip = QHBoxLayout(self._summary_strip)
        strip.setContentsMargins(16, 8, 16, 8)
        self._lbl_transactions = self._strip_lbl("Transactions", "—")
        self._lbl_items = self._strip_lbl("Unique items", "—")
        self._lbl_itemsets = self._strip_lbl("Frequent itemsets", "—")
        self._lbl_rules = self._strip_lbl("Rules", "—")
        for lbl in [
            self._lbl_transactions, self._lbl_items,
            self._lbl_itemsets, self._lbl_rules,
        ]:
            strip.addWidget(lbl)
        res.addWidget(self._summary_strip)

        self._tabs = QTabWidget()
        self._rules_table = self._make_table()
        self._itemsets_table = self._make_table()
        self._tabs.addTab(self._rules_table, "Association Rules")
        self._tabs.addTab(self._itemsets_table, "Frequent Itemsets")
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
        tbl.setMinimumHeight(320)
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
        cols = AppState.get_column_names()
        if not cols:
            return
        self._loaded_version = AppState.get_version()
        for combo in [self._transaction_combo, self._item_combo]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            combo.blockSignals(False)

        self._select_default_column(self._transaction_combo, INVOICE_ALIASES)
        self._select_default_column(self._item_combo, ITEM_ALIASES)

    @staticmethod
    def _select_default_column(combo: QComboBox, aliases: tuple[str, ...]):
        lower_map = {combo.itemText(i).lower(): i for i in range(combo.count())}
        for alias in aliases:
            if alias in lower_map:
                combo.setCurrentIndex(lower_map[alias])
                return
        for i in range(combo.count()):
            name = combo.itemText(i).lower()
            if any(alias in name for alias in aliases):
                combo.setCurrentIndex(i)
                return

    def _update_run_state(self):
        has_data = AppState.has_data()
        self._large_file_banner.setVisible(has_data)
        running = self._worker is not None and self._worker.isRunning()
        self._run_btn.setEnabled(has_data and not running)

    def _run(self):
        if self._worker and self._worker.isRunning():
            return

        df = AppState.get_dataframe()
        is_large = AppState.is_large()
        if df is None and not is_large:
            self._status.setText("❌  No dataset loaded. Please import data first.")
            return

        tx_col = self._transaction_combo.currentText()
        item_col = self._item_combo.currentText()

        if tx_col == item_col:
            self._status.setText("❌  Transaction and item columns must be different.")
            return

        slim_df = None
        if df is not None:
            slim_df = df[[tx_col, item_col]].copy()

        config = {
            "df": slim_df,
            "transaction_col": tx_col,
            "item_col": item_col,
            "min_support": self._support_spin.value(),
            "min_confidence": self._confidence_spin.value(),
            "is_large": is_large,
            "filepath": AppState.get_filepath() if is_large else None,
            "total_rows": AppState.get_row_count() if is_large else None,
            "encoding": AppState.get_load_encoding(),
        }

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Running market basket analysis…")
        self._results_widget.setVisible(False)
        self._placeholder.setVisible(True)
        self._export_frame.setVisible(False)

        self._worker = MarketBasketWorker(config)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: MarketBasketResult):
        self._result = result
        self._worker = None
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._update_run_state()

        sample_note = ""
        if result.sampled:
            parts = []
            if result.transactions_sampled or result.cap_transactions < result.source_transactions:
                parts.append(
                    f"{result.transaction_count:,} of {result.source_transactions:,} transactions"
                )
            if result.items_filtered or result.cap_unique_items < result.source_unique_items:
                parts.append(
                    f"top {result.cap_unique_items:,} of {result.source_unique_items:,} products"
                )
            if result.sampled and not parts:
                parts.append(f"{result.rows_used:,} line items sampled")
            if parts:
                sample_note = f" ({'; '.join(parts)})"
        self._status.setText(
            f"✅  Done — {len(result.rules)} rules from "
            f"{result.transaction_count:,} transactions{sample_note}."
        )

        self._lbl_transactions.setText(f"Transactions: {result.transaction_count:,}")
        self._lbl_items.setText(f"Unique items: {result.unique_items:,}")
        self._lbl_itemsets.setText(f"Frequent itemsets: {len(result.frequent_itemsets):,}")
        self._lbl_rules.setText(f"Rules: {len(result.rules):,}")

        rules_view = result.rules.head(DISPLAY_ROW_LIMIT)
        itemsets_view = result.frequent_itemsets.head(DISPLAY_ROW_LIMIT)
        self._populate_table(self._rules_table, rules_view)
        self._populate_table(self._itemsets_table, itemsets_view)

        if len(result.rules) > DISPLAY_ROW_LIMIT:
            self._tabs.setTabText(
                0,
                f"Association Rules (top {DISPLAY_ROW_LIMIT:,} of {len(result.rules):,})",
            )
        else:
            self._tabs.setTabText(0, "Association Rules")

        if len(result.frequent_itemsets) > DISPLAY_ROW_LIMIT:
            self._tabs.setTabText(
                1,
                f"Frequent Itemsets (top {DISPLAY_ROW_LIMIT:,} of "
                f"{len(result.frequent_itemsets):,})",
            )
        else:
            self._tabs.setTabText(1, "Frequent Itemsets")

        self._placeholder.setVisible(False)
        self._results_widget.setVisible(True)
        self._export_frame.setVisible(True)

    def _populate_table(self, table: QTableWidget, frame: pd.DataFrame):
        table.clear()
        if frame.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return

        pct_cols = {
            "support", "confidence", "lift",
            "antecedent support", "consequent support",
        }
        numeric_cols = {
            col for col in frame.columns
            if frame[col].dtype.kind in "fi" or col in pct_cols or col.endswith("_support")
        }

        display = frame.copy()
        for col in display.columns:
            if col in pct_cols or col.endswith("_support"):
                display[col] = display[col].map(
                    lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else "—"
                )
            elif col in numeric_cols:
                display[col] = display[col].map(
                    lambda v: f"{float(v):,.4f}" if pd.notna(v) else "—"
                )

        table.setRowCount(len(display))
        table.setColumnCount(len(display.columns))
        table.setHorizontalHeaderLabels([str(c) for c in display.columns])

        for r in range(len(display)):
            for c, col in enumerate(display.columns):
                val = display.iloc[r, c]
                text = str(val)
                if len(text) > CELL_TEXT_LIMIT:
                    text = text[: CELL_TEXT_LIMIT - 1] + "…"
                item = QTableWidgetItem(text)
                if col in numeric_cols or col in pct_cols:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                else:
                    item.setToolTip(str(val))
                table.setItem(r, c, item)

    def _on_error(self, msg: str):
        self._worker = None
        self._progress.setVisible(False)
        self._update_run_state()
        self._status.setText(f"❌  {msg}")
        QMessageBox.critical(self, "Market Basket Analysis Error", msg)

    def _export_table(self, kind: str):
        if not self._result:
            return
        frame = self._result.rules if kind == "rules" else self._result.frequent_itemsets
        default = f"market_basket_{kind}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", default, "Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            frame.to_excel(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
