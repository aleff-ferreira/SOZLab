"""SOZLab GUI entrypoint."""
from __future__ import annotations

import json
import threading
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters

from engine.analysis import SOZAnalysisEngine, load_project_json, write_project_json
from engine.export import export_results
from engine.extract import select_frames, write_extracted_trajectory
from engine.models import (
    ProjectConfig,
    SelectionSpec,
    SOZDefinition,
    SOZNode,
    SolventConfig,
    BridgeConfig,
    ResidueHydrationConfig,
    default_project,
)
from engine.preflight import run_preflight
from engine.serialization import to_jsonable
from engine.logging_utils import setup_run_logger
from engine.resolver import resolve_selection
from engine.solvent import build_solvent
from engine.soz_eval import EvaluationContext, evaluate_node
from engine.units import to_nm
from engine.report import generate_report


@dataclass
class ProjectState:
    project: Optional[ProjectConfig] = None
    path: Optional[Path] = None


class AnalysisWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, project: ProjectConfig, logger: logging.Logger | None = None):
        super().__init__()
        self.project = project
        self.cancel_event = threading.Event()
        self.logger = logger

    def cancel(self) -> None:
        self.cancel_event.set()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            engine = SOZAnalysisEngine(self.project)
            result = engine.run(
                progress=self._progress,
                cancel_flag=self.cancel_event.is_set,
                logger=self.logger,
            )
            self.finished.emit(result)
        except Exception as exc:
            if self.logger:
                self.logger.exception("Analysis failed")
            self.failed.emit(str(exc))

    def _progress(self, current: int, total: int, message: str) -> None:
        self.progress.emit(current, total, message)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SOZLab")
        self.resize(1400, 900)
        self.state = ProjectState()
        self.current_result = None
        self.analysis_thread = None
        self.analysis_worker = None
        self._table_proxies = {}
        self.run_logger: logging.Logger | None = None
        self.log_path: str | None = None
        self.run_project: ProjectConfig | None = None
        self._ui_scale = 1.0
        self._plot_line_width = 2.0
        self._plot_marker_size = 4.0
        self._timeline_event_cache: dict[str, object] = {}
        self._timeline_stats_cache: dict[str, object] = {}
        self._timeline_event_hover_index: int | None = None
        self._timeline_event_warning_shown: set[tuple[str, str]] = set()
        self._timeline_update_pending = False
        self._hist_update_pending = False
        self._matrix_update_pending = False
        self._event_update_pending = False
        self._extract_output_linked = True
        self._preflight_report = None
        self._preflight_universe = None
        self._preflight_key = None
        self._seed_validation_timer = QtCore.QTimer(self)
        self._seed_validation_timer.setSingleShot(True)
        self._seed_validation_timer.timeout.connect(self._run_seed_validation)
        self._seed_validation_cache: dict[str, dict] = {}
        self._analysis_running = False
        self._wizard_snapshot: dict | None = None
        self._time_window: tuple[float, float] | None = None
        self._selected_solvent_id: str | None = None
        self.timeline_region = None
        self.matrix_highlight_line = None
        self._log_raw_text = ""
        self._last_dt = None
        self.settings = QtCore.QSettings("SOZLab", "SOZLab")
        try:
            self._user_scale = float(self.settings.value("ui_scale", 1.0))
        except Exception:
            self._user_scale = 1.0
        theme_value = str(self.settings.value("ui_theme", "light")).lower()
        self._theme_mode = "dark" if theme_value == "dark" else "light"
        self._presentation_scale = 1.0

        self._base_font = QtWidgets.QApplication.instance().font()
        self._apply_plot_theme()
        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("RootContainer")
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.header_bar = self._build_header()
        root_layout.addWidget(self.header_bar)

        self.main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.project_panel = self._build_project_panel()
        self.page_stack = QtWidgets.QStackedWidget()
        self.inspector_stack = QtWidgets.QStackedWidget()
        self.project_scroll = QtWidgets.QScrollArea()
        self.project_scroll.setWidgetResizable(True)
        self.project_scroll.setWidget(self.project_panel)
        self.inspector_scroll = QtWidgets.QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setWidget(self.inspector_stack)
        for widget in (self.project_scroll, self.page_stack, self.inspector_scroll):
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        self.main_split.addWidget(self.project_scroll)
        self.main_split.addWidget(self.page_stack)
        self.main_split.addWidget(self.inspector_scroll)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setCollapsible(0, False)
        self.main_split.setCollapsible(1, False)
        self.main_split.setCollapsible(2, False)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setStretchFactor(2, 0)
        self.main_split.setSizes([320, 980, 0])
        self.main_split.setOpaqueResize(True)

        self.console_panel = self._build_console_panel()
        self.console_panel.setMaximumHeight(220)

        self.vertical_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vertical_split.addWidget(self.main_split)
        self.vertical_split.addWidget(self.console_panel)
        self.vertical_split.setStretchFactor(0, 1)
        self.vertical_split.setStretchFactor(1, 0)
        self.vertical_split.setChildrenCollapsible(False)
        self.vertical_split.setOpaqueResize(True)
        self.console_panel.setVisible(False)

        root_layout.addWidget(self.vertical_split, 1)
        self.setCentralWidget(root)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.run_progress = QtWidgets.QProgressBar()
        self.run_progress.setVisible(False)
        self.run_progress.setMaximumWidth(240)
        self.run_progress.setMinimumWidth(160)
        self.run_progress.setFormat("Run: %p%")
        self.status_bar.addPermanentWidget(self.run_progress)
        self._build_pages()
        self._build_inspectors()
        self._apply_scale()
        self._apply_initial_layout()
        self._set_active_step(0)
        self._set_run_ui_state(False)
        try:
            self._wizard_snapshot = self._wizard_state()
        except Exception:
            self._wizard_snapshot = None

    def _apply_ui_style(self, scale: float = 1.0) -> None:
        pad = max(6, int(6 * scale))
        pad_sm = max(4, int(4 * scale))
        radius = max(8, int(8 * scale))
        button_pad_y = max(6, int(6 * scale))
        button_pad_x = max(14, int(14 * scale))
        tooltip_pad = max(6, int(6 * scale))
        handle = max(6, int(6 * scale))
        scrollbar = max(10, int(10 * scale))
        scrollbar_margin = max(2, int(2 * scale))
        scrollbar_radius = max(4, int(4 * scale))
        tokens = self._get_theme_tokens()
        combo_drop = max(22, int(22 * scale))
        chevron_size = max(10, int(10 * scale))
        theme_mode = getattr(self, "_theme_mode", "light")
        arrow_down_file = "chevron-down-dark.png" if theme_mode == "dark" else "chevron-down-light.png"
        arrow_up_file = "chevron-up-dark.png" if theme_mode == "dark" else "chevron-up-light.png"
        assets_dir = Path(__file__).resolve().parent / "assets"
        arrow_down_path = (assets_dir / arrow_down_file).as_posix()
        arrow_up_path = (assets_dir / arrow_up_file).as_posix()
        combo_arrow = arrow_down_path
        style = """
            QMainWindow {{
                background: {base};
            }}
            QWidget {{
                color: {text};
                font-family: "Space Grotesk", "Noto Sans", "DejaVu Sans";
                font-size: 13px;
                background: {base};
            }}
            #RootContainer {{
                background: {base};
            }}
            #HeaderBar {{
                background: {base};
            }}
            #AppTitle {{
                font-size: 16px;
                font-weight: 700;
                color: {text};
            }}
            QSplitter::handle {{
                background: {border};
            }}
            QSplitter::handle:hover {{
                background: {border_hover};
            }}
            QSplitter::handle:pressed {{
                background: {accent};
            }}
            QSplitter {{
                background: {base};
            }}
            QSplitter::handle:horizontal {{
                width: {handle}px;
                margin: 0;
            }}
            QSplitter::handle:vertical {{
                height: {handle}px;
                margin: 0;
            }}
            QGroupBox {{
                border: 1px solid {border};
                border-radius: {radius}px;
                margin-top: 12px;
                background: {surface};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget {{
                background: {surface};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: {scrollbar}px;
                margin: {scrollbar_margin}px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {border_hover};
                border-radius: {scrollbar_radius}px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {accent};
            }}
            QScrollBar::handle:vertical:pressed {{
                background: {accent};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: {scrollbar}px;
                margin: {scrollbar_margin}px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {border_hover};
                border-radius: {scrollbar_radius}px;
                min-width: 24px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {accent};
            }}
            QScrollBar::handle:horizontal:pressed {{
                background: {accent};
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                width: 0px;
                background: transparent;
            }}
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: transparent;
            }}
            QStackedWidget {{
                background: {surface};
            }}
            QFrame {{
                border: 0;
                background: transparent;
            }}
            QAbstractItemView {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {selection};
                selection-color: {selection_text};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: {combo_drop}px;
                border-left: 1px solid {border};
                border-top-right-radius: {radius}px;
                border-bottom-right-radius: {radius}px;
                background: {panel};
            }}
            QComboBox::down-arrow {{
                image: url("{combo_arrow}");
                width: {chevron_size}px;
                height: {chevron_size}px;
            }}
            QComboBox QAbstractItemView {{
                background: {panel};
                color: {text};
            }}
            QMenu {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
            }}
            QMenu::item:selected {{
                background: {accent_soft};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 {pad_sm}px;
                color: {text_muted};
                font-weight: 600;
            }}
            QLabel {{
                color: {text};
            }}
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {panel};
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: {pad_sm}px {pad}px;
                selection-background-color: {selection};
                selection-color: {selection_text};
            }}
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {accent};
            }}
            QComboBox {{
                padding-right: {combo_drop}px;
            }}
            QSpinBox, QDoubleSpinBox {{
                padding-right: {combo_drop}px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: {combo_drop}px;
                border-left: 1px solid {border};
                border-top-right-radius: {radius}px;
                background: {panel};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
                background: {button_hover};
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
                background: {accent_soft};
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: {combo_drop}px;
                border-left: 1px solid {border};
                border-top: 1px solid {border};
                border-bottom-right-radius: {radius}px;
                background: {panel};
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: {button_hover};
            }}
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background: {accent_soft};
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url("{spin_up_arrow}");
                width: {chevron_size}px;
                height: {chevron_size}px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url("{spin_down_arrow}");
                width: {chevron_size}px;
                height: {chevron_size}px;
            }}
            QPushButton {{
                background: {button_bg};
                color: {text};
                border: 1px solid {border};
                padding: {button_pad_y}px {button_pad_x}px;
                border-radius: {radius}px;
            }}
            QPushButton:hover {{
                background: {button_hover};
            }}
            QPushButton:pressed {{
                background: {accent};
                color: {selection_text};
            }}
            QPushButton#PrimaryButton {{
                background: {accent};
                color: {selection_text};
                border: none;
            }}
            QPushButton#PrimaryButton:hover {{
                background: {accent};
            }}
            QPushButton:disabled {{
                background: {panel};
                color: {text_muted};
                border-color: {panel};
            }}
            QToolButton {{
                background: {panel};
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: {pad_sm}px;
            }}
            QToolButton:checked {{
                border: 1px solid {accent};
                background: {accent_soft};
            }}
            QToolTip {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                padding: {tooltip_pad}px;
                border-radius: {radius}px;
            }}
            QTableView {{
                background: {surface};
                border: 1px solid {border};
                border-radius: {radius}px;
                gridline-color: {grid};
                alternate-background-color: {panel};
            }}
            QHeaderView::section {{
                background: {panel};
                padding: {pad}px;
                border: 1px solid {border};
                font-weight: 600;
            }}
            QTabWidget::pane {{
                border: 0;
                border-radius: {radius}px;
                background: {surface};
            }}
            QTabBar::tab {{
                background: {tab_bg};
                border: 1px solid {border};
                padding: {pad_sm}px {pad}px;
                margin-right: 2px;
                border-top-left-radius: {radius}px;
                border-top-right-radius: {radius}px;
                color: {text_muted};
            }}
            QTabBar::tab:selected {{
                background: {surface};
                color: {text};
                border-color: {accent};
            }}
            QStatusBar {{
                background: {base};
                border-top: 1px solid {border};
                color: {text_muted};
            }}
            QMainWindow::separator {{
                background: {border};
                width: {handle}px;
                height: {handle}px;
            }}
        """.format(
            radius=radius,
            button_pad_y=button_pad_y,
            button_pad_x=button_pad_x,
            pad_sm=pad_sm,
            pad=pad,
            tooltip_pad=tooltip_pad,
            handle=handle,
            scrollbar=scrollbar,
            scrollbar_margin=scrollbar_margin,
            scrollbar_radius=scrollbar_radius,
            combo_drop=combo_drop,
            combo_arrow=combo_arrow,
            chevron_size=chevron_size,
            spin_up_arrow=arrow_up_path,
            spin_down_arrow=arrow_down_path,
            base=tokens["base"],
            surface=tokens["surface"],
            panel=tokens["panel"],
            text=tokens["text"],
            text_muted=tokens["text_muted"],
            border=tokens["border"],
            border_hover=tokens["border_hover"],
            accent=tokens["accent"],
            accent_soft=tokens["accent_soft"],
            selection=tokens["selection"],
            selection_text=tokens["selection_text"],
            tab_bg=tokens["tab_bg"],
            button_bg=tokens["button_bg"],
            button_hover=tokens["button_hover"],
            grid=tokens["grid"],
        )
        self.setStyleSheet(style)

    def _build_header(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        bar.setObjectName("HeaderBar")
        self.header_layout = layout

        title = QtWidgets.QLabel("SOZLab")
        title.setObjectName("AppTitle")
        layout.addWidget(title)

        self.stepper_group = QtWidgets.QButtonGroup(self)
        self.stepper_group.setExclusive(True)
        self.stepper_buttons = []
        for idx, name in enumerate(["Project", "Define", "QC", "Explore", "Export"]):
            btn = QtWidgets.QToolButton()
            btn.setText(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, i=idx: self._set_active_step(i))
            btn.setToolTip(f"Go to {name}")
            self.stepper_group.addButton(btn, idx)
            self.stepper_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch(1)

        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self._load_project)
        self.load_btn.setToolTip("Load a project configuration (JSON/YAML).")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self._save_project)
        self.save_btn.setToolTip("Save the current project configuration.")
        self.new_soz_btn = QtWidgets.QPushButton("New SOZ")
        self.new_soz_btn.clicked.connect(self._add_soz_from_builder)
        self.new_soz_btn.setToolTip("Create or update a SOZ definition from the Wizard.")
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setToolTip("Run analysis with the current settings.")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        self.cancel_btn.setToolTip("Cancel the running analysis.")
        self.quick_btn = QtWidgets.QPushButton("Quick")
        self.quick_btn.clicked.connect(self._quick_subset)
        self.quick_btn.setToolTip("Quick subset preview.")
        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setToolTip("Export CSV/JSON outputs to the output directory.")
        self.report_btn = QtWidgets.QPushButton("Export Report")
        self.report_btn.clicked.connect(self._export_report)
        self.report_btn.setToolTip("Generate a report (HTML/Markdown).")
        self.console_toggle_btn = QtWidgets.QToolButton()
        self.console_toggle_btn.setText("Console")
        self.console_toggle_btn.setCheckable(True)
        self.console_toggle_btn.toggled.connect(self._toggle_console)
        self.drawer_toggle_btn = QtWidgets.QToolButton()
        self.drawer_toggle_btn.setText("Drawer")
        self.drawer_toggle_btn.setCheckable(True)
        self.drawer_toggle_btn.setChecked(True)
        self.drawer_toggle_btn.toggled.connect(self._toggle_drawer)
        self.inspector_toggle_btn = QtWidgets.QToolButton()
        self.inspector_toggle_btn.setText("Inspector")
        self.inspector_toggle_btn.setCheckable(True)
        self.inspector_toggle_btn.setChecked(False)
        self.inspector_toggle_btn.toggled.connect(self._toggle_inspector)
        self.theme_label = QtWidgets.QLabel("Theme")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.theme_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.theme_combo.setMinimumContentsLength(4)
        self.scale_label = QtWidgets.QLabel("Scale")
        self.scale_combo = QtWidgets.QComboBox()
        self._scale_levels = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75]
        for level in self._scale_levels:
            self.scale_combo.addItem(f"{int(level * 100)}%", level)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        self.scale_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.scale_combo.setMinimumContentsLength(4)
        self.scale_combo.setToolTip("Scales the entire interface (fonts, buttons, spacing).")
        self.scale_label.setToolTip("Interface scale")
        self.theme_combo.setToolTip("Switch between light and dark themes.")
        self.theme_label.setToolTip("Theme")
        self.presentation_toggle = QtWidgets.QCheckBox("Presentation")
        self.presentation_toggle.toggled.connect(self._set_presentation_mode)

        for btn in (
            self.load_btn,
            self.save_btn,
            self.new_soz_btn,
            self.quick_btn,
            self.run_btn,
            self.cancel_btn,
            self.export_btn,
            self.report_btn,
        ):
            layout.addWidget(btn)
        layout.addWidget(self.console_toggle_btn)
        layout.addWidget(self.drawer_toggle_btn)
        layout.addWidget(self.inspector_toggle_btn)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)
        layout.addWidget(self.scale_label)
        layout.addWidget(self.scale_combo)
        layout.addWidget(self.presentation_toggle)
        self._sync_theme_combo()
        self._sync_scale_combo()
        return bar

    def _build_console_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Diagnostics Console"))
        header.addStretch(1)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_log_view)
        header.addWidget(refresh_btn)
        layout.addLayout(header)
        self.console_text = QtWidgets.QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setFont(QtGui.QFont("JetBrains Mono"))
        self.console_text.setMaximumHeight(180)
        layout.addWidget(self.console_text)
        return panel

    def _toggle_console(self, enabled: bool) -> None:
        self.console_panel.setVisible(enabled)
        if enabled:
            self._refresh_log_view()

    def _toggle_drawer(self, enabled: bool) -> None:
        if hasattr(self, "project_scroll"):
            self.project_scroll.setVisible(enabled)
            if hasattr(self, "main_split"):
                sizes = self.main_split.sizes()
                if enabled:
                    restored = int(getattr(self, "_last_drawer_size", 0) or 0)
                    if restored <= 0:
                        restored = int(300 * self._ui_scale)
                    center_size = max(sum(sizes) - restored - (sizes[2] if len(sizes) > 2 else 0), 200)
                    right_size = sizes[2] if len(sizes) > 2 else 0
                    self.main_split.setSizes([restored, center_size, right_size])
                else:
                    if len(sizes) == 3:
                        self._last_drawer_size = sizes[0]
                        self.main_split.setSizes([0, sizes[1] + sizes[0], sizes[2]])
                self._update_splitter_constraints()

    def _toggle_inspector(self, enabled: bool) -> None:
        if hasattr(self, "inspector_scroll"):
            self.inspector_scroll.setVisible(enabled)
            if hasattr(self, "main_split"):
                sizes = self.main_split.sizes()
                if enabled:
                    restored = int(getattr(self, "_last_inspector_size", 0) or 0)
                    if restored <= 0:
                        restored = int(240 * self._ui_scale)
                    left_size = sizes[0] if len(sizes) > 0 else 0
                    center_size = max(sum(sizes) - left_size - restored, 200)
                    self.main_split.setSizes([left_size, center_size, restored])
                else:
                    if len(sizes) == 3:
                        self._last_inspector_size = sizes[2]
                        self.main_split.setSizes([sizes[0], sizes[1] + sizes[2], 0])
                self._update_splitter_constraints()

    def _apply_initial_layout(self) -> None:
        if not hasattr(self, "main_split"):
            return
        scale = float(self._ui_scale) if hasattr(self, "_ui_scale") else 1.0
        drawer_width = int(320 * scale)
        center_width = int(900 * scale)
        self.project_scroll.setVisible(True)
        self.inspector_scroll.setVisible(False)
        if hasattr(self, "drawer_toggle_btn"):
            self.drawer_toggle_btn.blockSignals(True)
            self.drawer_toggle_btn.setChecked(True)
            self.drawer_toggle_btn.blockSignals(False)
        if hasattr(self, "inspector_toggle_btn"):
            self.inspector_toggle_btn.blockSignals(True)
            self.inspector_toggle_btn.setChecked(False)
            self.inspector_toggle_btn.blockSignals(False)
        self.main_split.setSizes([drawer_width, center_width, 0])
        self._update_splitter_constraints()

    def _build_pages(self) -> None:
        self.page_project = self._build_project_page()
        self.page_define = self._build_define_page()
        self.page_qc = self._build_qc_page()
        self.page_explore = self._build_explore_page()
        self.page_export = self._build_export_page()

        self.page_stack.addWidget(self.page_project)
        self.page_stack.addWidget(self.page_define)
        self.page_stack.addWidget(self.page_qc)
        self.page_stack.addWidget(self.page_explore)
        self.page_stack.addWidget(self.page_export)

    def _build_inspectors(self) -> None:
        self.inspector_project = self._build_project_inspector()
        self.inspector_define = self._build_define_inspector()
        self.inspector_qc = self._build_qc_inspector()
        self.inspector_explore = self._build_explore_inspector()
        self.inspector_export = self._build_export_inspector()
        for widget in (
            self.inspector_project,
            self.inspector_define,
            self.inspector_qc,
            self.inspector_explore,
            self.inspector_export,
        ):
            self.inspector_stack.addWidget(widget)

    def _set_active_step(self, index: int) -> None:
        if index < 0 or index >= self.page_stack.count():
            return
        self.page_stack.setCurrentIndex(index)
        self.inspector_stack.setCurrentIndex(index)
        if hasattr(self, "stepper_buttons") and index < len(self.stepper_buttons):
            self.stepper_buttons[index].setChecked(True)

    def _bind_shortcuts(self) -> None:
        shortcuts = {
            "Ctrl+O": self._load_project,
            "Ctrl+S": self._save_project,
            "Ctrl+R": self._run_analysis,
            "Ctrl+Shift+R": self._export_report,
            "Ctrl+E": self._export_data,
            "Ctrl+P": lambda: self.presentation_toggle.setChecked(not self.presentation_toggle.isChecked()),
            "Ctrl+K": self._open_command_palette,
            "Ctrl++": lambda: self._adjust_scale(1),
            "Ctrl+=": lambda: self._adjust_scale(1),
            "Ctrl+-": lambda: self._adjust_scale(-1),
            "Ctrl+0": lambda: self._set_user_scale(1.0),
            "Ctrl+Q": self.close,
        }
        for seq, handler in shortcuts.items():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            shortcut.activated.connect(handler)

        app = QtWidgets.QApplication.instance()
        if app:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Wheel:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._adjust_scale(1)
                elif delta < 0:
                    self._adjust_scale(-1)
                return True
        return super().eventFilter(obj, event)

    def _adjust_scale(self, step: int) -> None:
        if not hasattr(self, "_scale_levels"):
            return
        levels = self._scale_levels
        current = min(levels, key=lambda v: abs(v - self._user_scale))
        idx = levels.index(current)
        idx = max(0, min(len(levels) - 1, idx + step))
        self._set_user_scale(levels[idx])

    def _set_presentation_mode(self, enabled: bool) -> None:
        self._presentation_scale = 1.2 if enabled else 1.0
        self._apply_scale()
        if hasattr(self, "project_scroll"):
            self.project_scroll.setVisible(not enabled)
        if hasattr(self, "inspector_scroll"):
            self.inspector_scroll.setVisible(not enabled)
        if hasattr(self, "console_panel"):
            self.console_panel.setVisible(False)
            if hasattr(self, "console_toggle_btn"):
                self.console_toggle_btn.setChecked(False)
        if hasattr(self, "drawer_toggle_btn"):
            self.drawer_toggle_btn.setChecked(not enabled)

    def _apply_scale(self) -> None:
        total = float(self._user_scale) * float(self._presentation_scale)
        self._ui_scale = total
        self._plot_line_width = 2.0 * total
        self._plot_marker_size = 4.0 * total
        app = QtWidgets.QApplication.instance()
        if app and self._base_font:
            font = QtGui.QFont(self._base_font)
            size = font.pointSizeF() if font.pointSizeF() > 0 else font.pointSize()
            if size <= 0:
                size = 10.0
            font.setPointSizeF(size * total)
            app.setFont(font)
        self._apply_ui_style(total)
        self._set_timeline_summary_style()
        if hasattr(self, "timeline_plot"):
            self._style_plot(self.timeline_plot, "Occupancy Timeline")
            self.timeline_plot.setMinimumHeight(int(160 * total))
        if hasattr(self, "timeline_event_plot"):
            self._style_plot(self.timeline_event_plot, "Entry / Exit Rate")
            self.timeline_event_plot.setMinimumHeight(int(120 * total))
        if hasattr(self, "hist_zero_plot"):
            self._style_plot(self.hist_zero_plot, "Zero vs Non-zero")
        if hasattr(self, "hist_plot"):
            self._style_plot(self.hist_plot, "Distribution")
        if hasattr(self, "event_plot"):
            self._style_plot(self.event_plot, "Occupancy Events")
        if hasattr(self, "matrix_view"):
            try:
                self.matrix_view.setBackground(self._get_theme_tokens()["plot_bg"])
            except Exception:
                pass
        icon_size = int(18 * total)
        for attr in ("timeline_copy_btn", "timeline_export_btn", "plots_copy_btn"):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            try:
                btn.setIconSize(QtCore.QSize(icon_size, icon_size))
            except Exception:
                pass
        handle = max(6, int(6 * total))
        for splitter_name in (
            "main_split",
            "vertical_split",
            "explore_split",
            "lower_split",
            "timeline_split",
            "export_split",
        ):
            splitter = getattr(self, splitter_name, None)
            if splitter is not None:
                splitter.setHandleWidth(handle)
        header_min = max(80, int(80 * total))
        for table_attr in (
            "doctor_seed_table",
            "seed_validation_table",
            "bridge_table",
            "hydration_table",
            "extract_table",
            "selection_table",
        ):
            table = getattr(self, table_attr, None)
            if table is None:
                continue
            table.horizontalHeader().setMinimumSectionSize(header_min)
        for view_attr in ("per_frame_table", "per_solvent_table"):
            view = getattr(self, view_attr, None)
            if view is None:
                continue
            view.horizontalHeader().setMinimumSectionSize(header_min)
        if hasattr(self, "header_layout"):
            self.header_layout.setSpacing(max(6, int(8 * total)))
            self.header_bar.setMinimumHeight(int(44 * total))
        if hasattr(self, "timeline_plot"):
            self._update_timeline_plot()
        if hasattr(self, "hist_plot"):
            self._update_hist_plot()
        if hasattr(self, "event_plot"):
            self._update_event_plot()
        self._update_splitter_constraints()

    def _sync_scale_combo(self) -> None:
        if not hasattr(self, "scale_combo"):
            return
        idx = 0
        best = None
        for i, level in enumerate(self._scale_levels):
            if best is None or abs(level - self._user_scale) < abs(best - self._user_scale):
                best = level
                idx = i
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentIndex(idx)
        self.scale_combo.blockSignals(False)

    def _sync_theme_combo(self) -> None:
        if not hasattr(self, "theme_combo"):
            return
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentText("Dark" if self._theme_mode == "dark" else "Light")
        self.theme_combo.blockSignals(False)

    def _set_user_scale(self, scale: float) -> None:
        self._user_scale = float(scale)
        self.settings.setValue("ui_scale", self._user_scale)
        self._sync_scale_combo()
        self._apply_scale()

    def _on_scale_changed(self) -> None:
        if not hasattr(self, "scale_combo"):
            return
        data = self.scale_combo.currentData()
        try:
            scale = float(data)
        except Exception:
            scale = 1.0
        self._set_user_scale(scale)

    def _on_theme_changed(self) -> None:
        if not hasattr(self, "theme_combo"):
            return
        mode = "dark" if self.theme_combo.currentText().lower().startswith("dark") else "light"
        self._set_theme_mode(mode)

    def _set_theme_mode(self, mode: str) -> None:
        self._theme_mode = "dark" if str(mode).lower() == "dark" else "light"
        self.settings.setValue("ui_theme", self._theme_mode)
        self._sync_theme_combo()
        self._apply_plot_theme()
        self._apply_scale()

    def _get_theme_tokens(self) -> dict[str, str]:
        if getattr(self, "_theme_mode", "light") == "dark":
            return {
                "base": "#0A0F14",
                "surface": "#0F1620",
                "panel": "#121B26",
                "text": "#E7F0F8",
                "text_muted": "#A9B7C6",
                "border": "#223040",
                "border_hover": "#2B3E53",
                "accent": "#46D6FF",
                "accent_soft": "#1F2A3A",
                "accent_alt": "#2c7a7b",
                "selection": "#46D6FF",
                "selection_text": "#0A0F14",
                "grid": "#1B2836",
                "plot_bg": "#0F1620",
                "plot_fg": "#C7D2E0",
                "plot_axis": "#425166",
                "button_bg": "#1F2A3A",
                "button_hover": "#263548",
                "tab_bg": "#121B26",
                "entry": "#34D399",
                "exit": "#F87171",
            }
        return {
            "base": "#F5F7FB",
            "surface": "#FFFFFF",
            "panel": "#EEF2F7",
            "text": "#0B1A2A",
            "text_muted": "#4B5563",
            "border": "#CCD6E0",
            "border_hover": "#AEBCCD",
            "accent": "#0EA5E9",
            "accent_soft": "#D7EEF9",
            "accent_alt": "#1F7A8C",
            "selection": "#0EA5E9",
            "selection_text": "#0B1A2A",
            "grid": "#E1E7EF",
            "plot_bg": "#FFFFFF",
            "plot_fg": "#0B1A2A",
            "plot_axis": "#5B6B7C",
            "button_bg": "#E6ECF3",
            "button_hover": "#DDE6EF",
            "tab_bg": "#E9EEF4",
            "entry": "#15803D",
            "exit": "#B91C1C",
        }

    def _apply_plot_theme(self) -> None:
        tokens = self._get_theme_tokens()
        pg.setConfigOptions(
            antialias=True,
            foreground=tokens["plot_fg"],
            background=tokens["plot_bg"],
        )

    def _queue_timeline_update(self) -> None:
        if self._timeline_update_pending:
            return
        self._timeline_update_pending = True
        QtCore.QTimer.singleShot(60, self._run_timeline_update)

    def _run_timeline_update(self) -> None:
        self._timeline_update_pending = False
        self._update_timeline_plot()

    def _queue_hist_update(self) -> None:
        if self._hist_update_pending:
            return
        self._hist_update_pending = True
        QtCore.QTimer.singleShot(80, self._run_hist_update)

    def _run_hist_update(self) -> None:
        self._hist_update_pending = False
        self._update_hist_plot()

    def _queue_matrix_update(self) -> None:
        if self._matrix_update_pending:
            return
        self._matrix_update_pending = True
        QtCore.QTimer.singleShot(120, self._run_matrix_update)

    def _run_matrix_update(self) -> None:
        self._matrix_update_pending = False
        self._update_matrix_plot()

    def _queue_event_update(self) -> None:
        if self._event_update_pending:
            return
        self._event_update_pending = True
        QtCore.QTimer.singleShot(120, self._run_event_update)

    def _run_event_update(self) -> None:
        self._event_update_pending = False
        self._update_event_plot()

    def _set_timeline_summary_style(self) -> None:
        if not hasattr(self, "timeline_summary_frame"):
            return
        pad = int(8 * self._ui_scale)
        font_size = int(12 * self._ui_scale)
        tokens = self._get_theme_tokens()
        self.timeline_summary_frame.setStyleSheet(
            f"QFrame {{ background: {tokens['surface']}; border: 1px solid {tokens['border']}; "
            f"border-radius: 10px; }}"
            f"QLabel {{ padding: {pad}px; color: {tokens['text_muted']}; "
            f"font-size: {font_size}px; }}"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_splitter_constraints()

    def _update_splitter_constraints(self) -> None:
        scale = float(self._ui_scale) if hasattr(self, "_ui_scale") else 1.0
        # Main horizontal split (project | center | inspector)
        main_split = getattr(self, "main_split", None)
        if main_split and hasattr(self, "project_scroll") and hasattr(self, "inspector_scroll"):
            avail = main_split.width() or self.width()
            if avail <= 0:
                avail = self.width()
            left_base = int(300 * scale)
            right_base = int(240 * scale)
            center_min = int(420 * scale)
            inspector_visible = self.inspector_scroll.isVisible()
            drawer_visible = self.project_scroll.isVisible()
            if avail > 0:
                max_side = max(int(avail * 0.35), int(200 * scale))
                left_min = min(left_base, max_side) if drawer_visible else 0
                right_min = min(right_base, max_side) if inspector_visible else 0
                total_min = left_min + right_min + center_min
                if total_min > avail:
                    shrink = avail / max(total_min, 1)
                    left_min = max(int(left_min * shrink), int(120 * scale)) if drawer_visible else 0
                    right_min = max(int(right_min * shrink), int(120 * scale)) if inspector_visible else 0
                    center_min = max(int(center_min * shrink), int(160 * scale))
                self.project_scroll.setMinimumWidth(left_min if drawer_visible else 0)
                self.inspector_scroll.setMinimumWidth(right_min if inspector_visible else 0)
                self.page_stack.setMinimumWidth(center_min)
                sizes = main_split.sizes()
                if len(sizes) == 3:
                    if (
                        sizes[0] < left_min
                        or sizes[1] < center_min
                        or sizes[2] < right_min
                        or not inspector_visible
                        or not drawer_visible
                    ):
                        center_size = max(avail - left_min - right_min, center_min)
                        main_split.setSizes([left_min, center_size, right_min])

        # Explore split (timeline | plots/tables)
        explore_split = getattr(self, "explore_split", None)
        if explore_split and hasattr(self, "timeline_panel") and hasattr(self, "lower_split"):
            avail_h = explore_split.height()
            if avail_h > 0:
                top_min = min(int(260 * scale), int(avail_h * 0.75))
                bottom_min = min(int(200 * scale), int(avail_h * 0.6))
                if top_min + bottom_min > avail_h:
                    shrink = avail_h / max(top_min + bottom_min, 1)
                    top_min = max(int(top_min * shrink), int(160 * scale))
                    bottom_min = max(int(bottom_min * shrink), int(140 * scale))
                self.timeline_panel.setMinimumHeight(top_min)
                self.lower_split.setMinimumHeight(bottom_min)
                sizes = explore_split.sizes()
                if len(sizes) == 2:
                    if sizes[0] < top_min or sizes[1] < bottom_min:
                        bottom_size = max(avail_h - top_min, bottom_min)
                        explore_split.setSizes([top_min, bottom_size])

        # Timeline internal split (occupancy | entry/exit)
        timeline_split = getattr(self, "timeline_split", None)
        if timeline_split and hasattr(self, "timeline_top") and hasattr(self, "timeline_event_container"):
            avail_h = timeline_split.height()
            if avail_h > 0:
                top_min = min(int(200 * scale), int(avail_h * 0.7))
                bottom_min = min(int(160 * scale), int(avail_h * 0.6))
                if top_min + bottom_min > avail_h:
                    shrink = avail_h / max(top_min + bottom_min, 1)
                    top_min = max(int(top_min * shrink), int(140 * scale))
                    bottom_min = max(int(bottom_min * shrink), int(120 * scale))
                self.timeline_top.setMinimumHeight(top_min)
                self.timeline_event_container.setMinimumHeight(bottom_min)
                sizes = timeline_split.sizes()
                if len(sizes) == 2:
                    if sizes[0] < top_min or sizes[1] < bottom_min:
                        bottom_size = max(avail_h - top_min, bottom_min)
                        timeline_split.setSizes([top_min, bottom_size])

    def _build_project_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        inputs_group = QtWidgets.QGroupBox("Inputs")
        inputs_layout = QtWidgets.QVBoxLayout(inputs_group)
        self.topology_label = QtWidgets.QLabel("Topology: -")
        self.trajectory_label = QtWidgets.QLabel("Trajectory: -")
        self.metadata_label = QtWidgets.QLabel("Metadata: -")
        mono = QtGui.QFont("JetBrains Mono")
        for lbl in (self.topology_label, self.trajectory_label):
            lbl.setFont(mono)
        for lbl in (self.topology_label, self.trajectory_label, self.metadata_label):
            lbl.setWordWrap(True)
            inputs_layout.addWidget(lbl)
        layout.addWidget(inputs_group)

        analysis_group = QtWidgets.QGroupBox("Analysis Window")
        frame_controls = QtWidgets.QFormLayout(analysis_group)
        self.frame_start_spin = QtWidgets.QSpinBox()
        self.frame_start_spin.setRange(0, 10_000_000)
        self.frame_stop_spin = QtWidgets.QSpinBox()
        self.frame_stop_spin.setRange(-1, 10_000_000)
        self.frame_stop_spin.setSpecialValueText("End")
        self.frame_stride_spin = QtWidgets.QSpinBox()
        self.frame_stride_spin.setRange(1, 1_000_000)
        self.frame_stride_spin.setValue(1)
        frame_controls.addRow("Frame start", self.frame_start_spin)
        frame_controls.addRow("Frame stop", self.frame_stop_spin)
        frame_controls.addRow("Stride", self.frame_stride_spin)
        layout.addWidget(analysis_group)

        self.output_group = QtWidgets.QGroupBox("Output Settings")
        output_layout = QtWidgets.QFormLayout(self.output_group)
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("results/")
        self.output_dir_browse = QtWidgets.QPushButton("Browse")
        self.output_dir_browse.clicked.connect(self._browse_output_dir)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.output_dir_browse)
        self.report_format_combo = QtWidgets.QComboBox()
        self.report_format_combo.addItems(["html", "md"])
        self.write_per_frame_check = QtWidgets.QCheckBox("Write per-frame CSV")
        self.write_parquet_check = QtWidgets.QCheckBox("Write parquet")
        self.output_effective_label = None
        output_layout.addRow("Output dir", output_row)
        output_layout.addRow("Report format", self.report_format_combo)
        output_layout.addRow("", self.write_per_frame_check)
        output_layout.addRow("", self.write_parquet_check)
        layout.addWidget(self.output_group)

        self.report_format_combo.currentTextChanged.connect(self._on_output_settings_changed)
        self.write_per_frame_check.toggled.connect(self._on_output_settings_changed)
        self.write_parquet_check.toggled.connect(self._on_output_settings_changed)
        self.output_dir_edit.textChanged.connect(self._on_output_settings_changed)

        self.output_dir_edit.setToolTip("Base directory for analysis outputs and logs.")
        self.report_format_combo.setToolTip("Report format for Export Report (html or md).")
        self.write_per_frame_check.setToolTip("Write per-frame CSV outputs to disk.")
        self.write_parquet_check.setToolTip("Write parquet outputs where supported.")

        self.doctor_group = QtWidgets.QGroupBox("Project Doctor")
        doctor_layout = QtWidgets.QVBoxLayout(self.doctor_group)
        doctor_top = QtWidgets.QHBoxLayout()
        self.doctor_status_label = QtWidgets.QLabel("Not validated yet.")
        self.doctor_run_btn = QtWidgets.QPushButton("Run Project Doctor")
        self.doctor_run_btn.clicked.connect(self._run_project_doctor)
        self.doctor_pbc_btn = QtWidgets.QPushButton("PBC Helper")
        self.doctor_pbc_btn.clicked.connect(self._show_pbc_helper)
        self.doctor_run_btn.setToolTip(
            "Validate topology/trajectory, solvent, and selection matches."
        )
        self.doctor_pbc_btn.setToolTip("Show recommended GROMACS trjconv preprocessing commands.")
        doctor_top.addWidget(self.doctor_run_btn)
        doctor_top.addWidget(self.doctor_pbc_btn)
        doctor_top.addStretch(1)
        doctor_layout.addLayout(doctor_top)
        doctor_layout.addWidget(self.doctor_status_label)
        self.doctor_text = QtWidgets.QTextEdit()
        self.doctor_text.setReadOnly(True)
        self.doctor_text.setMinimumHeight(120)
        doctor_layout.addWidget(self.doctor_text)
        self.doctor_seed_table = QtWidgets.QTableWidget()
        self.doctor_seed_table.setColumnCount(6)
        self._configure_table_headers(
            self.doctor_seed_table,
            [
                "Selection label",
                "Count",
                "Require unique match",
                "Expect count",
                "Selection",
                "Suggestions",
            ],
        )
        doctor_layout.addWidget(self.doctor_seed_table)
        layout.addWidget(self.doctor_group)

        soz_group = QtWidgets.QGroupBox("Defined SOZs")
        soz_layout = QtWidgets.QVBoxLayout(soz_group)
        self.soz_list = QtWidgets.QListWidget()
        soz_layout.addWidget(self.soz_list)
        layout.addWidget(soz_group)

        layout.addStretch(1)
        return panel

    def _configure_table_headers(self, table: QtWidgets.QTableWidget, labels: list[str]) -> None:
        table.setHorizontalHeaderLabels(labels)
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        scale = float(getattr(self, "_ui_scale", 1.0))
        header.setMinimumSectionSize(max(80, int(80 * scale)))
        table.setWordWrap(False)

    def _configure_table_view_header(self, view: QtWidgets.QTableView) -> None:
        header = view.horizontalHeader()
        header.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        scale = float(getattr(self, "_ui_scale", 1.0))
        header.setMinimumSectionSize(max(80, int(80 * scale)))
        view.setWordWrap(False)

    def _build_project_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(12)

        summary_group = QtWidgets.QGroupBox("Project Summary")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        self.project_summary_label = QtWidgets.QLabel(
            "Load a project to view topology, trajectory, frames, and PBC status."
        )
        self.project_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.project_summary_label)
        layout.addWidget(summary_group)

        run_group = QtWidgets.QGroupBox("Run Summary")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        self.overview_card = QtWidgets.QLabel("Run analysis to populate summary cards.")
        self.overview_card.setWordWrap(True)
        run_layout.addWidget(self.overview_card)
        self.overview_raw_toggle = QtWidgets.QCheckBox("Show raw JSON")
        self.overview_raw_toggle.toggled.connect(self._toggle_overview_raw)
        run_layout.addWidget(self.overview_raw_toggle)
        self.overview_text = QtWidgets.QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setVisible(False)
        run_layout.addWidget(self.overview_text)
        layout.addWidget(run_group)

        layout.addStretch(1)
        return panel

    def _build_define_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.builder_panel = self._build_builder_panel()
        layout.addWidget(self.builder_panel)
        return panel

    def _build_qc_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.qc_tabs = QtWidgets.QTabWidget()
        qc_panel = QtWidgets.QWidget()
        qc_layout = QtWidgets.QVBoxLayout(qc_panel)
        self.qc_summary_label = QtWidgets.QLabel("QC summary will appear after analysis.")
        self.qc_summary_label.setWordWrap(True)
        qc_layout.addWidget(self.qc_summary_label)
        self.qc_raw_toggle = QtWidgets.QCheckBox("Show raw QC JSON")
        self.qc_raw_toggle.toggled.connect(self._toggle_qc_raw)
        qc_layout.addWidget(self.qc_raw_toggle)
        self.qc_text = QtWidgets.QTextEdit()
        self.qc_text.setReadOnly(True)
        self.qc_text.setVisible(False)
        qc_layout.addWidget(self.qc_text)
        self.qc_tabs.addTab(qc_panel, "QC Summary")
        self.qc_tabs.addTab(self._build_logs_tab(), "Diagnostics")
        layout.addWidget(self.qc_tabs)
        return panel

    def _build_explore_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(8)

        self.timeline_panel = self._build_timeline_tab()

        self.lower_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.lower_split.addWidget(self._build_plots_tab())
        self.lower_split.addWidget(self._build_tables_tab())
        self.lower_split.setStretchFactor(0, 2)
        self.lower_split.setStretchFactor(1, 1)
        self.lower_split.setChildrenCollapsible(False)
        self.lower_split.setCollapsible(0, False)
        self.lower_split.setCollapsible(1, False)
        self.lower_split.setSizes([700, 420])
        self.lower_split.setOpaqueResize(True)

        self.explore_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.explore_split.addWidget(self.timeline_panel)
        self.explore_split.addWidget(self.lower_split)
        self.explore_split.setStretchFactor(0, 2)
        self.explore_split.setStretchFactor(1, 1)
        self.explore_split.setChildrenCollapsible(False)
        self.explore_split.setCollapsible(0, False)
        self.explore_split.setCollapsible(1, False)
        self.explore_split.setSizes([520, 360])
        self.explore_split.setOpaqueResize(True)
        layout.addWidget(self.explore_split, 1)
        return panel

    def _build_export_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        action_row = QtWidgets.QHBoxLayout()
        export_data_btn = QtWidgets.QPushButton("Export Data")
        export_data_btn.clicked.connect(self._export_data)
        export_report_btn = QtWidgets.QPushButton("Export Report")
        export_report_btn.clicked.connect(self._export_report)
        action_row.addWidget(export_data_btn)
        action_row.addWidget(export_report_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        self.export_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.export_split.addWidget(self._build_extract_tab())
        self.export_split.addWidget(self._build_report_tab())
        self.export_split.setStretchFactor(0, 2)
        self.export_split.setStretchFactor(1, 1)
        self.export_split.setChildrenCollapsible(False)
        self.export_split.setCollapsible(0, False)
        self.export_split.setCollapsible(1, False)
        self.export_split.setOpaqueResize(True)
        layout.addWidget(self.export_split)
        return panel

    def _build_project_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(8)
        prov_group = QtWidgets.QGroupBox("Provenance Stamp")
        prov_layout = QtWidgets.QVBoxLayout(prov_group)
        self.provenance_text = QtWidgets.QTextEdit()
        self.provenance_text.setReadOnly(True)
        self.provenance_text.setFont(QtGui.QFont("JetBrains Mono"))
        prov_layout.addWidget(self.provenance_text)
        copy_btn = QtWidgets.QPushButton("Copy Provenance")
        copy_btn.clicked.connect(self._copy_provenance)
        prov_layout.addWidget(copy_btn)
        layout.addWidget(prov_group)
        return panel

    def _build_define_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        group = QtWidgets.QGroupBox("Selection Inspector")
        g_layout = QtWidgets.QVBoxLayout(group)
        self.define_inspector_text = QtWidgets.QLabel("Edit selections to see live validation.")
        self.define_inspector_text.setWordWrap(True)
        g_layout.addWidget(self.define_inspector_text)
        layout.addWidget(group)
        return panel

    def _build_qc_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        group = QtWidgets.QGroupBox("QC Status")
        g_layout = QtWidgets.QVBoxLayout(group)
        self.qc_inspector_text = QtWidgets.QLabel("Run Project Doctor to populate QC status.")
        self.qc_inspector_text.setWordWrap(True)
        g_layout.addWidget(self.qc_inspector_text)
        layout.addWidget(group)
        return panel

    def _build_explore_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        group = QtWidgets.QGroupBox("Explorer Inspector")
        g_layout = QtWidgets.QVBoxLayout(group)
        self.explore_inspector_text = QtWidgets.QLabel("Select a SOZ or solvent to inspect details.")
        self.explore_inspector_text.setWordWrap(True)
        g_layout.addWidget(self.explore_inspector_text)
        layout.addWidget(group)
        return panel

    def _build_export_inspector(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        group = QtWidgets.QGroupBox("Export Status")
        g_layout = QtWidgets.QVBoxLayout(group)
        self.export_inspector_text = QtWidgets.QLabel("Preview extraction rules and outputs.")
        self.export_inspector_text.setWordWrap(True)
        g_layout.addWidget(self.export_inspector_text)
        layout.addWidget(group)
        return panel

    def _build_center_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_overview_tab(), "Overview")
        tabs.addTab(self._build_qc_tab(), "QC Summary")
        tabs.addTab(self._build_timeline_tab(), "Timeline")
        tabs.addTab(self._build_plots_tab(), "Plots")
        tabs.addTab(self._build_tables_tab(), "Tables")
        tabs.addTab(self._build_report_tab(), "Report")
        tabs.addTab(self._build_logs_tab(), "Logs")
        tabs.addTab(self._build_extract_tab(), "Extract")
        return tabs

    def _build_overview_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.overview_text = QtWidgets.QTextEdit()
        self.overview_text.setReadOnly(True)
        layout.addWidget(self.overview_text)
        return panel

    def _build_qc_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.qc_text = QtWidgets.QTextEdit()
        self.qc_text.setReadOnly(True)
        layout.addWidget(self.qc_text)
        return panel

    def _build_timeline_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        controls.setContentsMargins(4, 4, 4, 4)
        controls.addWidget(QtWidgets.QLabel("SOZ"))
        self.timeline_soz_combo = QtWidgets.QComboBox()
        self.timeline_overlay = QtWidgets.QCheckBox("Overlay all SOZs")
        self.timeline_step_check = QtWidgets.QCheckBox("Step plot")
        self.timeline_step_check.setChecked(True)
        self.timeline_markers_check = QtWidgets.QCheckBox("Markers")
        self.timeline_clamp_check = QtWidgets.QCheckBox("Clamp y≥0")
        self.timeline_clamp_check.setChecked(True)
        self.timeline_mean_check = QtWidgets.QCheckBox("Mean line")
        self.timeline_mean_check.setChecked(True)
        self.timeline_median_check = QtWidgets.QCheckBox("Median line")
        self.timeline_median_check.setChecked(True)
        self.timeline_shade_check = QtWidgets.QCheckBox("Shade occupancy")
        self.timeline_shade_check.setChecked(True)
        self.timeline_shade_threshold = QtWidgets.QSpinBox()
        self.timeline_shade_threshold.setRange(1, 1_000_000)
        self.timeline_shade_threshold.setValue(1)
        self.timeline_brush_check = QtWidgets.QCheckBox("Brush time window")
        self.timeline_brush_clear = QtWidgets.QPushButton("Clear brush")
        self.timeline_brush_clear.clicked.connect(self._clear_time_brush)
        self.timeline_brush_clear.setEnabled(False)
        self.timeline_event_split_check = QtWidgets.QCheckBox("Split axes")
        self.timeline_event_split_check.setChecked(False)
        self.timeline_event_signed_check = QtWidgets.QCheckBox("Exits negative")
        self.timeline_event_signed_check.setChecked(False)
        self.timeline_event_mode_combo = QtWidgets.QComboBox()
        self.timeline_event_mode_combo.addItems(
            ["Events per frame", "Rate (events/ns)", "Cumulative events"]
        )
        self.timeline_event_norm_combo = QtWidgets.QComboBox()
        self.timeline_event_norm_combo.addItems(["per ns", "per 100 frames", "none"])
        self.timeline_event_norm_combo.setCurrentText("none")
        self.timeline_event_bin_spin = QtWidgets.QDoubleSpinBox()
        self.timeline_event_bin_spin.setRange(0.0, 100.0)
        self.timeline_event_bin_spin.setDecimals(3)
        self.timeline_event_bin_spin.setSingleStep(0.1)
        self.timeline_event_bin_spin.setValue(0.0)
        controls.addWidget(QtWidgets.QLabel("Metric"))
        self.timeline_metric_combo = QtWidgets.QComboBox()
        self.timeline_metric_combo.addItems(["n_solvent", "entries", "exits", "occupancy_fraction"])
        controls.addWidget(self.timeline_metric_combo)
        controls.addWidget(QtWidgets.QLabel("Secondary"))
        self.timeline_secondary_combo = QtWidgets.QComboBox()
        self.timeline_secondary_combo.addItems(["None", "occupancy_fraction", "entries", "exits", "n_solvent"])
        controls.addWidget(self.timeline_secondary_combo)
        self.timeline_smooth_check = QtWidgets.QCheckBox("Smooth")
        self.timeline_smooth_window = QtWidgets.QSpinBox()
        self.timeline_smooth_window.setRange(1, 500)
        self.timeline_smooth_window.setValue(5)
        controls.addWidget(self.timeline_smooth_check)
        controls.addWidget(QtWidgets.QLabel("Window"))
        controls.addWidget(self.timeline_smooth_window)

        self.timeline_soz_combo.setToolTip("Select which SOZ to display.")
        self.timeline_overlay.setToolTip("Overlay all SOZs in the same plot.")
        self.timeline_metric_combo.setToolTip("Primary metric to plot on the main axis.")
        self.timeline_secondary_combo.setToolTip("Optional secondary metric on the right axis.")
        self.timeline_step_check.setToolTip("Use a step plot for discrete counts.")
        self.timeline_markers_check.setToolTip("Show markers at sample points.")
        self.timeline_clamp_check.setToolTip("Clamp the y-axis to zero or above.")
        self.timeline_mean_check.setToolTip("Show the mean of the primary metric.")
        self.timeline_median_check.setToolTip("Show the median of the primary metric.")
        self.timeline_shade_check.setToolTip("Shade time spans where occupancy meets the threshold.")
        self.timeline_shade_threshold.setToolTip("Minimum n_solvent to shade occupancy spans.")
        self.timeline_brush_check.setToolTip("Select a time window to filter plots and tables.")
        self.timeline_smooth_check.setToolTip("Apply a rolling mean to improve visual clarity.")
        self.timeline_smooth_window.setToolTip("Window size for smoothing (frames).")
        self.timeline_event_split_check.setToolTip("Plot entries/exits on separate axes for clarity.")
        self.timeline_event_signed_check.setToolTip("Plot exits as negative values on the same axis.")
        self.timeline_event_mode_combo.setToolTip("Choose how to display entry/exit activity.")
        self.timeline_event_norm_combo.setToolTip("Normalize event series for easier comparison.")
        self.timeline_event_bin_spin.setToolTip("Aggregate events into time bins (ns). 0 disables binning.")
        self.timeline_save_btn = QtWidgets.QPushButton("Save Plot")
        self.timeline_save_btn.clicked.connect(
            lambda: self._export_plot(
                self.timeline_plot,
                "timeline.png",
                csv_exporter=self._write_timeline_csv,
            )
        )
        self.timeline_copy_btn = QtWidgets.QToolButton()
        self.timeline_copy_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton))
        self.timeline_copy_btn.setToolTip("Copy timeline plot to clipboard")
        self.timeline_copy_btn.clicked.connect(lambda: self._copy_widget_to_clipboard(self.timeline_plot))
        self.timeline_export_btn = QtWidgets.QToolButton()
        self.timeline_export_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.timeline_export_btn.setToolTip("Export timeline plot")
        self.timeline_export_btn.clicked.connect(
            lambda: self._export_plot(
                self.timeline_plot,
                "timeline.png",
                csv_exporter=self._write_timeline_csv,
            )
        )
        self.timeline_soz_combo.currentTextChanged.connect(self._on_soz_selection_changed)
        self.timeline_overlay.toggled.connect(self._queue_timeline_update)
        self.timeline_step_check.toggled.connect(self._queue_timeline_update)
        self.timeline_markers_check.toggled.connect(self._queue_timeline_update)
        self.timeline_clamp_check.toggled.connect(self._queue_timeline_update)
        self.timeline_mean_check.toggled.connect(self._queue_timeline_update)
        self.timeline_median_check.toggled.connect(self._queue_timeline_update)
        self.timeline_shade_check.toggled.connect(self._queue_timeline_update)
        self.timeline_shade_threshold.valueChanged.connect(self._queue_timeline_update)
        self.timeline_brush_check.toggled.connect(self._queue_timeline_update)
        self.timeline_event_split_check.toggled.connect(self._queue_timeline_update)
        self.timeline_event_signed_check.toggled.connect(self._queue_timeline_update)
        self.timeline_event_mode_combo.currentTextChanged.connect(self._queue_timeline_update)
        self.timeline_event_norm_combo.currentTextChanged.connect(self._queue_timeline_update)
        self.timeline_event_bin_spin.valueChanged.connect(self._queue_timeline_update)
        self.timeline_event_mode_combo.currentTextChanged.connect(self._update_event_controls_state)
        self.timeline_event_signed_check.toggled.connect(self._update_event_controls_state)
        self.timeline_metric_combo.currentTextChanged.connect(self._queue_timeline_update)
        self.timeline_secondary_combo.currentTextChanged.connect(self._queue_timeline_update)
        self.timeline_smooth_check.toggled.connect(self._queue_timeline_update)
        self.timeline_smooth_window.valueChanged.connect(self._queue_timeline_update)
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setSpacing(8)
        toggle_row.addWidget(self.timeline_overlay)
        toggle_row.addWidget(self.timeline_step_check)
        toggle_row.addWidget(self.timeline_markers_check)
        toggle_row.addWidget(self.timeline_clamp_check)
        toggle_row.addWidget(self.timeline_mean_check)
        toggle_row.addWidget(self.timeline_median_check)
        toggle_row.addWidget(self.timeline_shade_check)
        toggle_row.addWidget(QtWidgets.QLabel("Shade ≥"))
        toggle_row.addWidget(self.timeline_shade_threshold)
        toggle_row.addWidget(self.timeline_brush_check)
        toggle_row.addWidget(self.timeline_brush_clear)
        # Entry/exit controls live with the event plot below.
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.timeline_save_btn)
        toggle_row.addWidget(self.timeline_copy_btn)
        toggle_row.addWidget(self.timeline_export_btn)

        controls.addWidget(self.timeline_soz_combo)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addLayout(toggle_row)
        self.timeline_plot = pg.PlotWidget()
        self._style_plot(self.timeline_plot, "Occupancy Timeline")
        self.timeline_plot.setLabel("bottom", "Time", units="ns")
        self.timeline_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.timeline_plot.setLabel("left", "Metric")
        self.timeline_secondary_view = pg.ViewBox()
        self.timeline_plot.plotItem.showAxis("right")
        self.timeline_plot.plotItem.getAxis("right").setLabel("Secondary")
        self.timeline_plot.plotItem.scene().addItem(self.timeline_secondary_view)
        self.timeline_plot.plotItem.getAxis("right").linkToView(self.timeline_secondary_view)
        self.timeline_secondary_view.setXLink(self.timeline_plot.plotItem)
        self.timeline_plot.plotItem.vb.sigResized.connect(self._update_timeline_views)
        self._timeline_secondary_items = []
        self.timeline_help_label = QtWidgets.QLabel(
            "Occupancy Timeline shows solvent counts per frame. Brush a time window to "
            "filter histograms, heatmaps, and tables."
        )
        self.timeline_help_label.setWordWrap(True)

        self.timeline_top = QtWidgets.QWidget()
        timeline_top_layout = QtWidgets.QVBoxLayout(self.timeline_top)
        timeline_top_layout.setContentsMargins(0, 0, 0, 0)
        timeline_top_layout.addWidget(self.timeline_plot, 1)
        timeline_top_layout.addWidget(self.timeline_help_label)

        self.timeline_summary_frame = QtWidgets.QFrame()
        self.timeline_summary_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        summary_layout = QtWidgets.QVBoxLayout(self.timeline_summary_frame)
        summary_layout.setContentsMargins(6, 4, 6, 4)
        self.timeline_summary_label = QtWidgets.QLabel("")
        self.timeline_summary_label.setWordWrap(True)
        self.timeline_summary_label.setVisible(False)
        summary_layout.addWidget(self.timeline_summary_label)
        stats_row = QtWidgets.QHBoxLayout()
        self.timeline_stats_button = QtWidgets.QPushButton("Compute Timeline Statistics")
        self.timeline_stats_button.clicked.connect(self._compute_timeline_statistics)
        self.timeline_stats_button.setToolTip(
            "Compute occupancy and transition statistics for the selected SOZ."
        )
        self.timeline_stats_status = QtWidgets.QLabel("Not computed yet.")
        self.timeline_stats_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        stats_row.addWidget(self.timeline_stats_button)
        stats_row.addWidget(self.timeline_stats_status, 1)
        summary_layout.addLayout(stats_row)
        self._set_timeline_summary_style()

        event_controls = QtWidgets.QHBoxLayout()
        event_controls.setSpacing(8)
        event_controls.addWidget(QtWidgets.QLabel("Entry/Exit mode"))
        event_controls.addWidget(self.timeline_event_mode_combo)
        event_controls.addWidget(QtWidgets.QLabel("Normalize"))
        event_controls.addWidget(self.timeline_event_norm_combo)
        event_controls.addWidget(QtWidgets.QLabel("Bin (ns)"))
        event_controls.addWidget(self.timeline_event_bin_spin)
        event_controls.addWidget(self.timeline_event_split_check)
        event_controls.addWidget(self.timeline_event_signed_check)
        self.timeline_event_save_btn = QtWidgets.QPushButton("Save Plot")
        self.timeline_event_save_btn.clicked.connect(
            lambda: self._export_plot(
                self.timeline_event_plot,
                "entry_exit.png",
                csv_exporter=self._write_entry_exit_csv,
            )
        )
        self.timeline_event_export_btn = QtWidgets.QPushButton("Export Entry/Exit CSV")
        self.timeline_event_export_btn.clicked.connect(self._export_entry_exit_csv)
        event_controls.addStretch(1)
        event_controls.addWidget(self.timeline_event_save_btn)
        event_controls.addWidget(self.timeline_event_export_btn)

        self.timeline_event_plot = pg.PlotWidget()
        self._style_plot(self.timeline_event_plot, "Entry / Exit Rate")
        self.timeline_event_plot.setLabel("bottom", "Time", units="ns")
        self.timeline_event_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.timeline_event_plot.setLabel("left", "Entries / frame")
        self.timeline_event_plot.setToolTip(
            "Entry/Exit quantifies changes in occupancy count. Entries are positive transitions, "
            "exits are negative (or separate) transitions over time."
        )
        self.timeline_event_plot.setXLink(self.timeline_plot)
        try:
            self.timeline_event_plot.plotItem.legend.clear()
        except Exception:
            pass
        self.timeline_event_secondary_view = pg.ViewBox()
        self.timeline_event_plot.plotItem.showAxis("right")
        self.timeline_event_plot.plotItem.getAxis("right").setLabel("Exits / frame")
        self.timeline_event_plot.plotItem.getAxis("right").setVisible(False)
        self.timeline_event_plot.plotItem.scene().addItem(self.timeline_event_secondary_view)
        self.timeline_event_plot.plotItem.getAxis("right").linkToView(self.timeline_event_secondary_view)
        self.timeline_event_secondary_view.setXLink(self.timeline_event_plot.plotItem)
        self.timeline_event_plot.plotItem.vb.sigResized.connect(self._update_event_views)
        self.timeline_event_plot.scene().sigMouseMoved.connect(self._on_timeline_event_hover)
        self._timeline_event_secondary_items = []

        self.timeline_event_container = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(self.timeline_event_container)
        event_layout.setContentsMargins(0, 0, 0, 0)
        event_layout.addWidget(self.timeline_summary_frame)
        event_layout.addLayout(event_controls)
        event_layout.addWidget(self.timeline_event_plot, 1)

        self.timeline_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.timeline_split.addWidget(self.timeline_top)
        self.timeline_split.addWidget(self.timeline_event_container)
        self.timeline_split.setStretchFactor(0, 2)
        self.timeline_split.setStretchFactor(1, 1)
        self.timeline_split.setChildrenCollapsible(False)
        self.timeline_split.setCollapsible(0, False)
        self.timeline_split.setCollapsible(1, False)
        self.timeline_split.setSizes([460, 320])
        self.timeline_split.setOpaqueResize(True)
        layout.addWidget(self.timeline_split, 1)
        self._update_event_controls_state()
        return panel

    def _build_plots_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.hist_controls_row = QtWidgets.QWidget()
        hist_controls = QtWidgets.QHBoxLayout(self.hist_controls_row)
        hist_controls.setSpacing(8)
        hist_controls.setContentsMargins(4, 4, 4, 4)
        hist_controls.addWidget(QtWidgets.QLabel("Histogram metric"))
        self.hist_metric_combo = QtWidgets.QComboBox()
        self.hist_bins_spin = QtWidgets.QSpinBox()
        self.hist_bins_spin.setRange(5, 200)
        self.hist_bins_spin.setValue(30)
        self.hist_norm_check = QtWidgets.QCheckBox("Normalize")
        self.hist_log_check = QtWidgets.QCheckBox("Log Y")
        self.hist_zero_split_check = QtWidgets.QCheckBox("Split zeros")
        self.hist_zero_split_check.setChecked(True)
        plot_btn = QtWidgets.QPushButton("Plot Histogram")
        save_btn = QtWidgets.QPushButton("Save Plot")
        self.plots_copy_btn = QtWidgets.QToolButton()
        self.plots_copy_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton))
        self.plots_copy_btn.setToolTip("Copy current plot to clipboard")
        plot_btn.clicked.connect(self._update_hist_plot)
        save_btn.clicked.connect(self._export_current_plot)
        self.plots_copy_btn.clicked.connect(self._copy_current_plot)
        hist_controls.addWidget(self.hist_metric_combo)
        hist_controls.addWidget(QtWidgets.QLabel("Bins"))
        hist_controls.addWidget(self.hist_bins_spin)
        hist_controls.addWidget(self.hist_norm_check)
        hist_controls.addWidget(self.hist_log_check)
        hist_controls.addWidget(self.hist_zero_split_check)
        hist_controls.addStretch(1)
        hist_controls.addWidget(plot_btn)
        layout.addWidget(self.hist_controls_row)

        self.matrix_controls_row = QtWidgets.QWidget()
        matrix_controls = QtWidgets.QHBoxLayout(self.matrix_controls_row)
        matrix_controls.setSpacing(8)
        matrix_controls.setContentsMargins(4, 0, 4, 4)
        self.matrix_source_label = QtWidgets.QLabel("Matrix source")
        matrix_controls.addWidget(self.matrix_source_label)
        self.matrix_source_combo = QtWidgets.QComboBox()
        self.matrix_source_combo.addItems(["Solvent occupancy (top N)", "Residue hydration"])
        matrix_controls.addWidget(self.matrix_source_combo)
        matrix_controls.addWidget(QtWidgets.QLabel("Top solvents"))
        self.matrix_top_spin = QtWidgets.QSpinBox()
        self.matrix_top_spin.setRange(5, 500)
        self.matrix_top_spin.setValue(50)
        matrix_controls.addWidget(self.matrix_top_spin)
        matrix_controls.addWidget(QtWidgets.QLabel("Min occ %"))
        self.matrix_min_occ_spin = QtWidgets.QDoubleSpinBox()
        self.matrix_min_occ_spin.setRange(0, 100)
        self.matrix_min_occ_spin.setDecimals(1)
        self.matrix_min_occ_spin.setValue(0.0)
        matrix_controls.addWidget(self.matrix_min_occ_spin)
        matrix_controls.addStretch(1)
        layout.addWidget(self.matrix_controls_row)

        self.event_controls_row = QtWidgets.QWidget()
        event_controls = QtWidgets.QHBoxLayout(self.event_controls_row)
        event_controls.setSpacing(8)
        event_controls.setContentsMargins(4, 0, 4, 4)
        event_controls.addWidget(QtWidgets.QLabel("Event stride"))
        self.event_stride_spin = QtWidgets.QSpinBox()
        self.event_stride_spin.setRange(1, 1000)
        self.event_stride_spin.setValue(1)
        self.event_segment_check = QtWidgets.QCheckBox("Segments")
        self.event_min_duration_spin = QtWidgets.QSpinBox()
        self.event_min_duration_spin.setRange(1, 10_000)
        self.event_min_duration_spin.setValue(1)
        event_controls.addWidget(self.event_stride_spin)
        event_controls.addWidget(self.event_segment_check)
        event_controls.addWidget(QtWidgets.QLabel("Min duration"))
        event_controls.addWidget(self.event_min_duration_spin)
        event_controls.addStretch(1)
        layout.addWidget(self.event_controls_row)

        self.plot_actions_row = QtWidgets.QWidget()
        plot_actions = QtWidgets.QHBoxLayout(self.plot_actions_row)
        plot_actions.setSpacing(8)
        plot_actions.setContentsMargins(4, 0, 4, 4)
        plot_actions.addStretch(1)
        plot_actions.addWidget(save_btn)
        plot_actions.addWidget(self.plots_copy_btn)
        layout.addWidget(self.plot_actions_row)

        self.plots_tabs = QtWidgets.QTabWidget()
        self.hist_zero_plot = pg.PlotWidget()
        self._style_plot(self.hist_zero_plot, "Zero vs Non-zero")
        self.hist_zero_plot.setMaximumHeight(140)
        self.hist_zero_plot.setLabel("bottom", "Category")
        self.hist_zero_plot.setLabel("left", "Count")
        self.hist_plot = pg.PlotWidget()
        self._style_plot(self.hist_plot, "Distribution")
        self.hist_plot.setLabel("bottom", "Value")
        self.hist_plot.setLabel("left", "Count")
        self.hist_info = QtWidgets.QLabel()
        self.hist_info.setWordWrap(True)
        hist_container = QtWidgets.QWidget()
        hist_layout = QtWidgets.QVBoxLayout(hist_container)
        hist_layout.addWidget(self.hist_zero_plot)
        hist_layout.addWidget(self.hist_plot)
        hist_layout.addWidget(self.hist_info)
        self.matrix_view = pg.ImageView()
        self.matrix_view.ui.roiBtn.hide()
        self.matrix_view.ui.menuBtn.hide()
        try:
            self.matrix_view.setBackground(self._get_theme_tokens()["plot_bg"])
        except Exception:
            pass
        try:
            self.matrix_view.ui.histogram.setMinimumWidth(120)
        except Exception:
            pass
        self.matrix_info = QtWidgets.QLabel()
        self.matrix_info.setWordWrap(True)
        matrix_container = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QVBoxLayout(matrix_container)
        matrix_layout.addWidget(self.matrix_view)
        matrix_layout.addWidget(self.matrix_info)
        self.event_plot = pg.PlotWidget()
        self._style_plot(self.event_plot, "Occupancy Events")
        self.event_plot.setLabel("bottom", "Time", units="ns")
        self.event_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.event_plot.setLabel("left", "Solvent rank")
        self.event_info = QtWidgets.QLabel()
        self.event_info.setWordWrap(True)
        event_container = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(event_container)
        event_layout.addWidget(self.event_plot)
        event_layout.addWidget(self.event_info)
        self.plots_tabs.addTab(hist_container, "Histogram")
        self.plots_tabs.addTab(matrix_container, "Matrix / Heatmap")
        self.plots_tabs.addTab(event_container, "Event Raster")
        layout.addWidget(self.plots_tabs)
        self.plots_tabs.currentChanged.connect(self._update_plots_controls_state)

        self.hist_norm_check.toggled.connect(self._queue_hist_update)
        self.hist_log_check.toggled.connect(self._queue_hist_update)
        self.hist_zero_split_check.toggled.connect(self._queue_hist_update)
        self.matrix_source_combo.currentTextChanged.connect(self._queue_matrix_update)
        self.matrix_top_spin.valueChanged.connect(self._queue_matrix_update)
        self.matrix_min_occ_spin.valueChanged.connect(self._queue_matrix_update)
        self.matrix_top_spin.valueChanged.connect(self._queue_event_update)
        self.matrix_min_occ_spin.valueChanged.connect(self._queue_event_update)
        self.event_stride_spin.valueChanged.connect(self._queue_event_update)
        self.event_segment_check.toggled.connect(self._queue_event_update)
        self.event_min_duration_spin.valueChanged.connect(self._queue_event_update)

        self.hist_metric_combo.setToolTip("Per-frame metric for histogram.")
        self.hist_bins_spin.setToolTip("Number of histogram bins.")
        self.hist_zero_split_check.setToolTip("Show zeros separately and plot non-zero distribution.")
        self.hist_norm_check.setToolTip("Normalize histogram to sum to 1.")
        self.hist_log_check.setToolTip("Log scale on Y axis.")
        self.matrix_source_combo.setToolTip("Heatmap source: occupancy or residue hydration.")
        self.matrix_top_spin.setToolTip("Number of top solvents to show in heatmap.")
        self.matrix_min_occ_spin.setToolTip("Minimum occupancy percent to include in heatmap.")
        self.event_stride_spin.setToolTip("Plot every Nth frame in the event raster.")
        self.event_segment_check.setToolTip("Show continuous occupancy segments instead of dots.")
        self.event_min_duration_spin.setToolTip("Minimum duration (frames) for segment display.")
        self._update_plots_controls_state(self.plots_tabs.currentIndex())
        return panel

    def _build_tables_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter"))
        self.table_filter = QtWidgets.QLineEdit()
        self.table_filter.setPlaceholderText("Type to filter rows")
        self.table_filter.textChanged.connect(self._apply_table_filter)
        filter_row.addWidget(self.table_filter)
        layout.addLayout(filter_row)

        self.tables_tabs = QtWidgets.QTabWidget()
        self.per_frame_table = QtWidgets.QTableView()
        self.per_solvent_table = QtWidgets.QTableView()
        self.per_frame_table.setSortingEnabled(True)
        self.per_solvent_table.setSortingEnabled(True)
        self.per_frame_table.setAlternatingRowColors(True)
        self.per_solvent_table.setAlternatingRowColors(True)
        self.per_frame_table.horizontalHeader().setStretchLastSection(True)
        self.per_solvent_table.horizontalHeader().setStretchLastSection(True)
        self._configure_table_view_header(self.per_frame_table)
        self._configure_table_view_header(self.per_solvent_table)
        self.tables_tabs.addTab(self.per_frame_table, "Per Frame")
        self.tables_tabs.addTab(self.per_solvent_table, "Per Solvent")
        layout.addWidget(self.tables_tabs)
        return panel

    def _build_report_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.report_text = QtWidgets.QTextEdit()
        self.report_text.setReadOnly(True)
        layout.addWidget(self.report_text)
        return panel

    def _build_logs_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        top_row = QtWidgets.QHBoxLayout()
        self.log_path_label = QtWidgets.QLabel("Log: -")
        self.log_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.log_refresh_btn.clicked.connect(self._refresh_log_view)
        self.log_open_btn = QtWidgets.QPushButton("Open Log")
        self.log_open_btn.clicked.connect(self._open_log_file)
        self.log_copy_errors_btn = QtWidgets.QPushButton("Copy Errors")
        self.log_copy_errors_btn.clicked.connect(self._copy_log_errors)
        top_row.addWidget(self.log_path_label)
        top_row.addStretch(1)
        top_row.addWidget(self.log_refresh_btn)
        top_row.addWidget(self.log_open_btn)
        top_row.addWidget(self.log_copy_errors_btn)
        layout.addLayout(top_row)
        filter_row = QtWidgets.QHBoxLayout()
        self.log_level_combo = QtWidgets.QComboBox()
        self.log_level_combo.addItems(["All", "INFO", "WARNING", "ERROR"])
        self.log_search_edit = QtWidgets.QLineEdit()
        self.log_search_edit.setPlaceholderText("Search logs")
        self.log_collapse_check = QtWidgets.QCheckBox("Collapse tracebacks")
        self.log_collapse_check.setChecked(True)
        self.log_level_combo.currentTextChanged.connect(self._apply_log_filter)
        self.log_search_edit.textChanged.connect(self._apply_log_filter)
        self.log_collapse_check.toggled.connect(self._apply_log_filter)
        filter_row.addWidget(QtWidgets.QLabel("Level"))
        filter_row.addWidget(self.log_level_combo)
        filter_row.addWidget(self.log_search_edit)
        filter_row.addWidget(self.log_collapse_check)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)
        self.log_summary_label = QtWidgets.QLabel("No log loaded.")
        self.log_summary_label.setWordWrap(True)
        layout.addWidget(self.log_summary_label)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("JetBrains Mono"))
        layout.addWidget(self.log_text)
        return panel

    def _build_extract_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        form = QtWidgets.QFormLayout()
        self.extract_mode_combo = QtWidgets.QComboBox()
        self.extract_mode_combo.addItems(["Threshold", "Percentile", "Top N frames"])
        self.extract_soz_combo = QtWidgets.QComboBox()
        self.extract_metric_combo = QtWidgets.QComboBox()
        self.extract_metric_combo.addItems(["n_solvent", "occupancy_fraction"])
        self.extract_op_combo = QtWidgets.QComboBox()
        self.extract_op_combo.addItems([">=", ">", "==", "<=", "<"])
        self.extract_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.extract_threshold_spin.setRange(0, 1_000_000)
        self.extract_threshold_spin.setValue(1.0)
        self.extract_percentile_spin = QtWidgets.QDoubleSpinBox()
        self.extract_percentile_spin.setRange(0.0, 100.0)
        self.extract_percentile_spin.setDecimals(1)
        self.extract_percentile_spin.setSingleStep(1.0)
        self.extract_percentile_spin.setValue(90.0)
        self.extract_topn_spin = QtWidgets.QSpinBox()
        self.extract_topn_spin.setRange(1, 1_000_000)
        self.extract_topn_spin.setValue(100)
        self.extract_min_run_spin = QtWidgets.QSpinBox()
        self.extract_min_run_spin.setRange(1, 1_000_000)
        self.extract_min_run_spin.setValue(1)
        self.extract_gap_spin = QtWidgets.QSpinBox()
        self.extract_gap_spin.setRange(0, 1_000_000)
        self.extract_gap_spin.setValue(0)
        self.extract_format_combo = QtWidgets.QComboBox()
        self.extract_format_combo.addItems(["xtc"])
        self.extract_output_edit = QtWidgets.QLineEdit()
        self.extract_output_edit.setPlaceholderText("Uses Output Settings directory")
        self.extract_output_btn = QtWidgets.QPushButton("Browse")
        self.extract_output_btn.clicked.connect(self._browse_extract_output)
        self.extract_link_check = QtWidgets.QCheckBox("Use Output Settings directory")
        self.extract_link_check.setChecked(True)
        self.extract_link_check.toggled.connect(self._toggle_extract_output_link)
        self.extract_output_edit.textChanged.connect(self._on_extract_output_edited)

        self.extract_mode_label = QtWidgets.QLabel("Mode")
        self.extract_metric_label = QtWidgets.QLabel("Metric")
        self.extract_op_label = QtWidgets.QLabel("Operator")
        self.extract_threshold_label = QtWidgets.QLabel("Threshold")
        self.extract_percentile_label = QtWidgets.QLabel("Percentile")
        self.extract_topn_label = QtWidgets.QLabel("Top N frames")
        self.extract_min_run_label = QtWidgets.QLabel("Min run length")
        self.extract_gap_label = QtWidgets.QLabel("Gap tolerance")
        self.extract_format_label = QtWidgets.QLabel("Format")

        form.addRow(self.extract_mode_label, self.extract_mode_combo)
        form.addRow("SOZ", self.extract_soz_combo)
        form.addRow(self.extract_metric_label, self.extract_metric_combo)
        form.addRow(self.extract_op_label, self.extract_op_combo)
        form.addRow(self.extract_threshold_label, self.extract_threshold_spin)
        form.addRow(self.extract_percentile_label, self.extract_percentile_spin)
        form.addRow(self.extract_topn_label, self.extract_topn_spin)
        form.addRow(self.extract_min_run_label, self.extract_min_run_spin)
        form.addRow(self.extract_gap_label, self.extract_gap_spin)
        form.addRow(self.extract_format_label, self.extract_format_combo)
        self.extract_rule_preview = QtWidgets.QLabel()
        self.extract_rule_preview.setWordWrap(True)
        form.addRow("Rule preview", self.extract_rule_preview)
        self.extract_rule_note = QtWidgets.QLabel()
        self.extract_rule_note.setWordWrap(True)
        form.addRow("Rule note", self.extract_rule_note)
        self.extract_metric_info = QtWidgets.QLabel()
        self.extract_metric_info.setWordWrap(True)
        form.addRow("Metric stats", self.extract_metric_info)
        self.extract_rule_help = QtWidgets.QLabel(
            "Frames are kept when the rule is true. Increasing the threshold usually reduces the number of frames."
        )
        self.extract_rule_help.setWordWrap(True)
        form.addRow("Rule help", self.extract_rule_help)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.extract_output_edit)
        output_row.addWidget(self.extract_output_btn)
        form.addRow("Output directory", output_row)
        form.addRow("", self.extract_link_check)

        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.extract_preview_btn = QtWidgets.QPushButton("Preview")
        self.extract_run_btn = QtWidgets.QPushButton("Extract")
        self.extract_preview_btn.clicked.connect(self._preview_extraction)
        self.extract_run_btn.clicked.connect(self._run_extraction)
        buttons.addWidget(self.extract_preview_btn)
        buttons.addWidget(self.extract_run_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.extract_progress = QtWidgets.QProgressBar()
        self.extract_progress.setVisible(False)
        self.extract_progress.setFormat("Extraction: %p%")
        layout.addWidget(self.extract_progress)

        self.extract_summary = QtWidgets.QTextEdit()
        self.extract_summary.setReadOnly(True)
        layout.addWidget(self.extract_summary)

        self.extract_table = QtWidgets.QTableWidget()
        self.extract_table.setColumnCount(3)
        self._configure_table_headers(self.extract_table, ["frame", "time (ps)", "n_solvent"])
        layout.addWidget(self.extract_table)

        self._extract_selection = None
        self._last_extract_outputs = None

        self.extract_mode_combo.currentTextChanged.connect(self._update_extract_mode_ui)
        self.extract_metric_combo.currentTextChanged.connect(self._on_extract_metric_changed)
        self.extract_op_combo.currentTextChanged.connect(self._update_extract_rule_preview)
        self.extract_threshold_spin.valueChanged.connect(self._update_extract_rule_preview)
        self.extract_percentile_spin.valueChanged.connect(self._update_extract_rule_preview)
        self.extract_topn_spin.valueChanged.connect(self._update_extract_rule_preview)
        self._update_extract_mode_ui()
        self._on_extract_metric_changed()
        self._toggle_extract_output_link(True)

        return panel

    def _build_builder_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.builder_tabs = QtWidgets.QTabWidget()
        self.builder_tabs.addTab(self._build_wizard_tab(), "Wizard")
        self.builder_tabs.addTab(self._build_selection_builder_tab(), "Selection Builder")
        self.builder_tabs.addTab(self._build_bridges_tab(), "Bridges")
        self.builder_tabs.addTab(self._build_hydration_tab(), "Residue Hydration")
        self.builder_tabs.addTab(self._build_advanced_tab(), "Advanced")
        self.builder_tabs.addTab(self._build_selection_tester_tab(), "Selection Tester")

        layout.addWidget(self.builder_tabs)
        return panel

    def _build_wizard_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(panel)

        self.wizard_soz_name = QtWidgets.QLineEdit("SOZ_1")
        self.wizard_water_resnames = QtWidgets.QLineEdit("SOL,WAT,TIP3,HOH")
        self.wizard_water_oxygen = QtWidgets.QLineEdit("O,OW,OH2")
        self.wizard_include_ions = QtWidgets.QCheckBox("Include ions")
        self.wizard_ion_resnames = QtWidgets.QLineEdit("NA,CL,K,CA,MG")
        self.wizard_seed_a = QtWidgets.QLineEdit("protein and resid 10 and name CA")
        self.wizard_seed_a_unique = QtWidgets.QCheckBox("Require unique match")
        self.wizard_seed_a_unique.setChecked(True)
        self.wizard_seed_a_unique.setToolTip(
            "Require the selection to resolve to exactly one atom."
        )
        self.wizard_seed_b = QtWidgets.QLineEdit("")
        self.wizard_seed_b_unique = QtWidgets.QCheckBox("Require unique match")
        self.wizard_seed_b_unique.setChecked(True)
        self.wizard_seed_b_unique.setToolTip(
            "Require the selection to resolve to exactly one atom."
        )
        self.wizard_shell_cutoffs = QtWidgets.QLineEdit("3.5,5.0")
        self.wizard_atom_mode = QtWidgets.QComboBox()
        self.wizard_atom_mode.addItems(["O", "all"])
        self.wizard_boolean = QtWidgets.QComboBox()
        self.wizard_boolean.addItems(["AND", "OR"])
        self.wizard_b_cutoff = QtWidgets.QLineEdit("3.5")

        layout.addRow("SOZ name", self.wizard_soz_name)
        layout.addRow("Water resnames", self.wizard_water_resnames)
        layout.addRow("Water oxygen names", self.wizard_water_oxygen)
        layout.addRow("Include ions", self.wizard_include_ions)
        layout.addRow("Ion resnames", self.wizard_ion_resnames)
        layout.addRow("Selection A", self.wizard_seed_a)
        layout.addRow("Selection A unique match", self.wizard_seed_a_unique)
        self.wizard_seed_a_status = QtWidgets.QLabel("Selection A matches: -")
        self.wizard_seed_a_status.setWordWrap(True)
        layout.addRow("", self.wizard_seed_a_status)
        layout.addRow("Shell cutoffs (A)", self.wizard_shell_cutoffs)
        layout.addRow("Shell atom mode", self.wizard_atom_mode)
        layout.addRow("Selection B (optional)", self.wizard_seed_b)
        layout.addRow("Selection B unique match", self.wizard_seed_b_unique)
        self.wizard_seed_b_status = QtWidgets.QLabel("Selection B matches: -")
        self.wizard_seed_b_status.setWordWrap(True)
        layout.addRow("", self.wizard_seed_b_status)
        layout.addRow("Selection B cutoff (A)", self.wizard_b_cutoff)
        layout.addRow("Selection B combine", self.wizard_boolean)

        self.wizard_explain = QtWidgets.QTextEdit()
        self.wizard_explain.setReadOnly(True)
        layout.addRow("Explain my SOZ", self.wizard_explain)

        preview_btn = QtWidgets.QPushButton("Preview")
        preview_btn.clicked.connect(self._preview_soz)
        layout.addRow(preview_btn)

        self.seed_validation_group = QtWidgets.QGroupBox("Selection match preview")
        seed_layout = QtWidgets.QVBoxLayout(self.seed_validation_group)
        seed_options = QtWidgets.QHBoxLayout()
        self.seed_validation_live = QtWidgets.QCheckBox("Live")
        self.seed_validation_live.setChecked(True)
        self.seed_validation_use_traj = QtWidgets.QCheckBox("Use trajectory")
        self.seed_validation_use_traj.setChecked(True)
        self.seed_validation_limit_spin = QtWidgets.QSpinBox()
        self.seed_validation_limit_spin.setRange(1, 200)
        self.seed_validation_limit_spin.setValue(10)
        self.seed_validation_target_combo = QtWidgets.QComboBox()
        self.seed_validation_target_combo.addItems(["Selection A", "Selection B"])
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._run_seed_validation)
        seed_options.addWidget(self.seed_validation_live)
        seed_options.addWidget(self.seed_validation_use_traj)
        seed_options.addWidget(QtWidgets.QLabel("Max rows"))
        seed_options.addWidget(self.seed_validation_limit_spin)
        seed_options.addWidget(QtWidgets.QLabel("Preview"))
        seed_options.addWidget(self.seed_validation_target_combo)
        seed_options.addStretch(1)
        seed_options.addWidget(refresh_btn)
        seed_layout.addLayout(seed_options)
        self.seed_validation_table = QtWidgets.QTableWidget()
        self.seed_validation_table.setColumnCount(7)
        self._configure_table_headers(
            self.seed_validation_table,
            [
                "Atom index",
                "Atom name",
                "Residue name",
                "Residue number",
                "Segment ID",
                "Chain ID",
                "Molecule type",
            ],
        )
        seed_layout.addWidget(self.seed_validation_table)
        layout.addRow(self.seed_validation_group)

        self.wizard_seed_a.textChanged.connect(self._schedule_seed_validation)
        self.wizard_seed_b.textChanged.connect(self._schedule_seed_validation)
        self.wizard_seed_a_unique.toggled.connect(self._schedule_seed_validation)
        self.wizard_seed_b_unique.toggled.connect(self._schedule_seed_validation)
        self.seed_validation_live.toggled.connect(self._schedule_seed_validation)
        self.seed_validation_use_traj.toggled.connect(self._schedule_seed_validation)
        self.seed_validation_limit_spin.valueChanged.connect(self._schedule_seed_validation)
        self.seed_validation_target_combo.currentTextChanged.connect(self._run_seed_validation)

        return panel

    def _build_selection_builder_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        form = QtWidgets.QFormLayout()
        self.sel_builder_scope = QtWidgets.QComboBox()
        self.sel_builder_scope.addItems(["all atoms", "protein", "backbone", "sidechain", "nucleic"])
        self.sel_builder_resname = QtWidgets.QLineEdit()
        self.sel_builder_resname.setPlaceholderText("e.g., HSD or HS*")
        self.sel_builder_resid_mode = QtWidgets.QComboBox()
        self.sel_builder_resid_mode.addItems(["resid", "resnum"])
        self.sel_builder_resid = QtWidgets.QLineEdit()
        self.sel_builder_resid.setPlaceholderText("e.g., 223 or 10:20")
        self.sel_builder_atomname = QtWidgets.QLineEdit()
        self.sel_builder_atomname.setPlaceholderText("e.g., NE2")
        self.sel_builder_segid = QtWidgets.QLineEdit()
        self.sel_builder_segid.setPlaceholderText("e.g., seg_0_PROA")
        self.sel_builder_chainid = QtWidgets.QLineEdit()
        self.sel_builder_chainid.setPlaceholderText("e.g., A")

        form.addRow("Scope", self.sel_builder_scope)
        form.addRow("Resname", self.sel_builder_resname)
        resid_row = QtWidgets.QHBoxLayout()
        resid_row.addWidget(self.sel_builder_resid_mode)
        resid_row.addWidget(self.sel_builder_resid)
        form.addRow("Resid/resnum", resid_row)
        form.addRow("Atom name", self.sel_builder_atomname)
        form.addRow("Segid", self.sel_builder_segid)
        form.addRow("ChainID", self.sel_builder_chainid)
        layout.addLayout(form)

        self.sel_builder_output = QtWidgets.QLineEdit()
        self.sel_builder_output.setReadOnly(True)
        self.sel_builder_output.setPlaceholderText("Selection string will appear here")
        layout.addWidget(self.sel_builder_output)

        buttons = QtWidgets.QHBoxLayout()
        build_btn = QtWidgets.QPushButton("Build")
        build_btn.clicked.connect(self._update_selection_builder_output)
        seed_a_btn = QtWidgets.QPushButton("Use as Selection A")
        seed_b_btn = QtWidgets.QPushButton("Use as Selection B")
        tester_btn = QtWidgets.QPushButton("Send to Tester")
        seed_a_btn.clicked.connect(lambda: self._apply_builder_to_seed("A"))
        seed_b_btn.clicked.connect(lambda: self._apply_builder_to_seed("B"))
        tester_btn.clicked.connect(self._apply_builder_to_tester)
        buttons.addWidget(build_btn)
        buttons.addWidget(seed_a_btn)
        buttons.addWidget(seed_b_btn)
        buttons.addWidget(tester_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        for widget in (
            self.sel_builder_scope,
            self.sel_builder_resname,
            self.sel_builder_resid_mode,
            self.sel_builder_resid,
            self.sel_builder_atomname,
            self.sel_builder_segid,
            self.sel_builder_chainid,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._update_selection_builder_output)
            else:
                widget.textChanged.connect(self._update_selection_builder_output)

        self._update_selection_builder_output()
        return panel

    def _build_bridges_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Bridge")
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        add_btn.clicked.connect(self._add_bridge_row)
        remove_btn.clicked.connect(self._remove_bridge_row)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.bridge_table = QtWidgets.QTableWidget()
        self.bridge_table.setColumnCount(7)
        self._configure_table_headers(
            self.bridge_table,
            ["name", "selection_a", "selection_b", "cutoff_a", "cutoff_b", "unit", "atom_mode"],
        )
        self.bridge_table.itemChanged.connect(self._on_bridge_table_changed)
        layout.addWidget(self.bridge_table)
        return panel

    def _build_hydration_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Hydration")
        remove_btn = QtWidgets.QPushButton("Remove Selected")
        add_btn.clicked.connect(self._add_hydration_row)
        remove_btn.clicked.connect(self._remove_hydration_row)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.hydration_table = QtWidgets.QTableWidget()
        self.hydration_table.setColumnCount(5)
        self._configure_table_headers(
            self.hydration_table,
            ["name", "residue_selection", "cutoff", "unit", "soz_name"],
        )
        self.hydration_table.itemChanged.connect(self._on_hydration_table_changed)
        layout.addWidget(self.hydration_table)
        return panel

    def _update_selection_builder_output(self) -> None:
        if not hasattr(self, "sel_builder_scope"):
            return
        parts = []
        scope = self.sel_builder_scope.currentText()
        if scope and scope != "all atoms":
            parts.append(scope)
        resname = self.sel_builder_resname.text().strip()
        resid_mode = self.sel_builder_resid_mode.currentText()
        resid_val = self.sel_builder_resid.text().strip()
        atomname = self.sel_builder_atomname.text().strip()
        segid = self.sel_builder_segid.text().strip()
        chainid = self.sel_builder_chainid.text().strip()
        if resname:
            parts.append(f"resname {resname}")
        if resid_val:
            parts.append(f"{resid_mode} {resid_val}")
        if atomname:
            parts.append(f"name {atomname}")
        if segid:
            parts.append(f"segid {segid}")
        if chainid:
            parts.append(f"chainID {chainid}")
        selection = " and ".join(parts)
        self.sel_builder_output.setText(selection)

    def _apply_builder_to_seed(self, which: str) -> None:
        selection = self.sel_builder_output.text().strip()
        if not selection:
            return
        if which == "A":
            self.wizard_seed_a.setText(selection)
            self.builder_tabs.setCurrentIndex(0)
        elif which == "B":
            self.wizard_seed_b.setText(selection)
            self.builder_tabs.setCurrentIndex(0)

    def _apply_builder_to_tester(self) -> None:
        selection = self.sel_builder_output.text().strip()
        if not selection:
            return
        self.selection_input.setText(selection)
        if hasattr(self, "selection_tester_tab"):
            self.builder_tabs.setCurrentWidget(self.selection_tester_tab)

    def _refresh_bridge_table(self) -> None:
        if not hasattr(self, "bridge_table"):
            return
        project = self.state.project
        if not project:
            return
        self._bridge_table_refreshing = True
        self.bridge_table.setRowCount(0)
        self.bridge_table.setRowCount(len(project.bridges))
        for row, bridge in enumerate(project.bridges):
            values = [
                bridge.name,
                bridge.selection_a,
                bridge.selection_b,
                bridge.cutoff_a,
                bridge.cutoff_b,
                bridge.unit,
                bridge.atom_mode,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.bridge_table.setItem(row, col, item)
        self._bridge_table_refreshing = False

    def _refresh_hydration_table(self) -> None:
        if not hasattr(self, "hydration_table"):
            return
        project = self.state.project
        if not project:
            return
        self._hydration_table_refreshing = True
        self.hydration_table.setRowCount(0)
        self.hydration_table.setRowCount(len(project.residue_hydration))
        for row, cfg in enumerate(project.residue_hydration):
            values = [
                cfg.name,
                cfg.residue_selection,
                cfg.cutoff,
                cfg.unit,
                cfg.soz_name or "",
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.hydration_table.setItem(row, col, item)
        self._hydration_table_refreshing = False

    def _add_bridge_row(self) -> None:
        if not self._ensure_project():
            return
        row = self.bridge_table.rowCount()
        self.bridge_table.insertRow(row)
        defaults = ["bridge", "selection_a", "selection_b", "3.5", "3.5", "A", "O"]
        for col, value in enumerate(defaults):
            self.bridge_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self._on_bridge_table_changed()

    def _remove_bridge_row(self) -> None:
        rows = sorted({idx.row() for idx in self.bridge_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.bridge_table.removeRow(row)
        self._on_bridge_table_changed()

    def _add_hydration_row(self) -> None:
        if not self._ensure_project():
            return
        row = self.hydration_table.rowCount()
        self.hydration_table.insertRow(row)
        defaults = ["hydration", "protein", "3.5", "A", ""]
        for col, value in enumerate(defaults):
            self.hydration_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self._on_hydration_table_changed()

    def _remove_hydration_row(self) -> None:
        rows = sorted({idx.row() for idx in self.hydration_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.hydration_table.removeRow(row)
        self._on_hydration_table_changed()

    def _on_bridge_table_changed(self) -> None:
        if getattr(self, "_bridge_table_refreshing", False):
            return
        if not self.state.project:
            return
        bridges = []
        for row in range(self.bridge_table.rowCount()):
            values = []
            for col in range(self.bridge_table.columnCount()):
                item = self.bridge_table.item(row, col)
                values.append(item.text().strip() if item else "")
            if not any(values):
                continue
            try:
                bridge = BridgeConfig(
                    name=values[0] or f"bridge_{row+1}",
                    selection_a=values[1] or "selection_a",
                    selection_b=values[2] or "selection_b",
                    cutoff_a=float(values[3] or 3.5),
                    cutoff_b=float(values[4] or 3.5),
                    unit=values[5] or "A",
                    atom_mode=values[6] or "O",
                )
                bridges.append(bridge)
            except Exception:
                continue
        self.state.project.bridges = bridges

    def _on_hydration_table_changed(self) -> None:
        if getattr(self, "_hydration_table_refreshing", False):
            return
        if not self.state.project:
            return
        configs = []
        for row in range(self.hydration_table.rowCount()):
            values = []
            for col in range(self.hydration_table.columnCount()):
                item = self.hydration_table.item(row, col)
                values.append(item.text().strip() if item else "")
            if not any(values):
                continue
            try:
                cfg = ResidueHydrationConfig(
                    name=values[0] or f"hydration_{row+1}",
                    residue_selection=values[1] or "protein",
                    cutoff=float(values[2] or 3.5),
                    unit=values[3] or "A",
                    soz_name=values[4] or None,
                )
                configs.append(cfg)
            except Exception:
                continue
        self.state.project.residue_hydration = configs

    def _build_advanced_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.advanced_json = QtWidgets.QPlainTextEdit()
        self.advanced_json.setPlaceholderText("SOZ JSON definition")
        apply_btn = QtWidgets.QPushButton("Apply JSON to SOZ list")
        apply_btn.clicked.connect(self._apply_advanced_json)
        layout.addWidget(self.advanced_json)
        layout.addWidget(apply_btn)
        return panel

    def _build_selection_tester_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        self.selection_tester_tab = panel
        layout = QtWidgets.QVBoxLayout(panel)

        input_row = QtWidgets.QHBoxLayout()
        self.selection_input = QtWidgets.QLineEdit()
        self.selection_input.setPlaceholderText("MDAnalysis selection string")
        input_row.addWidget(self.selection_input)

        seed_a_btn = QtWidgets.QPushButton("Use Selection A")
        seed_b_btn = QtWidgets.QPushButton("Use Selection B")
        seed_a_btn.clicked.connect(lambda: self.selection_input.setText(self.wizard_seed_a.text().strip()))
        seed_b_btn.clicked.connect(lambda: self.selection_input.setText(self.wizard_seed_b.text().strip()))
        input_row.addWidget(seed_a_btn)
        input_row.addWidget(seed_b_btn)

        layout.addLayout(input_row)

        options_row = QtWidgets.QHBoxLayout()
        self.selection_limit_spin = QtWidgets.QSpinBox()
        self.selection_limit_spin.setRange(1, 500)
        self.selection_limit_spin.setValue(25)
        self.selection_use_trajectory = QtWidgets.QCheckBox("Load trajectory")
        self.selection_use_trajectory.setChecked(True)
        test_btn = QtWidgets.QPushButton("Test Selection")
        test_btn.clicked.connect(self._test_selection)
        options_row.addWidget(QtWidgets.QLabel("Max rows"))
        options_row.addWidget(self.selection_limit_spin)
        options_row.addWidget(self.selection_use_trajectory)
        options_row.addWidget(test_btn)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        self.selection_results = QtWidgets.QTextEdit()
        self.selection_results.setReadOnly(True)
        layout.addWidget(self.selection_results)

        self.selection_table = QtWidgets.QTableWidget()
        self.selection_table.setColumnCount(8)
        self._configure_table_headers(
            self.selection_table,
            ["index", "resname", "resid", "resnum", "segid", "chainID", "moltype", "name"],
        )
        layout.addWidget(self.selection_table)

        self._tester_universe = None
        self._tester_key = None

        return panel

    def _load_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project files (*.json *.yaml *.yml);;JSON (*.json);;YAML (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        project = load_project_json(path)
        self.state = ProjectState(project=project, path=Path(path))
        self._refresh_project_ui()

    def _save_project(self) -> None:
        if not self.state.project:
            return
        if not self.state.path:
            path, selected = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Project",
                "",
                "JSON (*.json);;YAML (*.yaml *.yml)",
            )
            if not path:
                return
            if not Path(path).suffix:
                if "yaml" in selected.lower():
                    path += ".yaml"
                else:
                    path += ".json"
            self.state.path = Path(path)
        write_project_json(self.state.project, str(self.state.path))
        self.status_bar.showMessage(f"Saved project to {self.state.path}", 5000)

    def _refresh_project_ui(self) -> None:
        project = self.state.project
        if not project:
            return
        self.topology_label.setText(f"Topology: {project.inputs.topology}")
        self.trajectory_label.setText(f"Trajectory: {project.inputs.trajectory}")
        meta_text = (
            f"SOZs: {len(project.sozs)} | Selections: {len(project.selections)} "
            f"| Stride: {project.analysis.stride}"
        )
        try:
            import MDAnalysis as mda

            if project.inputs.trajectory:
                universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
            else:
                universe = mda.Universe(project.inputs.topology)
            n_frames = len(universe.trajectory)
            dt = getattr(universe.trajectory, "dt", None)
            dims = universe.trajectory.ts.dimensions
            has_box = bool(dims is not None and all(val > 0 for val in dims[:3]))
            meta_text += f" | Frames: {n_frames} | dt: {dt} | PBC: {has_box}"
            self._last_dt = dt
        except Exception:
            pass
        self.metadata_label.setText(meta_text)
        self._update_project_summary(meta_text)
        self.frame_start_spin.setValue(project.analysis.frame_start)
        self.frame_stop_spin.setValue(
            project.analysis.frame_stop if project.analysis.frame_stop is not None else -1
        )
        self.frame_stride_spin.setValue(project.analysis.stride)
        self._refresh_output_controls()
        self._refresh_bridge_table()
        self._refresh_hydration_table()
        self.soz_list.clear()
        for soz in project.sozs:
            self.soz_list.addItem(soz.name)
        self._update_explain_text()
        try:
            self._wizard_snapshot = self._wizard_state()
        except Exception:
            pass
        self._update_provenance_stamp()

    def _refresh_output_controls(self) -> None:
        project = self.state.project
        if not project:
            return
        self.output_dir_edit.blockSignals(True)
        self.report_format_combo.blockSignals(True)
        self.write_per_frame_check.blockSignals(True)
        self.write_parquet_check.blockSignals(True)
        self.output_dir_edit.setText(project.outputs.output_dir)
        self.report_format_combo.setCurrentText(project.outputs.report_format)
        self.write_per_frame_check.setChecked(project.outputs.write_per_frame)
        self.write_parquet_check.setChecked(project.outputs.write_parquet)
        self.output_dir_edit.blockSignals(False)
        self.report_format_combo.blockSignals(False)
        self.write_per_frame_check.blockSignals(False)
        self.write_parquet_check.blockSignals(False)
        # Effective output label removed.
        if getattr(self, "_extract_output_linked", False):
            try:
                self.extract_output_edit.blockSignals(True)
                self.extract_output_edit.setText(project.outputs.output_dir)
            finally:
                self.extract_output_edit.blockSignals(False)

    def _on_output_settings_changed(self) -> None:
        if not self.state.project:
            return
        prev_output = self.state.project.outputs.output_dir
        output_dir = self.output_dir_edit.text().strip() or "results"
        report_format = self.report_format_combo.currentText() or "html"
        self.state.project.outputs.output_dir = output_dir
        self.state.project.outputs.report_format = report_format
        self.state.project.outputs.write_per_frame = self.write_per_frame_check.isChecked()
        self.state.project.outputs.write_parquet = self.write_parquet_check.isChecked()
        if output_dir != prev_output:
            self.log_path = None
        # Effective output label removed.
        if getattr(self, "_extract_output_linked", False):
            try:
                self.extract_output_edit.blockSignals(True)
                self.extract_output_edit.setText(output_dir)
            finally:
                self.extract_output_edit.blockSignals(False)
        self._refresh_log_view()

    def _effective_output_dir(self) -> Path | None:
        if not hasattr(self, "output_dir_edit"):
            return None
        raw = self.output_dir_edit.text().strip()
        if not raw:
            return None
        try:
            return Path(raw).expanduser().resolve()
        except Exception:
            return Path(raw)

    def _toggle_extract_output_link(self, enabled: bool) -> None:
        self._extract_output_linked = bool(enabled)
        if not hasattr(self, "extract_output_edit"):
            return
        if self._extract_output_linked:
            base = self._effective_output_dir()
            if base is not None:
                self.extract_output_edit.setText(str(base))
        self.extract_output_edit.setEnabled(not self._extract_output_linked)
        self.extract_output_btn.setEnabled(not self._extract_output_linked)

    def _browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir_edit.setText(path)

    def _open_directory(self, path_raw: str) -> None:
        if not path_raw:
            return
        try:
            path = Path(path_raw).expanduser().resolve()
        except Exception:
            path = Path(path_raw).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.status_bar.showMessage(f"Unable to create directory: {exc}", 8000)
            return
        url = QtCore.QUrl.fromLocalFile(str(path))
        if not QtGui.QDesktopServices.openUrl(url):
            try:
                import subprocess

                subprocess.run(["xdg-open", str(path)], check=False)
            except Exception as exc:
                self.status_bar.showMessage(f"Unable to open directory: {exc}", 8000)

    def _prompt_export_directory(self, title: str, default_dir: str) -> str | None:
        base = default_dir or "results"
        try:
            base = str(Path(base).expanduser().resolve())
        except Exception:
            pass
        path = QtWidgets.QFileDialog.getExistingDirectory(self, title, base)
        return path or None

    def _open_output_dir(self) -> None:
        path_raw = self.output_dir_edit.text().strip()
        self._open_directory(path_raw)

    def _update_explain_text(self) -> None:
        raw_cutoffs = self.wizard_shell_cutoffs.text().strip() or "3.5"
        cutoff_values = [val.strip() for val in raw_cutoffs.split(",") if val.strip()]
        cutoff_text = []
        for val in cutoff_values:
            try:
                nm = to_nm(float(val), "A")
                cutoff_text.append(f"{val}A ({nm:.3f}nm)")
            except Exception:
                cutoff_text.append(val)
        atom_mode = self.wizard_atom_mode.currentText()
        text = f"shell(selection_a, cutoffs=[{', '.join(cutoff_text)}], atom_mode={atom_mode})"
        if self.wizard_seed_b.text().strip():
            text += (
                f" {self.wizard_boolean.currentText().lower()} "
                f"distance(selection_b, cutoff={self.wizard_b_cutoff.text() or '3.5'}A)"
            )
        self.wizard_explain.setText(text)

    def _toggle_overview_raw(self, visible: bool) -> None:
        self.overview_text.setVisible(visible)

    def _toggle_qc_raw(self, visible: bool) -> None:
        self.qc_text.setVisible(visible)

    def _update_project_summary(self, meta_text: str | None = None) -> None:
        project = self.state.project
        if not project:
            self.project_summary_label.setText("Load a project to view metadata.")
            return
        if meta_text is None:
            meta_text = self.metadata_label.text()
        summary = (
            f"{meta_text}\n"
            f"Inputs: {project.inputs.topology} | {project.inputs.trajectory or 'No trajectory'}\n"
            f"Outputs: {project.outputs.output_dir} | Report: {project.outputs.report_format}"
        )
        self.project_summary_label.setText(summary)

    def _format_qc_summary(self, qc_json: dict) -> str:
        lines = []
        preflight = qc_json.get("preflight", {})
        warnings = preflight.get("warnings", [])
        errors = preflight.get("errors", [])
        if errors:
            lines.append(f"Preflight errors: {len(errors)}")
            lines.extend([f"  - {err}" for err in errors[:5]])
        else:
            lines.append("Preflight: OK")
        if warnings:
            lines.append(f"Warnings: {len(warnings)}")
            lines.extend([f"  - {warn}" for warn in warnings[:5]])
        zero_sozs = qc_json.get("zero_occupancy_sozs", [])
        if zero_sozs:
            lines.append("Zero-occupancy SOZs: " + ", ".join(zero_sozs))
        return "\n".join(lines) if lines else "QC summary unavailable."

    def _update_provenance_stamp(self) -> None:
        project = self.state.project
        if not project:
            if hasattr(self, "provenance_text"):
                self.provenance_text.setText("Load a project to see provenance.")
            return
        lines = []
        lines.append(f"Topology: {project.inputs.topology}")
        lines.append(f"Trajectory: {project.inputs.trajectory or 'None'}")
        if self._last_dt is not None:
            lines.append(f"dt: {self._last_dt}")
        lines.append(f"Frames: start {project.analysis.frame_start} stop {project.analysis.frame_stop or 'end'} stride {project.analysis.stride}")
        lines.append(f"Outputs: {project.outputs.output_dir} | Report: {project.outputs.report_format}")
        lines.append(f"Water resnames: {', '.join(project.solvent.water_resnames)}")
        lines.append(f"Oxygen names: {', '.join(project.solvent.water_oxygen_names)}")
        lines.append(f"Include ions: {project.solvent.include_ions} ({', '.join(project.solvent.ion_resnames)})")
        lines.append(f"Timestamp: {QtCore.QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}")
        def describe_node(node: SOZNode, indent: int = 2) -> list[str]:
            pad = " " * indent
            desc = [f"{pad}{node.type}"]
            if node.type == "shell":
                cutoffs = node.params.get("cutoffs")
                unit = node.params.get("unit", "A")
                if cutoffs is not None:
                    desc.append(f"{pad}  cutoffs: {cutoffs} {unit}")
                desc.append(f"{pad}  atom_mode: {node.params.get('atom_mode')}")
                desc.append(f"{pad}  selection: {node.params.get('selection_label')}")
            if node.type == "distance":
                cutoff = node.params.get("cutoff")
                unit = node.params.get("unit", "A")
                desc.append(f"{pad}  cutoff: {cutoff} {unit}")
                desc.append(f"{pad}  atom_mode: {node.params.get('atom_mode')}")
                desc.append(f"{pad}  selection: {node.params.get('selection_label')}")
            for child in node.children:
                desc.extend(describe_node(child, indent=indent + 2))
            return desc

        for soz in project.sozs:
            lines.append(f"SOZ: {soz.name}")
            lines.extend(describe_node(soz.root, indent=2))
        for label, selection in project.selections.items():
            lines.append(f"Selection {label}: {selection.selection}")
        if self.current_result and self.current_result.qc_summary:
            qc = self.current_result.qc_summary
            lines.append(f"PBC: {qc.get('pbc', {})}")
            if "time_unit" in qc:
                lines.append(f"Time unit: {qc.get('time_unit')}")
            versions = qc.get("versions", {})
            if versions:
                lines.append("Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
        if hasattr(self, "provenance_text"):
            self.provenance_text.setText("\n".join(lines))

    def _copy_provenance(self) -> None:
        if hasattr(self, "provenance_text"):
            QtWidgets.QApplication.clipboard().setText(self.provenance_text.toPlainText())
            self.status_bar.showMessage("Provenance copied to clipboard.", 3000)

    def _add_soz_from_builder(self) -> None:
        if not self._ensure_project():
            return
        self._apply_wizard_to_project(update_existing=True)
        self._refresh_project_ui()

    def _apply_advanced_json(self) -> None:
        text = self.advanced_json.toPlainText().strip()
        if not text:
            return
        try:
            data = json.loads(text)
            soz = SOZDefinition.from_dict(data)
            if not self._ensure_project():
                return
            self.state.project.sozs.append(soz)
            self._refresh_project_ui()
        except Exception as exc:
            self.status_bar.showMessage(f"Invalid JSON: {exc}", 5000)

    def _run_analysis(self) -> None:
        if not self._ensure_project():
            self.status_bar.showMessage("Load a project first", 5000)
            return
        if self._analysis_running:
            self.status_bar.showMessage("Analysis already running. Please wait.", 4000)
            return

        self.state.project.analysis.frame_start = self.frame_start_spin.value()
        frame_stop = self.frame_stop_spin.value()
        self.state.project.analysis.frame_stop = None if frame_stop < 0 else frame_stop
        self.state.project.analysis.stride = self.frame_stride_spin.value()

        if not self._maybe_apply_wizard_changes():
            return

        if not self._run_project_doctor(require_ok=True):
            return

        self._set_run_ui_state(True)
        self._clear_results_view()

        self.run_logger, self.log_path = setup_run_logger(self.state.project.outputs.output_dir)
        if self.run_logger:
            self.run_logger.info("GUI analysis requested")
        if self.log_path:
            self.status_bar.showMessage(f"Analysis started. Log: {self.log_path}", 5000)

        # Ensure GUI has per-frame data even if export is configured to skip it.
        if not self.state.project.sozs:
            # Fall back to the wizard definition if the project has no SOZs.
            project_for_run = self._project_from_wizard()
            if self.run_logger:
                self.run_logger.warning("Project has no SOZs; using Wizard definition for this run.")
            self.status_bar.showMessage("No SOZs in project; using Wizard definition for this run.", 8000)
        else:
            project_for_run = ProjectConfig.from_dict(self.state.project.to_dict())
        if not project_for_run.outputs.write_per_frame:
            project_for_run.outputs.write_per_frame = True
            if self.run_logger:
                self.run_logger.info("GUI override: write_per_frame forced true for display")
        self.run_project = project_for_run

        self.analysis_worker = AnalysisWorker(project_for_run, logger=self.run_logger)
        self.analysis_thread = QtCore.QThread()
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._on_progress)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_thread.start()

    def _cancel_analysis(self) -> None:
        if self.analysis_worker:
            self.analysis_worker.cancel()
            self.status_bar.showMessage("Cancellation requested", 5000)

    def _quick_subset(self) -> None:
        if not self._ensure_project():
            return
        if not self._maybe_apply_wizard_changes():
            return
        try:
            import MDAnalysis as mda

            if self.state.project.inputs.trajectory:
                universe = mda.Universe(
                    self.state.project.inputs.topology,
                    self.state.project.inputs.trajectory,
                )
            else:
                universe = mda.Universe(self.state.project.inputs.topology)
            total_frames = len(universe.trajectory)
            stride = max(1, int(total_frames / max(self.state.project.analysis.preview_frames, 1)))
            self.state.project.analysis.frame_start = 0
            self.state.project.analysis.frame_stop = None
            self.state.project.analysis.stride = stride
            self.frame_start_spin.setValue(0)
            self.frame_stop_spin.setValue(-1)
            self.frame_stride_spin.setValue(stride)
        except Exception as exc:
            self.status_bar.showMessage(f"Quick subset failed: {exc}", 5000)
            return
        self._run_analysis()

    def _export_report(self) -> None:
        if not self.current_result:
            QtWidgets.QMessageBox.information(self, "Export Report", "Run an analysis first.")
            return
        export_project = self.run_project or self.state.project
        if not export_project:
            QtWidgets.QMessageBox.warning(self, "Export Report", "No project loaded.")
            return
        out_dir = self._prompt_export_directory(
            "Select report output directory", export_project.outputs.output_dir
        )
        if out_dir:
            self.output_dir_edit.setText(out_dir)
            export_project.outputs.output_dir = out_dir
        try:
            export_results(self.current_result, export_project)
            report_path = generate_report(self.current_result, export_project, command_line="GUI")
            self.report_text.setText(f"Report written to {report_path}")
            QtWidgets.QApplication.clipboard().setText(str(report_path))
            self.status_bar.showMessage(f"Report written to {report_path} (copied)", 6000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Report export failed")
            self.status_bar.showMessage(f"Report export failed: {exc}", 8000)

    def _export_data(self) -> None:
        if not self.current_result:
            QtWidgets.QMessageBox.information(self, "Export Data", "Run an analysis first.")
            return
        export_project = self.run_project or self.state.project
        if not export_project:
            QtWidgets.QMessageBox.warning(self, "Export Data", "No project loaded.")
            return
        out_dir = self._prompt_export_directory(
            "Select data export directory", export_project.outputs.output_dir
        )
        if out_dir:
            self.output_dir_edit.setText(out_dir)
            export_project.outputs.output_dir = out_dir
        try:
            export_results(self.current_result, export_project)
            msg = f"Exported data to {export_project.outputs.output_dir}"
            self.status_bar.showMessage(msg, 5000)
            self.report_text.setText(msg)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Data export failed")
            self.status_bar.showMessage(f"Data export failed: {exc}", 8000)

    def _preview_soz(self) -> None:
        self._update_explain_text()
        if not self._ensure_project():
            return
        try:
            project = self._project_from_wizard()
            if not self.run_logger and project:
                self.run_logger, self.log_path = setup_run_logger(project.outputs.output_dir)
            if self.run_logger:
                self.run_logger.info("Preview requested")
            import MDAnalysis as mda

            if project.inputs.trajectory:
                universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
            else:
                universe = mda.Universe(project.inputs.topology)
            preflight = run_preflight(project, universe)
            if not preflight.ok:
                self.wizard_explain.setText(
                    "Preflight failed:\n" + "\n".join(preflight.errors)
                )
                return
            solvent = build_solvent(universe, project.solvent)
            selections = {
                label: resolve_selection(universe, spec)
                for label, spec in project.selections.items()
            }
            context = EvaluationContext(universe, solvent, selections)
            universe.trajectory[0]
            frame_set = evaluate_node(project.sozs[0].root, context)
            selection_counts = {
                label: len(selection.group) for label, selection in selections.items()
            }
            summary = (
                f"Selections: {selection_counts} | Solvent residues: {len(solvent.residues)} | "
                f"SOZ count (frame 0): {len(frame_set)}"
            )
            self.wizard_explain.setText(summary)
            if self.run_logger:
                self.run_logger.info("Preview summary: %s", summary)
        except Exception as exc:
            self.status_bar.showMessage(f"Preview failed: {exc}", 5000)

    def _project_from_wizard(self) -> ProjectConfig:
        base = self.state.project
        soz_name = self.wizard_soz_name.text().strip() or "SOZ"
        selection_a_label = f"{soz_name}_selection_a"
        selection_b_label = f"{soz_name}_selection_b"

        water_resnames = [x.strip() for x in self.wizard_water_resnames.text().split(",") if x.strip()]
        water_oxygen = [x.strip() for x in self.wizard_water_oxygen.text().split(",") if x.strip()]
        ion_resnames = [x.strip() for x in self.wizard_ion_resnames.text().split(",") if x.strip()]
        solvent_cfg = SolventConfig(
            water_resnames=water_resnames or base.solvent.water_resnames,
            water_oxygen_names=water_oxygen or base.solvent.water_oxygen_names,
            water_hydrogen_names=base.solvent.water_hydrogen_names,
            ion_resnames=ion_resnames or base.solvent.ion_resnames,
            include_ions=self.wizard_include_ions.isChecked(),
        )

        selection_a_sel = self.wizard_seed_a.text().strip()
        selections = {
            selection_a_label: SelectionSpec(
                label=selection_a_label,
                selection=selection_a_sel,
                require_unique=self.wizard_seed_a_unique.isChecked(),
            )
        }

        shell_cutoffs = [float(x.strip()) for x in self.wizard_shell_cutoffs.text().split(",") if x.strip()]
        if not shell_cutoffs:
            shell_cutoffs = [3.5]
        atom_mode = self.wizard_atom_mode.currentText()
        shell_node = SOZNode(
            type="shell",
            params={
                "selection_label": selection_a_label,
                "cutoffs": shell_cutoffs,
                "unit": "A",
                "atom_mode": atom_mode,
            },
        )

        root_node = shell_node
        selection_b_sel = self.wizard_seed_b.text().strip()
        if selection_b_sel:
            selections[selection_b_label] = SelectionSpec(
                label=selection_b_label,
                selection=selection_b_sel,
                require_unique=self.wizard_seed_b_unique.isChecked(),
            )
            dist_node = SOZNode(
                type="distance",
                params={
                    "selection_label": selection_b_label,
                    "cutoff": float(self.wizard_b_cutoff.text() or 3.5),
                    "unit": "A",
                    "atom_mode": atom_mode,
                },
            )
            combine = self.wizard_boolean.currentText().lower()
            root_node = SOZNode(type=combine, children=[shell_node, dist_node])

        soz = SOZDefinition(name=soz_name, description="Wizard-generated SOZ", root=root_node)
        return ProjectConfig(
            inputs=base.inputs,
            solvent=solvent_cfg,
            selections=selections,
            sozs=[soz],
            analysis=base.analysis,
            outputs=base.outputs,
            bridges=[],
            residue_hydration=[],
            version=base.version,
        )

    def _wizard_state(self) -> dict:
        return {
            "soz_name": self.wizard_soz_name.text().strip() or "SOZ",
            "water_resnames": self.wizard_water_resnames.text().strip(),
            "water_oxygen": self.wizard_water_oxygen.text().strip(),
            "include_ions": self.wizard_include_ions.isChecked(),
            "ion_resnames": self.wizard_ion_resnames.text().strip(),
            "selection_a": self.wizard_seed_a.text().strip(),
            "selection_a_unique": self.wizard_seed_a_unique.isChecked(),
            "selection_b": self.wizard_seed_b.text().strip(),
            "selection_b_unique": self.wizard_seed_b_unique.isChecked(),
            "shell_cutoffs": self.wizard_shell_cutoffs.text().strip(),
            "atom_mode": self.wizard_atom_mode.currentText(),
            "selection_b_cutoff": self.wizard_b_cutoff.text().strip(),
            "selection_b_combine": self.wizard_boolean.currentText(),
        }

    def _wizard_is_dirty(self) -> bool:
        if self._wizard_snapshot is None:
            return False
        return self._wizard_snapshot != self._wizard_state()

    def _apply_wizard_to_project(self, update_existing: bool = True) -> bool:
        if not self._ensure_project():
            return False
        project = self.state.project
        if not project:
            return False
        soz_name = self.wizard_soz_name.text().strip() or "SOZ"
        selection_a_label = f"{soz_name}_selection_a"
        selection_b_label = f"{soz_name}_selection_b"

        water_resnames = [x.strip() for x in self.wizard_water_resnames.text().split(",") if x.strip()]
        water_oxygen = [x.strip() for x in self.wizard_water_oxygen.text().split(",") if x.strip()]
        ion_resnames = [x.strip() for x in self.wizard_ion_resnames.text().split(",") if x.strip()]
        project.solvent = SolventConfig(
            water_resnames=water_resnames or project.solvent.water_resnames,
            water_oxygen_names=water_oxygen or project.solvent.water_oxygen_names,
            water_hydrogen_names=project.solvent.water_hydrogen_names,
            ion_resnames=ion_resnames or project.solvent.ion_resnames,
            include_ions=self.wizard_include_ions.isChecked(),
        )

        selection_a_sel = self.wizard_seed_a.text().strip()
        if not selection_a_sel:
            self.status_bar.showMessage("Selection A required", 5000)
            return False

        project.selections[selection_a_label] = SelectionSpec(
            label=selection_a_label,
            selection=selection_a_sel,
            require_unique=self.wizard_seed_a_unique.isChecked(),
        )

        shell_cutoffs = [float(x.strip()) for x in self.wizard_shell_cutoffs.text().split(",") if x.strip()]
        if not shell_cutoffs:
            shell_cutoffs = [3.5]
        atom_mode = self.wizard_atom_mode.currentText()
        shell_node = SOZNode(
            type="shell",
            params={
                "selection_label": selection_a_label,
                "cutoffs": shell_cutoffs,
                "unit": "A",
                "atom_mode": atom_mode,
            },
        )

        root_node = shell_node
        selection_b_sel = self.wizard_seed_b.text().strip()
        if selection_b_sel:
            project.selections[selection_b_label] = SelectionSpec(
                label=selection_b_label,
                selection=selection_b_sel,
                require_unique=self.wizard_seed_b_unique.isChecked(),
            )
            dist_node = SOZNode(
                type="distance",
                params={
                    "selection_label": selection_b_label,
                    "cutoff": float(self.wizard_b_cutoff.text() or 3.5),
                    "unit": "A",
                    "atom_mode": atom_mode,
                },
            )
            combine = self.wizard_boolean.currentText().lower()
            root_node = SOZNode(type=combine, children=[shell_node, dist_node])
        else:
            if selection_b_label in project.selections:
                project.selections.pop(selection_b_label, None)

        soz = SOZDefinition(name=soz_name, description="Wizard-generated SOZ", root=root_node)
        if update_existing:
            replaced = False
            for idx, existing in enumerate(project.sozs):
                if existing.name == soz_name:
                    project.sozs[idx] = soz
                    replaced = True
                    break
            if not replaced:
                project.sozs.append(soz)
        else:
            project.sozs.append(soz)

        self._wizard_snapshot = self._wizard_state()
        return True

    def _maybe_apply_wizard_changes(self) -> bool:
        if not self.state.project:
            return True
        if not self._wizard_is_dirty():
            return True
        project = self.state.project
        wizard_name = self.wizard_soz_name.text().strip() or "SOZ"
        auto_apply = False
        if not project.sozs:
            auto_apply = True
        elif len(project.sozs) == 1 and project.sozs[0].name == wizard_name:
            auto_apply = True

        if auto_apply:
            applied = self._apply_wizard_to_project(update_existing=True)
            if applied:
                self.status_bar.showMessage("Wizard changes applied to the project.", 4000)
            return applied

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Wizard changes detected")
        msg_box.setText(
            "Wizard settings have changed since they were last applied to the project.\n"
            "Apply the Wizard changes before running?"
        )
        apply_btn = msg_box.addButton("Apply Wizard", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        run_btn = msg_box.addButton("Run Project", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg_box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(apply_btn)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == cancel_btn:
            return False
        if clicked == apply_btn:
            return self._apply_wizard_to_project(update_existing=True)
        # Run Project: update snapshot to avoid repeated prompts until wizard changes again.
        self._wizard_snapshot = self._wizard_state()
        return True

    def _ensure_tester_universe(self, use_trajectory: bool):
        if not self._ensure_project():
            return None
        project = self.state.project
        if not project:
            return None
        key = (
            project.inputs.topology,
            project.inputs.trajectory if use_trajectory else None,
        )
        if getattr(self, "_tester_universe", None) is not None and self._tester_key == key:
            return self._tester_universe
        import MDAnalysis as mda

        if project.inputs.trajectory and use_trajectory:
            universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
        else:
            universe = mda.Universe(project.inputs.topology)
        self._tester_universe = universe
        self._tester_key = key
        return universe

    def _schedule_seed_validation(self) -> None:
        if not hasattr(self, "seed_validation_live"):
            return
        if not self.seed_validation_live.isChecked():
            return
        self._seed_validation_timer.start(600)

    def _run_seed_validation(self) -> None:
        if not hasattr(self, "seed_validation_table"):
            return
        if not self.state.project:
            self.wizard_seed_a_status.setText("Selection A matches: - (load a project)")
            self.wizard_seed_b_status.setText("Selection B matches: - (load a project)")
            self.seed_validation_table.setRowCount(0)
            return
        use_traj = self.seed_validation_use_traj.isChecked()
        limit = int(self.seed_validation_limit_spin.value())
        sel_a = self.wizard_seed_a.text().strip()
        sel_b = self.wizard_seed_b.text().strip()
        res_a = self._evaluate_selection(
            sel_a, use_traj, limit, require_unique=self.wizard_seed_a_unique.isChecked()
        )
        res_b = self._evaluate_selection(
            sel_b, use_traj, limit, require_unique=self.wizard_seed_b_unique.isChecked()
        )
        self._seed_validation_cache["A"] = res_a
        self._seed_validation_cache["B"] = res_b
        self.wizard_seed_a_status.setText(f"Selection A: {res_a.get('status', '-')}")
        self.wizard_seed_b_status.setText(f"Selection B: {res_b.get('status', '-')}")
        if hasattr(self, "define_inspector_text"):
            define_lines = [
                self.wizard_seed_a_status.text(),
                self.wizard_seed_b_status.text(),
            ]
            if res_a.get("suggestions"):
                define_lines.append("Selection A tips: " + " | ".join(res_a.get("suggestions")[:2]))
            if res_b.get("suggestions"):
                define_lines.append("Selection B tips: " + " | ".join(res_b.get("suggestions")[:2]))
            self.define_inspector_text.setText("\n".join(define_lines))
        target = self.seed_validation_target_combo.currentText()
        key = "A" if "A" in target else "B"
        result = self._seed_validation_cache.get(key, {})
        rows = result.get("rows", [])
        self.seed_validation_table.setRowCount(0)
        self.seed_validation_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.seed_validation_table.setItem(row, col, item)

    def _evaluate_selection(
        self,
        selection: str,
        use_traj: bool,
        limit: int,
        require_unique: bool = False,
        expect_count: int | None = None,
    ) -> dict:
        result = {
            "selection": selection,
            "count": 0,
            "summary_lines": [],
            "rows": [],
            "suggestions": [],
            "status": "",
            "error": None,
        }
        if not selection:
            result["status"] = "matches: 0 (empty selection)"
            result["summary_lines"] = ["Empty selection string."]
            return result
        universe = self._ensure_tester_universe(use_traj)
        if universe is None:
            result["status"] = "matches: - (load a project)"
            result["summary_lines"] = ["Load a project to test selections."]
            return result
        try:
            group = universe.select_atoms(selection)
        except Exception as exc:
            result["status"] = f"matches: error ({exc})"
            result["summary_lines"] = [f"Selection error: {exc}"]
            result["error"] = str(exc)
            return result

        n_hits = len(group)
        result["count"] = n_hits
        summary_lines = [f"Selection: {selection}", f"Matches: {n_hits}"]
        suggestions = []

        if n_hits == 1:
            atom = group[0]
            res = atom.residue
            residue_number = getattr(res, "resnum", None)
            if residue_number in (None, ""):
                residue_number = int(res.resid)
            summary_lines.append(
                "Unique match: "
                f"index {int(atom.index)} | {atom.name} | {res.resname} {residue_number} "
                f"| segid {res.segid} | chainID {getattr(res, 'chainID', '')} "
                f"| moltype {getattr(res, 'moltype', '')}"
            )
        if n_hits == 0:
            resnames = sorted({str(res.resname) for res in universe.residues})
            atom_names = sorted({str(name) for name in universe.atoms.names})
            if "HS" in selection.upper() or "HIS" in selection.upper():
                his_names = [name for name in resnames if name.upper().startswith(("HS", "HI"))]
                if his_names:
                    suggestions.append("Histidine-like resnames: " + ", ".join(his_names[:10]))
            if "LYS" in selection.upper():
                lys_names = [name for name in resnames if name.upper().startswith("LY")]
                if lys_names:
                    suggestions.append("Lys-like resnames: " + ", ".join(lys_names[:10]))
            suggestions.append("Example resnames: " + ", ".join(resnames[:12]))
            suggestions.append("Example atom names: " + ", ".join(atom_names[:12]))
        elif n_hits > 1:
            def uniq(vals):
                return sorted({v for v in vals if v not in (None, "")})

            segids = uniq([atom.residue.segid for atom in group])
            chain_ids = uniq([getattr(atom.residue, "chainID", None) for atom in group])
            moltypes = uniq([getattr(atom.residue, "moltype", None) for atom in group])
            resids = uniq([int(atom.residue.resid) for atom in group])
            resnums = uniq([getattr(atom.residue, "resnum", None) for atom in group])
            if segids:
                suggestions.append("Segids: " + ", ".join(segids[:10]))
            if chain_ids:
                suggestions.append("ChainIDs: " + ", ".join(map(str, chain_ids[:10])))
            if moltypes:
                suggestions.append("Moltypes: " + ", ".join(map(str, moltypes[:10])))
            if resids and len(resids) <= 20:
                suggestions.append("Resids: " + ", ".join(map(str, resids)))
            if resnums and len(resnums) <= 20:
                suggestions.append("Resnums: " + ", ".join(map(str, resnums)))
            suggestions.append("Suggestion: add segid/chainID/moltype and resid/resnum to narrow.")

        if require_unique and n_hits != 1:
            summary_lines.append("Warning: require_unique enabled but selection is not unique.")
        if expect_count is not None and n_hits != expect_count:
            summary_lines.append(f"Warning: expected {expect_count} atoms but found {n_hits}.")

        rows = []
        for atom in group[:limit]:
            res = atom.residue
            residue_number = getattr(res, "resnum", None)
            if residue_number in (None, ""):
                residue_number = int(res.resid)
            rows.append(
                [
                    int(atom.index),
                    str(atom.name),
                    str(res.resname),
                    residue_number,
                    str(res.segid),
                    getattr(res, "chainID", ""),
                    getattr(res, "moltype", ""),
                ]
            )

        status = f"matches: {n_hits}"
        if require_unique and n_hits != 1:
            status += " (not unique)"
        if n_hits == 0 and suggestions:
            status += f" | {suggestions[0]}"

        result["summary_lines"] = summary_lines + suggestions
        result["rows"] = rows
        result["suggestions"] = suggestions
        result["status"] = status
        return result

    def _ensure_preflight_universe(self):
        if not self._ensure_project():
            return None
        project = self.state.project
        if not project:
            return None
        key = (project.inputs.topology, project.inputs.trajectory)
        if self._preflight_universe is not None and self._preflight_key == key:
            return self._preflight_universe
        import MDAnalysis as mda

        if project.inputs.trajectory:
            universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
        else:
            universe = mda.Universe(project.inputs.topology)
        self._preflight_universe = universe
        self._preflight_key = key
        return universe

    def _run_project_doctor(self, require_ok: bool = False) -> bool:
        if not self._ensure_project():
            return False
        project = self.state.project
        if not project:
            return False
        try:
            universe = self._ensure_preflight_universe()
        except Exception as exc:
            msg = f"Preflight failed to load inputs: {exc}"
            self.doctor_status_label.setText("Preflight error")
            self.doctor_text.setText(msg)
            self.doctor_seed_table.setRowCount(0)
            self.status_bar.showMessage(msg, 8000)
            if self.run_logger:
                self.run_logger.exception("Preflight load failed")
            return False
        if universe is None:
            return False
        report = run_preflight(project, universe)
        self._preflight_report = report
        self._update_project_doctor_ui(report)
        if report.ok:
            self.status_bar.showMessage("Preflight OK. Ready to run.", 5000)
            return True
        self.status_bar.showMessage("Preflight failed. Fix errors in Project Doctor.", 8000)
        return False if require_ok else report.ok

    def _update_project_doctor_ui(self, report) -> None:
        status = "OK" if report.ok else f"Errors: {len(report.errors)}"
        self.doctor_status_label.setText(f"Project Doctor: {status}")
        lines = [f"Status: {status}"]
        if report.errors:
            lines.append("Errors:")
            lines.extend([f"  - {err}" for err in report.errors])
        if report.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {warn}" for warn in report.warnings])
        solvent = report.solvent_summary or {}
        water_matches = solvent.get("water_matches", [])
        ion_matches = solvent.get("ion_matches", [])
        lines.append(f"Water matches: {', '.join(map(str, water_matches)) or 'none'}")
        if solvent.get("include_ions"):
            lines.append(f"Ion matches: {', '.join(map(str, ion_matches)) or 'none'}")
        pbc = report.pbc_summary or {}
        lines.append(f"PBC box present: {pbc.get('has_box')}")
        gmx = report.gmx_summary or {}
        lines.append(
            f"GROMACS detected: {gmx.get('available')} {gmx.get('version') or ''}"
        )
        self.doctor_text.setText("\n".join(lines))

        selection_checks = report.selection_checks or report.seed_checks or {}
        self.doctor_seed_table.setRowCount(0)
        if selection_checks:
            self.doctor_seed_table.setRowCount(len(selection_checks))
            for row, check in enumerate(selection_checks.values()):
                values = [
                    check.label,
                    str(check.count),
                    str(check.require_unique),
                    str(check.expect_count) if check.expect_count is not None else "",
                    check.selection,
                    " | ".join(check.suggestions[:3]),
                ]
                for col, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.doctor_seed_table.setItem(row, col, item)
        if hasattr(self, "qc_inspector_text"):
            status_line = "QC OK" if report.ok else f"QC errors: {len(report.errors)}"
            self.qc_inspector_text.setText(status_line)

    def _show_pbc_helper(self) -> None:
        if not self.state.project:
            self.status_bar.showMessage("Load a project first.", 4000)
            return
        project = self.state.project
        topo = project.inputs.topology
        traj = project.inputs.trajectory
        if not traj:
            QtWidgets.QMessageBox.information(
                self, "PBC Helper", "No trajectory provided. Add a trajectory to use PBC helpers."
            )
            return
        import shutil

        gmx_bin = shutil.which("gmx_mpi") or shutil.which("gmx") or "gmx"
        cmd_lines = [
            "Suggested preprocessing commands (edit group selections as needed):",
            "",
            f"{gmx_bin} trjconv -s {topo} -f {traj} -o processed_nojump.xtc -pbc nojump",
            f"{gmx_bin} trjconv -s {topo} -f processed_nojump.xtc -o processed_center.xtc -pbc mol -center -ur compact",
            "",
            "Tip: use a protein group for centering and the system group for output.",
        ]
        QtWidgets.QMessageBox.information(self, "PBC Helper", "\n".join(cmd_lines))

    def _test_selection(self) -> None:
        selection = self.selection_input.text().strip()
        if not selection:
            self.selection_results.setText("Enter a selection string to test.")
            return
        if not self.state.project:
            self.selection_results.setText("Load a project first.")
            return
        if not self.run_logger:
            self.run_logger, self.log_path = setup_run_logger(self.state.project.outputs.output_dir)
        if self.run_logger:
            self.run_logger.info("Selection test: %s", selection)

        use_traj = self.selection_use_trajectory.isChecked()
        limit = int(self.selection_limit_spin.value())
        result = self._evaluate_selection(selection, use_traj, limit)
        self.selection_results.setText("\n".join(result.get("summary_lines", [])))
        if result.get("error") and self.run_logger:
            self.run_logger.error("Selection test failed: %s", result.get("error"))
        rows = result.get("rows", [])
        self.selection_table.setRowCount(0)
        self.selection_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.selection_table.setItem(row, col, item)

    def _ensure_project(self) -> bool:
        if self.state.project:
            return True
        topology, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Topology", "", "Topology (*)")
        if not topology:
            return False
        trajectory, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Trajectory (optional)", "", "Trajectory (*)"
        )
        project = default_project(topology, trajectory or None)
        self.state = ProjectState(project=project, path=None)
        self._refresh_project_ui()
        return True

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.status_bar.showMessage(f"{message} ({current}/{total})")
        if hasattr(self, "run_progress"):
            if total > 0:
                self.run_progress.setRange(0, total)
                self.run_progress.setValue(current)
            else:
                self.run_progress.setRange(0, 0)
        if self.run_logger:
            self.run_logger.info("%s (%d/%d)", message, current, total)

    def _on_analysis_finished(self, result: object) -> None:
        self.current_result = result
        self._timeline_stats_cache = {}
        self._timeline_event_cache = {}
        self._set_run_ui_state(False)
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        try:
            self._update_results_view()
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("GUI results update failed")
            self.status_bar.showMessage(f"Analysis complete, UI update failed: {exc}", 8000)
        if self.run_logger:
            self.run_logger.info("Analysis complete")
        if self.current_result and self.run_logger:
            try:
                summaries = []
                for name, soz in self.current_result.soz_results.items():
                    summaries.append(
                        f"{name}: per_frame={len(soz.per_frame)} per_solvent={len(soz.per_solvent)}"
                    )
                if summaries:
                    self.run_logger.info("GUI result sizes: %s", " | ".join(summaries))
            except Exception:
                self.run_logger.exception("Failed to summarize GUI result sizes")
        export_project = self.run_project or self.state.project
        if self.current_result and export_project:
            try:
                export_results(self.current_result, export_project)
                msg = f"Analysis complete. Results saved to {export_project.outputs.output_dir}"
                if self.log_path:
                    msg += f" | Log: {self.log_path}"
                self.status_bar.showMessage(msg, 8000)
                if self.report_text:
                    self.report_text.setText(msg)
            except Exception as exc:
                if self.run_logger:
                    self.run_logger.exception("Auto-export failed")
                self.status_bar.showMessage(f"Analysis complete, export failed: {exc}", 8000)
        else:
            self.status_bar.showMessage("Analysis complete", 5000)
        self._refresh_log_view()
        self._set_active_step(3)

    def _on_analysis_failed(self, message: str) -> None:
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        self._set_run_ui_state(False)
        if self.run_logger:
            self.run_logger.error("Analysis failed: %s", message)
        msg = f"Analysis failed: {message}"
        if self.log_path:
            msg += f" | Log: {self.log_path}"
        self.status_bar.showMessage(msg, 8000)
        self._refresh_log_view()

    def _update_results_view(self) -> None:
        if not self.current_result:
            return
        if not self.current_result.soz_results:
            msg = "Analysis completed but produced no SOZ results."
            if self.log_path:
                msg += f" Log: {self.log_path}"
            self.status_bar.showMessage(msg, 8000)
            if self.run_logger:
                self.run_logger.warning("No SOZ results produced.")
            return
        if self.run_logger:
            self.run_logger.info(
                "Updating GUI with %d SOZ results",
                len(self.current_result.soz_results),
            )
        if self.current_result.qc_summary is not None:
            qc_json = to_jsonable(self.current_result.qc_summary)
            self.qc_text.setText(json.dumps(qc_json, indent=2))
            self.qc_summary_label.setText(self._format_qc_summary(qc_json))
        soz_names = list(self.current_result.soz_results.keys())
        self.timeline_soz_combo.blockSignals(True)
        self.timeline_soz_combo.clear()
        self.timeline_soz_combo.addItems(soz_names)
        self.timeline_soz_combo.blockSignals(False)
        if soz_names:
            self.timeline_soz_combo.setCurrentIndex(0)
        self.extract_soz_combo.blockSignals(True)
        self.extract_soz_combo.clear()
        self.extract_soz_combo.addItems(soz_names)
        self.extract_soz_combo.blockSignals(False)
        if soz_names:
            self.extract_soz_combo.setCurrentIndex(0)

        for fn, name in (
            (self._update_overview, "overview"),
            (self._update_timeline_plot, "timeline"),
            (self._update_histogram_controls, "histogram_controls"),
            (self._update_hist_plot, "histogram"),
            (self._update_matrix_plot, "matrix"),
            (self._update_event_plot, "event_plot"),
            (self._update_tables_for_selected_soz, "tables"),
        ):
            try:
                fn()
            except Exception as exc:
                if self.run_logger:
                    self.run_logger.exception("GUI update failed: %s", name)
                self.status_bar.showMessage(f"GUI update failed ({name}): {exc}", 8000)
        if hasattr(self, "explore_inspector_text"):
            soz_name = self._selected_soz_name()
            if soz_name:
                self.explore_inspector_text.setText(f"SOZ: {soz_name}")

    def _update_overview(self) -> None:
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        soz = self.current_result.soz_results[soz_name]
        overview = {"summary": soz.summary, "warnings": self.current_result.warnings}
        summary_lines = []
        if isinstance(soz.summary, dict):
            for key, value in soz.summary.items():
                summary_lines.append(f"{key}: {value}")
        else:
            summary_lines.append(str(soz.summary))
        if self.current_result.warnings:
            summary_lines.append("Warnings: " + "; ".join(self.current_result.warnings))
        self.overview_card.setText("\n".join(summary_lines))
        self.overview_text.setText(json.dumps(to_jsonable(overview), indent=2))
        self._update_provenance_stamp()

    def _update_histogram_controls(self) -> None:
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        per_frame = self.current_result.soz_results[soz_name].per_frame
        self.hist_metric_combo.blockSignals(True)
        self.hist_metric_combo.clear()
        self.hist_metric_combo.addItems(per_frame.columns.tolist())
        self.hist_metric_combo.blockSignals(False)
        columns = per_frame.columns.tolist()
        if not columns:
            return
        if "n_solvent" in columns:
            self.hist_metric_combo.setCurrentText("n_solvent")
        else:
            self.hist_metric_combo.setCurrentIndex(0)

    def _update_tables_for_selected_soz(self) -> None:
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        soz = self.current_result.soz_results[soz_name]
        per_frame = self._filtered_per_frame(soz.per_frame)
        self._set_table(self.per_frame_table, per_frame)
        self._set_table(self.per_solvent_table, soz.per_solvent)

    def _selected_soz_name(self) -> Optional[str]:
        if not self.current_result or not self.current_result.soz_results:
            return None
        return self.timeline_soz_combo.currentText() or list(self.current_result.soz_results.keys())[0]

    def _filtered_per_frame(self, per_frame: pd.DataFrame) -> pd.DataFrame:
        if self._time_window is None or "time" not in per_frame.columns:
            return per_frame
        t0, t1 = self._time_window
        time_ns = pd.to_numeric(per_frame["time"], errors="coerce") / 1000.0
        mask = (time_ns >= min(t0, t1)) & (time_ns <= max(t0, t1))
        return per_frame[mask]

    def _on_soz_selection_changed(self) -> None:
        if not self.current_result:
            return
        self._update_overview()
        self._queue_timeline_update()
        self._update_histogram_controls()
        self._queue_hist_update()
        self._queue_matrix_update()
        self._queue_event_update()
        self._update_tables_for_selected_soz()

    def _update_timeline_plot(self) -> None:
        if not self.current_result:
            return
        tokens = self._get_theme_tokens()
        self.timeline_plot.clear()
        self.timeline_event_plot.clear()
        self._update_event_controls_state()
        if self.timeline_event_plot.plotItem.legend is None:
            self.timeline_event_plot.addLegend()
        if self.timeline_plot.plotItem.legend is None:
            self.timeline_plot.addLegend()
        clamp = self.timeline_clamp_check.isChecked()
        step_mode = self.timeline_step_check.isChecked()
        show_markers = self.timeline_markers_check.isChecked()
        show_mean = self.timeline_mean_check.isChecked()
        show_median = self.timeline_median_check.isChecked()
        shade_occupancy = self.timeline_shade_check.isChecked()
        metric = self.timeline_metric_combo.currentText()
        secondary_metric = self.timeline_secondary_combo.currentText()
        smooth = self.timeline_smooth_check.isChecked()
        smooth_window = int(self.timeline_smooth_window.value())
        max_y = 0.0
        self._clear_timeline_secondary()
        if secondary_metric == "None":
            self.timeline_plot.plotItem.getAxis("right").setVisible(False)
        else:
            self.timeline_plot.plotItem.getAxis("right").setVisible(True)
        if self.timeline_overlay.isChecked():
            colors = [
                tokens["accent"],
                tokens["exit"],
                tokens["entry"],
                "#b7791f",
                tokens["accent_alt"],
            ]
            for idx, (name, soz) in enumerate(self.current_result.soz_results.items()):
                if soz.per_frame.empty:
                    continue
                color = colors[idx % len(colors)]
                time_ps = pd.to_numeric(soz.per_frame["time"], errors="coerce").to_numpy()
                series = self._metric_series(soz.per_frame, metric)
                if series is None:
                    continue
                y_raw = series
                if np.any(y_raw < 0) and self.run_logger:
                    self.run_logger.warning("Negative %s values detected for %s.", metric, name)
                y = np.maximum(y_raw, 0) if clamp else y_raw
                x, y = self._downsample(time_ps / 1000.0, y)
                if smooth and smooth_window > 1:
                    y = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
                if step_mode:
                    x, y = self._step_series(x, y)
                if y.size:
                    max_y = max(max_y, float(np.nanmax(y)))
                self.timeline_plot.plot(
                    x,
                    y,
                    pen=pg.mkPen(color=color, width=self._plot_line_width),
                    name=name,
                    symbol="o" if show_markers else None,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=pg.mkBrush(color) if show_markers else None,
                )
        else:
            soz_name = self._selected_soz_name()
            if not soz_name:
                pass
            else:
                soz = self.current_result.soz_results[soz_name]
                if not soz.per_frame.empty:
                    time_ps = pd.to_numeric(soz.per_frame["time"], errors="coerce").to_numpy()
                    series = self._metric_series(soz.per_frame, metric)
                    if series is None:
                        series = np.array([])
                    y_raw = series
                    if y_raw.size and np.any(y_raw < 0) and self.run_logger:
                        self.run_logger.warning("Negative %s values detected for %s.", metric, soz_name)
                    y = np.maximum(y_raw, 0) if clamp else y_raw
                    x, y = self._downsample(time_ps / 1000.0, y)
                    if smooth and smooth_window > 1:
                        y = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
                    if step_mode:
                        x, y = self._step_series(x, y)
                    if y.size:
                        max_y = max(max_y, float(np.nanmax(y)))
                        self.timeline_plot.plot(
                            x,
                            y,
                            pen=pg.mkPen(color=tokens["accent"], width=self._plot_line_width),
                            symbol="o" if show_markers else None,
                            symbolSize=self._plot_marker_size if show_markers else None,
                            symbolBrush=pg.mkBrush(tokens["accent"]) if show_markers else None,
                        )
                    if shade_occupancy and "n_solvent" in soz.per_frame.columns:
                        threshold = int(self.timeline_shade_threshold.value())
                        occ = pd.to_numeric(soz.per_frame["n_solvent"], errors="coerce").fillna(0).to_numpy() >= threshold
                        if occ.size:
                            time_ns = time_ps / 1000.0
                            dt = float(np.median(np.diff(time_ns))) if time_ns.size > 1 else 0.0
                            start = None
                            for i, val in enumerate(occ):
                                if val and start is None:
                                    start = i
                                if (not val or i == len(occ) - 1) and start is not None:
                                    end = i if val else i - 1
                                    x0 = time_ns[start]
                                    x1 = time_ns[end] + (dt if dt else 0.0)
                                    region = pg.LinearRegionItem(
                                        values=(x0, x1),
                                        brush=pg.mkBrush(46, 125, 50, 40),
                                        movable=False,
                                    )
                                    region.setZValue(-10)
                                    try:
                                        region.setPen(pg.mkPen(None))
                                    except AttributeError:
                                        for line in getattr(region, "lines", []):
                                            line.setPen(pg.mkPen(None))
                                    self.timeline_plot.addItem(region)
                                    start = None
                    if secondary_metric != "None" and secondary_metric != metric:
                        sec_series = self._metric_series(soz.per_frame, secondary_metric)
                        if sec_series is not None and sec_series.size:
                            y2 = sec_series
                            y2 = np.maximum(y2, 0) if clamp else y2
                            x2, y2 = self._downsample(time_ps / 1000.0, y2)
                            if smooth and smooth_window > 1:
                                y2 = pd.Series(y2).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
                            if step_mode:
                                x2, y2 = self._step_series(x2, y2)
                            pen = pg.mkPen(
                                tokens["accent_alt"],
                                width=self._plot_line_width,
                                style=QtCore.Qt.PenStyle.DashLine,
                            )
                            item = pg.PlotDataItem(x2, y2, pen=pen)
                            self.timeline_secondary_view.addItem(item)
                            self._timeline_secondary_items.append(item)
                            self.timeline_plot.plotItem.getAxis("right").setLabel(secondary_metric)
                    if show_mean and y_raw.size:
                        mean_val = float(np.mean(y_raw))
                        self.timeline_plot.addItem(
                            pg.InfiniteLine(
                                pos=mean_val,
                                angle=0,
                                pen=pg.mkPen(
                                    tokens["exit"],
                                    width=max(1, int(self._plot_line_width)),
                                    style=QtCore.Qt.PenStyle.DashLine,
                                ),
                            )
                        )
                    if show_median and y_raw.size:
                        median_val = float(np.median(y_raw))
                        self.timeline_plot.addItem(
                            pg.InfiniteLine(
                                pos=median_val,
                                angle=0,
                                pen=pg.mkPen(
                                    tokens["entry"],
                                    width=max(1, int(self._plot_line_width)),
                                    style=QtCore.Qt.PenStyle.DashLine,
                                ),
                            )
                        )
        if clamp:
            upper = max(1.0, max_y * 1.05)
            self.timeline_plot.setLimits(yMin=0)
            self.timeline_plot.setYRange(0, upper, padding=0)
        self.timeline_plot.setLabel("left", self._hist_metric_label(metric))
        self._update_time_brush()
        self._update_timeline_event_plot(
            smooth=smooth,
            smooth_window=smooth_window,
            step_mode=step_mode,
            show_markers=show_markers,
        )
        self._update_timeline_summary()
        if hasattr(self, "explore_inspector_text"):
            window_text = (
                f"{self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
                if self._time_window
                else "full range"
            )
            self.explore_inspector_text.setText(
                f"SOZ: {self._selected_soz_name()}\n"
                f"Metric: {self._hist_metric_label(metric)}\n"
                f"Window: {window_text}"
            )

    def _update_time_brush(self) -> None:
        if not hasattr(self, "timeline_brush_check"):
            return
        if self.timeline_brush_check.isChecked():
            if self.timeline_region is None:
                self.timeline_region = pg.LinearRegionItem(movable=True, brush=pg.mkBrush(70, 214, 255, 40))
                self.timeline_region.sigRegionChanged.connect(self._on_time_brush_changed)
                self.timeline_plot.addItem(self.timeline_region)
            if self._time_window:
                self.timeline_region.setRegion(self._time_window)
            self.timeline_brush_clear.setEnabled(self._time_window is not None)
        else:
            if self.timeline_region is not None:
                try:
                    self.timeline_plot.removeItem(self.timeline_region)
                except Exception:
                    pass
                self.timeline_region = None
            if self._time_window is not None:
                self._time_window = None
                self._update_hist_plot()
                self._update_matrix_plot()
                self._update_event_plot()
                self._update_tables_for_selected_soz()
            self.timeline_brush_clear.setEnabled(False)

    def _on_time_brush_changed(self) -> None:
        if self.timeline_region is None:
            return
        region = self.timeline_region.getRegion()
        if region and len(region) == 2:
            self._time_window = (float(region[0]), float(region[1]))
        else:
            self._time_window = None
        if self._selected_soz_name():
            self._timeline_stats_cache.pop(self._selected_soz_name(), None)
        self._queue_hist_update()
        self._queue_matrix_update()
        self._queue_event_update()
        self._update_tables_for_selected_soz()
        if hasattr(self, "explore_inspector_text"):
            if self._time_window:
                self.explore_inspector_text.setText(
                    f"Time window: {self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
                )
            else:
                self.explore_inspector_text.setText("Time window: full")

    def _clear_time_brush(self) -> None:
        self._time_window = None
        if self.timeline_region is not None:
            try:
                self.timeline_plot.removeItem(self.timeline_region)
            except Exception:
                pass
            self.timeline_region = None
        if hasattr(self, "timeline_brush_clear"):
            self.timeline_brush_clear.setEnabled(False)
        self._queue_hist_update()
        self._queue_matrix_update()
        self._queue_event_update()
        self._update_tables_for_selected_soz()

    def _update_hist_plot(self) -> None:
        if not self.current_result:
            return
        tokens = self._get_theme_tokens()
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        metric = self.hist_metric_combo.currentText()
        if not metric:
            return
        df_full = self.current_result.soz_results[soz_name].per_frame
        df = self._filtered_per_frame(df_full)
        if metric not in df.columns:
            return
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        if metric == "time":
            values = values / 1000.0
        if values.size == 0:
            return
        bins = self.hist_bins_spin.value()
        zero_mask = values == 0
        zero_count = int(np.sum(zero_mask))
        zero_frac = zero_count / len(values)
        positive_values = values[~zero_mask]

        split_zeros = self.hist_zero_split_check.isChecked()
        self.hist_zero_plot.setVisible(split_zeros)
        self.hist_zero_plot.clear()

        if split_zeros:
            zero_axis = self.hist_zero_plot.getAxis("bottom")
            zero_axis.setTicks([[(0, "zero"), (1, "non-zero")]])
            zero_axis.setLabel("Category")
            if self.hist_norm_check.isChecked():
                zero_heights = [zero_frac, 1.0 - zero_frac]
                y_label = "Fraction of frames"
            else:
                zero_heights = [zero_count, len(values) - zero_count]
                y_label = "Frames"
            self.hist_zero_plot.setLabel("left", y_label)
            bar = pg.BarGraphItem(
                x=[0, 1],
                height=zero_heights,
                width=0.6,
                brush=pg.mkBrush(120, 140, 180, 180),
                pen=pg.mkPen("#2b6cb0", width=max(1, int(self._plot_line_width))),
            )
            self.hist_zero_plot.addItem(bar)

        if positive_values.size == 0:
            self.hist_plot.clear()
            self.hist_plot.addItem(
                pg.TextItem("No non-zero values to plot.", color=tokens["text_muted"])
            )
            self.hist_info.setText(
                f"n={len(values)} | zeros={zero_count} ({zero_frac:.1%}) | "
                "no non-zero values"
            )
            return

        if split_zeros:
            values_for_hist = positive_values
        else:
            values_for_hist = values

        if np.allclose(values_for_hist, np.round(values_for_hist)) and values_for_hist.size > 0:
            vmin = int(np.min(values_for_hist))
            vmax = int(np.max(values_for_hist))
            if vmax - vmin <= 200:
                bins = np.arange(vmin, vmax + 2) - 0.5
        y, x = np.histogram(values_for_hist, bins=bins)
        if self.hist_norm_check.isChecked():
            total = y.sum()
            y = y / total if total > 0 else y
        if self.hist_log_check.isChecked():
            self.hist_plot.setLogMode(y=True)
            y = np.where(y > 0, y, np.nan)
        else:
            self.hist_plot.setLogMode(y=False)
        self.hist_plot.clear()
        if y.size == 0:
            return
        centers = (x[:-1] + x[1:]) / 2.0
        widths = np.diff(x)
        bar = pg.BarGraphItem(
            x=centers,
            height=y,
            width=widths,
            brush=pg.mkBrush(100, 140, 220, 150),
            pen=pg.mkPen("#2b6cb0", width=max(1, int(self._plot_line_width))),
        )
        self.hist_plot.addItem(bar)

        mean_val = float(np.mean(values))
        med_val = float(np.median(values))
        mean_nonzero = float(np.mean(positive_values)) if positive_values.size else 0.0
        med_nonzero = float(np.median(positive_values)) if positive_values.size else 0.0
        line_mean = mean_val if not split_zeros else mean_nonzero
        line_median = med_val if not split_zeros else med_nonzero
        self.hist_plot.addItem(
            pg.InfiniteLine(pos=line_mean, angle=90, pen=pg.mkPen("#c53030", width=self._plot_line_width))
        )
        self.hist_plot.addItem(
            pg.InfiniteLine(pos=line_median, angle=90, pen=pg.mkPen("#2f855a", width=self._plot_line_width))
        )
        try:
            ymax = float(np.nanmax(y))
        except Exception:
            ymax = 0.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        mean_label = "mean (non-zero)" if split_zeros else "mean (all frames)"
        median_label = "median (non-zero)" if split_zeros else "median (all frames)"
        mean_text = pg.TextItem(mean_label, color="#c53030")
        median_text = pg.TextItem(median_label, color="#2f855a")
        mean_text.setPos(line_mean, ymax * 0.95)
        median_text.setPos(line_median, ymax * 0.85)
        self.hist_plot.addItem(mean_text)
        self.hist_plot.addItem(median_text)
        if metric == "time":
            self.hist_plot.setLabel("bottom", "Time", units="ns")
        else:
            pretty = self._hist_metric_label(metric)
            label = f"{pretty} (non-zero)" if split_zeros else pretty
            self.hist_plot.setLabel("bottom", label)
        if self.hist_norm_check.isChecked():
            self.hist_plot.setLabel("left", "Fraction of frames")
        else:
            self.hist_plot.setLabel("left", "Frames")

        hint = ""
        if zero_frac > 0.3 and not split_zeros:
            hint = " | Tip: enable 'Split zeros' to inspect non-zero structure."
        line_info = "Red = mean, Green = median"
        subset_info = " (non-zero subset)" if split_zeros else " (all frames)"
        self.hist_info.setText(
            f"n={len(values)} frames | zeros={zero_count} ({zero_frac:.1%}) | "
            f"mean={mean_val:.3f} median={med_val:.3f} | "
            f"non-zero mean={mean_nonzero:.3f} median={med_nonzero:.3f} | bins={bins} | "
            f"{line_info}{subset_info}{hint}"
        )
        if self._time_window:
            self.hist_info.setText(
                self.hist_info.text()
                + f" | window={self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
            )

    def _update_matrix_plot(self) -> None:
        if not self.current_result:
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        tokens = self._get_theme_tokens()
        source = self.matrix_source_combo.currentText()
        if source.startswith("Residue hydration"):
            if not self.current_result.hydration_results:
                self.matrix_view.setImage(np.zeros((1, 1)), autoLevels=True)
                self.matrix_view.setPredefinedGradient("viridis")
                self.matrix_info.setText("No hydration results configured for this run.")
                if self.run_logger:
                    self.run_logger.info("No hydration results; matrix plot left empty.")
                return
            hydration = next(iter(self.current_result.hydration_results.values()))
            table = hydration.table
            if table.empty:
                self.matrix_view.setImage(np.zeros((1, 1)), autoLevels=True)
                self.matrix_view.setPredefinedGradient("viridis")
                self.matrix_info.setText("Hydration table is empty.")
                return
            data = table["hydration_freq"].to_numpy().reshape(-1, 1)
            self.matrix_view.setImage(data, autoLevels=True)
            self.matrix_view.setPredefinedGradient("viridis")
            self.matrix_info.setText(f"Residue hydration rows: {len(table)}")
            return

        per_frame_full = self.current_result.soz_results[soz_name].per_frame
        per_frame = self._filtered_per_frame(per_frame_full)
        per_solvent = self.current_result.soz_results[soz_name].per_solvent
        matrix, ids, time_ns, msg = self._build_presence_matrix(
            per_frame,
            per_solvent,
            top_n=self.matrix_top_spin.value(),
            min_occ_pct=self.matrix_min_occ_spin.value(),
        )
        if matrix is None:
            self.matrix_view.setImage(np.zeros((1, 1)), autoLevels=True)
            self.matrix_view.setPredefinedGradient("viridis")
            self.matrix_info.setText(msg or "No data available for occupancy heatmap.")
            return
        self.matrix_view.setImage(matrix, xvals=time_ns, autoLevels=False)
        self.matrix_view.setLevels(0, 1)
        self.matrix_view.setPredefinedGradient("viridis")
        try:
            view = self.matrix_view.getView()
            view.setAspectLocked(False)
            view.enableAutoRange()
        except Exception:
            view = None
        plot_item = self._matrix_plot_item()
        if plot_item:
            plot_item.setLabel("bottom", "Time", units="ns")
            plot_item.setLabel("left", "Solvent (ranked by occupancy)")
            plot_item.getAxis("bottom").enableAutoSIPrefix(False)
        try:
            self.matrix_view.getView().invertY(True)
        except Exception:
            pass
        if ids:
            short_ids = [self._short_solvent_label(sid) for sid in ids]
            if plot_item:
                max_labels = 12
                step = max(1, int(len(short_ids) / max_labels))
                ticks = [(i, short_ids[i]) for i in range(0, len(short_ids), step)]
                plot_item.getAxis("left").setTicks([ticks])
                if hasattr(self, "matrix_highlight_line") and self.matrix_highlight_line is not None:
                    try:
                        plot_item.removeItem(self.matrix_highlight_line)
                    except Exception:
                        pass
                    self.matrix_highlight_line = None
                if self._selected_solvent_id and self._selected_solvent_id in ids:
                    row = ids.index(self._selected_solvent_id)
                    self.matrix_highlight_line = pg.InfiniteLine(
                        pos=row, angle=0, pen=pg.mkPen(tokens["accent"], width=2)
                    )
                    plot_item.addItem(self.matrix_highlight_line)
            preview = ", ".join(f"{idx}:{sid}" for idx, sid in enumerate(short_ids[:10]))
            suffix = " ..." if len(short_ids) > 10 else ""
            self.matrix_info.setText(
                "Rows = top solvents by occupancy (row 0 = most occupied). "
                f"Columns = time (ns). Color = occupancy (1 = present, 0 = absent). "
                f"Row map: {preview}{suffix}"
            )
        else:
            self.matrix_info.setText("No solvent IDs available.")

    def _update_event_plot(self) -> None:
        if not self.current_result:
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        tokens = self._get_theme_tokens()
        per_frame_full = self.current_result.soz_results[soz_name].per_frame
        per_frame = self._filtered_per_frame(per_frame_full)
        per_solvent = self.current_result.soz_results[soz_name].per_solvent
        matrix, ids, time_ns, msg = self._build_presence_matrix(
            per_frame,
            per_solvent,
            top_n=self.matrix_top_spin.value(),
            min_occ_pct=self.matrix_min_occ_spin.value(),
        )
        if matrix is None:
            self.event_plot.clear()
            self.event_plot.addItem(
                pg.TextItem(msg or "No solvent IDs available", color=tokens["text_muted"])
            )
            self.event_info.setText(msg or "")
            return
        stride = max(1, int(self.event_stride_spin.value()))
        if stride > 1:
            matrix = matrix[:, ::stride]
            time_ns = time_ns[::stride]
        rows, cols = np.where(matrix > 0)
        self.event_plot.clear()
        if rows.size == 0:
            self.event_plot.addItem(
                pg.TextItem("No events for selected solvents", color=tokens["text_muted"])
            )
            self.event_info.setText("No occupancy events found for the chosen top solvents.")
            return
        segment_mode = self.event_segment_check.isChecked()
        min_duration = int(self.event_min_duration_spin.value())
        if segment_mode:
            sel_idx = None
            if self._selected_solvent_id and self._selected_solvent_id in ids:
                sel_idx = ids.index(self._selected_solvent_id)
            for row_idx in range(matrix.shape[0]):
                row = matrix[row_idx]
                if not np.any(row):
                    continue
                start = None
                for col_idx, val in enumerate(row):
                    if val and start is None:
                        start = col_idx
                    if (not val or col_idx == len(row) - 1) and start is not None:
                        end = col_idx if val else col_idx - 1
                        length = end - start + 1
                        if length >= min_duration:
                            x0 = time_ns[start]
                            x1 = time_ns[end]
                            color = (
                                tokens["accent"]
                                if sel_idx is not None and row_idx == sel_idx
                                else tokens["accent_alt"]
                            )
                            line = pg.PlotDataItem(
                                [x0, x1],
                                [row_idx, row_idx],
                                pen=pg.mkPen(color, width=max(2, int(self._plot_line_width))),
                            )
                            self.event_plot.addItem(line)
                        start = None
        else:
            xs = time_ns[cols]
            ys = rows
            scatter = pg.ScatterPlotItem(
                xs,
                ys,
                size=max(3, int(self._plot_marker_size)),
                brush=pg.mkBrush(tokens["accent_alt"]),
            )
            self.event_plot.addItem(scatter)
        if self._selected_solvent_id and self._selected_solvent_id in ids:
            sel_idx = ids.index(self._selected_solvent_id)
            sel_mask = rows == sel_idx
            if np.any(sel_mask):
                sel_x = time_ns[cols[sel_mask]]
                sel_y = rows[sel_mask]
                highlight = pg.ScatterPlotItem(
                    sel_x,
                    sel_y,
                    size=max(6, int(self._plot_marker_size) + 2),
                    brush=pg.mkBrush(tokens["accent"]),
                )
                self.event_plot.addItem(highlight)
        axis = self.event_plot.getAxis("left")
        if ids:
            max_labels = 30
            step = max(1, int(len(ids) / max_labels))
            ticks = [(i, self._short_solvent_label(ids[i])) for i in range(0, len(ids), step)]
            axis.setTicks([ticks])
        mode_label = "segments" if segment_mode else "points"
        self.event_info.setText(
            f"Top {len(ids)} solvents by occupancy | stride={stride} | {mode_label} | "
            f"min duration={min_duration}"
        )

    def _build_presence_matrix(
        self,
        per_frame: pd.DataFrame,
        per_solvent: pd.DataFrame,
        top_n: int,
        min_occ_pct: float,
    ):
        if per_frame.empty or "solvent_ids" not in per_frame.columns:
            return None, [], None, "No solvent IDs available (enable per-frame IDs)."
        if per_solvent.empty or "solvent_id" not in per_solvent.columns:
            return None, [], None, "Per-solvent table is empty."
        df = per_solvent.sort_values("occupancy_pct", ascending=False)
        if min_occ_pct > 0:
            df = df[df["occupancy_pct"] >= min_occ_pct]
        ids = df["solvent_id"].head(top_n).tolist()
        if not ids:
            return None, [], None, "No solvents matched the occupancy filter."
        id_map = {sid: idx for idx, sid in enumerate(ids)}
        n_frames = len(per_frame)
        matrix = np.zeros((len(ids), n_frames), dtype=np.uint8)
        for col, ids_str in enumerate(per_frame["solvent_ids"].fillna("").astype(str)):
            if not ids_str:
                continue
            for sid in ids_str.split(";"):
                row = id_map.get(sid)
                if row is not None:
                    matrix[row, col] = 1
        if "time" in per_frame.columns:
            time_ns = per_frame["time"].to_numpy() / 1000.0
        else:
            time_ns = np.arange(n_frames, dtype=float)
        return matrix, ids, time_ns, ""

    def _browse_extract_output(self) -> None:
        if getattr(self, "_extract_output_linked", False):
            self.extract_link_check.setChecked(False)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.extract_output_edit.setText(path)

    def _open_extract_output(self) -> None:
        path_raw = self.extract_output_edit.text().strip()
        self._open_directory(path_raw)

    def _on_extract_output_edited(self, text: str) -> None:
        if getattr(self, "_extract_output_linked", False):
            self.extract_link_check.setChecked(False)

    def _resolve_log_path(self) -> Optional[str]:
        if self.log_path:
            return self.log_path
        if self.state.project:
            candidate = os.path.join(self.state.project.outputs.output_dir, "sozlab.log")
            return candidate
        return None

    def _refresh_log_view(self) -> None:
        if not hasattr(self, "log_text"):
            return
        path = self._resolve_log_path()
        if not path:
            self.log_text.setText("No log path available yet.")
            self.log_path_label.setText("Log: -")
            self.log_summary_label.setText("No log loaded.")
            return
        self.log_path_label.setText(f"Log: {path}")
        if not os.path.exists(path):
            self.log_text.setText("Log file not found yet.")
            self.log_summary_label.setText("Log file not found.")
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception as exc:
            self.log_text.setText(f"Failed to read log: {exc}")
            self.log_summary_label.setText("Unable to read log file.")
            return
        self._log_raw_text = text
        self._apply_log_filter()
        if hasattr(self, "console_text"):
            tail_lines = text.splitlines()[-200:]
            self.console_text.setText("\n".join(tail_lines))
        error_blocks = self._extract_error_blocks(text)
        error_count = len(error_blocks)
        summary = f"Errors detected: {error_count}"
        if error_blocks:
            summary += f" | Last error: {error_blocks[-1].splitlines()[0][:120]}"
        self.log_summary_label.setText(summary)

    def _open_log_file(self) -> None:
        path = self._resolve_log_path()
        if not path:
            return
        url = QtCore.QUrl.fromLocalFile(path)
        QtGui.QDesktopServices.openUrl(url)

    def _copy_log_errors(self) -> None:
        path = self._resolve_log_path()
        if not path or not os.path.exists(path):
            self.status_bar.showMessage("No log file to copy errors from.", 4000)
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception as exc:
            self.status_bar.showMessage(f"Failed to read log: {exc}", 6000)
            return
        blocks = self._extract_error_blocks(text)
        if not blocks:
            self.status_bar.showMessage("No error blocks found in log.", 4000)
            return
        QtWidgets.QApplication.clipboard().setText("\n\n".join(blocks))
        self.status_bar.showMessage("Copied error blocks to clipboard.", 4000)

    def _apply_log_filter(self) -> None:
        text = self._log_raw_text or ""
        level = self.log_level_combo.currentText() if hasattr(self, "log_level_combo") else "All"
        search = self.log_search_edit.text().strip() if hasattr(self, "log_search_edit") else ""
        collapse = self.log_collapse_check.isChecked() if hasattr(self, "log_collapse_check") else True

        lines = text.splitlines()
        filtered = []
        for line in lines:
            if level != "All" and f"| {level} |" not in line:
                continue
            if search and search.lower() not in line.lower():
                continue
            filtered.append(line)

        if collapse:
            filtered = self._collapse_tracebacks(filtered)

        self.log_text.setText("\n".join(filtered))
        if hasattr(self, "console_text"):
            self.console_text.setText("\n".join(filtered[-200:]))
        error_blocks = self._extract_error_blocks(text)
        error_count = len(error_blocks)
        summary = f"Errors detected: {error_count}"
        if error_blocks:
            summary += f" | Last error: {error_blocks[-1].splitlines()[0][:120]}"
        self.log_summary_label.setText(summary)

    def _collapse_tracebacks(self, lines: list[str]) -> list[str]:
        collapsed = []
        in_trace = False
        for line in lines:
            if line.startswith("Traceback"):
                collapsed.append("Traceback (collapsed)")
                in_trace = True
                continue
            if in_trace:
                if line.strip() == "":
                    in_trace = False
                continue
            collapsed.append(line)
        return collapsed

    def _extract_error_blocks(self, text: str) -> list[str]:
        lines = text.splitlines()
        blocks = []
        current = []
        in_block = False
        for line in lines:
            if "ERROR" in line or line.startswith("Traceback"):
                if current:
                    blocks.append("\n".join(current))
                    current = []
                in_block = True
                current.append(line)
                continue
            if in_block:
                if line.strip() == "" and current:
                    blocks.append("\n".join(current))
                    current = []
                    in_block = False
                    continue
                if line.startswith(" ") or line.startswith("\t") or line.startswith("Traceback"):
                    current.append(line)
                    continue
                if " | " in line:
                    blocks.append("\n".join(current))
                    current = []
                    in_block = False
            if in_block and current:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def _update_extract_mode_ui(self) -> None:
        mode = self.extract_mode_combo.currentText()
        is_threshold = mode == "Threshold"
        is_percentile = mode == "Percentile"
        is_topn = mode == "Top N frames"

        self.extract_op_label.setVisible(is_threshold)
        self.extract_op_combo.setVisible(is_threshold)
        self.extract_threshold_label.setVisible(is_threshold)
        self.extract_threshold_spin.setVisible(is_threshold)

        self.extract_percentile_label.setVisible(is_percentile)
        self.extract_percentile_spin.setVisible(is_percentile)

        self.extract_topn_label.setVisible(is_topn)
        self.extract_topn_spin.setVisible(is_topn)

        allow_runs = not is_topn
        self.extract_min_run_label.setEnabled(allow_runs)
        self.extract_min_run_spin.setEnabled(allow_runs)
        self.extract_gap_label.setEnabled(allow_runs)
        self.extract_gap_spin.setEnabled(allow_runs)

        if is_topn:
            self.extract_rule_help.setText(
                "Top N frames selects the highest-occupancy frames (ties may include more). "
                "Run-length and gap settings are disabled for this mode."
            )
        elif is_percentile:
            self.extract_rule_help.setText(
                "Percentile selects frames at or above the chosen percentile of the metric distribution."
            )
        else:
            self.extract_rule_help.setText(
                "Frames are kept when the rule is true. Increasing the threshold usually reduces the number of frames."
            )
        self._update_extract_rule_preview()

    def _on_extract_metric_changed(self) -> None:
        metric = self.extract_metric_combo.currentText()
        if metric == "occupancy_fraction":
            self.extract_threshold_spin.blockSignals(True)
            self.extract_threshold_spin.setDecimals(3)
            self.extract_threshold_spin.setSingleStep(0.05)
            self.extract_threshold_spin.setRange(0.0, 1.0)
            if self.extract_threshold_spin.value() > 1.0:
                self.extract_threshold_spin.setValue(0.5)
            self.extract_threshold_spin.blockSignals(False)
        else:
            self.extract_threshold_spin.blockSignals(True)
            self.extract_threshold_spin.setDecimals(0)
            self.extract_threshold_spin.setSingleStep(1.0)
            self.extract_threshold_spin.setRange(0.0, 1_000_000.0)
            if self.extract_threshold_spin.value() < 1.0:
                self.extract_threshold_spin.setValue(1.0)
        self.extract_threshold_spin.blockSignals(False)
        self._update_extract_rule_preview()
        self._update_extract_metric_info()

    def _update_extract_rule_preview(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            mode = self.extract_mode_combo.currentText()
            if mode == "Threshold":
                rule = self._build_extract_rule()
                self.extract_rule_preview.setText(rule)
                self.extract_rule_note.setText("Run an analysis to preview selections.")
            else:
                self.extract_rule_preview.setText("Run an analysis to compute this rule.")
                self.extract_rule_note.setText("")
            return
        soz_name = self.extract_soz_combo.currentText()
        if not soz_name:
            self.extract_rule_preview.setText("Select a SOZ.")
            self.extract_rule_note.setText("")
            return
        per_frame = self.current_result.soz_results[soz_name].per_frame
        rule, note = self._compute_extraction_rule(per_frame)
        if rule:
            metric = self.extract_metric_combo.currentText()
            extra = ""
            if metric == "occupancy_fraction":
                extra = " (computed per-frame: 1 if n_solvent>0 else 0 when not present)"
            self.extract_rule_preview.setText(f"{rule}{extra}")
        else:
            self.extract_rule_preview.setText("Unable to compute rule.")
        self.extract_rule_note.setText(note or "")
        self._update_extract_metric_info()

    def _extract_metric_values(self, per_frame: pd.DataFrame, metric: str) -> np.ndarray:
        if metric in per_frame.columns:
            return pd.to_numeric(per_frame[metric], errors="coerce").dropna().to_numpy()
        if metric == "occupancy_fraction" and "n_solvent" in per_frame.columns:
            values = pd.to_numeric(per_frame["n_solvent"], errors="coerce").fillna(0)
            return (values > 0).astype(float).to_numpy()
        return np.array([])

    def _compute_extraction_rule(self, per_frame: pd.DataFrame) -> tuple[str | None, str]:
        metric = self.extract_metric_combo.currentText()
        mode = self.extract_mode_combo.currentText()
        if mode == "Threshold":
            return self._build_extract_rule(), ""

        values = self._extract_metric_values(per_frame, metric)
        if values.size == 0:
            return None, "Metric values unavailable."

        note = ""
        if mode == "Percentile":
            percentile = float(self.extract_percentile_spin.value())
            threshold = float(np.percentile(values, percentile))
            if metric == "n_solvent":
                threshold = float(np.ceil(threshold))
            note = f"Percentile {percentile:.1f}% → threshold {threshold:.3f}"
            return f"{metric}>={threshold}", note

        if mode == "Top N frames":
            top_n = int(self.extract_topn_spin.value())
            if top_n <= 0:
                return None, "Top N must be >= 1."
            if top_n >= values.size:
                threshold = float(np.min(values))
                note = "Top N exceeds total frames; selecting all frames."
            else:
                threshold = float(np.sort(values)[-top_n])
            if metric == "n_solvent":
                threshold = float(np.ceil(threshold))
            selected = int(np.sum(values >= threshold))
            if selected > top_n:
                note = f"Top {top_n} by {metric} → threshold {threshold:.3f}; ties select {selected} frames"
            else:
                note = f"Top {top_n} by {metric} → threshold {threshold:.3f}"
            return f"{metric}>={threshold}", note

        return None, "Unknown extraction mode."

    def _update_extract_metric_info(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.extract_metric_info.setText("Run an analysis to see metric statistics.")
            return
        soz_name = self.extract_soz_combo.currentText()
        if not soz_name:
            self.extract_metric_info.setText("Select a SOZ to see metric statistics.")
            return
        metric = self.extract_metric_combo.currentText()
        per_frame = self.current_result.soz_results[soz_name].per_frame
        if per_frame.empty:
            self.extract_metric_info.setText("No per-frame data available.")
            return
        values = self._extract_metric_values(per_frame, metric)
        if values.size == 0:
            self.extract_metric_info.setText(f"Metric '{metric}' not available.")
            return
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        mean_v = float(np.mean(values))
        p5 = float(np.percentile(values, 5))
        p95 = float(np.percentile(values, 95))
        self.extract_metric_info.setText(
            f"min={min_v:.3f} | p5={p5:.3f} | mean={mean_v:.3f} | p95={p95:.3f} | max={max_v:.3f}"
        )

    def _build_extract_rule(self) -> str:
        metric = self.extract_metric_combo.currentText()
        op = self.extract_op_combo.currentText()
        threshold = self.extract_threshold_spin.value()
        return f"{metric}{op}{threshold}"

    def _preview_extraction(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.extract_summary.setText("Run an analysis first.")
            return
        soz_name = self.extract_soz_combo.currentText()
        if not soz_name:
            self.extract_summary.setText("Select a SOZ.")
            return
        per_frame = self.current_result.soz_results[soz_name].per_frame
        time_unit = "ps"
        if self.current_result.qc_summary:
            time_unit = self.current_result.qc_summary.get("time_unit", "ps")
        rule, note = self._compute_extraction_rule(per_frame)
        if not rule:
            self.extract_summary.setText(note or "Unable to compute extraction rule.")
            return
        mode = self.extract_mode_combo.currentText()
        min_run = self.extract_min_run_spin.value()
        gap = self.extract_gap_spin.value()
        if mode == "Top N frames":
            min_run = 1
            gap = 0
        try:
            selection = select_frames(
                per_frame,
                rule=rule,
                min_run_length=min_run,
                gap_tolerance=gap,
                time_unit=time_unit,
            )
        except Exception as exc:
            self.extract_summary.setText(f"Preview failed: {exc}")
            if self.run_logger:
                self.run_logger.exception("Extraction preview failed")
            return
        self._extract_selection = selection
        if selection.frame_indices:
            times = selection.manifest["time"]
            time_range = f"{times.min():.3f}–{times.max():.3f} {time_unit}"
            msg = f"Selected {len(selection.frame_indices)} frames using rule {rule}. Time range: {time_range}."
        else:
            msg = f"Selected 0 frames using rule {rule}."
        if note:
            msg += f" {note}"
        self.extract_summary.setText(msg)
        self.extract_run_btn.setEnabled(bool(selection.frame_indices))
        self.extract_table.setRowCount(0)
        rows = min(len(selection.frame_indices), 200)
        self.extract_table.setRowCount(rows)
        for row, (_, rec) in enumerate(selection.manifest.head(rows).iterrows()):
            values = [rec.get("frame"), rec.get("time"), rec.get("n_solvent")]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.extract_table.setItem(row, col, item)
        if hasattr(self, "export_inspector_text"):
            self.export_inspector_text.setText(msg)

    def _run_extraction(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.extract_summary.setText("Run an analysis first.")
            return
        if not self.state.project or not self.state.project.inputs.trajectory:
            self.extract_summary.setText("Extraction requires a trajectory.")
            return
        if not self._extract_selection:
            self._preview_extraction()
        selection = self._extract_selection
        if selection is None:
            return
        if not selection.frame_indices:
            self.extract_summary.setText("No frames selected. Adjust the threshold or rule.")
            return
        out_dir = self.extract_output_edit.text().strip()
        if not out_dir:
            out_dir = self.state.project.extraction.output_dir
            self.extract_output_edit.setText(out_dir)
        fmt = self.extract_format_combo.currentText() or "xtc"
        try:
            os.makedirs(out_dir, exist_ok=True)
            import MDAnalysis as mda

            universe = mda.Universe(
                self.state.project.inputs.topology,
                self.state.project.inputs.trajectory,
            )
            if hasattr(self, "extract_progress"):
                self.extract_progress.setVisible(True)
                self.extract_progress.setRange(0, len(selection.frame_indices))
                self.extract_progress.setValue(0)

            def _update_extract_progress(current: int, total: int, message: str) -> None:
                if hasattr(self, "extract_progress"):
                    self.extract_progress.setRange(0, max(total, 1))
                    self.extract_progress.setValue(current)
                    self.extract_progress.setFormat(f"{message}: %p%")
                QtWidgets.QApplication.processEvents()

            outputs = write_extracted_trajectory(
                universe,
                selection,
                output_dir=out_dir,
                prefix="extracted",
                fmt=fmt,
                progress_cb=_update_extract_progress,
            )
            self._last_extract_outputs = outputs
            lines = [f"Wrote {len(selection.frame_indices)} frames to {out_dir}"]
            lines.extend(f"{key}: {path}" for key, path in outputs.items())
            self.extract_summary.setText("\n".join(lines))
            if hasattr(self, "export_inspector_text"):
                self.export_inspector_text.setText("\n".join(lines))
            if self.run_logger:
                self.run_logger.info("Extraction outputs: %s", outputs)
        except Exception as exc:
            self.extract_summary.setText(f"Extraction failed: {exc}")
            if self.run_logger:
                self.run_logger.exception("Extraction failed")
        finally:
            if hasattr(self, "extract_progress"):
                self.extract_progress.setVisible(False)

    def _copy_widget_to_clipboard(self, widget: QtWidgets.QWidget) -> None:
        pixmap = widget.grab()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)
        self.status_bar.showMessage("Plot copied to clipboard", 3000)

    def _open_command_palette(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Command Palette")
        dialog.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(dialog)
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Type a command…")
        layout.addWidget(search)
        list_widget = QtWidgets.QListWidget()
        layout.addWidget(list_widget)

        commands = [
            ("Load project", self._load_project),
            ("Save project", self._save_project),
            ("Run analysis", self._run_analysis),
            ("Run Project Doctor", self._run_project_doctor),
            ("Export data", self._export_data),
            ("Export report", self._export_report),
            ("Open output directory", self._open_output_dir),
            ("Toggle console", lambda: self.console_toggle_btn.toggle()),
            ("Go to Project", lambda: self._set_active_step(0)),
            ("Go to Define", lambda: self._set_active_step(1)),
            ("Go to QC", lambda: self._set_active_step(2)),
            ("Go to Explore", lambda: self._set_active_step(3)),
            ("Go to Export", lambda: self._set_active_step(4)),
        ]

        def refresh_list(text: str = "") -> None:
            list_widget.clear()
            for name, _ in commands:
                if text.lower() in name.lower():
                    list_widget.addItem(name)

        def run_selected() -> None:
            item = list_widget.currentItem()
            if not item:
                return
            name = item.text()
            for cmd_name, handler in commands:
                if cmd_name == name:
                    handler()
                    dialog.accept()
                    return

        search.textChanged.connect(refresh_list)
        list_widget.itemActivated.connect(lambda _: run_selected())
        search.returnPressed.connect(run_selected)
        refresh_list()
        search.setFocus()
        dialog.exec()

    def _set_run_ui_state(self, running: bool) -> None:
        self._analysis_running = running
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(not running)
        if hasattr(self, "quick_btn"):
            self.quick_btn.setEnabled(not running)
        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setEnabled(running)
        self.doctor_run_btn.setEnabled(not running)
        self.doctor_pbc_btn.setEnabled(True)
        if hasattr(self, "run_progress"):
            if running:
                self.run_progress.setVisible(True)
                self.run_progress.setRange(0, 0)
                self.run_progress.setValue(0)
            else:
                self.run_progress.setVisible(False)
        if running:
            self.status_bar.showMessage("Analysis running…", 3000)
        else:
            self.status_bar.showMessage("Ready", 2000)

    def _clear_results_view(self) -> None:
        self.current_result = None
        self._timeline_stats_cache = {}
        self._timeline_event_cache = {}
        self._time_window = None
        if self.timeline_region is not None:
            try:
                self.timeline_plot.removeItem(self.timeline_region)
            except Exception:
                pass
            self.timeline_region = None
        self.overview_card.setText("Running analysis…")
        self.overview_text.setText("")
        self.qc_summary_label.setText("Running analysis…")
        self.qc_text.setText("")
        self.report_text.setText("")
        self.timeline_plot.clear()
        self.timeline_event_plot.clear()
        self.timeline_summary_label.setText("")
        self.timeline_summary_label.setVisible(False)
        self.timeline_stats_status.setText("Not computed yet.")
        self.hist_plot.clear()
        self.hist_zero_plot.clear()
        self.hist_info.setText("")
        self.matrix_view.setImage(np.zeros((1, 1)), autoLevels=True)
        self.matrix_info.setText("")
        self.event_plot.clear()
        self.event_info.setText("")
        empty_model = QtGui.QStandardItemModel()
        self.per_frame_table.setModel(empty_model)
        self.per_solvent_table.setModel(empty_model)
        if hasattr(self, "explore_inspector_text"):
            self.explore_inspector_text.setText("Awaiting results…")

    def _copy_current_plot(self) -> None:
        index = self.plots_tabs.currentIndex()
        if index == 0:
            self._copy_widget_to_clipboard(self.hist_plot)
        elif index == 1:
            self._copy_widget_to_clipboard(self.matrix_view)
        elif index == 2:
            self._copy_widget_to_clipboard(self.event_plot)

    def _ensure_export_extension(self, path: str, selected_filter: str) -> str:
        if not path:
            return path
        if Path(path).suffix:
            return path
        ext_map = {
            "PNG": ".png",
            "SVG": ".svg",
            "PDF": ".pdf",
            "CSV": ".csv",
        }
        for key, ext in ext_map.items():
            if key in selected_filter:
                return f"{path}{ext}"
        return path

    def _export_current_plot(self) -> None:
        index = self.plots_tabs.currentIndex()
        if index == 0:
            self._export_plot(
                self.hist_plot,
                "histogram.png",
                csv_exporter=self._write_histogram_csv,
            )
        elif index == 1:
            self._export_plot(
                self.matrix_view,
                "heatmap.png",
                csv_exporter=self._write_matrix_csv,
            )
        elif index == 2:
            self._export_plot(
                self.event_plot,
                "events.png",
                csv_exporter=self._write_event_raster_csv,
            )

    def _export_plot(
        self,
        plot_widget: QtWidgets.QWidget,
        default_name: str,
        csv_exporter: Callable[[str], None] | None = None,
        title: str = "Export Plot",
    ) -> None:
        default_path = default_name
        out_dir = self._effective_output_dir()
        if out_dir is not None:
            default_path = str(out_dir / default_name)
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            default_path,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;CSV (*.csv)",
        )
        if not path:
            return
        path = self._ensure_export_extension(path, selected_filter)
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            if csv_exporter is None:
                self.status_bar.showMessage("CSV export not available for this plot.", 5000)
                return
            csv_exporter(path)
            return
        try:
            if suffix == ".pdf":
                pdf_writer = QtGui.QPdfWriter(path)
                painter = QtGui.QPainter(pdf_writer)
                plot_widget.render(painter)
                painter.end()
                return
            if suffix == ".svg":
                if isinstance(plot_widget, pg.PlotWidget):
                    exporter = pg.exporters.SVGExporter(plot_widget.plotItem)
                    exporter.export(path)
                else:
                    self._render_widget_svg(plot_widget, path)
                return
            if isinstance(plot_widget, pg.PlotWidget):
                exporter = pg.exporters.ImageExporter(plot_widget.plotItem)
                exporter.export(path)
            else:
                pixmap = plot_widget.grab()
                pixmap.save(path)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Plot export failed")
            self.status_bar.showMessage(f"Plot export failed: {exc}", 8000)

    def _render_widget_svg(self, widget: QtWidgets.QWidget, path: str) -> None:
        try:
            from PyQt6 import QtSvg
        except Exception:
            self.status_bar.showMessage("SVG export requires QtSvg.", 5000)
            return
        generator = QtSvg.QSvgGenerator()
        generator.setFileName(path)
        size = widget.size()
        generator.setSize(size)
        generator.setViewBox(QtCore.QRect(0, 0, size.width(), size.height()))
        painter = QtGui.QPainter(generator)
        widget.render(painter)
        painter.end()

    def _write_timeline_csv(self, path: str) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.status_bar.showMessage("No timeline data available to export.", 5000)
            return
        metric = self.timeline_metric_combo.currentText()
        metric_col = metric or "value"
        secondary_metric = self.timeline_secondary_combo.currentText()
        clamp = self.timeline_clamp_check.isChecked()
        smooth = self.timeline_smooth_check.isChecked()
        smooth_window = int(self.timeline_smooth_window.value())
        overlay = self.timeline_overlay.isChecked()
        try:
            if overlay:
                rows = []
                for name, soz in self.current_result.soz_results.items():
                    if soz.per_frame.empty or "time" not in soz.per_frame.columns:
                        continue
                    time_ns = (
                        pd.to_numeric(soz.per_frame["time"], errors="coerce").to_numpy() / 1000.0
                    )
                    series = self._metric_series(soz.per_frame, metric)
                    if series is None or series.size == 0:
                        continue
                    y = np.maximum(series, 0) if clamp else series
                    if smooth and smooth_window > 1:
                        y = (
                            pd.Series(y)
                            .rolling(smooth_window, center=True, min_periods=1)
                            .mean()
                            .to_numpy()
                        )
                    valid = np.isfinite(time_ns)
                    if not np.all(valid):
                        time_ns = time_ns[valid]
                        y = y[valid]
                    rows.append(pd.DataFrame({"soz": name, "time_ns": time_ns, metric_col: y}))
                if not rows:
                    self.status_bar.showMessage("Timeline series missing for export.", 5000)
                    return
                df = pd.concat(rows, ignore_index=True)
            else:
                soz_name = self._selected_soz_name()
                if not soz_name:
                    self.status_bar.showMessage("Select a SOZ to export the timeline.", 5000)
                    return
                soz = self.current_result.soz_results[soz_name]
                if soz.per_frame.empty or "time" not in soz.per_frame.columns:
                    self.status_bar.showMessage("Timeline data missing for selected SOZ.", 5000)
                    return
                time_ns = (
                    pd.to_numeric(soz.per_frame["time"], errors="coerce").to_numpy() / 1000.0
                )
                series = self._metric_series(soz.per_frame, metric)
                if series is None or series.size == 0:
                    self.status_bar.showMessage("Timeline metric missing for export.", 5000)
                    return
                y = np.maximum(series, 0) if clamp else series
                if smooth and smooth_window > 1:
                    y = (
                        pd.Series(y)
                        .rolling(smooth_window, center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )
                valid = np.isfinite(time_ns)
                time_ns = time_ns[valid]
                y = y[valid]
                data = {"time_ns": time_ns, metric_col: y}
                if secondary_metric not in ("None", metric):
                    sec_series = self._metric_series(soz.per_frame, secondary_metric)
                    if sec_series is not None and sec_series.size:
                        y2 = np.maximum(sec_series, 0) if clamp else sec_series
                        if smooth and smooth_window > 1:
                            y2 = (
                                pd.Series(y2)
                                .rolling(smooth_window, center=True, min_periods=1)
                                .mean()
                                .to_numpy()
                            )
                        y2 = y2[valid]
                        data[secondary_metric] = y2
                df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            self.status_bar.showMessage(f"Timeline CSV saved: {path}", 5000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Failed to export timeline CSV")
            self.status_bar.showMessage(f"Timeline export failed: {exc}", 8000)

    def _write_histogram_csv(self, path: str) -> None:
        if not self.current_result:
            self.status_bar.showMessage("No histogram data available to export.", 5000)
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            self.status_bar.showMessage("Select a SOZ to export the histogram.", 5000)
            return
        metric = self.hist_metric_combo.currentText()
        if not metric:
            self.status_bar.showMessage("Select a histogram metric to export.", 5000)
            return
        df_full = self.current_result.soz_results[soz_name].per_frame
        df = self._filtered_per_frame(df_full)
        if metric not in df.columns:
            self.status_bar.showMessage("Histogram metric not found in data.", 5000)
            return
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        if metric == "time":
            values = values / 1000.0
        if values.size == 0:
            self.status_bar.showMessage("Histogram data is empty.", 5000)
            return
        bins = self.hist_bins_spin.value()
        zero_mask = values == 0
        split_zeros = self.hist_zero_split_check.isChecked()
        if split_zeros:
            values_for_hist = values[~zero_mask]
        else:
            values_for_hist = values
        if values_for_hist.size == 0:
            self.status_bar.showMessage("No non-zero values to export.", 5000)
            return
        if np.allclose(values_for_hist, np.round(values_for_hist)):
            vmin = int(np.min(values_for_hist))
            vmax = int(np.max(values_for_hist))
            if vmax - vmin <= 200:
                bins = np.arange(vmin, vmax + 2) - 0.5
        counts, edges = np.histogram(values_for_hist, bins=bins)
        if self.hist_norm_check.isChecked():
            total = counts.sum()
            counts = counts / total if total > 0 else counts
        centers = (edges[:-1] + edges[1:]) / 2.0
        data = {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "count": counts,
        }
        try:
            pd.DataFrame(data).to_csv(path, index=False)
            self.status_bar.showMessage(f"Histogram CSV saved: {path}", 5000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Failed to export histogram CSV")
            self.status_bar.showMessage(f"Histogram export failed: {exc}", 8000)

    def _write_matrix_csv(self, path: str) -> None:
        if not self.current_result:
            self.status_bar.showMessage("No heatmap data available to export.", 5000)
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            self.status_bar.showMessage("Select a SOZ to export the heatmap.", 5000)
            return
        source = self.matrix_source_combo.currentText()
        try:
            if source.startswith("Residue hydration"):
                if not self.current_result.hydration_results:
                    self.status_bar.showMessage("No hydration results available to export.", 5000)
                    return
                hydration = next(iter(self.current_result.hydration_results.values()))
                table = hydration.table
                if table.empty:
                    self.status_bar.showMessage("Hydration table is empty.", 5000)
                    return
                table.to_csv(path, index=False)
                self.status_bar.showMessage(f"Heatmap CSV saved: {path}", 5000)
                return

            per_frame_full = self.current_result.soz_results[soz_name].per_frame
            per_frame = self._filtered_per_frame(per_frame_full)
            per_solvent = self.current_result.soz_results[soz_name].per_solvent
            matrix, ids, time_ns, msg = self._build_presence_matrix(
                per_frame,
                per_solvent,
                top_n=self.matrix_top_spin.value(),
                min_occ_pct=self.matrix_min_occ_spin.value(),
            )
            if matrix is None:
                self.status_bar.showMessage(msg or "Heatmap data unavailable.", 5000)
                return
            if time_ns is None:
                time_ns = np.arange(matrix.shape[1], dtype=float)
            if len(time_ns) != matrix.shape[1]:
                min_len = min(len(time_ns), matrix.shape[1])
                time_ns = time_ns[:min_len]
                matrix = matrix[:, :min_len]
            columns = [f"{val:.3f}" for val in time_ns[: matrix.shape[1]]]
            df = pd.DataFrame(matrix.astype(int), columns=columns)
            df.insert(0, "solvent_id", ids)
            df.to_csv(path, index=False)
            self.status_bar.showMessage(f"Heatmap CSV saved: {path}", 5000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Failed to export heatmap CSV")
            self.status_bar.showMessage(f"Heatmap export failed: {exc}", 8000)

    def _write_event_raster_csv(self, path: str) -> None:
        if not self.current_result:
            self.status_bar.showMessage("No event raster data available to export.", 5000)
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            self.status_bar.showMessage("Select a SOZ to export the event raster.", 5000)
            return
        try:
            per_frame_full = self.current_result.soz_results[soz_name].per_frame
            per_frame = self._filtered_per_frame(per_frame_full)
            per_solvent = self.current_result.soz_results[soz_name].per_solvent
            matrix, ids, time_ns, msg = self._build_presence_matrix(
                per_frame,
                per_solvent,
                top_n=self.matrix_top_spin.value(),
                min_occ_pct=self.matrix_min_occ_spin.value(),
            )
            if matrix is None:
                self.status_bar.showMessage(msg or "Event raster data unavailable.", 5000)
                return
            if time_ns is None:
                time_ns = np.arange(matrix.shape[1], dtype=float)
            if len(time_ns) != matrix.shape[1]:
                min_len = min(len(time_ns), matrix.shape[1])
                time_ns = time_ns[:min_len]
                matrix = matrix[:, :min_len]
            stride = max(1, int(self.event_stride_spin.value()))
            if stride > 1:
                matrix = matrix[:, ::stride]
                time_ns = time_ns[::stride]
            segment_mode = self.event_segment_check.isChecked()
            min_duration = int(self.event_min_duration_spin.value())
            if segment_mode:
                rows = []
                for row_idx in range(matrix.shape[0]):
                    row = matrix[row_idx]
                    if not np.any(row):
                        continue
                    start = None
                    for col_idx, val in enumerate(row):
                        if val and start is None:
                            start = col_idx
                        if (not val or col_idx == len(row) - 1) and start is not None:
                            end = col_idx if val else col_idx - 1
                            length = end - start + 1
                            if length >= min_duration:
                                rows.append(
                                    {
                                        "solvent_id": ids[row_idx],
                                        "row": row_idx,
                                        "start_frame": start * stride,
                                        "end_frame": end * stride,
                                        "start_ns": time_ns[start],
                                        "end_ns": time_ns[end],
                                        "duration_frames": length,
                                    }
                                )
                            start = None
                if not rows:
                    self.status_bar.showMessage("No events available to export.", 5000)
                    return
                df = pd.DataFrame(rows)
            else:
                rows_idx, cols_idx = np.where(matrix > 0)
                if rows_idx.size == 0:
                    self.status_bar.showMessage("No events available to export.", 5000)
                    return
                df = pd.DataFrame(
                    {
                        "solvent_id": [ids[idx] for idx in rows_idx],
                        "row": rows_idx,
                        "frame": cols_idx * stride,
                        "time_ns": time_ns[cols_idx],
                    }
                )
            df.to_csv(path, index=False)
            self.status_bar.showMessage(f"Event raster CSV saved: {path}", 5000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Failed to export event raster CSV")
            self.status_bar.showMessage(f"Event raster export failed: {exc}", 8000)

    def _downsample(self, x, y, max_points: int = 5000):
        if len(x) <= max_points:
            return x, y
        stride = max(1, int(len(x) / max_points))
        return x[::stride], y[::stride]

    def _step_series(self, x, y):
        if len(x) <= 1:
            return x, y
        x_step = np.repeat(np.asarray(x), 2)[1:]
        y_step = np.repeat(np.asarray(y), 2)[:-1]
        return x_step, y_step

    def _style_plot(self, plot: pg.PlotWidget, title: str | None = None) -> None:
        tokens = self._get_theme_tokens()
        plot.setBackground(tokens["plot_bg"])
        plot.showGrid(x=True, y=True, alpha=0.2)
        if title:
            size = int(10 * self._ui_scale)
            plot.setTitle(title, color=tokens["text"], size=f"{size}pt")
        for axis_name in ("bottom", "left", "right"):
            axis = plot.getAxis(axis_name)
            if axis is None:
                continue
            axis.setPen(pg.mkPen(tokens["plot_axis"]))
            axis.setTextPen(pg.mkPen(tokens["plot_fg"]))
        plot.plotItem.legend = None

    def _update_timeline_views(self) -> None:
        if not hasattr(self, "timeline_secondary_view"):
            return
        vb = self.timeline_plot.plotItem.vb
        self.timeline_secondary_view.setGeometry(vb.sceneBoundingRect())
        self.timeline_secondary_view.linkedViewChanged(vb, self.timeline_secondary_view.XAxis)

    def _update_event_views(self) -> None:
        if not hasattr(self, "timeline_event_secondary_view"):
            return
        vb = self.timeline_event_plot.plotItem.vb
        self.timeline_event_secondary_view.setGeometry(vb.sceneBoundingRect())
        self.timeline_event_secondary_view.linkedViewChanged(vb, self.timeline_event_secondary_view.XAxis)

    def _clear_timeline_secondary(self) -> None:
        items = getattr(self, "_timeline_secondary_items", [])
        for item in items:
            try:
                self.timeline_secondary_view.removeItem(item)
            except Exception:
                pass
        self._timeline_secondary_items = []

    def _clear_event_secondary(self) -> None:
        items = getattr(self, "_timeline_event_secondary_items", [])
        for item in items:
            try:
                self.timeline_event_secondary_view.removeItem(item)
            except Exception:
                pass
        self._timeline_event_secondary_items = []

    def _metric_series(self, per_frame: pd.DataFrame, metric: str) -> np.ndarray | None:
        if metric == "occupancy_fraction":
            values = pd.to_numeric(per_frame.get("n_solvent", []), errors="coerce").fillna(0)
            return (values > 0).astype(float).to_numpy()
        if metric in per_frame.columns:
            return pd.to_numeric(per_frame[metric], errors="coerce").fillna(0).to_numpy()
        return None

    def _compute_occupancy_counts(
        self, per_frame: pd.DataFrame, soz_name: str
    ) -> tuple[np.ndarray, str]:
        if "n_solvent" in per_frame.columns:
            counts = pd.to_numeric(per_frame["n_solvent"], errors="coerce").fillna(0).to_numpy()
            return counts, "n_solvent"
        if "solvent_ids" in per_frame.columns:
            ids_series = per_frame["solvent_ids"].fillna("").astype(str)
            counts = np.array([len([val for val in ids.split(";") if val]) for ids in ids_series], dtype=float)
            if self.run_logger:
                self.run_logger.info(
                    "Timeline event plot: n_solvent missing for %s; derived counts from solvent_ids.",
                    soz_name,
                )
            return counts, "solvent_ids"
        if self.run_logger:
            self.run_logger.warning(
                "Timeline event plot: no n_solvent or solvent_ids for %s; counts set to zero.",
                soz_name,
            )
        return np.zeros(len(per_frame), dtype=float), "none"

    def _compute_entry_exit_series(
        self, per_frame: pd.DataFrame, counts: np.ndarray, soz_name: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        entries = np.array([])
        exits = np.array([])
        source = "none"

        if "entries" in per_frame.columns and "exits" in per_frame.columns:
            entries = pd.to_numeric(per_frame["entries"], errors="coerce").fillna(0).to_numpy()
            exits = pd.to_numeric(per_frame["exits"], errors="coerce").fillna(0).to_numpy()
            source = "per_frame"

        if source == "per_frame" and (entries.sum() > 0 or exits.sum() > 0):
            return entries, exits, source

        if counts.size:
            diff = np.diff(counts, prepend=counts[0])
            entries = np.maximum(diff, 0)
            exits = np.maximum(-diff, 0)
            source = "count_diff"
            if self.run_logger:
                self.run_logger.warning(
                    "Timeline event plot: entries/exits missing for %s; using Δcount approximation.",
                    soz_name,
                )
            return entries, exits, source

        if self.run_logger and source == "per_frame":
            self.run_logger.info("Timeline event plot: entries/exits empty for %s", soz_name)
        return entries, exits, source

    def _update_timeline_event_plot(
        self,
        smooth: bool,
        smooth_window: int,
        step_mode: bool,
        show_markers: bool,
    ) -> None:
        tokens = self._get_theme_tokens()
        self.timeline_event_plot.clear()
        if self.timeline_event_plot.plotItem.legend is None:
            self.timeline_event_plot.addLegend()
        self._clear_event_secondary()
        self._timeline_event_cache = {}
        self._timeline_event_hover_index = None

        mode = self.timeline_event_mode_combo.currentText()
        norm = self.timeline_event_norm_combo.currentText()
        signed = self.timeline_event_signed_check.isChecked()
        if mode == "Cumulative events":
            y_units = "cumulative events"
            title = "Entry / Exit (Cumulative)"
        elif mode == "Rate (events/ns)":
            y_units = "events/ns"
            title = "Entry / Exit Rate"
        else:
            if norm == "per ns":
                y_units = "events/ns"
            elif norm == "per 100 frames":
                y_units = "events/100 frames"
            else:
                y_units = "events/frame"
            title = "Entry / Exit Events"
        self.timeline_event_plot.setTitle(
            title, color=tokens["text"], size=f"{int(10 * self._ui_scale)}pt"
        )
        if signed:
            self.timeline_event_plot.setLabel("left", f"Entries (+) / Exits (-) ({y_units})")
        else:
            self.timeline_event_plot.setLabel("left", f"Events ({y_units})")

        try:
            if not self.current_result or not self.current_result.soz_results:
                self.timeline_event_plot.addItem(
                    pg.TextItem(
                        "Run analysis to compute entry/exit rates.", color=tokens["text_muted"]
                    )
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return

            soz_name = self._selected_soz_name()
            if not soz_name:
                self.timeline_event_plot.addItem(
                    pg.TextItem("Select a SOZ to view entry/exit rates.", color=tokens["text_muted"])
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return

            soz = self.current_result.soz_results[soz_name]
            if soz.per_frame.empty:
                self.timeline_event_plot.addItem(
                    pg.TextItem("No per-frame data available.", color=tokens["text_muted"])
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return
            if "time" not in soz.per_frame.columns:
                self.timeline_event_plot.addItem(
                    pg.TextItem("Per-frame timing data missing.", color=tokens["text_muted"])
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return

            time_ps = pd.to_numeric(soz.per_frame["time"], errors="coerce").to_numpy()
            if time_ps.size == 0:
                self.timeline_event_plot.addItem(
                    pg.TextItem("No time data available.", color=tokens["text_muted"])
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return

            counts, count_source = self._compute_occupancy_counts(soz.per_frame, soz_name)
            entries, exits, source = self._compute_entry_exit_series(soz.per_frame, counts, soz_name)
            if count_source == "none":
                self.timeline_event_plot.addItem(
                    pg.TextItem(
                        "No occupancy counts available (n_solvent missing).",
                        color=tokens["text_muted"],
                    )
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return
            if source != "per_frame":
                warn_key = (soz_name, source)
                if warn_key not in self._timeline_event_warning_shown:
                    self._timeline_event_warning_shown.add(warn_key)
                    self.status_bar.showMessage(
                        "Entry/Exit derived from occupancy counts (explicit entries/exits missing). "
                        "Re-run analysis to compute explicit transition stats.",
                        8000,
                    )

            min_len = min(len(time_ps), len(counts), len(entries), len(exits))
            if min_len == 0:
                self.timeline_event_plot.addItem(
                    pg.TextItem(
                        "Entry/exit series is empty after filtering.", color=tokens["text_muted"]
                    )
                )
                self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                return
            if min_len != len(time_ps) and self.run_logger:
                self.run_logger.warning(
                    "Timeline event plot length mismatch for %s (time=%d counts=%d entries=%d exits=%d).",
                    soz_name,
                    len(time_ps),
                    len(counts),
                    len(entries),
                    len(exits),
                )
            time_ps = time_ps[:min_len]
            counts = counts[:min_len]
            entries = entries[:min_len]
            exits = exits[:min_len]

            valid_mask = np.isfinite(time_ps)
            if not np.all(valid_mask):
                if self.run_logger:
                    self.run_logger.warning(
                        "Timeline event plot: dropping %d frames with non-finite time values.",
                        int(np.sum(~valid_mask)),
                    )
                time_ps = time_ps[valid_mask]
                counts = counts[valid_mask]
                entries = entries[valid_mask]
                exits = exits[valid_mask]
                if time_ps.size == 0:
                    self.timeline_event_plot.addItem(
                        pg.TextItem("No valid time values available.", color=tokens["text_muted"])
                    )
                    self.timeline_event_plot.setYRange(0, 1.0, padding=0)
                    return

            time_ns = time_ps / 1000.0
            dt_eff = float(np.median(np.diff(time_ns))) if len(time_ns) > 1 else 0.0
            if dt_eff <= 0 and self.run_logger:
                self.run_logger.warning("Timeline event plot: invalid dt for %s (dt=%.6f).", soz_name, dt_eff)

            bin_ns = float(self.timeline_event_bin_spin.value())
            if bin_ns > 0:
                start = float(time_ns[0])
                bin_index = np.floor((time_ns - start) / bin_ns).astype(int)
                n_bins = int(bin_index.max()) + 1 if bin_index.size else 0
                bin_time = []
                bin_counts = []
                bin_entries = []
                bin_exits = []
                bin_frames = []
                bin_duration = []
                for idx in range(n_bins):
                    mask = bin_index == idx
                    if not np.any(mask):
                        continue
                    times = time_ns[mask]
                    bin_time.append(start + (idx + 0.5) * bin_ns)
                    bin_counts.append(float(np.mean(counts[mask])))
                    bin_entries.append(float(np.sum(entries[mask])))
                    bin_exits.append(float(np.sum(exits[mask])))
                    bin_frames.append(int(np.sum(mask)))
                    if len(times) > 1:
                        duration = float(times.max() - times.min())
                    else:
                        duration = dt_eff if dt_eff > 0 else bin_ns
                    if duration <= 0:
                        duration = bin_ns if bin_ns > 0 else 1.0
                    bin_duration.append(duration)
                time_ns = np.asarray(bin_time, dtype=float)
                counts = np.asarray(bin_counts, dtype=float)
                entries = np.asarray(bin_entries, dtype=float)
                exits = np.asarray(bin_exits, dtype=float)
                frames_per_bin = np.asarray(bin_frames, dtype=float)
                bin_duration = np.asarray(bin_duration, dtype=float)
            else:
                frames_per_bin = np.ones_like(entries, dtype=float)
                bin_duration = np.full_like(entries, dt_eff if dt_eff > 0 else 1.0, dtype=float)

            delta = entries - exits

            entries_plot = entries.copy()
            exits_plot = exits.copy()
            y_units = "events/frame"

            if mode == "Events per frame":
                entries_plot = entries / frames_per_bin
                exits_plot = exits / frames_per_bin
                if norm == "per ns" and dt_eff > 0:
                    entries_plot = entries_plot / dt_eff
                    exits_plot = exits_plot / dt_eff
                    y_units = "events/ns"
                elif norm == "per 100 frames":
                    entries_plot = entries_plot * 100.0
                    exits_plot = exits_plot * 100.0
                    y_units = "events/100 frames"
            elif mode == "Rate (events/ns)":
                entries_plot = entries / bin_duration
                exits_plot = exits / bin_duration
                y_units = "events/ns"
            elif mode == "Cumulative events":
                entries_plot = np.cumsum(entries)
                exits_plot = np.cumsum(exits)
                y_units = "cumulative events"

            if smooth and smooth_window > 1 and entries_plot.size > 0:
                if smooth_window >= len(entries_plot) and self.run_logger:
                    self.run_logger.warning(
                        "Timeline event plot: smoothing window %d >= series length %d.",
                        smooth_window,
                        len(entries_plot),
                    )
                entries_plot = pd.Series(entries_plot).rolling(
                    smooth_window, center=True, min_periods=1
                ).mean().to_numpy()
                exits_plot = pd.Series(exits_plot).rolling(
                    smooth_window, center=True, min_periods=1
                ).mean().to_numpy()

            entries_plot = np.nan_to_num(entries_plot, nan=0.0, posinf=0.0, neginf=0.0)
            exits_plot = np.nan_to_num(exits_plot, nan=0.0, posinf=0.0, neginf=0.0)
            if (
                entries_plot.size
                and exits_plot.size
                and np.sum(entries) > 0
                and np.max(np.abs(entries_plot)) == 0
                and self.run_logger
            ):
                self.run_logger.warning(
                    "Timeline event plot: entries/exits non-zero but plot values are all zero. "
                    "Check normalization/binning settings."
                )

            x_entries, entry_vals = self._downsample(time_ns, entries_plot)
            x_exits, exit_vals = self._downsample(time_ns, exits_plot)
            if step_mode:
                x_entries, entry_vals = self._step_series(x_entries, entry_vals)
                x_exits, exit_vals = self._step_series(x_exits, exit_vals)

            entry_symbol = "o" if show_markers else None
            exit_symbol = "o" if show_markers else None
            entry_brush = pg.mkBrush(tokens["entry"]) if show_markers else None
            exit_brush = pg.mkBrush(tokens["exit"]) if show_markers else None

            signed = self.timeline_event_signed_check.isChecked()
            split_axes = self.timeline_event_split_check.isChecked() and not signed
            if mode == "Cumulative events":
                title = "Entry / Exit (Cumulative)"
                y_label = "Cumulative events"
            elif mode == "Rate (events/ns)":
                title = "Entry / Exit Rate"
                y_label = f"Events ({y_units})"
            else:
                title = "Entry / Exit Events"
                y_label = f"Events ({y_units})"
            self.timeline_event_plot.setTitle(
                title, color=tokens["text"], size=f"{int(10 * self._ui_scale)}pt"
            )

            if signed:
                exit_vals = -exit_vals
                self.timeline_event_plot.plotItem.getAxis("right").setVisible(False)
                self.timeline_event_plot.setLabel("left", f"Entries (+) / Exits (-) ({y_units})")
                self.timeline_event_plot.plot(
                    x_entries,
                    entry_vals,
                    pen=pg.mkPen(tokens["entry"], width=self._plot_line_width),
                    name="entries",
                    symbol=entry_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=entry_brush,
                )
                self.timeline_event_plot.plot(
                    x_exits,
                    exit_vals,
                    pen=pg.mkPen(tokens["exit"], width=self._plot_line_width),
                    name="exits",
                    symbol=exit_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=exit_brush,
                )
            elif split_axes:
                self.timeline_event_plot.plotItem.getAxis("right").setVisible(True)
                self.timeline_event_plot.setLabel("left", f"Entries ({y_units})")
                pen_exit = pg.mkPen(
                    tokens["exit"], width=self._plot_line_width, style=QtCore.Qt.PenStyle.DashLine
                )
                exit_item = pg.PlotDataItem(
                    x_exits,
                    exit_vals,
                    pen=pen_exit,
                    symbol=exit_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=exit_brush,
                )
                self.timeline_event_secondary_view.addItem(exit_item)
                self._timeline_event_secondary_items.append(exit_item)
                self.timeline_event_plot.plot(
                    x_entries,
                    entry_vals,
                    pen=pg.mkPen(tokens["entry"], width=self._plot_line_width),
                    name="entries",
                    symbol=entry_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=entry_brush,
                )
                self.timeline_event_plot.plotItem.getAxis("right").setLabel(f"Exits ({y_units})")
            else:
                self.timeline_event_plot.plotItem.getAxis("right").setVisible(False)
                self.timeline_event_plot.setLabel("left", y_label)
                self.timeline_event_plot.plot(
                    x_entries,
                    entry_vals,
                    pen=pg.mkPen(tokens["entry"], width=self._plot_line_width),
                    name="entries",
                    symbol=entry_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=entry_brush,
                )
                self.timeline_event_plot.plot(
                    x_exits,
                    exit_vals,
                    pen=pg.mkPen(tokens["exit"], width=self._plot_line_width),
                    name="exits",
                    symbol=exit_symbol,
                    symbolSize=self._plot_marker_size if show_markers else None,
                    symbolBrush=exit_brush,
                )

            max_entry = float(np.nanmax(entry_vals)) if entry_vals.size else 0.0
            max_exit = float(np.nanmax(exit_vals)) if exit_vals.size else 0.0
            max_ev = float(max(abs(max_entry), abs(max_exit)))
            if max_ev <= 0:
                self.timeline_event_plot.addItem(
                    pg.TextItem("No entry/exit events detected.", color=tokens["text_muted"])
                )
                self.timeline_event_plot.setYRange(-1.0 if signed else 0, 1.0, padding=0)
            else:
                self.timeline_event_plot.setYRange(
                    -max_ev * 1.2 if signed else 0,
                    max_ev * 1.2 if signed else max_ev * 1.2,
                    padding=0,
                )
                if split_axes:
                    self.timeline_event_secondary_view.setYRange(0, max_exit * 1.2, padding=0)

            self._timeline_event_cache = {
                "time_ns": time_ns,
                "count": counts,
                "delta": delta,
                "entries": entries,
                "exits": exits,
                "entries_plot": entries_plot,
                "exits_plot": exits_plot,
                "entry_rate": entries / bin_duration,
                "exit_rate": exits / bin_duration,
                "frames_per_bin": frames_per_bin,
                "bin_duration": bin_duration,
                "dt_eff": dt_eff,
                "source": source,
                "count_source": count_source,
                "mode": mode,
                "norm": norm,
                "units": y_units,
                "signed": signed,
            }

            if source != "per_frame" and self.run_logger:
                self.run_logger.info(
                    "Timeline event plot for %s uses %s-derived entries/exits.", soz_name, source
                )
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Timeline event plot failed")
            self.timeline_event_plot.addItem(
                pg.TextItem(f"Entry/exit plot error: {exc}", color=tokens["text_muted"])
            )
            self.timeline_event_plot.setYRange(0, 1.0, padding=0)

    def _update_event_controls_state(self) -> None:
        mode = self.timeline_event_mode_combo.currentText()
        if mode == "Rate (events/ns)":
            self.timeline_event_norm_combo.setCurrentText("per ns")
            self.timeline_event_norm_combo.setEnabled(False)
        elif mode == "Cumulative events":
            self.timeline_event_norm_combo.setCurrentText("none")
            self.timeline_event_norm_combo.setEnabled(False)
        else:
            self.timeline_event_norm_combo.setEnabled(True)

        if self.timeline_event_signed_check.isChecked():
            self.timeline_event_split_check.setChecked(False)
            self.timeline_event_split_check.setEnabled(False)
        else:
            self.timeline_event_split_check.setEnabled(True)

    def _update_plots_controls_state(self, index: int | None = None) -> None:
        if not hasattr(self, "plots_tabs"):
            return
        if index is None:
            index = self.plots_tabs.currentIndex()
        show_hist = index == 0
        show_matrix = index in (1, 2)
        show_event = index == 2
        if hasattr(self, "hist_controls_row"):
            self.hist_controls_row.setVisible(show_hist)
        if hasattr(self, "matrix_controls_row"):
            self.matrix_controls_row.setVisible(show_matrix)
        if hasattr(self, "event_controls_row"):
            self.event_controls_row.setVisible(show_event)
        show_source = index == 1
        if hasattr(self, "matrix_source_label"):
            self.matrix_source_label.setVisible(show_source)
        if hasattr(self, "matrix_source_combo"):
            self.matrix_source_combo.setVisible(show_source)

    def _on_timeline_event_hover(self, pos: QtCore.QPointF) -> None:
        cache = self._timeline_event_cache
        if not cache or "time_ns" not in cache:
            return
        vb = self.timeline_event_plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        times = np.asarray(cache.get("time_ns", []), dtype=float)
        if times.size == 0:
            return
        idx = int(np.searchsorted(times, x))
        idx = max(0, min(idx, len(times) - 1))
        if self._timeline_event_hover_index == idx:
            return
        self._timeline_event_hover_index = idx

        count = float(np.asarray(cache.get("count", []), dtype=float)[idx])
        delta = float(np.asarray(cache.get("delta", []), dtype=float)[idx])
        entries = float(np.asarray(cache.get("entries_plot", []), dtype=float)[idx])
        exits = float(np.asarray(cache.get("exits_plot", []), dtype=float)[idx])
        if bool(cache.get("signed", False)):
            exits = -exits
        dt_eff = float(np.asarray(cache.get("bin_duration", []), dtype=float)[idx]) if "bin_duration" in cache else 0.0
        units = str(cache.get("units", "events/frame"))

        text = (
            f"t = {times[idx]:.3f} ns\n"
            f"count = {count:.3f}\n"
            f"delta = {delta:.3f}\n"
            f"entries = {entries:.3f} ({units})\n"
            f"exits = {exits:.3f} ({units})\n"
            f"dt_eff = {dt_eff:.4f} ns"
        )
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self)

    def _write_entry_exit_csv(self, path: str) -> None:
        cache = self._timeline_event_cache
        if not cache or "time_ns" not in cache:
            self.status_bar.showMessage("Entry/exit data not available to export.", 5000)
            return
        try:
            signed = bool(cache.get("signed", False))
            entries_plot = np.asarray(cache.get("entries_plot", []), dtype=float)
            exits_plot = np.asarray(cache.get("exits_plot", []), dtype=float)
            if signed:
                exits_plot = -exits_plot
            data = {
                "time_ns": cache.get("time_ns", []),
                "count": cache.get("count", []),
                "delta": cache.get("delta", []),
                "entries": entries_plot,
                "exits": exits_plot,
                "entry_rate": cache.get("entry_rate", []),
                "exit_rate": cache.get("exit_rate", []),
            }
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            self.status_bar.showMessage(f"Entry/exit CSV saved: {path}", 5000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Failed to export entry/exit CSV")
            self.status_bar.showMessage(f"Entry/exit export failed: {exc}", 8000)

    def _export_entry_exit_csv(self) -> None:
        default_path = "entry_exit.csv"
        out_dir = self._effective_output_dir()
        if out_dir is not None:
            default_path = str(out_dir / default_path)
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Entry/Exit CSV", default_path, "CSV (*.csv)"
        )
        if not path:
            return
        path = self._ensure_export_extension(path, selected_filter)
        self._write_entry_exit_csv(path)

    def _update_timeline_summary(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            return
        overlay_note = ""
        if self.timeline_overlay.isChecked():
            soz_name = self._selected_soz_name()
            if soz_name:
                overlay_note = f"Overlay mode enabled; stats shown for '{soz_name}'."
            else:
                overlay_note = "Overlay mode enabled."
        soz_name = self._selected_soz_name()
        if not soz_name:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            return
        soz = self.current_result.soz_results[soz_name]
        if soz.per_frame.empty:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            return
        stats = self._timeline_stats_cache.get(soz_name)
        if stats:
            summary = stats.get("summary", "Timeline statistics computed.")
            if overlay_note:
                summary = f"{summary}\n{overlay_note}"
            if self._time_window:
                summary = f"{summary}\nWindow: {self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
            self.timeline_summary_label.setText(summary)
            self.timeline_summary_label.setVisible(True)
            self.timeline_stats_status.setText(stats.get("status", "Computed"))
        else:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")

    def _compute_timeline_statistics(self) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            self.status_bar.showMessage("No analysis loaded.", 5000)
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            self.status_bar.showMessage("Select a SOZ to compute statistics.", 5000)
            return
        soz = self.current_result.soz_results[soz_name]
        if soz.per_frame.empty:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            self.status_bar.showMessage("No per-frame data available for statistics.", 5000)
            return

        self.timeline_stats_status.setText("Computing…")
        QtWidgets.QApplication.processEvents()

        per_frame = self._filtered_per_frame(soz.per_frame)
        time_ps = pd.to_numeric(per_frame.get("time", []), errors="coerce").to_numpy()
        time_ns = time_ps / 1000.0 if time_ps.size else np.array([])
        dt_eff = float(np.median(np.diff(time_ns))) if len(time_ns) > 1 else 0.0

        counts, _ = self._compute_occupancy_counts(per_frame, soz_name)
        counts = counts[: len(time_ns)] if time_ns.size else counts
        if counts.size == 0:
            self.timeline_summary_label.setText("")
            self.timeline_summary_label.setVisible(False)
            self.timeline_stats_status.setText("Not computed yet.")
            self.status_bar.showMessage("No occupancy counts available for statistics.", 5000)
            return

        entries, exits, _ = self._compute_entry_exit_series(per_frame, counts, soz_name)
        min_len = min(len(counts), len(entries), len(exits))
        counts = counts[:min_len]
        entries = entries[:min_len]
        exits = exits[:min_len]

        mean_occ = float(np.mean(counts))
        median_occ = float(np.median(counts))
        max_occ = float(np.max(counts))
        occ_fraction = float(np.mean(counts > 0))

        total_entries = float(np.sum(entries))
        total_exits = float(np.sum(exits))
        total_time = float((len(counts) * dt_eff) if dt_eff > 0 else len(counts))
        mean_entry_rate = total_entries / total_time if total_time > 0 else 0.0
        mean_exit_rate = total_exits / total_time if total_time > 0 else 0.0

        occupied = counts > 0
        segments = []
        start = None
        for idx, flag in enumerate(occupied):
            if flag and start is None:
                start = idx
            if (not flag or idx == len(occupied) - 1) and start is not None:
                end = idx if flag else idx - 1
                length = end - start + 1
                segments.append(length)
                start = None
        if dt_eff <= 0:
            dt_eff = 1.0
        seg_times = [length * dt_eff for length in segments] if segments else []
        mean_res = float(np.mean(seg_times)) if seg_times else 0.0
        median_res = float(np.median(seg_times)) if seg_times else 0.0
        max_res = float(np.max(seg_times)) if seg_times else 0.0

        summary = (
            f"SOZ: {soz_name} | mean {mean_occ:.3f} | median {median_occ:.3f} | max {max_occ:.3f} | "
            f"% frames occupied {occ_fraction:.3f} | total entries {total_entries:.0f} | total exits {total_exits:.0f} | "
            f"mean entry rate {mean_entry_rate:.3f} | mean exit rate {mean_exit_rate:.3f} | "
            f"residence (mean/median/max) {mean_res:.3f}/{median_res:.3f}/{max_res:.3f} ns"
        )

        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        status = f"Computed ({timestamp})"
        self._timeline_stats_cache[soz_name] = {"summary": summary, "status": status}
        self.timeline_summary_label.setText(summary)
        self.timeline_summary_label.setVisible(True)
        self.timeline_stats_status.setText(status)

    def _hist_metric_label(self, metric: str) -> str:
        if metric == "n_solvent":
            return "Waters within cutoff per frame"
        if metric == "entries":
            return "Entry events per frame"
        if metric == "exits":
            return "Exit events per frame"
        if metric == "occupancy_fraction":
            return "Occupied (0/1 per frame)"
        return metric

    def _short_solvent_label(self, solvent_id: str) -> str:
        parts = solvent_id.split(":")
        if len(parts) >= 2:
            resname = parts[0]
            resid = parts[1]
            segid = parts[2] if len(parts) > 2 else ""
            seg = f" {segid}" if segid and segid != "-" else ""
            return f"{resname} {resid}{seg}"
        return solvent_id

    def _matrix_plot_item(self):
        try:
            view = self.matrix_view.getView()
        except Exception:
            return None
        if hasattr(view, "getAxis"):
            return view
        parent = view.parentItem() if hasattr(view, "parentItem") else None
        if parent is not None and hasattr(parent, "getAxis"):
            return parent
        return None

    def _apply_table_filter(self, text: str) -> None:
        for proxy in self._table_proxies.values():
            proxy.setFilterRegularExpression(text)

    def _on_per_solvent_selection_changed(self, selected, deselected=None) -> None:
        sel_model = self.per_solvent_table.selectionModel()
        if sel_model is None or not sel_model.selectedRows():
            self._selected_solvent_id = None
            self._update_matrix_plot()
            self._update_event_plot()
            if hasattr(self, "explore_inspector_text"):
                self.explore_inspector_text.setText("Selected solvent: none")
            return
        proxy = self.per_solvent_table.model()
        if proxy is None:
            return
        # Find solvent_id column.
        solvent_col = None
        for col in range(proxy.columnCount()):
            header = proxy.headerData(col, QtCore.Qt.Orientation.Horizontal)
            if str(header) == "solvent_id":
                solvent_col = col
                break
        if solvent_col is None:
            return
        row = selected.indexes()[0].row()
        index = proxy.index(row, solvent_col)
        solvent_id = str(proxy.data(index))
        if not solvent_id:
            return
        self._selected_solvent_id = solvent_id
        self._update_matrix_plot()
        self._update_event_plot()
        if hasattr(self, "explore_inspector_text"):
            self.explore_inspector_text.setText(f"Selected solvent: {solvent_id}")

    def _set_table(self, view: QtWidgets.QTableView, df: pd.DataFrame) -> None:
        max_rows = 5000
        if len(df) > max_rows:
            df = df.head(max_rows)
            self.status_bar.showMessage(
                f"Table truncated to first {max_rows} rows for responsiveness", 5000
            )
        model = QtGui.QStandardItemModel()
        model.setColumnCount(len(df.columns))
        model.setHorizontalHeaderLabels(df.columns.tolist())
        for _, row in df.iterrows():
            items = [QtGui.QStandardItem(str(val)) for val in row.tolist()]
            model.appendRow(items)
        proxy = QtCore.QSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        proxy.setFilterRegularExpression(self.table_filter.text())
        view.setModel(proxy)
        self._table_proxies[view] = proxy
        header = view.horizontalHeader()
        header.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        scale = float(getattr(self, "_ui_scale", 1.0))
        header.setMinimumSectionSize(max(80, int(80 * scale)))
        view.resizeColumnsToContents()
        if view is self.per_solvent_table:
            if hasattr(self, "_per_solvent_sel_model") and self._per_solvent_sel_model is not None:
                try:
                    self._per_solvent_sel_model.selectionChanged.disconnect(self._on_per_solvent_selection_changed)
                except Exception:
                    pass
            self._per_solvent_sel_model = view.selectionModel()
            if self._per_solvent_sel_model is not None:
                self._per_solvent_sel_model.selectionChanged.connect(self._on_per_solvent_selection_changed)


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
