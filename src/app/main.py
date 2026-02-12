"""SOZLab GUI entrypoint."""
from __future__ import annotations

import copy
import json
import threading
import logging
import os
import sys
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Iterable

import pandas as pd
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, QtPrintSupport
import pyqtgraph as pg
import pyqtgraph.exporters
if str(os.environ.get("SOZLAB_DISABLE_WEBGL", "")).strip().lower() in {"1", "true", "yes"}:
    Density3DWidget = None
else:
    try:
        from .viz_3d import Density3DWidget
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Failed to import Density3DWidget (relative): {e}")
        # Fallback or local dev
        try:
            from app.viz_3d import Density3DWidget
        except ImportError as e2:
            logging.getLogger(__name__).error(f"Failed to import Density3DWidget (absolute): {e2}")
            Density3DWidget = None

logger = logging.getLogger(__name__)


from engine.analysis import SOZAnalysisEngine, load_project_json, write_project_json
from engine.export import export_results
from engine.extract import select_frames, write_extracted_trajectory
from engine.models import (
    ProjectConfig,
    SelectionSpec,
    SOZDefinition,
    SOZNode,
    ProbeConfig,
    SolventConfig,
    DistanceBridgeConfig,
    HbondWaterBridgeConfig,
    HbondHydrationConfig,
    DensityMapConfig,
    WaterDynamicsConfig,
    default_project,
)
from engine.preflight import run_preflight
from engine.serialization import to_jsonable
from engine.logging_utils import setup_run_logger
from engine.resolver import resolve_selection, sanitize_selection_string
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
        self._plot_insights: dict[str, list[str]] = {}
        self._plot_insight_labels: dict[str, list[QtWidgets.QLabel]] = {}
        self._timeline_update_pending = False
        self._hist_update_pending = False
        self._event_update_pending = False
        self._density_update_pending = False
        self._run_progress_total: int | None = None
        self._run_progress_current: int = 0
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
        self._wizard_synced_selection_a: str | None = None
        self._wizard_synced_selection_b: str | None = None
        self._time_window: tuple[float, float] | None = None
        self._selected_solvent_id: str | None = None
        self._event_ids_current: list[str] = []
        self._event_rows_current: np.ndarray | None = None
        self._event_cols_current: np.ndarray | None = None
        self._event_time_current: np.ndarray | None = None
        self._event_scatter_item = None
        self._event_highlight_item = None
        self._syncing_solvent_selection = False
        self._suppress_nav_hover_until_ms = 0
        self._nav_rail_expanded = True
        self._nav_rail_pinned = True
        self._nav_rail_animation = None
        self._toolbar_button_groups: dict[
            QtWidgets.QToolButton, tuple[QtWidgets.QButtonGroup, list[QtWidgets.QToolButton]]
        ] = {}
        self._feature_toolbars: list[QtWidgets.QFrame] = []
        self.timeline_region = None
        self.timeline_highlight_line = None
        self._log_raw_text = ""
        self._last_dt = None
        self._last_dt_source: str | None = None
        self._last_dt_effective: float | None = None
        self._last_dt_warning: str | None = None
        self.settings = QtCore.QSettings("SOZLab", "SOZLab")
        try:
            self._user_scale = float(self.settings.value("ui_scale", 1.0))
        except Exception:
            self._user_scale = 1.0
        theme_value = str(self.settings.value("ui_theme", "light")).lower()
        self._theme_mode = "dark" if theme_value == "dark" else "light"
        density_value = str(self.settings.value("ui_density", "comfortable")).lower()
        self._ui_density = "compact" if density_value == "compact" else "comfortable"
        self._nav_rail_pinned = True
        self._presentation_scale = 1.0
        self._default_debounce_ms = 150
        self._sidebar_content_min_px = 420
        self._sidebar_content_pref_px = 460
        self._sidebar_content_max_px = 760
        self._nav_rail_collapsed_px = 0
        self._nav_rail_expanded_px = 0
        self._header_scroll_initialized = False
        self._density_update_timer = QtCore.QTimer(self)
        self._density_update_timer.setSingleShot(True)
        self._density_update_timer.setInterval(self._default_debounce_ms)
        self._density_update_timer.timeout.connect(self._run_density_update)
        self._doctor_refresh_timer = QtCore.QTimer(self)
        self._doctor_refresh_timer.setSingleShot(True)
        self._doctor_refresh_timer.setInterval(self._default_debounce_ms)
        self._doctor_refresh_timer.timeout.connect(self._run_project_doctor_silent_if_initialized)

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
        self.project_panel.setMinimumWidth(self._sidebar_content_min_width())
        self.page_stack = QtWidgets.QStackedWidget()
        self.project_scroll = QtWidgets.QScrollArea()
        self.project_scroll.setWidgetResizable(True)
        self.project_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.project_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.project_scroll.setWidget(self.project_panel)
        self.nav_rail = None
        self.drawer_shell = QtWidgets.QWidget()
        self.drawer_shell.setObjectName("DrawerShell")
        drawer_layout = QtWidgets.QHBoxLayout(self.drawer_shell)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)
        self.project_scroll.setMinimumWidth(self._sidebar_content_min_width())
        self.project_scroll.setMaximumWidth(self._sidebar_content_max_width())
        drawer_layout.addWidget(self.project_scroll, 1)
        self.drawer_shell.installEventFilter(self)
        self._set_nav_rail_expanded(True)
        for widget in (self.drawer_shell, self.page_stack):
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        self.main_split.addWidget(self.drawer_shell)
        self.main_split.addWidget(self.page_stack)
        self.main_split.splitterMoved.connect(self._on_main_splitter_moved)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setCollapsible(0, False)
        self.main_split.setCollapsible(1, False)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([350, 980])
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
        self._apply_scale()
        self._apply_initial_layout()
        QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(120, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(320, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(0, self._deferred_sidebar_reflow)
        QtCore.QTimer.singleShot(120, self._deferred_sidebar_reflow)
        self._set_active_step(0)
        self._set_run_ui_state(False)
        try:
            self._wizard_snapshot = self._wizard_state()
        except Exception:
            self._wizard_snapshot = None

    def _apply_ui_style(self, scale: float = 1.0) -> None:
        compact = self._ui_density == "compact"
        base_grid = 6 if compact else 8
        font_base = max(14, int((14 if compact else 15) * scale))
        font_title = max(font_base + 1, int((17 if compact else 18) * scale))
        font_label = max(13, int((13 if compact else 14) * scale))
        font_helper = max(12, int((12 if compact else 13) * scale))
        font_section = max(13, int((13 if compact else 14) * scale))
        font_badge = max(12, int((12 if compact else 13) * scale))
        font_table_header = max(12, int((12 if compact else 13) * scale))
        pad = max(6, int(base_grid * scale))
        pad_sm = max(4, int((base_grid - 2) * scale))
        panel_pad = max(10, int((12 if compact else 16) * scale))
        radius = max(6, int(7 * scale))
        button_pad_y = max(5, int((5 if compact else 7) * scale))
        button_pad_x = max(12, int((12 if compact else 16) * scale))
        button_min_h = max(34, int((34 if compact else 38) * scale))
        tooltip_pad = max(6, int(6 * scale))
        handle = max(6, int(6 * scale))
        toolbar_height = max(40, int((40 if compact else 44) * scale))
        toolbar_height_inner = max(24, toolbar_height - max(14, int(16 * scale)))
        toolbar_radius = max(6, int(6 * scale))
        segmented_radius = max(toolbar_radius, int(10 * scale))
        toolbar_pad_x = max(8, int(8 * scale))
        toolbar_pad_y = max(6, int(6 * scale))
        toolbar_border = "rgba(148, 163, 184, 0.22)" if getattr(self, "_theme_mode", "light") == "dark" else "rgba(148, 163, 184, 0.32)"
        scrollbar = max(10, int(10 * scale))
        scrollbar_margin = max(2, int(2 * scale))
        scrollbar_radius = max(4, int(4 * scale))
        tokens = self._get_theme_tokens()
        combo_drop = max(22, int(22 * scale))
        chevron_size = max(10, int(10 * scale))
        input_min_h = max(36, int((36 if compact else 40) * scale))
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
                font-family: "Inter", "SF Pro Text", "Noto Sans", "DejaVu Sans";
                font-size: {font_base}px;
                background: {base};
            }}
            #RootContainer {{
                background: {base};
            }}
            #HeaderBar {{
                background: {surface};
                border-bottom: 1px solid {grid};
            }}
            #HeaderScroll {{
                background: transparent;
                border: none;
            }}
            #HeaderContent {{
                background: transparent;
            }}
            #AppTitle {{
                font-size: {font_title}px;
                font-weight: 700;
                color: {text};
            }}
            QFrame#HeaderNavGroup,
            QFrame#HeaderActionGroup,
            QFrame#HeaderSettingsGroup {{
                background: {panel};
                border: 1px solid {border};
                border-radius: {radius}px;
            }}
            QLabel#HeaderFieldLabel {{
                color: {text_muted};
                font-size: {font_helper}px;
                font-weight: 600;
            }}
            #NavRail {{
                background: {surface};
                border-right: 1px solid {grid};
            }}
            #DrawerShell {{
                background: {surface};
                border-right: 1px solid {grid};
            }}
            QToolButton#RailStepButton {{
                border: 1px solid transparent;
                border-radius: {radius}px;
                padding: {pad_sm}px {pad}px;
                text-align: left;
                color: {text_muted};
                background: transparent;
            }}
            QToolButton#RailStepButton:hover {{
                background: {panel};
                color: {text};
            }}
            QToolButton#RailStepButton:checked {{
                background: {accent_soft};
                border-color: {accent};
                color: {text};
            }}
            QToolButton#RailIconButton {{
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: {pad_sm}px;
                background: {panel};
            }}
            QToolButton#StepperButton {{
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: {pad_sm}px {pad}px;
                color: {text_muted};
                background: transparent;
                font-weight: 600;
            }}
            QToolButton#StepperButton:hover {{
                color: {text};
                background: {surface};
            }}
            QToolButton#StepperButton:checked {{
                border-color: {accent};
                color: {text};
                background: {accent_soft};
            }}
            QFrame#FeatureToolbar {{
                background: {panel};
                border: 1px solid {toolbar_border};
                border-radius: {toolbar_radius}px;
                min-height: {toolbar_height}px;
                max-height: {toolbar_height}px;
                padding: 2px {pad_sm}px;
            }}
            QScrollArea#FeatureToolbarScroll {{
                background: transparent;
                border: none;
            }}
            QScrollArea#FeatureToolbarScroll > QWidget > QWidget {{
                background: transparent;
            }}
            QWidget#FeatureToolbarRow {{
                background: transparent;
            }}
            QToolButton#FeatureToolbarButton {{
                border: 1px solid transparent;
                border-radius: {toolbar_radius}px;
                padding: 0px {toolbar_pad_x}px;
                color: {text_muted};
                background: transparent;
                font-size: {font_label}px;
                font-weight: 600;
                font-style: normal;
                min-height: {toolbar_height_inner}px;
                max-height: {toolbar_height_inner}px;
                margin: 0px;
            }}
            QToolButton#FeatureToolbarButton:hover {{
                background: {accent_soft};
                border-color: {accent};
                color: {text};
            }}
            QToolButton#FeatureToolbarButton:checked {{
                border-color: {accent};
                background: {accent};
                color: {selection_text};
            }}
            QToolButton#FeatureToolbarButton:pressed {{
                background: {accent};
                color: {selection_text};
            }}
            QToolButton#FeatureToolbarButton:focus {{
                border-color: {accent};
            }}
            QFrame#FeatureToolbar[toolbarRole="explore"] QToolButton#FeatureToolbarButton,
            QFrame#FeatureToolbar[toolbarRole="soz-views"] QToolButton#FeatureToolbarButton {{
                border: 1px solid {border};
                border-radius: {segmented_radius}px;
                background: transparent;
                color: {text_muted};
            }}
            QFrame#FeatureToolbar[toolbarRole="explore"] QToolButton#FeatureToolbarButton:hover,
            QFrame#FeatureToolbar[toolbarRole="soz-views"] QToolButton#FeatureToolbarButton:hover {{
                background: {accent_soft};
                border-color: {accent};
                color: {text};
            }}
            QFrame#FeatureToolbar[toolbarRole="explore"] QToolButton#FeatureToolbarButton:checked,
            QFrame#FeatureToolbar[toolbarRole="soz-views"] QToolButton#FeatureToolbarButton:checked {{
                border-color: {accent};
                background: {accent};
                color: {selection_text};
            }}
            QToolButton#FeatureToolbarButton QLabel {{
                color: inherit;
                background: transparent;
            }}
            QToolButton#FeatureToolbarNavButton {{
                border: 1px solid {toolbar_border};
                border-radius: {toolbar_radius}px;
                min-width: {toolbar_height_inner}px;
                max-width: {toolbar_height_inner}px;
                min-height: {toolbar_height_inner}px;
                max-height: {toolbar_height_inner}px;
                padding: 0px;
                color: {text_muted};
                background: {surface};
                font-weight: 600;
            }}
            QToolButton#FeatureToolbarNavButton:hover {{
                border-color: {accent};
                background: {accent_soft};
                color: {text};
            }}
            QToolButton#FeatureToolbarNavButton:disabled {{
                color: {text_muted};
                border-color: {border};
                background: transparent;
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
            #ProjectPanel QLabel[role="form-label"] {{
                color: {text_muted};
                font-size: {font_label}px;
                font-weight: 500;
            }}
            #ProjectPanel QLabel[role="helper-text"] {{
                color: {text_muted};
                font-size: {font_helper}px;
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
            QFrame[role="card"] {{
                background: {surface};
                border: 1px solid {border};
                border-radius: {radius}px;
            }}
            QWidget[role="card-header"] {{
                background: transparent;
            }}
            QLabel[role="card-title"] {{
                color: {text_muted};
                font-size: {font_section}px;
                font-weight: 600;
                letter-spacing: 0.6px;
            }}
            QFrame[role="card-divider"] {{
                background: {grid};
                min-height: 1px;
                max-height: 1px;
            }}
            QLabel[role="path-pill"] {{
                background: {panel};
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: {pad_sm}px {pad}px;
                color: {text};
            }}
            QLabel[role="status-badge"] {{
                border-radius: 999px;
                padding: 3px {pad}px;
                font-size: {font_badge}px;
                font-weight: 600;
                border: 1px solid {border_hover};
                background: {button_bg};
                color: {text};
            }}
            QLabel[role="status-badge"][tone="success"] {{
                background: {success_soft};
                border-color: {success};
                color: {success_text};
            }}
            QLabel[role="status-badge"][tone="warning"] {{
                background: {warning_soft};
                border-color: {warning};
                color: {warning_text};
            }}
            QLabel[role="status-badge"][tone="error"] {{
                background: {error_soft};
                border-color: {error};
                color: {error_text};
            }}
            QFrame[role="status-card"] {{
                border: 1px solid {border};
                border-radius: {radius}px;
                background: {panel};
            }}
            QFrame[role="status-card"][tone="success"] {{
                border-color: {success};
                background: {success_soft};
            }}
            QFrame[role="status-card"][tone="warning"] {{
                border-color: {warning};
                background: {warning_soft};
            }}
            QFrame[role="status-card"][tone="error"] {{
                border-color: {error};
                background: {error_soft};
            }}
            QLabel[role="status-headline"] {{
                font-weight: 600;
                font-size: 14px;
                color: {text};
            }}
            QLabel[role="doctor-warnings"] {{
                color: {warning_text};
                background: {warning_soft};
                border: 1px solid {warning};
                border-radius: {radius}px;
                padding: {pad_sm}px {pad}px;
            }}
            QLabel[role="meta-caption"] {{
                color: {text_muted};
                font-size: {font_helper}px;
            }}
            QListWidget[role="doctor-findings"] {{
                border: 1px solid {border};
                border-radius: {radius}px;
                background: {panel};
                padding: {pad_sm}px;
            }}
            QListWidget[role="doctor-findings"]::item {{
                padding: {pad_sm}px {pad}px;
                border-radius: {radius}px;
            }}
            QListWidget[role="doctor-findings"]::item:selected {{
                background: {accent_soft};
                color: {text};
            }}
            QFrame[role="inline-group"] {{
                border: 1px solid {border};
                border-radius: {radius}px;
                background: {surface};
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
                border: 1px solid {border};
                selection-background-color: {accent_soft};
                selection-color: {text};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                background: transparent;
                color: {text};
                min-height: 26px;
                padding: {pad_sm}px {pad}px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: {accent_soft};
                color: {text};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background: {accent_soft};
                color: {text};
            }}
            QComboBox QAbstractItemView::item:disabled {{
                color: {text_muted};
            }}
            QMenu {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                padding: {pad_sm}px;
            }}
            QMenu::item {{
                background: transparent;
                color: {text};
                padding: {pad_sm}px {pad}px;
                border-radius: {radius}px;
            }}
            QMenu::item:hover {{
                background: {accent_soft};
                color: {text};
            }}
            QMenu::item:selected {{
                background: {accent_soft};
                color: {text};
            }}
            QMenu::item:disabled {{
                color: {text_muted};
            }}
            QMenu::separator {{
                background: {border};
                height: 1px;
                margin: {pad_sm}px {pad}px;
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
                min-height: {input_min_h}px;
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
            QCheckBox {{
                spacing: {pad_sm}px;
                color: {text};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {border_hover};
                border-radius: 4px;
                background: {surface};
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid {accent};
                background: {accent};
            }}
            QCheckBox[variant="pill"] {{
                border: 1px solid {border};
                border-radius: 999px;
                padding: 4px 10px;
                background: {panel};
            }}
            QCheckBox[variant="pill"]::indicator {{
                width: 0px;
                height: 0px;
                border: none;
            }}
            QCheckBox[variant="pill"]:hover {{
                background: {button_hover};
            }}
            QCheckBox[variant="pill"]:checked {{
                border-color: {accent};
                background: {accent_soft};
                color: {text};
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background: {border};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {surface};
                border: 1px solid {accent};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {accent_soft};
            }}
            QPushButton {{
                background: {button_bg};
                color: {text};
                border: 1px solid {border};
                padding: {button_pad_y}px {button_pad_x}px;
                border-radius: {radius}px;
                font-size: {font_label}px;
                font-weight: 500;
                min-height: {button_min_h}px;
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
            QTableView, QTableWidget {{
                background: {surface};
                border: 1px solid {border};
                border-radius: {radius}px;
                gridline-color: transparent;
                alternate-background-color: {panel};
                selection-background-color: {accent_soft};
                selection-color: {text};
            }}
            QTableView::item, QTableWidget::item {{
                border-bottom: 1px solid {grid};
                padding: {pad_sm}px {pad}px;
            }}
            QTableView::item:hover, QTableWidget::item:hover {{
                background: {panel};
            }}
            QTableView::item:selected, QTableWidget::item:selected {{
                background: {accent_soft};
                color: {text};
            }}
            QTableCornerButton::section {{
                background: {panel};
                border: none;
            }}
            QLabel[role="section"] {{
                color: {text_muted};
                font-size: {font_section}px;
                font-weight: 600;
                padding-top: {pad_sm}px;
            }}
            QFrame[role="divider"] {{
                background: {grid};
                min-height: 1px;
                max-height: 1px;
            }}
            QLabel[role="insight-chip"] {{
                border: 1px solid {border};
                border-radius: {radius}px;
                background: {panel};
                color: {text_muted};
                padding: {pad_sm}px {pad}px;
            }}
            QHeaderView::section {{
                background: {panel};
                padding: {pad_sm}px {pad}px;
                border: none;
                border-bottom: 1px solid {grid};
                font-weight: 600;
                font-size: {font_table_header}px;
                color: {text_muted};
            }}
            QHeaderView::section:vertical {{
                color: {text_muted};
                font-size: {font_table_header}px;
                padding: {pad_sm}px;
                border-right: 1px solid {grid};
            }}
            QTabWidget::pane {{
                border: 1px solid {border};
                border-radius: {radius}px;
                background: {surface};
                top: -1px;
            }}
            QTabWidget::tab-bar {{
                left: {pad}px;
            }}
            QTabBar {{
                background: {panel};
                border-bottom: 1px solid {border};
                padding-left: {pad_sm}px;
            }}
            QTabBar::tab {{
                background: {tab_bg};
                border: 1px solid {border};
                border-bottom-color: {border};
                padding: {pad_sm}px {pad}px;
                margin-right: 4px;
                margin-top: 4px;
                border-top-left-radius: {radius}px;
                border-top-right-radius: {radius}px;
                color: {text_muted};
            }}
            QTabBar::tab:hover {{
                background: {panel};
                color: {text};
            }}
            QTabBar::tab:selected {{
                background: {surface};
                color: {text};
                border-color: {accent};
                border-bottom-color: {surface};
                margin-top: 0px;
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
            font_base=font_base,
            font_title=font_title,
            font_label=font_label,
            font_helper=font_helper,
            font_section=font_section,
            font_badge=font_badge,
            font_table_header=font_table_header,
            radius=radius,
            button_pad_y=button_pad_y,
            button_pad_x=button_pad_x,
            button_min_h=button_min_h,
            pad_sm=pad_sm,
            pad=pad,
            tooltip_pad=tooltip_pad,
            handle=handle,
            toolbar_height=toolbar_height,
            toolbar_height_inner=toolbar_height_inner,
            toolbar_radius=toolbar_radius,
            segmented_radius=segmented_radius,
            toolbar_pad_x=toolbar_pad_x,
            toolbar_pad_y=toolbar_pad_y,
            toolbar_border=toolbar_border,
            scrollbar=scrollbar,
            scrollbar_margin=scrollbar_margin,
            scrollbar_radius=scrollbar_radius,
            combo_drop=combo_drop,
            combo_arrow=combo_arrow,
            chevron_size=chevron_size,
            input_min_h=input_min_h,
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
            success=tokens["success"],
            warning=tokens["warning"],
            error=tokens["error"],
            success_soft=tokens["success_soft"],
            warning_soft=tokens["warning_soft"],
            error_soft=tokens["error_soft"],
            success_text=tokens["success_text"],
            warning_text=tokens["warning_text"],
            error_text=tokens["error_text"],
        )
        self.setStyleSheet(style)

    def _build_nav_rail(self) -> QtWidgets.QWidget:
        rail = QtWidgets.QFrame()
        rail.setObjectName("NavRail")
        rail_layout = QtWidgets.QVBoxLayout(rail)
        rail_layout.setContentsMargins(8, 8, 8, 8)
        rail_layout.setSpacing(6)

        self.nav_pin_btn = QtWidgets.QToolButton()
        self.nav_pin_btn.setCheckable(True)
        self.nav_pin_btn.setChecked(self._nav_rail_pinned)
        self.nav_pin_btn.setToolTip("Pin navigation drawer")
        self.nav_pin_btn.toggled.connect(self._toggle_nav_rail_pin)
        rail_layout.addWidget(self.nav_pin_btn)

        self.nav_step_group = QtWidgets.QButtonGroup(self)
        self.nav_step_group.setExclusive(True)
        self.nav_step_buttons = []
        step_icons = [
            QtWidgets.QStyle.StandardPixmap.SP_DirHomeIcon,
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView,
            QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton,
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView,
            QtWidgets.QStyle.StandardPixmap.SP_DriveFDIcon,
        ]
        for idx, (name, icon_enum) in enumerate(
            zip(["Project", "Define", "QC", "Explore", "Export"], step_icons)
        ):
            btn = QtWidgets.QToolButton()
            btn.setObjectName("RailStepButton")
            btn.setCheckable(True)
            btn.setText(name)
            btn.setIcon(self.style().standardIcon(icon_enum))
            btn.setToolTip(name)
            btn.clicked.connect(lambda _=False, i=idx: self._set_active_step(i))
            self.nav_step_group.addButton(btn, idx)
            self.nav_step_buttons.append(btn)
            rail_layout.addWidget(btn)

        rail_layout.addStretch(1)

        quick_row = QtWidgets.QHBoxLayout()
        quick_row.setSpacing(4)
        self.nav_load_btn = QtWidgets.QToolButton()
        self.nav_load_btn.setObjectName("RailIconButton")
        self.nav_load_btn.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        self.nav_load_btn.setToolTip("Load project")
        self.nav_load_btn.clicked.connect(self._load_project)
        quick_row.addWidget(self.nav_load_btn)

        self.nav_run_btn = QtWidgets.QToolButton()
        self.nav_run_btn.setObjectName("RailIconButton")
        self.nav_run_btn.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.nav_run_btn.setToolTip("Run analysis")
        self.nav_run_btn.clicked.connect(self._run_analysis)
        quick_row.addWidget(self.nav_run_btn)
        rail_layout.addLayout(quick_row)
        return rail

    def _toggle_nav_rail_pin(self, enabled: bool) -> None:
        self._nav_rail_pinned = bool(enabled)
        self.settings.setValue("nav_rail_pinned", self._nav_rail_pinned)
        self._set_nav_rail_expanded(self._nav_rail_pinned, animate=True)

    def _sidebar_content_min_width(self) -> int:
        scale = float(getattr(self, "_ui_scale", 1.0))
        return max(self._sidebar_content_min_px, int(self._sidebar_content_min_px * max(1.0, scale)))

    def _sidebar_content_preferred_width(self) -> int:
        scale = float(getattr(self, "_ui_scale", 1.0))
        preferred = max(
            self._sidebar_content_min_px,
            int(self._sidebar_content_pref_px * max(1.0, scale)),
        )
        return max(self._sidebar_content_min_width(), preferred)

    def _sidebar_content_max_width(self) -> int:
        scale = float(getattr(self, "_ui_scale", 1.0))
        max_width = max(
            self._sidebar_content_pref_px,
            int(self._sidebar_content_max_px * max(1.0, scale)),
        )
        return max(self._sidebar_content_preferred_width(), max_width)

    def _nav_rail_width(self, expanded: bool) -> int:
        scale = float(getattr(self, "_ui_scale", 1.0))
        base = self._nav_rail_expanded_px if expanded else self._nav_rail_collapsed_px
        min_px = self._nav_rail_expanded_px if expanded else self._nav_rail_collapsed_px
        return max(min_px, int(base * max(1.0, scale)))

    def _sidebar_total_min_width(self, expanded: bool) -> int:
        rail = self._nav_rail_width(expanded)
        content = self._sidebar_content_min_width() if expanded else 0
        return rail + content

    def _restored_drawer_width(self, expanded: bool) -> int:
        if not expanded:
            return self._sidebar_total_min_width(False)
        saved = self.settings.value("drawer_width", None)
        try:
            width = int(float(saved))
        except Exception:
            width = int(getattr(self, "_last_drawer_size", 0) or 0)
        if width <= 0:
            if expanded:
                width = self._nav_rail_width(True) + self._sidebar_content_preferred_width()
            else:
                width = self._sidebar_total_min_width(False)
        return max(width, self._sidebar_total_min_width(expanded))

    def _clamp_drawer_width(self, proposed: int, expanded: bool, inspector_visible: bool) -> int:
        scale = float(getattr(self, "_ui_scale", 1.0))
        avail = 0
        if hasattr(self, "main_split"):
            avail = self.main_split.width()
        if avail <= 0:
            avail = self.width()
        center_min = int(420 * scale)
        right_min = int(240 * scale) if inspector_visible else 0
        min_width = self._sidebar_total_min_width(expanded)
        drawer_cap = self._nav_rail_width(expanded) + (
            self._sidebar_content_max_width() if expanded else 0
        )
        max_width = proposed
        if avail > 0:
            max_width = max(min_width, avail - center_min - right_min)
        max_width = min(max_width, drawer_cap)
        return int(max(min_width, min(proposed, max_width)))

    def _set_nav_rail_expanded(self, expanded: bool, animate: bool = False) -> None:
        if not hasattr(self, "project_scroll"):
            return
        if hasattr(self, "project_panel"):
            self.project_panel.setMinimumWidth(self._sidebar_content_min_width())
        if not hasattr(self, "nav_rail") or self.nav_rail is None:
            self._nav_rail_expanded = True
            self.project_scroll.setVisible(True)
            self.project_scroll.setMinimumWidth(self._sidebar_content_min_width())
            self.project_scroll.setMaximumWidth(self._sidebar_content_max_width())
            self._update_splitter_constraints()
            return
        expanded = bool(expanded)
        self._nav_rail_expanded = expanded
        scale = float(getattr(self, "_ui_scale", 1.0))
        rail_width = self._nav_rail_width(expanded)
        self.nav_rail.setFixedWidth(rail_width)
        button_style = (
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            if expanded
            else QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        pin_text = "Unpin" if self._nav_rail_pinned else "Pin"
        self.nav_pin_btn.setText(pin_text)
        self.nav_pin_btn.setToolButtonStyle(button_style)
        self.nav_pin_btn.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarUnshadeButton)
        )
        button_min_width = max(0, rail_width - int(16 * scale)) if expanded else 0
        self.nav_pin_btn.setMinimumWidth(button_min_width)
        for btn in getattr(self, "nav_step_buttons", []):
            btn.setToolButtonStyle(button_style)
            btn.setMinimumWidth(button_min_width)

        target_width = self._sidebar_content_preferred_width() if expanded else 0
        if expanded:
            self.project_scroll.setVisible(True)
            self.project_scroll.setMaximumWidth(max(1, self.project_scroll.maximumWidth()))
        if self._nav_rail_animation is not None:
            try:
                self._nav_rail_animation.stop()
            except Exception:
                pass
        if animate:
            anim = QtCore.QPropertyAnimation(self.project_scroll, b"maximumWidth", self)
            anim.setDuration(self._default_debounce_ms)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
            start_value = self.project_scroll.maximumWidth()
            if start_value <= 0:
                start_value = self.project_scroll.width()
            anim.setStartValue(start_value)
            anim.setEndValue(target_width)

            def _finish() -> None:
                if expanded:
                    self.project_scroll.setMinimumWidth(self._sidebar_content_min_width())
                    self.project_scroll.setMaximumWidth(self._sidebar_content_max_width())
                else:
                    self.project_scroll.setMinimumWidth(0)
                    self.project_scroll.setMaximumWidth(0)
                    self.project_scroll.setVisible(False)
                self._update_splitter_constraints()

            anim.finished.connect(_finish)
            self._nav_rail_animation = anim
            anim.start()
            return

        if expanded:
            self.project_scroll.setVisible(True)
            self.project_scroll.setMinimumWidth(self._sidebar_content_min_width())
            self.project_scroll.setMaximumWidth(self._sidebar_content_max_width())
        else:
            self.project_scroll.setMinimumWidth(0)
            self.project_scroll.setMaximumWidth(0)
            self.project_scroll.setVisible(False)
        self._update_splitter_constraints()

    def _on_main_splitter_moved(self, _pos: int, _index: int) -> None:
        if not hasattr(self, "main_split") or not hasattr(self, "drawer_shell"):
            return
        if not self.drawer_shell.isVisible():
            return
        buttons = QtWidgets.QApplication.mouseButtons()
        if not (buttons & QtCore.Qt.MouseButton.LeftButton):
            return
        sizes = self.main_split.sizes()
        if len(sizes) < 2:
            return
        drawer_width = int(sizes[0])
        if drawer_width <= 0:
            return
        self._suppress_nav_hover_until_ms = QtCore.QDateTime.currentMSecsSinceEpoch() + 300
        self._normalize_main_splitter_sizes()
        sizes = self.main_split.sizes()
        if len(sizes) >= 2:
            drawer_width = int(sizes[0])
            if drawer_width > 0:
                self._last_drawer_size = drawer_width
                self.settings.setValue("drawer_width", drawer_width)

    def _deferred_sidebar_reflow(self) -> None:
        if not hasattr(self, "main_split") or not hasattr(self, "drawer_shell"):
            return
        if not self.drawer_shell.isVisible():
            return
        self._normalize_main_splitter_sizes()
        self._update_splitter_constraints()

    def _normalize_main_splitter_sizes(self) -> None:
        if not hasattr(self, "main_split") or not hasattr(self, "drawer_shell"):
            return
        if not self.drawer_shell.isVisible():
            return
        sizes = self.main_split.sizes()
        if len(sizes) < 2:
            return
        scale = float(getattr(self, "_ui_scale", 1.0))
        center_min = int(420 * scale)
        total = sum(sizes) if sum(sizes) > 0 else max(self.main_split.width(), self.width())
        if total <= 0:
            return
        expanded = bool(getattr(self, "_nav_rail_expanded", False))
        saved_width = self._restored_drawer_width(expanded)
        current_left = int(sizes[0]) if int(sizes[0]) > 0 else saved_width
        left_min = self._sidebar_total_min_width(expanded)
        left_max = self._clamp_drawer_width(total, expanded=expanded, inspector_visible=False)
        left_max = max(left_min, min(left_max, max(left_min, total - center_min)))
        clamped_left = max(left_min, min(current_left, left_max))
        center = max(center_min, total - clamped_left)
        if clamped_left == int(sizes[0]) and center == int(sizes[1]):
            return
        self.main_split.blockSignals(True)
        self.main_split.setSizes([clamped_left, center])
        self.main_split.blockSignals(False)

    def _add_section_header(self, layout: QtWidgets.QLayout, text: str) -> None:
        label = QtWidgets.QLabel(text)
        label.setProperty("role", "section")
        divider = QtWidgets.QFrame()
        divider.setProperty("role", "divider")
        layout.addWidget(label)
        layout.addWidget(divider)

    def _build_card(
        self,
        title: str,
        *,
        collapsible: bool = False,
        default_open: bool = True,
        role: str = "card",
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout, QtWidgets.QWidget, QtWidgets.QToolButton | None]:
        scale = float(getattr(self, "_ui_scale", 1.0))
        card = QtWidgets.QFrame()
        card.setProperty("role", role)
        outer = QtWidgets.QVBoxLayout(card)
        outer.setContentsMargins(max(10, int(14 * scale)), max(8, int(10 * scale)), max(10, int(14 * scale)), max(8, int(10 * scale)))
        outer.setSpacing(max(6, int(8 * scale)))

        header = QtWidgets.QWidget()
        header.setProperty("role", "card-header")
        header_row = QtWidgets.QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(max(6, int(8 * scale)))
        title_label = QtWidgets.QLabel(title.upper())
        title_label.setProperty("role", "card-title")
        header_row.addWidget(title_label)
        header_row.addStretch(1)
        toggle_btn = None
        if collapsible:
            toggle_btn = QtWidgets.QToolButton()
            toggle_btn.setCheckable(True)
            toggle_btn.setChecked(default_open)
            toggle_btn.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if default_open else QtCore.Qt.ArrowType.RightArrow
            )
            toggle_btn.setAutoRaise(False)
            toggle_btn.setToolTip(f"Show or hide {title}")
            header_row.addWidget(toggle_btn)
        outer.addWidget(header)

        divider = QtWidgets.QFrame()
        divider.setProperty("role", "card-divider")
        outer.addWidget(divider)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(max(8, int(10 * scale)))
        outer.addWidget(body)

        if toggle_btn is not None:
            toggle_btn.toggled.connect(
                lambda checked, b=body, t=toggle_btn: self._set_card_open_state(b, t, checked)
            )
            self._set_card_open_state(body, toggle_btn, default_open)
        return card, body_layout, body, toggle_btn

    def _set_card_open_state(
        self,
        body: QtWidgets.QWidget,
        toggle_btn: QtWidgets.QToolButton,
        is_open: bool,
    ) -> None:
        body.setVisible(bool(is_open))
        toggle_btn.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if is_open else QtCore.Qt.ArrowType.RightArrow
        )

    def _fit_combo_width(self, combo: QtWidgets.QComboBox) -> None:
        if combo is None or combo.property("wideControl"):
            return
        scale = float(getattr(self, "_ui_scale", 1.0))
        min_px_raw = combo.property("compactMinPx")
        max_px_raw = combo.property("compactMaxPx")
        min_px = int(min_px_raw) if min_px_raw is not None else 96
        max_px = int(max_px_raw) if max_px_raw is not None else 320
        min_px = max(72, int(min_px * max(1.0, scale)))
        max_px = max(min_px, int(max_px * max(1.0, scale)))
        base_max_raw = combo.property("compactBaseMaxPx")
        auto_applied = bool(combo.property("compactAutoFitApplied"))
        if base_max_raw is None:
            if not auto_applied:
                explicit_max = combo.maximumWidth()
                if explicit_max > 0 and explicit_max < 100000:
                    combo.setProperty("compactBaseMaxPx", int(explicit_max))
                    max_px = min(max_px, explicit_max)
        else:
            try:
                max_px = min(max_px, int(base_max_raw))
            except Exception:
                pass
        fm = combo.fontMetrics()
        texts: list[str] = []
        for idx in range(combo.count()):
            txt = combo.itemText(idx).strip()
            if txt:
                texts.append(txt)
        current = combo.currentText().strip()
        if current:
            texts.append(current)
        if combo.isEditable():
            edit = combo.lineEdit()
            if edit is not None:
                placeholder = edit.placeholderText().strip()
                if placeholder:
                    texts.append(placeholder)
        longest = max((fm.horizontalAdvance(text) for text in texts), default=0)
        chrome = max(36, int(52 * max(1.0, scale)))
        target = max(min_px, min(max_px, longest + chrome))
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        combo.setMinimumWidth(target)
        combo.setMaximumWidth(target)
        combo.setProperty("compactAutoFitApplied", True)
        try:
            view = combo.view()
            if view is not None:
                view.setMinimumWidth(target)
        except Exception:
            pass

    def _apply_compact_control_sizing(self) -> None:
        scale = float(getattr(self, "_ui_scale", 1.0))
        for combo in self.findChildren(QtWidgets.QComboBox):
            if combo.property("wideControl"):
                continue
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            if not bool(combo.property("compactAutoFitHooked")):
                combo.currentTextChanged.connect(
                    lambda _text, c=combo: self._fit_combo_width(c)
                )
                model = combo.model()
                if model is not None:
                    model.rowsInserted.connect(lambda *_args, c=combo: self._fit_combo_width(c))
                    model.rowsRemoved.connect(lambda *_args, c=combo: self._fit_combo_width(c))
                    model.modelReset.connect(lambda c=combo: self._fit_combo_width(c))
                    model.dataChanged.connect(lambda *_args, c=combo: self._fit_combo_width(c))
                combo.setProperty("compactAutoFitHooked", True)
            self._fit_combo_width(combo)
        spin_cap = max(112, int(142 * max(1.0, scale)))
        for spin_type in (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox):
            for spin in self.findChildren(spin_type):
                if spin.property("wideControl"):
                    continue
                if spin.maximumWidth() >= 100000:
                    spin.setMaximumWidth(spin_cap)
                if spin.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding:
                    spin.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Maximum,
                        QtWidgets.QSizePolicy.Policy.Fixed,
                    )
        text_buttons = (
            getattr(self, "topology_change_btn", None),
            getattr(self, "trajectory_change_btn", None),
            getattr(self, "trajectory_clear_btn", None),
            getattr(self, "topology_copy_btn", None),
            getattr(self, "trajectory_copy_btn", None),
            getattr(self, "output_dir_browse", None),
        )
        for btn in text_buttons:
            if not isinstance(btn, QtWidgets.QAbstractButton):
                continue
            text = (btn.text() or "").strip()
            if not text:
                continue
            pad = int((24 if isinstance(btn, QtWidgets.QPushButton) else 20) * max(1.0, scale))
            target_w = max(
                btn.minimumWidth(),
                btn.minimumSizeHint().width(),
                btn.fontMetrics().horizontalAdvance(text) + pad,
            )
            btn.setMinimumWidth(target_w)

    def _apply_form_layout_defaults(self, form: QtWidgets.QFormLayout, label_px: int = 120) -> None:
        scale = float(getattr(self, "_ui_scale", 1.0))
        form.setHorizontalSpacing(max(10, int(12 * scale)))
        form.setVerticalSpacing(max(8, int(10 * scale)))
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        label_w = max(label_px, int(label_px * scale))
        for row in range(form.rowCount()):
            item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            if item is None:
                continue
            label = item.widget()
            if isinstance(label, QtWidgets.QLabel):
                label.setProperty("role", "form-label")
                label.setMinimumWidth(label_w)

    def _set_status_badge(self, widget: QtWidgets.QLabel, text: str, tone: str = "neutral") -> None:
        widget.setText(text)
        widget.setProperty("tone", tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_status_card_tone(self, widget: QtWidgets.QWidget, tone: str) -> None:
        widget.setProperty("tone", tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _copy_to_clipboard(self, text: str, ok_message: str = "Copied to clipboard.") -> None:
        payload = (text or "").strip()
        if not payload:
            self._toast("Nothing to copy.", level="warning")
            return
        QtWidgets.QApplication.clipboard().setText(payload)
        self._toast(ok_message)

    def _path_from_label(self, label: QtWidgets.QLabel) -> str:
        if not isinstance(label, QtWidgets.QLabel):
            return ""
        tip = (label.toolTip() or "").strip()
        text = (label.text() or "").strip()
        return tip if tip and tip != text else text

    def _setup_modern_table(self, table: QtWidgets.QTableWidget) -> None:
        scale = float(getattr(self, "_ui_scale", 1.0))
        table.setProperty("role", "modern-table")
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(True)
        table.verticalHeader().setDefaultSectionSize(max(34, int(38 * scale)))
        table.verticalHeader().setMinimumWidth(max(24, int(32 * scale)))
        table.horizontalHeader().setMinimumHeight(max(26, int(30 * scale)))
        table.horizontalHeader().setStretchLastSection(True)

    def _build_insight_strip(self, key: str, chips: int = 4) -> QtWidgets.QWidget:
        container = QtWidgets.QFrame()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        labels: list[QtWidgets.QLabel] = []
        for _ in range(chips):
            label = QtWidgets.QLabel("...")
            label.setProperty("role", "insight-chip")
            labels.append(label)
            row.addWidget(label)
        row.addStretch(1)
        self._plot_insight_labels[key] = labels
        return container

    def _set_plot_insights(self, key: str, values: Iterable[str]) -> None:
        values_list = [str(v) for v in values if str(v).strip()]
        self._plot_insights[key] = values_list
        labels = self._plot_insight_labels.get(key, [])
        for idx, label in enumerate(labels):
            if idx < len(values_list):
                label.setText(values_list[idx])
                label.setVisible(True)
            else:
                label.setVisible(False)

    def _build_feature_toolbar(
        self,
        labels: list[str],
        *,
        object_name: str,
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QButtonGroup, list[QtWidgets.QToolButton]]:
        scale = float(getattr(self, "_ui_scale", 1.0))
        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("FeatureToolbar")
        toolbar.setProperty("toolbarRole", object_name)
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(max(6, int(8 * scale)), 4, max(6, int(8 * scale)), 4)
        toolbar_layout.setSpacing(max(4, int(4 * scale)))

        left_nav = QtWidgets.QToolButton()
        left_nav.setObjectName("FeatureToolbarNavButton")
        left_nav.setText("◀")
        left_nav.setToolTip("Scroll left")
        left_nav.setAutoRaise(False)
        left_nav.setVisible(False)
        toolbar_layout.addWidget(left_nav, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("FeatureToolbarScroll")
        scroll.setWidgetResizable(False)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        row = QtWidgets.QWidget()
        row.setObjectName("FeatureToolbarRow")
        row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(max(4, int(6 * scale)))

        group = QtWidgets.QButtonGroup(toolbar)
        group.setExclusive(True)
        buttons: list[QtWidgets.QToolButton] = []

        for idx, label in enumerate(labels):
            button = QtWidgets.QToolButton()
            button.setObjectName("FeatureToolbarButton")
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setText(label)
            button.setProperty("fullLabel", label)
            button.setToolTip(label)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.installEventFilter(self)
            group.addButton(button, idx)
            buttons.append(button)
            row_layout.addWidget(button, 0)

        scroll.setWidget(row)
        toolbar_layout.addWidget(scroll, 1)

        right_nav = QtWidgets.QToolButton()
        right_nav.setObjectName("FeatureToolbarNavButton")
        right_nav.setText("▶")
        right_nav.setToolTip("Scroll right")
        right_nav.setAutoRaise(False)
        right_nav.setVisible(False)
        toolbar_layout.addWidget(right_nav, 0)

        hbar = scroll.horizontalScrollBar()

        def _scroll_toolbar(direction: int) -> None:
            step = max(80, int(scroll.viewport().width() * 0.7))
            hbar.setValue(hbar.value() + (step * direction))

        def _sync_toolbar_nav() -> None:
            max_v = hbar.maximum()
            min_v = hbar.minimum()
            has_overflow = max_v > min_v
            left_nav.setVisible(has_overflow)
            right_nav.setVisible(has_overflow)
            left_nav.setEnabled(has_overflow and hbar.value() > min_v)
            right_nav.setEnabled(has_overflow and hbar.value() < max_v)

        def _ensure_toolbar_button_visible(btn: QtWidgets.QToolButton | None) -> None:
            if btn is None:
                return
            current = hbar.value()
            viewport_w = scroll.viewport().width()
            left = btn.x()
            right_edge = left + btn.width()
            target = current
            if left < current:
                target = left
            elif right_edge > current + viewport_w:
                target = right_edge - viewport_w
            if target != current:
                hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), target)))

        def _refresh_toolbar_buttons() -> None:
            local_scale = float(getattr(self, "_ui_scale", 1.0))
            min_button_w = max(96, int(104 * local_scale))
            horizontal_padding = max(40, int(44 * local_scale))
            for btn in buttons:
                full_label = str(btn.property("fullLabel") or btn.text() or "").strip()
                if not full_label:
                    continue
                if btn.text() != full_label:
                    btn.setText(full_label)
                btn.setToolTip(full_label)
                fm = QtGui.QFontMetrics(btn.font())
                target_w = max(min_button_w, fm.horizontalAdvance(full_label) + horizontal_padding)
                btn.setFixedWidth(target_w)
            row.adjustSize()
            _ensure_toolbar_button_visible(group.checkedButton())
            QtCore.QTimer.singleShot(0, _sync_toolbar_nav)

        left_nav.clicked.connect(lambda: _scroll_toolbar(-1))
        right_nav.clicked.connect(lambda: _scroll_toolbar(1))
        group.idClicked.connect(
            lambda idx: _ensure_toolbar_button_visible(buttons[idx] if 0 <= idx < len(buttons) else None)
        )
        hbar.rangeChanged.connect(lambda *_: _sync_toolbar_nav())
        hbar.valueChanged.connect(lambda *_: _sync_toolbar_nav())
        QtCore.QTimer.singleShot(0, _refresh_toolbar_buttons)
        toolbar._sync_toolbar_nav = _sync_toolbar_nav  # keep callable alive for signal closures
        toolbar._refresh_toolbar_buttons = _refresh_toolbar_buttons

        for button in buttons:
            self._toolbar_button_groups[button] = (group, buttons)
        self._feature_toolbars.append(toolbar)
        return toolbar, group, buttons

    def _set_feature_toolbar_index(
        self,
        buttons: list[QtWidgets.QToolButton],
        index: int,
    ) -> None:
        if 0 <= index < len(buttons):
            button = buttons[index]
            if not button.isChecked():
                button.setChecked(True)
            scroll = None
            parent = button.parentWidget()
            while parent is not None:
                if (
                    isinstance(parent, QtWidgets.QScrollArea)
                    and parent.objectName() == "FeatureToolbarScroll"
                ):
                    scroll = parent
                    break
                parent = parent.parentWidget()
            if scroll is not None:
                hbar = scroll.horizontalScrollBar()
                current = hbar.value()
                viewport_w = scroll.viewport().width()
                left = button.x()
                right_edge = left + button.width()
                target = current
                if left < current:
                    target = left
                elif right_edge > current + viewport_w:
                    target = right_edge - viewport_w
                if target != current:
                    hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), target)))

    def _refresh_feature_toolbar_metrics(self) -> None:
        for toolbar in list(getattr(self, "_feature_toolbars", [])):
            if toolbar is None:
                continue
            refresh = getattr(toolbar, "_refresh_toolbar_buttons", None)
            if callable(refresh):
                refresh()

    def _refresh_header_metrics(self) -> None:
        if not hasattr(self, "header_bar"):
            return
        scale = float(getattr(self, "_ui_scale", 1.0))
        if hasattr(self, "header_layout"):
            self.header_layout.setContentsMargins(
                max(8, int(12 * scale)),
                max(6, int(8 * scale)),
                max(8, int(12 * scale)),
                max(6, int(8 * scale)),
            )
            self.header_layout.setSpacing(max(6, int(8 * scale)))
        if hasattr(self, "header_content_layout"):
            self.header_content_layout.setContentsMargins(0, 0, 0, 0)
            self.header_content_layout.setSpacing(max(6, int(8 * scale)))

        header_buttons = list(getattr(self, "stepper_buttons", []))
        header_buttons.extend(
            [
                getattr(self, "load_btn", None),
                getattr(self, "save_btn", None),
                getattr(self, "new_soz_btn", None),
                getattr(self, "quick_btn", None),
                getattr(self, "run_btn", None),
                getattr(self, "cancel_btn", None),
                getattr(self, "export_btn", None),
                getattr(self, "report_btn", None),
                getattr(self, "console_toggle_btn", None),
                getattr(self, "drawer_toggle_btn", None),
            ]
        )
        for btn in header_buttons:
            if not isinstance(btn, QtWidgets.QAbstractButton):
                continue
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            fm = btn.fontMetrics()
            text_w = fm.horizontalAdvance(btn.text()) if hasattr(btn, "text") else 0
            icon_w = 0
            try:
                if not btn.icon().isNull():
                    icon_w = btn.iconSize().width() + int(8 * scale)
            except Exception:
                icon_w = 0
            content_w = text_w + icon_w + int(22 * scale)
            hint_w = max(
                btn.sizeHint().width(),
                btn.minimumSizeHint().width(),
                max(int(64 * scale), content_w),
            )
            if hint_w > 0:
                btn.setMinimumWidth(hint_w)
                btn.setMaximumWidth(16777215)

        for lbl in (
            getattr(self, "theme_label", None),
            getattr(self, "density_label", None),
            getattr(self, "scale_label", None),
            getattr(self, "workers_header_label", None),
        ):
            if not isinstance(lbl, QtWidgets.QLabel):
                continue
            label_w = max(50, lbl.fontMetrics().horizontalAdvance(lbl.text()) + int(16 * scale))
            lbl.setMinimumWidth(label_w)
            lbl.setMaximumWidth(label_w)

        for combo in (
            getattr(self, "ui_theme_combo", None),
            getattr(self, "ui_density_combo", None),
            getattr(self, "ui_scale_combo", None),
        ):
            if not isinstance(combo, QtWidgets.QComboBox):
                continue
            combo.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._fit_combo_width(combo)
            target_w = max(combo.maximumWidth(), combo.sizeHint().width())
            combo.setMinimumWidth(target_w)
            combo.setMaximumWidth(target_w)

        workers_spin = getattr(self, "workers_spin", None)
        if isinstance(workers_spin, QtWidgets.QSpinBox):
            workers_spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            workers_w = max(workers_spin.sizeHint().width(), max(92, int(116 * scale)))
            workers_spin.setMinimumWidth(workers_w)
            workers_spin.setMaximumWidth(workers_w)

        if hasattr(self, "presentation_toggle"):
            self.presentation_toggle.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            toggle_w = max(
                self.presentation_toggle.sizeHint().width(),
                self.presentation_toggle.minimumSizeHint().width(),
            )
            self.presentation_toggle.setMinimumWidth(toggle_w)

        if hasattr(self, "header_content") and hasattr(self, "header_scroll"):
            self.header_content.adjustSize()
            target_w = max(self.header_content.sizeHint().width(), self.header_scroll.viewport().width())
            self.header_content.setMinimumWidth(target_w)
            hbar = self.header_scroll.horizontalScrollBar()
            content_h = max(
                self.header_content.sizeHint().height(),
                self.header_content.minimumSizeHint().height(),
            )
            hbar_h = 0
            if hbar is not None and hbar.maximum() > hbar.minimum():
                hbar_h = max(hbar.sizeHint().height(), hbar.minimumSizeHint().height())
            scroll_h = max(content_h + hbar_h, self.header_scroll.minimumSizeHint().height())
            self.header_scroll.setMinimumHeight(scroll_h)
            self.header_scroll.setMaximumHeight(scroll_h)
            if hasattr(self, "header_layout"):
                margins = self.header_layout.contentsMargins()
                needed_h = scroll_h + margins.top() + margins.bottom()
                if needed_h > self.header_bar.minimumHeight():
                    self.header_bar.setMinimumHeight(needed_h)
            if hbar is not None and not getattr(self, "_header_scroll_initialized", False):
                hbar.setValue(hbar.minimum())
                self._header_scroll_initialized = True

    def _toast(self, message: str, timeout_ms: int = 3500, level: str = "info") -> None:
        if not hasattr(self, "status_bar"):
            return
        prefix = ""
        if level == "error":
            prefix = "Error: "
        elif level == "warning":
            prefix = "Warning: "
        self.status_bar.showMessage(f"{prefix}{message}", timeout_ms)

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

        self.header_scroll = QtWidgets.QScrollArea()
        self.header_scroll.setObjectName("HeaderScroll")
        self.header_scroll.setWidgetResizable(False)
        self.header_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.header_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.header_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.header_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.header_content = QtWidgets.QWidget()
        self.header_content.setObjectName("HeaderContent")
        content_layout = QtWidgets.QHBoxLayout(self.header_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        self.header_content_layout = content_layout

        def _std_icon(name: str, fallback: QtWidgets.QStyle.StandardPixmap) -> QtGui.QIcon:
            pix = getattr(QtWidgets.QStyle.StandardPixmap, name, fallback)
            return self.style().standardIcon(pix)

        nav_group = QtWidgets.QFrame()
        nav_group.setObjectName("HeaderNavGroup")
        nav_layout = QtWidgets.QHBoxLayout(nav_group)
        nav_layout.setContentsMargins(6, 4, 6, 4)
        nav_layout.setSpacing(4)

        self.stepper_group = QtWidgets.QButtonGroup(self)
        self.stepper_group.setExclusive(True)
        self.stepper_buttons = []
        step_defs = [
            ("Project", _std_icon("SP_DirHomeIcon", QtWidgets.QStyle.StandardPixmap.SP_DirIcon)),
            ("Define", _std_icon("SP_FileDialogDetailedView", QtWidgets.QStyle.StandardPixmap.SP_FileIcon)),
            ("QC", _std_icon("SP_DialogApplyButton", QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton)),
            ("Explore", _std_icon("SP_FileDialogContentsView", QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)),
            ("Export", _std_icon("SP_DialogSaveButton", QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)),
        ]
        for idx, (name, icon) in enumerate(step_defs):
            btn = QtWidgets.QToolButton()
            btn.setObjectName("StepperButton")
            btn.setText(name)
            btn.setIcon(icon)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, i=idx: self._set_active_step(i))
            btn.setToolTip(f"Go to {name}")
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.stepper_group.addButton(btn, idx)
            self.stepper_buttons.append(btn)
            nav_layout.addWidget(btn)
        content_layout.addWidget(nav_group, 0)

        action_group = QtWidgets.QFrame()
        action_group.setObjectName("HeaderActionGroup")
        action_layout = QtWidgets.QHBoxLayout(action_group)
        action_layout.setContentsMargins(6, 4, 6, 4)
        action_layout.setSpacing(4)

        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setIcon(_std_icon("SP_DialogOpenButton", QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.load_btn.clicked.connect(self._load_project)
        self.load_btn.setToolTip("Load a project configuration (JSON/YAML).")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setIcon(_std_icon("SP_DialogSaveButton", QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_btn.clicked.connect(self._save_project)
        self.save_btn.setToolTip("Save the current project configuration.")
        self.new_soz_btn = QtWidgets.QPushButton("New SOZ")
        self.new_soz_btn.setIcon(_std_icon("SP_FileDialogNewFolder", QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
        self.new_soz_btn.clicked.connect(self._add_soz_from_builder)
        self.new_soz_btn.setToolTip("Create or update a SOZ definition from the Wizard.")
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setIcon(_std_icon("SP_MediaPlay", QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setToolTip("Run analysis with the current settings.")
        self.workers_header_label = QtWidgets.QLabel("CPU workers")
        self.workers_header_label.setObjectName("HeaderFieldLabel")
        self.workers_spin = QtWidgets.QSpinBox()
        max_workers = max(1, int(os.cpu_count() or 1))
        self.workers_spin.setRange(0, max_workers)
        self.workers_spin.setSpecialValueText("Auto")
        self.workers_spin.setToolTip(
            "CPU workers/threads for optional modules. 0 = auto (os.cpu_count()-1)."
        )
        self.workers_spin.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.workers_spin.setMaximumWidth(max(92, int(116 * float(getattr(self, "_ui_scale", 1.0)))))
        self.workers_spin.valueChanged.connect(self._on_analysis_settings_changed)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setIcon(_std_icon("SP_DialogCancelButton", QtWidgets.QStyle.StandardPixmap.SP_DialogCancelButton))
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        self.cancel_btn.setToolTip("Cancel the running analysis.")
        self.quick_btn = QtWidgets.QPushButton("Quick")
        self.quick_btn.setIcon(_std_icon("SP_MediaSeekForward", QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
        self.quick_btn.clicked.connect(self._quick_subset)
        self.quick_btn.setToolTip("Quick subset preview.")
        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.setIcon(_std_icon("SP_DriveHDIcon", QtWidgets.QStyle.StandardPixmap.SP_DriveHDIcon))
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setToolTip("Export CSV/JSON outputs to the output directory.")
        self.report_btn = QtWidgets.QPushButton("Export Report")
        self.report_btn.setIcon(_std_icon("SP_FileIcon", QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
        self.report_btn.clicked.connect(self._export_report)
        self.report_btn.setToolTip("Generate a report (HTML/Markdown).")
        self.console_toggle_btn = QtWidgets.QToolButton()
        self.console_toggle_btn.setText("Console")
        self.console_toggle_btn.setIcon(_std_icon("SP_MessageBoxInformation", QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.console_toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.console_toggle_btn.setCheckable(True)
        self.console_toggle_btn.toggled.connect(self._toggle_console)
        self.drawer_toggle_btn = QtWidgets.QToolButton()
        self.drawer_toggle_btn.setText("Drawer")
        self.drawer_toggle_btn.setIcon(_std_icon("SP_FileDialogListView", QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView))
        self.drawer_toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.drawer_toggle_btn.setCheckable(True)
        self.drawer_toggle_btn.setChecked(True)
        self.drawer_toggle_btn.toggled.connect(self._toggle_drawer)

        self.theme_label = QtWidgets.QLabel("Theme")
        self.theme_label.setObjectName("HeaderFieldLabel")
        self.ui_theme_combo = QtWidgets.QComboBox()
        self.ui_theme_combo.addItems(["Light", "Dark"])
        self.ui_theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.ui_theme_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ui_theme_combo.setMinimumContentsLength(4)
        self.ui_theme_combo.setProperty("compactMinPx", 90)
        self.ui_theme_combo.setProperty("compactMaxPx", 170)
        self.density_label = QtWidgets.QLabel("Density")
        self.density_label.setObjectName("HeaderFieldLabel")
        self.ui_density_combo = QtWidgets.QComboBox()
        self.ui_density_combo.addItems(["Comfortable", "Compact"])
        self.ui_density_combo.currentTextChanged.connect(self._on_density_changed)
        self.ui_density_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ui_density_combo.setMinimumContentsLength(11)
        self.ui_density_combo.setProperty("compactMinPx", 130)
        self.ui_density_combo.setProperty("compactMaxPx", 240)
        self.scale_label = QtWidgets.QLabel("Scale")
        self.scale_label.setObjectName("HeaderFieldLabel")
        self.ui_scale_combo = QtWidgets.QComboBox()
        self._scale_levels = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75]
        for level in self._scale_levels:
            self.ui_scale_combo.addItem(f"{int(level * 100)}%", level)
        self.ui_scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        self.ui_scale_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ui_scale_combo.setMinimumContentsLength(4)
        self.ui_scale_combo.setProperty("compactMinPx", 84)
        self.ui_scale_combo.setProperty("compactMaxPx", 150)
        self.ui_scale_combo.setToolTip("Scales the entire interface (fonts, buttons, spacing).")
        self.scale_label.setToolTip("Interface scale")
        self.ui_theme_combo.setToolTip("Switch between light and dark themes.")
        self.ui_density_combo.setToolTip("Choose UI density.")
        self.theme_label.setToolTip("Theme")
        self.presentation_toggle = QtWidgets.QCheckBox("Presentation")
        self.presentation_toggle.toggled.connect(self._set_presentation_mode)

        for btn in (
            self.load_btn,
            self.save_btn,
            self.new_soz_btn,
            self.quick_btn,
            self.run_btn,
            self.workers_header_label,
            self.workers_spin,
            self.cancel_btn,
            self.export_btn,
            self.report_btn,
        ):
            action_layout.addWidget(btn)
        content_layout.addWidget(action_group, 0)

        settings_group = QtWidgets.QFrame()
        settings_group.setObjectName("HeaderSettingsGroup")
        settings_layout = QtWidgets.QHBoxLayout(settings_group)
        settings_layout.setContentsMargins(6, 4, 6, 4)
        settings_layout.setSpacing(6)
        settings_layout.addWidget(self.console_toggle_btn)
        settings_layout.addWidget(self.drawer_toggle_btn)
        settings_layout.addWidget(self.theme_label)
        settings_layout.addWidget(self.ui_theme_combo)
        settings_layout.addWidget(self.density_label)
        settings_layout.addWidget(self.ui_density_combo)
        settings_layout.addWidget(self.scale_label)
        settings_layout.addWidget(self.ui_scale_combo)
        settings_layout.addWidget(self.presentation_toggle)
        content_layout.addWidget(settings_group, 0)

        self.header_scroll.setWidget(self.header_content)
        layout.addWidget(self.header_scroll, 1)

        self._sync_theme_combo()
        self._sync_density_combo()
        self._sync_scale_combo()
        QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
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
        if hasattr(self, "drawer_shell"):
            self.drawer_shell.setVisible(enabled)
            if hasattr(self, "main_split"):
                sizes = self.main_split.sizes()
                if enabled:
                    expanded = bool(getattr(self, "_nav_rail_pinned", False))
                    self._set_nav_rail_expanded(expanded, animate=False)
                    restored = self._restored_drawer_width(expanded)
                    restored = self._clamp_drawer_width(restored, expanded=expanded, inspector_visible=False)
                    center_size = max(sum(sizes) - restored, 200)
                    self.main_split.setSizes([restored, center_size])
                else:
                    if len(sizes) == 2:
                        self._last_drawer_size = sizes[0]
                        self.settings.setValue("drawer_width", self._last_drawer_size)
                        self.main_split.setSizes([0, sizes[1] + sizes[0]])
                    elif len(sizes) == 3:
                        self._last_drawer_size = sizes[0]
                        self.settings.setValue("drawer_width", self._last_drawer_size)
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
        expanded = bool(getattr(self, "_nav_rail_pinned", False))
        drawer_width = self._restored_drawer_width(expanded)
        drawer_width = self._clamp_drawer_width(
            drawer_width,
            expanded=expanded,
            inspector_visible=False,
        )
        center_width = int(900 * scale)
        if hasattr(self, "drawer_shell"):
            self.drawer_shell.setVisible(True)
        self._set_nav_rail_expanded(expanded, animate=False)
        if hasattr(self, "drawer_toggle_btn"):
            self.drawer_toggle_btn.blockSignals(True)
            self.drawer_toggle_btn.setChecked(True)
            self.drawer_toggle_btn.blockSignals(False)
        self.main_split.setSizes([drawer_width, center_width])
        self._last_drawer_size = drawer_width
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
        if hasattr(self, "inspector_stack"):
            self.inspector_stack.setCurrentIndex(index)
        if hasattr(self, "stepper_buttons") and index < len(self.stepper_buttons):
            self.stepper_buttons[index].setChecked(True)
        if hasattr(self, "nav_step_buttons") and index < len(self.nav_step_buttons):
            self.nav_step_buttons[index].setChecked(True)

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
        if obj in {getattr(self, "nav_rail", None), getattr(self, "drawer_shell", None)}:
            etype = event.type()
            now_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
            if (
                now_ms < int(getattr(self, "_suppress_nav_hover_until_ms", 0))
                or (QtWidgets.QApplication.mouseButtons() & QtCore.Qt.MouseButton.LeftButton)
            ):
                return super().eventFilter(obj, event)
            if etype == QtCore.QEvent.Type.Enter:
                if not self._nav_rail_pinned and getattr(self, "drawer_shell", None) and self.drawer_shell.isVisible():
                    self._set_nav_rail_expanded(True, animate=True)
            elif etype == QtCore.QEvent.Type.Leave:
                if not self._nav_rail_pinned and getattr(self, "drawer_shell", None) and self.drawer_shell.isVisible():
                    cursor_local = self.drawer_shell.mapFromGlobal(QtGui.QCursor.pos())
                    if not self.drawer_shell.rect().contains(cursor_local):
                        self._set_nav_rail_expanded(False, animate=True)
        if (
            obj is getattr(self, "analysis_window_fields_host", None)
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._layout_analysis_window_fields()
            return False
        if (
            obj is getattr(self, "_density_split_handle", None)
            and event.type() == QtCore.QEvent.Type.MouseButtonDblClick
        ):
            self._reset_density_split_to_half()
            return True
        if (
            event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(obj, QtWidgets.QToolButton)
            and obj in self._toolbar_button_groups
        ):
            key_event = event
            key = key_event.key()
            group, buttons = self._toolbar_button_groups[obj]
            if not buttons:
                return True
            current = group.checkedId()
            if current < 0:
                try:
                    current = buttons.index(obj)
                except ValueError:
                    current = 0

            target = None
            if key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Down):
                target = (current + 1) % len(buttons)
            elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Up):
                target = (current - 1) % len(buttons)
            elif key == QtCore.Qt.Key.Key_Home:
                target = 0
            elif key == QtCore.Qt.Key.Key_End:
                target = len(buttons) - 1
            elif key in (
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Enter,
                QtCore.Qt.Key.Key_Space,
            ):
                obj.click()
                return True

            if target is not None:
                buttons[target].setFocus(QtCore.Qt.FocusReason.TabFocusReason)
                buttons[target].click()
                return True
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
        if hasattr(self, "drawer_shell"):
            self.drawer_shell.setVisible(not enabled)
        if hasattr(self, "console_panel"):
            self.console_panel.setVisible(False)
            if hasattr(self, "console_toggle_btn"):
                self.console_toggle_btn.setChecked(False)
        if hasattr(self, "drawer_toggle_btn"):
            self.drawer_toggle_btn.setChecked(not enabled)

    def _apply_scale(self) -> None:
        density_factor = 0.9 if self._ui_density == "compact" else 1.0
        total = float(self._user_scale) * float(self._presentation_scale) * density_factor
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
        for attr, title in (
            ("distance_bridge_timeseries_plot", "Distance Bridge"),
            ("distance_bridge_residence_plot", "Residence summary"),
            ("distance_bridge_top_plot", "Top bridges"),
            ("hbond_bridge_timeseries_plot", "H-bond Water Bridge"),
            ("hbond_bridge_residence_plot", "Residence summary"),
            ("hbond_bridge_top_plot", "Top bridges"),
            ("bridge_compare_plot", "Bridge type comparator"),
            ("hbond_bridge_network_plot", "Bridge network"),
            ("hydration_frequency_plot", "Residue frequency profile"),
            ("hydration_top_plot", "Top residues"),
            ("hydration_timeline_plot", "Residue contact timeline"),
            ("water_sp_plot", "SP(τ)"),
            ("water_hbl_plot", "H-bond lifetime"),
            ("water_wor_plot", "Water orientational relaxation"),
        ):
            plot = getattr(self, attr, None)
            if plot is not None:
                self._style_plot(plot, title)
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
            "distance_bridge_table",
            "hbond_bridge_table",
            "distance_contact_table",
            "hbond_hydration_table",
            "density_table",
            "water_dynamics_table",
            "water_dynamics_summary",
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
            self.header_bar.setMinimumHeight(int(52 * total))
        toolbar_h = max(40, int((40 if self._ui_density == "compact" else 44) * total))
        for toolbar_attr in (
            "builder_toolbar",
            "explore_mode_toolbar",
            "qc_toolbar",
            "soz_explore_toolbar",
            "timeline_mode_toolbar",
            "plots_mode_toolbar",
            "density_view_toolbar",
        ):
            toolbar = getattr(self, toolbar_attr, None)
            if toolbar is None:
                continue
            toolbar.setMinimumHeight(toolbar_h)
            toolbar.setMaximumHeight(toolbar_h)
        self._refresh_feature_toolbar_metrics()
        self._apply_compact_control_sizing()
        self._refresh_header_metrics()
        if hasattr(self, "nav_rail"):
            self._set_nav_rail_expanded(
                self._nav_rail_pinned or self._nav_rail_expanded,
                animate=False,
            )
        if hasattr(self, "timeline_plot"):
            self._update_timeline_plot()
        if hasattr(self, "hist_plot"):
            self._update_hist_plot()
        if hasattr(self, "event_plot"):
            self._update_event_plot()
        self._update_splitter_constraints()

    def _sync_scale_combo(self) -> None:
        if not hasattr(self, "ui_scale_combo"):
            return
        idx = 0
        best = None
        for i, level in enumerate(self._scale_levels):
            if best is None or abs(level - self._user_scale) < abs(best - self._user_scale):
                best = level
                idx = i
        self.ui_scale_combo.blockSignals(True)
        self.ui_scale_combo.setCurrentIndex(idx)
        self.ui_scale_combo.blockSignals(False)

    def _sync_theme_combo(self) -> None:
        if not hasattr(self, "ui_theme_combo"):
            return
        self.ui_theme_combo.blockSignals(True)
        self.ui_theme_combo.setCurrentText("Dark" if self._theme_mode == "dark" else "Light")
        self.ui_theme_combo.blockSignals(False)

    def _sync_density_combo(self) -> None:
        if not hasattr(self, "ui_density_combo"):
            return
        self.ui_density_combo.blockSignals(True)
        self.ui_density_combo.setCurrentText(
            "Compact" if self._ui_density == "compact" else "Comfortable"
        )
        self.ui_density_combo.blockSignals(False)

    def _set_user_scale(self, scale: float) -> None:
        self._user_scale = float(scale)
        self.settings.setValue("ui_scale", self._user_scale)
        self._sync_scale_combo()
        self._apply_scale()

    def _on_scale_changed(self) -> None:
        if not hasattr(self, "ui_scale_combo"):
            return
        data = self.ui_scale_combo.currentData()
        try:
            scale = float(data)
        except Exception:
            scale = 1.0
        self._set_user_scale(scale)

    def _on_theme_changed(self) -> None:
        if not hasattr(self, "ui_theme_combo"):
            return
        mode = "dark" if self.ui_theme_combo.currentText().lower().startswith("dark") else "light"
        self._set_theme_mode(mode)

    def _on_density_changed(self) -> None:
        if not hasattr(self, "ui_density_combo"):
            return
        text = self.ui_density_combo.currentText().strip().lower()
        self._set_density_mode("compact" if text.startswith("compact") else "comfortable")

    def _set_theme_mode(self, mode: str) -> None:
        self._theme_mode = "dark" if str(mode).lower() == "dark" else "light"
        self.settings.setValue("ui_theme", self._theme_mode)
        self._sync_theme_combo()
        self._apply_plot_theme()
        self._apply_scale()

    def _set_density_mode(self, mode: str) -> None:
        self._ui_density = "compact" if str(mode).lower() == "compact" else "comfortable"
        self.settings.setValue("ui_density", self._ui_density)
        self._sync_density_combo()
        self._apply_scale()

    def _get_theme_tokens(self) -> dict[str, str]:
        if getattr(self, "_theme_mode", "light") == "dark":
            return {
                "base": "#0B1220",
                "surface": "#111827",
                "panel": "#0F172A",
                "text": "#E5E7EB",
                "text_muted": "#94A3B8",
                "border": "#334155",
                "border_hover": "#475569",
                "accent": "#3B82F6",
                "accent_soft": "#1E3A5F",
                "accent_alt": "#14B8A6",
                "selection": "#3B82F6",
                "selection_text": "#F8FAFC",
                "grid": "#1F2937",
                "plot_bg": "#0B1220",
                "plot_fg": "#CBD5E1",
                "plot_axis": "#64748B",
                "button_bg": "#1E293B",
                "button_hover": "#273449",
                "tab_bg": "#111827",
                "entry": "#34D399",
                "exit": "#F87171",
                "success": "#22C55E",
                "warning": "#F59E0B",
                "error": "#EF4444",
                "success_soft": "#052E1B",
                "warning_soft": "#3A2504",
                "error_soft": "#3F1111",
                "success_text": "#BBF7D0",
                "warning_text": "#FDE68A",
                "error_text": "#FECACA",
            }
        return {
            "base": "#F9FAFB",
            "surface": "#FFFFFF",
            "panel": "#F8FAFC",
            "text": "#111827",
            "text_muted": "#4B5563",
            "border": "#D1D5DB",
            "border_hover": "#9CA3AF",
            "accent": "#3B82F6",
            "accent_soft": "#DBEAFE",
            "accent_alt": "#0F766E",
            "selection": "#3B82F6",
            "selection_text": "#F8FAFC",
            "grid": "#E5E7EB",
            "plot_bg": "#FFFFFF",
            "plot_fg": "#111827",
            "plot_axis": "#64748B",
            "button_bg": "#F3F4F6",
            "button_hover": "#E5E7EB",
            "tab_bg": "#FFFFFF",
            "entry": "#059669",
            "exit": "#DC2626",
            "success": "#22C55E",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "success_soft": "#DCFCE7",
            "warning_soft": "#FEF3C7",
            "error_soft": "#FEE2E2",
            "success_text": "#166534",
            "warning_text": "#92400E",
            "error_text": "#991B1B",
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
        if hasattr(self, "timeline_help_label"):
            self.timeline_help_label.setText("Updating timeline…")
        QtCore.QTimer.singleShot(self._default_debounce_ms, self._run_timeline_update)

    def _run_timeline_update(self) -> None:
        self._timeline_update_pending = False
        self._update_timeline_plot()

    def _queue_hist_update(self) -> None:
        if self._hist_update_pending:
            return
        self._hist_update_pending = True
        if hasattr(self, "hist_info"):
            self.hist_info.setText("Updating histogram…")
        QtCore.QTimer.singleShot(self._default_debounce_ms, self._run_hist_update)

    def _run_hist_update(self) -> None:
        self._hist_update_pending = False
        self._update_hist_plot()

    def _queue_event_update(self) -> None:
        if self._event_update_pending:
            return
        self._event_update_pending = True
        if hasattr(self, "event_info"):
            self.event_info.setText("Updating event raster…")
        QtCore.QTimer.singleShot(self._default_debounce_ms, self._run_event_update)

    def _run_event_update(self) -> None:
        self._event_update_pending = False
        self._update_event_plot()

    def _queue_density_update(self) -> None:
        self._density_update_pending = True
        self._density_update_timer.start(self._default_debounce_ms)

    def _run_density_update(self) -> None:
        self._density_update_pending = False
        self._update_density_plots()

    def _set_density_hero_mode(self, index: int) -> None:
        if not hasattr(self, "density_splitter"):
            return
        mode = int(index)
        self._density_hero_mode = mode
        if mode == 0:  # 3D
            self.density_slice_card.setVisible(False)
            self.density_viewer_card.setVisible(True)
            self.density_splitter.setSizes([0, max(1, self.density_splitter.width())])
            if hasattr(self, "density_3d_widget") and self.density_3d_widget:
                try:
                    self.density_3d_widget.set_context_panel_visible(True)
                except Exception:
                    pass
            return
        if mode == 1:  # Plots
            self.density_slice_card.setVisible(True)
            self.density_viewer_card.setVisible(False)
            self.density_splitter.setSizes([max(1, self.density_splitter.width()), 0])
            return
        self.density_slice_card.setVisible(True)
        self.density_viewer_card.setVisible(True)
        self._reset_density_split_to_half()
        if hasattr(self, "density_3d_widget") and self.density_3d_widget:
            try:
                self.density_3d_widget.set_context_panel_visible(True)
            except Exception:
                pass

    def _reset_density_split_to_half(self) -> None:
        if not hasattr(self, "density_splitter"):
            return
        width = self.density_splitter.width()
        if width <= 0:
            width = self.width()
        half = max(1, int(width / 2))
        self.density_splitter.setSizes([half, half])

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
        QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
        self._update_splitter_constraints()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(120, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(320, self._refresh_header_metrics)
        QtCore.QTimer.singleShot(0, self._deferred_sidebar_reflow)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            if not self.isMinimized():
                QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
                QtCore.QTimer.singleShot(0, self._deferred_sidebar_reflow)
        else:
            tracked = {
                QtCore.QEvent.Type.FontChange,
                QtCore.QEvent.Type.ApplicationFontChange,
            }
            screen_change = getattr(QtCore.QEvent.Type, "ScreenChangeInternal", None)
            dpi_change = getattr(QtCore.QEvent.Type, "DpiChange", None)
            if screen_change is not None:
                tracked.add(screen_change)
            if dpi_change is not None:
                tracked.add(dpi_change)
            if event.type() not in tracked:
                return
            QtCore.QTimer.singleShot(0, self._refresh_header_metrics)
            QtCore.QTimer.singleShot(0, self._deferred_sidebar_reflow)

    def _update_splitter_constraints(self) -> None:
        scale = float(self._ui_scale) if hasattr(self, "_ui_scale") else 1.0
        # Main horizontal split (project | center)
        main_split = getattr(self, "main_split", None)
        if main_split and hasattr(self, "drawer_shell"):
            avail = main_split.width() or self.width()
            if avail <= 0:
                avail = self.width()
            expanded = bool(getattr(self, "_nav_rail_expanded", False))
            left_floor_pref = self._sidebar_total_min_width(expanded)
            center_min = int(420 * scale)
            drawer_visible = self.drawer_shell.isVisible()
            if avail > 0:
                drawer_cap = self._nav_rail_width(expanded) + (
                    self._sidebar_content_max_width() if expanded else 0
                )
                max_side = max(left_floor_pref, min(int(avail * 0.62), drawer_cap))
                left_min = left_floor_pref if drawer_visible else 0
                total_min = left_min + center_min
                if total_min > avail:
                    shrink = avail / max(total_min, 1)
                    rail_only = self._sidebar_total_min_width(False)
                    left_floor = rail_only if not expanded else min(
                        left_floor_pref,
                        max(rail_only, avail - center_min),
                    )
                    left_min = max(int(left_min * shrink), left_floor) if drawer_visible else 0
                    center_min = max(int(center_min * shrink), int(160 * scale))
                self.drawer_shell.setMinimumWidth(left_min if drawer_visible else 0)
                self.drawer_shell.setMaximumWidth(drawer_cap if drawer_visible else 0)
                self.page_stack.setMinimumWidth(center_min)
                sizes = main_split.sizes()
                if len(sizes) >= 2:
                    if (
                        sizes[0] < left_min
                        or sizes[0] > drawer_cap
                        or sizes[1] < center_min
                        or not drawer_visible
                    ):
                        left_size = self._clamp_drawer_width(
                            int(sizes[0]),
                            expanded=expanded,
                            inspector_visible=False,
                        ) if drawer_visible else 0
                        left_size = min(left_size, max_side)
                        center_size = max(avail - left_size, center_min)
                        main_split.setSizes([left_size, center_size])

        # Events split (plots | tables)
        lower_split = getattr(self, "lower_split", None)
        if lower_split and lower_split.count() >= 2:
            avail_w = lower_split.width()
            if avail_w > 0:
                left_min = int(460 * scale)
                right_min = int(300 * scale)
                if left_min + right_min > avail_w:
                    shrink = avail_w / max(left_min + right_min, 1)
                    left_min = max(int(left_min * shrink), int(240 * scale))
                    right_min = max(int(right_min * shrink), int(200 * scale))
                left_widget = lower_split.widget(0)
                right_widget = lower_split.widget(1)
                if left_widget is not None:
                    left_widget.setMinimumWidth(left_min)
                if right_widget is not None:
                    right_widget.setMinimumWidth(right_min)
                sizes = lower_split.sizes()
                if len(sizes) == 2:
                    if sizes[0] < left_min or sizes[1] < right_min:
                        left_size = max(left_min, min(int(sizes[0]), max(left_min, avail_w - right_min)))
                        right_size = max(right_min, avail_w - left_size)
                        lower_split.setSizes([left_size, right_size])

        # Export split (extract | report)
        export_split = getattr(self, "export_split", None)
        if export_split and export_split.count() >= 2:
            avail_w = export_split.width()
            if avail_w > 0:
                left_min = int(360 * scale)
                right_min = int(300 * scale)
                if left_min + right_min > avail_w:
                    shrink = avail_w / max(left_min + right_min, 1)
                    left_min = max(int(left_min * shrink), int(220 * scale))
                    right_min = max(int(right_min * shrink), int(200 * scale))
                left_widget = export_split.widget(0)
                right_widget = export_split.widget(1)
                if left_widget is not None:
                    left_widget.setMinimumWidth(left_min)
                if right_widget is not None:
                    right_widget.setMinimumWidth(right_min)
                sizes = export_split.sizes()
                if len(sizes) == 2:
                    if sizes[0] < left_min or sizes[1] < right_min:
                        left_size = max(left_min, min(int(sizes[0]), max(left_min, avail_w - right_min)))
                        right_size = max(right_min, avail_w - left_size)
                        export_split.setSizes([left_size, right_size])

        # Density split (slice explorer | 3D viewer) when in split mode
        density_splitter = getattr(self, "density_splitter", None)
        if density_splitter and density_splitter.count() >= 2:
            left_widget = density_splitter.widget(0)
            right_widget = density_splitter.widget(1)
            if (
                left_widget is not None
                and right_widget is not None
                and left_widget.isVisible()
                and right_widget.isVisible()
            ):
                avail_w = density_splitter.width()
                if avail_w > 0:
                    left_min = int(320 * scale)
                    right_min = int(340 * scale)
                    if left_min + right_min > avail_w:
                        shrink = avail_w / max(left_min + right_min, 1)
                        left_min = max(int(left_min * shrink), int(180 * scale))
                        right_min = max(int(right_min * shrink), int(200 * scale))
                    left_widget.setMinimumWidth(left_min)
                    right_widget.setMinimumWidth(right_min)
                    sizes = density_splitter.sizes()
                    if len(sizes) == 2 and (sizes[0] < left_min or sizes[1] < right_min):
                        left_size = max(left_min, min(int(sizes[0]), max(left_min, avail_w - right_min)))
                        right_size = max(right_min, avail_w - left_size)
                        density_splitter.setSizes([left_size, right_size])

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
        panel.setObjectName("ProjectPanel")
        scale = float(getattr(self, "_ui_scale", 1.0))
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(max(12, int(14 * scale)))

        inputs_card, inputs_body, _, _ = self._build_card("Project Inputs")
        layout.addWidget(inputs_card)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        status_label = QtWidgets.QLabel("Status")
        status_label.setProperty("role", "form-label")
        self.project_inputs_status = QtWidgets.QLabel("Not loaded")
        self.project_inputs_status.setProperty("role", "status-badge")
        self._set_status_badge(self.project_inputs_status, "Not loaded", "neutral")
        status_row.addWidget(status_label)
        status_row.addWidget(self.project_inputs_status)
        status_row.addStretch(1)
        inputs_body.addLayout(status_row)

        inputs_layout = QtWidgets.QGridLayout()
        inputs_layout.setHorizontalSpacing(max(8, int(10 * scale)))
        inputs_layout.setVerticalSpacing(max(8, int(10 * scale)))
        inputs_layout.setColumnStretch(1, 1)
        self.topology_label = QtWidgets.QLabel("No topology selected")
        self.trajectory_label = QtWidgets.QLabel("No trajectory selected")
        self.topology_change_btn = QtWidgets.QPushButton("Change")
        self.trajectory_change_btn = QtWidgets.QPushButton("Change")
        self.topology_clear_btn = QtWidgets.QPushButton("Clear")
        self.trajectory_clear_btn = QtWidgets.QPushButton("Clear")
        self.topology_copy_btn = QtWidgets.QPushButton("Copy")
        self.trajectory_copy_btn = QtWidgets.QPushButton("Copy")
        for btn in (self.topology_copy_btn, self.trajectory_copy_btn):
            btn.setToolTip("Copy full path")
        text_fit_extra = max(20, int(24 * scale))
        for btn in (
            self.topology_change_btn,
            self.topology_clear_btn,
            self.trajectory_change_btn,
            self.trajectory_clear_btn,
        ):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            width_hint = btn.fontMetrics().horizontalAdvance(btn.text()) + text_fit_extra
            button_width = max(max(68, int(72 * scale)), width_hint, btn.sizeHint().width())
            btn.setMinimumWidth(button_width)
        for btn in (self.topology_copy_btn, self.trajectory_copy_btn):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            copy_hint = btn.fontMetrics().horizontalAdvance(btn.text()) + max(18, int(22 * scale))
            copy_width = max(max(56, int(60 * scale)), copy_hint, btn.sizeHint().width())
            btn.setMinimumWidth(copy_width)
        self.topology_change_btn.clicked.connect(self._browse_topology)
        self.topology_clear_btn.clicked.connect(self._clear_topology)
        self.trajectory_change_btn.clicked.connect(self._browse_trajectory)
        self.trajectory_clear_btn.clicked.connect(self._clear_trajectory)
        self.topology_copy_btn.clicked.connect(
            lambda: self._copy_to_clipboard(
                self._path_from_label(self.topology_label),
                "Topology path copied.",
            )
        )
        self.trajectory_copy_btn.clicked.connect(
            lambda: self._copy_to_clipboard(
                self._path_from_label(self.trajectory_label),
                "Trajectory path copied.",
            )
        )

        mono = QtGui.QFont("JetBrains Mono", max(10, int(11 * float(getattr(self, "_ui_scale", 1.0)))))
        for lbl in (self.topology_label, self.trajectory_label):
            lbl.setFont(mono)
        for lbl in (self.topology_label, self.trajectory_label):
            lbl.setProperty("role", "path-pill")
            lbl.setWordWrap(False)
            lbl.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            lbl.setMinimumWidth(max(180, int(220 * scale)))
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        inputs_layout.addWidget(QtWidgets.QLabel("Topology"), 0, 0)
        inputs_layout.addWidget(self.topology_label, 0, 1)
        top_btns = QtWidgets.QHBoxLayout()
        top_btns.setContentsMargins(0, 0, 0, 0)
        top_btns.setSpacing(6)
        top_btns.addWidget(self.topology_change_btn)
        top_btns.addWidget(self.topology_clear_btn)
        top_btns.addWidget(self.topology_copy_btn)
        inputs_layout.addLayout(top_btns, 0, 2)

        inputs_layout.addWidget(QtWidgets.QLabel("Trajectory"), 1, 0)
        inputs_layout.addWidget(self.trajectory_label, 1, 1)
        traj_btns = QtWidgets.QHBoxLayout()
        traj_btns.setContentsMargins(0, 0, 0, 0)
        traj_btns.setSpacing(6)
        traj_btns.addWidget(self.trajectory_change_btn)
        traj_btns.addWidget(self.trajectory_clear_btn)
        traj_btns.addWidget(self.trajectory_copy_btn)
        inputs_layout.addLayout(traj_btns, 1, 2)
        col2_min = (
            self.trajectory_change_btn.minimumWidth()
            + self.trajectory_clear_btn.minimumWidth()
            + self.trajectory_copy_btn.minimumWidth()
            + traj_btns.spacing() * 2
        )
        inputs_layout.setColumnMinimumWidth(2, col2_min)
        inputs_body.addLayout(inputs_layout)

        run_card, run_body, _, _ = self._build_card("Run Configuration")
        layout.addWidget(run_card)

        analysis_card, analysis_body, _, _ = self._build_card(
            "Analysis Window",
            collapsible=True,
            default_open=True,
        )
        self.frame_start_spin = QtWidgets.QSpinBox()
        self.frame_start_spin.setRange(0, 10_000_000)
        self.frame_stop_spin = QtWidgets.QSpinBox()
        self.frame_stop_spin.setRange(-1, 10_000_000)
        self.frame_stop_spin.setSpecialValueText("End")
        self.frame_stride_spin = QtWidgets.QSpinBox()
        self.frame_stride_spin.setRange(1, 1_000_000)
        self.frame_stride_spin.setValue(1)
        compact_input_width = max(112, int(132 * scale))
        for spin in (
            self.frame_start_spin,
            self.frame_stop_spin,
            self.frame_stride_spin,
        ):
            spin.setMaximumWidth(compact_input_width)
            spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )

        def _compact_field(label_text: str, field: QtWidgets.QWidget) -> QtWidgets.QWidget:
            wrap = QtWidgets.QWidget()
            row_layout = QtWidgets.QVBoxLayout(wrap)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_label = QtWidgets.QLabel(label_text)
            row_label.setProperty("role", "form-label")
            row_layout.addWidget(row_label)
            row_layout.addWidget(field)
            return wrap

        self._analysis_window_field_widgets = [
            _compact_field("Frame start", self.frame_start_spin),
            _compact_field("Frame stop", self.frame_stop_spin),
            _compact_field("Stride", self.frame_stride_spin),
        ]
        self.analysis_window_fields_host = QtWidgets.QWidget()
        self.analysis_window_fields_layout = QtWidgets.QGridLayout(self.analysis_window_fields_host)
        self.analysis_window_fields_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_window_fields_layout.setHorizontalSpacing(max(8, int(10 * scale)))
        self.analysis_window_fields_layout.setVerticalSpacing(max(6, int(8 * scale)))
        for field_widget in self._analysis_window_field_widgets:
            field_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        self._analysis_window_field_columns = 0
        self.analysis_window_fields_host.installEventFilter(self)
        analysis_body.addWidget(self.analysis_window_fields_host)
        QtCore.QTimer.singleShot(0, self._layout_analysis_window_fields)
        run_body.addWidget(analysis_card)

        self.output_group, output_body, _, _ = self._build_card(
            "Output Settings",
            collapsible=True,
            default_open=True,
        )
        output_layout = QtWidgets.QFormLayout()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("results/")
        self.output_dir_browse = QtWidgets.QPushButton("Browse")
        browse_w = max(
            max(72, int(80 * scale)),
            self.output_dir_browse.fontMetrics().horizontalAdvance(self.output_dir_browse.text())
            + max(18, int(22 * scale)),
            self.output_dir_browse.sizeHint().width(),
        )
        self.output_dir_browse.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.output_dir_browse.setMinimumWidth(browse_w)
        self.output_dir_browse.clicked.connect(self._browse_output_dir)
        output_row_widget = QtWidgets.QWidget()
        output_row = QtWidgets.QHBoxLayout(output_row_widget)
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(6)
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.output_dir_browse, 0)
        self.report_format_combo = QtWidgets.QComboBox()
        self.report_format_combo.addItems(["html", "md"])
        self.write_per_frame_check = QtWidgets.QCheckBox("Write per-frame CSV")
        self.write_parquet_check = QtWidgets.QCheckBox("Write parquet")
        self.output_effective_label = None
        output_layout.addRow("Output dir", output_row_widget)
        output_layout.addRow("Report format", self.report_format_combo)
        flags_row = QtWidgets.QWidget()
        flags_layout = QtWidgets.QVBoxLayout(flags_row)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.setSpacing(max(6, int(8 * scale)))
        flags_layout.addWidget(self.write_per_frame_check)
        flags_layout.addWidget(self.write_parquet_check)
        for chk in (self.write_per_frame_check, self.write_parquet_check):
            chk.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            chk.setMinimumWidth(chk.sizeHint().width())
        output_layout.addRow("Outputs", flags_row)
        self._apply_form_layout_defaults(output_layout, label_px=120)
        self.report_format_combo.setMaximumWidth(max(120, int(140 * scale)))
        self.report_format_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        output_body.addLayout(output_layout)
        run_body.addWidget(self.output_group)

        self.report_format_combo.currentTextChanged.connect(self._on_output_settings_changed)
        self.write_per_frame_check.toggled.connect(self._on_output_settings_changed)
        self.write_parquet_check.toggled.connect(self._on_output_settings_changed)
        self.output_dir_edit.textChanged.connect(self._on_output_settings_changed)

        self.output_dir_edit.setToolTip("Base directory for analysis outputs and logs.")
        self.report_format_combo.setToolTip("Report format for Export Report (html or md).")
        self.write_per_frame_check.setToolTip("Write per-frame CSV outputs to disk.")
        self.write_parquet_check.setToolTip("Write parquet outputs where supported.")
        self.topology_change_btn.setToolTip("Select a different topology file.")
        self.trajectory_change_btn.setToolTip("Select a different trajectory file.")
        self.trajectory_clear_btn.setToolTip("Clear the trajectory path.")

        self.doctor_group, doctor_layout, _, _ = self._build_card("Project Doctor")
        doctor_top = QtWidgets.QHBoxLayout()
        doctor_top.setContentsMargins(0, 0, 0, 0)
        doctor_top.setSpacing(8)
        self.doctor_status_label = QtWidgets.QLabel("Not validated yet.")
        self.doctor_status_label.setProperty("role", "status-headline")
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

        self.doctor_status_card = QtWidgets.QFrame()
        self.doctor_status_card.setProperty("role", "status-card")
        self._set_status_card_tone(self.doctor_status_card, "neutral")
        doctor_status_layout = QtWidgets.QHBoxLayout(self.doctor_status_card)
        doctor_status_layout.setContentsMargins(10, 8, 10, 8)
        doctor_status_layout.setSpacing(6)
        doctor_status_layout.addWidget(self.doctor_status_label)
        doctor_status_layout.addStretch(1)
        doctor_layout.addWidget(self.doctor_status_card)

        doctor_badges = QtWidgets.QGridLayout()
        doctor_badges.setContentsMargins(0, 0, 0, 0)
        doctor_badges.setHorizontalSpacing(6)
        doctor_badges.setVerticalSpacing(6)
        self.doctor_errors_badge = QtWidgets.QLabel("Errors 0")
        self.doctor_warnings_badge = QtWidgets.QLabel("Warnings 0")
        self.doctor_solvent_badge = QtWidgets.QLabel("Solvent atoms -")
        self.doctor_probe_badge = QtWidgets.QLabel("Probe atoms -")
        self.doctor_frames_badge = QtWidgets.QLabel("Frames -")
        self.doctor_pbc_badge = QtWidgets.QLabel("PBC -")
        for badge in (
            self.doctor_errors_badge,
            self.doctor_warnings_badge,
            self.doctor_solvent_badge,
            self.doctor_probe_badge,
            self.doctor_frames_badge,
            self.doctor_pbc_badge,
        ):
            badge.setProperty("role", "status-badge")
            self._set_status_badge(badge, badge.text(), "neutral")
        doctor_badges.addWidget(self.doctor_errors_badge, 0, 0)
        doctor_badges.addWidget(self.doctor_warnings_badge, 0, 1)
        doctor_badges.addWidget(self.doctor_solvent_badge, 1, 0)
        doctor_badges.addWidget(self.doctor_probe_badge, 1, 1)
        doctor_badges.addWidget(self.doctor_frames_badge, 2, 0)
        doctor_badges.addWidget(self.doctor_pbc_badge, 2, 1)
        doctor_badges.setColumnStretch(2, 1)
        doctor_layout.addLayout(doctor_badges)

        findings_label = QtWidgets.QLabel("Actionable findings")
        findings_label.setProperty("role", "meta-caption")
        doctor_layout.addWidget(findings_label)
        self.doctor_findings_list = QtWidgets.QListWidget()
        self.doctor_findings_list.setProperty("role", "doctor-findings")
        self.doctor_findings_list.setMinimumHeight(max(84, int(100 * scale)))
        self.doctor_findings_list.setMaximumHeight(max(130, int(150 * scale)))
        self.doctor_findings_list.addItem("Run Project Doctor to generate actionable findings.")
        doctor_layout.addWidget(self.doctor_findings_list)

        details_toggle_row = QtWidgets.QHBoxLayout()
        details_toggle_row.setContentsMargins(0, 0, 0, 0)
        details_toggle_row.setSpacing(6)
        self.doctor_details_toggle = QtWidgets.QToolButton()
        self.doctor_details_toggle.setText("Diagnostics details")
        self.doctor_details_toggle.setCheckable(True)
        self.doctor_details_toggle.setChecked(False)
        self.doctor_details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.doctor_details_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        details_toggle_row.addWidget(self.doctor_details_toggle)
        details_toggle_row.addStretch(1)
        doctor_layout.addLayout(details_toggle_row)

        self.doctor_text = QtWidgets.QTextEdit()
        self.doctor_text.setReadOnly(True)
        self.doctor_text.setMinimumHeight(max(90, int(110 * scale)))
        self.doctor_text.setMaximumHeight(max(140, int(170 * scale)))
        self.doctor_text.setVisible(False)
        self.doctor_details_toggle.toggled.connect(self.doctor_text.setVisible)
        self.doctor_details_toggle.toggled.connect(
            lambda checked: self.doctor_details_toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )
        )
        doctor_layout.addWidget(self.doctor_text)
        layout.addWidget(self.doctor_group)

        selection_group, selection_layout, _, _ = self._build_card(
            "Selection Table",
            collapsible=True,
            default_open=False,
        )
        selection_caption = QtWidgets.QLabel("Selection checks and matching diagnostics")
        selection_caption.setProperty("role", "meta-caption")
        selection_layout.addWidget(selection_caption)
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
        self._setup_modern_table(self.doctor_seed_table)
        header = self.doctor_seed_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.doctor_seed_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.doctor_seed_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        selection_layout.addWidget(self.doctor_seed_table)
        layout.addWidget(selection_group)

        soz_group, soz_layout, _, _ = self._build_card(
            "Defined SOZs",
            collapsible=True,
            default_open=False,
        )
        soz_header_row = QtWidgets.QHBoxLayout()
        soz_header_row.setContentsMargins(0, 0, 0, 0)
        soz_header_row.setSpacing(6)
        self.soz_count_badge = QtWidgets.QLabel("SOZs 0")
        self.soz_count_badge.setProperty("role", "status-badge")
        self._set_status_badge(self.soz_count_badge, "SOZs 0", "neutral")
        soz_header_row.addWidget(self.soz_count_badge)
        soz_header_row.addStretch(1)
        soz_layout.addLayout(soz_header_row)
        self.soz_list = QtWidgets.QListWidget()
        soz_layout.addWidget(self.soz_list)
        self.soz_hint_label = QtWidgets.QLabel("")
        self.soz_hint_label.setWordWrap(True)
        self.soz_hint_label.setProperty("role", "meta-caption")
        self.soz_hint_label.setVisible(False)
        soz_layout.addWidget(self.soz_hint_label)
        layout.addWidget(soz_group)

        layout.addStretch(1)
        return panel

    def _layout_analysis_window_fields(self) -> None:
        host = getattr(self, "analysis_window_fields_host", None)
        grid = getattr(self, "analysis_window_fields_layout", None)
        fields = getattr(self, "_analysis_window_field_widgets", None)
        if host is None or grid is None or not fields:
            return

        margins = grid.contentsMargins()
        available_width = max(0, host.width() - margins.left() - margins.right())
        spacing = max(0, grid.horizontalSpacing())
        field_width = max(
            max(widget.sizeHint().width(), widget.minimumSizeHint().width()) for widget in fields
        )
        if available_width >= (field_width * 3 + spacing * 2):
            columns = 3
        elif available_width >= (field_width * 2 + spacing):
            columns = 2
        else:
            columns = 1

        if columns == getattr(self, "_analysis_window_field_columns", None) and grid.count() == len(fields):
            return
        self._analysis_window_field_columns = columns

        while grid.count():
            grid.takeAt(0)

        for index, widget in enumerate(fields):
            row = index // columns
            col = index % columns
            grid.addWidget(
                widget,
                row,
                col,
                alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            )

        for col in range(4):
            grid.setColumnStretch(col, 0)
        grid.setColumnStretch(columns, 1)

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
        scale = float(getattr(self, "_ui_scale", 1.0))
        layout.setSpacing(max(10, int(12 * scale)))

        summary_card, summary_body, _, _ = self._build_card("Project Summary")
        self.project_summary_label = QtWidgets.QLabel(
            "Load a project to view topology, trajectory, and output metadata. "
            "Use Project Doctor for frame count and PBC checks."
        )
        self.project_summary_label.setWordWrap(True)
        self.project_summary_label.setProperty("role", "helper-text")
        summary_body.addWidget(self.project_summary_label)
        layout.addWidget(summary_card)

        run_card, run_body, _, _ = self._build_card("Run Summary")
        self.overview_card = QtWidgets.QLabel("Run analysis to populate summary cards.")
        self.overview_card.setWordWrap(True)
        run_body.addWidget(self.overview_card)
        self.overview_raw_toggle = QtWidgets.QCheckBox("Show raw JSON")
        self.overview_raw_toggle.toggled.connect(self._toggle_overview_raw)
        run_body.addWidget(self.overview_raw_toggle)
        self.overview_text = QtWidgets.QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setVisible(False)
        self.overview_text.setMinimumHeight(max(150, int(180 * scale)))
        run_body.addWidget(self.overview_text)
        layout.addWidget(run_card)

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
        layout.setSpacing(10)
        (
            self.qc_toolbar,
            self.qc_toolbar_group,
            self.qc_toolbar_buttons,
        ) = self._build_feature_toolbar(
            ["QC Summary", "Diagnostics"],
            object_name="qc-view",
        )
        layout.addWidget(self.qc_toolbar, 0)
        self.qc_tabs = QtWidgets.QStackedWidget()
        qc_panel = QtWidgets.QWidget()
        qc_layout = QtWidgets.QVBoxLayout(qc_panel)
        qc_layout.setContentsMargins(0, 0, 0, 0)
        qc_layout.setSpacing(10)

        self.qc_health_card = QtWidgets.QFrame()
        self.qc_health_card.setProperty("role", "status-card")
        self._set_status_card_tone(self.qc_health_card, "neutral")
        health_layout = QtWidgets.QVBoxLayout(self.qc_health_card)
        health_layout.setContentsMargins(10, 8, 10, 8)
        health_layout.setSpacing(8)
        self.qc_health_headline = QtWidgets.QLabel("Project health unavailable")
        self.qc_health_headline.setProperty("role", "status-headline")
        self.qc_health_headline.setWordWrap(True)
        health_layout.addWidget(self.qc_health_headline)
        self.qc_health_detail = QtWidgets.QLabel("Run analysis to compute QC checks and diagnostics.")
        self.qc_health_detail.setWordWrap(True)
        health_layout.addWidget(self.qc_health_detail)
        chip_row = QtWidgets.QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(8)
        self.qc_preflight_badge = QtWidgets.QLabel("Preflight: -")
        self.qc_preflight_badge.setProperty("role", "status-badge")
        self.qc_warning_badge = QtWidgets.QLabel("Warnings: -")
        self.qc_warning_badge.setProperty("role", "status-badge")
        self.qc_analysis_badge = QtWidgets.QLabel("Analysis warnings: -")
        self.qc_analysis_badge.setProperty("role", "status-badge")
        self.qc_zero_badge = QtWidgets.QLabel("Zero SOZs: -")
        self.qc_zero_badge.setProperty("role", "status-badge")
        for badge in (
            self.qc_preflight_badge,
            self.qc_warning_badge,
            self.qc_analysis_badge,
            self.qc_zero_badge,
        ):
            self._set_status_badge(badge, badge.text(), "neutral")
            chip_row.addWidget(badge)
        chip_row.addStretch(1)
        health_layout.addLayout(chip_row)
        qc_layout.addWidget(self.qc_health_card)

        findings_frame = QtWidgets.QFrame()
        findings_frame.setProperty("role", "inline-group")
        findings_layout = QtWidgets.QVBoxLayout(findings_frame)
        findings_layout.setContentsMargins(10, 8, 10, 8)
        findings_layout.setSpacing(8)
        findings_title = QtWidgets.QLabel("Key Findings")
        findings_title.setProperty("role", "section")
        findings_layout.addWidget(findings_title)
        self.qc_summary_label = QtWidgets.QLabel("QC summary will appear after analysis.")
        self.qc_summary_label.setWordWrap(True)
        findings_layout.addWidget(self.qc_summary_label)
        self.qc_findings_text = QtWidgets.QTextEdit()
        self.qc_findings_text.setReadOnly(True)
        self.qc_findings_text.setMinimumHeight(120)
        self.qc_findings_text.setMaximumHeight(220)
        self.qc_findings_text.setPlaceholderText("Warnings and errors will be highlighted here.")
        findings_layout.addWidget(self.qc_findings_text)
        qc_layout.addWidget(findings_frame)

        self.qc_raw_toggle = QtWidgets.QCheckBox("Show raw QC JSON")
        self.qc_raw_toggle.toggled.connect(self._toggle_qc_raw)
        qc_layout.addWidget(self.qc_raw_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.qc_text = QtWidgets.QTextEdit()
        self.qc_text.setReadOnly(True)
        self.qc_text.setVisible(False)
        qc_layout.addWidget(self.qc_text)
        qc_layout.addStretch(1)

        self.qc_tabs.addWidget(qc_panel)
        self.qc_tabs.addWidget(self._build_logs_tab())
        self.qc_toolbar_group.idClicked.connect(self.qc_tabs.setCurrentIndex)
        self.qc_tabs.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.qc_toolbar_buttons, idx)
        )
        self._set_feature_toolbar_index(self.qc_toolbar_buttons, 0)
        layout.addWidget(self.qc_tabs, 1)
        self._update_qc_overview(None)
        return panel

    def _build_explore_page(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        explore_modes = [
            "SOZ Explorer",
            "Bridges",
            "Density",
        ]
        (
            self.explore_mode_toolbar,
            self.explore_mode_group,
            self.explore_mode_buttons,
        ) = self._build_feature_toolbar(
            explore_modes,
            object_name="explore",
        )
        layout.addWidget(self.explore_mode_toolbar, 0)

        self.explore_tabs = QtWidgets.QStackedWidget()
        self.explore_tabs.addWidget(self._build_soz_explore_tab())
        self.explore_tabs.addWidget(self._build_bridge_explore_tab())
        self.explore_tabs.addWidget(self._build_density_explore_tab())
        self.explore_mode_group.idClicked.connect(self.explore_tabs.setCurrentIndex)
        self.explore_tabs.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.explore_mode_buttons, idx)
        )
        self._set_feature_toolbar_index(self.explore_mode_buttons, 0)
        layout.addWidget(self.explore_tabs, 1)
        return panel

    def _build_soz_explore_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)
        section = QtWidgets.QLabel("SOZ Explorer Views")
        section.setProperty("role", "section")
        layout.addWidget(section)
        (
            self.soz_explore_toolbar,
            self.soz_explore_group,
            self.soz_explore_buttons,
        ) = self._build_feature_toolbar(
            ["Overview", "Events & Details"],
            object_name="soz-views",
        )
        layout.addWidget(self.soz_explore_toolbar, 0)

        self.soz_explore_stack = QtWidgets.QStackedWidget()
        self.soz_explore_subtabs = self.soz_explore_stack  # compatibility alias

        overview_tab = QtWidgets.QWidget()
        ov_layout = QtWidgets.QVBoxLayout(overview_tab)
        ov_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline_panel = self._build_timeline_tab()
        ov_layout.addWidget(self.timeline_panel)
        self.soz_explore_stack.addWidget(overview_tab)

        events_tab = QtWidgets.QWidget()
        ev_layout = QtWidgets.QVBoxLayout(events_tab)
        ev_layout.setContentsMargins(0, 0, 0, 0)

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
        self.lower_split.splitterMoved.connect(lambda *_: self._update_splitter_constraints())

        ev_layout.addWidget(self.lower_split)
        self.soz_explore_stack.addWidget(events_tab)
        self.soz_explore_group.idClicked.connect(self.soz_explore_stack.setCurrentIndex)
        self.soz_explore_stack.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.soz_explore_buttons, idx)
        )
        self._set_feature_toolbar_index(self.soz_explore_buttons, 0)

        layout.addWidget(self.soz_explore_stack, 1)
        return panel

    def _build_bridge_explore_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(self._build_distance_bridge_explore_panel(), 1)
        return panel

    def _build_distance_bridge_explore_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        self.distance_bridge_combo = QtWidgets.QComboBox()
        self.distance_bridge_combo.currentTextChanged.connect(self._update_distance_bridge_plots)
        self.distance_bridge_smooth_check = QtWidgets.QCheckBox("Smooth")
        self.distance_bridge_smooth_check.toggled.connect(self._update_distance_bridge_plots)
        self.distance_bridge_smooth_window = QtWidgets.QSpinBox()
        self.distance_bridge_smooth_window.setRange(1, 500)
        self.distance_bridge_smooth_window.setValue(5)
        self.distance_bridge_smooth_window.valueChanged.connect(self._update_distance_bridge_plots)
        self.distance_bridge_export_combo = QtWidgets.QComboBox()
        self.distance_bridge_export_combo.addItems(["Time series", "Residence", "Top bridges"])
        self.distance_bridge_export_btn = QtWidgets.QPushButton("Export Plot")
        self.distance_bridge_export_btn.clicked.connect(self._export_distance_bridge_plot)
        controls.addStretch(1)
        controls.addWidget(self.distance_bridge_export_btn)
        layout.addLayout(controls)
        layout.addWidget(self._build_insight_strip("bridge_distance"), 0)

        self.distance_bridge_timeseries_plot = pg.PlotWidget()
        self._style_plot(self.distance_bridge_timeseries_plot, "Distance Bridge")
        self.distance_bridge_timeseries_plot.setLabel("bottom", "Time", units="ns")
        self.distance_bridge_timeseries_plot.setLabel("left", "Bridging solvent residues")
        self.distance_bridge_timeseries_plot.setToolTip(
            "Distance Bridge (distance-cutoff intersection): number of bridging solvent residues per frame."
        )

        bottom_row = QtWidgets.QHBoxLayout()
        self.distance_bridge_residence_plot = pg.PlotWidget()
        self._style_plot(
            self.distance_bridge_residence_plot,
            "Distance Bridge (distance-cutoff intersection): Residence",
        )
        self.distance_bridge_residence_plot.setLabel("bottom", "Residence (time)")
        self.distance_bridge_residence_plot.setLabel("left", "Survival")
        self.distance_bridge_residence_plot.setToolTip(
            "Distance Bridge (distance-cutoff intersection): residence survival (continuous segments)."
        )
        self.distance_bridge_top_plot = pg.PlotWidget()
        self._style_plot(
            self.distance_bridge_top_plot,
            "Distance Bridge (distance-cutoff intersection): Top bridges",
        )
        self.distance_bridge_top_plot.setLabel("bottom", "Occupancy %")
        self.distance_bridge_top_plot.setLabel("left", "Bridge ID")
        self.distance_bridge_top_plot.setToolTip(
            "Distance Bridge (distance-cutoff intersection): top bridging solvent residues by occupancy."
        )
        bottom_row.addWidget(self.distance_bridge_residence_plot, 1)
        bottom_row.addWidget(self.distance_bridge_top_plot, 1)

        layout.addWidget(self.distance_bridge_timeseries_plot, 2)
        layout.addLayout(bottom_row, 1)
        return panel

    def _build_hbond_bridge_explore_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("H-bond Water Bridge"))
        self.hbond_bridge_combo = QtWidgets.QComboBox()
        self.hbond_bridge_combo.currentTextChanged.connect(self._update_hbond_bridge_plots)
        self.hbond_bridge_smooth_check = QtWidgets.QCheckBox("Smooth")
        self.hbond_bridge_smooth_check.toggled.connect(self._update_hbond_bridge_plots)
        self.hbond_bridge_smooth_window = QtWidgets.QSpinBox()
        self.hbond_bridge_smooth_window.setRange(1, 500)
        self.hbond_bridge_smooth_window.setValue(5)
        self.hbond_bridge_smooth_window.valueChanged.connect(self._update_hbond_bridge_plots)
        controls.addWidget(self.hbond_bridge_combo)
        controls.addWidget(self.hbond_bridge_smooth_check)
        controls.addWidget(QtWidgets.QLabel("Window"))
        controls.addWidget(self.hbond_bridge_smooth_window)
        self.hbond_bridge_export_combo = QtWidgets.QComboBox()
        self.hbond_bridge_export_combo.addItems(
            ["Time series", "Residence", "Top bridges", "Comparator", "Network"]
        )
        self.hbond_bridge_export_btn = QtWidgets.QPushButton("Export Plot")
        self.hbond_bridge_export_btn.clicked.connect(self._export_hbond_bridge_plot)
        controls.addWidget(self.hbond_bridge_export_combo)
        controls.addWidget(self.hbond_bridge_export_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.hbond_bridge_timeseries_plot = pg.PlotWidget()
        self._style_plot(self.hbond_bridge_timeseries_plot, "H-bond Water Bridge")
        self.hbond_bridge_timeseries_plot.setLabel("bottom", "Time", units="ns")
        self.hbond_bridge_timeseries_plot.setLabel("left", "# bridging waters")
        self.hbond_bridge_timeseries_plot.setToolTip(
            "H-bond Water Bridge: number of bridging waters per frame (MDAnalysis WaterBridgeAnalysis definition)."
        )

        self.bridge_compare_plot = pg.PlotWidget()
        self._style_plot(self.bridge_compare_plot, "Distance vs H-bond Bridge")
        self.bridge_compare_plot.setLabel("bottom", "Time", units="ns")
        self.bridge_compare_plot.setLabel("left", "# bridging waters")
        self.bridge_compare_plot.setToolTip(
            "Comparator: Distance Bridge vs H-bond Water Bridge time series."
        )

        # [Audit Fix] Add checkbox to toggle comparator overlay
        self.bridge_compare_check = QtWidgets.QCheckBox("Show Comparator")
        self.bridge_compare_check.setToolTip(
            "Overlay Distance Bridge time series onto H-bond Bridge plot for comparison."
        )
        self.bridge_compare_check.setChecked(False)  # Default off to prevent confusion
        self.bridge_compare_check.toggled.connect(self._update_hbond_bridge_plots)
        
        # Insert check into controls
        controls.insertWidget(controls.count() - 2, self.bridge_compare_check)
        layout.addWidget(self._build_insight_strip("bridge_hbond"), 0)

        bottom_row = QtWidgets.QHBoxLayout()
        self.hbond_bridge_residence_plot = pg.PlotWidget()
        self._style_plot(self.hbond_bridge_residence_plot, "H-bond Water Bridge: Residence")
        self.hbond_bridge_residence_plot.setLabel("bottom", "Residence (time)")
        self.hbond_bridge_residence_plot.setLabel("left", "Survival")
        self.hbond_bridge_residence_plot.setToolTip(
            "H-bond Water Bridge: residence survival (continuous segments)."
        )
        self.hbond_bridge_top_plot = pg.PlotWidget()
        self._style_plot(self.hbond_bridge_top_plot, "H-bond Water Bridge: Top bridges")
        self.hbond_bridge_top_plot.setLabel("bottom", "Occupancy %")
        self.hbond_bridge_top_plot.setLabel("left", "Bridge ID")
        self.hbond_bridge_top_plot.setToolTip(
            "H-bond Water Bridge: top bridging waters by occupancy."
        )
        bottom_row.addWidget(self.hbond_bridge_residence_plot, 1)
        bottom_row.addWidget(self.hbond_bridge_top_plot, 1)

        self.hbond_bridge_network_plot = pg.PlotWidget()
        self._style_plot(self.hbond_bridge_network_plot, "H-bond Water Bridge: Network")
        self.hbond_bridge_network_plot.setLabel("bottom", "Network")
        self.hbond_bridge_network_plot.setLabel("left", "")
        self.hbond_bridge_network_plot.setToolTip(
            "Optional network view: partner ↔ water ↔ partner edges for top bridges."
        )

        layout.addWidget(self.hbond_bridge_timeseries_plot, 2)
        layout.addWidget(self.bridge_compare_plot, 1)
        layout.addLayout(bottom_row, 1)
        layout.addWidget(self.hbond_bridge_network_plot, 1)
        return panel

    def _build_hbond_hydration_explore_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("H-bond Hydration"))
        self.hydration_config_combo = QtWidgets.QComboBox()
        self.hydration_config_combo.currentTextChanged.connect(self._update_hydration_plots)
        controls.addWidget(self.hydration_config_combo)
        controls.addWidget(QtWidgets.QLabel("Metric"))
        self.hydration_metric_combo = QtWidgets.QComboBox()
        self.hydration_metric_combo.addItems(["freq_total", "freq_given_soz"])
        self.hydration_metric_combo.currentTextChanged.connect(self._update_hydration_plots)
        controls.addWidget(self.hydration_metric_combo)
        self.hydration_export_combo = QtWidgets.QComboBox()
        self.hydration_export_combo.addItems(["Frequency", "Top residues", "Timeline"])
        self.hydration_export_btn = QtWidgets.QPushButton("Export Plot")
        self.hydration_export_btn.clicked.connect(self._export_hydration_plot)
        controls.addWidget(self.hydration_export_combo)
        controls.addWidget(self.hydration_export_btn)
        
        controls.addWidget(QtWidgets.QLabel("Colormap"))
        self.hydration_cmap_combo = QtWidgets.QComboBox()
        self.hydration_cmap_combo.addItems(["viridis", "magma", "plasma", "inferno", "cividis", "hot", "cool", "GnuPlot", "HLS"])
        self.hydration_cmap_combo.setCurrentText("viridis")
        self.hydration_cmap_combo.currentTextChanged.connect(self._update_hydration_plots)
        controls.addWidget(self.hydration_cmap_combo)

        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self._build_insight_strip("hydration"), 0)

        self.hydration_frequency_plot = pg.PlotWidget()
        self._style_plot(self.hydration_frequency_plot, "Residue frequency profile")
        self.hydration_frequency_plot.setLabel("bottom", "Residue ID")
        self.hydration_frequency_plot.setLabel("left", "Frequency")
        self.hydration_scatter = pg.ScatterPlotItem()
        self.hydration_frequency_plot.addItem(self.hydration_scatter)
        self.hydration_scatter.sigClicked.connect(self._on_hydration_point_clicked)

        self.hydration_top_plot = pg.PlotWidget()
        self._style_plot(self.hydration_top_plot, "Top residues")
        self.hydration_top_plot.setLabel("bottom", "Frequency")
        self.hydration_top_plot.setLabel("left", "Residue")

        self.hydration_timeline_plot = pg.PlotWidget()
        self._style_plot(self.hydration_timeline_plot, "Residue contact timeline")
        self.hydration_timeline_plot.setLabel("bottom", "Time", units="ns")
        self.hydration_timeline_plot.setLabel("left", "Contact")

        layout.addWidget(self.hydration_frequency_plot, 2)
        layout.addWidget(self.hydration_top_plot, 1)
        layout.addWidget(self.hydration_timeline_plot, 1)
        return panel

    def _build_density_explore_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        controls_frame = QtWidgets.QFrame()
        controls_frame.setObjectName("DensityExploreToolbar")
        controls_shell = QtWidgets.QVBoxLayout(controls_frame)
        controls_shell.setContentsMargins(10, 8, 10, 8)
        controls_shell.setSpacing(6)

        top_group = QtWidgets.QFrame()
        top_group.setProperty("role", "inline-group")
        top_row = QtWidgets.QHBoxLayout(top_group)
        top_row.setContentsMargins(8, 6, 8, 6)
        top_row.setSpacing(8)
        self.density_combo = QtWidgets.QComboBox()
        self.density_combo.currentTextChanged.connect(self._queue_density_update)
        self.density_combo.setProperty("compactMinPx", 150)
        self.density_combo.setProperty("compactMaxPx", 360)
        top_row.addWidget(QtWidgets.QLabel("View"), 0)
        self.density_explore_view_combo = QtWidgets.QComboBox()
        self.density_explore_view_combo.addItems(["physical", "relative", "score"])
        self.density_explore_view_combo.currentTextChanged.connect(self._queue_density_update)
        self.density_explore_view_combo.setMaximumWidth(150)
        top_row.addWidget(self.density_explore_view_combo, 0)
        top_row.addWidget(QtWidgets.QLabel("Colormap"), 0)
        self.density_cmap_combo = QtWidgets.QComboBox()
        self.density_cmap_combo.addItems(
            ["viridis", "magma", "plasma", "inferno", "cividis", "hot", "cool", "GnuPlot", "HLS"]
        )
        self.density_cmap_combo.setCurrentText("viridis")
        self.density_cmap_combo.currentTextChanged.connect(self._queue_density_update)
        self.density_cmap_combo.setMaximumWidth(190)
        top_row.addWidget(self.density_cmap_combo, 0)
        top_row.addStretch(1)
        controls_shell.addWidget(top_group)

        self.density_overlay_check = QtWidgets.QCheckBox("Overlay reference")
        self.density_overlay_check.toggled.connect(self._queue_density_update)
        self.density_export_btn = QtWidgets.QPushButton("Export Figure Pack")
        self.density_export_btn.clicked.connect(self._export_density_figure)

        slices_group = QtWidgets.QFrame()
        slices_group.setProperty("role", "inline-group")
        slices_row = QtWidgets.QHBoxLayout(slices_group)
        slices_row.setContentsMargins(8, 6, 8, 6)
        slices_row.setSpacing(8)
        slices_row.addWidget(QtWidgets.QLabel("Slices"), 0)
        self.density_slice_x = QtWidgets.QSpinBox()
        self.density_slice_y = QtWidgets.QSpinBox()
        self.density_slice_z = QtWidgets.QSpinBox()
        for spin in (self.density_slice_x, self.density_slice_y, self.density_slice_z):
            spin.setRange(0, 0)
            spin.valueChanged.connect(self._queue_density_update)
            spin.setMaximumWidth(92)
        slices_row.addWidget(QtWidgets.QLabel("X"), 0)
        slices_row.addWidget(self.density_slice_x, 0)
        slices_row.addWidget(QtWidgets.QLabel("Y"), 0)
        slices_row.addWidget(self.density_slice_y, 0)
        slices_row.addWidget(QtWidgets.QLabel("Z"), 0)
        slices_row.addWidget(self.density_slice_z, 0)
        self.density_center_btn = QtWidgets.QPushButton("Center Slices")
        self.density_center_btn.clicked.connect(self._center_density_slices)
        slices_row.addWidget(self.density_center_btn, 0)
        slices_row.addStretch(1)
        slices_row.addWidget(self.density_overlay_check, 0)
        slices_row.addWidget(self.density_export_btn, 0)
        controls_shell.addWidget(slices_group)
        layout.addWidget(controls_frame)
        layout.addWidget(self._build_insight_strip("density"), 0)

        (
            self.density_view_toolbar,
            self.density_view_group,
            self.density_view_buttons,
        ) = self._build_feature_toolbar(
            ["3D", "Plots", "Split"],
            object_name="density-view",
        )
        layout.addWidget(self.density_view_toolbar, 0)

        self.density_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.density_splitter.setChildrenCollapsible(False)
        self.density_splitter.setOpaqueResize(True)

        self.density_slice_card = QtWidgets.QGroupBox("Slice Explorer")
        slice_layout = QtWidgets.QVBoxLayout(self.density_slice_card)
        slice_layout.setContentsMargins(8, 8, 8, 8)
        self.density_layout = pg.GraphicsLayoutWidget()
        slice_layout.addWidget(self.density_layout)
        self.density_splitter.addWidget(self.density_slice_card)

        self.density_viewer_card = QtWidgets.QFrame()
        self.density_viewer_card.setObjectName("DensityViewerCard")
        viewer_layout = QtWidgets.QVBoxLayout(self.density_viewer_card)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        if Density3DWidget:
            self.density_3d_widget = Density3DWidget(self)
            try:
                self.density_3d_widget.sig_pick_event.connect(self._on_density_3d_pick_event)
            except Exception:
                pass
            viewer_layout.addWidget(self.density_3d_widget, 1)
            self.density_summary = getattr(self.density_3d_widget, "density_insights_text", None)
        else:
            unavailable = QtWidgets.QLabel("3D viewer unavailable on this setup.")
            unavailable.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            viewer_layout.addWidget(unavailable, 1)
            self.density_summary = QtWidgets.QTextEdit()
            self.density_summary.setReadOnly(True)
            self.density_summary.setPlaceholderText("Density map key findings will appear here.")
            self.density_summary.setMinimumHeight(110)
            self.density_summary.setMaximumHeight(170)
            viewer_layout.addWidget(self.density_summary, 0)

        self.density_splitter.addWidget(self.density_viewer_card)
        self.density_splitter.setCollapsible(0, False)
        self.density_splitter.setCollapsible(1, False)
        self.density_splitter.setStretchFactor(0, 1)
        self.density_splitter.setStretchFactor(1, 2)
        self.density_splitter.setSizes([0, 1])
        self.density_splitter.splitterMoved.connect(lambda *_: self._update_splitter_constraints())
        layout.addWidget(self.density_splitter, 1)

        self.density_view_group.idClicked.connect(self._set_density_hero_mode)
        self._set_feature_toolbar_index(self.density_view_buttons, 0)
        self._set_density_hero_mode(0)
        if self.density_splitter.count() > 1:
            self._density_split_handle = self.density_splitter.handle(1)
            self._density_split_handle.installEventFilter(self)

        self.density_plots = {}
        self.density_images = {}
        for row, col, key, title in [
            (0, 0, "xy", "XY slice"),
            (0, 1, "xz", "XZ slice"),
            (1, 0, "yz", "YZ slice"),
            (1, 1, "max_projection", "Max projection"),
        ]:
            plot = self.density_layout.addPlot(row=row, col=col, title=title)
            plot.setLabel("bottom", "X", units="Å")
            plot.setLabel("left", "Y", units="Å")
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setToolTip(f"Density map view: {title}.")
            self._simplify_plot_context_menu(plot)
            img = pg.ImageItem()
            plot.addItem(img)
            self.density_plots[key] = plot
            self.density_images[key] = img

        self.density_hist = pg.HistogramLUTItem()
        self.density_hist.axis.setLabel("Density", units="")
        self.density_layout.addItem(self.density_hist, row=0, col=2, rowspan=2)
        return panel

    def _build_water_dynamics_explore_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Water dynamics"))
        self.water_dynamics_combo = QtWidgets.QComboBox()
        self.water_dynamics_combo.currentTextChanged.connect(self._update_water_dynamics_plots)
        self.water_dynamics_log_check = QtWidgets.QCheckBox("Log τ")
        self.water_dynamics_log_check.toggled.connect(self._update_water_dynamics_plots)
        controls.addWidget(self.water_dynamics_combo)
        controls.addWidget(self.water_dynamics_log_check)
        self.water_dynamics_export_combo = QtWidgets.QComboBox()
        self.water_dynamics_export_combo.addItems(["SP(τ)", "HBL", "WOR"])
        self.water_dynamics_export_btn = QtWidgets.QPushButton("Export Plot")
        self.water_dynamics_export_btn.clicked.connect(self._export_water_dynamics_plot)
        controls.addWidget(self.water_dynamics_export_combo)
        controls.addWidget(self.water_dynamics_export_btn)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self._build_insight_strip("water"), 0)

        self.water_sp_plot = pg.PlotWidget()
        self._style_plot(self.water_sp_plot, "SP(τ)")
        self.water_sp_plot.setLabel("bottom", "Tau")
        self.water_sp_plot.setLabel("left", "Survival")
        self.water_hbl_plot = pg.PlotWidget()
        self._style_plot(self.water_hbl_plot, "H-bond lifetime")
        self.water_hbl_plot.setLabel("bottom", "Tau")
        self.water_hbl_plot.setLabel("left", "Correlation")
        self.water_wor_plot = pg.PlotWidget()
        self._style_plot(self.water_wor_plot, "Water orientational relaxation")
        self.water_wor_plot.setLabel("bottom", "Tau")
        self.water_wor_plot.setLabel("left", "Correlation")
        self.water_dynamics_note = QtWidgets.QLabel("")
        self.water_dynamics_note.setWordWrap(True)
        self.water_dynamics_summary = QtWidgets.QTableWidget()
        self.water_dynamics_summary.setColumnCount(5)
        self._configure_table_headers(
            self.water_dynamics_summary,
            ["resindex", "resid", "resname", "segid", "mean_lifetime"],
        )

        layout.addWidget(self.water_sp_plot, 1)
        layout.addWidget(self.water_hbl_plot, 1)
        layout.addWidget(self.water_wor_plot, 1)
        layout.addWidget(self.water_dynamics_note)
        layout.addWidget(self.water_dynamics_summary)
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
        self.export_split.splitterMoved.connect(lambda *_: self._update_splitter_constraints())
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
        soz_row = QtWidgets.QHBoxLayout()
        soz_row.setSpacing(8)
        soz_row.setContentsMargins(4, 4, 4, 4)
        soz_row.addWidget(QtWidgets.QLabel("SOZ"))
        self.timeline_soz_combo = QtWidgets.QComboBox()
        self.timeline_soz_combo.setToolTip("Select which SOZ to display.")
        soz_row.addWidget(self.timeline_soz_combo)
        soz_row.addStretch(1)
        layout.addLayout(soz_row)
        self._add_section_header(layout, "Occupancy Controls")

        self.timeline_overlay = QtWidgets.QCheckBox("Overlay all SOZs")
        self.timeline_step_check = QtWidgets.QCheckBox("Step plot")
        # Keep legacy control hidden and default to continuous line rendering.
        self.timeline_step_check.setChecked(False)
        self.timeline_step_check.setVisible(False)
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
        self.timeline_event_signed_check.setChecked(True)
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
        self.timeline_metric_combo = QtWidgets.QComboBox()
        self.timeline_metric_combo.addItems(["n_solvent", "entries", "exits", "occupancy_fraction"])
        self.timeline_secondary_combo = QtWidgets.QComboBox()
        self.timeline_secondary_combo.addItems(["None", "occupancy_fraction", "entries", "exits", "n_solvent"])
        self.timeline_smooth_check = QtWidgets.QCheckBox("Smooth")
        self.timeline_smooth_window = QtWidgets.QSpinBox()
        self.timeline_smooth_window.setRange(1, 500)
        self.timeline_smooth_window.setValue(5)
        for toggle in (
            self.timeline_overlay,
            self.timeline_step_check,
            self.timeline_markers_check,
            self.timeline_clamp_check,
            self.timeline_mean_check,
            self.timeline_median_check,
            self.timeline_shade_check,
            self.timeline_brush_check,
            self.timeline_smooth_check,
        ):
            toggle.setProperty("variant", "pill")

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
        controls_panel = QtWidgets.QWidget()
        controls_panel_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_panel_layout.setContentsMargins(0, 0, 0, 0)
        controls_panel_layout.setSpacing(8)

        metric_group = QtWidgets.QFrame()
        metric_group.setProperty("role", "inline-group")
        metric_row = QtWidgets.QHBoxLayout(metric_group)
        metric_row.setContentsMargins(8, 6, 8, 6)
        metric_row.setSpacing(8)
        metric_row.addWidget(QtWidgets.QLabel("Metric"))
        metric_row.addWidget(self.timeline_metric_combo)
        metric_row.addWidget(QtWidgets.QLabel("Secondary"))
        metric_row.addWidget(self.timeline_secondary_combo)
        metric_row.addStretch(1)
        metric_group.setVisible(False)
        controls_panel_layout.addWidget(metric_group)

        display_group = QtWidgets.QFrame()
        display_group.setProperty("role", "inline-group")
        display_row = QtWidgets.QHBoxLayout(display_group)
        display_row.setContentsMargins(8, 6, 8, 6)
        display_row.setSpacing(8)
        display_row.addWidget(QtWidgets.QLabel("Display Options"))
        display_row.addWidget(self.timeline_clamp_check)
        display_row.addWidget(self.timeline_brush_check)
        display_row.addWidget(self.timeline_brush_clear)
        display_row.addStretch(1)
        controls_panel_layout.addWidget(display_group)

        stats_group = QtWidgets.QFrame()
        stats_group.setProperty("role", "inline-group")
        stats_row = QtWidgets.QHBoxLayout(stats_group)
        stats_row.setContentsMargins(8, 6, 8, 6)
        stats_row.setSpacing(8)
        stats_row.addWidget(QtWidgets.QLabel("Statistics"))
        stats_row.addWidget(self.timeline_mean_check)
        stats_row.addWidget(self.timeline_median_check)
        stats_row.addWidget(self.timeline_shade_check)
        stats_row.addWidget(QtWidgets.QLabel("Shade ≥"))
        stats_row.addWidget(self.timeline_shade_threshold)
        stats_row.addStretch(1)
        controls_panel_layout.addWidget(stats_group)

        smoothing_group = QtWidgets.QFrame()
        smoothing_group.setProperty("role", "inline-group")
        smoothing_row = QtWidgets.QHBoxLayout(smoothing_group)
        smoothing_row.setContentsMargins(8, 6, 8, 6)
        smoothing_row.setSpacing(8)
        smoothing_row.addWidget(QtWidgets.QLabel("Smoothing"))
        smoothing_row.addWidget(self.timeline_smooth_check)
        smoothing_row.addWidget(QtWidgets.QLabel("Window"))
        smoothing_row.addWidget(self.timeline_smooth_window)
        smoothing_row.addStretch(1)
        smoothing_group.setVisible(False)
        controls_panel_layout.addWidget(smoothing_group)

        self.timeline_advanced_toggle = QtWidgets.QCheckBox("<> Advanced display")
        self.timeline_advanced_toggle.setChecked(False)
        controls_panel_layout.addWidget(self.timeline_advanced_toggle)
        self.timeline_advanced_panel = QtWidgets.QFrame()
        self.timeline_advanced_panel.setProperty("role", "inline-group")
        advanced_row = QtWidgets.QHBoxLayout(self.timeline_advanced_panel)
        advanced_row.setContentsMargins(8, 6, 8, 6)
        advanced_row.setSpacing(8)
        advanced_row.addWidget(self.timeline_overlay)
        advanced_row.addWidget(self.timeline_markers_check)
        advanced_row.addStretch(1)
        self.timeline_advanced_panel.setVisible(False)
        self.timeline_advanced_toggle.toggled.connect(self.timeline_advanced_panel.setVisible)
        controls_panel_layout.addWidget(self.timeline_advanced_panel)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addStretch(1)
        action_row.addWidget(self.timeline_save_btn)
        action_row.addWidget(self.timeline_copy_btn)
        action_row.addWidget(self.timeline_export_btn)
        controls_panel_layout.addLayout(action_row)

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
            "filter histograms, event rasters, and tables."
        )
        self.timeline_help_label.setWordWrap(True)

        self.timeline_top = QtWidgets.QWidget()
        timeline_top_layout = QtWidgets.QVBoxLayout(self.timeline_top)
        timeline_top_layout.setContentsMargins(0, 0, 0, 0)
        timeline_top_layout.addWidget(controls_panel, 0)
        timeline_top_layout.addWidget(self._build_insight_strip("timeline"), 0)
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
        timeline_top_layout.addWidget(self.timeline_summary_frame)

        event_controls = QtWidgets.QHBoxLayout()
        event_controls.setSpacing(8)
        event_section = QtWidgets.QLabel("Entry/Exit Controls")
        event_section.setProperty("role", "section")
        event_layout_header = QtWidgets.QVBoxLayout()
        event_layout_header.setContentsMargins(0, 0, 0, 0)
        event_layout_header.setSpacing(4)
        event_layout_header.addWidget(event_section)
        divider = QtWidgets.QFrame()
        divider.setProperty("role", "divider")
        event_layout_header.addWidget(divider)
        event_layout_header.addLayout(event_controls)
        event_controls.addWidget(QtWidgets.QLabel("Entry/Exit mode"))
        event_controls.addWidget(self.timeline_event_mode_combo)
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
        event_layout.addLayout(event_layout_header)
        event_layout.addWidget(self.timeline_event_plot, 1)

        (
            self.timeline_mode_toolbar,
            self.timeline_mode_group,
            self.timeline_mode_buttons,
        ) = self._build_feature_toolbar(
            ["Timeline", "Entry/Exit"],
            object_name="timeline-view",
        )
        layout.addWidget(self.timeline_mode_toolbar, 0)
        self.timeline_tabs = QtWidgets.QStackedWidget()
        self.timeline_tabs.addWidget(self.timeline_top)
        self.timeline_tabs.addWidget(self.timeline_event_container)
        self.timeline_mode_group.idClicked.connect(self.timeline_tabs.setCurrentIndex)
        self.timeline_tabs.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.timeline_mode_buttons, idx)
        )
        self._set_feature_toolbar_index(self.timeline_mode_buttons, 0)
        self.timeline_selector = self.timeline_mode_toolbar  # compatibility alias
        layout.addWidget(self.timeline_tabs, 1)
        self._update_event_controls_state()
        return panel

    def _build_plots_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.hist_controls_row = QtWidgets.QWidget()
        hist_controls = QtWidgets.QHBoxLayout(self.hist_controls_row)
        hist_controls.setSpacing(8)
        hist_controls.setContentsMargins(4, 4, 4, 4)
        plot_btn = QtWidgets.QPushButton("Plot Histogram")
        save_btn = QtWidgets.QPushButton("Save Plot")
        self.plots_copy_btn = QtWidgets.QToolButton()
        self.plots_copy_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton))
        self.plots_copy_btn.setToolTip("Copy current plot to clipboard")
        plot_btn.clicked.connect(self._update_hist_plot)
        save_btn.clicked.connect(self._export_current_plot)
        self.plots_copy_btn.clicked.connect(self._copy_current_plot)
        hist_controls.addStretch(1)
        hist_controls.addWidget(plot_btn)
        layout.addWidget(self.hist_controls_row)

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

        (
            self.plots_mode_toolbar,
            self.plots_mode_group,
            self.plots_mode_buttons,
        ) = self._build_feature_toolbar(
            ["Histogram", "Event Raster"],
            object_name="plots-view",
        )
        layout.addWidget(self.plots_mode_toolbar, 0)
        self.plots_tabs = QtWidgets.QStackedWidget()
        self.hist_zero_plot = pg.PlotWidget()
        self._style_plot(self.hist_zero_plot, "Zero vs Non-zero")
        self.hist_zero_plot.setMaximumHeight(140)
        self.hist_zero_plot.setLabel("bottom", "Category")
        self.hist_zero_plot.setLabel("left", "Count")
        self.hist_zero_plot.setVisible(False)
        self.hist_plot = pg.PlotWidget()
        self._style_plot(self.hist_plot, "Distribution")
        self.hist_plot.setLabel("bottom", "Value")
        self.hist_plot.setLabel("left", "Count")
        self.hist_info = QtWidgets.QLabel()
        self.hist_info.setWordWrap(True)
        hist_container = QtWidgets.QWidget()
        hist_layout = QtWidgets.QVBoxLayout(hist_container)
        hist_layout.addWidget(self._build_insight_strip("hist"), 0)
        hist_layout.addWidget(self.hist_zero_plot)
        hist_layout.addWidget(self.hist_plot)
        hist_layout.addWidget(self.hist_info)
        self.event_plot = pg.PlotWidget()
        self._style_plot(self.event_plot, "Occupancy Events")
        self.event_plot.setLabel("bottom", "Time", units="ns")
        self.event_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.event_plot.setLabel("left", "Solvent rank")
        try:
            self.event_plot.scene().sigMouseClicked.connect(self._on_event_plot_clicked)
        except Exception:
            pass
        self.event_info = QtWidgets.QLabel()
        self.event_info.setWordWrap(True)
        event_container = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(event_container)
        event_layout.addWidget(self._build_insight_strip("event"), 0)
        event_layout.addWidget(self.event_plot)
        event_layout.addWidget(self.event_info)
        self.plots_tabs.addWidget(hist_container)
        self.plots_tabs.addWidget(event_container)
        self.plots_mode_group.idClicked.connect(self.plots_tabs.setCurrentIndex)
        self.plots_tabs.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.plots_mode_buttons, idx)
        )
        self._set_feature_toolbar_index(self.plots_mode_buttons, 0)
        self.plots_selector = self.plots_mode_toolbar  # compatibility alias
        layout.addWidget(self.plots_tabs, 1)
        self.plots_tabs.currentChanged.connect(self._update_plots_controls_state)

        self.event_stride_spin.valueChanged.connect(self._queue_event_update)
        self.event_segment_check.toggled.connect(self._queue_event_update)
        self.event_min_duration_spin.valueChanged.connect(self._queue_event_update)
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

        self.tables_selector = QtWidgets.QComboBox()
        self.tables_selector.addItems(["Per Frame", "Per Solvent"])
        layout.addWidget(self.tables_selector)
        self.tables_tabs = QtWidgets.QStackedWidget()
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
        self.tables_tabs.addWidget(self.per_frame_table)
        self.tables_tabs.addWidget(self.per_solvent_table)
        self.tables_selector.currentIndexChanged.connect(self.tables_tabs.setCurrentIndex)
        layout.addWidget(self.tables_tabs, 1)
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
        self._setup_modern_table(self.extract_table)
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
        builder_pages = [
            ("Wizard", self._build_wizard_tab()),
            ("Distance Bridges", self._build_bridges_tab()),
            ("Density Maps", self._build_density_tab()),
            ("Selection Builder", self._build_selection_builder_tab()),
            ("Selection Tester", self._build_selection_tester_tab()),
            ("Advanced", self._build_advanced_tab()),
        ]
        (
            self.builder_toolbar,
            self.builder_group,
            self.builder_buttons,
        ) = self._build_feature_toolbar(
            [name for name, _ in builder_pages],
            object_name="define",
        )
        layout.addWidget(self.builder_toolbar, 0)

        self.builder_tabs = QtWidgets.QStackedWidget()
        for name, widget in builder_pages:
            self.builder_tabs.addWidget(widget)
        self.builder_group.idClicked.connect(self.builder_tabs.setCurrentIndex)
        self.builder_tabs.currentChanged.connect(
            lambda idx: self._set_feature_toolbar_index(self.builder_buttons, idx)
        )
        self._set_feature_toolbar_index(self.builder_buttons, 0)
        layout.addWidget(self.builder_tabs, 1)
        return panel

    def _build_wizard_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(panel)

        self.wizard_soz_name = QtWidgets.QLineEdit("SOZ_1")
        self.wizard_solvent_label = QtWidgets.QLineEdit("Water")
        self.wizard_solvent_label.setPlaceholderText("e.g., Water, Methanol, DMSO")
        self.wizard_solvent_label.setToolTip(
            "Display label for the solvent; does not change selection logic."
        )
        self.wizard_water_resnames = QtWidgets.QLineEdit("SOL,WAT,TIP3,HOH")
        self.wizard_water_resnames.setToolTip(
            "Residue names used to identify solvent molecules."
        )
        self.wizard_probe_selection = QtWidgets.QLineEdit("name O OW OH2")
        self.wizard_probe_selection.setToolTip(
            "MDAnalysis selection for probe atoms within solvent residues."
        )
        self.wizard_probe_position = QtWidgets.QComboBox()
        self.wizard_probe_position.addItems(["atom", "com", "cog"])
        self.wizard_probe_position.setToolTip(
            "How each solvent molecule is positioned for distances (atom, COM, COG)."
        )
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
        self.wizard_atom_mode.addItems(["probe", "atom", "com", "cog", "all"])
        self.wizard_atom_mode.setToolTip(
            "SOZ-specific probe mode override; 'probe' uses the global probe settings."
        )
        self.wizard_boolean = QtWidgets.QComboBox()
        self.wizard_boolean.addItem("Both (A and B)", "AND")
        self.wizard_boolean.addItem("Either (A or B)", "OR")
        self.wizard_boolean.setToolTip(
            "Combine A+B logic for the SOZ: 'Both' means a water must satisfy A and B; "
            "'Either' means it can satisfy A or B."
        )
        self.wizard_b_cutoff = QtWidgets.QLineEdit("3.5")

        layout.addRow("SOZ name", self.wizard_soz_name)
        layout.addRow("Solvent label", self.wizard_solvent_label)
        layout.addRow("Solvent resnames", self.wizard_water_resnames)
        layout.addRow("Probe selection", self.wizard_probe_selection)
        layout.addRow("Probe position", self.wizard_probe_position)
        self.wizard_solvent_note = QtWidgets.QLabel(
            "Solvent resnames define solvent residues; probe selection/position define solvent positioning. "
            "Update them for methanol, DMSO, mixed solvents, or ionic liquids."
        )
        self.wizard_solvent_note.setWordWrap(True)
        layout.addRow("", self.wizard_solvent_note)
        layout.addRow("Include ions", self.wizard_include_ions)
        layout.addRow("Ion resnames", self.wizard_ion_resnames)
        layout.addRow("Selection A", self.wizard_seed_a)
        layout.addRow("Selection A unique match", self.wizard_seed_a_unique)
        self.wizard_seed_a_status = QtWidgets.QLabel("Selection A matches: -")
        self.wizard_seed_a_status.setWordWrap(True)
        layout.addRow("", self.wizard_seed_a_status)
        layout.addRow("Shell cutoffs (A)", self.wizard_shell_cutoffs)
        layout.addRow("Mode", self.wizard_atom_mode)
        layout.addRow("Selection B (optional)", self.wizard_seed_b)
        layout.addRow("Selection B unique match", self.wizard_seed_b_unique)
        self.wizard_seed_b_status = QtWidgets.QLabel("Selection B matches: -")
        self.wizard_seed_b_status.setWordWrap(True)
        layout.addRow("", self.wizard_seed_b_status)
        layout.addRow("Selection B cutoff (A)", self.wizard_b_cutoff)
        layout.addRow("Combine A + B", self.wizard_boolean)
        self.wizard_boolean_note = QtWidgets.QLabel(
            "Both = water must satisfy Selection A and Selection B. "
            "Either = water can satisfy A or B."
        )
        self.wizard_boolean_note.setWordWrap(True)
        layout.addRow("", self.wizard_boolean_note)

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
        self._setup_modern_table(self.seed_validation_table)
        seed_layout.addWidget(self.seed_validation_table)
        layout.addRow(self.seed_validation_group)

        self.wizard_soz_name.textChanged.connect(self._schedule_seed_validation)
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
        layout.addWidget(self._build_selection_input_row(self.sel_builder_output, include_saved=True))

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

        intro = QtWidgets.QLabel(
            "Distance bridges identify solvent residues that sit within two distance cutoffs at the same time "
            "(selection A AND selection B). Use this to track shared waters between two sites.\n"
            "Explore: Bridges tab shows the time series, residence survival, and top bridge list."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QtWidgets.QHBoxLayout()

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.addWidget(QtWidgets.QLabel("Configured bridges"))
        self.distance_bridge_list = QtWidgets.QListWidget()
        self.distance_bridge_list.currentRowChanged.connect(self._on_distance_bridge_selected)
        list_layout.addWidget(self.distance_bridge_list, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        dup_btn = QtWidgets.QPushButton("Duplicate")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_distance_bridge_row)
        dup_btn.clicked.connect(self._duplicate_distance_bridge_row)
        remove_btn.clicked.connect(self._remove_distance_bridge_row)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addStretch(1)
        list_layout.addLayout(list_buttons)
        body.addWidget(list_panel, 1)

        self.distance_bridge_detail_scroll = QtWidgets.QScrollArea()
        self.distance_bridge_detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        details_group = QtWidgets.QGroupBox("Bridge details")
        form = QtWidgets.QFormLayout(details_group)

        self.distance_bridge_name_edit = QtWidgets.QLineEdit()
        self.distance_bridge_name_edit.setPlaceholderText("e.g., active_site_bridge")
        form.addRow("Name", self.distance_bridge_name_edit)

        self.distance_bridge_sel_a_combo = QtWidgets.QComboBox()
        self._register_selection_combo(self.distance_bridge_sel_a_combo)
        sel_a_row = self._build_selection_row(self.distance_bridge_sel_a_combo)
        form.addRow("Selection A", sel_a_row)

        self.distance_bridge_sel_b_combo = QtWidgets.QComboBox()
        self._register_selection_combo(self.distance_bridge_sel_b_combo)
        sel_b_row = self._build_selection_row(self.distance_bridge_sel_b_combo)
        form.addRow("Selection B", sel_b_row)

        self.distance_bridge_cutoff_a_spin = QtWidgets.QDoubleSpinBox()
        self.distance_bridge_cutoff_a_spin.setRange(0.0, 100.0)
        self.distance_bridge_cutoff_a_spin.setDecimals(3)
        self.distance_bridge_cutoff_a_spin.setSingleStep(0.1)
        form.addRow("Cutoff A", self.distance_bridge_cutoff_a_spin)

        self.distance_bridge_cutoff_b_spin = QtWidgets.QDoubleSpinBox()
        self.distance_bridge_cutoff_b_spin.setRange(0.0, 100.0)
        self.distance_bridge_cutoff_b_spin.setDecimals(3)
        self.distance_bridge_cutoff_b_spin.setSingleStep(0.1)
        form.addRow("Cutoff B", self.distance_bridge_cutoff_b_spin)

        self.distance_bridge_unit_combo = QtWidgets.QComboBox()
        self.distance_bridge_unit_combo.addItems(["A", "nm"])
        form.addRow("Unit", self.distance_bridge_unit_combo)

        self.distance_bridge_probe_combo = QtWidgets.QComboBox()
        self.distance_bridge_probe_combo.addItems(["probe", "atom", "com", "cog", "all"])
        self.distance_bridge_probe_combo.setToolTip(
            "Which solvent positions to use. 'probe' uses the global probe settings."
        )
        form.addRow("Mode", self.distance_bridge_probe_combo)

        detail_layout.addWidget(details_group)

        self.distance_bridge_warning = QtWidgets.QLabel("")
        self.distance_bridge_warning.setWordWrap(True)
        detail_layout.addWidget(self.distance_bridge_warning)
        detail_layout.addStretch(1)

        self.distance_bridge_detail_scroll.setWidget(detail_panel)
        body.addWidget(self.distance_bridge_detail_scroll, 2)

        layout.addLayout(body)

        for widget in (
            self.distance_bridge_name_edit,
            self.distance_bridge_sel_a_combo,
            self.distance_bridge_sel_b_combo,
            self.distance_bridge_unit_combo,
            self.distance_bridge_probe_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_distance_bridge_form_changed)
            else:
                widget.textChanged.connect(self._on_distance_bridge_form_changed)
        self.distance_bridge_cutoff_a_spin.valueChanged.connect(self._on_distance_bridge_form_changed)
        self.distance_bridge_cutoff_b_spin.valueChanged.connect(self._on_distance_bridge_form_changed)

        return panel

    def _build_hbond_bridges_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        intro = QtWidgets.QLabel(
            "H-bond water bridges identify solvent waters that hydrogen-bond to both selections "
            "within distance/angle thresholds. This captures bridging water networks.\n"
            "Explore: Bridges tab shows time series, residence, top bridges, comparator, and network view."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QtWidgets.QHBoxLayout()

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.addWidget(QtWidgets.QLabel("Configured H-bond bridges"))
        self.hbond_bridge_list = QtWidgets.QListWidget()
        self.hbond_bridge_list.currentRowChanged.connect(self._on_hbond_bridge_selected)
        list_layout.addWidget(self.hbond_bridge_list, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        dup_btn = QtWidgets.QPushButton("Duplicate")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_hbond_bridge_row)
        dup_btn.clicked.connect(self._duplicate_hbond_bridge_row)
        remove_btn.clicked.connect(self._remove_hbond_bridge_row)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addStretch(1)
        list_layout.addLayout(list_buttons)
        body.addWidget(list_panel, 1)

        self.hbond_bridge_detail_scroll = QtWidgets.QScrollArea()
        self.hbond_bridge_detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        details_group = QtWidgets.QGroupBox("Bridge details")
        form = QtWidgets.QFormLayout(details_group)

        self.hbond_bridge_name_edit = QtWidgets.QLineEdit()
        self.hbond_bridge_name_edit.setPlaceholderText("e.g., catalytic_bridge")
        form.addRow("Name", self.hbond_bridge_name_edit)

        self.hbond_bridge_sel_a_combo = QtWidgets.QComboBox()
        self._register_selection_combo(self.hbond_bridge_sel_a_combo)
        sel_a_row = self._build_selection_row(self.hbond_bridge_sel_a_combo, allow_raw=True)
        form.addRow("Selection A", sel_a_row)

        self.hbond_bridge_sel_b_combo = QtWidgets.QComboBox()
        self._register_selection_combo(self.hbond_bridge_sel_b_combo)
        sel_b_row = self._build_selection_row(self.hbond_bridge_sel_b_combo, allow_raw=True)
        form.addRow("Selection B", sel_b_row)

        self.hbond_bridge_distance_spin = QtWidgets.QDoubleSpinBox()
        self.hbond_bridge_distance_spin.setRange(0.0, 10.0)
        self.hbond_bridge_distance_spin.setDecimals(3)
        self.hbond_bridge_distance_spin.setSingleStep(0.1)
        form.addRow("Distance cutoff (A)", self.hbond_bridge_distance_spin)

        self.hbond_bridge_angle_spin = QtWidgets.QDoubleSpinBox()
        self.hbond_bridge_angle_spin.setRange(0.0, 180.0)
        self.hbond_bridge_angle_spin.setDecimals(1)
        self.hbond_bridge_angle_spin.setSingleStep(1.0)
        form.addRow("Angle cutoff (deg)", self.hbond_bridge_angle_spin)

        self.hbond_bridge_backend_combo = QtWidgets.QComboBox()
        self.hbond_bridge_backend_combo.addItems(["auto", "waterbridge", "hbond_analysis"])
        self.hbond_bridge_backend_combo.setToolTip(
            "auto: Try WaterBridgeAnalysis, fallback to HBA.\n"
            "waterbridge: Force WaterBridgeAnalysis (error if missing).\n"
            "hbond_analysis: Force HBA (fallback)."
        )
        form.addRow("Backend", self.hbond_bridge_backend_combo)

        detail_layout.addWidget(details_group)

        self.hbond_bridge_advanced_toggle = QtWidgets.QCheckBox("<> Advanced")
        self.hbond_bridge_advanced_toggle.toggled.connect(self._toggle_hbond_bridge_advanced)
        detail_layout.addWidget(self.hbond_bridge_advanced_toggle)

        self.hbond_bridge_advanced_group = QtWidgets.QGroupBox("Advanced selection overrides (optional)")
        advanced_form = QtWidgets.QFormLayout(self.hbond_bridge_advanced_group)
        self.hbond_bridge_water_edit = QtWidgets.QLineEdit()
        self.hbond_bridge_water_edit.setPlaceholderText("Defaults to solvent water selection")
        advanced_form.addRow(
            "Water selection",
            self._build_selection_input_row(self.hbond_bridge_water_edit, include_saved=True),
        )
        self.hbond_bridge_donors_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Donors selection",
            self._build_selection_input_row(self.hbond_bridge_donors_edit, include_saved=True),
        )
        self.hbond_bridge_hydrogens_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Hydrogens selection",
            self._build_selection_input_row(self.hbond_bridge_hydrogens_edit, include_saved=True),
        )
        self.hbond_bridge_acceptors_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Acceptors selection",
            self._build_selection_input_row(self.hbond_bridge_acceptors_edit, include_saved=True),
        )
        self.hbond_bridge_update_check = QtWidgets.QCheckBox("Update selections each frame")
        self.hbond_bridge_update_check.setChecked(True)
        advanced_form.addRow("", self.hbond_bridge_update_check)
        self.hbond_bridge_advanced_group.setVisible(False)
        detail_layout.addWidget(self.hbond_bridge_advanced_group)

        self.hbond_bridge_warning = QtWidgets.QLabel("")
        self.hbond_bridge_warning.setWordWrap(True)
        detail_layout.addWidget(self.hbond_bridge_warning)
        detail_layout.addStretch(1)

        self.hbond_bridge_detail_scroll.setWidget(detail_panel)
        body.addWidget(self.hbond_bridge_detail_scroll, 2)
        layout.addLayout(body)

        for widget in (
            self.hbond_bridge_name_edit,
            self.hbond_bridge_sel_a_combo,
            self.hbond_bridge_sel_b_combo,
            self.hbond_bridge_water_edit,
            self.hbond_bridge_donors_edit,
            self.hbond_bridge_hydrogens_edit,
            self.hbond_bridge_acceptors_edit,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_hbond_bridge_form_changed)
            else:
                widget.textChanged.connect(self._on_hbond_bridge_form_changed)
        self.hbond_bridge_distance_spin.valueChanged.connect(self._on_hbond_bridge_form_changed)
        self.hbond_bridge_angle_spin.valueChanged.connect(self._on_hbond_bridge_form_changed)
        self.hbond_bridge_update_check.toggled.connect(self._on_hbond_bridge_form_changed)

        return panel


    def _build_hbond_hydration_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        intro = QtWidgets.QLabel(
            "H-bond hydration measures hydrogen bonds between solvent and a residue selection. "
            "You can condition counts on SOZ membership or compute across all frames.\n"
            "Explore: Hydration tab shows residue frequency, top contacts, and timeline."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QtWidgets.QHBoxLayout()

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.addWidget(QtWidgets.QLabel("Configured H-bond hydration"))
        self.hbond_hydration_list = QtWidgets.QListWidget()
        self.hbond_hydration_list.currentRowChanged.connect(self._on_hbond_hydration_selected)
        list_layout.addWidget(self.hbond_hydration_list, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        dup_btn = QtWidgets.QPushButton("Duplicate")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_hbond_hydration_row)
        dup_btn.clicked.connect(self._duplicate_hbond_hydration_row)
        remove_btn.clicked.connect(self._remove_hbond_hydration_row)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addStretch(1)
        list_layout.addLayout(list_buttons)
        body.addWidget(list_panel, 1)

        self.hbond_hydration_detail_scroll = QtWidgets.QScrollArea()
        self.hbond_hydration_detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        details_group = QtWidgets.QGroupBox("Hydration details")
        form = QtWidgets.QFormLayout(details_group)

        self.hbond_hydration_name_edit = QtWidgets.QLineEdit()
        self.hbond_hydration_name_edit.setPlaceholderText("e.g., hbond_hydration")
        form.addRow("Name", self.hbond_hydration_name_edit)

        self.hbond_hydration_residue_edit = QtWidgets.QLineEdit()
        self.hbond_hydration_residue_edit.setPlaceholderText("e.g., protein or resname LIG")
        form.addRow(
            "Residue selection",
            self._build_selection_input_row(self.hbond_hydration_residue_edit, include_saved=True),
        )

        self.hbond_hydration_distance_spin = QtWidgets.QDoubleSpinBox()
        self.hbond_hydration_distance_spin.setRange(0.0, 10.0)
        self.hbond_hydration_distance_spin.setDecimals(3)
        self.hbond_hydration_distance_spin.setSingleStep(0.1)
        form.addRow("Distance cutoff (A)", self.hbond_hydration_distance_spin)

        self.hbond_hydration_angle_spin = QtWidgets.QDoubleSpinBox()
        self.hbond_hydration_angle_spin.setRange(0.0, 180.0)
        self.hbond_hydration_angle_spin.setDecimals(1)
        self.hbond_hydration_angle_spin.setSingleStep(1.0)
        form.addRow("Angle cutoff (deg)", self.hbond_hydration_angle_spin)

        self.hbond_hydration_conditioning_combo = QtWidgets.QComboBox()
        self.hbond_hydration_conditioning_combo.addItems(
            ["SOZ-conditioned", "Unconditioned (all frames)"]
        )
        form.addRow("Conditioning", self.hbond_hydration_conditioning_combo)

        self.hbond_hydration_soz_combo = QtWidgets.QComboBox()
        self._register_soz_combo(self.hbond_hydration_soz_combo, allow_default=True)
        form.addRow("SOZ name", self.hbond_hydration_soz_combo)

        detail_layout.addWidget(details_group)

        self.hbond_hydration_advanced_toggle = QtWidgets.QCheckBox("<> Advanced")
        self.hbond_hydration_advanced_toggle.toggled.connect(self._toggle_hbond_hydration_advanced)
        detail_layout.addWidget(self.hbond_hydration_advanced_toggle)

        self.hbond_hydration_advanced_group = QtWidgets.QGroupBox("Advanced selection overrides (optional)")
        advanced_form = QtWidgets.QFormLayout(self.hbond_hydration_advanced_group)
        self.hbond_hydration_water_edit = QtWidgets.QLineEdit()
        self.hbond_hydration_water_edit.setPlaceholderText("Defaults to solvent water selection")
        advanced_form.addRow(
            "Water selection",
            self._build_selection_input_row(self.hbond_hydration_water_edit, include_saved=True),
        )
        self.hbond_hydration_donors_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Donors selection",
            self._build_selection_input_row(self.hbond_hydration_donors_edit, include_saved=True),
        )
        self.hbond_hydration_hydrogens_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Hydrogens selection",
            self._build_selection_input_row(self.hbond_hydration_hydrogens_edit, include_saved=True),
        )
        self.hbond_hydration_acceptors_edit = QtWidgets.QLineEdit()
        advanced_form.addRow(
            "Acceptors selection",
            self._build_selection_input_row(self.hbond_hydration_acceptors_edit, include_saved=True),
        )
        self.hbond_hydration_update_check = QtWidgets.QCheckBox("Update selections each frame")
        self.hbond_hydration_update_check.setChecked(True)
        advanced_form.addRow("", self.hbond_hydration_update_check)
        self.hbond_hydration_advanced_group.setVisible(False)
        detail_layout.addWidget(self.hbond_hydration_advanced_group)

        self.hbond_hydration_warning = QtWidgets.QLabel("")
        self.hbond_hydration_warning.setWordWrap(True)
        detail_layout.addWidget(self.hbond_hydration_warning)
        detail_layout.addStretch(1)

        self.hbond_hydration_detail_scroll.setWidget(detail_panel)
        body.addWidget(self.hbond_hydration_detail_scroll, 2)
        layout.addLayout(body)

        for widget in (
            self.hbond_hydration_name_edit,
            self.hbond_hydration_residue_edit,
            self.hbond_hydration_conditioning_combo,
            self.hbond_hydration_soz_combo,
            self.hbond_hydration_water_edit,
            self.hbond_hydration_donors_edit,
            self.hbond_hydration_hydrogens_edit,
            self.hbond_hydration_acceptors_edit,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_hbond_hydration_form_changed)
            else:
                widget.textChanged.connect(self._on_hbond_hydration_form_changed)
        self.hbond_hydration_distance_spin.valueChanged.connect(self._on_hbond_hydration_form_changed)
        self.hbond_hydration_angle_spin.valueChanged.connect(self._on_hbond_hydration_form_changed)
        self.hbond_hydration_update_check.toggled.connect(self._on_hbond_hydration_form_changed)

        return panel

    def _build_density_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        intro = QtWidgets.QLabel(
            "Density maps compute 3D solvent (or atom) density grids from your selection. "
            "Use this to visualize preferred water sites.\n"
            "Explore: Density tab shows slices and summary metadata."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QtWidgets.QHBoxLayout()

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.addWidget(QtWidgets.QLabel("Configured density maps"))
        self.density_list = QtWidgets.QListWidget()
        self.density_list.currentRowChanged.connect(self._on_density_selected)
        list_layout.addWidget(self.density_list, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        dup_btn = QtWidgets.QPushButton("Duplicate")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_density_row)
        dup_btn.clicked.connect(self._duplicate_density_row)
        remove_btn.clicked.connect(self._remove_density_row)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addStretch(1)
        list_layout.addLayout(list_buttons)
        body.addWidget(list_panel, 1)

        self.density_detail_scroll = QtWidgets.QScrollArea()
        self.density_detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        details_group = QtWidgets.QGroupBox("Density map details")
        form = QtWidgets.QFormLayout(details_group)

        self.density_name_edit = QtWidgets.QLineEdit()
        self.density_name_edit.setPlaceholderText("e.g., water_density")
        form.addRow("Name", self.density_name_edit)

        self.density_selection_edit = QtWidgets.QLineEdit()
        self.density_selection_edit.setPlaceholderText("e.g., name O or resname HOH")
        form.addRow(
            "Density species",
            self._build_selection_input_row(self.density_selection_edit, include_saved=True),
        )

        self.density_grid_spin = QtWidgets.QDoubleSpinBox()
        self.density_grid_spin.setRange(0.1, 10.0)
        self.density_grid_spin.setDecimals(3)
        self.density_grid_spin.setSingleStep(0.1)
        form.addRow("Grid spacing (A)", self.density_grid_spin)

        self.density_padding_spin = QtWidgets.QDoubleSpinBox()
        self.density_padding_spin.setRange(0.0, 20.0)
        self.density_padding_spin.setDecimals(3)
        self.density_padding_spin.setSingleStep(0.5)
        form.addRow("Padding (A)", self.density_padding_spin)

        self.density_stride_spin = QtWidgets.QSpinBox()
        self.density_stride_spin.setRange(1, 1000000)
        self.density_stride_spin.setValue(1)
        form.addRow("Stride", self.density_stride_spin)

        self.density_align_check = QtWidgets.QCheckBox("Align trajectory before density")
        form.addRow("Alignment", self.density_align_check)

        self.density_align_selection_edit = QtWidgets.QLineEdit()
        self.density_align_selection_edit.setPlaceholderText("Alignment selection (e.g., protein and backbone)")
        form.addRow(
            "Align selection",
            self._build_selection_input_row(self.density_align_selection_edit, include_saved=True),
        )

        self.density_align_reference_combo = QtWidgets.QComboBox()
        self.density_align_reference_combo.addItems(["first_frame", "structure"])
        form.addRow("Align reference", self.density_align_reference_combo)

        align_path_row = QtWidgets.QHBoxLayout()
        self.density_align_path_edit = QtWidgets.QLineEdit()
        self.density_align_path_edit.setPlaceholderText("Reference structure path (optional)")
        self.density_align_path_btn = QtWidgets.QPushButton("Browse")
        self.density_align_path_btn.clicked.connect(self._browse_density_align_path)
        align_path_row.addWidget(self.density_align_path_edit)
        align_path_row.addWidget(self.density_align_path_btn)
        form.addRow("Reference path", align_path_row)

        detail_layout.addWidget(details_group)

        self.density_advanced_toggle = QtWidgets.QCheckBox("<> Advanced")
        detail_layout.addWidget(self.density_advanced_toggle)
        self.density_advanced_group = QtWidgets.QGroupBox("Conditioning & Visualization")
        opt_form = QtWidgets.QFormLayout(self.density_advanced_group)

        self.density_conditioning_policy_combo = QtWidgets.QComboBox()
        self.density_conditioning_policy_combo.addItems(["strict", "warn", "unsafe"])
        opt_form.addRow("Conditioning Policy", self.density_conditioning_policy_combo)

        self.density_view_mode_combo = QtWidgets.QComboBox()
        self.density_view_mode_combo.addItems(["physical", "relative", "score"])
        opt_form.addRow("View Mode", self.density_view_mode_combo)
        self.density_advanced_group.setVisible(False)
        self.density_advanced_toggle.toggled.connect(self.density_advanced_group.setVisible)
        detail_layout.addWidget(self.density_advanced_group)

        self.density_warning = QtWidgets.QLabel("")
        self.density_warning.setWordWrap(True)
        detail_layout.addWidget(self.density_warning)
        detail_layout.addStretch(1)

        self.density_detail_scroll.setWidget(detail_panel)
        body.addWidget(self.density_detail_scroll, 2)
        layout.addLayout(body)

        for widget in (
            self.density_name_edit,
            self.density_selection_edit,
            self.density_align_check,
            self.density_align_selection_edit,
            self.density_align_reference_combo,
            self.density_align_path_edit,
            self.density_conditioning_policy_combo,
            self.density_view_mode_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_density_form_changed)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._on_density_form_changed)
            else:
                widget.textChanged.connect(self._on_density_form_changed)
        self.density_grid_spin.valueChanged.connect(self._on_density_form_changed)
        self.density_padding_spin.valueChanged.connect(self._on_density_form_changed)
        self.density_stride_spin.valueChanged.connect(self._on_density_form_changed)

        return panel

    def _build_water_dynamics_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        intro = QtWidgets.QLabel(
            "Water dynamics computes residence survival (SP(tau)) and optional HBL/WOR metrics. "
            "Define the region via SOZ or an explicit selection.\n"
            "Explore: Water Dynamics tab shows SP(tau), HBL, and WOR plots plus summaries."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QtWidgets.QHBoxLayout()

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.addWidget(QtWidgets.QLabel("Configured water dynamics"))
        self.water_dynamics_list = QtWidgets.QListWidget()
        self.water_dynamics_list.currentRowChanged.connect(self._on_water_dynamics_selected)
        list_layout.addWidget(self.water_dynamics_list, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        dup_btn = QtWidgets.QPushButton("Duplicate")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_water_dynamics_row)
        dup_btn.clicked.connect(self._duplicate_water_dynamics_row)
        remove_btn.clicked.connect(self._remove_water_dynamics_row)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(remove_btn)
        list_buttons.addStretch(1)
        list_layout.addLayout(list_buttons)
        body.addWidget(list_panel, 1)

        self.water_dynamics_detail_scroll = QtWidgets.QScrollArea()
        self.water_dynamics_detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        details_group = QtWidgets.QGroupBox("Water dynamics details")
        form = QtWidgets.QFormLayout(details_group)

        self.water_dynamics_name_edit = QtWidgets.QLineEdit()
        self.water_dynamics_name_edit.setPlaceholderText("e.g., water_dynamics")
        form.addRow("Name", self.water_dynamics_name_edit)

        self.water_dynamics_region_mode_combo = QtWidgets.QComboBox()
        self.water_dynamics_region_mode_combo.addItems(["soz", "selection"])
        form.addRow("Region mode", self.water_dynamics_region_mode_combo)

        self.water_dynamics_soz_combo = QtWidgets.QComboBox()
        self._register_soz_combo(self.water_dynamics_soz_combo, allow_default=True)
        form.addRow("SOZ name", self.water_dynamics_soz_combo)

        self.water_dynamics_region_selection_edit = QtWidgets.QLineEdit()
        self.water_dynamics_region_selection_edit.setPlaceholderText("Selection for region when mode=selection")
        form.addRow(
            "Region selection",
            self._build_selection_input_row(self.water_dynamics_region_selection_edit, include_saved=True),
        )

        self.water_dynamics_region_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.water_dynamics_region_cutoff_spin.setRange(0.0, 100.0)
        self.water_dynamics_region_cutoff_spin.setDecimals(3)
        self.water_dynamics_region_cutoff_spin.setSingleStep(0.1)
        form.addRow("Region cutoff", self.water_dynamics_region_cutoff_spin)

        self.water_dynamics_region_unit_combo = QtWidgets.QComboBox()
        self.water_dynamics_region_unit_combo.addItems(["A", "nm"])
        form.addRow("Region unit", self.water_dynamics_region_unit_combo)

        self.water_dynamics_region_probe_combo = QtWidgets.QComboBox()
        self.water_dynamics_region_probe_combo.addItems(["probe", "atom", "com", "cog", "all"])
        form.addRow("Region probe mode", self.water_dynamics_region_probe_combo)

        self.water_dynamics_residence_combo = QtWidgets.QComboBox()
        self.water_dynamics_residence_combo.addItems(["continuous", "intermittent"])
        form.addRow("Residence mode", self.water_dynamics_residence_combo)

        self.water_dynamics_solute_edit = QtWidgets.QLineEdit()
        self.water_dynamics_solute_edit.setPlaceholderText("Solute selection for HBL (optional)")
        form.addRow(
            "Solute selection",
            self._build_selection_input_row(self.water_dynamics_solute_edit, include_saved=True),
        )

        self.water_dynamics_water_edit = QtWidgets.QLineEdit()
        self.water_dynamics_water_edit.setPlaceholderText("Water selection (optional)")
        form.addRow(
            "Water selection",
            self._build_selection_input_row(self.water_dynamics_water_edit, include_saved=True),
        )

        self.water_dynamics_hbond_distance_spin = QtWidgets.QDoubleSpinBox()
        self.water_dynamics_hbond_distance_spin.setRange(0.0, 10.0)
        self.water_dynamics_hbond_distance_spin.setDecimals(3)
        self.water_dynamics_hbond_distance_spin.setSingleStep(0.1)
        form.addRow("H-bond distance (A)", self.water_dynamics_hbond_distance_spin)

        self.water_dynamics_hbond_angle_spin = QtWidgets.QDoubleSpinBox()
        self.water_dynamics_hbond_angle_spin.setRange(0.0, 180.0)
        self.water_dynamics_hbond_angle_spin.setDecimals(1)
        self.water_dynamics_hbond_angle_spin.setSingleStep(1.0)
        form.addRow("H-bond angle (deg)", self.water_dynamics_hbond_angle_spin)

        self.water_dynamics_update_check = QtWidgets.QCheckBox("Update selections each frame")
        self.water_dynamics_update_check.setChecked(True)
        form.addRow("", self.water_dynamics_update_check)

        detail_layout.addWidget(details_group)

        self.water_dynamics_warning = QtWidgets.QLabel("")
        self.water_dynamics_warning.setWordWrap(True)
        detail_layout.addWidget(self.water_dynamics_warning)
        detail_layout.addStretch(1)

        self.water_dynamics_detail_scroll.setWidget(detail_panel)
        body.addWidget(self.water_dynamics_detail_scroll, 2)
        layout.addLayout(body)

        for widget in (
            self.water_dynamics_name_edit,
            self.water_dynamics_region_mode_combo,
            self.water_dynamics_soz_combo,
            self.water_dynamics_region_selection_edit,
            self.water_dynamics_region_unit_combo,
            self.water_dynamics_region_probe_combo,
            self.water_dynamics_residence_combo,
            self.water_dynamics_solute_edit,
            self.water_dynamics_water_edit,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_water_dynamics_form_changed)
            else:
                widget.textChanged.connect(self._on_water_dynamics_form_changed)
        self.water_dynamics_region_cutoff_spin.valueChanged.connect(self._on_water_dynamics_form_changed)
        self.water_dynamics_hbond_distance_spin.valueChanged.connect(self._on_water_dynamics_form_changed)
        self.water_dynamics_hbond_angle_spin.valueChanged.connect(self._on_water_dynamics_form_changed)
        self.water_dynamics_update_check.toggled.connect(self._on_water_dynamics_form_changed)

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

    def _selection_spec_to_string(self, spec: SelectionSpec) -> str:
        if spec.selection:
            return spec.selection
        parts = []
        if spec.resid is not None:
            parts.append(f"resid {spec.resid}")
        if spec.resname:
            parts.append(f"resname {spec.resname}")
        if spec.atomname:
            parts.append(f"name {spec.atomname}")
        if spec.segid:
            parts.append(f"segid {spec.segid}")
        if spec.chain:
            parts.append(f"chainID {spec.chain}")
        return " and ".join(parts)

    def _selection_preview_from_string(self, selection: str) -> str:
        if not selection:
            return "selection"
        text = selection.strip()
        resname = None
        resid = None
        resnum = None
        atom = None
        segid = None
        chain = None
        try:
            resname_match = re.search(r"\\bresname\\s+([^\\s)]+)", text, re.IGNORECASE)
            resid_match = re.search(r"\\bresid\\s+([^\\s)]+)", text, re.IGNORECASE)
            resnum_match = re.search(r"\\bresnum\\s+([^\\s)]+)", text, re.IGNORECASE)
            atom_match = re.search(r"\\bname\\s+([^\\s)]+)", text, re.IGNORECASE)
            segid_match = re.search(r"\\bsegid\\s+([^\\s)]+)", text, re.IGNORECASE)
            chain_match = re.search(r"\\bchainID\\s+([^\\s)]+)", text, re.IGNORECASE)
            resname = resname_match.group(1) if resname_match else None
            resid = resid_match.group(1) if resid_match else None
            resnum = resnum_match.group(1) if resnum_match else None
            atom = atom_match.group(1) if atom_match else None
            segid = segid_match.group(1) if segid_match else None
            chain = chain_match.group(1) if chain_match else None
        except Exception:
            resname = None
        parts = []
        residue_id = resid or resnum
        if resname and residue_id:
            parts.append(f"{resname}{residue_id}")
        elif resname:
            parts.append(resname)
        if atom:
            parts.append(atom)
        label = " ".join(parts) if parts else text
        if segid:
            label += f" (segid {segid})"
        elif chain:
            label += f" (chain {chain})"
        if len(label) > 60:
            label = label[:57] + "..."
        return label

    def _selection_display_label(self, label: str, spec: SelectionSpec | None) -> str:
        prefix = ""
        lowered = label.lower()
        if lowered.endswith("_selection_a"):
            prefix = "A: "
        elif lowered.endswith("_selection_b"):
            prefix = "B: "
        base = None
        if spec is not None:
            if spec.display_label:
                base = spec.display_label.strip()
            else:
                base = self._selection_preview_from_string(
                    spec.selection or self._selection_spec_to_string(spec)
                )
        if not base:
            base = label
        if prefix and not base.lower().startswith(("a:", "b:")):
            return prefix + base
        return base

    def _selection_tooltip(self, spec: SelectionSpec | None) -> str:
        if spec is None:
            return ""
        text = spec.selection or self._selection_spec_to_string(spec)
        if not text:
            return ""
        if len(text) > 120:
            return text[:117] + "..."
        return text

    def _selection_display_for_label(self, label: str | None) -> str:
        if not label:
            return "selection"
        project = self.state.project
        if project and label in project.selections:
            return self._selection_display_label(label, project.selections[label])
        return self._selection_preview_from_string(label)

    def _selection_combo_value(self, combo: QtWidgets.QComboBox) -> str:
        current_text = combo.currentText().strip() if combo.currentText() else ""
        current_data = combo.currentData()
        if isinstance(current_data, str) and current_data.strip():
            return current_data.strip()
        project = self.state.project
        if project and current_text in project.selections:
            return current_text
        display_map = getattr(self, "_selection_display_map_cache", {})
        if current_text in display_map:
            return display_map[current_text]
        if combo.property("allow_raw"):
            return current_text
        return ""

    def _set_selection_combo_value(self, combo: QtWidgets.QComboBox, value: str | None) -> None:
        if value is None:
            value = ""
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if data == value:
                combo.setCurrentIndex(idx)
                return
        for idx in range(combo.count()):
            if combo.itemText(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if combo.isEditable() and value:
            combo.setEditText(str(value))

    def _register_selection_combo(self, combo: QtWidgets.QComboBox) -> None:
        if not hasattr(self, "_selection_combos"):
            self._selection_combos = []
        if combo not in self._selection_combos:
            self._selection_combos.append(combo)
        combo.setProperty("wideControl", True)
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        try:
            combo.lineEdit().setPlaceholderText("Select a saved selection")
        except Exception:
            pass
        self._refresh_selection_combos()

    def _register_soz_combo(self, combo: QtWidgets.QComboBox, allow_default: bool = True) -> None:
        if not hasattr(self, "_soz_combos"):
            self._soz_combos = []
        if combo not in self._soz_combos:
            self._soz_combos.append(combo)
        combo.setProperty("allow_default", bool(allow_default))
        self._refresh_soz_combos()

    def _refresh_selection_combos(self) -> None:
        project = self.state.project
        entries = []
        display_map = {}
        if project:
            for label in sorted(project.selections.keys()):
                spec = project.selections[label]
                display = self._selection_display_label(label, spec)
                if display in display_map and display_map[display] != label:
                    display = f"{display} [{label}]"
                display_map[display] = label
                tooltip = self._selection_tooltip(spec)
                entries.append((display, label, tooltip))
        self._selection_display_map_cache = display_map
        for combo in getattr(self, "_selection_combos", []):
            current_text = combo.currentText().strip() if combo.currentText() else ""
            current_value = combo.currentData() or current_text
            if project and current_text in project.selections:
                current_value = current_text
            elif current_text in display_map:
                current_value = display_map[current_text]
            combo.blockSignals(True)
            combo.clear()
            for display, label, tooltip in entries:
                combo.addItem(display, label)
                idx = combo.count() - 1
                if tooltip:
                    combo.setItemData(idx, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
            if current_value:
                self._set_selection_combo_value(combo, str(current_value))
            combo.blockSignals(False)
            self._fit_combo_width(combo)

    def _refresh_soz_combos(self) -> None:
        project = self.state.project
        names = [soz.name for soz in project.sozs] if project else []
        for combo in getattr(self, "_soz_combos", []):
            allow_default = bool(combo.property("allow_default"))
            current_data = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if allow_default:
                combo.addItem("Default (first SOZ)", None)
            for name in names:
                combo.addItem(name, name)
            if current_data is None and allow_default:
                combo.setCurrentIndex(0)
            else:
                self._set_combo_by_data(combo, current_data)
            combo.blockSignals(False)
            self._fit_combo_width(combo)

    def _refresh_defined_soz_panel(self) -> None:
        if not hasattr(self, "soz_list"):
            return
        project = self.state.project
        project_names = [soz.name for soz in project.sozs] if project else []
        run_names = []
        if self.current_result and getattr(self.current_result, "soz_results", None):
            run_names = list(self.current_result.soz_results.keys())

        display_names = project_names if project_names else run_names
        self.soz_list.clear()
        self.soz_list.addItems(display_names)

        badge_text = f"SOZs {len(project_names)}"
        badge_tone = "neutral"
        hint_text = ""
        if project_names:
            hint_text = ""
        elif run_names:
            badge_text = f"SOZs {len(run_names)} (run)"
            badge_tone = "warning"
            hint_text = (
                "No SOZs are saved in this project. Showing SOZs from the latest run "
                "(Wizard fallback). Use Define SOZ -> Add SOZ to persist definitions."
            )
        else:
            badge_text = "SOZs 0"
            hint_text = "No SOZs defined yet. Use Define SOZ -> Add SOZ to create one."

        if hasattr(self, "soz_count_badge"):
            self._set_status_badge(self.soz_count_badge, badge_text, badge_tone)
        if hasattr(self, "soz_hint_label"):
            self.soz_hint_label.setText(hint_text)
            self.soz_hint_label.setVisible(bool(hint_text))

    def _set_combo_by_data(self, combo: QtWidgets.QComboBox, value: object) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if combo.isEditable() and value not in (None, ""):
            combo.setEditText(str(value))

    def _build_selection_row(
        self, combo: QtWidgets.QComboBox, allow_raw: bool = False
    ) -> QtWidgets.QWidget:
        combo.setProperty("allow_raw", bool(allow_raw))
        if allow_raw:
            try:
                combo.lineEdit().setPlaceholderText("Selection label or raw selection string")
            except Exception:
                pass
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(combo, 1)
        edit_btn = QtWidgets.QPushButton("Edit")
        new_btn = QtWidgets.QPushButton("New")
        wiz_a_btn = QtWidgets.QToolButton()
        wiz_a_btn.setText("A")
        wiz_a_btn.setToolTip("Use Wizard Selection A")
        wiz_b_btn = QtWidgets.QToolButton()
        wiz_b_btn.setText("B")
        wiz_b_btn.setToolTip("Use Wizard Selection B")
        edit_btn.clicked.connect(lambda: self._edit_selection_from_combo(combo))
        new_btn.clicked.connect(lambda: self._new_selection_from_combo(combo))
        wiz_a_btn.clicked.connect(lambda: self._apply_wizard_selection_to_combo(combo, "A"))
        wiz_b_btn.clicked.connect(lambda: self._apply_wizard_selection_to_combo(combo, "B"))
        row_layout.addWidget(edit_btn)
        row_layout.addWidget(new_btn)
        row_layout.addWidget(wiz_a_btn)
        row_layout.addWidget(wiz_b_btn)
        return row_widget

    def _build_selection_input_row(
        self,
        line_edit: QtWidgets.QLineEdit,
        include_saved: bool = True,
    ) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit, 1)

        wiz_a_btn = QtWidgets.QToolButton()
        wiz_a_btn.setText("A")
        wiz_a_btn.setToolTip("Use Wizard Selection A")
        wiz_a_btn.clicked.connect(
            lambda: self._apply_wizard_selection_to_line_edit(line_edit, "A")
        )
        row_layout.addWidget(wiz_a_btn)

        wiz_b_btn = QtWidgets.QToolButton()
        wiz_b_btn.setText("B")
        wiz_b_btn.setToolTip("Use Wizard Selection B")
        wiz_b_btn.clicked.connect(
            lambda: self._apply_wizard_selection_to_line_edit(line_edit, "B")
        )
        row_layout.addWidget(wiz_b_btn)

        if include_saved:
            saved_btn = QtWidgets.QToolButton()
            saved_btn.setText("Saved")
            saved_btn.setToolTip("Insert a saved selection")
            saved_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QtWidgets.QMenu(saved_btn)
            saved_btn.setMenu(menu)

            def _refresh_menu() -> None:
                self._populate_saved_selection_menu(menu, line_edit.setText)

            menu.aboutToShow.connect(_refresh_menu)
            row_layout.addWidget(saved_btn)

        return row_widget

    def _wizard_selection_label(self, which: str) -> str:
        soz_name = self.wizard_soz_name.text().strip() or "SOZ"
        suffix = "a" if which.upper() == "A" else "b"
        return f"{soz_name}_selection_{suffix}"

    def _wizard_selection_text(self, which: str) -> str:
        if which.upper() == "A":
            return self.wizard_seed_a.text().strip()
        return self.wizard_seed_b.text().strip()

    def _wizard_boolean_value(self) -> str:
        if not hasattr(self, "wizard_boolean"):
            return "and"
        data = self.wizard_boolean.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip().lower()
        text = (self.wizard_boolean.currentText() or "").strip().lower()
        if "or" in text:
            return "or"
        return "and"

    def _coerce_finite_float(self, value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _estimate_dt_from_frame_times(self, universe, max_samples: int = 8) -> float | None:
        trajectory = getattr(universe, "trajectory", None)
        if trajectory is None:
            return None
        try:
            n_frames = int(len(trajectory))
        except Exception:
            return None
        if n_frames < 2:
            return None

        sample_count = min(max_samples, n_frames)
        current_frame = self._coerce_finite_float(getattr(trajectory, "frame", None))
        times: list[float] = []
        try:
            for idx in range(sample_count):
                ts = trajectory[idx]
                time_value = self._coerce_finite_float(getattr(ts, "time", None))
                if time_value is not None:
                    times.append(time_value)
        finally:
            if current_frame is not None:
                frame_idx = int(current_frame)
                if 0 <= frame_idx < n_frames:
                    try:
                        trajectory[frame_idx]
                    except Exception:
                        pass

        if len(times) < 2:
            return None
        diffs = np.diff(np.asarray(times, dtype=float))
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[np.abs(diffs) > 1e-12]
        if diffs.size == 0:
            return None
        return float(np.median(diffs))

    def _snapshot_dt_details(self, universe, stride: int) -> dict[str, object]:
        trajectory = getattr(universe, "trajectory", None)
        dt_attr = self._coerce_finite_float(getattr(trajectory, "dt", None)) if trajectory else None
        dt_time = self._estimate_dt_from_frame_times(universe)

        dt_value = None
        dt_source = "unavailable"
        dt_warning = None
        if dt_attr is not None and dt_attr > 0:
            dt_value = dt_attr
            dt_source = "trajectory.dt"
            if dt_time is not None and dt_time > 0:
                rel_diff = abs(dt_attr - dt_time) / max(abs(dt_attr), abs(dt_time), 1e-12)
                if rel_diff > 0.05:
                    dt_warning = (
                        f"trajectory.dt={dt_attr:.6g} differs from median delta(ts.time)={dt_time:.6g}"
                    )
        elif dt_time is not None and dt_time > 0:
            dt_value = dt_time
            dt_source = "median delta(ts.time)"

        effective_dt = None
        if dt_value is not None:
            effective_dt = dt_value * max(1, int(stride))

        return {
            "dt": dt_value,
            "effective_dt": effective_dt,
            "source": dt_source,
            "warning": dt_warning,
            "trajectory_dt": dt_attr,
            "frame_time_dt": dt_time,
        }

    def _refresh_selection_state_ui(self, refresh_doctor: bool = True) -> None:
        project = self.state.project
        if not project:
            return
        self._refresh_selection_combos()
        if refresh_doctor:
            self._refresh_project_doctor_if_initialized()

    def _ensure_wizard_selection(self, which: str) -> str | None:
        if not self._ensure_project():
            return None
        selection = self._wizard_selection_text(which)
        if not selection:
            self.status_bar.showMessage(f"Wizard Selection {which} is empty.", 4000)
            return None
        label = self._wizard_selection_label(which)
        project = self.state.project
        if not project:
            return None
        spec = project.selections.get(label)
        if spec is None:
            spec = SelectionSpec(label=label, selection=selection)
        spec.selection = selection
        if which.upper() == "A":
            spec.require_unique = self.wizard_seed_a_unique.isChecked()
        else:
            spec.require_unique = self.wizard_seed_b_unique.isChecked()
        project.selections[label] = spec
        # Keep selection widgets in sync first; caller decides when to refresh doctor.
        self._refresh_selection_state_ui(refresh_doctor=False)
        return label

    def _apply_wizard_selection_to_combo(
        self, combo: QtWidgets.QComboBox, which: str
    ) -> None:
        label = self._ensure_wizard_selection(which)
        if label:
            self._set_selection_combo_value(combo, label)
            # Force downstream form sync even when value did not emit a change signal.
            combo.currentTextChanged.emit(combo.currentText())
            self._refresh_project_doctor_if_initialized()

    def _apply_wizard_selection_to_line_edit(
        self, line_edit: QtWidgets.QLineEdit, which: str
    ) -> None:
        selection = self._wizard_selection_text(which)
        if not selection:
            self.status_bar.showMessage(f"Wizard Selection {which} is empty.", 4000)
            return
        line_edit.setText(selection)

    def _populate_saved_selection_menu(
        self,
        menu: QtWidgets.QMenu,
        setter: Callable[[str], None],
    ) -> None:
        menu.clear()
        project = self.state.project
        if not project or not project.selections:
            empty_action = menu.addAction("No saved selections")
            empty_action.setEnabled(False)
            return
        for label in sorted(project.selections.keys()):
            spec = project.selections[label]
            selection_text = self._selection_spec_to_string(spec)
            display = self._selection_display_label(label, spec)
            action_text = f"{display} [{label}]" if display != label else label
            action = menu.addAction(action_text)
            action.triggered.connect(
                lambda checked=False, sel=selection_text: setter(sel)
            )

    def _edit_selection_from_combo(self, combo: QtWidgets.QComboBox) -> None:
        if not self.state.project:
            return
        label = self._selection_combo_value(combo)
        if not label:
            return
        spec = self.state.project.selections.get(label)
        if spec is None:
            raw_text = combo.currentText().strip() if combo.currentText() else ""
            selection_text = raw_text if combo.property("allow_raw") else None
            self._open_selection_editor(None, selection_text)
            return
        selection_text = self._selection_spec_to_string(spec)
        self._open_selection_editor(label, selection_text, spec.require_unique)

    def _new_selection_from_combo(self, combo: QtWidgets.QComboBox) -> None:
        label_text = combo.currentText().strip() if combo.currentText() else ""
        existing = False
        if self.state.project:
            existing = label_text in self.state.project.selections
            if label_text in getattr(self, "_selection_display_map_cache", {}):
                existing = True
        self._open_selection_editor(None if existing else (label_text or None), None)

    def _open_selection_editor(
        self,
        label: str | None,
        selection_text: str | None,
        require_unique: bool | None = None,
    ) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Selection Editor")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        label_edit = QtWidgets.QLineEdit(label or "")
        if label and label in project.selections:
            label_edit.setReadOnly(True)
        display_label_edit = QtWidgets.QLineEdit()
        if label and label in project.selections:
            display_label_edit.setText(project.selections[label].display_label or "")
        selection_edit = QtWidgets.QPlainTextEdit()
        selection_edit.setPlainText(selection_text or "")
        unique_check = QtWidgets.QCheckBox("Require unique match")
        if require_unique is not None:
            unique_check.setChecked(require_unique)
        form.addRow("Label", label_edit)
        form.addRow("Display label", display_label_edit)
        form.addRow("Selection string", selection_edit)
        form.addRow("", unique_check)
        layout.addLayout(form)
        helper = QtWidgets.QLabel(
            "Tip: use the Selection Builder/Tester tabs to craft and validate selection strings."
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def _save() -> None:
            new_label = label_edit.text().strip()
            sel_text = selection_edit.toPlainText().strip()
            if not new_label:
                self.status_bar.showMessage("Selection label is required.", 4000)
                return
            if not sel_text:
                self.status_bar.showMessage("Selection string is required.", 4000)
                return
            spec = project.selections.get(new_label)
            if spec is None:
                spec = SelectionSpec(label=new_label, selection=sel_text)
            spec.selection = sel_text
            display_text = display_label_edit.text().strip()
            spec.display_label = display_text or None
            spec.require_unique = unique_check.isChecked()
            project.selections[new_label] = spec
            self._refresh_selection_state_ui()
            dialog.accept()

        buttons.accepted.connect(_save)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def _toggle_hbond_bridge_advanced(self, enabled: bool) -> None:
        if hasattr(self, "hbond_bridge_advanced_group"):
            self.hbond_bridge_advanced_group.setVisible(enabled)

    def _toggle_hbond_hydration_advanced(self, enabled: bool) -> None:
        if hasattr(self, "hbond_hydration_advanced_group"):
            self.hbond_hydration_advanced_group.setVisible(enabled)

    def _browse_density_align_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select reference structure",
            "",
            "Structure files (*)",
        )
        if path and hasattr(self, "density_align_path_edit"):
            self.density_align_path_edit.setText(path)

    def _unique_name(self, base: str, existing: list[str]) -> str:
        if base not in existing:
            return base
        idx = 2
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    def _distance_bridge_item_text(self, bridge: DistanceBridgeConfig) -> str:
        sel_a = self._selection_display_for_label(bridge.selection_a)
        sel_b = self._selection_display_for_label(bridge.selection_b)
        return f"{bridge.name} ({sel_a} <-> {sel_b})"

    def _hbond_bridge_item_text(self, bridge: HbondWaterBridgeConfig) -> str:
        sel_a = self._selection_display_for_label(bridge.selection_a)
        sel_b = self._selection_display_for_label(bridge.selection_b)
        return f"{bridge.name} ({sel_a} <-> {sel_b})"


    def _hbond_hydration_item_text(self, cfg: HbondHydrationConfig) -> str:
        return f"{cfg.name} ({cfg.residue_selection})"

    def _density_item_text(self, cfg: DensityMapConfig) -> str:
        return f"{cfg.name} ({cfg.selection})"

    def _water_dynamics_item_text(self, cfg: WaterDynamicsConfig) -> str:
        region = cfg.region_mode
        return f"{cfg.name} ({region})"

    def _refresh_distance_bridge_table(self) -> None:
        if not hasattr(self, "distance_bridge_list"):
            return
        project = self.state.project
        if not project:
            return
        self._distance_bridge_refreshing = True
        current = self.distance_bridge_list.currentRow()
        self.distance_bridge_list.clear()
        for bridge in project.distance_bridges:
            self.distance_bridge_list.addItem(self._distance_bridge_item_text(bridge))
        self._distance_bridge_refreshing = False
        if project.distance_bridges:
            idx = current if 0 <= current < len(project.distance_bridges) else 0
            self.distance_bridge_list.setCurrentRow(idx)
        else:
            self._load_distance_bridge_form(None)

    def _refresh_hbond_bridge_table(self) -> None:
        if not hasattr(self, "hbond_bridge_list"):
            return
        project = self.state.project
        if not project:
            return
        self._hbond_bridge_refreshing = True
        current = self.hbond_bridge_list.currentRow()
        self.hbond_bridge_list.clear()
        for bridge in project.hbond_water_bridges:
            self.hbond_bridge_list.addItem(self._hbond_bridge_item_text(bridge))
        self._hbond_bridge_refreshing = False
        if project.hbond_water_bridges:
            idx = current if 0 <= current < len(project.hbond_water_bridges) else 0
            self.hbond_bridge_list.setCurrentRow(idx)
        else:
            self._load_hbond_bridge_form(None)


    def _refresh_hbond_hydration_table(self) -> None:
        if not hasattr(self, "hbond_hydration_list"):
            return
        project = self.state.project
        if not project:
            return
        self._hbond_hydration_refreshing = True
        current = self.hbond_hydration_list.currentRow()
        self.hbond_hydration_list.clear()
        for cfg in project.hbond_hydration:
            self.hbond_hydration_list.addItem(self._hbond_hydration_item_text(cfg))
        self._hbond_hydration_refreshing = False
        if project.hbond_hydration:
            idx = current if 0 <= current < len(project.hbond_hydration) else 0
            self.hbond_hydration_list.setCurrentRow(idx)
        else:
            self._load_hbond_hydration_form(None)

    def _refresh_density_table(self) -> None:
        if not hasattr(self, "density_list"):
            return
        project = self.state.project
        if not project:
            return
        self._density_refreshing = True
        current = self.density_list.currentRow()
        self.density_list.clear()
        for cfg in project.density_maps:
            self.density_list.addItem(self._density_item_text(cfg))
        self._density_refreshing = False
        if project.density_maps:
            idx = current if 0 <= current < len(project.density_maps) else 0
            self.density_list.setCurrentRow(idx)
        else:
            self._load_density_form(None)

    def _refresh_water_dynamics_table(self) -> None:
        if not hasattr(self, "water_dynamics_list"):
            return
        project = self.state.project
        if not project:
            return
        self._water_dynamics_refreshing = True
        current = self.water_dynamics_list.currentRow()
        self.water_dynamics_list.clear()
        for cfg in project.water_dynamics:
            self.water_dynamics_list.addItem(self._water_dynamics_item_text(cfg))
        self._water_dynamics_refreshing = False
        if project.water_dynamics:
            idx = current if 0 <= current < len(project.water_dynamics) else 0
            self.water_dynamics_list.setCurrentRow(idx)
        else:
            self._load_water_dynamics_form(None)

    def _add_distance_bridge_row(self) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        labels = list(project.selections.keys())
        sel_a = labels[0] if labels else ""
        sel_b = labels[1] if len(labels) > 1 else (labels[0] if labels else "")
        name = self._unique_name("distance_bridge", [b.name for b in project.distance_bridges])
        bridge = DistanceBridgeConfig(
            name=name,
            selection_a=sel_a,
            selection_b=sel_b,
            cutoff_a=3.5,
            cutoff_b=3.5,
            unit="A",
            atom_mode="probe",
        )
        project.distance_bridges.append(bridge)
        self._refresh_distance_bridge_table()
        self.distance_bridge_list.setCurrentRow(len(project.distance_bridges) - 1)

    def _duplicate_distance_bridge_row(self) -> None:
        if not self.state.project or not hasattr(self, "distance_bridge_list"):
            return
        idx = self.distance_bridge_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.distance_bridges):
            return
        clone = copy.deepcopy(self.state.project.distance_bridges[idx])
        clone.name = self._unique_name(f"{clone.name}_copy", [b.name for b in self.state.project.distance_bridges])
        self.state.project.distance_bridges.append(clone)
        self._refresh_distance_bridge_table()
        self.distance_bridge_list.setCurrentRow(len(self.state.project.distance_bridges) - 1)

    def _remove_distance_bridge_row(self) -> None:
        if not self.state.project or not hasattr(self, "distance_bridge_list"):
            return
        idx = self.distance_bridge_list.currentRow()
        if idx < 0:
            return
        self.state.project.distance_bridges.pop(idx)
        self._refresh_distance_bridge_table()

    def _add_hbond_bridge_row(self) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        labels = list(project.selections.keys())
        sel_a = labels[0] if labels else ""
        sel_b = labels[1] if len(labels) > 1 else (labels[0] if labels else "")
        name = self._unique_name("hbond_bridge", [b.name for b in project.hbond_water_bridges])
        bridge = HbondWaterBridgeConfig(
            name=name,
            selection_a=sel_a,
            selection_b=sel_b,
            distance=3.0,
            angle=150.0,
            update_selections=True,
        )
        project.hbond_water_bridges.append(bridge)
        self._refresh_hbond_bridge_table()
        self.hbond_bridge_list.setCurrentRow(len(project.hbond_water_bridges) - 1)

    def _duplicate_hbond_bridge_row(self) -> None:
        if not self.state.project or not hasattr(self, "hbond_bridge_list"):
            return
        idx = self.hbond_bridge_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.hbond_water_bridges):
            return
        clone = copy.deepcopy(self.state.project.hbond_water_bridges[idx])
        clone.name = self._unique_name(f"{clone.name}_copy", [b.name for b in self.state.project.hbond_water_bridges])
        self.state.project.hbond_water_bridges.append(clone)
        self._refresh_hbond_bridge_table()
        self.hbond_bridge_list.setCurrentRow(len(self.state.project.hbond_water_bridges) - 1)

    def _remove_hbond_bridge_row(self) -> None:
        if not self.state.project or not hasattr(self, "hbond_bridge_list"):
            return
        idx = self.hbond_bridge_list.currentRow()
        if idx < 0:
            return
        self.state.project.hbond_water_bridges.pop(idx)
        self._refresh_hbond_bridge_table()


    def _add_hbond_hydration_row(self) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        name = self._unique_name("hbond_hydration", [c.name for c in project.hbond_hydration])
        cfg = HbondHydrationConfig(
            name=name,
            residue_selection="protein",
            distance=3.0,
            angle=150.0,
            conditioning="soz",
            soz_name=None,
            update_selections=True,
        )
        project.hbond_hydration.append(cfg)
        self._refresh_hbond_hydration_table()
        self.hbond_hydration_list.setCurrentRow(len(project.hbond_hydration) - 1)

    def _duplicate_hbond_hydration_row(self) -> None:
        if not self.state.project or not hasattr(self, "hbond_hydration_list"):
            return
        idx = self.hbond_hydration_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.hbond_hydration):
            return
        clone = copy.deepcopy(self.state.project.hbond_hydration[idx])
        clone.name = self._unique_name(f"{clone.name}_copy", [c.name for c in self.state.project.hbond_hydration])
        self.state.project.hbond_hydration.append(clone)
        self._refresh_hbond_hydration_table()
        self.hbond_hydration_list.setCurrentRow(len(self.state.project.hbond_hydration) - 1)

    def _remove_hbond_hydration_row(self) -> None:
        if not self.state.project or not hasattr(self, "hbond_hydration_list"):
            return
        idx = self.hbond_hydration_list.currentRow()
        if idx < 0:
            return
        self.state.project.hbond_hydration.pop(idx)
        self._refresh_hbond_hydration_table()

    def _add_density_row(self) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        name = self._unique_name("density_map", [c.name for c in project.density_maps])
        cfg = DensityMapConfig(
            name=name,
            species_selection="name O",
            grid_spacing=1.0,
            padding=2.0,
            stride=1,
            align=False,
            align_reference="first_frame",
        )
        project.density_maps.append(cfg)
        self._refresh_density_table()
        self.density_list.setCurrentRow(len(project.density_maps) - 1)

    def _duplicate_density_row(self) -> None:
        if not self.state.project or not hasattr(self, "density_list"):
            return
        idx = self.density_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.density_maps):
            return
        clone = copy.deepcopy(self.state.project.density_maps[idx])
        clone.name = self._unique_name(f"{clone.name}_copy", [c.name for c in self.state.project.density_maps])
        self.state.project.density_maps.append(clone)
        self._refresh_density_table()
        self.density_list.setCurrentRow(len(self.state.project.density_maps) - 1)

    def _remove_density_row(self) -> None:
        if not self.state.project or not hasattr(self, "density_list"):
            return
        idx = self.density_list.currentRow()
        if idx < 0:
            return
        self.state.project.density_maps.pop(idx)
        self._refresh_density_table()

    def _add_water_dynamics_row(self) -> None:
        if not self._ensure_project():
            return
        project = self.state.project
        if not project:
            return
        name = self._unique_name("water_dynamics", [c.name for c in project.water_dynamics])
        cfg = WaterDynamicsConfig(
            name=name,
            region_mode="soz",
            region_cutoff=3.5,
            region_unit="A",
            region_probe_mode="probe",
            residence_mode="continuous",
            hbond_distance=3.0,
            hbond_angle=150.0,
            update_selections=True,
        )
        project.water_dynamics.append(cfg)
        self._refresh_water_dynamics_table()
        self.water_dynamics_list.setCurrentRow(len(project.water_dynamics) - 1)

    def _duplicate_water_dynamics_row(self) -> None:
        if not self.state.project or not hasattr(self, "water_dynamics_list"):
            return
        idx = self.water_dynamics_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.water_dynamics):
            return
        clone = copy.deepcopy(self.state.project.water_dynamics[idx])
        clone.name = self._unique_name(f"{clone.name}_copy", [c.name for c in self.state.project.water_dynamics])
        self.state.project.water_dynamics.append(clone)
        self._refresh_water_dynamics_table()
        self.water_dynamics_list.setCurrentRow(len(self.state.project.water_dynamics) - 1)

    def _remove_water_dynamics_row(self) -> None:
        if not self.state.project or not hasattr(self, "water_dynamics_list"):
            return
        idx = self.water_dynamics_list.currentRow()
        if idx < 0:
            return
        self.state.project.water_dynamics.pop(idx)
        self._refresh_water_dynamics_table()

    def _on_distance_bridge_selected(self, row: int) -> None:
        if getattr(self, "_distance_bridge_refreshing", False):
            return
        self._load_distance_bridge_form(row)

    def _on_hbond_bridge_selected(self, row: int) -> None:
        if getattr(self, "_hbond_bridge_refreshing", False):
            return
        self._load_hbond_bridge_form(row)


    def _on_hbond_hydration_selected(self, row: int) -> None:
        if getattr(self, "_hbond_hydration_refreshing", False):
            return
        self._load_hbond_hydration_form(row)

    def _on_density_selected(self, row: int) -> None:
        if getattr(self, "_density_refreshing", False):
            return
        self._load_density_form(row)

    def _on_water_dynamics_selected(self, row: int) -> None:
        if getattr(self, "_water_dynamics_refreshing", False):
            return
        self._load_water_dynamics_form(row)

    def _load_distance_bridge_form(self, row: int | None) -> None:
        if not self.state.project or not hasattr(self, "distance_bridge_detail_scroll"):
            return
        bridges = self.state.project.distance_bridges
        if row is None or row < 0 or row >= len(bridges):
            self.distance_bridge_detail_scroll.setEnabled(False)
            if hasattr(self, "distance_bridge_warning"):
                self.distance_bridge_warning.setText("No distance bridges configured yet.")
            return
        bridge = bridges[row]
        self._distance_bridge_form_updating = True
        self.distance_bridge_detail_scroll.setEnabled(True)
        self.distance_bridge_name_edit.setText(bridge.name)
        self._set_selection_combo_value(self.distance_bridge_sel_a_combo, bridge.selection_a)
        self._set_selection_combo_value(self.distance_bridge_sel_b_combo, bridge.selection_b)
        self.distance_bridge_cutoff_a_spin.setValue(float(bridge.cutoff_a))
        self.distance_bridge_cutoff_b_spin.setValue(float(bridge.cutoff_b))
        self.distance_bridge_unit_combo.setCurrentText(bridge.unit or "A")
        self.distance_bridge_probe_combo.setCurrentText(bridge.atom_mode or "probe")
        self._distance_bridge_form_updating = False
        self._validate_distance_bridge_form()

    def _load_hbond_bridge_form(self, row: int | None) -> None:
        if not self.state.project or not hasattr(self, "hbond_bridge_detail_scroll"):
            return
        bridges = self.state.project.hbond_water_bridges
        if row is None or row < 0 or row >= len(bridges):
            self.hbond_bridge_detail_scroll.setEnabled(False)
            if hasattr(self, "hbond_bridge_warning"):
                self.hbond_bridge_warning.setText("No H-bond bridges configured yet.")
            return
        bridge = bridges[row]
        self._hbond_bridge_form_updating = True
        self.hbond_bridge_detail_scroll.setEnabled(True)
        self.hbond_bridge_name_edit.setText(bridge.name)
        self._set_selection_combo_value(self.hbond_bridge_sel_a_combo, bridge.selection_a)
        self._set_selection_combo_value(self.hbond_bridge_sel_b_combo, bridge.selection_b)
        self.hbond_bridge_distance_spin.setValue(float(bridge.distance))
        self.hbond_bridge_angle_spin.setValue(float(bridge.angle))
        self.hbond_bridge_water_edit.setText(bridge.water_selection or "")
        self.hbond_bridge_donors_edit.setText(bridge.donors_selection or "")
        self.hbond_bridge_hydrogens_edit.setText(bridge.hydrogens_selection or "")
        self.hbond_bridge_acceptors_edit.setText(bridge.acceptors_selection or "")
        self.hbond_bridge_update_check.setChecked(bool(bridge.update_selections))
        idx_backend = self.hbond_bridge_backend_combo.findText(bridge.backend)
        if idx_backend >= 0:
            self.hbond_bridge_backend_combo.setCurrentIndex(idx_backend)
        else:
            self.hbond_bridge_backend_combo.setCurrentIndex(0)
        advanced_on = any(
            [
                bridge.water_selection,
                bridge.donors_selection,
                bridge.hydrogens_selection,
                bridge.acceptors_selection,
            ]
        )
        self.hbond_bridge_advanced_toggle.setChecked(advanced_on)
        self.hbond_bridge_advanced_group.setVisible(advanced_on)
        self._hbond_bridge_form_updating = False
        self._validate_hbond_bridge_form()


    def _load_hbond_hydration_form(self, row: int | None) -> None:
        if not self.state.project or not hasattr(self, "hbond_hydration_detail_scroll"):
            return
        configs = self.state.project.hbond_hydration
        if row is None or row < 0 or row >= len(configs):
            self.hbond_hydration_detail_scroll.setEnabled(False)
            if hasattr(self, "hbond_hydration_warning"):
                self.hbond_hydration_warning.setText("No H-bond hydration configs yet.")
            return
        cfg = configs[row]
        self._hbond_hydration_form_updating = True
        self.hbond_hydration_detail_scroll.setEnabled(True)
        self.hbond_hydration_name_edit.setText(cfg.name)
        self.hbond_hydration_residue_edit.setText(cfg.residue_selection)
        self.hbond_hydration_distance_spin.setValue(float(cfg.distance))
        self.hbond_hydration_angle_spin.setValue(float(cfg.angle))
        if cfg.conditioning == "soz":
            self.hbond_hydration_conditioning_combo.setCurrentIndex(0)
        else:
            self.hbond_hydration_conditioning_combo.setCurrentIndex(1)
        self._set_combo_by_data(self.hbond_hydration_soz_combo, cfg.soz_name)
        self.hbond_hydration_water_edit.setText(cfg.water_selection or "")
        self.hbond_hydration_donors_edit.setText(cfg.donors_selection or "")
        self.hbond_hydration_hydrogens_edit.setText(cfg.hydrogens_selection or "")
        self.hbond_hydration_acceptors_edit.setText(cfg.acceptors_selection or "")
        self.hbond_hydration_update_check.setChecked(bool(cfg.update_selections))
        advanced_on = any(
            [
                cfg.water_selection,
                cfg.donors_selection,
                cfg.hydrogens_selection,
                cfg.acceptors_selection,
            ]
        )
        self.hbond_hydration_advanced_toggle.setChecked(advanced_on)
        self.hbond_hydration_advanced_group.setVisible(advanced_on)
        self._hbond_hydration_form_updating = False
        self._sync_hbond_hydration_controls()
        self._validate_hbond_hydration_form()

    def _load_density_form(self, row: int | None) -> None:
        if not self.state.project or not hasattr(self, "density_detail_scroll"):
            return
        configs = self.state.project.density_maps
        if row is None or row < 0 or row >= len(configs):
            self.density_detail_scroll.setEnabled(False)
            if hasattr(self, "density_warning"):
                self.density_warning.setText("No density maps configured yet.")
            return
        cfg = configs[row]
        normalized_species = sanitize_selection_string(cfg.species_selection)
        if normalized_species != cfg.species_selection:
            cfg.species_selection = normalized_species
        self._density_form_updating = True
        self.density_detail_scroll.setEnabled(True)
        self.density_name_edit.setText(cfg.name)
        self.density_selection_edit.setText(normalized_species)
        self.density_grid_spin.setValue(float(cfg.grid_spacing))
        self.density_padding_spin.setValue(float(cfg.padding))
        self.density_stride_spin.setValue(int(cfg.stride))
        self.density_align_check.setChecked(bool(cfg.align))
        self.density_align_selection_edit.setText(cfg.align_selection or "")
        self.density_align_reference_combo.setCurrentText(cfg.align_reference or "first_frame")
        self.density_align_path_edit.setText(cfg.align_reference_path or "")
        self.density_view_mode_combo.setCurrentText(cfg.view_mode or "physical")
        self.density_conditioning_policy_combo.setCurrentText(cfg.conditioning_policy or "strict")
        self._density_form_updating = False
        self._sync_density_align_controls()
        self._validate_density_form()

    def _load_water_dynamics_form(self, row: int | None) -> None:
        if not self.state.project or not hasattr(self, "water_dynamics_detail_scroll"):
            return
        configs = self.state.project.water_dynamics
        if row is None or row < 0 or row >= len(configs):
            self.water_dynamics_detail_scroll.setEnabled(False)
            if hasattr(self, "water_dynamics_warning"):
                self.water_dynamics_warning.setText("No water dynamics configs yet.")
            return
        cfg = configs[row]
        self._water_dynamics_form_updating = True
        self.water_dynamics_detail_scroll.setEnabled(True)
        self.water_dynamics_name_edit.setText(cfg.name)
        self.water_dynamics_region_mode_combo.setCurrentText(cfg.region_mode)
        self._set_combo_by_data(self.water_dynamics_soz_combo, cfg.soz_name)
        self.water_dynamics_region_selection_edit.setText(cfg.region_selection or "")
        self.water_dynamics_region_cutoff_spin.setValue(float(cfg.region_cutoff))
        self.water_dynamics_region_unit_combo.setCurrentText(cfg.region_unit or "A")
        self.water_dynamics_region_probe_combo.setCurrentText(cfg.region_probe_mode or "probe")
        self.water_dynamics_residence_combo.setCurrentText(cfg.residence_mode or "continuous")
        self.water_dynamics_solute_edit.setText(cfg.solute_selection or "")
        self.water_dynamics_water_edit.setText(cfg.water_selection or "")
        self.water_dynamics_hbond_distance_spin.setValue(float(cfg.hbond_distance))
        self.water_dynamics_hbond_angle_spin.setValue(float(cfg.hbond_angle))
        self.water_dynamics_update_check.setChecked(bool(cfg.update_selections))
        self._water_dynamics_form_updating = False
        self._sync_water_dynamics_controls()
        self._validate_water_dynamics_form()

    def _on_distance_bridge_form_changed(self) -> None:
        if getattr(self, "_distance_bridge_form_updating", False):
            return
        if not self.state.project or not hasattr(self, "distance_bridge_list"):
            return
        idx = self.distance_bridge_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.distance_bridges):
            return
        bridge = self.state.project.distance_bridges[idx]
        name = self.distance_bridge_name_edit.text().strip()
        bridge.name = name or f"distance_bridge_{idx+1}"
        bridge.selection_a = self._selection_combo_value(self.distance_bridge_sel_a_combo)
        bridge.selection_b = self._selection_combo_value(self.distance_bridge_sel_b_combo)
        bridge.cutoff_a = float(self.distance_bridge_cutoff_a_spin.value())
        bridge.cutoff_b = float(self.distance_bridge_cutoff_b_spin.value())
        bridge.unit = self.distance_bridge_unit_combo.currentText().strip() or "A"
        bridge.atom_mode = self.distance_bridge_probe_combo.currentText().strip() or "probe"
        item = self.distance_bridge_list.item(idx)
        if item is not None:
            item.setText(self._distance_bridge_item_text(bridge))
        self._validate_distance_bridge_form()
        self._refresh_project_doctor_if_initialized()

    def _on_hbond_bridge_form_changed(self) -> None:
        if getattr(self, "_hbond_bridge_form_updating", False):
            return
        if not self.state.project or not hasattr(self, "hbond_bridge_list"):
            return
        idx = self.hbond_bridge_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.hbond_water_bridges):
            return
        bridge = self.state.project.hbond_water_bridges[idx]
        name = self.hbond_bridge_name_edit.text().strip()
        bridge.name = name or f"hbond_bridge_{idx+1}"
        bridge.selection_a = self._selection_combo_value(self.hbond_bridge_sel_a_combo)
        bridge.selection_b = self._selection_combo_value(self.hbond_bridge_sel_b_combo)
        bridge.distance = float(self.hbond_bridge_distance_spin.value())
        bridge.angle = float(self.hbond_bridge_angle_spin.value())
        bridge.water_selection = self.hbond_bridge_water_edit.text().strip() or None
        bridge.donors_selection = self.hbond_bridge_donors_edit.text().strip() or None
        bridge.hydrogens_selection = self.hbond_bridge_hydrogens_edit.text().strip() or None
        bridge.acceptors_selection = self.hbond_bridge_acceptors_edit.text().strip() or None
        bridge.update_selections = bool(self.hbond_bridge_update_check.isChecked())
        bridge.backend = self.hbond_bridge_backend_combo.currentText().strip()
        item = self.hbond_bridge_list.item(idx)
        if item is not None:
            item.setText(self._hbond_bridge_item_text(bridge))
        self._validate_hbond_bridge_form()


    def _on_hbond_hydration_form_changed(self) -> None:
        if getattr(self, "_hbond_hydration_form_updating", False):
            return
        if not self.state.project or not hasattr(self, "hbond_hydration_list"):
            return
        idx = self.hbond_hydration_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.hbond_hydration):
            return
        cfg = self.state.project.hbond_hydration[idx]
        name = self.hbond_hydration_name_edit.text().strip()
        cfg.name = name or f"hbond_hydration_{idx+1}"
        cfg.residue_selection = self.hbond_hydration_residue_edit.text().strip() or "protein"
        cfg.distance = float(self.hbond_hydration_distance_spin.value())
        cfg.angle = float(self.hbond_hydration_angle_spin.value())
        cfg.conditioning = "soz" if self.hbond_hydration_conditioning_combo.currentIndex() == 0 else "all"
        cfg.soz_name = self.hbond_hydration_soz_combo.currentData()
        cfg.water_selection = self.hbond_hydration_water_edit.text().strip() or None
        cfg.donors_selection = self.hbond_hydration_donors_edit.text().strip() or None
        cfg.hydrogens_selection = self.hbond_hydration_hydrogens_edit.text().strip() or None
        cfg.acceptors_selection = self.hbond_hydration_acceptors_edit.text().strip() or None
        cfg.update_selections = bool(self.hbond_hydration_update_check.isChecked())
        item = self.hbond_hydration_list.item(idx)
        if item is not None:
            item.setText(self._hbond_hydration_item_text(cfg))
        self._sync_hbond_hydration_controls()
        self._validate_hbond_hydration_form()

    def _on_density_form_changed(self) -> None:
        if getattr(self, "_density_form_updating", False):
            return
        if not self.state.project or not hasattr(self, "density_list"):
            return
        idx = self.density_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.density_maps):
            return
        cfg = self.state.project.density_maps[idx]
        name = self.density_name_edit.text().strip()
        cfg.name = name or f"density_map_{idx+1}"
        raw_selection = self.density_selection_edit.text().strip()
        normalized_selection = sanitize_selection_string(raw_selection)
        if normalized_selection and normalized_selection != raw_selection:
            self.density_selection_edit.blockSignals(True)
            self.density_selection_edit.setText(normalized_selection)
            self.density_selection_edit.blockSignals(False)
            self.status_bar.showMessage(
                f"Density selection normalized to '{normalized_selection}'.",
                5000,
            )
        cfg.species_selection = normalized_selection or "name O"
        cfg.grid_spacing = float(self.density_grid_spin.value())
        cfg.padding = float(self.density_padding_spin.value())
        cfg.stride = int(self.density_stride_spin.value())
        cfg.align = bool(self.density_align_check.isChecked())
        cfg.align_selection = self.density_align_selection_edit.text().strip() or None
        cfg.align_reference = self.density_align_reference_combo.currentText().strip() or "first_frame"
        cfg.align_reference_path = self.density_align_path_edit.text().strip() or None
        cfg.view_mode = self.density_view_mode_combo.currentText().strip() or "physical"
        cfg.conditioning_policy = self.density_conditioning_policy_combo.currentText().strip() or "strict"
        item = self.density_list.item(idx)
        if item is not None:
            item.setText(self._density_item_text(cfg))
        self._sync_density_align_controls()
        self._validate_density_form()

    def _on_water_dynamics_form_changed(self) -> None:
        if getattr(self, "_water_dynamics_form_updating", False):
            return
        if not self.state.project or not hasattr(self, "water_dynamics_list"):
            return
        idx = self.water_dynamics_list.currentRow()
        if idx < 0 or idx >= len(self.state.project.water_dynamics):
            return
        cfg = self.state.project.water_dynamics[idx]
        name = self.water_dynamics_name_edit.text().strip()
        cfg.name = name or f"water_dynamics_{idx+1}"
        cfg.region_mode = self.water_dynamics_region_mode_combo.currentText().strip() or "soz"
        cfg.soz_name = self.water_dynamics_soz_combo.currentData()
        cfg.region_selection = self.water_dynamics_region_selection_edit.text().strip() or None
        cfg.region_cutoff = float(self.water_dynamics_region_cutoff_spin.value())
        cfg.region_unit = self.water_dynamics_region_unit_combo.currentText().strip() or "A"
        cfg.region_probe_mode = self.water_dynamics_region_probe_combo.currentText().strip() or "probe"
        cfg.residence_mode = self.water_dynamics_residence_combo.currentText().strip() or "continuous"
        cfg.solute_selection = self.water_dynamics_solute_edit.text().strip() or None
        cfg.water_selection = self.water_dynamics_water_edit.text().strip() or None
        cfg.hbond_distance = float(self.water_dynamics_hbond_distance_spin.value())
        cfg.hbond_angle = float(self.water_dynamics_hbond_angle_spin.value())
        cfg.update_selections = bool(self.water_dynamics_update_check.isChecked())
        item = self.water_dynamics_list.item(idx)
        if item is not None:
            item.setText(self._water_dynamics_item_text(cfg))
        self._sync_water_dynamics_controls()
        self._validate_water_dynamics_form()

    def _validate_distance_bridge_form(self) -> None:
        if not hasattr(self, "distance_bridge_warning") or not self.state.project:
            return
        messages = []
        sel_a = self._selection_combo_value(self.distance_bridge_sel_a_combo)
        sel_b = self._selection_combo_value(self.distance_bridge_sel_b_combo)
        if not sel_a:
            messages.append("Selection A is required.")
        elif sel_a not in self.state.project.selections:
            messages.append(f"Selection A '{sel_a}' is not defined. Click New to create it.")
        if not sel_b:
            messages.append("Selection B is required.")
        elif sel_b not in self.state.project.selections:
            messages.append(f"Selection B '{sel_b}' is not defined. Click New to create it.")
        if self.distance_bridge_cutoff_a_spin.value() <= 0:
            messages.append("Cutoff A must be > 0.")
        if self.distance_bridge_cutoff_b_spin.value() <= 0:
            messages.append("Cutoff B must be > 0.")
        self.distance_bridge_warning.setText(" ".join(messages))

    def _validate_hbond_bridge_form(self) -> None:
        if not hasattr(self, "hbond_bridge_warning"):
            return
        messages = []
        sel_a = self._selection_combo_value(self.hbond_bridge_sel_a_combo)
        sel_b = self._selection_combo_value(self.hbond_bridge_sel_b_combo)
        if not sel_a:
            messages.append("Selection A is required.")
        if not sel_b:
            messages.append("Selection B is required.")
        if self.hbond_bridge_distance_spin.value() <= 0:
            messages.append("Distance cutoff must be > 0.")
        if self.hbond_bridge_angle_spin.value() <= 0:
            messages.append("Angle cutoff must be > 0.")
        self.hbond_bridge_warning.setText(" ".join(messages))

    def _validate_hbond_hydration_form(self) -> None:
        if not hasattr(self, "hbond_hydration_warning"):
            return
        messages = []
        if not self.hbond_hydration_residue_edit.text().strip():
            messages.append("Residue selection is required.")
        if self.hbond_hydration_distance_spin.value() <= 0:
            messages.append("Distance cutoff must be > 0.")
        if self.hbond_hydration_angle_spin.value() <= 0:
            messages.append("Angle cutoff must be > 0.")
        if (
            self.state.project
            and not self.state.project.sozs
            and self.hbond_hydration_conditioning_combo.currentIndex() == 0
        ):
            messages.append("No SOZs defined; SOZ conditioning will be empty.")
        self.hbond_hydration_warning.setText(" ".join(messages))

    def _validate_density_form(self) -> None:
        if not hasattr(self, "density_warning"):
            return
        messages = []
        if not self.density_selection_edit.text().strip():
            messages.append("Selection is required.")
        if self.density_grid_spin.value() <= 0:
            messages.append("Grid spacing must be > 0.")
        self.density_warning.setText(" ".join(messages))

    def _validate_water_dynamics_form(self) -> None:
        if not hasattr(self, "water_dynamics_warning"):
            return
        messages = []
        if not self.water_dynamics_region_mode_combo.currentText().strip():
            messages.append("Region mode is required.")
        if self.water_dynamics_region_mode_combo.currentText().strip() == "selection":
            if not self.water_dynamics_region_selection_edit.text().strip():
                messages.append("Region selection is required for selection mode.")
        if (
            self.state.project
            and not self.state.project.sozs
            and self.water_dynamics_region_mode_combo.currentText().strip() == "soz"
        ):
            messages.append("No SOZs defined; SOZ-based residence will be empty.")
        if self.water_dynamics_region_cutoff_spin.value() <= 0:
            messages.append("Region cutoff must be > 0.")
        self.water_dynamics_warning.setText(" ".join(messages))

    def _sync_hbond_hydration_controls(self) -> None:
        if not hasattr(self, "hbond_hydration_conditioning_combo"):
            return
        conditioned = self.hbond_hydration_conditioning_combo.currentIndex() == 0
        self.hbond_hydration_soz_combo.setEnabled(conditioned)

    def _sync_density_align_controls(self) -> None:
        if not hasattr(self, "density_align_check"):
            return
        align_enabled = self.density_align_check.isChecked()
        self.density_align_selection_edit.setEnabled(align_enabled)
        self.density_align_reference_combo.setEnabled(align_enabled)
        ref_is_structure = self.density_align_reference_combo.currentText() == "structure"
        self.density_align_path_edit.setEnabled(align_enabled and ref_is_structure)
        self.density_align_path_btn.setEnabled(align_enabled and ref_is_structure)

    def _sync_water_dynamics_controls(self) -> None:
        if not hasattr(self, "water_dynamics_region_mode_combo"):
            return
        mode = self.water_dynamics_region_mode_combo.currentText().strip()
        selection_mode = mode == "selection"
        self.water_dynamics_soz_combo.setEnabled(not selection_mode)
        self.water_dynamics_region_selection_edit.setEnabled(selection_mode)
        self.water_dynamics_region_cutoff_spin.setEnabled(True)
        self.water_dynamics_region_unit_combo.setEnabled(True)
        self.water_dynamics_region_probe_combo.setEnabled(selection_mode)

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
        probe_btn = QtWidgets.QPushButton("Test Probe")
        probe_btn.clicked.connect(self._test_probe_selection)
        options_row.addWidget(QtWidgets.QLabel("Max rows"))
        options_row.addWidget(self.selection_limit_spin)
        options_row.addWidget(self.selection_use_trajectory)
        options_row.addWidget(test_btn)
        options_row.addWidget(probe_btn)
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
        self._setup_modern_table(self.selection_table)
        layout.addWidget(self.selection_table)

        self._tester_universe = None
        self._tester_key = None

        return panel

    def _strip_removed_analysis_options(self, project: ProjectConfig | None) -> None:
        if not project:
            return
        project.hbond_water_bridges.clear()
        project.hbond_hydration.clear()
        project.water_dynamics.clear()

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
        self._strip_removed_analysis_options(project)
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
        self._strip_removed_analysis_options(project)
        topology_path = str(project.inputs.topology or "").strip()
        trajectory_path = str(project.inputs.trajectory or "").strip()
        self.topology_label.setText(topology_path or "No topology selected")
        self.topology_label.setToolTip(topology_path or "No topology selected")
        self.trajectory_label.setText(trajectory_path or "No trajectory selected")
        self.trajectory_label.setToolTip(trajectory_path or "No trajectory selected")
        if hasattr(self, "trajectory_clear_btn"):
            self.trajectory_clear_btn.setEnabled(bool(project.inputs.trajectory))
        if hasattr(self, "project_inputs_status"):
            if topology_path and trajectory_path:
                self._set_status_badge(self.project_inputs_status, "Loaded", "success")
            elif topology_path:
                self._set_status_badge(self.project_inputs_status, "Topology only", "warning")
            else:
                self._set_status_badge(self.project_inputs_status, "Missing topology", "error")
        stride = int(project.analysis.stride)
        try:
            universe = self._ensure_preflight_universe()
            if universe is None:
                raise RuntimeError("Inputs unavailable")
            dt_details = self._snapshot_dt_details(universe, stride)
            dt_value = dt_details["dt"]
            dt_effective = dt_details["effective_dt"]
            dt_source = str(dt_details["source"])
            dt_warning = dt_details["warning"]
            self._last_dt = dt_value
            self._last_dt_source = dt_source if dt_value is not None else None
            self._last_dt_effective = dt_effective if isinstance(dt_effective, (int, float)) else None
            self._last_dt_warning = dt_warning if isinstance(dt_warning, str) else None
        except Exception:
            self._last_dt = None
            self._last_dt_source = None
            self._last_dt_effective = None
            self._last_dt_warning = None
        self._update_project_summary()
        self.frame_start_spin.setValue(project.analysis.frame_start)
        self.frame_stop_spin.setValue(
            project.analysis.frame_stop if project.analysis.frame_stop is not None else -1
        )
        self.frame_stride_spin.setValue(project.analysis.stride)
        if hasattr(self, "workers_spin"):
            workers_val = project.analysis.workers or 0
            self.workers_spin.setValue(int(workers_val))
        self._refresh_output_controls()
        self._refresh_selection_combos()
        self._refresh_soz_combos()
        self._refresh_distance_bridge_table()
        self._refresh_density_table()
        self._refresh_defined_soz_panel()
        self._refresh_soz_combos()
        self._update_explain_text()
        try:
            self._wizard_snapshot = self._wizard_state()
        except Exception:
            pass
        self._update_provenance_stamp()

    def _reset_input_caches(self) -> None:
        self._preflight_universe = None
        self._preflight_key = None
        self._tester_universe = None
        self._tester_key = None

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

    def _on_analysis_settings_changed(self) -> None:
        if not self.state.project:
            return
        if hasattr(self, "workers_spin"):
            workers_val = int(self.workers_spin.value())
            self.state.project.analysis.workers = None if workers_val <= 0 else workers_val

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
        probe_mode = self.wizard_atom_mode.currentText()
        text = f"shell(selection_a, cutoffs=[{', '.join(cutoff_text)}], probe_mode={probe_mode})"
        if self.wizard_seed_b.text().strip():
            text += (
                f" {self._wizard_boolean_value()} "
                f"distance(selection_b, cutoff={self.wizard_b_cutoff.text() or '3.5'}A)"
            )
        self.wizard_explain.setText(text)

    def _parse_probe_selection(self) -> tuple[str, list[str], str]:
        raw = self.wizard_probe_selection.text().strip()
        if not raw:
            return "", [], "wizard_empty"
        lowered = raw.lower()
        selection_keywords = (" and ", " or ", " not ", "name ", "resname ", "segid ", "chain", "index ", "bynum ")
        if lowered in ("all", "protein", "backbone", "sidechain", "nucleic"):
            return raw, [], "wizard_selection"
        if any(keyword in lowered for keyword in selection_keywords):
            return raw, [], "wizard_selection"
        if "," in raw or " " not in raw:
            names = [part.strip() for part in raw.split(",") if part.strip()]
            if names:
                return "name " + " ".join(names), names, "wizard_atom_names"
        return raw, [], "wizard_selection"

    def _toggle_overview_raw(self, visible: bool) -> None:
        self.overview_text.setVisible(visible)

    def _toggle_qc_raw(self, visible: bool) -> None:
        self.qc_text.setVisible(visible)

    def _update_qc_overview(self, qc_json: dict | None) -> None:
        if not hasattr(self, "qc_summary_label"):
            return
        if not qc_json:
            self.qc_summary_label.setText("QC summary will appear after analysis.")
            if hasattr(self, "qc_findings_text"):
                self.qc_findings_text.setPlainText("No QC diagnostics yet.")
            if hasattr(self, "qc_health_headline"):
                self.qc_health_headline.setText("Project health unavailable")
            if hasattr(self, "qc_health_detail"):
                self.qc_health_detail.setText("Run analysis to compute QC checks and diagnostics.")
            if hasattr(self, "qc_health_card"):
                self._set_status_card_tone(self.qc_health_card, "neutral")
            if hasattr(self, "qc_preflight_badge"):
                self._set_status_badge(self.qc_preflight_badge, "Preflight: -", "neutral")
            if hasattr(self, "qc_warning_badge"):
                self._set_status_badge(self.qc_warning_badge, "Warnings: -", "neutral")
            if hasattr(self, "qc_analysis_badge"):
                self._set_status_badge(self.qc_analysis_badge, "Analysis warnings: -", "neutral")
            if hasattr(self, "qc_zero_badge"):
                self._set_status_badge(self.qc_zero_badge, "Zero SOZs: -", "neutral")
            return

        preflight = qc_json.get("preflight", {}) if isinstance(qc_json, dict) else {}
        errors = list(preflight.get("errors", []) or [])
        warnings = list(preflight.get("warnings", []) or [])
        analysis_warnings = list(qc_json.get("analysis_warnings", []) or [])
        zero_sozs = list(qc_json.get("zero_occupancy_sozs", []) or [])
        zero_diag = qc_json.get("zero_occupancy_diagnostics", {}) or {}

        if errors:
            tone = "error"
            headline = "Action required"
        elif warnings or analysis_warnings or zero_sozs:
            tone = "warning"
            headline = "Warnings detected"
        else:
            tone = "success"
            headline = "Healthy"

        if hasattr(self, "qc_health_card"):
            self._set_status_card_tone(self.qc_health_card, tone)
        if hasattr(self, "qc_health_headline"):
            self.qc_health_headline.setText(f"Project health: {headline}")
        if hasattr(self, "qc_health_detail"):
            self.qc_health_detail.setText(
                f"Preflight errors: {len(errors)} | warnings: {len(warnings)} | "
                f"analysis warnings: {len(analysis_warnings)} | zero-occupancy SOZs: {len(zero_sozs)}"
            )

        if hasattr(self, "qc_preflight_badge"):
            preflight_tone = "error" if errors else "success"
            preflight_text = f"Preflight: {len(errors)} errors" if errors else "Preflight: OK"
            self._set_status_badge(self.qc_preflight_badge, preflight_text, preflight_tone)
        if hasattr(self, "qc_warning_badge"):
            warn_tone = "warning" if warnings else "success"
            self._set_status_badge(self.qc_warning_badge, f"Warnings: {len(warnings)}", warn_tone)
        if hasattr(self, "qc_analysis_badge"):
            analysis_tone = "warning" if analysis_warnings else "success"
            self._set_status_badge(
                self.qc_analysis_badge,
                f"Analysis warnings: {len(analysis_warnings)}",
                analysis_tone,
            )
        if hasattr(self, "qc_zero_badge"):
            zero_tone = "warning" if zero_sozs else "success"
            self._set_status_badge(self.qc_zero_badge, f"Zero SOZs: {len(zero_sozs)}", zero_tone)

        self.qc_summary_label.setText(self._format_qc_summary(qc_json))

        findings: list[str] = []
        if errors:
            findings.append("Preflight errors:")
            findings.extend([f"- {entry}" for entry in errors[:6]])
        if warnings:
            findings.append("Preflight warnings:")
            findings.extend([f"- {entry}" for entry in warnings[:6]])
        if analysis_warnings:
            findings.append("Analysis warnings:")
            findings.extend([f"- {entry}" for entry in analysis_warnings[:6]])
        if zero_sozs:
            findings.append("Zero-occupancy SOZs: " + ", ".join(zero_sozs))
        if zero_diag:
            findings.append("Zero-occupancy diagnostics:")
            for name, notes in list(zero_diag.items())[:4]:
                if notes:
                    findings.append(f"- {name}: {notes[0]}")
        if not findings:
            findings.append("No warnings or errors reported.")
        if hasattr(self, "qc_findings_text"):
            self.qc_findings_text.setPlainText("\n".join(findings))

    def _update_project_summary(self) -> None:
        project = self.state.project
        if not project:
            self.project_summary_label.setText("Load a project to view metadata.")
            return
        topo_full = str(project.inputs.topology or "")
        traj_full = str(project.inputs.trajectory or "")
        topo_name = Path(topo_full).name if topo_full else "None"
        traj_name = Path(traj_full).name if traj_full else "No trajectory"
        summary = (
            f"Topology: {topo_name}\n"
            f"Trajectory: {traj_name}\n"
            f"Outputs: {project.outputs.output_dir} ({project.outputs.report_format})"
        )
        self.project_summary_label.setText(summary)
        self.project_summary_label.setToolTip(
            f"Topology: {topo_full or '-'}\n"
            f"Trajectory: {traj_full or '-'}\n"
            f"Output dir: {project.outputs.output_dir}"
        )

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
        analysis_warnings = qc_json.get("analysis_warnings", [])
        if analysis_warnings:
            lines.append(f"Analysis warnings: {len(analysis_warnings)}")
            lines.extend([f"  - {warn}" for warn in analysis_warnings[:5]])
        zero_sozs = qc_json.get("zero_occupancy_sozs", [])
        if zero_sozs:
            lines.append("Zero-occupancy SOZs: " + ", ".join(zero_sozs))
        zero_diag = qc_json.get("zero_occupancy_diagnostics", {})
        if zero_diag:
            lines.append("Zero-occupancy diagnostics:")
            for name, notes in list(zero_diag.items())[:3]:
                if notes:
                    lines.append(f"  - {name}: {notes[0]}")
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
            dt_line = f"dt: {self._last_dt}"
            if self._last_dt_source:
                dt_line += f" [{self._last_dt_source}]"
            lines.append(dt_line)
            if self._last_dt_effective is not None and int(project.analysis.stride) > 1:
                lines.append(
                    f"effective dt (stride {project.analysis.stride}): {self._last_dt_effective}"
                )
            if self._last_dt_warning:
                lines.append(f"dt warning: {self._last_dt_warning}")
        lines.append(f"Frames: start {project.analysis.frame_start} stop {project.analysis.frame_stop or 'end'} stride {project.analysis.stride}")
        lines.append(f"Outputs: {project.outputs.output_dir} | Report: {project.outputs.report_format}")
        lines.append(f"Solvent label: {project.solvent.solvent_label}")
        lines.append(f"Solvent resnames: {', '.join(project.solvent.water_resnames)}")
        lines.append(f"Probe selection: {project.solvent.probe.selection}")
        lines.append(f"Probe position: {project.solvent.probe.position}")
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
                desc.append(f"{pad}  probe_mode: {node.params.get('probe_mode') or node.params.get('atom_mode')}")
                desc.append(f"{pad}  selection: {node.params.get('selection_label')}")
            if node.type == "distance":
                cutoff = node.params.get("cutoff")
                unit = node.params.get("unit", "A")
                desc.append(f"{pad}  cutoff: {cutoff} {unit}")
                desc.append(f"{pad}  probe_mode: {node.params.get('probe_mode') or node.params.get('atom_mode')}")
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
        applied = self._apply_wizard_to_project(update_existing=True)
        self._refresh_project_ui()
        if applied:
            self._refresh_project_doctor_if_initialized()

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
            self._refresh_project_doctor_if_initialized()
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
        if hasattr(self, "workers_spin"):
            workers_val = int(self.workers_spin.value())
            self.state.project.analysis.workers = None if workers_val <= 0 else workers_val

        if not self._maybe_apply_wizard_changes():
            return

        self._strip_removed_analysis_options(self.state.project)
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
        self._strip_removed_analysis_options(project_for_run)
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

        solvent_label = self.wizard_solvent_label.text().strip() or base.solvent.solvent_label
        water_resnames = [x.strip() for x in self.wizard_water_resnames.text().split(",") if x.strip()]
        probe_selection, probe_names, probe_source = self._parse_probe_selection()
        probe_position = self.wizard_probe_position.currentText().strip() or "atom"
        ion_resnames = [x.strip() for x in self.wizard_ion_resnames.text().split(",") if x.strip()]
        water_oxygen = probe_names or base.solvent.water_oxygen_names
        solvent_cfg = SolventConfig(
            solvent_label=solvent_label,
            water_resnames=water_resnames or base.solvent.water_resnames,
            water_oxygen_names=water_oxygen or base.solvent.water_oxygen_names,
            water_hydrogen_names=base.solvent.water_hydrogen_names,
            ion_resnames=ion_resnames or base.solvent.ion_resnames,
            include_ions=self.wizard_include_ions.isChecked(),
            probe=ProbeConfig(selection=probe_selection, position=probe_position),
        )
        solvent_cfg.probe_source = probe_source

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
                "probe_mode": atom_mode,
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
                    "probe_mode": atom_mode,
                },
            )
            combine = self._wizard_boolean_value()
            root_node = SOZNode(type=combine, children=[shell_node, dist_node])

        soz = SOZDefinition(name=soz_name, description="Wizard-generated SOZ", root=root_node)
        return ProjectConfig(
            inputs=base.inputs,
            solvent=solvent_cfg,
            selections=selections,
            sozs=[soz],
            analysis=base.analysis,
            outputs=base.outputs,
            distance_bridges=base.distance_bridges,
            hbond_water_bridges=[],
            hbond_hydration=[],
            density_maps=base.density_maps,
            water_dynamics=[],
            version=base.version,
        )

    def _wizard_state(self) -> dict:
        return {
            "soz_name": self.wizard_soz_name.text().strip() or "SOZ",
            "solvent_label": self.wizard_solvent_label.text().strip(),
            "water_resnames": self.wizard_water_resnames.text().strip(),
            "probe_selection": self.wizard_probe_selection.text().strip(),
            "probe_position": self.wizard_probe_position.currentText(),
            "include_ions": self.wizard_include_ions.isChecked(),
            "ion_resnames": self.wizard_ion_resnames.text().strip(),
            "selection_a": self.wizard_seed_a.text().strip(),
            "selection_a_unique": self.wizard_seed_a_unique.isChecked(),
            "selection_b": self.wizard_seed_b.text().strip(),
            "selection_b_unique": self.wizard_seed_b_unique.isChecked(),
            "shell_cutoffs": self.wizard_shell_cutoffs.text().strip(),
            "atom_mode": self.wizard_atom_mode.currentText(),
            "selection_b_cutoff": self.wizard_b_cutoff.text().strip(),
            "selection_b_combine": self._wizard_boolean_value().upper(),
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

        solvent_label = self.wizard_solvent_label.text().strip() or project.solvent.solvent_label
        water_resnames = [x.strip() for x in self.wizard_water_resnames.text().split(",") if x.strip()]
        probe_selection, probe_names, probe_source = self._parse_probe_selection()
        probe_position = self.wizard_probe_position.currentText().strip() or "atom"
        ion_resnames = [x.strip() for x in self.wizard_ion_resnames.text().split(",") if x.strip()]
        water_oxygen = probe_names or project.solvent.water_oxygen_names
        project.solvent = SolventConfig(
            solvent_label=solvent_label,
            water_resnames=water_resnames or project.solvent.water_resnames,
            water_oxygen_names=water_oxygen or project.solvent.water_oxygen_names,
            water_hydrogen_names=project.solvent.water_hydrogen_names,
            ion_resnames=ion_resnames or project.solvent.ion_resnames,
            include_ions=self.wizard_include_ions.isChecked(),
            probe=ProbeConfig(selection=probe_selection, position=probe_position),
        )
        project.solvent.probe_source = probe_source

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
                "probe_mode": atom_mode,
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
                    "probe_mode": atom_mode,
                },
            )
            combine = self._wizard_boolean_value()
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

    def _sync_wizard_selection_defs_to_project(self) -> None:
        project = self.state.project
        if not project or not hasattr(self, "wizard_soz_name"):
            return

        label_a = self._wizard_selection_label("A")
        label_b = self._wizard_selection_label("B")
        prev_a = getattr(self, "_wizard_synced_selection_a", None)
        prev_b = getattr(self, "_wizard_synced_selection_b", None)

        for stale in (prev_a, prev_b):
            if stale and stale not in {label_a, label_b}:
                project.selections.pop(stale, None)

        selection_a = self.wizard_seed_a.text().strip()
        selection_b = self.wizard_seed_b.text().strip()

        if selection_a:
            project.selections[label_a] = SelectionSpec(
                label=label_a,
                selection=selection_a,
                require_unique=self.wizard_seed_a_unique.isChecked(),
            )
            self._wizard_synced_selection_a = label_a
        else:
            project.selections.pop(label_a, None)
            if prev_a == label_a:
                self._wizard_synced_selection_a = None

        if selection_b:
            project.selections[label_b] = SelectionSpec(
                label=label_b,
                selection=selection_b,
                require_unique=self.wizard_seed_b_unique.isChecked(),
            )
            self._wizard_synced_selection_b = label_b
        else:
            project.selections.pop(label_b, None)
            if prev_b == label_b:
                self._wizard_synced_selection_b = None

        self._refresh_selection_state_ui(refresh_doctor=False)

    def _schedule_seed_validation(self) -> None:
        self._sync_wizard_selection_defs_to_project()
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

    def _run_project_doctor_silent_if_initialized(self) -> None:
        if self._preflight_report is None:
            return
        if getattr(self, "_analysis_running", False):
            return
        self._run_project_doctor(silent=True)

    def _refresh_project_doctor_if_initialized(self) -> None:
        if self._preflight_report is None:
            return
        if getattr(self, "_analysis_running", False):
            return
        timer = getattr(self, "_doctor_refresh_timer", None)
        if timer is None:
            self._run_project_doctor(silent=True)
            return
        timer.start()

    def _run_project_doctor(self, require_ok: bool = False, silent: bool = False) -> bool:
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
            if hasattr(self, "doctor_status_card"):
                self._set_status_card_tone(self.doctor_status_card, "error")
            if hasattr(self, "doctor_errors_badge"):
                self._set_status_badge(self.doctor_errors_badge, "Errors 1", "error")
            if hasattr(self, "doctor_warnings_badge"):
                self._set_status_badge(self.doctor_warnings_badge, "Warnings 0", "neutral")
            if hasattr(self, "doctor_frames_badge"):
                self._set_status_badge(self.doctor_frames_badge, "Frames -", "warning")
                self.doctor_frames_badge.setToolTip("Unable to read trajectory.")
            if hasattr(self, "doctor_pbc_badge"):
                self._set_status_badge(self.doctor_pbc_badge, "PBC unknown", "warning")
                self.doctor_pbc_badge.setToolTip("Unable to evaluate PBC without readable trajectory.")
            if hasattr(self, "doctor_findings_list"):
                self.doctor_findings_list.clear()
                self.doctor_findings_list.addItem(f"Error: {msg}")
            self.doctor_text.setText(msg)
            self.doctor_seed_table.setRowCount(0)
            if not silent:
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
            if not silent:
                self.status_bar.showMessage("Preflight OK. Ready to run.", 5000)
            return True
        if not silent:
            self.status_bar.showMessage("Preflight failed. Fix errors in Project Doctor.", 8000)
        return False if require_ok else report.ok

    def _update_project_doctor_ui(self, report) -> None:
        status = "Healthy" if report.ok else f"{len(report.errors)} errors"
        tone = "success" if report.ok else ("error" if report.errors else "warning")
        self.doctor_status_label.setText(f"Project Doctor: {status}")
        if hasattr(self, "doctor_status_card"):
            self._set_status_card_tone(self.doctor_status_card, tone)

        solvent = report.solvent_summary or {}
        solvent_matches = solvent.get("solvent_matches", [])
        solvent_residue_counts = solvent.get("solvent_residue_counts", {})
        probe = solvent.get("probe_summary", {})
        try:
            if isinstance(solvent_residue_counts, dict) and solvent_residue_counts:
                solvent_total = int(sum(int(v) for v in solvent_residue_counts.values()))
            else:
                solvent_total = int(sum(int(v) for v in solvent_matches))
        except Exception:
            solvent_total = 0
        probe_atom_count = int(probe.get("probe_atom_count", 0) or 0)

        if hasattr(self, "doctor_errors_badge"):
            err_tone = "error" if report.errors else "success"
            self._set_status_badge(
                self.doctor_errors_badge,
                f"Errors {len(report.errors)}",
                err_tone,
            )
        if hasattr(self, "doctor_warnings_badge"):
            warn_tone = "warning" if report.warnings else "success"
            self._set_status_badge(
                self.doctor_warnings_badge,
                f"Warnings {len(report.warnings)}",
                warn_tone,
            )
        if hasattr(self, "doctor_solvent_badge"):
            solvent_tone = "success" if solvent_total > 0 else "warning"
            self._set_status_badge(
                self.doctor_solvent_badge,
                f"Solvent atoms {solvent_total}",
                solvent_tone,
            )
        if hasattr(self, "doctor_probe_badge"):
            probe_tone = "success" if probe_atom_count > 0 else "warning"
            self._set_status_badge(
                self.doctor_probe_badge,
                f"Probe atoms {probe_atom_count}",
                probe_tone,
            )
        trajectory = report.trajectory_summary or {}
        n_frames = trajectory.get("n_frames")
        try:
            n_frames_int = int(n_frames) if n_frames is not None else None
        except Exception:
            n_frames_int = None
        if hasattr(self, "doctor_frames_badge"):
            frames_text = "Frames -" if n_frames_int is None else f"Frames {n_frames_int}"
            frames_tone = "neutral"
            if n_frames_int is not None:
                frames_tone = "success" if n_frames_int > 0 else "error"
            self._set_status_badge(self.doctor_frames_badge, frames_text, frames_tone)
            self.doctor_frames_badge.setToolTip(
                "Number of trajectory frames validated by Project Doctor."
            )
        pbc = report.pbc_summary or {}
        has_box = pbc.get("has_box")
        if hasattr(self, "doctor_pbc_badge"):
            if has_box is True:
                pbc_text = "PBC valid"
                pbc_tone = "success"
            elif has_box is False:
                pbc_text = "PBC missing"
                pbc_tone = "warning"
            else:
                pbc_text = "PBC unknown"
                pbc_tone = "neutral"
            self._set_status_badge(self.doctor_pbc_badge, pbc_text, pbc_tone)
            dims = pbc.get("dimensions")
            self.doctor_pbc_badge.setToolTip(
                f"Box vectors from trajectory: {dims}" if dims is not None else "No trajectory box vectors detected."
            )

        findings: list[tuple[str, str]] = []
        for err in report.errors:
            findings.append(("error", f"Error: {err}"))
        for warn in report.warnings:
            findings.append(("warning", f"Warning: {warn}"))

        missing_count = int(probe.get("probe_residues_missing_count", 0) or 0)
        multi_count = int(probe.get("probe_residues_multi_count", 0) or 0)
        if missing_count:
            findings.append(("warning", f"Probe missing residues: {missing_count}"))
        if multi_count:
            findings.append(("warning", f"Probe multi atoms/residue: {multi_count}"))
        if not findings:
            findings.append(("success", "No blocking issues detected. Project is ready to run."))

        if hasattr(self, "doctor_findings_list"):
            self.doctor_findings_list.clear()
            tokens = self._get_theme_tokens()
            tone_colors = {
                "error": QtGui.QColor(tokens.get("error_text", "#991B1B")),
                "warning": QtGui.QColor(tokens.get("warning_text", "#92400E")),
                "success": QtGui.QColor(tokens.get("success_text", "#166534")),
            }
            for tone_name, text in findings:
                item = QtWidgets.QListWidgetItem(text)
                item.setForeground(QtGui.QBrush(tone_colors.get(tone_name, QtGui.QColor(tokens.get("text", "#111827")))))
                self.doctor_findings_list.addItem(item)

        lines = [f"Status: {status}"]
        if report.errors:
            lines.append("Errors:")
            lines.extend([f"  - {err}" for err in report.errors])
        if report.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {warn}" for warn in report.warnings])
        ion_matches = solvent.get("ion_matches", [])
        lines.append(f"Solvent matches: {', '.join(map(str, solvent_matches)) or 'none'}")
        if solvent.get("include_ions"):
            lines.append(f"Ion matches: {', '.join(map(str, ion_matches)) or 'none'}")
        if probe:
            lines.append(f"Probe selection: {probe.get('selection', '')}")
            lines.append(f"Probe position: {probe.get('position', '')}")
            lines.append(
                "Probe atoms: "
                f"{probe.get('probe_atom_count', 0)} | "
                f"residues w/probe: {probe.get('residues_with_probe', 0)}"
            )
            missing_count = probe.get("probe_residues_missing_count", 0)
            missing_sample = probe.get("probe_residues_missing_sample", [])
            if missing_count:
                lines.append(
                    "Probe missing residues: "
                    f"{missing_count} (sample: {', '.join(map(str, missing_sample))})"
                )
            multi_count = probe.get("probe_residues_multi_count", 0)
            multi_sample = probe.get("probe_residues_multi_sample", [])
            if multi_count:
                lines.append(
                    "Probe multi atoms/residue: "
                    f"{multi_count} (sample: {', '.join(map(str, multi_sample))})"
                )
        lines.append(f"Frames: {n_frames_int if n_frames_int is not None else '-'}")
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
            tokens = self._get_theme_tokens()
            req_bg = QtGui.QColor(tokens.get("success_soft", "#DCFCE7"))
            req_fg = QtGui.QColor(tokens.get("success_text", "#166534"))
            opt_bg = QtGui.QColor(tokens.get("panel", "#F3F4F6"))
            opt_fg = QtGui.QColor(tokens.get("text_muted", "#6B7280"))
            for row, check in enumerate(selection_checks.values()):
                required = bool(check.require_unique)
                values = [
                    check.label,
                    str(check.count),
                    "Required" if required else "Optional",
                    str(check.expect_count) if check.expect_count is not None else "",
                    check.selection,
                    " | ".join(check.suggestions[:3]),
                ]
                for col, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if col == 2:
                        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                        item.setForeground(QtGui.QBrush(req_fg if required else opt_fg))
                        item.setBackground(QtGui.QBrush(req_bg if required else opt_bg))
                    if col in (4, 5):
                        item.setToolTip(str(value))
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

    def _browse_topology(self) -> None:
        if not self.state.project:
            self.status_bar.showMessage("Load a project first.", 4000)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Topology", "", "Topology (*)"
        )
        if not path:
            return
        self.state.project.inputs.topology = path
        self._reset_input_caches()
        self._refresh_project_ui()
        self.status_bar.showMessage("Topology updated. Rerun analysis.", 5000)

    def _browse_trajectory(self) -> None:
        if not self.state.project:
            self.status_bar.showMessage("Load a project first.", 4000)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Trajectory", "", "Trajectory (*)"
        )
        if not path:
            return
        self.state.project.inputs.trajectory = path
        self._reset_input_caches()
        self._refresh_project_ui()
        self.status_bar.showMessage("Trajectory updated. Rerun analysis.", 5000)

    def _clear_topology(self) -> None:
        if not self.state.project:
            self.status_bar.showMessage("Load a project first.", 4000)
            return
        self.state.project.inputs.topology = ""
        self._reset_input_caches()
        self._refresh_project_ui()
        self.status_bar.showMessage("Topology cleared. Rerun analysis.", 5000)

    def _clear_trajectory(self) -> None:
        if not self.state.project:
            self.status_bar.showMessage("Load a project first.", 4000)
            return
        self.state.project.inputs.trajectory = None
        self._reset_input_caches()
        self._refresh_project_ui()
        self.status_bar.showMessage("Trajectory cleared. Rerun analysis.", 5000)

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

    def _test_probe_selection(self) -> None:
        if not self.state.project:
            self.selection_results.setText("Load a project first.")
            return
        if not self.run_logger:
            self.run_logger, self.log_path = setup_run_logger(self.state.project.outputs.output_dir)
        if self.run_logger:
            self.run_logger.info("Probe selection test requested")

        use_traj = self.selection_use_trajectory.isChecked()
        limit = int(self.selection_limit_spin.value())
        universe = self._ensure_tester_universe(use_traj)
        if universe is None:
            self.selection_results.setText("Unable to load inputs for probe test.")
            return

        solvent_label = self.wizard_solvent_label.text().strip() or self.state.project.solvent.solvent_label
        water_resnames = [x.strip() for x in self.wizard_water_resnames.text().split(",") if x.strip()]
        ion_resnames = [x.strip() for x in self.wizard_ion_resnames.text().split(",") if x.strip()]
        probe_selection, probe_names, probe_source = self._parse_probe_selection()
        probe_position = self.wizard_probe_position.currentText().strip() or "atom"
        water_oxygen = probe_names or self.state.project.solvent.water_oxygen_names
        solvent_cfg = SolventConfig(
            solvent_label=solvent_label,
            water_resnames=water_resnames or self.state.project.solvent.water_resnames,
            water_oxygen_names=water_oxygen or self.state.project.solvent.water_oxygen_names,
            water_hydrogen_names=self.state.project.solvent.water_hydrogen_names,
            ion_resnames=ion_resnames or self.state.project.solvent.ion_resnames,
            include_ions=self.wizard_include_ions.isChecked(),
            probe=ProbeConfig(selection=probe_selection, position=probe_position),
        )
        solvent_cfg.probe_source = probe_source

        try:
            solvent = build_solvent(universe, solvent_cfg)
        except Exception as exc:
            self.selection_results.setText(f"Probe selection failed: {exc}")
            self.selection_table.setRowCount(0)
            if self.run_logger:
                self.run_logger.error("Probe selection failed: %s", exc)
            return

        probe_counts = {
            resindex: len(indices)
            for resindex, indices in solvent.probe.resindex_to_atom_indices.items()
        }
        multi = [idx for idx, count in probe_counts.items() if count > 1]
        missing = [idx for idx, count in probe_counts.items() if count == 0]

        lines = [
            f"Probe selection: {solvent.probe.selection}",
            f"Probe position: {solvent.probe.position}",
            f"Solvent residues: {len(solvent.residues)}",
            f"Probe atoms: {len(solvent.probe.atom_indices)}",
            f"Residues with probe: {sum(1 for count in probe_counts.values() if count > 0)}",
        ]
        if multi:
            lines.append(f"Multi-probe residues: {len(multi)} (sample: {', '.join(map(str, multi[:10]))})")
        if missing:
            lines.append(f"Missing probe residues: {len(missing)} (sample: {', '.join(map(str, missing[:10]))})")
        if multi and solvent.probe.position == "atom":
            lines.append("Warning: probe matches multiple atoms per residue.")
        self.selection_results.setText("\n".join(lines))

        rows = []
        for atom in solvent.probe.atoms[:limit]:
            res = atom.residue
            residue_number = getattr(res, "resnum", None)
            if residue_number in (None, ""):
                residue_number = int(res.resid)
            rows.append(
                [
                    int(atom.index),
                    str(res.resname),
                    int(res.resid),
                    residue_number,
                    str(res.segid),
                    getattr(res, "chainID", ""),
                    getattr(res, "moltype", ""),
                    str(atom.name),
                ]
            )
        self.selection_table.setRowCount(0)
        self.selection_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.selection_table.setItem(row, col, item)

    def _ensure_project(self) -> bool:
        if self.state.project:
            self._strip_removed_analysis_options(self.state.project)
            return True
        topology, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Topology", "", "Topology (*)")
        if not topology:
            return False
        trajectory, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Trajectory (optional)", "", "Trajectory (*)"
        )
        project = default_project(topology, trajectory or None)
        self._strip_removed_analysis_options(project)
        self.state = ProjectState(project=project, path=None)
        self._refresh_project_ui()
        return True

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._run_progress_current = int(current)
        self._run_progress_total = int(total) if total > 0 else None
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
        if self._run_progress_total:
            self._on_progress(
                max(self._run_progress_total - 1, 0),
                self._run_progress_total,
                "Writing outputs...",
            )
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
        if self._run_progress_total:
            self._on_progress(self._run_progress_total, self._run_progress_total, "Finalizing outputs...")
        self._set_run_ui_state(False)
        self._refresh_project_doctor_if_initialized()
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
        if self.run_logger:
            self.run_logger.info(
                "Updating GUI with %d SOZ results",
                len(self.current_result.soz_results),
            )
        if self.current_result.qc_summary is not None:
            qc_json = to_jsonable(self.current_result.qc_summary)
            self.qc_text.setText(json.dumps(qc_json, indent=2))
            self._update_qc_overview(qc_json)
        else:
            self._update_qc_overview(None)
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
        self._refresh_defined_soz_panel()

        for fn, name in (
            (self._update_overview, "overview"),
            (self._update_timeline_plot, "timeline"),
            (self._update_hist_plot, "histogram"),
            (self._update_event_plot, "event_plot"),
            (self._update_tables_for_selected_soz, "tables"),
            (self._update_bridge_explore, "bridge_explore"),
            (self._update_density_explore, "density_explore"),
        ):
            try:
                fn()
            except Exception as exc:
                if self.run_logger:
                    self.run_logger.exception("GUI update failed: %s", name)
                self.status_bar.showMessage(f"GUI update failed ({name}): {exc}", 8000)
        self._apply_compact_control_sizing()
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

    def _update_tables_for_selected_soz(self) -> None:
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        soz = self.current_result.soz_results[soz_name]
        per_frame = self._filtered_per_frame(soz.per_frame)
        self._set_table(self.per_frame_table, per_frame)
        self._set_table(self.per_solvent_table, soz.per_solvent)

    def _update_bridge_explore(self) -> None:
        if not self.current_result:
            return
        distance_names = list(self.current_result.distance_bridge_results.keys())
        self.distance_bridge_combo.blockSignals(True)
        self.distance_bridge_combo.clear()
        self.distance_bridge_combo.addItems(distance_names)
        if distance_names:
            self.distance_bridge_combo.setCurrentIndex(0)
        self.distance_bridge_combo.blockSignals(False)
        self._update_distance_bridge_plots()

    def _bridge_time_axis(self, per_frame: pd.DataFrame) -> tuple[np.ndarray, str]:
        if "time" in per_frame.columns:
            time_ns = pd.to_numeric(per_frame["time"], errors="coerce").to_numpy() / 1000.0
            return time_ns, "Time (ns)"
        frames = pd.to_numeric(per_frame.get("frame", []), errors="coerce").to_numpy()
        return frames, "Frame"

    def _update_distance_bridge_plots(self) -> None:
        self.distance_bridge_timeseries_plot.clear()
        self.distance_bridge_residence_plot.clear()
        self.distance_bridge_top_plot.clear()

        # [UX Improvement] Helper for consistent titles
        def set_title(plot, title, caption):
            plot.setTitle(
                "<div style='text-align:center'>"
                f"<span style='font-size: 10pt; font-weight: bold'>{title}</span><br>"
                f"<span style='font-size: 8pt; color: #a0a0a0'>{caption}</span>"
                "</div>"
            )

        if not self.current_result or not self.current_result.distance_bridge_results:
            self.distance_bridge_timeseries_plot.addItem(
                pg.TextItem("No distance bridge results.", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_distance", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        name = self.distance_bridge_combo.currentText()
        if not name:
            name = next(iter(self.current_result.distance_bridge_results.keys()), "")
        bridge = self.current_result.distance_bridge_results.get(name)
        if bridge is None:
            self.distance_bridge_timeseries_plot.addItem(
                pg.TextItem("No data for selected bridge.", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_distance", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        if bridge.per_frame.empty:
            self.distance_bridge_timeseries_plot.addItem(
                pg.TextItem("Bridge has no frames (check selections).", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_distance", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        
        set_title(self.distance_bridge_timeseries_plot, "Distance Bridge", "Number of bridging solvent residues over time")
        set_title(self.distance_bridge_residence_plot, "Residence Time Distribution", "How long bridging events persist (Survival Probability)")
        set_title(self.distance_bridge_top_plot, "Top Bridging Residues", "Solvent residues with highest bridging occupancy")

        x, x_label = self._bridge_time_axis(bridge.per_frame)
        y = pd.to_numeric(bridge.per_frame["n_solvent"], errors="coerce").fillna(0).to_numpy()
        if self.distance_bridge_smooth_check.isChecked():
            window = int(self.distance_bridge_smooth_window.value() or 1)
            y = pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()
        self.distance_bridge_timeseries_plot.setLabel("bottom", x_label)
        self.distance_bridge_timeseries_plot.plot(
            x,
            y,
            pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
        )

        durations = []
        dt = float(bridge.summary.get("dt", 1.0))
        time_unit = bridge.summary.get("time_unit", "ps")
        scale_to_ns = 1.0
        if time_unit == "ps":
            scale_to_ns = 1e-3
        elif time_unit == "fs":
            scale_to_ns = 1e-6
            
        for lengths in bridge.residence_cont.values():
            durations.extend([length * dt * scale_to_ns for length in lengths])
        if durations:
            durations = np.array(durations, dtype=float)
            durations.sort()
            survival = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
            # [UX Improvement] Explicit units
            self.distance_bridge_residence_plot.setLabel("bottom", "Residence Time (ns)")
            self.distance_bridge_residence_plot.plot(
                durations,
                survival,
                pen=pg.mkPen(self._get_theme_tokens()["accent_alt"], width=self._plot_line_width),
                stepMode=False,
            )
            self.distance_bridge_residence_plot.autoRange()
        else:
            self.distance_bridge_residence_plot.addItem(pg.TextItem("No residence events found", anchor=(0.5, 0.5)))

        per_solvent = bridge.per_solvent.head(10)
        if not per_solvent.empty:
            values = per_solvent["occupancy_pct"].to_numpy()
            y_pos = np.arange(len(values))
            bar = pg.BarGraphItem(
                x0=0,
                y=y_pos,
                height=0.6,
                width=values,
                brush=self._get_theme_tokens()["accent"],
            )
            self.distance_bridge_top_plot.addItem(bar)
            axis = self.distance_bridge_top_plot.getAxis("left")
            labels = [
                (idx, sid) for idx, sid in enumerate(per_solvent["solvent_id"].tolist())
            ]
            axis.setTicks([labels])
            self.distance_bridge_top_plot.autoRange()
        else:
            self.distance_bridge_top_plot.addItem(pg.TextItem("No top bridges found", anchor=(0.5, 0.5)))
        top3 = ", ".join(
            f"{str(row.solvent_id)}:{float(row.occupancy_pct):.1f}%"
            for row in bridge.per_solvent.head(3).itertuples()
        )
        self._set_plot_insights(
            "bridge_distance",
            [
                f"n={len(y)}",
                f"mean={float(np.mean(y)):.3f}",
                f"max={float(np.max(y)):.3f}",
                f"top3={top3 or 'none'}",
            ],
        )

    def _update_hbond_bridge_plots(self) -> None:
        self.hbond_bridge_timeseries_plot.clear()
        self.hbond_bridge_residence_plot.clear()
        self.hbond_bridge_top_plot.clear()
        self.bridge_compare_plot.clear()
        self.hbond_bridge_network_plot.clear()
        
        def set_title(plot, title, caption):
             plot.setTitle(f"<span style='font-size: 10pt; font-weight: bold'>{title}</span><br><span style='font-size: 8pt; color: #a0a0a0'>{caption}</span>")
             
        if not self.current_result or not self.current_result.hbond_bridge_results:
            self.hbond_bridge_timeseries_plot.addItem(
                pg.TextItem("No H-bond bridge results.", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_hbond", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        name = self.hbond_bridge_combo.currentText()
        bridge = self.current_result.hbond_bridge_results.get(name)
        if bridge is None:
            self.hbond_bridge_timeseries_plot.addItem(
                pg.TextItem("No data for selected bridge.", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_hbond", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        if bridge.per_frame.empty:
            self.hbond_bridge_timeseries_plot.addItem(
                pg.TextItem("Bridge has no frames (check selections).", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("bridge_hbond", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
            
        set_title(self.hbond_bridge_timeseries_plot, f"H-bond Bridge: {name}", "Number of bridging waters per frame (H-bond definition)")
        set_title(self.hbond_bridge_residence_plot, "Residence Time Distribution", "How long waters bridge continuously (Survival Probability)")
        set_title(self.hbond_bridge_top_plot, "Top Bridging Waters", "Waters with highest bridging occupancy")
        set_title(self.bridge_compare_plot, "Comparator: Distance vs H-bond", "Overlay of Distance (geometric) and H-bond (energetic) definitions")
        set_title(self.hbond_bridge_network_plot, "Bridge Network", "Connectivity between selection groups via water")

        x, x_label = self._bridge_time_axis(bridge.per_frame)
        y = pd.to_numeric(bridge.per_frame["n_solvent"], errors="coerce").fillna(0).to_numpy()
        if self.hbond_bridge_smooth_check.isChecked():
            window = int(self.hbond_bridge_smooth_window.value() or 1)
            y = pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()
        self.hbond_bridge_timeseries_plot.setLabel("bottom", x_label)
        self.hbond_bridge_timeseries_plot.plot(
            x,
            y,
            pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
            name="H-bond",
        )

        durations = []
        dt = float(bridge.summary.get("dt", 1.0))
        durations = []
        dt = float(bridge.summary.get("dt", 1.0))
        time_unit = bridge.summary.get("time_unit", "ps")
        scale_to_ns = 1.0
        if time_unit == "ps":
            scale_to_ns = 1e-3
        elif time_unit == "fs":
            scale_to_ns = 1e-6

        for lengths in bridge.residence_cont.values():
            durations.extend([length * dt * scale_to_ns for length in lengths])
        if durations:
            durations = np.array(durations, dtype=float)
            durations.sort()
            survival = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
            # [UX Improvement] Explicit units
            self.hbond_bridge_residence_plot.setLabel("bottom", "Residence Time (ns)")
            self.hbond_bridge_residence_plot.plot(
                durations,
                survival,
                pen=pg.mkPen(self._get_theme_tokens()["accent_alt"], width=self._plot_line_width),
                stepMode=False,
            )
            self.hbond_bridge_residence_plot.autoRange()
        else:
            self.hbond_bridge_residence_plot.addItem(pg.TextItem("No residence events found", anchor=(0.5, 0.5)))

        per_solvent = bridge.per_solvent.head(10)
        if not per_solvent.empty:
            values = per_solvent["occupancy_pct"].to_numpy()
            y_pos = np.arange(len(values))
            bar = pg.BarGraphItem(
                x0=0,
                y=y_pos,
                height=0.6,
                width=values,
                brush=self._get_theme_tokens()["accent"],
            )
            self.hbond_bridge_top_plot.addItem(bar)
            axis = self.hbond_bridge_top_plot.getAxis("left")
            labels = [
                (idx, sid) for idx, sid in enumerate(per_solvent["solvent_id"].tolist())
            ]
            axis.setTicks([labels])
            self.hbond_bridge_top_plot.autoRange()
        else:
            self.hbond_bridge_top_plot.addItem(pg.TextItem("No top bridges found", anchor=(0.5, 0.5)))

        distance_name = self.distance_bridge_combo.currentText()
        distance_bridge = (
            self.current_result.distance_bridge_results.get(distance_name)
            if self.current_result
            else None
        )
        if self.bridge_compare_check.isChecked() and distance_bridge and not distance_bridge.per_frame.empty:
            x_d, _ = self._bridge_time_axis(distance_bridge.per_frame)
            y_d = (
                pd.to_numeric(distance_bridge.per_frame["n_solvent"], errors="coerce")
                .fillna(0)
                .to_numpy()
            )
            self.bridge_compare_plot.plot(
                x_d,
                y_d,
                pen=pg.mkPen(self._get_theme_tokens()["accent_alt"], width=self._plot_line_width),
                name="Distance",
            )
        self.bridge_compare_plot.plot(
            x,
            y,
            pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
            name="H-bond",
        )

        if bridge.edge_list is None or bridge.edge_list.empty:
            self.hbond_bridge_network_plot.addItem(
                pg.TextItem("Edge list unavailable.", color=self._get_theme_tokens()["text_muted"])
            )
        else:
            edge_df = bridge.edge_list.sort_values(by="frames_present", ascending=False).head(30)
            nodes = list(dict.fromkeys(edge_df["source"].tolist() + edge_df["target"].tolist()))
            if nodes:
                angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
                pos = np.column_stack((np.cos(angles), np.sin(angles)))
                node_index = {node: idx for idx, node in enumerate(nodes)}
                edges = np.array(
                    [
                        [node_index[src], node_index[tgt]]
                        for src, tgt in edge_df[["source", "target"]].itertuples(index=False)
                    ],
                    dtype=int,
                )
                graph = pg.GraphItem()
                graph.setData(pos=pos, adj=edges, size=12, symbol="o", pen=pg.mkPen("#94a3b8"))
                self.hbond_bridge_network_plot.addItem(graph)
                for node, (xv, yv) in zip(nodes, pos):
                    text = pg.TextItem(str(node), anchor=(0.5, -0.2))
                    text.setPos(xv, yv)
                    self.hbond_bridge_network_plot.addItem(text)
        top3 = ", ".join(
            f"{str(row.solvent_id)}:{float(row.occupancy_pct):.1f}%"
            for row in bridge.per_solvent.head(3).itertuples()
        )
        self._set_plot_insights(
            "bridge_hbond",
            [
                f"n={len(y)}",
                f"mean={float(np.mean(y)):.3f}",
                f"max={float(np.max(y)):.3f}",
                f"top3={top3 or 'none'}",
            ],
        )

    def _update_hydration_setup(self) -> None:
        """Auto-select populated hydration mode if needed."""
        # No-op since only H-bond hydration remains
        pass

    def _refresh_hydration_combo(self) -> Dict[str, object]:
        if not self.current_result:
            return {}
        results = self.current_result.hbond_hydration_results
        names = list(results.keys())
        self.hydration_config_combo.blockSignals(True)
        self.hydration_config_combo.clear()
        self.hydration_config_combo.addItems(names)
        if names:
            self.hydration_config_combo.setCurrentIndex(0)
        self.hydration_config_combo.blockSignals(False)
        return results

    def _update_hydration_plots(self) -> None:
        self.hydration_frequency_plot.clear()
        self.hydration_top_plot.clear()
        self.hydration_timeline_plot.clear()
        if not self.current_result:
            return
        results = self._refresh_hydration_combo()
        if not results:
            text = pg.TextItem("No hydration results available.", color=self._get_theme_tokens()["text_muted"], anchor=(0.5, 0.5))
            self.hydration_frequency_plot.addItem(text)
            self.hydration_frequency_plot.autoRange()
            self.hydration_top_plot.addItem(pg.TextItem("No data", anchor=(0.5, 0.5)))
            self._set_plot_insights("hydration", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        name = self.hydration_config_combo.currentText()
        hydration = results.get(name)
        if hydration is None or hydration.table.empty:
            text = pg.TextItem("No data for this configuration.", color=self._get_theme_tokens()["text_muted"], anchor=(0.5, 0.5))
            self.hydration_frequency_plot.addItem(text)
            self.hydration_frequency_plot.autoRange()
            self.hydration_top_plot.addItem(pg.TextItem("No data", anchor=(0.5, 0.5)))
            self._set_plot_insights("hydration", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        mode_label = "H-bond Hydration"
        metric = self.hydration_metric_combo.currentText()
        try:
            # [UX Improvement] Interpretive captions
            def set_title(plot, title, caption):
                 plot.setTitle(f"<span style='font-size: 10pt; font-weight: bold'>{title}</span><br><span style='font-size: 8pt; color: #a0a0a0'>{caption}</span>")

            set_title(self.hydration_frequency_plot, f"{mode_label}: Frequency Profile", "Probability of solvent contact per residue")
            set_title(self.hydration_top_plot, f"{mode_label}: Top Residues", "Residues with highest solvent contact frequency")
            set_title(self.hydration_timeline_plot, f"{mode_label}: Contact Timeline", "Frame-by-frame contact status (selected residue)")
        except Exception as e:
            # Fallback if HTML fails or something else goes wrong
            print(f"Title update failed: {e}")
            self.hydration_frequency_plot.setTitle(f"{mode_label}: Frequency")
            self.hydration_top_plot.setTitle(f"{mode_label}: Top Residues")
            self.hydration_timeline_plot.setTitle(f"{mode_label}: Timeline")
        tooltip = (
            f"{mode_label} ({metric}) — "
            f"{'SOZ-conditioned' if metric == 'freq_given_soz' else 'unconditioned total'}."
        )
        self.hydration_frequency_plot.setToolTip(tooltip)
        self.hydration_top_plot.setToolTip(tooltip)
        self.hydration_timeline_plot.setToolTip(
            f"{mode_label} timeline ({metric}) — contact presence per frame."
        )
        freq = hydration.table[metric].to_numpy(dtype=float)
        resid = hydration.table["resid"].to_numpy(dtype=int)
        resindex = hydration.table["resindex"].to_numpy(dtype=int)
        resname = hydration.table["resname"].astype(str).to_numpy()

        # [Visualization Upgrade] Bar Chart with Biochemical Coloring
        
        # Define Biochemical Colors
        # Hydrophobic (A, V, L, I, M, F, W, P, G): Greenish/Gray
        # Polar (S, T, C, Y, N, Q): Purple/Cyan
        # Basic (K, R, H): Blue
        # Acidic (D, E): Red
        
        colors = []
        residue_brushes = []
        
        theme = self._get_theme_tokens()
        c_hydrophobic = pg.mkColor(100, 150, 100)
        c_polar = pg.mkColor(100, 200, 200)
        c_basic = pg.mkColor(100, 100, 250)
        c_acidic = pg.mkColor(250, 100, 100)
        c_default = pg.mkColor(theme["accent"])

        for rname in resname:
            r = rname.upper()
            if r in ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"]:
                residue_brushes.append(c_hydrophobic)
            elif r in ["SER", "THR", "CYS", "TYR", "ASN", "GLN"]:
                residue_brushes.append(c_polar)
            elif r in ["LYS", "ARG", "HIS"]:
                residue_brushes.append(c_basic)
            elif r in ["ASP", "GLU"]:
                residue_brushes.append(c_acidic)
            else:
                residue_brushes.append(c_default)

        x_vals = np.arange(len(resid))
        
        # Remove old items
        self.hydration_frequency_plot.clear()
        
        # Add Legend/Key? (Maybe tooltip is enough)
        
        bar_item = pg.BarGraphItem(
            x=x_vals,
            height=freq,
            width=0.6, # Slightly thinner bars
            brushes=residue_brushes
        )
        self.hydration_frequency_plot.addItem(bar_item)
        
        # Make interactive (Click to select) via invisible scatter overlay
        spots = []
        for idx, (xv, yv) in enumerate(zip(x_vals, freq)):
            spots.append({
                "pos": (float(xv), float(yv)),
                "data": {"resindex": int(resindex[idx]), "resid": int(resid[idx]), "resname": str(resname[idx])},
                "size": 14, # Larger hit area
                "pen": None,
                "brush": (0,0,0,0) # Invisible
            })
        
        self.hydration_scatter.setData(spots)
        self.hydration_frequency_plot.addItem(self.hydration_scatter)
        
        # Set x-axis ticks
        axis = self.hydration_frequency_plot.getAxis("bottom")
        if len(resid) <= 50:
            ticks = [(i, f"{n}{r}") for i, n, r in zip(x_vals, resname, resid)]
            axis.setTicks([ticks])
        else:
            step = max(1, len(resid) // 30)
            ticks = [(i, f"{n}{r}") for i, n, r in zip(x_vals[::step], resname[::step], resid[::step])]
            axis.setTicks([ticks])
            
        if len(resid) < 10:
             # Fix scaling for small N to avoid "mega-bar" filling the screen
             self.hydration_frequency_plot.setXRange(-1, max(len(resid), 5))
        else:
             self.hydration_frequency_plot.autoRange()

        top = hydration.table.sort_values(by=metric, ascending=False).head(10)
        if not top.empty:
            values = top[metric].to_numpy()
            y_pos = np.arange(len(values))
            bar = pg.BarGraphItem(
                x0=0,
                y=y_pos,
                height=0.6,
                width=values,
                brush=self._get_theme_tokens()["accent"],
            )
            self.hydration_top_plot.addItem(bar)
            axis = self.hydration_top_plot.getAxis("left")
            labels = [
                (idx, f"{row.resname}{row.resid}")
                for idx, row in enumerate(top.itertuples())
            ]
            axis.setTicks([labels])

        self._current_hydration_result = hydration
        if not hasattr(self, "_hydration_selected_resindex"):
            self._hydration_selected_resindex = int(resindex[0]) if len(resindex) else None
        self._update_hydration_timeline(hydration)
        top_rows = hydration.table.sort_values(by=metric, ascending=False).head(3)
        top_text = ", ".join(
            f"{str(r.resname)}{int(r.resid)}:{float(getattr(r, metric)):.3f}"
            for r in top_rows.itertuples()
        )
        self._set_plot_insights(
            "hydration",
            [
                f"n={len(freq)}",
                f"mean={float(np.mean(freq)):.3f}",
                f"max={float(np.max(freq)):.3f}",
                f"top3={top_text or 'none'}",
            ],
        )

    def _on_hydration_point_clicked(self, _, points) -> None:
        if not points:
            return
        data = points[0].data()
        if not data:
            return
        self._hydration_selected_resindex = data.get("resindex")
        if hasattr(self, "_current_hydration_result"):
            self._update_hydration_timeline(self._current_hydration_result)

    def _update_hydration_timeline(self, hydration) -> None:
        self.hydration_timeline_plot.clear()
        if self._hydration_selected_resindex is None:
            return
        if self.hydration_metric_combo.currentText() == "freq_given_soz":
            frames = hydration.contact_frames_given_soz.get(self._hydration_selected_resindex, [])
        else:
            frames = hydration.contact_frames_total.get(self._hydration_selected_resindex, [])
        n_frames = len(hydration.frame_times)
        presence = np.zeros(n_frames, dtype=float)
        for idx in frames:
            if 0 <= idx < n_frames:
                presence[idx] = 1.0
        time_ns = np.array(hydration.frame_times, dtype=float) / 1000.0
        self.hydration_timeline_plot.plot(
            time_ns,
            presence,
            pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
            stepMode=False,
            symbol="o",
            symbolSize=4,
            symbolBrush=self._get_theme_tokens()["accent"],
        )

    def _update_density_plots(self) -> None:
        if not self.current_result or not self.current_result.density_results:
            for img in getattr(self, "density_images", {}).values():
                img.clear()
            self._set_plot_insights("density", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        name = self.density_combo.currentText()
        if not name:
            name = next(iter(self.current_result.density_results.keys()), "")
        density = self.current_result.density_results.get(name)
        if density is None or density.grid is None:
            self._set_plot_insights("density", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        raw_grid = np.asarray(density.grid, dtype=float)
        if raw_grid.ndim != 3 or raw_grid.size == 0:
            self._set_plot_insights("density", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return

        view_mode = self.density_explore_view_combo.currentText()
        rho_bulk = max(float(density.metadata.get("rho_bulk_approx", 0.0334) or 0.0334), 1e-6)

        grid = np.nan_to_num(raw_grid, nan=0.0, posinf=0.0, neginf=0.0)
        if view_mode == "relative":
            grid = grid / rho_bulk
        elif view_mode == "score":
            safe_grid = np.maximum(grid, 1e-6)
            grid = -np.log(safe_grid / rho_bulk)

        shape = grid.shape
        self.density_slice_x.setRange(0, max(shape[0] - 1, 0))
        self.density_slice_y.setRange(0, max(shape[1] - 1, 0))
        self.density_slice_z.setRange(0, max(shape[2] - 1, 0))

        view_key = (name, view_mode, tuple(shape))
        if view_key != getattr(self, "_density_view_key", None):
            self._density_view_key = view_key
            for spin, idx in (
                (self.density_slice_x, shape[0] // 2),
                (self.density_slice_y, shape[1] // 2),
                (self.density_slice_z, shape[2] // 2),
            ):
                spin.blockSignals(True)
                spin.setValue(idx)
                spin.blockSignals(False)

        x_idx = min(self.density_slice_x.value(), shape[0] - 1)
        y_idx = min(self.density_slice_y.value(), shape[1] - 1)
        z_idx = min(self.density_slice_z.value(), shape[2] - 1)
        slices = {
            "xy": grid[:, :, z_idx],
            "xz": grid[:, y_idx, :],
            "yz": grid[x_idx, :, :],
            "max_projection": grid.max(axis=2),
        }

        cmap_name = self.density_cmap_combo.currentText()
        levels = (float(np.nanmin(grid)), float(np.nanmax(grid))) if grid.size else (0.0, 1.0)
        if levels[1] <= levels[0]:
            levels = (levels[0], levels[0] + 1e-6)

        cmap_obj = None
        lut = None
        try:
            cmap_obj = pg.colormap.get(cmap_name)
            self.density_hist.gradient.loadPreset(cmap_name)
            lut = self.density_hist.gradient.getLookupTable(256)
        except Exception:
            if cmap_obj is not None:
                try:
                    lut = cmap_obj.getLookupTable(0.0, 1.0, 256)
                except Exception:
                    lut = None

        first_img = next(iter(self.density_images.values()), None)
        if first_img is not None:
            self.density_hist.setImageItem(first_img)
            self.density_hist.setLevels(*levels)

        if hasattr(self, "density_3d_widget") and self.density_3d_widget:
            spacing = float(density.metadata.get("grid_spacing", 1.0))
            origin = np.array(density.metadata.get("origin", [0, 0, 0]), dtype=float)
            self.density_3d_widget.set_colormap(cmap_obj)
            self.density_3d_widget.set_data(grid, spacing, origin, view_mode=view_mode)
            atoms, structure_key = self._get_density_structure_atoms(density)
            if atoms is not None and structure_key != getattr(self, "_density_widget_structure_key", None):
                self.density_3d_widget.set_structure(atoms)
                self._density_widget_structure_key = structure_key

        if hasattr(self, "density_summary") and self.density_summary is not None:
            min_val = float(np.nanmin(grid))
            max_val = float(np.nanmax(grid))
            max_idx = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
            spacing = float(density.metadata.get("grid_spacing", 1.0))
            summary_text = (
                f"<h3>Density Map Analysis: {name}</h3>"
                f"<p><b>Data Range:</b> {min_val:.3f} — {max_val:.3f} (probability density)</p>"
                f"<p><b>Map Dimensions:</b> {shape[0]}x{shape[1]}x{shape[2]} grid points ({spacing:.3f} A spacing)</p>"
                f"<p><b>Highest Density:</b> {max_val:.3f} at index {max_idx}</p>"
                "<p><i>Higher values indicate regions where solvent is more likely to be found. "
                "Use the sliders to explore different slices through the volume.</i></p>"
            )
            self.density_summary.setHtml(summary_text)

        axes = density.axes or {}
        captions = {
            "xy": "Cross-section in XY plane (Z-axis slice)",
            "xz": "Cross-section in XZ plane (Y-axis slice)",
            "yz": "Cross-section in YZ plane (X-axis slice)",
            "max_projection": "Maximum density projected through volume",
        }

        for key, plot in self.density_plots.items():
            title = "Max Projection" if key == "max_projection" else f"{key.upper()} Slice"
            caption = captions.get(key, "Density map visualization")
            try:
                plot.setTitle(
                    f"<span style='font-size: 10pt; font-weight: 600'>{title}</span><br>"
                    f"<span style='font-size: 8pt; color: #94a3b8'>{caption}</span>"
                )
            except Exception:
                pass

        for key, data in slices.items():
            img = self.density_images.get(key)
            if img is None:
                continue
            img.setImage(data.T, autoLevels=False)
            img.setLevels(levels)
            if lut is not None:
                img.setLookupTable(lut)
            try:
                if key == "xy":
                    x_axis = axes.get("x", np.arange(data.shape[0]))
                    y_axis = axes.get("y", np.arange(data.shape[1]))
                elif key == "xz":
                    x_axis = axes.get("x", np.arange(data.shape[0]))
                    y_axis = axes.get("z", np.arange(data.shape[1]))
                elif key == "yz":
                    x_axis = axes.get("y", np.arange(data.shape[0]))
                    y_axis = axes.get("z", np.arange(data.shape[1]))
                else:
                    x_axis = axes.get("x", np.arange(data.shape[0]))
                    y_axis = axes.get("y", np.arange(data.shape[1]))

                if len(x_axis) > 1 and len(y_axis) > 1:
                    rect = QtCore.QRectF(
                        float(x_axis[0]),
                        float(y_axis[0]),
                        float(x_axis[-1] - x_axis[0]),
                        float(y_axis[-1] - y_axis[0]),
                    )
                    img.setRect(rect)
            except Exception:
                pass

            plot = self.density_plots.get(key)
            if plot:
                if key == "xy":
                    plot.setLabel("bottom", "X", units="Å")
                    plot.setLabel("left", "Y", units="Å")
                    plot.setTitle(f"XY Slice (Z index={z_idx})")
                elif key == "xz":
                    plot.setLabel("bottom", "X", units="Å")
                    plot.setLabel("left", "Z", units="Å")
                    plot.setTitle(f"XZ Slice (Y={y_idx})")
                elif key == "yz":
                    plot.setLabel("bottom", "Y", units="Å")
                    plot.setLabel("left", "Z", units="Å")
                    plot.setTitle(f"YZ Slice (X={x_idx})")
                elif key == "max_projection":
                    plot.setLabel("bottom", "X", units="Å")
                    plot.setLabel("left", "Y", units="Å")
                    plot.setTitle("Max Projection")

        if self.density_overlay_check.isChecked():
            self._update_density_overlay(density)
        else:
            self._clear_density_overlay()
        flat = np.ravel(grid)
        top_idx = np.argsort(flat)[::-1][:3]
        top_text = ", ".join(
            f"{np.unravel_index(int(i), grid.shape)}:{float(flat[int(i)]):.3f}" for i in top_idx
        )
        self._set_plot_insights(
            "density",
            [
                f"n={int(grid.size)}",
                f"mean={float(np.mean(grid)):.3f}",
                f"max={float(np.max(grid)):.3f}",
                f"top3={top_text or 'none'}",
            ],
        )

    def _center_density_slices(self) -> None:
        for spin in (self.density_slice_x, self.density_slice_y, self.density_slice_z):
            spin.blockSignals(True)
            spin.setValue(max(0, spin.maximum() // 2))
            spin.blockSignals(False)
        self._queue_density_update()

    def _get_density_structure_atoms(self, density):
        align_selection = density.metadata.get("align_selection") or "protein and name CA"
        align_enabled = bool(density.metadata.get("align", False))
        align_reference = str(density.metadata.get("align_reference", "first_frame") or "first_frame")
        align_reference_path = str(density.metadata.get("align_reference_path", "") or "").strip()
        frame_start = density.metadata.get("frame_start", None)
        frame_stop = density.metadata.get("frame_stop", None)
        frame_stride = density.metadata.get("stride", None)
        project = self.run_project or self.state.project
        if not project:
            return None, None
        solvent_cfg = getattr(project, "solvent", None)
        excluded_resnames = ["SOL", "WAT", "TIP3", "HOH", "NA", "CL", "K", "CA", "MG"]
        for key in ("water_resnames", "ion_resnames"):
            values = getattr(solvent_cfg, key, []) if solvent_cfg is not None else []
            for value in values:
                token = str(value or "").strip()
                if token:
                    excluded_resnames.append(token)
        if excluded_resnames:
            default_structure_selection = (
                "protein or (not protein and not resname "
                + " ".join(sorted(set(excluded_resnames)))
                + ")"
            )
        else:
            default_structure_selection = "all"
        requested_selection = (
            str(density.metadata.get("structure_selection", "") or "").strip() or default_structure_selection
        )

        cache_key = (
            str(project.inputs.topology),
            str(project.inputs.trajectory or ""),
            requested_selection,
            align_selection,
            align_enabled,
            align_reference,
            align_reference_path,
            frame_start,
            frame_stop,
            frame_stride,
        )
        if getattr(self, "_structure_cache_key", None) == cache_key:
            return getattr(self, "_structure_cache", {}).get("atoms"), cache_key

        try:
            import MDAnalysis as mda
            from MDAnalysis.analysis import align as align_module

            if project.inputs.trajectory:
                universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
            else:
                universe = mda.Universe(project.inputs.topology)

            if align_enabled and project.inputs.trajectory and align_selection:
                ref = universe
                if align_reference == "structure" and align_reference_path:
                    try:
                        ref = mda.Universe(align_reference_path)
                    except Exception as exc:
                        logger.warning(
                            "Density overlay reference load failed (%s); using first frame as reference.",
                            exc,
                        )
                        ref = universe
                align_kwargs = {"select": align_selection, "in_memory": True}
                try:
                    aligner = align_module.AlignTraj(universe, ref, **align_kwargs)
                except Exception:
                    aligner = align_module.AlignTraj(universe, ref, weights=None, **align_kwargs)
                aligner.run(start=frame_start, stop=frame_stop, step=frame_stride)
                try:
                    target_frame = int(frame_start) if frame_start is not None else 0
                    universe.trajectory[target_frame]
                except Exception:
                    pass

            max_pdb_atoms = 99999
            candidates = [
                requested_selection,
                default_structure_selection,
                "protein",
                "all",
                align_selection,
                "protein and name CA",
                "protein and backbone",
                "backbone",
            ]
            seen = set()
            atoms = None
            used_selection = None
            for sel in candidates:
                sel = (sel or "").strip()
                if not sel or sel in seen:
                    continue
                seen.add(sel)
                try:
                    attempt = universe.select_atoms(sel)
                except Exception:
                    continue
                n_attempt = int(getattr(attempt, "n_atoms", 0) or 0)
                if n_attempt <= 0:
                    continue
                if n_attempt > max_pdb_atoms:
                    logger.warning(
                        "Structure overlay selection '%s' matched %d atoms (PDB limit %d); trying fallback.",
                        sel,
                        n_attempt,
                        max_pdb_atoms,
                    )
                    continue
                if n_attempt > 0:
                    atoms = attempt
                    used_selection = sel
                    break

            if atoms is None:
                try:
                    attempt = universe.select_atoms("all")
                    n_attempt = int(getattr(attempt, "n_atoms", 0) or 0)
                    if 0 < n_attempt <= max_pdb_atoms:
                        atoms = attempt
                        used_selection = "all"
                except Exception:
                    atoms = None

            if atoms is None:
                logger.warning(
                    "Structure overlay failed: no atoms found for selection '%s'.",
                    requested_selection,
                )
                return None, None

            self._structure_cache = {"atoms": atoms, "u": universe}
            self._structure_cache_key = cache_key
            if used_selection and used_selection != requested_selection:
                logger.warning(
                    "Structure overlay fallback selection used: '%s' (requested: '%s').",
                    used_selection,
                    requested_selection,
                )
            return atoms, cache_key
        except Exception as exc:
            msg = f"Failed to load structure for Density3D: {exc}"
            logger.error(msg, exc_info=True)
            if getattr(self, "run_logger", None):
                self.run_logger.warning(msg)
            return None, None

    def _update_density_explore(self) -> None:
        if not self.current_result:
            return
        names = list(self.current_result.density_results.keys())
        self.density_combo.blockSignals(True)
        self.density_combo.clear()
        self.density_combo.addItems(names)
        if names:
            self.density_combo.setCurrentIndex(0)
        self._density_view_key = None
        self._density_widget_structure_key = None
        self.density_combo.blockSignals(False)
        self._queue_density_update()

    def _update_density_overlay(self, density) -> None:
        if not hasattr(self, "_density_overlay_items"):
            self._density_overlay_items = {}
        coords = self._get_density_overlay_coords(density)
        if not coords:
            self._clear_density_overlay()
            return
        for key, (x_vals, y_vals) in coords.items():
            plot = self.density_plots.get(key)
            if plot is None:
                continue
            scatter = self._density_overlay_items.get(key)
            if scatter is None:
                scatter = pg.ScatterPlotItem(
                    pen=pg.mkPen(self._get_theme_tokens()["text"], width=1),
                    brush=pg.mkBrush(255, 255, 255, 80),
                    size=max(4, int(self._plot_marker_size)),
                )
                plot.addItem(scatter)
                self._density_overlay_items[key] = scatter
            scatter.setData(x=x_vals, y=y_vals)

    def _clear_density_overlay(self) -> None:
        if not hasattr(self, "_density_overlay_items"):
            return
        for scatter in self._density_overlay_items.values():
            try:
                scatter.setData([], [])
            except Exception:
                pass

    def _get_density_overlay_coords(self, density) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        selection = density.metadata.get("align_selection") or "protein and name CA"
        cache_key = (density.name, selection)
        if not hasattr(self, "_density_overlay_cache"):
            self._density_overlay_cache = {}
        if cache_key in self._density_overlay_cache:
            return self._density_overlay_cache[cache_key]
        project = self.run_project or self.state.project
        if not project:
            return {}
        try:
            import MDAnalysis as mda

            if project.inputs.trajectory:
                u = mda.Universe(project.inputs.topology, project.inputs.trajectory)
            else:
                u = mda.Universe(project.inputs.topology)
            coords = u.select_atoms(selection).positions
            if coords.size == 0:
                return {}
            coords = np.asarray(coords, dtype=float)
            overlay = {
                "xy": (coords[:, 0], coords[:, 1]),
                "xz": (coords[:, 0], coords[:, 2]),
                "yz": (coords[:, 1], coords[:, 2]),
            }
            self._density_overlay_cache[cache_key] = overlay
            return overlay
        except Exception:
            return {}

    def _update_water_dynamics_plots(self) -> None:
        self.water_sp_plot.clear()
        self.water_hbl_plot.clear()
        self.water_wor_plot.clear()
        self.water_dynamics_note.setText("")
        log_x = self.water_dynamics_log_check.isChecked()
        try:
            self.water_sp_plot.setLogMode(x=log_x, y=False)
            self.water_hbl_plot.setLogMode(x=log_x, y=False)
            self.water_wor_plot.setLogMode(x=log_x, y=False)
        except Exception:
            pass
        if not self.current_result or not self.current_result.water_dynamics_results:
            self.water_sp_plot.addItem(
                pg.TextItem("No water dynamics results.", color=self._get_theme_tokens()["text_muted"])
            )
            self._set_plot_insights("water", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        name = self.water_dynamics_combo.currentText()
        dynamics = self.current_result.water_dynamics_results.get(name)
        if dynamics is None:
            self._set_plot_insights("water", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            return
        try:
            self.water_sp_plot.setTitle(f"SP(τ) [{dynamics.residence_mode}]")
        except Exception:
            pass
        self.water_sp_plot.setToolTip(
            f"Survival probability SP(τ) — {dynamics.residence_mode} residence; stride impacts τ."
        )
        self.water_hbl_plot.setToolTip("Hydrogen-bond lifetimes (HBL).")
        self.water_wor_plot.setToolTip("Water orientational relaxation (WOR).")
        sp = dynamics.sp_tau
        if not sp.empty:
            self.water_sp_plot.setLabel("bottom", "Time τ", units="ps")
            self.water_sp_plot.setLabel("left", "Probability P(τ)")
            self.water_sp_plot.plot(
                sp["tau"],
                sp["survival"],
                pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
            )
            self.water_sp_plot.autoRange()

        if dynamics.hbl is not None and not dynamics.hbl.empty:
            self.water_hbl_plot.setLabel("bottom", "Time τ", units="ps")
            self.water_hbl_plot.setLabel("left", "Correlation C(τ)")
            self.water_hbl_plot.plot(
                dynamics.hbl["tau"],
                dynamics.hbl["correlation"],
                pen=pg.mkPen(self._get_theme_tokens()["accent_alt"], width=self._plot_line_width),
            )
            self.water_hbl_plot.autoRange()
        else:
            # [Audit Fix] Actionable empty state
            msg = "No H-bonds detected in region"
            if dynamics.hbl is None:
                msg = "HBL analysis not enabled"
            self.water_hbl_plot.addItem(
                pg.TextItem(msg, color=self._get_theme_tokens()["text_muted"], anchor=(0.5, 0.5))
            )

        if dynamics.wor is not None and not dynamics.wor.empty:
            self.water_wor_plot.setLabel("bottom", "Time τ", units="ps")
            self.water_wor_plot.setLabel("left", "Correlation C(τ)")
            self.water_wor_plot.plot(
                dynamics.wor["tau"],
                dynamics.wor["correlation"],
                pen=pg.mkPen(self._get_theme_tokens()["accent"], width=self._plot_line_width),
            )
            self.water_wor_plot.autoRange()
        else:
            # [Audit Fix] Actionable empty state
            msg = "No solvent in region or WOR disabled"
            if dynamics.wor is None:
                msg = "WOR analysis not enabled"
            self.water_wor_plot.addItem(
                pg.TextItem(msg, color=self._get_theme_tokens()["text_muted"], anchor=(0.5, 0.5))
            )
        if dynamics.notes:
            self.water_dynamics_note.setText("; ".join(dynamics.notes))
        if hasattr(self, "water_dynamics_summary"):
            self.water_dynamics_summary.setRowCount(0)
            summary = dynamics.hbl_summary
            if summary is not None and not summary.empty:
                top = summary.head(10)
                self.water_dynamics_summary.setRowCount(len(top))
                for row_idx, row in enumerate(top.itertuples(index=False)):
                    values = [
                        getattr(row, "resindex", ""),
                        getattr(row, "resid", ""),
                        getattr(row, "resname", ""),
                        getattr(row, "segid", ""),
                        f"{getattr(row, 'mean_lifetime', 0.0):.3f}",
                    ]
                    for col_idx, value in enumerate(values):
                        item = QtWidgets.QTableWidgetItem(str(value))
                        self.water_dynamics_summary.setItem(row_idx, col_idx, item)
        sp_values = dynamics.sp_tau["survival"].to_numpy(dtype=float) if dynamics.sp_tau is not None and not dynamics.sp_tau.empty else np.array([], dtype=float)
        top_text = "none"
        if dynamics.hbl_summary is not None and not dynamics.hbl_summary.empty:
            top_text = ", ".join(
                f"{str(r.resname)}{int(r.resid)}:{float(r.mean_lifetime):.2f}"
                for r in dynamics.hbl_summary.head(3).itertuples()
            )
        self._set_plot_insights(
            "water",
            [
                f"n={len(sp_values)}",
                f"mean={float(np.mean(sp_values)) if sp_values.size else 0.0:.3f}",
                f"max={float(np.max(sp_values)) if sp_values.size else 0.0:.3f}",
                f"top3={top_text}",
            ],
        )

    def _update_water_dynamics_explore(self) -> None:
        if not self.current_result:
            return
        names = list(self.current_result.water_dynamics_results.keys())
        self.water_dynamics_combo.blockSignals(True)
        self.water_dynamics_combo.clear()
        self.water_dynamics_combo.addItems(names)
        if names:
            self.water_dynamics_combo.setCurrentIndex(0)
        self.water_dynamics_combo.blockSignals(False)
        self._update_water_dynamics_plots()

    def _selected_soz_name(self) -> Optional[str]:
        if not self.current_result or not self.current_result.soz_results:
            return None
        return self.timeline_soz_combo.currentText() or list(self.current_result.soz_results.keys())[0]

    def _selected_soz_definition(self) -> SOZDefinition | None:
        soz_name = self._selected_soz_name()
        if not soz_name:
            return None
        project = self.run_project or self.state.project
        if not project:
            return None
        for soz in project.sozs:
            if soz.name == soz_name:
                return soz
        return None

    def _friendly_selection_label(self, label: str | None) -> str:
        return self._selection_display_for_label(label)

    def _describe_soz_node(self, node: SOZNode) -> str:
        if node.type in ("and", "or"):
            joiner = " AND " if node.type == "and" else " OR "
            parts = [self._describe_soz_node(child) for child in node.children]
            parts = [part for part in parts if part]
            return joiner.join(parts) if parts else node.type.upper()
        if node.type == "not":
            if node.children:
                return f"NOT {self._describe_soz_node(node.children[0])}"
            return "NOT"
        if node.type == "shell":
            label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
            return f"Shell({self._friendly_selection_label(label)})"
        if node.type == "distance":
            label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
            return f"Distance({self._friendly_selection_label(label)})"
        return node.type.upper()

    def _selected_soz_logic(self) -> str | None:
        soz = self._selected_soz_definition()
        if not soz:
            return None
        return self._describe_soz_node(soz.root)

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
        self._queue_hist_update()
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
        step_mode = False
        show_markers = self.timeline_markers_check.isChecked()
        show_mean = self.timeline_mean_check.isChecked()
        show_median = self.timeline_median_check.isChecked()
        shade_occupancy = self.timeline_shade_check.isChecked()
        metric = "n_solvent"
        secondary_metric = "None"
        smooth = False
        smooth_window = 1
        max_y = 0.0
        plotted_any = False
        self.timeline_highlight_line = None
        insight_values: list[float] = []
        insight_top_items: list[str] = []
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
                if y_raw.size:
                    insight_values.extend(y_raw.tolist())
                    top_val = float(np.nanmax(y_raw))
                    insight_top_items.append(f"{name}:{top_val:.2f}")
                y = np.maximum(y_raw, 0) if clamp else y_raw
                x, y = self._downsample(time_ps / 1000.0, y)
                if smooth and smooth_window > 1:
                    y = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
                if step_mode:
                    x, y = self._step_series(x, y)
                if y.size:
                    max_y = max(max_y, float(np.nanmax(y)))
                    plotted_any = True
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
                    if y_raw.size:
                        insight_values.extend(y_raw.tolist())
                        top_idx = np.argsort(y_raw)[::-1][:3]
                        insight_top_items = [
                            f"{(time_ps[i] / 1000.0):.2f}ns:{float(y_raw[i]):.2f}" for i in top_idx
                        ]
                    y = np.maximum(y_raw, 0) if clamp else y_raw
                    x, y = self._downsample(time_ps / 1000.0, y)
                    if smooth and smooth_window > 1:
                        y = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
                    if step_mode:
                        x, y = self._step_series(x, y)
                    if y.size:
                        max_y = max(max_y, float(np.nanmax(y)))
                        plotted_any = True
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
        if not plotted_any:
            soz_name = self._selected_soz_name()
            per_frame = (
                self.current_result.soz_results.get(soz_name).per_frame
                if soz_name and self.current_result
                else pd.DataFrame()
            )
            reason, level = self._plot_empty_reason(per_frame, metric, "timeline")
            self.timeline_plot.addItem(pg.TextItem(reason, color=tokens["text_muted"]))
            self._log_plot_reason("timeline", reason, level)
            self._set_plot_insights("timeline", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
        else:
            values = np.asarray(insight_values, dtype=float)
            if values.size == 0:
                self._set_plot_insights("timeline", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            else:
                top_text = ", ".join(insight_top_items[:3]) if insight_top_items else "none"
                self._set_plot_insights(
                    "timeline",
                    [
                        f"n={len(values)}",
                        f"mean={float(np.nanmean(values)):.3f}",
                        f"max={float(np.nanmax(values)):.3f}",
                        f"top3={top_text}",
                    ],
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
        if hasattr(self, "timeline_help_label"):
            self.timeline_help_label.setText(
                "Occupancy Timeline shows solvent counts per frame. Brush a time window to "
                "filter histograms, event rasters, and tables."
            )
        if hasattr(self, "explore_inspector_text"):
            window_text = (
                f"{self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
                if self._time_window
                else "full range"
            )
            logic = self._selected_soz_logic()
            logic_line = f"Logic: {logic}\n" if logic else ""
            self.explore_inspector_text.setText(
                f"SOZ: {self._selected_soz_name()}\n"
                f"{logic_line}"
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
        self._queue_event_update()
        self._update_tables_for_selected_soz()

    def _update_hist_plot(self) -> None:
        if not self.current_result:
            return
        tokens = self._get_theme_tokens()
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        metric = "n_solvent"
        df_full = self.current_result.soz_results[soz_name].per_frame
        if df_full.empty:
            self.hist_plot.clear()
            self.hist_plot.addItem(
                pg.TextItem("No per-frame data available for histogram.", color=tokens["text_muted"])
            )
            self.hist_info.setText("No per-frame data available for histogram.")
            self._set_plot_insights("hist", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            self._log_plot_reason("histogram", "No per-frame data available for histogram.", "warning")
            return
        df = self._filtered_per_frame(df_full)
        if metric not in df.columns:
            self.hist_plot.clear()
            self.hist_plot.addItem(
                pg.TextItem("Selected metric not available.", color=tokens["text_muted"])
            )
            self.hist_info.setText("Selected metric not available.")
            self._set_plot_insights("hist", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            self._log_plot_reason("histogram", f"Metric '{metric}' missing from per-frame table.", "warning")
            return
        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        if metric == "time":
            values = values / 1000.0
        if values.size == 0:
            reason, level = self._plot_empty_reason(df, metric, "histogram")
            self.hist_plot.clear()
            self.hist_plot.addItem(pg.TextItem(reason, color=tokens["text_muted"]))
            self.hist_info.setText(reason)
            self._set_plot_insights("hist", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
            self._log_plot_reason("histogram", reason, level)
            return
        bins = 30
        zero_mask = values == 0
        zero_count = int(np.sum(zero_mask))
        zero_frac = zero_count / len(values)
        positive_values = values[~zero_mask]

        split_zeros = False
        self.hist_zero_plot.setVisible(split_zeros)
        self.hist_zero_plot.clear()

        if split_zeros:
            zero_axis = self.hist_zero_plot.getAxis("bottom")
            zero_axis.setTicks([[(0, "zero"), (1, "non-zero")]])
            zero_axis.setLabel("Category")
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

        if split_zeros and positive_values.size == 0:
            mean_all = float(np.mean(values)) if values.size else 0.0
            self.hist_plot.clear()
            self.hist_plot.addItem(
                pg.TextItem("No non-zero values to plot.", color=tokens["text_muted"])
            )
            self.hist_info.setText(
                f"n={len(values)} | zeros={zero_count} ({zero_frac:.1%}) | "
                "no non-zero values"
            )
            self._set_plot_insights(
                "hist",
                [
                    f"n={len(values)}",
                    f"mean={mean_all:.3f}",
                    f"max={float(np.max(values)) if values.size else 0.0:.3f}",
                    "top3=none",
                ],
            )
            self._log_plot_reason("histogram", "No non-zero values to plot.", "info")
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
        if metric == "time":
            self.hist_plot.setLabel("bottom", "Time", units="ns")
        else:
            pretty = self._hist_metric_label(metric)
            label = f"{pretty} (non-zero)" if split_zeros else pretty
            self.hist_plot.setLabel("bottom", label)
        self.hist_plot.setLabel("left", "Frames")

        if split_zeros:
            self.hist_info.setText(
                f"n={len(values)} frames | zeros={zero_count} ({zero_frac:.1%}) | "
                f"mean={mean_val:.3f} median={med_val:.3f} | "
                f"non-zero mean={mean_nonzero:.3f} median={med_nonzero:.3f} | bins={bins}"
            )
        else:
            self.hist_info.setText(
                f"n={len(values)} frames | zeros={zero_count} ({zero_frac:.1%}) | "
                f"mean={mean_val:.3f} median={med_val:.3f} | bins={bins}"
            )
        if self._time_window:
            self.hist_info.setText(
                self.hist_info.text()
                + f" | window={self._time_window[0]:.3f}–{self._time_window[1]:.3f} ns"
            )
        try:
            top_bins_idx = np.argsort(np.nan_to_num(y, nan=0.0))[::-1][:3]
            top_bins = ", ".join(
                f"{centers[i]:.2f}:{0.0 if np.isnan(y[i]) else y[i]:.2f}" for i in top_bins_idx if i < len(centers)
            )
        except Exception:
            top_bins = ""
        self._set_plot_insights(
            "hist",
            [
                f"n={len(values)}",
                f"mean={mean_val:.3f}",
                f"max={float(np.max(values)):.3f}",
                f"top3={top_bins or 'none'}",
            ],
        )

    def _highlight_timeline_time(self, time_ns: float | None) -> None:
        if time_ns is None or not hasattr(self, "timeline_plot"):
            return
        if self.timeline_highlight_line is not None:
            try:
                self.timeline_plot.removeItem(self.timeline_highlight_line)
            except Exception:
                pass
            self.timeline_highlight_line = None
        self.timeline_highlight_line = pg.InfiniteLine(
            pos=float(time_ns),
            angle=90,
            pen=pg.mkPen(self._get_theme_tokens()["accent"], width=2),
        )
        self.timeline_plot.addItem(self.timeline_highlight_line)

    def _on_event_points_clicked(self, _item, points) -> None:
        if not points:
            return
        point = points[0]
        row = int(round(point.pos().y()))
        if row < 0 or row >= len(self._event_ids_current):
            return
        self._set_selected_solvent(self._event_ids_current[row], sync_table=True)
        self._highlight_timeline_time(float(point.pos().x()))

    def _on_event_plot_clicked(self, event) -> None:
        if not self._event_ids_current:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.event_plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        mouse = vb.mapSceneToView(event.scenePos())
        row = int(round(mouse.y()))
        if row < 0 or row >= len(self._event_ids_current):
            return
        self._set_selected_solvent(self._event_ids_current[row], sync_table=True)
        self._highlight_timeline_time(float(mouse.x()))

    def _update_event_plot(self) -> None:
        if not self.current_result:
            return
        soz_name = self._selected_soz_name()
        if not soz_name:
            return
        self._event_ids_current = []
        self._event_rows_current = None
        self._event_cols_current = None
        self._event_time_current = None
        self._event_scatter_item = None
        self._event_highlight_item = None
        tokens = self._get_theme_tokens()
        per_frame_full = self.current_result.soz_results[soz_name].per_frame
        per_frame = self._filtered_per_frame(per_frame_full)
        per_solvent = self.current_result.soz_results[soz_name].per_solvent
        matrix, ids, time_ns, msg = self._build_presence_matrix(
            per_frame,
            per_solvent,
            top_n=50,
            min_occ_pct=0.0,
        )
        if matrix is None:
            self.event_plot.clear()
            self.event_plot.addItem(
                pg.TextItem(msg or "No solvent IDs available", color=tokens["text_muted"])
            )
            self.event_info.setText(msg or "")
            self._set_plot_insights("event", ["n=0", "events=0", "max=0", "top3=none"])
            self._log_plot_reason("event raster", msg or "No solvent IDs available", "warning")
            return
        stride = max(1, int(self.event_stride_spin.value()))
        if stride > 1:
            matrix = matrix[:, ::stride]
            time_ns = time_ns[::stride]
        self._event_ids_current = list(ids)
        self._event_time_current = np.asarray(time_ns)
        rows, cols = np.where(matrix > 0)
        self._event_rows_current = rows
        self._event_cols_current = cols
        self.event_plot.clear()
        if rows.size == 0:
            self.event_plot.addItem(
                pg.TextItem("No events for selected solvents", color=tokens["text_muted"])
            )
            self.event_info.setText("No occupancy events found for the chosen top solvents.")
            self._set_plot_insights(
                "event",
                [f"n={len(ids)}", "events=0", "max=0", "top3=none"],
            )
            self._log_plot_reason("event raster", "No occupancy events found for selected solvents.", "info")
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
            scatter.sigClicked.connect(self._on_event_points_clicked)
            self._event_scatter_item = scatter
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
                self._event_highlight_item = highlight
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
        row_counts = np.sum(matrix, axis=1).astype(int)
        top_idx = np.argsort(row_counts)[::-1][:3]
        top_text = ", ".join(
            f"{self._short_solvent_label(ids[i])}:{row_counts[i]}" for i in top_idx if i < len(ids)
        )
        self._set_plot_insights(
            "event",
            [
                f"n={len(ids)}",
                f"events={int(rows.size)}",
                f"max={int(np.max(row_counts)) if row_counts.size else 0}",
                f"top3={top_text or 'none'}",
            ],
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
            warnings_list = outputs.get("warnings", [])
            lines.extend(
                f"{key}: {path}" for key, path in outputs.items() if key != "warnings"
            )
            if warnings_list:
                lines.append("Warnings:")
                lines.extend(f"  - {warning}" for warning in warnings_list)
            self.extract_summary.setText("\n".join(lines))
            if hasattr(self, "export_inspector_text"):
                self.export_inspector_text.setText("\n".join(lines))
            if self.run_logger:
                self.run_logger.info("Extraction outputs: %s", outputs)
                if warnings_list:
                    self.run_logger.warning("Extraction warnings: %s", warnings_list)
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
        self._toast("Plot copied to clipboard", 3000)

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
            self._toast("Analysis running…", 3000)
        else:
            if not self.status_bar.currentMessage():
                self._toast("Ready", 2000)

    def _clear_results_view(self) -> None:
        self.current_result = None
        self._timeline_stats_cache = {}
        self._timeline_event_cache = {}
        self._time_window = None
        self._selected_solvent_id = None
        self._event_ids_current = []
        self._event_rows_current = None
        self._event_cols_current = None
        self._event_time_current = None
        if self.timeline_region is not None:
            try:
                self.timeline_plot.removeItem(self.timeline_region)
            except Exception:
                pass
            self.timeline_region = None
        self._set_plot_insights("timeline", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
        self._set_plot_insights("hist", ["n=0", "mean=0.000", "max=0.000", "top3=none"])
        self._set_plot_insights("event", ["n=0", "events=0", "max=0", "top3=none"])
        self.overview_card.setText("Running analysis…")
        self.overview_text.setText("")
        self.qc_summary_label.setText("Running analysis…")
        self.qc_text.setText("")
        if hasattr(self, "qc_findings_text"):
            self.qc_findings_text.setPlainText("Running analysis…")
        if hasattr(self, "qc_health_headline"):
            self.qc_health_headline.setText("Project health: Running analysis…")
        if hasattr(self, "qc_health_detail"):
            self.qc_health_detail.setText("QC checks are being recomputed.")
        if hasattr(self, "qc_health_card"):
            self._set_status_card_tone(self.qc_health_card, "neutral")
        self.report_text.setText("")
        self.timeline_plot.clear()
        self.timeline_event_plot.clear()
        self.timeline_summary_label.setText("")
        self.timeline_summary_label.setVisible(False)
        self.timeline_stats_status.setText("Not computed yet.")
        self.hist_plot.clear()
        self.hist_zero_plot.clear()
        self.hist_info.setText("")
        self.event_plot.clear()
        self.event_info.setText("")
        empty_model = QtGui.QStandardItemModel()
        self.per_frame_table.setModel(empty_model)
        self.per_solvent_table.setModel(empty_model)
        for attr in (
            "distance_bridge_timeseries_plot",
            "distance_bridge_residence_plot",
            "distance_bridge_top_plot",
            "hbond_bridge_timeseries_plot",
            "hbond_bridge_residence_plot",
            "hbond_bridge_top_plot",
            "bridge_compare_plot",
            "hbond_bridge_network_plot",
            "hydration_frequency_plot",
            "hydration_top_plot",
            "hydration_timeline_plot",
            "water_sp_plot",
            "water_hbl_plot",
            "water_wor_plot",
        ):
            plot = getattr(self, attr, None)
            if plot is not None:
                plot.clear()
        if hasattr(self, "density_images"):
            for img in self.density_images.values():
                img.clear()
        for combo_attr in (
            "distance_bridge_combo",
            "hbond_bridge_combo",
            "hydration_config_combo",
            "density_combo",
            "water_dynamics_combo",
        ):
            combo = getattr(self, combo_attr, None)
            if combo is not None:
                combo.clear()
        if hasattr(self, "water_dynamics_summary"):
            self.water_dynamics_summary.setRowCount(0)
        if hasattr(self, "explore_inspector_text"):
            self.explore_inspector_text.setText("Awaiting results…")
        self._refresh_defined_soz_panel()

    def _copy_current_plot(self) -> None:
        index = self.plots_tabs.currentIndex()
        if index == 0:
            self._copy_widget_to_clipboard(self.hist_plot)
        elif index == 1:
            self._copy_widget_to_clipboard(self.event_plot)

    def _ensure_export_extension(self, path: str, selected_filter: str) -> str:
        if not path:
            return path
        if Path(path).suffix:
            return path
        ext_map = {
            "PNG": ".png",
            "SVG": ".svg",
            "EMF": ".emf",
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
                self.event_plot,
                "events.png",
                csv_exporter=self._write_event_raster_csv,
            )

    def _export_distance_bridge_plot(self) -> None:
        widget = self.distance_bridge_timeseries_plot
        filename = "distance_bridge_timeseries.png"
        self._export_plot(widget, filename, title="Export Distance Bridge Plot")

    def _export_hbond_bridge_plot(self) -> None:
        choice = self.hbond_bridge_export_combo.currentText()
        widget = self.hbond_bridge_timeseries_plot
        filename = "hbond_bridge_timeseries.png"
        if choice.startswith("Residence"):
            widget = self.hbond_bridge_residence_plot
            filename = "hbond_bridge_residence.png"
        elif choice.startswith("Top"):
            widget = self.hbond_bridge_top_plot
            filename = "hbond_bridge_top.png"
        elif choice.startswith("Comparator"):
            widget = self.bridge_compare_plot
            filename = "bridge_comparator.png"
        elif choice.startswith("Network"):
            widget = self.hbond_bridge_network_plot
            filename = "hbond_bridge_network.png"
        self._export_plot(widget, filename, title="Export H-bond Bridge Plot")

    def _export_hydration_plot(self) -> None:
        choice = self.hydration_export_combo.currentText()
        widget = self.hydration_frequency_plot
        filename = "hydration_frequency.png"
        if choice.startswith("Top"):
            widget = self.hydration_top_plot
            filename = "hydration_top.png"
        elif choice.startswith("Timeline"):
            widget = self.hydration_timeline_plot
            filename = "hydration_timeline.png"
        self._export_plot(widget, filename, title="Export Hydration Plot")

    def _export_density_figure(self) -> None:
        self._export_plot(self.density_layout, "density_figure_pack.png", title="Export Density Figure Pack")

    def _export_water_dynamics_plot(self) -> None:
        choice = self.water_dynamics_export_combo.currentText()
        widget = self.water_sp_plot
        filename = "water_dynamics_sp.png"
        if choice.startswith("HBL"):
            widget = self.water_hbl_plot
            filename = "water_dynamics_hbl.png"
        elif choice.startswith("WOR"):
            widget = self.water_wor_plot
            filename = "water_dynamics_wor.png"
        self._export_plot(widget, filename, title="Export Water Dynamics Plot")

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
        filters = ["PNG (*.png)", "PNG @2x (*.png)", "SVG (*.svg)"]
        if self._emf_available():
            filters.append("EMF (*.emf)")
        filters.extend(["PDF (*.pdf)", "CSV (*.csv)"])
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            default_path,
            ";;".join(filters),
        )
        if not path:
            return
        path = self._ensure_export_extension(path, selected_filter)
        suffix = Path(path).suffix.lower()
        png_scale = 2.0 if "@2x" in selected_filter else 1.0
        if suffix == ".csv":
            if csv_exporter is None:
                self.status_bar.showMessage("CSV export not available for this plot.", 5000)
                return
            csv_exporter(path)
            return
        try:
            if suffix == ".emf":
                if self._export_plot_emf(plot_widget, path):
                    self._notify_powerpoint_export(
                        "EMF",
                        path,
                        "In PowerPoint: Insert > Pictures, then Group > Ungroup (accept the conversion prompt).",
                    )
                return
            if suffix == ".pdf":
                self._export_plot_pdf(plot_widget, path)
                return
            if suffix == ".svg":
                if self._export_plot_svg(plot_widget, path, sanitize=True):
                    self._notify_powerpoint_export(
                        "SVG",
                        path,
                        "If Ungroup fails, use Insert > Pictures and try Ungroup after paste.",
                    )
                return
            if suffix == ".png":
                if not self._export_plot_raster(plot_widget, path, scale=png_scale):
                    raise RuntimeError("PNG export failed.")
                self._toast(f"Saved plot: {path}", 4000)
                return
            if isinstance(plot_widget, pg.PlotWidget):
                exporter = pg.exporters.ImageExporter(plot_widget.plotItem)
                exporter.export(path)
            else:
                pixmap = plot_widget.grab()
                pixmap.save(path)
            self._toast(f"Saved plot: {path}", 4000)
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("Plot export failed")
            self._toast(f"Plot export failed: {exc}", 8000, level="error")

    def _export_plot_svg(
        self, plot_widget: QtWidgets.QWidget, path: str, sanitize: bool = True
    ) -> bool:
        if isinstance(plot_widget, pg.PlotWidget):
            ok = self._export_plot_svg_pg(plot_widget, path)
        else:
            ok = self._render_widget_svg(plot_widget, path)
        if ok:
            self._normalize_svg_text(path)
            if sanitize:
                self._sanitize_svg_for_powerpoint(path)
        return ok

    def _export_plot_svg_pg(self, plot_widget: pg.PlotWidget, path: str) -> bool:
        plot_item = getattr(plot_widget, "plotItem", None)
        if plot_item is None:
            return self._render_widget_svg(plot_widget, path)
        clip_state = None
        if hasattr(plot_item, "setClipToView"):
            clip_attr = getattr(plot_item, "clipToView", None)
            if callable(clip_attr):
                clip_state = clip_attr()
            elif isinstance(clip_attr, bool):
                clip_state = clip_attr
            plot_item.setClipToView(False)
        try:
            exporter = pg.exporters.SVGExporter(plot_item)
            rect = plot_item.sceneBoundingRect()
            export_size = QtCore.QSize(
                max(1, int(round(rect.width()))),
                max(1, int(round(rect.height()))),
            )
            try:
                params = exporter.parameters()
                if hasattr(params, "keys"):
                    if "width" in params:
                        params["width"] = export_size.width()
                    if "height" in params:
                        params["height"] = export_size.height()
            except Exception:
                pass
            exporter.export(path)
            return True
        except Exception:
            return self._render_widget_svg(plot_widget, path)
        finally:
            if clip_state is not None and hasattr(plot_item, "setClipToView"):
                plot_item.setClipToView(clip_state)

    def _export_target_size(self, widget: QtWidgets.QWidget) -> QtCore.QSize:
        size = widget.size()
        if size.width() <= 0 or size.height() <= 0:
            size = widget.sizeHint()
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(960, 640)
        return size

    def _render_widget_svg(self, widget: QtWidgets.QWidget, path: str) -> bool:
        try:
            from PyQt6 import QtSvg
        except Exception:
            self.status_bar.showMessage("SVG export requires QtSvg.", 5000)
            return False
        try:
            generator = QtSvg.QSvgGenerator()
            generator.setFileName(path)
            size = self._export_target_size(widget)
            generator.setSize(size)
            generator.setViewBox(QtCore.QRect(0, 0, size.width(), size.height()))
            generator.setResolution(96)
            painter = QtGui.QPainter(generator)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            widget.render(painter)
            painter.end()
            return True
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("SVG export failed")
            self.status_bar.showMessage(f"SVG export failed: {exc}", 8000)
            return False

    def _export_plot_pdf(self, plot_widget: QtWidgets.QWidget, path: str) -> bool:
        if self._export_plot_via_inkscape(plot_widget, path, "pdf"):
            return True
        if self._render_widget_pdf(plot_widget, path):
            self.status_bar.showMessage(
                "PDF exported with Qt fallback; install Inkscape for best fidelity.",
                8000,
            )
            return True
        self.status_bar.showMessage("PDF export failed.", 8000)
        return False

    def _emf_available(self) -> bool:
        return bool(self._find_inkscape())

    def _export_plot_emf(self, plot_widget: QtWidgets.QWidget, path: str) -> bool:
        if not self._emf_available():
            self.status_bar.showMessage("EMF export requires Inkscape.", 8000)
            return False
        if self._export_plot_via_inkscape(plot_widget, path, "emf"):
            return True
        self.status_bar.showMessage("EMF export failed. Check Inkscape installation.", 8000)
        return False

    def _export_plot_via_inkscape(
        self, plot_widget: QtWidgets.QWidget, path: str, fmt: str
    ) -> bool:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                svg_path = str(Path(tmp_dir) / "plot.svg")
                if self._export_plot_svg(plot_widget, svg_path, sanitize=False):
                    if fmt == "emf":
                        return self._convert_svg_to_emf(svg_path, path)
                    if fmt == "pdf":
                        return self._convert_svg_to_pdf(svg_path, path)
                png_path = str(Path(tmp_dir) / "plot.png")
                if not self._export_plot_raster(plot_widget, png_path):
                    return False
                if fmt == "emf":
                    return self._convert_svg_to_emf(png_path, path)
                if fmt == "pdf":
                    return self._convert_svg_to_pdf(png_path, path)
        except Exception:
            return False
        return False

    def _export_plot_raster(self, plot_widget: QtWidgets.QWidget, path: str, scale: float = 1.0) -> bool:
        try:
            if isinstance(plot_widget, pg.PlotWidget):
                exporter = pg.exporters.ImageExporter(plot_widget.plotItem)
                if scale > 1.0:
                    try:
                        params = exporter.parameters()
                        if hasattr(params, "__setitem__"):
                            params["width"] = int(plot_widget.width() * scale)
                        else:
                            exporter.parameters()["width"] = int(plot_widget.width() * scale)
                    except Exception:
                        pass
                exporter.export(path)
                return Path(path).exists()
            pixmap = plot_widget.grab()
            if scale > 1.0:
                pixmap = pixmap.scaled(
                    int(pixmap.width() * scale),
                    int(pixmap.height() * scale),
                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            return pixmap.save(path)
        except Exception:
            return False

    def _render_widget_pdf(self, widget: QtWidgets.QWidget, path: str) -> bool:
        try:
            export_size = self._export_target_size(widget)
            dpi = 96.0
            width_in = export_size.width() / dpi
            height_in = export_size.height() / dpi
            pdf_writer = QtGui.QPdfWriter(path)
            pdf_writer.setResolution(int(dpi))
            try:
                pdf_writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
            except Exception:
                pass
            try:
                page_size = QtGui.QPageSize(
                    QtCore.QSizeF(width_in * 72.0, height_in * 72.0),
                    QtGui.QPageSize.Unit.Point,
                )
                pdf_writer.setPageSize(page_size)
            except Exception:
                pass
            painter = QtGui.QPainter(pdf_writer)
            if not painter.isActive():
                return False
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            target = QtCore.QRectF(0, 0, pdf_writer.width(), pdf_writer.height())
            source = QtCore.QRectF(0, 0, export_size.width(), export_size.height())
            widget.render(painter, target, source)
            painter.end()
            pdf_path = Path(path)
            return pdf_path.exists() and pdf_path.stat().st_size > 0
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("PDF export failed")
            self.status_bar.showMessage(f"PDF export failed: {exc}", 8000)
            return False

    def _sanitize_svg_for_powerpoint(self, path: str) -> None:
        self._normalize_svg_text(path)
        if self._convert_svg_to_plain_svg(path):
            self._normalize_svg_text(path)

    def _normalize_svg_text(self, path: str) -> None:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            return
        updated = text
        updated = re.sub(r'<!DOCTYPE[^>]*>', '', updated)
        updated = updated.replace('baseProfile="tiny"', "")
        updated = updated.replace('version="1.2"', 'version="1.1"')
        if "xmlns=\"http://www.w3.org/2000/svg\"" not in updated:
            updated = updated.replace("<svg ", "<svg xmlns=\"http://www.w3.org/2000/svg\" ")
        if "xmlns:xlink" not in updated:
            updated = updated.replace(
                "<svg ",
                "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" ",
            )
        updated = re.sub(r'width="(\\d+)"', r'width="\\1px"', updated)
        updated = re.sub(r'height="(\\d+)"', r'height="\\1px"', updated)
        if "viewBox" not in updated:
            width_match = re.search(r'width="([\\d\\.]+)px"', updated)
            height_match = re.search(r'height="([\\d\\.]+)px"', updated)
            if width_match and height_match:
                view_box = f'viewBox="0 0 {width_match.group(1)} {height_match.group(1)}"'
                updated = updated.replace("<svg ", f"<svg {view_box} ", 1)
        if updated != text:
            try:
                Path(path).write_text(updated, encoding="utf-8")
            except Exception:
                pass

    def _render_widget_emf_native(self, widget: QtWidgets.QWidget, path: str) -> bool:
        try:
            printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QtPrintSupport.QPrinter.OutputFormat.NativeFormat)
            printer.setOutputFileName(path)
            if not printer.isValid():
                return False
            try:
                printer.setFullPage(True)
            except Exception:
                pass
            size = widget.size()
            dpi_x = float(widget.logicalDpiX()) if hasattr(widget, "logicalDpiX") else 96.0
            dpi_y = float(widget.logicalDpiY()) if hasattr(widget, "logicalDpiY") else 96.0
            if dpi_x <= 0:
                dpi_x = 96.0
            if dpi_y <= 0:
                dpi_y = 96.0
            width_in = size.width() / dpi_x
            height_in = size.height() / dpi_y
            try:
                page_size = QtGui.QPageSize(
                    QtCore.QSizeF(width_in * 72.0, height_in * 72.0),
                    QtGui.QPageSize.Unit.Point,
                )
                printer.setPageSize(page_size)
            except Exception:
                pass
            painter = QtGui.QPainter(printer)
            if not painter.isActive():
                return False
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            widget.render(painter)
            painter.end()
            emf_path = Path(path)
            return emf_path.exists() and emf_path.stat().st_size > 0
        except Exception as exc:
            if self.run_logger:
                self.run_logger.exception("EMF export failed")
            self.status_bar.showMessage(f"EMF export failed: {exc}", 8000)
            return False

    def _convert_svg_to_emf(self, svg_path: str, emf_path: str) -> bool:
        inkscape = self._find_inkscape()
        if not inkscape:
            return False
        commands = [
            [inkscape, svg_path, "--export-type=emf", "--export-area-drawing", "--export-filename", emf_path],
            [inkscape, svg_path, "--export-type=emf", "--export-area-page", "--export-filename", emf_path],
            [inkscape, svg_path, "--export-type=emf", f"--export-filename={emf_path}"],
            [inkscape, svg_path, "--export-emf", emf_path],
            [inkscape, svg_path, f"--export-emf={emf_path}"],
        ]
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                continue
            if result.returncode == 0:
                out_path = Path(emf_path)
                if out_path.exists() and out_path.stat().st_size > 0:
                    return True
        return False

    def _convert_svg_to_pdf(self, svg_path: str, pdf_path: str) -> bool:
        inkscape = self._find_inkscape()
        if not inkscape:
            return False
        commands = [
            [inkscape, svg_path, "--export-type=pdf", "--export-area-drawing", "--export-filename", pdf_path],
            [inkscape, svg_path, "--export-type=pdf", "--export-area-page", "--export-filename", pdf_path],
            [inkscape, svg_path, "--export-type=pdf", f"--export-filename={pdf_path}"],
            [inkscape, svg_path, "--export-pdf", pdf_path],
            [inkscape, svg_path, f"--export-pdf={pdf_path}"],
        ]
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                continue
            if result.returncode == 0:
                out_path = Path(pdf_path)
                if out_path.exists() and out_path.stat().st_size > 0:
                    return True
        return False

    def _convert_svg_to_plain_svg(self, svg_path: str) -> bool:
        inkscape = self._find_inkscape()
        if not inkscape:
            return False
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_path = str(Path(tmp_dir) / "plain.svg")
                commands = [
                    [inkscape, svg_path, "--export-plain-svg", "--export-area-drawing", "--export-filename", out_path],
                    [inkscape, svg_path, "--export-plain-svg", "--export-area-page", "--export-filename", out_path],
                    [inkscape, svg_path, "--export-plain-svg", "--export-filename", out_path],
                    [inkscape, svg_path, "--export-plain-svg", f"--export-filename={out_path}"],
                    [inkscape, svg_path, "--export-plain-svg", out_path],
                    [inkscape, svg_path, "--export-type=svg", "--export-plain-svg", "--export-filename", out_path],
                    [inkscape, svg_path, "--export-type=svg", f"--export-filename={out_path}"],
                ]
                for cmd in commands:
                    try:
                        result = subprocess.run(
                            cmd,
                            check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    except Exception:
                        continue
                    if result.returncode == 0 and Path(out_path).exists():
                        shutil.copyfile(out_path, svg_path)
                        return True
        except Exception:
            return False
        return False

    def _find_inkscape(self) -> str | None:
        candidate = shutil.which("inkscape")
        if candidate:
            return candidate
        if sys.platform == "darwin":
            app_path = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
            if Path(app_path).exists():
                return app_path
            return None
        if sys.platform != "win32":
            return None
        roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ]
        candidates = []
        for root in roots:
            if not root:
                continue
            candidates.append(Path(root) / "Inkscape" / "inkscape.exe")
            candidates.append(Path(root) / "Inkscape" / "inkscape.com")
            candidates.append(Path(root) / "Inkscape" / "bin" / "inkscape.exe")
            candidates.append(Path(root) / "Inkscape" / "bin" / "inkscape.com")
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _notify_powerpoint_export(self, fmt: str, path: str, tip: str) -> None:
        self.status_bar.showMessage(f"{fmt} saved: {path}. PowerPoint tip: {tip}", 8000)

    def _write_timeline_csv(self, path: str) -> None:
        if not self.current_result or not self.current_result.soz_results:
            self.status_bar.showMessage("No timeline data available to export.", 5000)
            return
        metric = "n_solvent"
        metric_col = metric or "value"
        secondary_metric = "None"
        clamp = self.timeline_clamp_check.isChecked()
        smooth = False
        smooth_window = 1
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
        metric = "n_solvent"
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
        bins = 30
        zero_mask = values == 0
        split_zeros = False
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
                top_n=50,
                min_occ_pct=0.0,
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

    def _simplify_plot_context_menu(self, plot_obj) -> None:
        if plot_obj is None:
            return
        plot_item = None
        if hasattr(plot_obj, "plotItem"):
            plot_item = plot_obj.plotItem
        elif hasattr(plot_obj, "vb") and hasattr(plot_obj, "ctrlMenu"):
            plot_item = plot_obj
        elif hasattr(plot_obj, "parentItem"):
            parent = plot_obj.parentItem()
            if hasattr(parent, "vb") and hasattr(parent, "ctrlMenu"):
                plot_item = parent
        if plot_item is None:
            return
        try:
            visibility_map = {
                "Transforms": False,
                "Downsample": False,
                "Average": False,
                "Points": False,
                "Alpha": True,
                "Grid": True,
            }
            for action_name, visible in visibility_map.items():
                plot_item.setContextMenuActionVisible(action_name, visible)
        except Exception:
            pass
        try:
            vb = plot_item.vb
            menu = vb.menu
            scene = vb.scene()
            export_action = None
            if scene is not None:
                context_actions = getattr(scene, "contextMenu", None)
                if isinstance(context_actions, list):
                    for action in context_actions:
                        if isinstance(action, QtGui.QAction) and "Export" in action.text():
                            export_action = action
                            break
            menu.clear()
            if export_action is not None:
                menu.addAction(export_action)
            menu.addAction(plot_item.ctrlMenu.menuAction())
        except Exception:
            pass

    def _style_plot(self, plot: pg.PlotWidget, title: str | None = None) -> None:
        tokens = self._get_theme_tokens()
        plot.setBackground(tokens["plot_bg"])
        plot.showGrid(x=True, y=True, alpha=0.1)
        if title:
            size = int(10 * self._ui_scale)
            plot.setTitle(title, color=tokens["text"], size=f"{size}pt")
        for axis_name in ("bottom", "left", "right"):
            axis = plot.getAxis(axis_name)
            if axis is None:
                continue
            axis.setPen(pg.mkPen(tokens["plot_axis"]))
            axis.setTextPen(pg.mkPen(tokens["plot_fg"]))
            try:
                axis.setStyle(tickTextOffset=10, tickLength=6, autoExpandTextSpace=True)
            except Exception:
                pass
        plot.plotItem.legend = None
        self._simplify_plot_context_menu(plot)

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

    def _plot_empty_reason(
        self,
        per_frame: pd.DataFrame,
        metric: str | None,
        context: str,
    ) -> tuple[str, str]:
        if per_frame.empty:
            return (f"No per-frame data computed for {context}.", "warning")
        if not metric:
            return (f"No metric selected for {context}.", "warning")
        series = self._metric_series(per_frame, metric)
        if series is None:
            return (f"Metric '{metric}' missing from per-frame table.", "warning")
        if series.size == 0:
            return (f"No data computed for '{metric}'.", "warning")
        if np.nanmax(series) <= 0:
            return ("No events detected (all frames are zero).", "info")
        return ("No data available for plot.", "warning")

    def _log_plot_reason(self, context: str, reason: str, level: str) -> None:
        if not self.run_logger:
            return
        if level == "warning":
            self.run_logger.warning("%s plot empty: %s", context, reason)
        else:
            self.run_logger.info("%s plot empty: %s", context, reason)

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
        step_mode = False
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
                self._log_plot_reason("timeline events", "No entry/exit events detected.", "info")
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
        show_event = index == 1
        if hasattr(self, "hist_controls_row"):
            self.hist_controls_row.setVisible(show_hist)
        if hasattr(self, "event_controls_row"):
            self.event_controls_row.setVisible(show_event)

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
        # [UX Improvement] 10s tooltip duration
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self, QtCore.QRect(), 10000)

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
            note = ""
            if isinstance(soz.summary, dict):
                occ_frac = float(soz.summary.get("occupancy_fraction", 0.0))
                if occ_frac == 0.0:
                    note = (
                        "Occupancy is 0% (no solvent met the SOZ definition). "
                        "If using Both (A and B), try widening cutoffs or switch to Either (A or B)."
                    )
                else:
                    note = f"Occupancy fraction: {occ_frac:.3f}"
            if overlay_note:
                note = f"{note}\n{overlay_note}" if note else overlay_note
            if note:
                self.timeline_summary_label.setText(note)
                self.timeline_summary_label.setVisible(True)
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
            return "Solvent molecules within cutoff"
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

    def _apply_table_filter(self, text: str) -> None:
        for proxy in self._table_proxies.values():
            proxy.setFilterRegularExpression(text)

    def _select_per_solvent_row(self, solvent_id: str) -> None:
        if not solvent_id or not hasattr(self, "per_solvent_table"):
            return
        model = self.per_solvent_table.model()
        if model is None:
            return
        solvent_col = None
        for col in range(model.columnCount()):
            header = model.headerData(col, QtCore.Qt.Orientation.Horizontal)
            if str(header) == "solvent_id":
                solvent_col = col
                break
        if solvent_col is None:
            return
        for row in range(model.rowCount()):
            idx = model.index(row, solvent_col)
            if str(model.data(idx)) == solvent_id:
                sel_model = self.per_solvent_table.selectionModel()
                if sel_model is None:
                    return
                self._syncing_solvent_selection = True
                sel_model.select(
                    idx,
                    QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )
                self.per_solvent_table.scrollTo(idx)
                self._syncing_solvent_selection = False
                return

    def _set_selected_solvent(self, solvent_id: str | None, sync_table: bool = True) -> None:
        solvent_id = (solvent_id or "").strip()
        self._selected_solvent_id = solvent_id or None
        if sync_table and self._selected_solvent_id:
            self._select_per_solvent_row(self._selected_solvent_id)
        self._update_event_plot()
        label = self._selected_solvent_id or "none"
        if hasattr(self, "explore_inspector_text"):
            self.explore_inspector_text.setText(f"Selected solvent: {label}")
        if hasattr(self, "density_3d_widget") and self.density_3d_widget:
            try:
                self.density_3d_widget.set_focus_label(self._selected_solvent_id)
            except Exception:
                pass

    def _match_solvent_id_from_pick_event(self, event: dict) -> str | None:
        table = getattr(self, "per_solvent_table", None)
        if table is None:
            return None
        model = table.model()
        if model is None:
            return None

        solvent_col = None
        for col in range(model.columnCount()):
            header = str(model.headerData(col, QtCore.Qt.Orientation.Horizontal))
            if header.strip().lower() == "solvent_id":
                solvent_col = col
                break
        if solvent_col is None:
            return None

        resname = str(event.get("resname", "")).strip()
        resno = str(event.get("resno", "")).strip()
        atom_name = str(event.get("atomName", "")).strip()
        label = str(event.get("label", "")).strip()
        chain = str(event.get("chain", "")).strip()
        candidates = []
        for token in (
            f"{resname}{resno}" if resname and resno else "",
            f"{resname}:{resno}" if resname and resno else "",
            f"{resname}-{resno}" if resname and resno else "",
            resno,
            atom_name,
            chain,
            label,
        ):
            token = token.strip()
            if token:
                candidates.append(token.lower())

        if not candidates:
            return None

        for row in range(model.rowCount()):
            idx = model.index(row, solvent_col)
            sid = str(model.data(idx) or "").strip()
            if not sid:
                continue
            sid_l = sid.lower()
            if any(token and token in sid_l for token in candidates):
                return sid
        return None

    def _on_density_3d_pick_event(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        kind = str(event.get("kind", "")).strip().lower()
        if kind != "atom":
            return
        matched = self._match_solvent_id_from_pick_event(event)
        if matched:
            self._set_selected_solvent(matched, sync_table=True)
            return
        if hasattr(self, "explore_inspector_text"):
            atom = str(event.get("atomName", "")).strip()
            resname = str(event.get("resname", "")).strip()
            resno = str(event.get("resno", "")).strip()
            self.explore_inspector_text.setText(
                f"3D picked: {atom} {resname}{resno}"
            )

    def _on_per_solvent_selection_changed(self, selected, deselected=None) -> None:
        if self._syncing_solvent_selection:
            return
        sel_model = self.per_solvent_table.selectionModel()
        if sel_model is None or not sel_model.selectedRows():
            self._set_selected_solvent(None, sync_table=False)
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
        self._set_selected_solvent(solvent_id, sync_table=False)

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


# Workaround for Wayland EGL crashes (eglSwapBuffers failed with 0x300d)
# Force xcb by default if not set
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"


def _configure_qt_opengl_backend() -> None:
    """
    Select OpenGL backend before QApplication creation.
    Default is auto so QtWebEngine/NGL can negotiate a valid WebGL path.
    Override with SOZLAB_OPENGL_BACKEND=desktop|auto|software.
    """
    mode = str(os.environ.get("SOZLAB_OPENGL_BACKEND", "auto")).strip().lower()
    try:
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        )
    except Exception:
        pass
    try:
        if mode == "desktop":
            QtCore.QCoreApplication.setAttribute(
                QtCore.Qt.ApplicationAttribute.AA_UseDesktopOpenGL
            )
        elif mode == "auto":
            return
        else:
            QtCore.QCoreApplication.setAttribute(
                QtCore.Qt.ApplicationAttribute.AA_UseSoftwareOpenGL
            )
    except Exception:
        pass


def _configure_qt_webengine_runtime() -> None:
    """
    Configure QtWebEngine paths/flags early for embedded NGL reliability.
    """
    try:
        libexec = Path(
            QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.LibraryExecutablesPath)
        )
        translations = Path(
            QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
        )
        locale_dir = translations / "qtwebengine_locales"
        if "QTWEBENGINEPROCESS_PATH" not in os.environ:
            process = libexec / "QtWebEngineProcess"
            if process.exists():
                os.environ["QTWEBENGINEPROCESS_PATH"] = str(process)
        if "QTWEBENGINE_LOCALES_PATH" not in os.environ and locale_dir.exists():
            os.environ["QTWEBENGINE_LOCALES_PATH"] = str(locale_dir)
        # Avoid noisy \"Path override failed\" warnings when dictionary dir is absent.
        if "QTWEBENGINE_DICTIONARIES_PATH" not in os.environ and locale_dir.exists():
            os.environ["QTWEBENGINE_DICTIONARIES_PATH"] = str(locale_dir)
    except Exception:
        pass

    if "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            "--ignore-gpu-blocklist "
            "--enable-webgl "
            "--use-gl=angle "
            "--use-angle=swiftshader "
            "--disable-features=Vulkan"
        )


def main() -> None:
    _configure_qt_opengl_backend()
    _configure_qt_webengine_runtime()
    app_args = list(sys.argv) if sys.argv else ["sozlab-gui"]
    if not app_args:
        app_args = ["sozlab-gui"]
    app = QtWidgets.QApplication(app_args)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
