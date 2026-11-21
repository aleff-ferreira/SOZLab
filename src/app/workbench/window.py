"""The SOZLab Workbench main window: one screen, five steps (Import → Define → Run → Inspect → Export).

Layout (all of it visible at 1366×768): inputs and run settings on the left rail, always in view; the 3D system and
the results in the middle; trust, run history and export on the right rail."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench import icons
from app.workbench.panels_left import ImportCard, RunCard, ZoneCard
from app.workbench.panels_results import ResultsPanel
from app.workbench.panels_right import ExportPanel, HistoryPanel, TrustPanel
from app.workbench.session import STEPS, Session
from app.workbench.theme import Theme, set_prop
from app.workbench.widgets import Card, Chip, Segmented, Stepper, ToastHost, make_label

Qt = QtCore.Qt

FILE_FILTER = ("Structure and trajectory files (*.gro *.pdb *.psf *.prmtop *.parm7 *.tpr *.top *.mol2 *.xyz *.xtc *.trr *.dcd *.nc *.ncdf "
               "*.crd *.mdcrd *.trj *.lammpstrj *.h5md);;SOZLab projects (*.json);;All files (*)")


def _t() -> Theme:
    return Theme.instance()


def _make_scene_widget(parent=None):
    """The 3D view; falls back to a plain message if the renderer cannot be created (never blocks the rest)."""
    try:
        from app.workbench.viewport3d import SceneView

        return SceneView(parent)
    except Exception as exc:  # pragma: no cover - defensive
        lab = make_label(f"The 3D view is unavailable ({exc}). Charts and results still work.", "muted", word_wrap=True)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.set_scene = lambda *_a, **_k: None  # type: ignore[attr-defined]
        lab.clear_scene = lambda: None  # type: ignore[attr-defined]
        lab.refresh_theme = lambda: None  # type: ignore[attr-defined]
        return lab


# ------------------------------------------------------------------------------------------------ run bar
class RunBar(QtWidgets.QFrame):
    """Sticky footer of the left rail: the Run button is always reachable, with the reason when it is not."""

    runClicked = QtCore.pyqtSignal()
    cancelClicked = QtCore.pyqtSignal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setObjectName("RunBar")
        self.session = session
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.run_btn = QtWidgets.QPushButton("Run analysis")
        set_prop(self.run_btn, "variant", "go")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setToolTip("Ctrl+R")
        self.run_btn.clicked.connect(self.runClicked)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        set_prop(self.cancel_btn, "variant", "danger")
        self.cancel_btn.setToolTip("Esc — keeps nothing from this run")
        self.cancel_btn.clicked.connect(self.cancelClicked)
        self.cancel_btn.hide()
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.cancel_btn)
        v.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        v.addWidget(self.progress)
        self.status = make_label("", "caption", word_wrap=True)
        v.addWidget(self.status)
        self.stale = Chip("Settings changed since the last run", "warn")
        self.stale.hide()
        v.addWidget(self.stale, 0, Qt.AlignmentFlag.AlignLeft)
        self._sticky = ""  # the outcome of the last run stays on screen until the user changes something
        for sig in (session.issuesChanged, session.stepsChanged):
            sig.connect(lambda *_: self.refresh())
        for sig in (session.zoneChanged, session.settingsChanged, session.summaryReady):
            sig.connect(lambda *_: self._clear_sticky())
        session.runStarted.connect(self._on_started)
        session.runProgress.connect(self._on_progress)
        session.resultReady.connect(self._on_finished)
        session.runFailed.connect(self._on_failed)
        session.runCancelled.connect(self._on_cancelled)
        self.refresh()

    def refresh(self) -> None:
        s = self.session
        running = s.running
        self.run_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress.setVisible(running)
        can = s.can_run()
        self.run_btn.setEnabled(can)
        self.run_btn.setText("Run again" if s.result is not None else "Run analysis")
        self.run_btn.setIcon(icons.icon("play", "#06201C" if _t().is_dark else "#FFFFFF", 18, 2.2))
        errs = [i for i in s.issues() if i.level == "error"]
        if running:
            pass
        elif self._sticky:
            self.status.setText(self._sticky)
        elif s.summary is None:
            self.status.setText("Open a topology and trajectory to begin.")
        elif errs:
            self.status.setText("To run: " + errs[0].message)
        else:
            self.status.setText("Ready. Nothing is written to disk until you export.")
        self.run_btn.setToolTip("Ctrl+R" if can else (errs[0].message if errs else "Open a topology and trajectory first"))
        self.stale.setVisible(s.is_stale() and not running)

    def _clear_sticky(self) -> None:
        self._sticky = ""
        self.refresh()

    def _on_started(self) -> None:
        self._sticky = ""
        self.progress.setRange(0, 0)
        self.status.setText("Starting…")
        self.refresh()

    def _on_progress(self, cur: int, tot: int, msg: str) -> None:
        if tot > 0:
            self.progress.setRange(0, tot)
            self.progress.setValue(cur)
        self.status.setText(msg)

    def _on_finished(self, result) -> None:
        self._sticky = "Done. Results are on the right; nothing is written until you export." if getattr(result, "complete", True)             else "Cancelled: the results shown are partial and are marked as such."
        self.refresh()

    def _on_failed(self, msg: str) -> None:
        self._sticky = f"The run failed: {msg}"
        self.refresh()

    def _on_cancelled(self) -> None:
        self._sticky = "Cancelled. Nothing was kept; change the settings or run again."
        self.refresh()


# ------------------------------------------------------------------------------------------------ stage
class StageView(QtWidgets.QFrame):
    """The 3D view plus frame transport. The slider walks the analysed frames; the timeline cursor follows."""

    frameChanged = QtCore.pyqtSignal(int)  # index into the analysis window
    foldToggled = QtCore.pyqtSignal(bool)  # True = the 3D canvas is folded away (only the transport bar remains)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setObjectName("Stage")
        self.session = session
        self._frames: List[int] = []
        self._waiting = False
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(self.empty)
        el.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo = QtWidgets.QLabel()
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(self.logo)
        self.empty_title = make_label("Your system will appear here", "h1")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_text = make_label("Add a topology and trajectory on the left. The solute, the solvent around it and the zone you define are drawn "
                                     "live, and every change you make is reflected here immediately.", "muted", word_wrap=True)
        self.empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_text.setMaximumWidth(_t().px(440))
        el.addWidget(self.empty_title)
        el.addWidget(self.empty_text, 0, Qt.AlignmentFlag.AlignCenter)
        self.scene = _make_scene_widget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.scene)
        # the renderer asks for 900x600; the splitter, not the widget, decides how tall the stage is
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.stack.setMinimumHeight(_t().px(150))
        v.addWidget(self.stack, 1)
        # transport
        bar = QtWidgets.QFrame()
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(12, 6, 12, 8)
        bl.setSpacing(10)
        self.play = QtWidgets.QToolButton()
        set_prop(self.play, "variant", "bar")
        self.play.setCheckable(True)
        self.play.setToolTip("Play through the analysed frames (Space)")
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip("Frame (← → to step)")
        self.readout = make_label("", "mono")
        self.readout.setMinimumWidth(_t().px(260))
        self.shot = QtWidgets.QToolButton()
        set_prop(self.shot, "variant", "bar")
        self.shot.setToolTip("Save a picture of this view")
        self.fold = QtWidgets.QToolButton()
        set_prop(self.fold, "variant", "bar")
        self.fold.setCheckable(True)
        self.fold.setToolTip("Fold the 3D view away to give the charts the full height (and back)")
        self.fold.toggled.connect(self._on_fold)
        bl.addWidget(self.play)
        bl.addWidget(self.slider, 1)
        bl.addWidget(self.readout)
        bl.addWidget(self.shot)
        bl.addWidget(self.fold)
        v.addWidget(bar)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(110)
        self.timer.timeout.connect(self._tick)
        self.play.toggled.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._slider_changed)
        self.shot.clicked.connect(self._save_picture)
        session.sceneReady.connect(self._on_scene)
        session.summaryReady.connect(self._on_summary)
        session.inputsChanged.connect(self._on_reset)
        session.settingsChanged.connect(self._on_window_changed)
        self.refresh_theme()
        self._enable(False)

    # ------------------------------------------------------------------------------------ state
    def refresh_theme(self) -> None:
        t = _t()
        self.logo.setPixmap(icons.logo_pixmap(t.t["solvent"], t.t["solute"], 76))
        self.play.setIcon(icons.icon("pause" if self.play.isChecked() else "play", t.t["ink"], 18))
        self.shot.setIcon(icons.icon("camera", t.t["ink2"], 18))
        self.fold.setIcon(icons.icon("chevron_down" if self.fold.isChecked() else "chevron_up", t.t["ink2"], 18))
        if hasattr(self.scene, "refresh_theme"):
            self.scene.refresh_theme()

    def _on_fold(self, folded: bool) -> None:
        self.stack.setVisible(not folded)
        self.fold.setIcon(icons.icon("chevron_down" if folded else "chevron_up", _t().t["ink2"], 18))
        self.foldToggled.emit(folded)

    def _enable(self, on: bool) -> None:
        for w in (self.play, self.slider, self.shot):
            w.setEnabled(on)

    def _on_reset(self) -> None:
        self.timer.stop()
        self.play.setChecked(False)
        self.stack.setCurrentIndex(0)
        self._enable(False)
        self.readout.setText("")
        if hasattr(self.scene, "clear_scene"):
            self.scene.clear_scene()

    def _on_summary(self, summary: dict) -> None:
        self._on_window_changed()
        self.stack.setCurrentIndex(1)
        self._enable(True)
        self.slider.setValue(0)
        self.request(0)

    def _on_window_changed(self) -> None:
        frames = self.session.window_frames()
        if frames == self._frames:
            return
        self._frames = frames
        pos = min(self.slider.value(), max(len(frames) - 1, 0))
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(frames) - 1, 0))
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.request(pos)

    # ------------------------------------------------------------------------------------ frames
    def request(self, index: Optional[int] = None) -> None:
        if not self._frames:
            return
        i = self.slider.value() if index is None else int(index)
        i = max(0, min(i, len(self._frames) - 1))
        frame = self._frames[i]
        prev = self._frames[i - 1] if i > 0 else None
        self._waiting = True
        self.session.set_preview_frame(frame, prev)

    def set_index(self, i: int) -> None:
        if self._frames:
            self.slider.setValue(max(0, min(int(i), len(self._frames) - 1)))

    def _slider_changed(self, i: int) -> None:
        self.request(i)
        self.frameChanged.emit(i)

    def _toggle_play(self, on: bool) -> None:
        self.play.setIcon(icons.icon("pause" if on else "play", _t().t["ink"], 18))
        if on and len(self._frames) > 1:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self) -> None:
        if self._waiting:  # never queue frames faster than they can be drawn
            return
        i = self.slider.value() + 1
        self.slider.setValue(0 if i >= self.slider.maximum() + 1 else i)

    def _on_scene(self, out: dict) -> None:
        self._waiting = False
        if not out.get("ok"):
            self.readout.setText("")
            return
        if hasattr(self.scene, "set_scene"):
            self.scene.set_scene(out)
        t = out.get("time_ps")
        extra = ""
        if out.get("n_entered") or out.get("n_left"):
            extra = f"  (+{out['n_entered']} / −{out['n_left']})"
        i = self.slider.value()
        self.readout.setText(f"frame {out['frame']}  ·  {t:.4g} ps  ·  {out['n_in_zone']} in zone{extra}" if t is not None else f"frame {out['frame']}")
        self.readout.setToolTip(f"Frame {i + 1} of {len(self._frames)} analysed; green = entered since the previous analysed frame, coral = left")

    def _save_picture(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save a picture of the view", "sozlab_view.png", "PNG image (*.png)")
        if path:
            self.stack.currentWidget().grab().save(path, "PNG")


# ------------------------------------------------------------------------------------------------ rails
class Rail(QtWidgets.QWidget):
    """Segmented switch + stacked scroll pages (right rail)."""

    def __init__(self, pages: List[tuple], parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        self.seg = Segmented([(title, key) for key, title, _w in pages], pages[0][0])
        v.addWidget(self.seg)
        self.stack = QtWidgets.QStackedWidget()
        self.keys = [k for k, _t_, _w in pages]
        for _k, _title, widget in pages:
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            host = QtWidgets.QWidget()
            host.setObjectName("Transparent")
            hl = QtWidgets.QVBoxLayout(host)
            hl.setContentsMargins(0, 0, 6, 0)
            hl.addWidget(widget)
            sa.setWidget(host)
            self.stack.addWidget(sa)
        v.addWidget(self.stack, 1)
        self.seg.changed.connect(lambda key: self.stack.setCurrentIndex(self.keys.index(key)))

    def show_page(self, key: str) -> None:
        self.seg.set_value(key)
        self.stack.setCurrentIndex(self.keys.index(key))


# ------------------------------------------------------------------------------------------------ window
class WorkbenchWindow(QtWidgets.QMainWindow):
    def __init__(self, session: Optional[Session] = None, demo_dir: Optional[str] = None):
        super().__init__()
        self.theme = _t()
        self.session = session or Session(self)
        s = self.session
        self.demo_dir = demo_dir or os.path.join(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation) or
                                                 os.path.expanduser("~/.sozlab"), "demo")
        self.setWindowTitle("SOZLab Workbench")
        self.setAcceptDrops(True)
        self.setMinimumSize(1080, 660)
        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_topbar())
        # ---- left rail
        self.import_card = ImportCard(s)
        self.zone_card = ZoneCard(s)
        self.run_card = RunCard(s)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QtWidgets.QWidget()
        host.setObjectName("Transparent")
        hl = QtWidgets.QVBoxLayout(host)
        hl.setContentsMargins(14, 12, 8, 12)
        hl.setSpacing(12)
        for c in (self.import_card, self.zone_card, self.run_card):
            hl.addWidget(c)
        hl.addStretch(1)
        left_scroll.setWidget(host)
        self.left_scroll = left_scroll
        self.runbar = RunBar(s)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        ll.addWidget(left_scroll, 1)
        ll.addWidget(self.runbar)
        # ---- centre
        self.stage = StageView(s)
        self.results = ResultsPanel(s)
        centre = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        centre.setChildrenCollapsible(False)
        stage_host = QtWidgets.QWidget()
        sh = QtWidgets.QVBoxLayout(stage_host)
        sh.setContentsMargins(0, 12, 0, 0)
        sh.addWidget(self.stage)
        # the results scroll vertically when the window is too short for them (they are never squeezed into overlapping)
        res_host = QtWidgets.QScrollArea()
        res_host.setWidgetResizable(True)
        res_host.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        res_host.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        res_inner = QtWidgets.QWidget()
        res_inner.setObjectName("Transparent")
        rh = QtWidgets.QVBoxLayout(res_inner)
        rh.setContentsMargins(0, 0, 4, 12)
        rh.addWidget(self.results)
        res_host.setWidget(res_inner)
        self.results_scroll = res_host
        centre.addWidget(stage_host)
        centre.addWidget(res_host)
        centre.setStretchFactor(0, 5)
        centre.setStretchFactor(1, 6)
        centre.setSizes([290, 380])
        self.centre = centre
        # ---- right rail
        self.trust = TrustPanel(s)
        self.history = HistoryPanel(s)
        self.export = ExportPanel(s)
        self.right = Rail([("trust", "Trust", self.trust), ("runs", "Runs", self.history), ("export", "Export", self.export)])
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(8, 12, 14, 12)
        rl.addWidget(self.right)
        self.body = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.body.setChildrenCollapsible(False)
        self.body.addWidget(left)
        self.body.addWidget(centre)
        self.body.addWidget(right)
        self.body.setStretchFactor(1, 1)
        self.body.setSizes([418, 632, 316])
        left.setMinimumWidth(_t().px(404))
        left.setMaximumWidth(_t().px(540))
        right.setMinimumWidth(_t().px(304))
        right.setMaximumWidth(_t().px(460))
        outer.addWidget(self.body, 1)
        outer.addWidget(self._build_statusbar())
        self.toast = ToastHost(root)
        self._wire()
        self._on_theme()
        self._sync_steps()
        self.resize(1366, 800)

    # ------------------------------------------------------------------------------------ chrome
    def _build_topbar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setObjectName("TopBar")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 8, 12, 8)
        row.setSpacing(12)
        self.logo = QtWidgets.QLabel()
        row.addWidget(self.logo)
        brand = QtWidgets.QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(make_label("SOZLab", "brand"))
        self.brand_sub = make_label("solvent occupancy workbench", "caption")
        brand.addWidget(self.brand_sub)
        row.addLayout(brand)
        row.addSpacing(18)
        self.stepper = Stepper(STEPS)
        self.stepper.setMinimumWidth(_t().px(520))
        row.addWidget(self.stepper, 1)
        self.project_chip = Chip("no files", "neutral")
        row.addWidget(self.project_chip)
        self.btn_file = QtWidgets.QToolButton()
        self.btn_file.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_file.setToolTip("Open files or a project (Ctrl+O)")
        menu = QtWidgets.QMenu(self.btn_file)
        self.act_open = menu.addAction("Open topology / trajectory…")
        self.act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.act_open_project = menu.addAction("Open a project file…")
        menu.addSeparator()
        self.act_save_project = menu.addAction("Save project…")
        self.act_save_project.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.act_demo = menu.addAction("Load the demo system")
        self.btn_file.setMenu(menu)
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setToolTip("Light / dark")
        self.btn_motion = QtWidgets.QToolButton()
        self.btn_motion.setCheckable(True)
        self.btn_motion.setToolTip("Reduce motion")
        self.btn_help = QtWidgets.QToolButton()
        self.btn_help.setToolTip("Short guide (F1)")
        for b in (self.btn_file, self.btn_motion, self.btn_theme, self.btn_help):
            set_prop(b, "variant", "bar")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        return bar

    def _build_statusbar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setObjectName("StatusBar")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 5, 16, 5)
        self.status_label = make_label("Ready.", "caption")
        row.addWidget(self.status_label, 1)
        self.task_bar = QtWidgets.QProgressBar()
        self.task_bar.setFixedWidth(_t().px(160))
        self.task_bar.setTextVisible(False)
        self.task_bar.hide()
        row.addWidget(self.task_bar)
        self.prov_chip = Chip("", "neutral")
        self.prov_chip.hide()
        row.addWidget(self.prov_chip)
        from engine._version import __version__

        row.addWidget(make_label(f"v{__version__}", "caption"))
        return bar

    # ------------------------------------------------------------------------------------ wiring
    def _wire(self) -> None:
        s = self.session
        ic, zc = self.import_card, self.zone_card
        ic.browse.connect(self.browse_files)
        ic.filesChosen.connect(s.open_paths)
        ic.demo.connect(self.load_demo)
        self.act_open.triggered.connect(self.browse_files)
        self.act_open_project.triggered.connect(self.open_project)
        self.act_save_project.triggered.connect(self.save_project)
        self.act_demo.triggered.connect(self.load_demo)
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_motion.toggled.connect(self._set_motion)
        self.btn_help.clicked.connect(self.show_guide)
        self.runbar.runClicked.connect(s.start_run)
        self.runbar.cancelClicked.connect(s.cancel_run)
        self.stepper.stepClicked.connect(self._goto_step)
        s.stepsChanged.connect(self._sync_steps)
        s.message.connect(self.notify)
        s.inputsChanged.connect(lambda: self.results.clear())
        s.summaryReady.connect(self._on_summary)
        s.summaryFailed.connect(lambda m: self.notify(f"Could not open the files: {m}", "err"))
        s.resultReady.connect(self._on_result)
        s.runFailed.connect(lambda m: self.notify(f"The run failed: {m}", "err"))
        s.runProgress.connect(lambda c, t, m: self.status_label.setText(m))
        s.taskProgress.connect(self._on_task_progress)
        s.taskDone.connect(self._on_task_done)
        s.taskFailed.connect(self._on_task_failed)
        s.seriesReady.connect(self.results.set_preview)
        s.snapshotsChanged.connect(self._sync_overlay)
        self.history.compareChanged.connect(lambda snap: self.results.timeline.set_compare(
            None if snap is None else snap.times_ps, None if snap is None else snap.n_solvent))
        self.results.cursorMoved.connect(self.stage.set_index)
        self.stage.frameChanged.connect(self.results.set_cursor)
        self.stage.foldToggled.connect(self._fold_stage)
        self.theme.changed.connect(self._on_theme)
        # live preview: any change to the zone or settings refreshes the scene and the occupancy preview
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(260)
        self._preview_timer.timeout.connect(s.ask_series)
        for sig in (s.zoneChanged, s.settingsChanged, s.selectionInfo):
            sig.connect(lambda *_: self._preview_timer.start())
        s.zoneChanged.connect(lambda: self.stage.request())
        for key, seq in (("run", "Ctrl+R"), ("cancel", "Esc"), ("export", "Ctrl+E"), ("play", "Space"), ("guide", "F1"), ("theme", "Ctrl+Shift+L")):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect({"run": s.start_run, "cancel": self._escape, "export": lambda: self._goto_step("export"),
                                  "play": lambda: self.stage.play.toggle() if self.stage.play.isEnabled() else None,
                                  "guide": self.show_guide, "theme": self.toggle_theme}[key])

    def _escape(self) -> None:
        if self.session.running:
            self.session.cancel_run()

    # ------------------------------------------------------------------------------------ actions
    def browse_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose a topology and (optionally) a trajectory", "", FILE_FILTER)
        if paths:
            if len(paths) == 1 and paths[0].lower().endswith(".json"):
                self._open_project_path(paths[0])
            else:
                self.session.open_paths(paths)

    def load_demo(self) -> None:
        self.import_card.stack.setCurrentIndex(2)
        self.import_card.loading_label.setText("Generating a synthetic hydrated solute (about 10 s, once)…")
        self.session.use_demo(self.demo_dir)

    def open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a SOZLab project", "", "SOZLab projects (*.json);;All files (*)")
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path: str) -> None:
        try:
            self.session.load_project(path)
        except Exception as exc:
            self.notify(f"Could not read the project: {exc}", "err")

    def save_project(self) -> None:
        if self.session.summary is None:
            self.notify("Open some files first; there is nothing to save yet.", "warn")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the project", "sozlab_project.json", "SOZLab projects (*.json)")
        if path:
            self.session.save_project(path)
            self.notify(f"Project saved: {path}", "ok")

    def toggle_theme(self) -> None:
        self.theme.set_mode("light" if self.theme.is_dark else "dark")

    def _set_motion(self, on: bool) -> None:
        self.theme.reduce_motion = on
        self.stepper.update()

    def show_guide(self) -> None:
        text = ("<b>Five steps, left to right.</b><br><br>"
                "<b>1 Import</b> — drop a topology and trajectory (or load the demo).<br>"
                "<b>2 Define</b> — pick what the zone surrounds and drag the line to set the distance; the 3D view and the occupancy preview update live.<br>"
                "<b>3 Run</b> — the settings stay visible next to the results; 'Advanced' holds everything else, with defaults and Reset.<br>"
                "<b>4 Inspect</b> — every number has an uncertainty, a unit and a trust badge; click the ? for its definition.<br>"
                "<b>5 Export</b> — data, a report and frames, with a manifest so the folder can be verified later.<br><br>"
                "Shortcuts: Ctrl+O open · Ctrl+R run · Esc cancel · Ctrl+E export · Space play · Ctrl+Shift+L light/dark.")
        QtWidgets.QMessageBox.information(self, "SOZLab in one minute", text)

    # ------------------------------------------------------------------------------------ session → chrome
    def notify(self, text: str, kind: str = "info") -> None:
        self.status_label.setText(text)
        self.toast.show_message(text, kind)

    def _sync_steps(self) -> None:
        st = self.session.step_state()
        for key, _title in STEPS:
            self.stepper.set_state(key, st[key])
        self.import_card.set_active(st["import"] == "active")
        self.zone_card.set_active(st["define"] == "active")
        self.run_card.set_active(st["run"] == "active")

    def _goto_step(self, key: str) -> None:
        if key in ("import", "define", "run"):
            card = {"import": self.import_card, "define": self.zone_card, "run": self.run_card}[key]
            self.left_scroll.ensureWidgetVisible(card, 0, 20)
        elif key == "inspect":
            self.right.show_page("trust")
        elif key == "export":
            self.right.show_page("export")

    def _on_summary(self, summary: dict) -> None:
        names = [os.path.basename(p) for p in (summary["topology"], summary["trajectory"]) if p]
        self.project_chip.set(" + ".join(names), "accent")
        self.project_chip.setToolTip("\n".join(p for p in (summary["topology"], summary["trajectory"]) if p))
        self.status_label.setText(f"{summary['n_atoms']:,} atoms · {summary['n_frames']:,} frames")
        QtCore.QTimer.singleShot(60, self._scroll_to_zone)

    def _fold_stage(self, folded: bool) -> None:
        sizes = self.centre.sizes()
        total = sum(sizes)
        if folded:
            self._stage_sizes = sizes
            bar = self.stage.minimumSizeHint().height() + 12
            self.centre.setSizes([bar, max(total - bar, 1)])
        else:
            prev = getattr(self, "_stage_sizes", None)
            self.centre.setSizes(prev if prev and sum(prev) > 0 else [290, max(total - 290, 1)])

    def _scroll_to_zone(self) -> None:
        try:  # the window may be gone by the time the timer fires
            self.left_scroll.verticalScrollBar().setValue(max(0, self.zone_card.y() - 14))
        except RuntimeError:
            pass

    def _on_result(self, result) -> None:
        if not getattr(result, "complete", True):
            self.notify("The run was cancelled; results are partial and are not exported unless you ask.", "warn")
        else:
            n_bad = sum(1 for i in result.qc_summary.get("trust", []) if i["level"] in ("caution", "warning"))
            self.notify("Run complete." + (f" {n_bad} caveat(s) to read in the Trust panel." if n_bad else " No caveats raised."), "ok" if not n_bad else "warn")
        self.results.set_result(result)
        prov = result.provenance or {}
        self.prov_chip.set(f"run {str(prov.get('run_id', ''))[:8]} · config {str(prov.get('config_sha256', ''))[:8]}", "accent")
        self.prov_chip.setToolTip("Identifies the exact software version, settings and input files behind the numbers on screen")
        self.prov_chip.show()
        self.right.show_page("trust")
        self._sync_steps()
        self.stage.request()

    def _sync_overlay(self) -> None:
        pass

    def _on_task_progress(self, name: str, cur: int, tot: int, msg: str) -> None:
        self.task_bar.show()
        if tot > 0:
            self.task_bar.setRange(0, tot)
            self.task_bar.setValue(cur)
        else:
            self.task_bar.setRange(0, 0)
        self.status_label.setText(msg)
        if name == "demo":
            self.import_card.loading_label.setText(msg or "Generating the demo system…")

    def _on_task_done(self, name: str, result) -> None:
        self.task_bar.hide()
        if name == "export_data" or name == "export_report":
            self.notify("Results saved. The folder includes a manifest you can verify later.", "ok")
        elif name == "extract":
            self.notify("Frames extracted (inputs verified by content before writing).", "ok")

    def _on_task_failed(self, name: str, msg: str) -> None:
        self.task_bar.hide()
        if name == "demo":
            self.import_card.stack.setCurrentIndex(0)
        self.notify(f"{name.replace('_', ' ')} failed: {msg}", "err")

    # ------------------------------------------------------------------------------------ theme
    def _on_theme(self) -> None:
        t = self.theme.t
        self.logo.setPixmap(icons.logo_pixmap(t["solvent"], t["solute"], 34))
        self.btn_file.setIcon(icons.icon("folder_open", t["ink2"], 20))
        self.btn_theme.setIcon(icons.icon("sun" if self.theme.is_dark else "moon", t["ink2"], 20))
        self.btn_motion.setIcon(icons.icon("wave", t["ink2"], 20))
        self.btn_help.setIcon(icons.icon("help", t["ink2"], 20))
        for card in self.findChildren(Card):
            card.refresh_theme()
        self.zone_card.guide.refresh_theme()
        self.import_card.drop.refresh_theme()
        self.results.refresh_theme()
        self.stage.refresh_theme()
        self.runbar.refresh()
        self.trust.refresh()
        self.stepper.update()

    # ------------------------------------------------------------------------------------ drag & drop, lifecycle
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            if len(paths) == 1 and paths[0].lower().endswith(".json"):
                self._open_project_path(paths[0])
            else:
                self.session.open_paths(paths)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if hasattr(self, "toast"):
            self.toast.reposition()

    def closeEvent(self, e):  # noqa: N802
        if self.session.running:
            ans = QtWidgets.QMessageBox.question(self, "Analysis running", "An analysis is still running. Cancel it and quit?")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.session.cancel_run()
        self.session.shutdown()
        e.accept()
