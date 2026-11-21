"""Left rail: Import → Define → Run. Every control shows units and its default; invalid states say how to fix them."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench import icons
from app.workbench.charts import CutoffGuide
from app.workbench.session import Issue, Session
from app.workbench.theme import Theme, set_prop
from app.workbench.widgets import Card, Chip, DropZone, FlowLayout, ParamField, make_label, dispose

Qt = QtCore.Qt


def _t() -> Theme:
    return Theme.instance()


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _fmt_time(ps: float) -> str:
    if ps is None or not math.isfinite(ps):
        return "n/a"
    return f"{ps / 1000:.3g} ns" if abs(ps) >= 1000 else f"{ps:.4g} ps"


# ------------------------------------------------------------------------------------------------ issues
class IssueList(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)

    def set_issues(self, issues: List[Issue]) -> None:
        while self._lay.count():
            it = self._lay.takeAt(0)
            if it.widget():
                dispose(it.widget())
        t = _t().t
        for issue in issues:
            fr = QtWidgets.QFrame()
            fr.setObjectName("Issue")
            fr.setProperty("level", issue.level)
            row = QtWidgets.QHBoxLayout(fr)
            row.setContentsMargins(10, 8, 8, 8)
            row.setSpacing(8)
            ic = QtWidgets.QLabel()
            color = t["err"] if issue.level == "error" else t["warn"] if issue.level == "warn" else t["accent"]
            ic.setPixmap(icons.pixmap("alert" if issue.level != "info" else "info", color, 17, 2.1))
            ic.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.addWidget(ic)
            lab = make_label(issue.message, "body", word_wrap=True)
            row.addWidget(lab, 1)
            if issue.fix and issue.fix_label:
                btn = QtWidgets.QPushButton(issue.fix_label)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(issue.fix)
                row.addWidget(btn)
            self._lay.addWidget(fr)
        self.setVisible(bool(issues))


# ------------------------------------------------------------------------------------------------ 1. import
class ImportCard(Card):
    filesChosen = QtCore.pyqtSignal(list)
    browse = QtCore.pyqtSignal()
    demo = QtCore.pyqtSignal()

    def __init__(self, session: Session, parent=None):
        super().__init__("1 · Import", "folder", "", parent=parent)
        self.session = session
        self.stack = QtWidgets.QStackedWidget()
        self.drop = DropZone()
        self.drop.filesDropped.connect(self.filesChosen)
        self.drop.browseClicked.connect(self.browse)
        self.drop.demoClicked.connect(self.demo)
        self.stack.addWidget(self.drop)  # 0
        # 1 summary
        self.summary = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(self.summary)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        self.file_rows = QtWidgets.QVBoxLayout()
        self.file_rows.setSpacing(4)
        sl.addLayout(self.file_rows)
        self.line1 = make_label("", "mono", word_wrap=True)
        self.line2 = make_label("", "caption", word_wrap=True)
        sl.addWidget(self.line1)
        sl.addWidget(self.line2)
        self.time_chip = Chip("", "neutral")
        sl.addWidget(self.time_chip, 0, Qt.AlignmentFlag.AlignLeft)
        self.notes = make_label("", "muted", word_wrap=True)
        sl.addWidget(self.notes)
        change = QtWidgets.QPushButton("Change files…")
        set_prop(change, "variant", "ghost")
        change.clicked.connect(self.browse)
        sl.addWidget(change, 0, Qt.AlignmentFlag.AlignLeft)
        self.stack.addWidget(self.summary)  # 1
        # 2 loading
        self.loading = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(self.loading)
        ll.setContentsMargins(0, 18, 0, 18)
        self.loading_label = make_label("Reading the files…", "muted")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ll.addWidget(self.loading_label)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)
        ll.addWidget(bar)
        self.stack.addWidget(self.loading)  # 2
        # 3 error
        self.error = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(self.error)
        el.setContentsMargins(0, 8, 0, 8)
        self.error_text = make_label("", "error", word_wrap=True, selectable=True)
        el.addWidget(self.error_text)
        el.addWidget(make_label("Check that the topology and trajectory describe the same atoms (same number of atoms).", "muted", word_wrap=True))
        retry = QtWidgets.QPushButton("Choose other files…")
        retry.clicked.connect(self.browse)
        el.addWidget(retry, 0, Qt.AlignmentFlag.AlignLeft)
        self.stack.addWidget(self.error)  # 3
        self.add(self.stack)
        session.inputsChanged.connect(self._on_loading)
        session.summaryReady.connect(self._on_summary)
        session.summaryFailed.connect(self._on_error)

    def refresh_theme(self) -> None:
        super().refresh_theme()
        if hasattr(self, "drop"):
            self.drop.refresh_theme()

    def _on_loading(self) -> None:
        self.stack.setCurrentIndex(2)
        self.set_status("loading…", "accent")

    def _on_error(self, msg: str) -> None:
        self.error_text.setText(msg)
        self.stack.setCurrentIndex(3)
        self.set_status("failed", "err")

    def _on_summary(self, s: dict) -> None:
        while self.file_rows.count():
            it = self.file_rows.takeAt(0)
            if it.widget():
                dispose(it.widget())
        for label, path in (("Topology", s["topology"]), ("Trajectory", s["trajectory"])):
            if path:
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(8)
                tag = Chip(label, "accent")
                tag.setFixedWidth(_t().px(84))
                tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
                name = make_label(os.path.basename(path), "mono")
                name.setToolTip(path)
                row.addWidget(tag)
                row.addWidget(name, 1)
                holder = QtWidgets.QWidget()
                holder.setLayout(row)
                self.file_rows.addWidget(holder)
        timing = s["timing"]
        n = int(s["n_frames"])
        dt = timing.get("dt_median")
        span = None if dt is None else dt * max(n - 1, 0)
        parts = [f"{_fmt_int(s['n_atoms'])} atoms", f"{_fmt_int(n)} frames"]
        if span is not None and n > 1:
            parts.append(f"{_fmt_time(span)}, every {_fmt_time(dt)}")
        self.line1.setText(" · ".join(parts))
        box = s["box"]
        box_txt = "no periodic box" if not box else "box " + " × ".join(f"{x:.1f}" for x in box[:3]) + " Å" + (
            "" if abs(box[3] - 90) + abs(box[4] - 90) + abs(box[5] - 90) < 1e-3 else " (triclinic)")
        sp = [f"{d['resname']} ×{_fmt_int(d['n_residues'])}" for d in s["species"] if d["kind"] == "water"]
        self.line2.setText(f"{box_txt} · solvent: {', '.join(sp[:2]) if sp else 'none detected'}")
        if s["time_assumed"]:
            self.time_chip.set("time stamps assumed (1 ps/frame)", "warn")
        elif not timing.get("uniform", True) and n > 1:
            self.time_chip.set("irregular time stamps", "warn")
        elif n > 1:
            self.time_chip.set("time stamps from file", "ok")
        else:
            self.time_chip.set("single frame", "warn")
        self.notes.setText(" ".join(s.get("notes", [])))
        self.notes.setVisible(bool(s.get("notes")))
        self.stack.setCurrentIndex(1)
        self.set_status("ready", "ok")


# ------------------------------------------------------------------------------------------------ 2. define
class ZoneCard(Card):
    def __init__(self, session: Session, parent=None):
        super().__init__("2 · Define the zone", "zone", "Solvent within a distance of part of the solute.",
                         help_key="zone", parent=parent)
        self.session = session
        self._chips: List[QtWidgets.QPushButton] = []
        self._building = False
        self.add(make_label("Around", "muted"))
        self.chip_host = QtWidgets.QWidget()
        self.chip_flow = FlowLayout(self.chip_host, 0, 6, 6)
        self.add(self.chip_host)
        self.selection = QtWidgets.QLineEdit()
        self.selection.setProperty("role", "mono")
        self.selection.setPlaceholderText("e.g.  protein and resid 10-20      or      resname LIG")
        self.selection.setToolTip("Any MDAnalysis selection. Pick a detected part above, or type your own.")
        self.add(self.selection)
        self.sel_status = make_label("", "caption", word_wrap=True)
        self.add(self.sel_status)
        # cutoff
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(5, 150)
        self.slider.setToolTip("Distance from the selection to the solvent's probe atom (usually the water oxygen)")
        self.cutoff = QtWidgets.QDoubleSpinBox()
        self.cutoff.setRange(0.5, 30.0)
        self.cutoff.setDecimals(2)
        self.cutoff.setSingleStep(0.1)
        self.cutoff.setSuffix(" Å")
        self.cutoff.setKeyboardTracking(False)
        self.cutoff.setValue(session.zone.cutoff)
        row.addWidget(make_label("Distance", "muted"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.cutoff)
        self.add(row)
        self.guide = CutoffGuide()
        self.guide.setFixedHeight(_t().px(250))
        self.add(self.guide)
        sug = QtWidgets.QHBoxLayout()
        self.suggest_btn = QtWidgets.QPushButton("Use the end of the first shell")
        self.suggest_btn.setToolTip("Picks the first minimum of the distance histogram above")
        self.suggest_btn.hide()
        sug.addWidget(self.suggest_btn)
        sug.addStretch(1)
        self.add(sug)
        self.description = make_label("", "caption", word_wrap=True, selectable=True)
        self.add(self.description)
        # advanced
        self.f_b = ParamField.text("Second selection (optional)", "", placeholder="e.g.  resname LIG", help_key=None,
                                   tooltip="Combine with another part of the solute")
        self.f_bcut = ParamField.spin("Second distance", 3.5, 0.5, 30.0, 0.1, "Å", decimals=2)
        self.f_comb = ParamField.combo("Combine", [("both (AND)", "AND"), ("either (OR)", "OR")], "AND")
        self.f_shells = ParamField.text("Extra radii (Å, comma separated)", "", placeholder="e.g.  5.0, 7.5",
                                        tooltip="Extra distances measured from the selection itself; the zone is the largest one")
        self.f_dynamic = ParamField.check("Re-evaluate the selection every frame", False,
                                          tooltip="Needed for geometric selections such as 'around 5 resname LIG'")
        self.f_probe = ParamField.combo("Solvent position", [("probe atom (e.g. oxygen)", "atom"), ("centre of geometry", "cog"),
                                                             ("centre of mass", "com")], "atom", help_key=None,
                                        tooltip="Which point of each solvent molecule is compared with the cutoff")
        self.f_names = ParamField.text("Solvent residue names", "SOL, WAT, TIP3, HOH", tooltip="Detected automatically from your system")
        self.f_probe_sel = ParamField.text("Probe atom selection", "name O OW OH2", tooltip="The atom used as each solvent molecule's position")
        self.f_ions = ParamField.check("Count ions as solvent", False)
        self.add_advanced([self.f_b, self.f_bcut, self.f_comb, self.f_shells, self.f_dynamic, self.f_probe, self.f_names,
                           self.f_probe_sel, self.f_ions], "Advanced settings")
        # wiring
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(self._push_advanced)
        self.selection.textEdited.connect(self._sel_edited)
        self.slider.valueChanged.connect(self._slider_changed)
        self.cutoff.valueChanged.connect(self._spin_changed)
        self.guide.cutoffChanged.connect(self._guide_changed)
        self.suggest_btn.clicked.connect(self._use_suggestion)
        for f in (self.f_b, self.f_bcut, self.f_comb, self.f_shells, self.f_dynamic, self.f_probe, self.f_names, self.f_probe_sel, self.f_ions):
            f.valueChanged.connect(lambda *_: self._debounce.start())
        session.summaryReady.connect(self._on_summary)
        session.selectionInfo.connect(self._on_selection_info)
        session.zoneChanged.connect(self._sync_from_session)
        session.curveReady.connect(self._on_curve)
        session.sceneReady.connect(self._on_scene)
        self.setEnabled(False)

    # ---------------------------------------------------------------------------- session → UI
    def _on_summary(self, s: dict) -> None:
        self.setEnabled(True)
        while self.chip_flow.count():
            it = self.chip_flow.takeAt(0)
            if it.widget():
                dispose(it.widget())
        self._chips = []
        for comp in s["components"]:
            b = QtWidgets.QPushButton(f"{comp['label']}")
            b.setCheckable(True)
            set_prop(b, "variant", "chip")
            b.setToolTip(f"{comp['selection']}  ({_fmt_int(comp['n_atoms'])} atoms)")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, sel=comp["selection"]: self._pick_component(sel))
            b.setProperty("selection", comp["selection"])
            self.chip_flow.addWidget(b)
            self._chips.append(b)
        custom = QtWidgets.QPushButton("Custom…")
        set_prop(custom, "variant", "chip")
        custom.clicked.connect(lambda: (self.selection.setFocus(), self.selection.selectAll()))
        self.chip_flow.addWidget(custom)
        self.f_names.set_value(", ".join(self.session.solvent.resnames))
        self.f_names.set_default(", ".join(self.session.solvent.resnames))
        self.f_probe_sel.set_value(self.session.solvent.probe_selection)
        self.f_probe_sel.set_default(self.session.solvent.probe_selection)
        # a project file may carry advanced zone settings: show them (silently; nothing is re-pushed to the session)
        z, sv = self.session.zone, self.session.solvent
        self.f_b.set_value(z.anchor_b)
        self.f_bcut.set_value(z.cutoff_b)
        self.f_comb.set_value(z.combine)
        self.f_shells.set_value(", ".join(f"{x:g}" for x in z.shells))
        self.f_dynamic.set_value(bool(z.updating))
        self.f_probe.set_value(sv.probe_position)
        self.f_ions.set_value(bool(sv.include_ions))
        self._sync_from_session()
        self.session.ask_curve(0)
        lim = s.get("min_image_limit")
        if lim:
            self.cutoff.setMaximum(max(2.0, lim * 1.5))

    def _sync_from_session(self) -> None:
        z = self.session.zone
        self._building = True
        if self.selection.text() != z.anchor:
            self.selection.setText(z.anchor)
        self.cutoff.setValue(z.cutoff)
        self.slider.setValue(int(round(z.cutoff * 10)))
        self.guide.set_cutoff(z.cutoff)
        for b in self._chips:
            on = b.property("selection") == z.anchor
            b.setChecked(on)
            set_prop(b, "checked", "true" if on else "false")
        self.description.setText(("Definition: " + z.describe()) if z.is_defined else "")
        self._building = False
        self.set_status("defined" if z.is_defined and self.session.selection_ok is not False else "incomplete",
                        "ok" if z.is_defined and self.session.selection_ok is not False else "warn")

    def _on_selection_info(self, info: dict) -> None:
        if info["ok"]:
            names = ", ".join(info["resnames"][:4]) + ("…" if len(info["resnames"]) > 4 else "")
            self.sel_status.setText(f"✓ {_fmt_int(info['n_atoms'])} atoms · {_fmt_int(info['n_residues'])} residues ({names})")
            self.sel_status.setStyleSheet(f"color: {_t().t['ok']}; font-size: 11px;")
            set_prop(self.selection, "state", "ok")
            self.session.ask_curve(getattr(self.session, "preview_frame", 0))
        else:
            self.sel_status.setText(info["message"])
            self.sel_status.setStyleSheet(f"color: {_t().t['err']}; font-size: 11px;")
            set_prop(self.selection, "state", "err" if self.selection.text().strip() else "")
        self._sync_from_session()

    def _on_curve(self, out: dict) -> None:
        if not out.get("ok"):
            return
        sug = self.guide.set_distances(out["nearest"], self.session.zone.cutoff)
        if sug:
            self.suggest_btn.setText(f"Use {sug:.2f} Å (end of the first shell)")
            self.suggest_btn.show()
        else:
            self.suggest_btn.hide()

    def _on_scene(self, out: dict) -> None:
        if out.get("ok"):
            self.guide.caption.setText(f"Frame {out['frame']}: {out['n_in_zone']} of {_fmt_int(out['n_solvent_total'])} solvent molecules are inside "
                                       "the zone. Drag the line to change the distance.")

    # ---------------------------------------------------------------------------- UI → session
    def _pick_component(self, selection: str) -> None:
        self.session.set_zone(anchor=selection)

    def _sel_edited(self, text: str) -> None:
        self.session.set_zone(anchor=text)

    def _slider_changed(self, v: int) -> None:
        if self._building:
            return
        self._building = True
        self.cutoff.setValue(v / 10.0)
        self._building = False
        self._set_cutoff(v / 10.0)

    def _spin_changed(self, v: float) -> None:
        if self._building:
            return
        self._building = True
        self.slider.setValue(int(round(v * 10)))
        self._building = False
        self._set_cutoff(v)

    def _guide_changed(self, r: float) -> None:
        if self._building:
            return
        self.cutoff.setValue(round(r, 2))

    def _set_cutoff(self, v: float) -> None:
        self.guide.set_cutoff(v)
        self.session.set_zone(cutoff=float(v))
        self.session.ask_scene(getattr(self.session, "preview_frame", 0))

    def _use_suggestion(self) -> None:
        s = self.guide.suggestion()
        if s:
            self.cutoff.setValue(round(s, 2))

    def _push_advanced(self) -> None:
        names = [n.strip() for n in self.f_names.value().split(",") if n.strip()]
        shells = []
        for tok in self.f_shells.value().replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    shells.append(float(tok))
                except ValueError:
                    pass
        self.session.set_solvent(resnames=names, probe_selection=self.f_probe_sel.value().strip() or "name O OW OH2",
                                 probe_position=self.f_probe.value(), include_ions=self.f_ions.value())
        self.session.set_zone(anchor_b=self.f_b.value().strip(), cutoff_b=float(self.f_bcut.value()), combine=self.f_comb.value(),
                              shells=shells, updating=bool(self.f_dynamic.value()), probe_mode="probe")
        self.session.ask_scene(getattr(self.session, "preview_frame", 0))


# ------------------------------------------------------------------------------------------------ 3. run
class RunCard(Card):
    """Run settings. The Run button itself lives in the sticky RunBar under the left rail, so it is always reachable."""

    def __init__(self, session: Session, parent=None):
        super().__init__("3 · Run settings", "bolt", "These stay visible next to the results, so every number can be traced to its settings.",
                         parent=parent)
        self.session = session
        self.f_start = ParamField.spin("First frame", 0, 0, 10_000_000, 1, "", tooltip="Index of the first frame to analyse (0 = start)")
        self.f_stop = ParamField.spin("Last frame", 0, 0, 10_000_000, 1, "", special="to the end",
                                      tooltip="Analyse up to (not including) this frame index; 0 means up to the last frame")
        self.f_stride = ParamField.spin("Use every n-th frame", 1, 1, 100_000, 1, "", help_key="stride_sensitivity",
                                        tooltip="A larger stride is faster but samples time more coarsely")
        self.add(self.f_stride)
        self.summary = make_label("", "body", word_wrap=True)
        self.add(self.summary)
        self.issues = IssueList()
        self.add(self.issues)
        # advanced
        self.f_gap = ParamField.spin("Tolerance for brief exits (ps)", 0.0, 0.0, 10_000.0, 0.5, "ps", decimals=2, help_key="visit_intermittent",
                                     tooltip="Absences shorter than this do not end a visit (stride independent). 0 = strict.")
        self.f_blocks = ParamField.spin("Blocks for uncertainty", 5, 2, 50, 1, "", help_key="uncertainty",
                                        tooltip="Number of contiguous blocks used for standard errors of residence times and rates")
        self.f_sens = ParamField.check("Report stride sensitivity", True, tooltip="Recomputes the headline numbers at 2× and 4× coarser sampling")
        self.f_workers = ParamField.spin("CPU workers (0 = auto)", 0, 0, 256, 1, "", tooltip="Processes start only when the run is long enough to benefit")
        self.f_dt = ParamField.spin("Frame interval override (ps/frame)", 0.0, 0.0, 1e6, 0.1, "ps", decimals=4,
                                    tooltip="Only for formats without time stamps. 0 = use the file.")
        self.f_density = ParamField.check("Compute a density map", False, help_key="density",
                                          tooltip="Time-averaged solvent density around the selection (superposed on it every frame)")
        self.f_spacing = ParamField.spin("Density grid spacing (Å)", 1.0, 0.25, 4.0, 0.25, "Å", decimals=2)
        self.f_padding = ParamField.spin("Density region beyond the selection (Å)", 12.0, 2.0, 40.0, 1.0, "Å", decimals=1)
        self.f_outdir = ParamField.text("Output folder", "", placeholder="results go here")
        browse = QtWidgets.QToolButton()
        browse.setText("Browse…")
        set_prop(browse, "variant", "ghost")
        browse.clicked.connect(self._browse_out)
        self.f_outdir.layout().addWidget(browse, 1, 2)
        self.add_advanced([self.f_start, self.f_stop, self.f_gap, self.f_blocks, self.f_sens, self.f_workers, self.f_dt, self.f_density,
                           self.f_spacing, self.f_padding, self.f_outdir], "Advanced settings")
        self.hint = make_label("", "caption", word_wrap=True)
        self.add(self.hint)
        for f in (self.f_start, self.f_stop, self.f_stride, self.f_gap, self.f_blocks, self.f_sens, self.f_workers, self.f_dt,
                  self.f_density, self.f_spacing, self.f_padding, self.f_outdir):
            f.valueChanged.connect(lambda *_: self._push())
        session.summaryReady.connect(self._on_summary)
        session.issuesChanged.connect(self._on_issues)
        session.runStarted.connect(lambda: self._set_running(True))
        session.resultReady.connect(lambda *_: self._set_running(False))
        session.runFailed.connect(lambda *_: self._set_running(False))
        self.setEnabled(False)

    def _browse_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the output folder", self.f_outdir.value() or "")
        if d:
            self.f_outdir.set_value(d, emit=True)

    def _on_summary(self, s: dict) -> None:
        self.setEnabled(True)
        n = int(s["n_frames"])
        self.f_stop.control.setMaximum(max(n, 1))
        self.f_start.control.setMaximum(max(n - 1, 0))
        for f, v in ((self.f_start, 0), (self.f_stop, 0), (self.f_stride, self.session.run.stride)):
            f.set_default(v)
            f.set_value(v)
        self.f_outdir.set_value(self.session.run.output_dir)
        self.f_outdir.set_default(self.session.run.output_dir)
        self.f_dt.setVisible(bool(s.get("time_assumed")))
        self._refresh_summary()

    def _push(self) -> None:
        stop = int(self.f_stop.value())
        self.session.set_run(
            frame_start=int(self.f_start.value()), frame_stop=None if stop <= 0 else stop, stride=int(self.f_stride.value()),
            gap_time=float(self.f_gap.value()), blocks=int(self.f_blocks.value()), stride_sensitivity=bool(self.f_sens.value()),
            workers=int(self.f_workers.value()), dt_override=float(self.f_dt.value()) or None, density=bool(self.f_density.value()),
            density_spacing=float(self.f_spacing.value()), density_padding=float(self.f_padding.value()),
            output_dir=self.f_outdir.value().strip() or self.session.run.output_dir,
        )
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        s = self.session
        if s.summary is None:
            return
        n = s.effective_frames()
        dt = s.effective_dt()
        parts = [f"<b>{_fmt_int(n)}</b> frames"]
        if dt:
            parts.append(f"every <b>{_fmt_time(dt)}</b>")
            parts.append(f"spanning <b>{_fmt_time(dt * max(n - 1, 0))}</b>")
        self.summary.setText(" · ".join(parts))
        self.summary.setTextFormat(Qt.TextFormat.RichText)

    def _on_issues(self, issues: List[Issue]) -> None:
        shown = [i for i in issues if i.level in ("error", "warn")]
        self.issues.set_issues(shown)
        errs = [i for i in issues if i.level == "error"]
        if errs:
            self.hint.setText("Fix the red items above to enable Run.")
        elif self.session.summary is not None:
            self.hint.setText("Ready. Results appear on the right; nothing is written until you export.")
        self.set_status("blocked" if errs else "ready", "err" if errs else "ok")

    def _set_running(self, running: bool) -> None:
        for f in (self.f_start, self.f_stop, self.f_stride, self.f_gap, self.f_blocks, self.f_sens, self.f_workers, self.f_dt,
                  self.f_density, self.f_spacing, self.f_padding, self.f_outdir):
            f.setEnabled(not running)
