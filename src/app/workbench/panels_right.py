"""Right rail: Trust (when can each number be used?), Runs (history and comparison), Export (data, report, frames)."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench import icons
from app.workbench.panels_left import IssueList
from app.workbench.session import Issue, Session, Snapshot
from app.workbench.theme import Theme, set_prop
from app.workbench.widgets import Card, Chip, DefinitionPopup, FlowLayout, Segmented, make_label, dispose

Qt = QtCore.Qt


def _t() -> Theme:
    return Theme.instance()


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _open_folder(path: str) -> None:
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


# ------------------------------------------------------------------------------------------------ trust
class TrustPanel(QtWidgets.QWidget):
    DEFINITION_CHIPS = (("zone", "Zone"), ("n_solvent", "Solvent in zone"), ("occupancy_fraction", "Occupancy"),
                        ("visit", "Visit"), ("survival", "Survival S(τ)"), ("tau_survival_integral", "Residence time τ"),
                        ("entry_rate_per_time", "Entry rate"), ("uncertainty", "Uncertainty"),
                        ("stride_sensitivity", "Stride sensitivity"), ("density", "Density map"))

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.head = make_label("Before you quote a number", "h2")
        lay.addWidget(self.head)
        self.ok_note = make_label("", "muted", word_wrap=True)
        lay.addWidget(self.ok_note)
        self.list = IssueList()
        lay.addWidget(self.list)
        lay.addWidget(make_label("What each number means", "h2"))
        lay.addWidget(make_label("Every quantity has a definition, a unit and a condition under which it can be trusted.", "caption", word_wrap=True))
        host = QtWidgets.QWidget()
        flow = FlowLayout(host, 0, 6, 6)
        for key, title in self.DEFINITION_CHIPS:
            b = QtWidgets.QPushButton(title)
            set_prop(b, "variant", "chip")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=key, w=b: self._show(k, w))
            flow.addWidget(b)
        lay.addWidget(host)
        lay.addStretch(1)
        session.issuesChanged.connect(lambda *_: self.refresh())
        session.resultReady.connect(lambda *_: self.refresh())
        self.refresh()

    def _show(self, key: str, anchor: QtWidgets.QWidget) -> None:
        pop = DefinitionPopup(key, anchor)
        pop.move(anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + 6)))
        pop.show()

    def refresh(self) -> None:
        res = self.session.result
        items: List[Issue] = []
        if res is not None:
            order = {"warning": 0, "caution": 1, "info": 2}
            for it in sorted(res.qc_summary.get("trust", []), key=lambda i: order.get(i["level"], 3)):
                level = {"warning": "error", "caution": "warn", "info": "info"}.get(it["level"], "info")
                items.append(Issue(level, it["id"], it["message"]))
            for w in res.warnings[:6]:
                items.append(Issue("info", "warning", w))
            n_bad = sum(1 for i in items if i.level in ("error", "warn"))
            self.head.setText("Before you quote a number")
            self.ok_note.setText("No caveats were raised for this run: the numbers are stated with their uncertainty and can be used as shown."
                                 if n_bad == 0 else f"{n_bad} caveat(s) apply to this run. Read them before using the affected numbers.")
        else:
            for i in self.session.issues():
                items.append(i)
            self.head.setText("Checks on your setup")
            self.ok_note.setText("Nothing to fix before you run." if not items else "These are updated as you change the setup.")
        self.list.set_issues(items)


# ------------------------------------------------------------------------------------------------ history
class HistoryPanel(QtWidgets.QWidget):
    compareChanged = QtCore.pyqtSignal(object)  # Snapshot or None

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._cmp: Optional[int] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(make_label("Runs this session", "h2"))
        lay.addWidget(make_label("Change a setting and run again: earlier runs stay here so you can see exactly what moved. Select one to overlay it on the timeline.",
                                 "caption", word_wrap=True))
        self.host = QtWidgets.QVBoxLayout()
        self.host.setSpacing(8)
        lay.addLayout(self.host)
        self.delta = QtWidgets.QFrame()
        self.delta.setObjectName("Plate")
        self.delta_lay = QtWidgets.QGridLayout(self.delta)
        self.delta_lay.setContentsMargins(12, 10, 12, 10)
        self.delta_lay.setHorizontalSpacing(10)
        self.delta.hide()
        lay.addWidget(self.delta)
        self.empty = make_label("No runs yet.", "muted")
        lay.addWidget(self.empty)
        lay.addStretch(1)
        session.snapshotsChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        while self.host.count():
            it = self.host.takeAt(0)
            if it.widget():
                dispose(it.widget())
        snaps = list(reversed(self.session.snapshots))
        self.empty.setVisible(not snaps)
        for sn in snaps:
            self.host.addWidget(self._card(sn, latest=sn is snaps[0]))
        self._fill_delta()

    def _card(self, sn: Snapshot, latest: bool) -> QtWidgets.QFrame:
        fr = QtWidgets.QFrame()
        fr.setObjectName("Plate")
        set_prop(fr, "selected", "true" if self._cmp == sn.id else "false")
        v = QtWidgets.QVBoxLayout(fr)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(make_label(f"{sn.label}  ·  {sn.created}", "body"))
        if latest:
            head.addWidget(Chip("latest", "ok"))
        if sn.pinned:
            head.addWidget(Chip("pinned", "accent"))
        head.addStretch(1)
        v.addLayout(head)
        st = sn.settings
        v.addWidget(make_label(f"{st['zone']} · stride {st['stride']} · {st['frames']} frames", "caption", word_wrap=True))
        k = sn.key
        v.addWidget(make_label(f"n̄ = {k['mean_n']:.3g} · τ = {k['tau']:.3g} ps · entry rate = {k['entry_rate']:.3g}/ps", "mono", word_wrap=True))
        row = QtWidgets.QHBoxLayout()
        if not latest:
            b = QtWidgets.QPushButton("Hide overlay" if self._cmp == sn.id else "Overlay on timeline")
            b.clicked.connect(lambda _=False, i=sn.id: self._toggle(i))
            row.addWidget(b)
        pin = QtWidgets.QToolButton()
        set_prop(pin, "variant", "ghost")
        pin.setText("Unpin" if sn.pinned else "Pin")
        pin.clicked.connect(lambda _=False, i=sn.id, p=sn.pinned: self.session.pin(i, not p))
        row.addWidget(pin)
        row.addStretch(1)
        v.addLayout(row)
        return fr

    def _toggle(self, snap_id: int) -> None:
        self._cmp = None if self._cmp == snap_id else snap_id
        snap = next((s for s in self.session.snapshots if s.id == self._cmp), None)
        self.compareChanged.emit(snap)
        self.refresh()

    def _fill_delta(self) -> None:
        while self.delta_lay.count():
            it = self.delta_lay.takeAt(0)
            if it.widget():
                dispose(it.widget())
        snaps = self.session.snapshots
        other = next((s for s in snaps if s.id == self._cmp), None)
        if other is None or len(snaps) < 2:
            self.delta.hide()
            return
        cur = snaps[-1]
        self.delta_lay.addWidget(make_label("", "caption"), 0, 0)
        self.delta_lay.addWidget(make_label(other.label, "caption"), 0, 1)
        self.delta_lay.addWidget(make_label(cur.label, "caption"), 0, 2)
        self.delta_lay.addWidget(make_label("change", "caption"), 0, 3)
        for i, (key, title, fmt) in enumerate((("mean_n", "n̄ (molecules)", "{:.3g}"), ("occupancy", "occupancy", "{:.3f}"),
                                               ("tau", "τ (ps)", "{:.3g}"), ("entry_rate", "entry rate (/ps)", "{:.3g}"),
                                               ("visits", "visits", "{:.0f}")), start=1):
            a, b = other.key.get(key, float("nan")), cur.key.get(key, float("nan"))
            self.delta_lay.addWidget(make_label(title, "body"), i, 0)
            self.delta_lay.addWidget(make_label(fmt.format(a) if _fin(a) else "n/a", "mono"), i, 1)
            self.delta_lay.addWidget(make_label(fmt.format(b) if _fin(b) else "n/a", "mono"), i, 2)
            ch = (b - a) / abs(a) if _fin(a) and _fin(b) and a != 0 else float("nan")
            lab = make_label(f"{ch:+.0%}" if _fin(ch) else "–", "mono")
            if _fin(ch) and abs(ch) > 0.1:
                lab.setStyleSheet(f"color: {_t().t['warn']}; font-weight: 700;")
            self.delta_lay.addWidget(lab, i, 3)
        self.delta.show()


# ------------------------------------------------------------------------------------------------ export
RULES = [("The zone holds at least N molecules", "n_solvent>={n}"), ("The zone holds exactly N molecules", "n_solvent=={n}"),
         ("The zone holds at most N molecules", "n_solvent<={n}")]


class ExportPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._busy = False
        self._last_dir: Optional[str] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        # data / report
        card = Card("Save results", "download", "Tables, settings, provenance and a SHA-256 manifest, so the folder can be checked later.")
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit()
        self.folder.setProperty("role", "mono")
        self.folder.setToolTip("Output folder")
        browse = QtWidgets.QToolButton()
        browse.setText("Browse…")
        set_prop(browse, "variant", "ghost")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        card.add(row)
        btns = QtWidgets.QHBoxLayout()
        self.save_data = QtWidgets.QPushButton("Save data")
        self.save_data.setToolTip("CSV tables, JSON summary, provenance, data dictionary")
        self.save_report = QtWidgets.QPushButton("Save data + report")
        set_prop(self.save_report, "variant", "primary")
        self.save_report.setToolTip("Everything above plus a readable report with definitions, settings and caveats")
        self.save_data.clicked.connect(lambda: self._export(False))
        self.save_report.clicked.connect(lambda: self._export(True))
        btns.addWidget(self.save_data)
        btns.addWidget(self.save_report, 1)
        card.add(btns)
        self.exp_status = make_label("", "caption", word_wrap=True, selectable=True)
        card.add(self.exp_status)
        links = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open folder")
        self.open_btn.clicked.connect(lambda: self._last_dir and _open_folder(self._last_dir))
        self.verify_btn = QtWidgets.QPushButton("Verify files")
        self.verify_btn.setToolTip("Re-hash every exported file against manifest.json")
        self.verify_btn.clicked.connect(self._verify)
        links.addWidget(self.open_btn)
        links.addWidget(self.verify_btn)
        links.addStretch(1)
        self.links_host = QtWidgets.QWidget()
        self.links_host.setLayout(links)
        self.links_host.hide()
        card.add(self.links_host)
        lay.addWidget(card)
        # frames
        fcard = Card("Extract frames", "scissors", "Write the frames where a rule holds to a new trajectory. The inputs are verified by content first.",
                     help_key=None)
        self.rule = QtWidgets.QComboBox()
        for title, tpl in RULES:
            self.rule.addItem(title, tpl)
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(0, 100000)
        self.n.setValue(1)
        self.n.setPrefix("N = ")
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(self.rule, 1)
        r1.addWidget(self.n)
        fcard.add(r1)
        self.min_run = QtWidgets.QSpinBox()
        self.min_run.setRange(1, 100000)
        self.min_run.setSuffix(" frames")
        self.min_run.setToolTip("Keep only runs of at least this many consecutive frames")
        self.gap = QtWidgets.QSpinBox()
        self.gap.setRange(0, 100000)
        self.gap.setSuffix(" frames")
        self.gap.setToolTip("Bridge interruptions of up to this many frames")
        self.fmt = QtWidgets.QComboBox()
        for f in ("xtc", "dcd", "trr", "pdb"):
            self.fmt.addItem(f.upper(), f)
        r2 = QtWidgets.QFormLayout()
        r2.setVerticalSpacing(6)
        r2.addRow(make_label("Shortest run", "caption"), self.min_run)
        r2.addRow(make_label("Bridge gaps up to", "caption"), self.gap)
        r2.addRow(make_label("Format", "caption"), self.fmt)
        fcard.add(r2)
        self.preview = make_label("", "body", word_wrap=True)
        fcard.add(self.preview)
        self.extract_btn = QtWidgets.QPushButton("Extract frames")
        self.extract_btn.clicked.connect(self._extract)
        fcard.add(self.extract_btn)
        self.ext_status = make_label("", "caption", word_wrap=True, selectable=True)
        fcard.add(self.ext_status)
        lay.addWidget(fcard)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        lay.addWidget(self.progress)
        lay.addStretch(1)
        for w in (self.rule, self.n, self.min_run, self.gap):
            sig = w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.valueChanged
            sig.connect(lambda *_: self._refresh_preview())
        session.resultReady.connect(self._on_result)
        session.taskProgress.connect(self._on_progress)
        session.taskDone.connect(self._on_done)
        session.taskFailed.connect(self._on_failed)
        session.settingsChanged.connect(self._sync_folder)
        self._on_result(None)

    # ------------------------------------------------------------------------------------ helpers
    def _sync_folder(self) -> None:
        if not self.folder.hasFocus() and self.session.run.output_dir:
            self.folder.setText(self.session.run.output_dir)

    def _on_result(self, result) -> None:
        has = self.session.result is not None
        for w in (self.save_data, self.save_report, self.extract_btn, self.rule, self.n, self.min_run, self.gap, self.fmt, self.folder):
            w.setEnabled(has and not self._busy)
        if not has:
            self.exp_status.setText("Run the analysis first; there is nothing to save yet.")
            self.preview.setText("")
            return
        self.exp_status.setText("")
        self._sync_folder()
        if self.session.result is not None:
            name = next(iter(self.session.result.soz_results))
            nmax = int(self.session.result.soz_results[name].summary.get("max_n_solvent", 1))
            self.n.setMaximum(max(nmax, 1))
            self.n.setValue(min(max(1, int(round(self.session.result.soz_results[name].summary["mean_n_solvent"]))), nmax))
        self._refresh_preview()

    def _rule(self) -> str:
        return self.rule.currentData().format(n=self.n.value())

    def _refresh_preview(self) -> None:
        if self.session.result is None:
            return
        try:
            sel = self.session.preview_extraction(self._rule(), self.min_run.value(), self.gap.value())
            n, tot = len(sel.frame_indices), sel.params["total_frames"]
            self.preview.setText(f"<b>{n}</b> of {tot} analysed frames match" + ("" if n else " — loosen the rule"))
            self.preview.setTextFormat(Qt.TextFormat.RichText)
            self.extract_btn.setEnabled(n > 0 and not self._busy)
        except Exception as exc:
            self.preview.setText(str(exc))
            self.extract_btn.setEnabled(False)

    def _browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the output folder", self.folder.text() or "")
        if d:
            self.folder.setText(d)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
        for w in (self.save_data, self.save_report, self.extract_btn):
            w.setEnabled(not busy and self.session.result is not None)

    # ------------------------------------------------------------------------------------ actions
    def _export(self, with_report: bool) -> None:
        directory = self.folder.text().strip()
        if not directory:
            self.exp_status.setText("Choose an output folder first.")
            return
        if os.path.isfile(os.path.join(directory, "manifest.json")):
            ans = QtWidgets.QMessageBox.question(self, "Replace earlier results?",
                                                 f"{directory}\ncontains results from an earlier run. They will be replaced by this run's results.")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.exp_status.setText("Writing…")
        self._set_busy(True)
        (self.session.export_report if with_report else self.session.export_data)(directory)

    def _extract(self) -> None:
        directory = os.path.join(self.folder.text().strip() or self.session.run.output_dir, "extracted")
        self.ext_status.setText("Verifying the inputs and writing frames…")
        self._set_busy(True)
        self.session.extract_frames(self._rule(), directory, self.fmt.currentData(), self.min_run.value(), self.gap.value())

    def _verify(self) -> None:
        if not self._last_dir:
            return
        from engine.export import verify_export

        try:
            problems = verify_export(self._last_dir)
        except Exception as exc:
            self.exp_status.setText(f"Could not verify: {exc}")
            return
        self.exp_status.setText("✓ Every file matches manifest.json." if not problems else "✗ " + "; ".join(problems[:5]))

    # ------------------------------------------------------------------------------------ task signals
    def _on_progress(self, name: str, cur: int, tot: int, msg: str) -> None:
        if name in ("export_data", "export_report", "extract") and tot > 0:
            self.progress.setRange(0, tot)
            self.progress.setValue(cur)
        (self.ext_status if name == "extract" else self.exp_status).setText(msg)

    def _on_done(self, name: str, result) -> None:
        if name not in ("export_data", "export_report", "extract"):
            return
        self._set_busy(False)
        self._refresh_preview()
        if name == "extract":
            paths = [v for v in (result or {}).values() if isinstance(v, str)]
            where = os.path.dirname(paths[0]) if paths else ""
            self.ext_status.setText(f"✓ Written to {where}" if where else "✓ Done")
            self._last_dir = where or self._last_dir
        else:
            out = result["output_dir"] if isinstance(result, dict) else (os.path.dirname(os.path.dirname(result)) if isinstance(result, str) else "")
            self._last_dir = out if isinstance(result, dict) else (self.folder.text().strip() or out)
            self.exp_status.setText(f"✓ Saved to {self._last_dir}")
            self.session.exported = True
            self.session.stepsChanged.emit()
        self.links_host.setVisible(bool(self._last_dir))

    def _on_failed(self, name: str, msg: str) -> None:
        if name not in ("export_data", "export_report", "extract"):
            return
        self._set_busy(False)
        (self.ext_status if name == "extract" else self.exp_status).setText(f"✗ {msg}")
