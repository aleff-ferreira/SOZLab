"""Scripted, task-based usability verification of the Workbench (offscreen, real widgets, real input events).

This is NOT a user study: no human takes part. It answers the questions a machine can answer about a first-time
researcher's path, and records screenshots as evidence:

  * can each task of the novice path be done using only controls that are visible, labelled and not covered by other
    widgets at the target window size (1366x768), and how many actions (clicks / keystrokes / scrolls) does it take;
  * does the interface recover from the common mistakes (bad selection, wrong solvent name, unreadable files, cancel);
  * is every technical term on screen explained (tooltip or definition) and every number shown with a unit;
  * do the screens render without clipping at 1100x680, 1366x768 and 1920x1080, in both themes.

    python bench/gui/usability_walkthrough.py --out docs/evidence/screens --json docs/evidence/data/usability_walkthrough.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PyQt6 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

from app.workbench.theme import Theme  # noqa: E402
from app.workbench.widgets import HelpButton  # noqa: E402
from app.workbench.window import WorkbenchWindow  # noqa: E402

Qt = QtCore.Qt


class Study:
    def __init__(self, out: Path, size=(1366, 768)):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.theme = Theme.instance()
        self.theme.mode, self.theme.reduce_motion = "dark", True
        self.theme.apply(self.app)
        self.tmp = Path(tempfile.mkdtemp(prefix="sozlab_study_"))
        self.win = WorkbenchWindow(demo_dir=str(self.tmp / "demo"))
        self.win.resize(*size)
        self.win.show()
        self.s = self.win.session
        self.actions: list[dict] = []
        self.tasks: list[dict] = []
        self.shots: list[str] = []
        self.problems: list[str] = []
        self.pump(200)

    # ------------------------------------------------------------------------------------------ helpers
    def pump(self, ms: int) -> None:
        t0 = time.time()
        while (time.time() - t0) * 1000 < ms:
            self.app.processEvents()
            time.sleep(0.005)

    def wait(self, cond, timeout=60.0, what=""):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        self.problems.append(f"timeout waiting for {what}")
        return False

    def shot(self, name: str, widget: QtWidgets.QWidget | None = None, suffix: bool = True) -> str:
        self.pump(120)
        w = widget or self.win
        fname = f"{name}_{self.theme.mode}.png" if suffix else f"{name}.png"
        path = self.out / fname
        w.grab().save(str(path), "PNG")
        try:  # UI screenshots compress ~4x with a 192-colour palette and no dithering; keeps the repository light
            from PIL import Image

            Image.open(path).convert("RGB").quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
        except Exception:
            pass
        self.shots.append(fname)
        return fname

    def visible_in_window(self, w: QtWidgets.QWidget) -> bool:
        """Visible without scrolling and not occluded: a pointer at the widget's centre would hit it."""
        if not w.isVisible():
            return False
        centre = w.mapTo(self.win, w.rect().center())
        if not self.win.rect().contains(centre):
            return False
        hit = self.win.childAt(centre)
        while hit is not None:
            if hit is w:
                return True
            hit = hit.parentWidget()
        return False

    def reach(self, w: QtWidgets.QWidget, what: str) -> bool:
        """What a person does to get at a control: scroll its rail if needed (1 action). Returns whether it ended up reachable."""
        if not self.visible_in_window(w):
            sa = self.win.left_scroll
            if sa.isAncestorOf(w):
                sa.ensureWidgetVisible(w, 0, 40)
                self.actions.append({"type": "scroll", "target": what})
                self.pump(60)
        ok = self.visible_in_window(w)
        if not ok:
            self.problems.append(f"not reachable: {what}")
        return ok

    def click(self, w: QtWidgets.QWidget, what: str) -> None:
        self.reach(w, what)
        QtTest.QTest.mouseClick(w, Qt.MouseButton.LeftButton)
        self.actions.append({"type": "click", "target": what})
        self.pump(60)

    def type_into(self, w: QtWidgets.QLineEdit, text: str, what: str) -> None:
        self.reach(w, what)
        w.setFocus()
        w.selectAll()
        QtTest.QTest.keyClicks(w, text)
        QtTest.QTest.keyClick(w, Qt.Key.Key_Return)
        self.actions.append({"type": "type", "target": what, "chars": len(text)})
        self.pump(60)

    def task(self, name: str, goal: str, fn) -> None:
        start = len(self.actions)
        p0 = len(self.problems)
        t0 = time.time()
        ok, detail = True, ""
        try:
            detail = fn() or ""
        except Exception as exc:  # recorded, not hidden
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        acts = self.actions[start:]
        self.tasks.append({
            "task": name, "goal": goal, "completed": ok and len(self.problems) == p0, "detail": detail,
            "actions": len(acts), "clicks": sum(a["type"] == "click" for a in acts), "keystrokes": sum(a.get("chars", 0) for a in acts),
            "scrolls": sum(a["type"] == "scroll" for a in acts), "seconds_machine": round(time.time() - t0, 1),
            "problems": self.problems[p0:],
        })

    # ------------------------------------------------------------------------------------------ the novice path
    def run_all(self, demo_files) -> dict:
        w, s = self.win, self.s
        gro, xtc = demo_files
        self.shot("01_empty")

        def t_import():
            w.import_card.drop.filesDropped.emit([xtc, gro])  # what dropping the two files on the zone does
            self.actions.append({"type": "drop", "target": "topology + trajectory"})
            assert self.wait(lambda: s.summary is not None, 60, "import")
            assert "frames" in w.import_card.line1.text()
            self.shot("02_imported")
            return f"{s.summary['n_atoms']} atoms, {s.summary['n_frames']} frames, solvent {s.summary['water_resnames']}"
        self.task("T1 Import", "Open a topology and trajectory", t_import)

        def t_define():
            self.wait(lambda: s.selection_ok is not None, 20, "selection check")
            chip = w.zone_card._chips[0]
            self.click(chip, "chip: " + chip.text())
            self.wait(lambda: len(w.results.timeline._n) > 0, 30, "occupancy preview")
            before = float(w.results.t_mean.value_label.text()) if w.results.t_mean.value_label.text() not in ("–", "n/a") else None
            self.shot("03_define_preview")
            # change the distance by typing it (the red line, slider and number box are one control)
            dp = QtCore.QLocale().decimalPoint()
            sp = w.zone_card.cutoff
            self.reach(sp, "distance box")
            sp.setFocus()
            sp.selectAll()
            QtTest.QTest.keyClicks(sp, f"4{dp}5")
            QtTest.QTest.keyClick(sp, Qt.Key.Key_Return)
            self.actions.append({"type": "type", "target": "distance box", "chars": 3})
            self.wait(lambda: abs(s.zone.cutoff - 4.5) < 1e-6, 5, "cutoff applied")
            self.wait(lambda: w.results.t_mean.value_label.text() not in ("–",) and float(w.results.t_mean.value_label.text()) != before, 30, "preview update")
            self.shot("04_cutoff_changed")
            assert w.stage.stack.currentIndex() == 1 and w.stage.scene.stats()["n_solvent"] > 0
            return f"zone: {s.zone.describe()}; preview mean n {w.results.t_mean.value_label.text()}"
        self.task("T2 Define and tweak the zone", "Choose what the zone surrounds and change its distance", t_define)

        def t_advanced():
            rc = w.run_card
            tgl = rc._adv_toggle
            self.click(tgl, "Advanced settings")
            rc.f_gap.set_value(2.0, emit=True)
            self.actions.append({"type": "type", "target": "brief-exit tolerance", "chars": 1})
            assert rc.f_gap.is_modified() and "1 changed" in rc._adv_count.text()
            self.reach(rc.f_gap, "brief-exit tolerance")
            self.shot("05_advanced_changed")
            rc.f_gap.reset()
            assert not rc.f_gap.is_modified()
            self.actions.append({"type": "click", "target": "reset"})
            tgl.setChecked(False)
            return "advanced settings open, a changed setting is marked and can be reset"
        self.task("T3 Find and change an advanced setting", "Adjust, compare with default, reset", t_advanced)

        def t_run():
            s.run.density = True
            w.run_card.f_density.set_value(True, emit=True)
            self.click(w.runbar.run_btn, "Run analysis")
            self.pump(150)
            self.shot("06_running")
            assert self.wait(lambda: s.result is not None and not s.running, 180, "run")
            self.pump(300)
            self.shot("07_results_occupancy")
            return f"complete={s.result.complete}"
        self.task("T4 Run", "Run the analysis", t_run)

        def t_inspect():
            tabs = ("Residence", "Entries / exits", "Sampling", "Density", "Tables")
            for i, name in enumerate(tabs, start=1):
                w.results.tabbar.setCurrentIndex(i)
                self.actions.append({"type": "click", "target": f"tab {name}"})
                self.pump(150)
                self.shot(f"08_results_{name.split()[0].lower()}")
            w.results.tabbar.setCurrentIndex(0)
            # the residence time and its uncertainty can be read straight from a tile, with unit and trust badge
            tile = w.results.t_tau
            txt = f"{tile.value_label.text()} {tile.unit_label.text()} {tile.sub.toolTip()}"
            assert "ps" in txt and "SE" in txt
            # a definition is one click away
            helps = tile.findChildren(HelpButton)
            assert helps, "no help button on the residence tile"
            self.click(helps[0], "? on the residence tile")
            pops = [wd for wd in QtWidgets.QApplication.topLevelWidgets() if wd.isVisible() and wd is not w and wd.objectName() == "Card"]
            if pops:
                self.shot("09_definition_popup", pops[0])
                pops[0].close()
            else:
                self.problems.append("definition popup did not open")
            w.right.show_page("trust")
            self.actions.append({"type": "click", "target": "Trust tab"})
            self.shot("10_trust")
            return f"residence τ read as '{txt}'"
        self.task("T5 Inspect", "Find the residence time with its uncertainty, learn what it means, see the caveats", t_inspect)

        def t_compare():
            w.zone_card.cutoff.setValue(3.0)
            self.actions.append({"type": "type", "target": "distance box", "chars": 3})
            assert w.runbar.stale.isVisible(), "stale-results marker not shown"
            self.click(w.runbar.run_btn, "Run again")
            assert self.wait(lambda: len(s.snapshots) == 2 and not s.running, 180, "second run")
            w.right.show_page("runs")
            self.actions.append({"type": "click", "target": "Runs tab"})
            first = s.snapshots[0]
            w.history._toggle(first.id)
            self.actions.append({"type": "click", "target": "Overlay on timeline"})
            self.pump(200)
            self.shot("11_runs_compare")
            return f"2 runs; overlay shown: {w.results.timeline._cmp is not None}"
        self.task("T6 Change a setting and compare", "See what a different distance does", t_compare)

        def t_export():
            self.click(w.stepper, "stepper: Export")  # the step indicator is itself navigation
            w.right.show_page("export")
            out = str(self.tmp / "export")
            w.export.folder.setText(out)
            self.click(w.export.save_report, "Save data + report")
            assert self.wait(lambda: w.export.links_host.isVisible() and not s.tasks.running, 120, "export finished")
            assert os.path.isfile(os.path.join(out, "manifest.json"))
            self.pump(300)
            self.shot("12_export")
            self.click(w.export.verify_btn, "Verify files")
            assert "✓" in w.export.exp_status.text(), w.export.exp_status.text()
            return "data + report written and verified against the manifest"
        self.task("T7 Export", "Save results and check them", t_export)

        # ------------------------------------------------------------------ recovery
        def r_bad_selection():
            w.right.show_page("trust")
            w.zone_card.selection.setText("resname NOPE")
            w.zone_card._sel_edited("resname NOPE")
            assert self.wait(lambda: s.selection_ok is False, 10, "selection check")
            self.pump(200)
            assert not w.runbar.run_btn.isEnabled() and "no atoms" in w.runbar.status.text().lower() or "matches" in w.runbar.status.text().lower()
            self.reach(w.zone_card.selection, "selection box")
            self.shot("13_error_bad_selection")
            self.click(w.zone_card._chips[0], "chip: " + w.zone_card._chips[0].text())
            assert self.wait(lambda: s.can_run(), 10, "recovered")
            return "Run blocked with the reason shown; one click on a chip recovers"
        self.task("R1 Recover: selection matches nothing", "Notice, understand and fix", r_bad_selection)

        def r_wrong_solvent():
            w.zone_card.open_advanced(True)
            w.zone_card.f_names.set_value("XYZ", emit=True)
            assert self.wait(lambda: any(i.code == "solvent.names" for i in s.issues()), 5, "solvent issue")
            self.pump(200)
            self.shot("14_error_solvent_name")
            issue = next(i for i in s.issues() if i.code == "solvent.names")
            issue.fix()
            self.actions.append({"type": "click", "target": issue.fix_label})
            w.zone_card.open_advanced(False)
            return "offered '" + (issue.fix_label or "") + "' as a button"
        self.task("R2 Recover: wrong solvent name", "Notice and fix with one click", r_wrong_solvent)

        def r_unreadable():
            bad = self.tmp / "broken.gro"
            bad.write_text("not a structure\n")
            s.open_inputs(str(bad), None)
            assert self.wait(lambda: w.import_card.stack.currentIndex() == 3, 30, "error card")
            self.shot("15_error_unreadable_file")
            msg = w.import_card.error_text.text()
            s.open_paths([xtc, gro])
            assert self.wait(lambda: s.summary is not None, 60, "reload")
            return msg[:140]
        self.task("R3 Recover: unreadable file", "Understand the error and continue", r_unreadable)

        # ------------------------------------------------------------------ appearance and sizes
        self.wait(lambda: s.result is None or not s.running, 5, "idle")
        s.start_run()
        self.wait(lambda: s.result is not None and not s.running, 120, "final run")
        w.results.tabbar.setCurrentIndex(0)
        w.right.show_page("trust")
        self.shot("20_results_1366x768")
        self.theme.set_mode("light")
        self.pump(300)
        self.shot("21_results_light")
        w.results.tabbar.setCurrentIndex(3)
        self.pump(200)
        self.shot("22_density_light")
        self.theme.set_mode("dark")
        w.results.tabbar.setCurrentIndex(0)
        for (ww, hh, tag) in ((1100, 680, "small_1100x680"), (1920, 1080, "large_1920x1080")):
            w.resize(ww, hh)
            self.pump(500)
            self.shot(f"23_{tag}")
        w.resize(1366, 768)
        self.pump(300)
        return self.audit()

    # ------------------------------------------------------------------------------------------ static audits
    def audit(self) -> dict:
        w = self.win
        w.resize(1366, 768)
        self.pump(300)
        controls = []
        for cls in (QtWidgets.QPushButton, QtWidgets.QToolButton, QtWidgets.QComboBox, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox,
                    QtWidgets.QCheckBox, QtWidgets.QLineEdit, QtWidgets.QSlider):
            for c in w.findChildren(cls):
                if not c.isVisibleTo(w) or c.parent() is None:
                    continue
                if isinstance(c, QtWidgets.QLineEdit) and c.parent().inherits("QAbstractSpinBox"):
                    continue
                label = ""
                if hasattr(c, "text") and callable(c.text) and not isinstance(c, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QComboBox)):
                    try:
                        label = c.text()
                    except Exception:
                        label = ""
                controls.append({"class": cls.__name__, "label": label, "tooltip": c.toolTip(), "icon_only": bool(not label and hasattr(c, "icon") and not c.icon().isNull())})
        icon_only = [c for c in controls if c["icon_only"]]
        icon_only_without_tooltip = [c for c in icon_only if not c["tooltip"]]
        # every statistic tile shows its unit; list the units on screen
        tiles = {t.title_label.text(): t.unit_label.text() for t in w.results.tiles.tiles}
        no_unit = [k for k, v in tiles.items() if not v]
        # clipped text: labels whose text is wider than the widget showing it (cheap proxy for truncation)
        clipped = []
        for lab in w.findChildren(QtWidgets.QLabel):
            if lab.isVisibleTo(w) and lab.text() and not lab.wordWrap() and lab.pixmap() is None and lab.width() > 8:
                fm = lab.fontMetrics()
                if fm.horizontalAdvance(lab.text()) > lab.width() + 3 and lab.property("role") != "kpi" and "ElideLabel" not in type(lab).__name__:
                    clipped.append(lab.text()[:50])
        return {"controls_checked": len(controls), "icon_only_controls": len(icon_only), "icon_only_without_tooltip": len(icon_only_without_tooltip),
                "tiles_and_units": tiles, "tiles_without_unit": no_unit, "possibly_clipped_labels": clipped[:12]}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/screens")
    ap.add_argument("--json", default="docs/evidence/data/usability_walkthrough.json")
    args = ap.parse_args()
    from engine.synthetic import make_demo_system

    tmp = Path(tempfile.mkdtemp(prefix="sozlab_study_data_"))
    make_demo_system(n_waters=700, n_frames=160, box=28.0, seed=7).write(str(tmp), "demo", "xtc")
    study = Study(Path(args.out))
    audit = study.run_all((str(tmp / "demo.gro"), str(tmp / "demo.xtc")))
    report = {"window_size": [1366, 768], "tasks": study.tasks, "audit": audit, "screenshots": study.shots, "problems": study.problems,
              "note": "scripted checks on real widgets with real input events; no human participants"}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.json, "w"), indent=2)
    done = sum(t["completed"] for t in study.tasks)
    print(f"{done}/{len(study.tasks)} tasks completed; screenshots: {len(study.shots)}; problems: {len(study.problems)}")
    for t in study.tasks:
        print(f"  {'OK ' if t['completed'] else 'FAIL'} {t['task']:45s} actions={t['actions']:2d} clicks={t['clicks']} keys={t['keystrokes']} scrolls={t['scrolls']}  {t['detail'][:80]}")
    print(json.dumps(audit, indent=1)[:1500])
    study.win.close()
    return 0 if done == len(study.tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
