"""Workbench GUI tests: the first-use path, validation/recovery, results, export, theme, and layout at 1366x768."""
import json
import os

import numpy as np
import pytest

from tests.gui.conftest import load, run  # noqa: F401  (also imported by sibling test modules)

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------------------------------- import
def test_dropped_files_are_classified_and_summarised(qtbot, window, demo_files):
    gro, xtc = demo_files
    with qtbot.waitSignal(window.session.summaryReady, timeout=30000):
        assert window.session.open_paths([xtc, gro])  # order does not matter
    s = window.session.summary
    assert s["n_frames"] == 70 and s["n_atoms"] > 1000 and s["water_resnames"] == ["SOL"]
    assert window.import_card.stack.currentIndex() == 1
    assert "70 frames" in window.import_card.line1.text()
    # a sensible zone exists before the user touches anything: the ligand
    assert window.session.zone.anchor == "resname LIG"
    assert window.session.step_state()["import"] == "done"


def test_unreadable_files_give_a_recoverable_error(qtbot, window, tmp_path, demo_files):
    bad = tmp_path / "broken.gro"
    bad.write_text("not a structure file\n")
    with qtbot.waitSignal(window.session.summaryFailed, timeout=30000):
        window.session.open_inputs(str(bad), None)
    assert window.import_card.stack.currentIndex() == 3
    assert window.import_card.error_text.text()
    # recovery: choosing good files afterwards works in the same window
    load(qtbot, window, demo_files)
    assert window.import_card.stack.currentIndex() == 1


def test_topology_trajectory_mismatch_is_explained(qtbot, window, demo_files, tmp_path):
    from engine.synthetic import make_demo_system

    other = tmp_path / "other"
    make_demo_system(n_waters=60, n_frames=5, box=20.0, seed=3).write(str(other), "o", "xtc")
    with qtbot.waitSignal(window.session.summaryFailed, timeout=30000):
        window.session.open_inputs(demo_files[0], str(other / "o.xtc"))
    assert window.import_card.stack.currentIndex() == 3


# ---------------------------------------------------------------------------------------------------- define
def test_live_preview_follows_the_cutoff(qtbot, loaded):
    s = loaded.session
    with qtbot.waitSignal(s.seriesReady, timeout=20000, raising=True):
        loaded.zone_card.cutoff.setValue(2.5)
    n_small = float(np.mean(loaded.results.timeline._n))
    with qtbot.waitSignal(s.seriesReady, timeout=20000, raising=True):
        loaded.zone_card.cutoff.setValue(5.0)
    n_big = float(np.mean(loaded.results.timeline._n))
    assert n_big > n_small > 0
    assert s.zone.cutoff == pytest.approx(5.0)
    assert "PREVIEW" in loaded.results.timeline.caption.text()


def test_stage_draws_the_system_and_the_zone(qtbot, loaded):
    qtbot.waitUntil(lambda: getattr(loaded.stage.scene, "stats", lambda: {})().get("n_solvent", 0) > 0, timeout=15000)
    st = loaded.stage.scene.stats()
    assert st["n_solute"] > 0 and st["n_solvent"] > 0
    assert "in zone" in loaded.stage.readout.text()


def test_invalid_selection_blocks_run_and_says_why(qtbot, loaded):
    s = loaded.session
    with qtbot.waitSignal(s.selectionInfo, timeout=10000):
        loaded.zone_card.selection.setText("resname DOES_NOT_EXIST")
        loaded.zone_card._sel_edited("resname DOES_NOT_EXIST")
    assert not s.can_run()
    assert not loaded.runbar.run_btn.isEnabled()
    assert "no atoms" in loaded.runbar.status.text().lower() or "matches" in loaded.runbar.status.text().lower()
    # recovery: pick a detected component with one click
    with qtbot.waitSignal(s.selectionInfo, timeout=10000):
        loaded.zone_card._chips[0].click()
    assert s.can_run() and loaded.runbar.run_btn.isEnabled()


def test_wrong_solvent_name_offers_a_one_click_fix(qtbot, loaded):
    s = loaded.session
    loaded.zone_card.f_names.set_value("XYZ", emit=True)
    qtbot.waitUntil(lambda: any(i.code == "solvent.names" for i in s.issues()), timeout=5000)
    issue = next(i for i in s.issues() if i.code == "solvent.names")
    assert issue.fix is not None and "SOL" in issue.fix_label
    issue.fix()
    assert not any(i.code == "solvent.names" for i in s.issues())
    assert s.can_run()


def test_advanced_settings_show_changes_and_reset(qtbot, loaded):
    rc = loaded.run_card
    assert not rc.f_gap.is_modified()
    rc.open_advanced(True)
    rc.f_gap.set_value(2.0, emit=True)
    assert rc.f_gap.is_modified() and rc._adv_count.isVisible() or rc._adv_count.text().startswith("1")
    assert loaded.session.run.gap_time == pytest.approx(2.0)
    rc.f_gap.reset()
    assert not rc.f_gap.is_modified()
    assert loaded.session.run.gap_time == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------- run / inspect
def test_run_fills_results_with_uncertainty_and_trust(qtbot, loaded):
    res = run(qtbot, loaded)
    assert res is not None and res.complete
    z = next(iter(res.soz_results.values()))
    summ = z.summary
    tiles = loaded.results
    assert tiles.t_mean.value_label.text() != "–"
    assert float(tiles.t_mean.value_label.text()) == pytest.approx(summ["mean_n_solvent"], rel=2e-3)
    assert "SE" in tiles.t_mean.sub.toolTip()
    assert tiles.t_tau.unit_label.text() == "ps"
    assert loaded.results.stack.count() == 6
    assert loaded.session.step_state()["inspect"] == "done"
    # settings that produced the numbers stay in view
    assert loaded.results.strip_host.isVisible()
    chips = [loaded.results.strip.itemAt(i).widget().text() for i in range(loaded.results.strip.count())]
    assert any("solvent within" in c for c in chips) and any("frames" in c and "Δt" in c for c in chips)
    assert "config" in loaded.prov_chip.text()  # the configuration hash stays visible in the status bar
    # run history
    assert len(loaded.session.snapshots) == 1


def test_changed_settings_mark_results_stale_and_history_compares(qtbot, loaded):
    run(qtbot, loaded)
    assert not loaded.runbar.stale.isVisible()
    loaded.zone_card.cutoff.setValue(4.5)
    assert loaded.session.is_stale() and loaded.runbar.stale.isVisible()
    run(qtbot, loaded)
    assert len(loaded.session.snapshots) == 2 and not loaded.runbar.stale.isVisible()
    a, b = loaded.session.snapshots
    assert b.key["mean_n"] > a.key["mean_n"]
    loaded.history._toggle(a.id)
    assert loaded.results.timeline._cmp is not None  # earlier run overlaid on the timeline


def test_cancelling_never_fails_silently_or_as_an_error(qtbot, loaded):
    s = loaded.session
    s.run.density = False
    s.start_run()
    s.cancel_run()
    qtbot.waitUntil(lambda: not s.running, timeout=60000)
    res = s.result
    status = loaded.runbar.status.text().lower()
    if res is None:  # cancelled before the first frame: a clean cancellation, not "failed"
        assert "cancelled" in status and "failed" not in status
    elif not res.complete:  # cancelled mid-run: the partial result is labelled and never offered for comparison
        assert "partial" in loaded.results.banner.text().lower() and loaded.results.banner.isVisible()
        assert not s.snapshots
    else:  # finished before the cancel arrived (fast machine)
        assert len(s.snapshots) == 1


def test_partial_results_are_labelled_on_screen(qtbot, loaded):
    s = loaded.session
    s.run.density = False
    state = {"n": 0}
    orig = s.runner.start

    # drive the engine with a cancel flag that trips after 20 frames, through the same code path the Cancel button uses
    def start(fn):
        def wrapped(progress, cancelled):
            import engine.analysis as an

            real = an.SOZAnalysisEngine.run

            def patched(self, *a, **k):
                def flag():
                    state["n"] += 1
                    return state["n"] > 20
                k["cancel_flag"] = flag
                return real(self, *a, **k)

            an.SOZAnalysisEngine.run = patched
            try:
                return fn(progress, cancelled)
            finally:
                an.SOZAnalysisEngine.run = real
        return orig(wrapped)

    s.runner.start = start
    with qtbot.waitSignal(s.resultReady, timeout=60000):
        s.start_run()
    assert not s.result.complete
    assert "PARTIAL" in loaded.results.banner.text() and loaded.results.banner.isVisible()
    assert not s.snapshots


def test_trust_panel_lists_caveats_after_a_run(qtbot, loaded):
    res = run(qtbot, loaded)
    items = res.qc_summary["trust"]
    shown = loaded.trust.list._lay.count()
    assert shown >= len(items)  # every caveat the engine raised is visible, none summarised away


# ---------------------------------------------------------------------------------------------------- export
def test_export_report_and_verify(qtbot, loaded, tmp_path):
    run(qtbot, loaded)
    out = str(tmp_path / "export")
    with qtbot.waitSignal(loaded.session.taskDone, timeout=60000):
        loaded.session.export_report(out)
    assert os.path.isfile(os.path.join(out, "manifest.json"))
    from engine.export import verify_export

    assert verify_export(out) == []
    assert os.path.isdir(os.path.join(out, "report"))
    prov = json.load(open(os.path.join(out, "provenance.json"), encoding="utf-8"))
    assert prov["config_sha256"] and prov["inputs"]["topology"]["sha256"]


def test_extract_frames_from_the_gui_path(qtbot, loaded, tmp_path):
    res = run(qtbot, loaded)
    z = next(iter(res.soz_results.values()))
    n_expected = int((z.per_frame["n_solvent"] >= 1).sum())
    sel = loaded.session.preview_extraction("n_solvent>=1")
    assert len(sel.frame_indices) == n_expected
    out = str(tmp_path / "frames")
    with qtbot.waitSignal(loaded.session.taskDone, timeout=60000) as blocker:
        loaded.session.extract_frames("n_solvent>=1", out, "xtc")
    assert blocker.args[0] == "extract"
    import MDAnalysis as mda

    xtcs = [f for f in os.listdir(out) if f.endswith(".xtc")]
    assert xtcs
    u = mda.Universe(loaded.session.topology, os.path.join(out, xtcs[0]))
    assert len(u.trajectory) == n_expected


def test_extraction_refuses_inputs_changed_after_analysis(qtbot, loaded, tmp_path, demo_files):
    run(qtbot, loaded)
    import shutil

    copy = str(tmp_path / "copy.xtc")
    shutil.copy(demo_files[1], copy)
    # mutate the trajectory file the run was made from (in a copy that the project now points to)
    loaded.session.run_project.inputs.trajectory = copy
    with open(copy, "r+b") as fh:
        fh.seek(200)
        fh.write(b"\x00\x01\x02\x03")
    with qtbot.waitSignal(loaded.session.taskFailed, timeout=60000):
        loaded.session.extract_frames("n_solvent>=1", str(tmp_path / "x"), "xtc")


# ---------------------------------------------------------------------------------------------------- chrome
def test_theme_and_motion_toggles_do_not_break_rendering(qtbot, loaded):
    run(qtbot, loaded)
    from app.workbench.theme import Theme

    th = Theme.instance()
    before = th.mode
    loaded.toggle_theme()
    assert th.mode != before
    loaded.grab()  # repaints every themed widget without raising
    loaded.toggle_theme()
    assert th.mode == before
    loaded.btn_motion.setChecked(True)
    assert th.reduce_motion


def test_everything_primary_fits_at_1366x768(qtbot, loaded):
    run(qtbot, loaded)
    win = loaded
    win.resize(1366, 768)
    qtbot.wait(150)
    # no horizontal scrolling anywhere in the rails
    assert win.left_scroll.horizontalScrollBar().maximum() == 0
    # the Run button, stepper, 3D stage, tiles and chart tabs are all inside the window
    frame = win.rect()
    for w in (win.runbar.run_btn, win.stepper, win.stage, win.results.tiles, win.results.tabbar, win.right.seg):
        top_left = w.mapTo(win, w.rect().topLeft())
        bottom_right = w.mapTo(win, w.rect().bottomRight())
        assert frame.contains(top_left) and frame.contains(bottom_right), w
    # tiles stay on a single row (readable) at this size
    ys = {t.mapTo(win, t.rect().topLeft()).y() for t in win.results.tiles.tiles if t.isVisible()}
    assert len(ys) == 1


def test_project_round_trip_restores_the_setup(qtbot, loaded, tmp_path):
    loaded.zone_card.cutoff.setValue(4.2)
    loaded.session.set_run(stride=2, gap_time=1.5, blocks=7)
    loaded.session.set_zone(shells=[4.5, 6.0])
    path = str(tmp_path / "p.json")
    loaded.session.save_project(path)
    loaded.session.zone.cutoff = 3.0
    loaded.session.set_run(stride=1, gap_time=0.0, blocks=5)
    with qtbot.waitSignal(loaded.session.summaryReady, timeout=30000):
        loaded.session.load_project(path)
    s = loaded.session
    assert s.zone.cutoff == pytest.approx(4.2)
    # run settings are the user's: opening the files again must not replace them with defaults
    assert (s.run.stride, s.run.gap_time, s.run.blocks) == (2, 1.5, 7)
    assert s.zone.shells == [4.5, 6.0] and loaded.zone_card.f_shells.value() == "4.5, 6"  # advanced zone settings are shown again


def test_a_plain_cli_project_opens_in_the_workbench(qtbot, loaded, tmp_path):
    """The other direction: a project written for the CLI (no Workbench block) opens with its zone, solvent and stride."""
    from engine.analysis import write_project_json

    s = loaded.session
    s.set_run(stride=3)
    s.zone.cutoff = 4.7
    proj = s.build_project()
    path = str(tmp_path / "cli_project.json")
    write_project_json(proj, path)
    s.zone.cutoff, s.run.stride = 3.0, 1
    with qtbot.waitSignal(s.summaryReady, timeout=30000):
        s.load_project(path)
    assert s.zone.cutoff == pytest.approx(4.7) and s.run.stride == 3
    assert s.zone.anchor == "resname LIG" and s.solvent.resnames == ["SOL"]


def test_demo_loads_without_any_files(qtbot, window):
    with qtbot.waitSignal(window.session.summaryReady, timeout=120000):
        window.load_demo()
    assert window.session.summary["n_frames"] == 200
    assert window.session.zone.anchor == "resname LIG"


def test_left_rail_never_exceeds_its_width_in_any_state(qtbot, loaded):
    """Regression: a card wider than the rail is clipped at the right edge, hiding labels and buttons."""
    win = loaded
    rail = win.left_scroll

    def fits(state: str):
        qtbot.wait(60)
        need = rail.widget().minimumSizeHint().width()
        assert need <= rail.viewport().width(), f"{state}: content needs {need}px, the rail has {rail.viewport().width()}px"

    fits("after import")
    win.zone_card.open_advanced(True)
    win.run_card.open_advanced(True)
    fits("advanced regions open")
    win.run_card.f_gap.set_value(2.0, emit=True)  # shows the '1 changed' chip and the Reset button
    win.zone_card.f_shells.set_value("4.5, 6.0", emit=True)
    fits("settings changed")
    win.zone_card.selection.setText("resname NOPE")
    win.zone_card._sel_edited("resname NOPE")
    qtbot.waitUntil(lambda: win.session.selection_ok is False, timeout=10000)
    fits("with an error message")
    win.session.set_run(stride=1, frame_start=0)
    run_card_fields = [win.run_card.f_start, win.run_card.f_stop, win.run_card.f_stride, win.run_card.f_gap, win.run_card.f_outdir]
    for f in run_card_fields:
        assert f.width() > 0
    for size in ((1100, 680), (1366, 768), (1920, 1080)):
        win.resize(*size)
        fits(f"window {size}")


def test_tiles_and_tabs_never_overlap_at_any_window_size(qtbot, loaded):
    """Regression: at narrow widths the tile grid wrapped into two rows that overlapped each other and the tab bar."""
    from PyQt6 import QtCore

    run(qtbot, loaded)
    win = loaded
    for size in ((1080, 660), (1100, 680), (1366, 768), (1920, 1080)):
        win.resize(*size)
        qtbot.wait(250)
        rects = []
        for t in win.results.tiles.tiles:
            if t.isVisible():
                rects.append((t, QtCore.QRect(t.mapTo(win, QtCore.QPoint(0, 0)), t.size())))
        for i, (a, ra) in enumerate(rects):
            for b, rb in rects[i + 1:]:
                assert not ra.intersects(rb), f"{size}: tiles overlap ({a.title_label.text()} / {b.title_label.text()})"
        tabs = QtCore.QRect(win.results.tabbar.mapTo(win, QtCore.QPoint(0, 0)), win.results.tabbar.size())
        for t, r in rects:
            assert not r.intersects(tabs), f"{size}: tile {t.title_label.text()} overlaps the chart tabs"


def test_scene_carries_water_hydrogens_attached_to_their_oxygen(qtbot, loaded):
    s = loaded.session
    with qtbot.waitSignal(s.sceneReady, timeout=20000) as blocker:
        s.ask_scene(5, 4)
    out = blocker.args[0]
    assert out["ok"] and out["solvent_h"] is not None
    h, o = out["solvent_h"], out["solvent_xyz"]
    assert h.shape == (len(o), 2, 3)
    finite = np.isfinite(h).all(axis=(1, 2))
    assert finite.mean() > 0.95
    # O-H bonds are ~1 A in TIP3P-like water: no hydrogen is left behind on the other side of the box
    d = np.linalg.norm(h[finite] - o[finite][:, None, :], axis=2)
    assert d.max() < 1.3 and d.min() > 0.5


def test_folding_the_3d_view_gives_the_charts_the_height(qtbot, loaded):
    run(qtbot, loaded)
    win = loaded
    qtbot.wait(150)
    before = win.results_scroll.height()
    win.stage.fold.setChecked(True)
    qtbot.wait(150)
    assert not win.stage.stack.isVisible()
    assert win.results_scroll.height() > before + 100
    win.stage.fold.setChecked(False)
    qtbot.wait(150)
    assert win.stage.stack.isVisible() and abs(win.results_scroll.height() - before) < 30
