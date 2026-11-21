"""Tests for the software-rendered 3D scene widget (``app.workbench.viewport3d.SceneView``).

Pure logic / pixel tests: they run on the ``offscreen`` Qt platform (text renders as boxes there, which is fine).
The module is loaded from its file path so the tests do not depend on the rest of ``app.workbench``.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

pytestmark = pytest.mark.gui

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "app" / "workbench" / "viewport3d.py"

DARK = {"bg0": "#080F1D", "bg1": "#0D1729", "line": "#233556", "ink": "#E9F1FB", "ink2": "#A8B9D1", "ink3": "#6B819F",
        "solvent": "#35E0C8", "solute": "#FF8A5B", "accent": "#7AA7FF", "ok": "#4ADE80", "warn": "#FBBF24", "err": "#FB7185"}
LIGHT = {"bg0": "#F3F0E8", "bg1": "#FAF8F3", "line": "#DDD5C4", "ink": "#13233B", "ink2": "#44566F", "ink3": "#7A8AA1",
         "solvent": "#0E9F8E", "solute": "#D9531E", "accent": "#2F6BFF", "ok": "#15803D", "warn": "#B45309", "err": "#BE123C"}


def _load_module():
    spec = importlib.util.spec_from_file_location("sozlab_viewport3d_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load_module()


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def view(qapp):
    v = V.SceneView()
    v.resize(600, 400)
    yield v
    v.close()
    v.deleteLater()
    QtWidgets.QApplication.processEvents()


# ------------------------------------------------------------------------------------------------ helpers
def qimage_to_array(img: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8 RGB."""
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(h * img.bytesPerLine())
    a = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :]
    return a[:, :, [2, 1, 0]].astype(np.int16)


def luminance(rgb: np.ndarray) -> float:
    return float((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean() / 255.0)


def non_bg_fraction(img_a: np.ndarray, img_empty: np.ndarray, thr: int = 14) -> float:
    return float((np.abs(img_a - img_empty).max(axis=2) > thr).mean())


def random_ball(n: int, radius: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * radius * rng.random((n, 1)) ** (1 / 3)).astype(np.float32)


def rich_scene(v, n_sol: int = 5000, n_sv: int = 300, n_dn: int = 4000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    v.set_solute(random_ball(n_sol, 18.0, seed), roles=(rng.random(n_sol) < 0.02).astype(np.uint8))
    ox = ((rng.random((n_sv, 3)) - 0.5) * 50).astype(np.float32)
    st = np.zeros(n_sv, np.uint8)
    st[:30] = 1
    st[30:34] = 2
    st[34:38] = 3
    v.set_solvent(ox, None, st)
    dn = (rng.normal(size=(n_dn, 3)) * 4 + np.array([12.0, 0, 0])).astype(np.float32)
    v.set_density(dn, rng.random(n_dn), 0.0, 1.0, 0.9)
    v._fade_anim.stop()
    v._dn_fade = 1.0
    v.set_box(np.array([50, 50, 50, 90, 90, 90.0]), origin=np.array([-25, -25, -25.0]))


# ------------------------------------------------------------------------------------------------ 1. empty
def test_empty_widget_renders_and_exports(view):
    img = view.render_to_image(400, 300)
    assert isinstance(img, QtGui.QImage)
    assert (img.width(), img.height()) == (400, 300)
    arr = qimage_to_array(img)
    assert arr.std() > 0.5                      # the background gradient is not a flat fill
    pm = view.grab()
    assert not pm.isNull()
    assert view._last_error is None
    s = view.stats()
    assert set(s) >= {"fps", "n_solute", "n_solvent", "n_density", "last_paint_ms"}
    assert (s["n_solute"], s["n_solvent"], s["n_density"]) == (0, 0, 0)


@pytest.mark.parametrize("size", [(1, 1), (7, 5), (123, 77), (64, 700)])
def test_export_odd_sizes(view, size):
    rich_scene(view, 500, 60, 400)
    img = view.render_to_image(*size)
    assert (img.width(), img.height()) == size


# ------------------------------------------------------------------------------------------------ 2. content + rotation
def test_scene_is_not_blank_and_rotation_changes_the_image(view):
    view.resize(480, 320)
    empty = qimage_to_array(view.render_to_image(480, 320))
    rich_scene(view, 5000, 300, 4000)
    a = qimage_to_array(view.render_to_image(480, 320))
    assert non_bg_fraction(a, empty) > 0.02
    st = view.camera_state()
    q = np.asarray(st["quat"])
    qd = V._qaxis((0.3, 1.0, 0.2), 1.3)
    st["quat"] = list(V._qmul(qd, q))
    view.set_camera_state(st)
    b = qimage_to_array(view.render_to_image(480, 320))
    assert non_bg_fraction(b, a, thr=20) > 0.01
    assert view._last_error is None


# ------------------------------------------------------------------------------------------------ 3. themes
def test_theme_switch_changes_luminance(view):
    view.resize(480, 320)
    rich_scene(view, 3000, 200, 2000)
    view.set_theme(DARK)
    dark = qimage_to_array(view.render_to_image(480, 320))
    view.set_theme(LIGHT)
    light = qimage_to_array(view.render_to_image(480, 320))
    assert luminance(light) - luminance(dark) > 0.45
    # neither render is blank: compare with the empty scene of the same theme
    empty_light = V.SceneView()
    empty_light.resize(480, 320)
    empty_light.set_theme(LIGHT)
    el = qimage_to_array(empty_light.render_to_image(480, 320))
    assert non_bg_fraction(light, el) > 0.02
    empty_dark = V.SceneView()
    empty_dark.resize(480, 320)
    empty_dark.set_theme(DARK)
    ed = qimage_to_array(empty_dark.render_to_image(480, 320))
    assert non_bg_fraction(dark, ed) > 0.02


def test_partial_and_invalid_tokens_are_tolerated(view):
    view.set_theme({"bg0": "#102030"})
    view.set_theme({"bg0": "not-a-colour", "solvent": "#00ff00", "nonsense": "#fff"})
    view.set_theme({})
    view.set_solute(random_ball(300, 6.0))
    img = view.render_to_image(160, 120)
    assert img.width() == 160


# ------------------------------------------------------------------------------------------------ 4. state colouring
def _solvent_patch(view, state: int, theme: dict) -> tuple[np.ndarray, np.ndarray]:
    view.set_theme(theme)
    oxy = np.zeros((1, 3), np.float32)
    hyd = np.array([[[0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]], np.float32)
    view.set_solvent(oxy, hyd, np.array([state], np.uint8))
    view.set_options(show_box=False, show_axes=False, show_scale_bar=False, reduce_motion=True)
    img = qimage_to_array(view.render_to_image(400, 300))
    (xy,), _ = view.project_points(np.zeros((1, 3)))
    x, y = int(round(xy[0])), int(round(xy[1]))
    return img[y - 30:y + 30, x - 30:x + 30], img


def test_state_one_is_brighter_and_more_solvent_coloured_than_bulk(view):
    view.resize(400, 300)
    view.set_theme(DARK)
    view.set_options(show_box=False, show_axes=False, show_scale_bar=False, reduce_motion=True)
    empty = qimage_to_array(view.render_to_image(400, 300))
    bulk, _ = _solvent_patch(view, 0, DARK)
    zone, _ = _solvent_patch(view, 1, DARK)
    (xy,), _ = view.project_points(np.zeros((1, 3)))
    x, y = int(round(xy[0])), int(round(xy[1]))
    bg = empty[y - 30:y + 30, x - 30:x + 30]
    assert luminance(zone) > luminance(bulk) * 1.25 and luminance(zone) > luminance(bg) * 1.25
    # direction of the solvent token (#35E0C8) away from the background; energy of the patch along it
    u = np.array([0x35, 0xE0, 0xC8], float) - bg.reshape(-1, 3).mean(axis=0)
    u /= np.linalg.norm(u)
    along = lambda patch: float(np.maximum((patch - bg) @ u, 0).sum())
    assert along(zone) > 2.5 * max(along(bulk), 1.0)
    # the molecule's typical pixel is aquamarine-ish: green and blue clearly above red
    flat = zone.reshape(-1, 3)
    mol = flat[np.abs((zone - bg).reshape(-1, 3)).max(axis=1) > 60]
    assert mol.shape[0] > 20
    med = np.median(mol, axis=0)
    assert med[1] > med[0] + 40 and med[2] > med[0] + 20
    # bulk stays dim: its brightest pixels are far below the zone's
    assert bulk.reshape(-1, 3).sum(axis=1).max() < 0.75 * flat.sum(axis=1).max()


def test_state_three_ghost_is_coral_ring(view):
    view.resize(400, 300)
    view.set_theme(DARK)
    view.set_solvent(np.zeros((1, 3), np.float32), None, np.array([3], np.uint8))
    view.set_options(show_box=False, show_axes=False, show_scale_bar=False, reduce_motion=True)
    img = qimage_to_array(view.render_to_image(400, 300))
    (xy,), _ = view.project_points(np.zeros((1, 3)))
    x, y = int(round(xy[0])), int(round(xy[1]))
    patch = img[y - 40:y + 40, x - 40:x + 40].reshape(-1, 3)
    coral = patch[(patch[:, 0] > 150) & (patch[:, 0] > patch[:, 2] + 30)]
    assert coral.shape[0] > 15                                   # a coral outline exists


def test_anchor_role_uses_solute_token_colour(view):
    view.resize(400, 300)
    view.set_theme(DARK)
    pts = random_ball(40, 3.0)
    roles = np.zeros(40, np.uint8)
    roles[:6] = 1
    view.set_solute(pts, roles=roles)
    view.set_options(show_box=False, show_axes=False, show_scale_bar=False, reduce_motion=True)
    with_anchor = qimage_to_array(view.render_to_image(400, 300)).reshape(-1, 3)
    view.set_solute(pts, roles=np.zeros(40, np.uint8))
    without = qimage_to_array(view.render_to_image(400, 300)).reshape(-1, 3)
    warm = lambda a: int(((a[:, 0] > 170) & (a[:, 0] > a[:, 2] + 45)).sum())      # coral: strong red, weak blue
    assert warm(with_anchor) > warm(without) + 20


# ------------------------------------------------------------------------------------------------ 5. hover picking
def test_hover_picks_the_right_solvent_molecule(view):
    view.resize(600, 400)
    view.set_solvent(np.array([[0, 0, 0], [9, 0, 0]], np.float32), None, np.array([1, 0], np.uint8),
                     labels=["WAT-A", "WAT-B"])
    view.show()
    QtWidgets.QApplication.processEvents()
    got: list = []
    view.hovered.connect(got.append)
    xy, depth = view.project_points(np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert xy.shape == (2, 2) and depth.shape == (2,)
    QtTest.QTest.mouseMove(view, QtCore.QPoint(int(round(xy[0, 0])), int(round(xy[0, 1]))))
    QtTest.QTest.qWait(30)
    assert got, "hovered did not fire"
    assert got[-1]["kind"] == "solvent" and got[-1]["index"] == 1
    assert got[-1]["label"] == "WAT-B" and got[-1]["state"] == 0
    np.testing.assert_allclose(got[-1]["pos"], (9.0, 0.0, 0.0), atol=1e-5)
    n = len(got)
    QtTest.QTest.mouseMove(view, QtCore.QPoint(int(round(xy[1, 0])), int(round(xy[1, 1]))))
    QtTest.QTest.qWait(60)
    assert len(got) > n and got[-1]["kind"] == "solvent" and got[-1]["index"] == 0
    QtTest.QTest.mouseMove(view, QtCore.QPoint(5, 395))                  # far away from everything
    QtTest.QTest.qWait(60)
    assert got[-1] == {}


def test_click_emits_payload_but_drag_does_not(view):
    view.resize(600, 400)
    view.set_solvent(np.array([[0, 0, 0], [9, 0, 0]], np.float32))
    view.show()
    QtWidgets.QApplication.processEvents()
    clicks: list = []
    view.clicked.connect(clicks.append)
    (xy,), _ = view.project_points(np.array([[9.0, 0.0, 0.0]]))
    pt = QtCore.QPoint(int(round(xy[0])), int(round(xy[1])))
    QtTest.QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=pt)
    assert len(clicks) == 1 and clicks[0]["index"] == 1
    QtTest.QTest.mousePress(view, QtCore.Qt.MouseButton.LeftButton, pos=pt)
    QtTest.QTest.mouseMove(view, pt + QtCore.QPoint(40, 25))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.MouseButton.LeftButton, pos=pt + QtCore.QPoint(40, 25))
    assert len(clicks) == 1                                              # a drag is not a click


def test_pick_prefers_the_nearer_atom_and_maps_indices_around_nan(view):
    view.resize(600, 400)
    xyz = np.array([[0, 0, -6], [0, 0, 6], [np.nan, 0, 0], [20, 0, 0]], np.float32)   # [1] is in front of [0]
    view.set_solute(xyz, radii=np.array([2, 2, 2, 2], np.float32), labels=["far", "near", "bad", "side"])
    view.set_view_preset("z", animate=False)                              # both z-atoms project to the same pixel
    (xy,), _ = view.project_points(np.zeros((1, 3)))
    hit = view.pick(xy[0], xy[1])
    assert hit["kind"] == "solute" and hit["index"] in (0, 1)
    d = view.project_points(np.array([[0, 0, -6.0], [0, 0, 6.0]]))[1]
    assert hit["index"] == int(np.argmin(d))
    (sxy,), _ = view.project_points(np.array([[20.0, 0, 0]]))
    hit = view.pick(sxy[0], sxy[1])
    assert hit["index"] == 3 and hit["label"] == "side"                   # original index survives the NaN drop


def test_pick_density_when_nothing_else(view):
    view.resize(600, 400)
    view.set_density(np.array([[0, 0, 0], [5, 5, 0]], np.float32), np.array([1.0, 2.0]), 0.0, 2.0)
    (xy,), _ = view.project_points(np.array([[5.0, 5.0, 0.0]]))
    hit = view.pick(xy[0], xy[1])
    assert hit["kind"] == "density" and hit["index"] == 1
    assert view.pick(2.0, 2.0) == {}


# ------------------------------------------------------------------------------------------------ 6. bad input
def test_nonfinite_and_zero_length_inputs_are_tolerated(view):
    nan = np.nan
    view.set_solute(np.array([[0, 0, 0], [nan, 1, 1], [1, np.inf, 0], [2, 2, 2]], np.float32),
                    radii=np.array([1.5, nan, 1.5, -2.0], np.float32), roles=np.array([0, 1, 2, 1], np.uint8))
    assert view.stats()["n_solute"] == 2
    view.set_solvent(np.array([[0, 0, 0], [nan, 0, 0], [3, 3, 3]], np.float32),
                     np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0]], [[nan, 0, 0], [0, 0, 1]]], np.float32),
                     np.array([1, 2, 9], np.uint8))
    assert view.stats()["n_solvent"] == 2
    view.set_density(np.array([[0, 0, 0], [nan, 0, 0], [1, 1, 1]], np.float32), np.array([1.0, 2.0, nan]), 0.0, 1.0)
    assert view.stats()["n_density"] == 1
    view.set_box(np.array([nan, 1, 1, 90, 90, 90.0]))
    view.set_box(np.array([30, 30, 30, 90, 90, 90.0]), origin=np.array([nan, 0, 0]))
    view.focus_on(np.array([[nan, nan, nan]]))
    view.set_center_of_rotation(np.array([[np.inf, 0, 0]]))
    img = view.render_to_image(200, 150)
    assert img.width() == 200
    view.grab()
    assert view._last_error is None


def test_empty_and_none_inputs(view):
    view.set_solute(None)
    view.set_solute(np.zeros((0, 3), np.float32))
    view.set_solvent(None)
    view.set_solvent(np.zeros((0, 3)), np.zeros((0, 2, 3)), np.zeros(0, np.uint8), [])
    view.set_density(None, None, 0, 1)
    view.set_density(np.zeros((0, 3)), np.zeros(0), 0, 1)
    view.clear_density()
    view.set_box(None)
    view.set_hud([])
    view.set_hud(None)
    view.focus_on(np.zeros((0, 3)))
    view.set_center_of_rotation(None)
    view.set_camera_state({})
    view.set_camera_state({"quat": [0, 0, 0, 0], "zoom": -1, "center": [np.nan, 0, 0]})
    view.reset_view(animate=False)
    view.set_view_preset("bogus")
    view.grab()
    assert view.render_to_image(100, 80).width() == 100
    assert view._last_error is None


def test_mismatched_optional_arrays_fall_back_to_defaults(view):
    view.set_solute(random_ball(10, 3.0), radii=np.ones(3), roles=np.ones(4), labels=["a"])
    view.set_solvent(random_ball(5, 3.0), hydrogen_xyz=np.zeros((2, 2, 3)), state=np.ones(7))
    view.set_density(random_ball(5, 3.0), np.ones(4), 0, 1)
    view.grab()
    assert view._last_error is None
    assert view.stats()["n_solute"] == 10 and view.stats()["n_density"] == 0


# ------------------------------------------------------------------------------------------------ 7. performance
def test_performance_smoke_30k_solute_3k_solvent_20k_density(view):
    rng = np.random.default_rng(3)
    view.resize(900, 600)
    n = 30000
    sol = random_ball(n, 38.0, 0)
    view.set_solute(sol, radii=rng.choice([1.2, 1.5, 1.7, 1.9], n).astype(np.float32),
                    roles=(rng.random(n) < 0.01).astype(np.uint8))
    m = 3000
    ox = ((rng.random((m, 3)) - 0.5) * 100).astype(np.float32)
    hyd = (ox[:, None, :] + rng.normal(size=(m, 2, 3)).astype(np.float32) * 0.6)
    st = (rng.random(m) < 0.1).astype(np.uint8)
    st[:20] = 2
    st[20:30] = 3
    view.set_solvent(ox, hyd, st)
    dn = (rng.normal(size=(20000, 3)) * 7 + np.array([40.0, 0, 0])).astype(np.float32)
    view.set_density(dn, rng.random(20000), 0.0, 1.0)
    view.set_box(np.array([100, 100, 100, 90, 90, 90.0]), origin=np.array([-50, -50, -50.0]))
    view.set_options(reduce_motion=True)
    view.reset_view(animate=False)
    assert view.stats()["n_solute"] == n

    def one_repaint(i: int, interactive: bool) -> float:
        st_ = view.camera_state()
        st_["quat"] = list(V._qmul(V._qaxis((0.2, 1.0, 0.1), 0.05 + 0.03 * i), np.asarray(st_["quat"])))
        view._interacting = interactive
        view.set_camera_state(st_)
        t0 = time.perf_counter()
        view.grab()
        return (time.perf_counter() - t0) * 1000.0

    one_repaint(0, False)                                                 # warm-up (sprite bank, LUTs)
    full = [one_repaint(i + 1, False) for i in range(10)]
    inter = [one_repaint(i + 20, True) for i in range(10)]
    view._interacting = False
    med_full, med_inter = statistics.median(full), statistics.median(inter)
    print(f"\n[viewport3d perf] full-quality grab median {med_full:.1f} ms, interactive median {med_inter:.1f} ms "
          f"(900x600, 30k solute + 3k solvent + 20k density)")
    assert view._last_error is None
    assert med_full < 250.0
    assert med_inter < 250.0


# ------------------------------------------------------------------------------------------------ 8. camera state
def test_camera_state_round_trips_through_set_camera_state(view):
    rich_scene(view, 800, 60, 500)
    st = view.camera_state()
    json.dumps(st)                                                       # JSON-able
    assert set(st) >= {"quat", "zoom", "center"}
    q = V._qmul(V._qaxis((1, 2, 3), 0.9), np.asarray(st["quat"]))
    st2 = dict(st, quat=list(q), zoom=st["zoom"] * 1.7, center=[1.5, -2.0, 3.0])
    view.set_camera_state(st2)
    got = view.camera_state()
    np.testing.assert_allclose(got["quat"], V._qnorm(q), atol=1e-9)
    assert got["zoom"] == pytest.approx(st2["zoom"])
    np.testing.assert_allclose(got["center"], st2["center"])
    view.set_camera_state(json.loads(json.dumps(got)))
    again = view.camera_state()
    np.testing.assert_allclose(again["quat"], got["quat"], atol=1e-12)
    assert again["zoom"] == pytest.approx(got["zoom"])
    np.testing.assert_allclose(again["center"], got["center"])


def test_camera_changed_signal_fires(view):
    rich_scene(view, 200, 20, 100)
    seen: list = []
    view.cameraChanged.connect(lambda: seen.append(1))
    st = view.camera_state()
    st["zoom"] *= 1.2
    view.set_camera_state(st)
    QtTest.QTest.qWait(40)
    assert seen


# ------------------------------------------------------------------------------------------------ 9. motion
def test_reduce_motion_makes_reset_view_synchronous(view):
    rich_scene(view, 500, 40, 300)
    view.set_options(reduce_motion=True)
    st = view.camera_state()
    st["quat"] = list(V._qmul(V._qaxis((0, 1, 0), 2.0), np.asarray(st["quat"])))
    st["zoom"] *= 3.0
    view.set_camera_state(st)
    view.reset_view(animate=True)
    assert not view.is_animating()
    assert view._cam_anim.state() != QtCore.QAbstractAnimation.State.Running
    ref = V.SceneView()
    ref.resize(600, 400)
    rich_scene(ref, 500, 40, 300)
    ref.reset_view(animate=False)
    np.testing.assert_allclose(view.camera_state()["quat"], ref.camera_state()["quat"], atol=1e-9)
    assert view.camera_state()["zoom"] == pytest.approx(ref.camera_state()["zoom"])
    view.set_view_preset("x", animate=True)
    view.focus_on(np.array([[1.0, 2.0, 3.0]]), animate=True)
    view.set_camera_state(st, animate=True)
    assert not view.is_animating()


def test_animated_reset_view_runs_then_lands_on_the_target(view):
    rich_scene(view, 500, 40, 300)
    ref = V.SceneView()
    ref.resize(600, 400)
    rich_scene(ref, 500, 40, 300)
    ref.reset_view(animate=False)
    st = view.camera_state()
    st["quat"] = list(V._qmul(V._qaxis((0, 1, 0), 2.0), np.asarray(st["quat"])))
    view.set_camera_state(st)
    view.reset_view(animate=True)
    assert view.is_animating()
    QtTest.QTest.qWait(700)
    assert not view.is_animating()
    np.testing.assert_allclose(view.camera_state()["quat"], ref.camera_state()["quat"], atol=1e-6)


def test_pulse_runs_only_with_entered_or_left_molecules_and_honours_reduce_motion(view):
    ox = random_ball(20, 5.0)
    view.set_solvent(ox, None, np.zeros(20, np.uint8))
    assert not view.pulse_running()
    st = np.zeros(20, np.uint8)
    st[0] = 2
    view.set_solvent(ox, None, st)
    assert view.pulse_running()
    view.set_options(reduce_motion=True)
    assert not view.pulse_running()
    view.set_solvent(ox, None, st)
    assert not view.pulse_running()
    view.set_options(reduce_motion=False)
    view.set_solvent(ox, None, st)
    assert view.pulse_running()
    view.set_pulse_phase(0.25)
    view.set_solvent(ox, None, np.zeros(20, np.uint8))
    assert not view.pulse_running()


def test_density_fade_in_and_opacity_option(view):
    rich_scene(view, 300, 20, 3000)
    view.set_options(reduce_motion=False)
    dn = (np.random.default_rng(0).normal(size=(2000, 3)) * 3).astype(np.float32)
    view.set_density(dn, np.random.default_rng(1).random(2000), 0, 1)
    assert view._fade_anim.state() == QtCore.QAbstractAnimation.State.Running      # fades in over ~250 ms
    QtTest.QTest.qWait(400)
    assert view._dn_fade == pytest.approx(1.0)
    view.set_options(reduce_motion=True, density_opacity=0.0)
    off = qimage_to_array(view.render_to_image(300, 200))
    view.set_options(reduce_motion=True, density_opacity=1.0)
    on = qimage_to_array(view.render_to_image(300, 200))
    assert np.abs(on - off).max() > 25


# ------------------------------------------------------------------------------------------------ views + interaction
def test_view_presets_look_down_the_requested_axis(view):
    view.resize(600, 400)
    view.set_solute(np.array([[0, 0, 0]], np.float32))
    ax = np.eye(3) * 5.0
    view.set_view_preset("z", animate=False)
    s0, _ = view.project_points(np.zeros((1, 3)))
    s, d = view.project_points(ax)
    assert s[0, 0] > s0[0, 0] + 10 and abs(s[0, 1] - s0[0, 1]) < 1e-6              # x -> right
    assert s[1, 1] < s0[0, 1] - 10 and abs(s[1, 0] - s0[0, 0]) < 1e-6              # y -> up
    assert d[2] < d[0] and d[2] < d[1]                                              # z points at the viewer
    view.set_view_preset("x", animate=False)
    s, d = view.project_points(ax)
    assert abs(s[0, 0] - s0[0, 0]) < 1e-6 and d[0] < d[1]                           # x towards the viewer
    view.set_view_preset("y", animate=False)
    s, d = view.project_points(ax)
    assert abs(s[1, 0] - s0[0, 0]) < 1e-6 and d[1] < d[0]
    view.set_view_preset("iso", animate=False)
    s, d = view.project_points(ax)
    assert all(abs(s[i, 0] - s0[0, 0]) > 1 or abs(s[i, 1] - s0[0, 1]) > 1 for i in range(3))


def test_wheel_zooms_towards_the_cursor_keeping_the_point_under_it(view):
    view.resize(600, 400)
    rich_scene(view, 500, 40, 300)
    view.set_options(reduce_motion=True)
    view.show()
    QtWidgets.QApplication.processEvents()
    world = np.array([[6.0, -4.0, 2.0]])
    (before,), _ = view.project_points(world)
    z0 = view.camera_state()["zoom"]
    pos = QtCore.QPointF(float(before[0]), float(before[1]))
    ev = QtGui.QWheelEvent(pos, view.mapToGlobal(pos), QtCore.QPoint(), QtCore.QPoint(0, 240),
                           QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                           QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(view, ev)
    assert view.camera_state()["zoom"] > z0 * 1.2
    (after,), _ = view.project_points(world)
    assert np.hypot(*(after - before)) < 0.6


def test_left_drag_rotates_and_middle_drag_pans(view):
    view.resize(600, 400)
    rich_scene(view, 800, 60, 400)
    view.set_options(reduce_motion=True)
    view.show()
    QtWidgets.QApplication.processEvents()
    seen: list = []
    view.cameraChanged.connect(lambda: seen.append(1))
    q0 = np.asarray(view.camera_state()["quat"])
    c0 = np.asarray(view.camera_state()["center"])
    centre = QtCore.QPoint(300, 200)
    QtTest.QTest.mousePress(view, QtCore.Qt.MouseButton.LeftButton, pos=centre)
    for i in range(1, 9):
        QtTest.QTest.mouseMove(view, centre + QtCore.QPoint(10 * i, 4 * i))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.MouseButton.LeftButton, pos=centre + QtCore.QPoint(80, 32))
    q1 = np.asarray(view.camera_state()["quat"])
    assert abs(float(np.dot(q0, q1))) < 0.999 and seen
    np.testing.assert_allclose(view.camera_state()["center"], c0, atol=1e-6)         # rotation keeps the centre
    QtTest.QTest.mousePress(view, QtCore.Qt.MouseButton.MiddleButton, pos=centre)
    for i in range(1, 6):
        QtTest.QTest.mouseMove(view, centre + QtCore.QPoint(8 * i, 0))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.MouseButton.MiddleButton, pos=centre + QtCore.QPoint(40, 0))
    assert np.linalg.norm(np.asarray(view.camera_state()["center"]) - c0) > 0.5
    QtTest.QTest.mouseDClick(view, QtCore.Qt.MouseButton.LeftButton, pos=centre)    # double click resets
    ref = V.SceneView()
    ref.resize(600, 400)
    rich_scene(ref, 800, 60, 400)
    ref.reset_view(animate=False)
    np.testing.assert_allclose(view.camera_state()["quat"], ref.camera_state()["quat"], atol=1e-9)


def test_interaction_renders_reduced_then_settles_to_full_quality(view):
    view.resize(600, 400)
    rng = np.random.default_rng(0)
    view.set_solute(random_ball(20000, 30.0), roles=(rng.random(20000) < 0.01).astype(np.uint8))
    view.set_options(reduce_motion=True)
    view.show()
    view.grab()
    assert view._scene.interactive is False
    centre = QtCore.QPoint(300, 200)
    QtTest.QTest.mousePress(view, QtCore.Qt.MouseButton.LeftButton, pos=centre)
    QtTest.QTest.mouseMove(view, centre + QtCore.QPoint(25, 10))
    QtTest.QTest.mouseMove(view, centre + QtCore.QPoint(50, 20))
    view.grab()                                                       # a frame while the mouse is moving
    assert view._interacting and view._scene.interactive is True
    QtTest.QTest.mouseRelease(view, QtCore.Qt.MouseButton.LeftButton, pos=centre + QtCore.QPoint(50, 20))
    QtTest.QTest.qWait(260)                                           # ~120 ms settle delay
    assert not view._interacting
    view.grab()
    assert view._scene.interactive is False                           # re-rendered at full quality
    # a click without movement never degrades the picture
    QtTest.QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=centre)
    view.grab()
    assert view._scene.interactive is False


def test_keyboard_shortcuts(view):
    view.resize(600, 400)
    rich_scene(view, 300, 20, 100)
    view.set_options(reduce_motion=True)
    view.show()
    view.setFocus()
    QtWidgets.QApplication.processEvents()
    assert view.focusPolicy() != QtCore.Qt.FocusPolicy.NoFocus
    iso = np.asarray(view.camera_state()["quat"])
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_Left)
    assert abs(float(np.dot(iso, view.camera_state()["quat"]))) < 0.99999
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_3)
    np.testing.assert_allclose(view.camera_state()["quat"], V._preset_quat("z"), atol=1e-9)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_1)
    np.testing.assert_allclose(np.abs(view.camera_state()["quat"]), np.abs(V._preset_quat("x")), atol=1e-9)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_2)
    np.testing.assert_allclose(np.abs(view.camera_state()["quat"]), np.abs(V._preset_quat("y")), atol=1e-9)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_0)
    np.testing.assert_allclose(np.abs(view.camera_state()["quat"]), np.abs(iso), atol=1e-9)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_Up)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_R)
    np.testing.assert_allclose(np.abs(view.camera_state()["quat"]), np.abs(iso), atol=1e-9)


def test_focus_on_centres_the_points_and_keeps_zoom(view):
    view.resize(600, 400)
    rich_scene(view, 500, 40, 100)
    view.set_options(reduce_motion=True)
    z0 = view.camera_state()["zoom"]
    pts = np.array([[4.0, 4.0, 4.0], [6.0, 4.0, 4.0]])
    view.focus_on(pts, animate=True)
    np.testing.assert_allclose(view.camera_state()["center"], [5.0, 4.0, 4.0], atol=1e-9)
    assert view.camera_state()["zoom"] == pytest.approx(z0)
    view.focus_on(np.array([[-300.0, 0, 0], [300.0, 0, 0]]), animate=True)         # would be off screen -> zoom out
    assert view.camera_state()["zoom"] < z0


def test_reset_view_frames_the_data_and_the_box_when_comparable(view):
    view.resize(600, 400)
    pts = random_ball(800, 10.0)
    view.set_solute(pts)
    corners = np.array([[i, j, k] for i in (-20, 20) for j in (-20, 20) for k in (-20, 20)], float)
    view.set_box(np.array([40, 40, 40, 90, 90, 90.0]), origin=np.array([-20, -20, -20.0]))
    view.reset_view(animate=False)
    xy, _ = view.project_points(corners)
    assert (xy[:, 0] > 0).all() and (xy[:, 0] < 600).all() and (xy[:, 1] > 0).all() and (xy[:, 1] < 400).all()
    # a cell that dwarfs the data is not framed: the molecule fills the view instead of shrinking to a speck
    view.set_box(np.array([300, 300, 300, 90, 90, 90.0]), origin=np.array([-150, -150, -150.0]))
    view.reset_view(animate=False)
    xy, _ = view.project_points(pts)
    assert xy[:, 1].max() - xy[:, 1].min() > 0.5 * 400
    assert (xy[:, 0] > 0).all() and (xy[:, 0] < 600).all() and (xy[:, 1] > 0).all() and (xy[:, 1] < 400).all()


def test_center_of_rotation_stays_fixed_on_screen(view):
    view.resize(600, 400)
    rich_scene(view, 300, 20, 100)
    view.set_options(reduce_motion=True)
    cor = np.array([[8.0, 3.0, -2.0]])
    view.set_center_of_rotation(cor)
    (before,), _ = view.project_points(cor)
    view._rotate_view(V._qaxis((0.3, 1.0, 0.1), 0.9))
    (after,), _ = view.project_points(cor)
    assert np.hypot(*(after - before)) < 1e-6
    view.set_center_of_rotation(None)
    view._rotate_view(V._qaxis((0.3, 1.0, 0.1), 0.4))


# ------------------------------------------------------------------------------------------------ box / options / HUD
def test_triclinic_box_and_options_toggle_overlays(view):
    view.resize(400, 300)
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=False)
    base = qimage_to_array(view.render_to_image(400, 300))
    view.set_box(np.array([30, 28, 26, 80, 95, 105.0]), origin=np.array([-15, -14, -13.0]))
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=False)
    hidden = qimage_to_array(view.render_to_image(400, 300))
    np.testing.assert_array_equal(base, hidden)                                     # box hidden -> identical
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=True)
    shown = qimage_to_array(view.render_to_image(400, 300))
    assert non_bg_fraction(shown, base, thr=10) > 0.004
    view.set_options(reduce_motion=True, show_axes=True, show_scale_bar=True, show_box=True)
    full = qimage_to_array(view.render_to_image(400, 300))
    assert non_bg_fraction(full, shown, thr=10) > 0.002
    view.set_hud(["Frame 12", "in zone 40", "+3 / -2"])
    hud = qimage_to_array(view.render_to_image(400, 300))
    assert non_bg_fraction(hud, full, thr=10) > 0.01
    assert view.camera_state()["zoom"] > 0


def test_solute_styles_render_differently(view):
    view.resize(300, 220)
    view.set_solute(random_ball(1500, 10.0))
    view.set_options(reduce_motion=True, solute_style="spheres", show_axes=False, show_scale_bar=False)
    sph = qimage_to_array(view.render_to_image(300, 220))
    view.set_options(reduce_motion=True, solute_style="points", show_axes=False, show_scale_bar=False)
    pts = qimage_to_array(view.render_to_image(300, 220))
    view.set_options(reduce_motion=True, solute_style="ribbon-points", show_axes=False, show_scale_bar=False)
    rib = qimage_to_array(view.render_to_image(300, 220))
    view.set_options(solute_style="nonsense")
    assert view._opts["solute_style"] == "spheres"
    assert np.abs(sph - pts).max() > 30 and np.abs(pts - rib).max() > 0


def test_light_theme_density_darkens_instead_of_washing_out(view):
    view.resize(300, 220)
    view.set_theme(LIGHT)
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=False)
    rng = np.random.default_rng(2)
    view.set_density((rng.normal(size=(6000, 3)) * 2.5).astype(np.float32), rng.random(6000), 0, 1, 1.0)
    on = qimage_to_array(view.render_to_image(300, 220))
    view.clear_density()
    off = qimage_to_array(view.render_to_image(300, 220))
    assert luminance(on) < luminance(off) - 0.004
    assert (on.astype(int) - off.astype(int)).max() <= 6                           # multiply never brightens


def test_dark_theme_density_adds_light(view):
    view.resize(300, 220)
    view.set_theme(DARK)
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=False)
    rng = np.random.default_rng(2)
    view.set_density((rng.normal(size=(6000, 3)) * 2.5).astype(np.float32), rng.random(6000), 0, 1, 1.0)
    on = qimage_to_array(view.render_to_image(300, 220))
    view.clear_density()
    off = qimage_to_array(view.render_to_image(300, 220))
    assert luminance(on) > luminance(off) + 0.01
    assert (on.astype(int) - off.astype(int)).min() >= -6


def test_dense_solute_occludes_density_behind_it(view):
    view.resize(300, 220)
    view.set_theme(DARK)
    view.set_options(reduce_motion=True, show_axes=False, show_scale_bar=False, show_box=False)
    view.set_view_preset("z", animate=False)
    view.set_solute(random_ball(3000, 6.0))
    rng = np.random.default_rng(4)
    cloud = (rng.normal(size=(3000, 3)) * 1.2).astype(np.float32)
    cloud[:, 2] -= 25.0                                                              # far behind the solute
    view.set_density(cloud, rng.random(3000), 0, 1, 1.0)
    behind = qimage_to_array(view.render_to_image(300, 220))
    cloud[:, 2] += 50.0                                                              # now in front of it
    view.set_density(cloud, rng.random(3000), 0, 1, 1.0)
    front = qimage_to_array(view.render_to_image(300, 220))
    view.clear_density()
    none = qimage_to_array(view.render_to_image(300, 220))
    gain_behind = float(np.maximum(behind - none, 0).sum())
    gain_front = float(np.maximum(front - none, 0).sum())
    assert gain_front > gain_behind * 1.5


# ------------------------------------------------------------------------------------------------ misc / contract
def test_hidpi_widget_grab_has_physical_pixel_size(view):
    view.resize(300, 200)
    rich_scene(view, 300, 20, 200)
    pm = view.grab()
    assert pm.width() == int(round(300 * pm.devicePixelRatio()))


def test_widget_paint_and_export_agree(view):
    """grab() (baked overlays, native resolution) and render_to_image() (1.5x supersampled) show the same picture."""
    view.resize(360, 240)
    rich_scene(view, 1500, 120, 1500)
    view.set_hud(["Frame 7", "in zone 30"])
    view.set_options(reduce_motion=True)
    view.show()
    pm = view.grab()
    dpr = pm.devicePixelRatio()
    a = qimage_to_array(pm.toImage())
    b = qimage_to_array(view.render_to_image(a.shape[1], a.shape[0]))
    assert a.shape == b.shape and dpr >= 1.0
    assert np.abs(a - b).mean() < 6.0                                  # same layout; only edge smoothing differs
    assert (np.abs(a - b).max(axis=2) > 90).mean() < 0.02


def test_resize_and_repeated_theme_changes_keep_working(view):
    rich_scene(view, 600, 40, 300)
    for w, h in ((320, 240), (801, 333), (200, 150)):
        view.resize(w, h)
        view.grab()
    for tk in (DARK, LIGHT, DARK):
        view.set_theme(tk)
        view.grab()
    assert view._last_error is None


def test_set_scene_accepts_workbench_scene_dict(view):
    rng = np.random.default_rng(0)
    out = {"ok": True, "frame": 3, "solute_xyz": random_ball(200, 8.0), "solute_roles": np.r_[np.ones(10), np.zeros(190)].astype(np.uint8),
           "solvent_xyz": random_ball(50, 12.0), "solvent_state": rng.integers(0, 4, 50).astype(np.uint8),
           "solvent_labels": [f"WAT{i}" for i in range(50)], "box": np.array([40, 40, 40, 90, 90, 90.0]),
           "center": np.zeros(3, np.float32)}
    view.set_scene(out)
    assert view.stats()["n_solute"] == 200 and view.stats()["n_solvent"] == 50
    view.grab()
    assert view._last_error is None
    view.clear_scene()
    assert view.stats()["n_solute"] == 0


def test_module_only_needs_pyqt6_and_numpy():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    allowed = {"__future__", "functools", "math", "sys", "time", "traceback", "typing", "numpy", "PyQt6", "app"}
    assert roots <= allowed, roots - allowed
    assert not ({"scipy", "pyqtgraph", "OpenGL", "matplotlib"} & roots)


def test_public_api_surface():
    expected = ["set_theme", "set_solute", "set_solvent", "set_density", "clear_density", "set_box", "set_hud",
                "set_options", "reset_view", "set_view_preset", "focus_on", "set_center_of_rotation", "camera_state",
                "set_camera_state", "render_to_image", "stats", "set_pulse_phase", "project_points"]
    for name in expected:
        assert callable(getattr(V.SceneView, name)), name
    for sig in ("hovered", "clicked", "cameraChanged"):
        assert hasattr(V.SceneView, sig)
