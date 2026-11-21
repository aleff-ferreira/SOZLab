"""Render the SOZLab demo video from the real Workbench (no screen recorder, fully scripted and reproducible).

The script starts the actual application offscreen, drives it through the same widgets and signals a person uses (drop files,
pick a part of the molecule, drag the distance line, Run, scrub the frames, switch tabs, export, verify), grabs the window
once per video frame and adds a virtual mouse pointer plus short captions. Nothing is mocked: every number, chart and 3D frame
in the video is computed by the engine while it is being filmed.

    python bench/media/make_demo_video.py                        # water around Na+ / Cl- (TIP3P, OpenMM; see bench/data)
    python bench/media/make_demo_video.py --system demo          # the synthetic demo system, no other data needed
    python bench/media/make_demo_video.py --preview              # quick low-resolution dry run (10 fps)

Outputs (docs/media): sozlab-demo.mp4, sozlab-hero.gif (short loop for the README), poster.png, social-preview.png, timeline.json.
Requires ffmpeg on PATH and Pillow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from app.workbench import icons  # noqa: E402
from app.workbench.theme import Theme  # noqa: E402
from app.workbench.window import WorkbenchWindow  # noqa: E402

Qt = QtCore.Qt


# ------------------------------------------------------------------------------------------------ small maths
def smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def qaxis(axis, angle):
    s = math.sin(angle / 2)
    n = math.sqrt(sum(c * c for c in axis)) or 1.0
    return [math.cos(angle / 2), axis[0] / n * s, axis[1] / n * s, axis[2] / n * s]


# ------------------------------------------------------------------------------------------------ video encoder
class Encoder:
    def __init__(self, path: Path, size, fps: int, crf: int = 20):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg is required on PATH to encode the video")
        w, h = size
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, image: QtGui.QImage) -> None:
        img = image if image.format() == QtGui.QImage.Format.Format_RGB32 else image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        self.proc.stdin.write(bytes(ptr))
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


# ------------------------------------------------------------------------------------------------ director
class Director:
    def __init__(self, out_mp4: Path, size=(1600, 900), fps=30, system="tip3p", preview=False):
        self.size, self.fps, self.preview = size, fps, preview
        self.band = 70 if preview else 86  # caption band above the application, so a caption never covers the interface
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.theme = Theme.instance()
        self.theme.mode, self.theme.reduce_motion = "dark", True
        self.theme.apply(self.app)
        self.tmp = Path(tempfile.mkdtemp(prefix="sozlab_video_"))
        self.win = WorkbenchWindow(demo_dir=str(self.tmp / "demo"))
        self.win.toast.show_message = lambda *a, **k: None  # transient toasts would cover the charts in a film
        self.win.resize(size[0], size[1] - self.band)
        self.win.show()
        self.win.body.setSizes([430, size[0] - 430 - 330, 330])
        self.s = self.win.session
        self.enc = Encoder(out_mp4, size, fps, crf=26 if preview else 19)
        self.cursor = QtCore.QPointF(size[0] * 0.62, size[1] * 0.58)
        self.cursor_on = False
        self.ripple = None  # progress 0..1 of the click ripple
        self.caption = None  # (step, text, alpha)
        self.card = None  # (kind, alpha)
        self.pills = []  # flying file pills: (label, QPointF, alpha)
        self.marks = {}
        self.frame_no = 0
        self.got = {"scene": 0, "series": 0}
        self.s.sceneReady.connect(lambda o: self.got.__setitem__("scene", o.get("token", 0)))
        self.s.seriesReady.connect(lambda o: self.got.__setitem__("series", o.get("token", 0)))
        self.win._preview_timer.setInterval(0)
        self.orbit_deg = 0.0
        self.pump(300)

    # ------------------------------------------------------------------------------------------ frame capture
    def pump(self, ms: int = 0) -> None:
        t0 = time.time()
        self.app.processEvents()
        while (time.time() - t0) * 1000 < ms:
            self.app.processEvents()
            time.sleep(0.003)

    def settle(self, timeout: float = 15.0) -> None:
        """Wait until the newest scene / occupancy preview requested from the background threads has arrived."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.app.processEvents()
            pv = self.s.preview
            quiet = not self.win._preview_timer.isActive()
            if quiet and self.got["scene"] >= pv.latest.get("scene", 0) and self.got["series"] >= pv.latest.get("series", 0):
                return
            time.sleep(0.003)

    def wait(self, cond, timeout=120.0, what=""):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.005)
        raise TimeoutError(what)

    def compose(self) -> QtGui.QImage:
        W, H = self.size
        canvas = QtGui.QImage(W, H, QtGui.QImage.Format.Format_RGB32)
        p = QtGui.QPainter(canvas)
        p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing
                         | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self._paint_band(p, W)
        p.drawImage(0, self.band, self.win.grab().toImage())
        p.save()
        p.translate(0, self.band)  # window coordinates from here on
        for label, pos, alpha in self.pills:
            self._paint_pill(p, label, pos, alpha)
        if self.cursor_on:
            self._paint_cursor(p)
        p.restore()
        if self.card:
            self._paint_card(p, W, H, *self.card)
        p.end()
        return canvas

    def frame(self, hold: int = 1) -> None:
        img = self.compose()
        for _ in range(hold):
            self.enc.write(img)
            self.frame_no += 1

    def step(self, hold: int = 1) -> None:
        self.pump()
        self.frame(hold)

    def hold(self, seconds: float, orbit: float = 0.0) -> None:
        for _ in range(max(1, int(seconds * self.fps))):
            if orbit:
                self.orbit(orbit)
            self.step()

    def mark(self, name: str) -> None:
        self.marks[name] = round(self.frame_no / self.fps, 2)

    # ------------------------------------------------------------------------------------------ overlays
    def _font(self, size: float, bold=True, mono=False) -> QtGui.QFont:
        f = self.theme.font("mono" if mono else "display", size, QtGui.QFont.Weight.Bold if bold else QtGui.QFont.Weight.Normal)
        return f

    def _paint_band(self, p, W):
        """The strip above the application: brand on the left, the current step's caption in the middle."""
        t = self.theme.t
        grad = QtGui.QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QtGui.QColor(t["bg0"]))
        grad.setColorAt(1, QtGui.QColor("#0B1B33"))
        p.fillRect(QtCore.QRectF(0, 0, W, self.band), grad)
        p.fillRect(QtCore.QRectF(0, self.band - 2, W, 2), QtGui.QColor(t["line"]))
        if not self.caption:
            return
        step, text, alpha = self.caption
        if alpha <= 0:
            return
        p.save()
        p.setOpacity(alpha)
        font = self._font(self.band * 0.30)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        badge = self.band * 0.5
        total = badge + 16 + tw
        x = (W - total) / 2
        y = (self.band - badge) / 2 + (1 - alpha) * 8
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(t["solvent"]))
        p.drawEllipse(QtCore.QRectF(x, y, badge, badge))
        p.setPen(QtGui.QColor(t["accent_ink"]))
        p.setFont(self._font(self.band * 0.24))
        p.drawText(QtCore.QRectF(x, y, badge, badge), Qt.AlignmentFlag.AlignCenter, str(step))
        p.setPen(QtGui.QColor(t["ink"]))
        p.setFont(font)
        p.drawText(QtCore.QRectF(x + badge + 16, 0, tw + 30, self.band), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        p.restore()

    def _paint_cursor(self, p):
        t = self.theme.t
        if self.ripple is not None:
            r = 8 + 34 * ease_out(self.ripple)
            c = QtGui.QColor(t["accent_hi"] if hasattr(t, "get") and "accent_hi" in t else t["accent"])
            c.setAlphaF(max(0.0, 0.8 * (1 - self.ripple)))
            p.setPen(QtGui.QPen(c, 3.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self.cursor, r, r)
        pts = [(0, 0), (0, 21), (5.6, 16.4), (9.6, 25.6), (13.2, 24), (9.2, 15.2), (16.4, 15.2)]
        s = 1.3
        poly = QtGui.QPolygonF([QtCore.QPointF(self.cursor.x() + x * s, self.cursor.y() + y * s) for x, y in pts])
        shadow = QtGui.QPolygonF([QtCore.QPointF(self.cursor.x() + x * s + 2, self.cursor.y() + y * s + 3) for x, y in pts])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 110))
        p.drawPolygon(shadow)
        p.setPen(QtGui.QPen(QtGui.QColor(10, 14, 24), 1.6, join=Qt.PenJoinStyle.RoundJoin))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawPolygon(poly)

    def _paint_pill(self, p, label, pos, alpha):
        t = self.theme.t
        p.save()
        p.setOpacity(alpha)
        font = self._font(12.5, bold=True, mono=True)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        w, h = fm.horizontalAdvance(label) + 54, 44
        rect = QtCore.QRectF(pos.x() - w / 2, pos.y() - h / 2, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 110))
        p.drawRoundedRect(rect.translated(0, 4), 12, 12)
        p.setBrush(QtGui.QColor(t["bg3"]))
        p.setPen(QtGui.QPen(QtGui.QColor(t["accent"]), 1.8))
        p.drawRoundedRect(rect, 12, 12)
        p.drawPixmap(QtCore.QRectF(rect.x() + 12, rect.y() + 11, 22, 22).toRect(), icons.pixmap("file", t["accent"], 22, 1.9))
        p.setPen(QtGui.QColor(t["ink"]))
        p.drawText(QtCore.QRectF(rect.x() + 42, rect.y(), w - 46, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        p.restore()

    def _paint_card(self, p, W, H, kind, alpha):
        if alpha <= 0:
            return
        t = self.theme.t
        p.save()
        p.setOpacity(alpha)
        grad = QtGui.QLinearGradient(0, 0, W * 0.4, H)
        grad.setColorAt(0, QtGui.QColor(t["bg0"]))
        grad.setColorAt(1, QtGui.QColor("#0A1B33"))
        p.fillRect(QtCore.QRectF(0, 0, W, H), grad)
        # faint solvent dots: the idea of "water around something" without needing data
        rng = [(0.18, 0.28, 7), (0.82, 0.22, 9), (0.12, 0.74, 8), (0.9, 0.72, 6), (0.3, 0.88, 5), (0.7, 0.12, 5), (0.5, 0.93, 6)]
        for fx, fy, r in rng:
            c = QtGui.QColor(t["solvent"])
            c.setAlpha(60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(QtCore.QPointF(W * fx, H * fy), r * 2.2, r * 2.2)
        logo = icons.logo_pixmap(t["solvent"], t["solute"], 150)
        p.drawPixmap(QtCore.QPointF(W / 2 - 75, H * 0.20), logo)
        p.setPen(QtGui.QColor(t["ink"]))
        p.setFont(self._font(64))
        p.drawText(QtCore.QRectF(0, H * 0.20 + 160, W, 90), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "SOZLab")
        p.setPen(QtGui.QColor(t["ink2"]))
        p.setFont(self._font(23, bold=False))
        if kind == "title":
            lines = ["See the solvent around your molecule.", "Occupancy, residence and density, with the uncertainty attached."]
        else:
            lines = ["Solvent occupancy zones you can trust.", "Every number: its unit, its uncertainty, when to believe it."]
        for i, line in enumerate(lines):
            p.drawText(QtCore.QRectF(0, H * 0.20 + 258 + i * 40, W, 40), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, line)
        if kind == "end":
            p.setFont(self._font(18, bold=True, mono=True))
            p.setPen(QtGui.QColor(t["solvent"]))
            p.drawText(QtCore.QRectF(0, H * 0.20 + 372, W, 40), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       "pip install -e .     sozlab-gui --demo     sozlab-cli selftest")
            p.setFont(self._font(15, bold=False))
            p.setPen(QtGui.QColor(t["ink3"]))
            p.drawText(QtCore.QRectF(0, H * 0.20 + 416, W, 36), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       "github.com/aleff-ferreira/SOZLab   ·   MIT license   ·   Windows / Linux / macOS (Qt)")
        p.restore()

    # ------------------------------------------------------------------------------------------ cursor and captions
    def centre(self, w: QtWidgets.QWidget, dx=0.0, dy=0.0) -> QtCore.QPointF:
        pt = w.mapTo(self.win, w.rect().center())
        return QtCore.QPointF(pt.x() + dx, pt.y() + dy)

    def cursor_to(self, target: QtCore.QPointF, seconds: float, arc: float = 0.10) -> None:
        self.cursor_on = True
        a = QtCore.QPointF(self.cursor)
        d = QtCore.QPointF(target.x() - a.x(), target.y() - a.y())
        dist = math.hypot(d.x(), d.y()) or 1.0
        ctrl = QtCore.QPointF((a.x() + target.x()) / 2 - d.y() / dist * dist * arc, (a.y() + target.y()) / 2 + d.x() / dist * dist * arc)
        n = max(1, int(seconds * self.fps))
        for i in range(1, n + 1):
            t = smooth(i / n)
            x = (1 - t) ** 2 * a.x() + 2 * (1 - t) * t * ctrl.x() + t ** 2 * target.x()
            y = (1 - t) ** 2 * a.y() + 2 * (1 - t) * t * ctrl.y() + t ** 2 * target.y()
            self.cursor = QtCore.QPointF(x, y)
            self.step()

    def click(self, widget, action=None, travel=0.8, dx=0.0, dy=0.0) -> None:
        self.cursor_to(self.centre(widget, dx, dy), travel)
        button = widget if isinstance(widget, QtWidgets.QAbstractButton) else None
        if button is not None:
            button.setDown(True)
        for _ in range(4):
            self.step()
        if button is not None:
            button.setDown(False)
        (action or (button.click if button is not None else (lambda: None)))()
        for i in range(12):
            self.ripple = i / 12
            self.step()
        self.ripple = None

    def show_caption(self, step: int, text: str, fade: float = 0.35) -> None:
        n = max(1, int(fade * self.fps))
        for i in range(1, n + 1):
            self.caption = (step, text, smooth(i / n))
            self.step()

    def hide_caption(self, fade: float = 0.3) -> None:
        if not self.caption:
            return
        step, text, _ = self.caption
        n = max(1, int(fade * self.fps))
        for i in range(1, n + 1):
            self.caption = (step, text, 1 - smooth(i / n))
            self.step()
        self.caption = None

    def fade_card(self, kind: str, fade_in: bool, seconds: float = 0.5) -> None:
        n = max(1, int(seconds * self.fps))
        for i in range(1, n + 1):
            self.card = (kind, smooth(i / n) if fade_in else 1 - smooth(i / n))
            self.step()
        if not fade_in:
            self.card = None

    # ------------------------------------------------------------------------------------------ 3D and layout
    def orbit(self, degrees: float) -> None:
        sc = self.win.stage.scene
        if not hasattr(sc, "camera_state"):
            return
        st = sc.camera_state()
        st["quat"] = qmul(qaxis((0, 1, 0), math.radians(degrees)), st["quat"])
        sc.set_camera_state(st, animate=False)
        self.orbit_deg += degrees

    def zoom_scene(self, factor: float) -> None:
        sc = self.win.stage.scene
        st = sc.camera_state()
        st["zoom"] = st["zoom"] * factor
        sc.set_camera_state(st, animate=False)

    def split(self, stage_h: int, seconds: float = 0.8) -> None:
        sizes = self.win.centre.sizes()
        total = sum(sizes)
        n = max(1, int(seconds * self.fps))
        for i in range(1, n + 1):
            t = smooth(i / n)
            top = int(lerp(sizes[0], stage_h, t))
            self.win.centre.setSizes([top, max(total - top, 1)])
            self.step()

    def line_point(self, r: float) -> QtCore.QPointF:
        guide = self.win.zone_card.guide
        vb = guide.plot.getPlotItem().getViewBox()
        rect = vb.viewRect()
        pt = vb.mapViewToScene(QtCore.QPointF(r, rect.top() + rect.height() * 0.62))
        local = guide.plot.mapFromScene(pt)
        g = guide.plot.mapTo(self.win, local)
        return QtCore.QPointF(g.x(), g.y())

    def drag_cutoff(self, r0: float, r1: float, seconds: float) -> None:
        guide = self.win.zone_card.guide
        n = max(1, int(seconds * self.fps))
        for i in range(1, n + 1):
            r = lerp(r0, r1, smooth(i / n))
            guide._line.setValue(r)  # the same signal path as dragging the line with the mouse
            self.cursor = self.line_point(r)
            self.settle()
            self.pump(2)
            self.frame()

    def count_up(self, seconds: float = 0.6) -> None:
        tiles = [t for t in self.win.results.tiles.tiles if t.isVisible() and t._value is not None]
        finals = [(t, t.value_label.text()) for t in tiles]
        n = max(1, int(seconds * self.fps))
        for i in range(1, n):
            e = ease_out(i / n)
            for t in tiles:
                t.value_label.setText(t._fmt(t._value * e))
            self.step()
        for t, txt in finals:
            t.value_label.setText(txt)

    # ------------------------------------------------------------------------------------------ data
    def load(self, topology: str, trajectory: str) -> None:
        def fly():
            drop = self.win.import_card.drop
            target = self.centre(drop)
            labels = [Path(topology).name, Path(trajectory).name]
            n = int(1.3 * self.fps)
            for i in range(1, n + 1):
                t = ease_out(i / n)
                self.pills = []
                for k, lab in enumerate(labels):
                    start = QtCore.QPointF(self.size[0] * (0.74 + 0.05 * k), self.size[1] * (0.30 + 0.16 * k))
                    end = QtCore.QPointF(target.x(), target.y() + (k - 0.5) * 56)
                    self.pills.append((lab, QtCore.QPointF(lerp(start.x(), end.x(), t), lerp(start.y(), end.y(), t)), min(1.0, i / 8)))
                self.step()
            for i in range(6):
                self.pills = [(lab, pos, 1 - i / 6) for lab, pos, _a in self.pills]
                self.step()
            self.pills = []

        fly()
        self.win.import_card.drop.filesDropped.emit([topology, trajectory])  # what dropping the two files does
        self.wait(lambda: self.s.summary is not None, 120, "import")
        self.wait(lambda: self.s.selection_ok is not None, 30, "selection check")
        self.settle()
        self.pump(400)
        self.settle()

    def finish(self) -> None:
        self.enc.close()


# ------------------------------------------------------------------------------------------------ the film
def film(d: Director, topology: str, trajectory: str) -> None:
    w, s = d.win, d.s
    # 0 title
    d.card = ("title", 1.0)
    d.mark("title")
    for _ in range(int(1.6 * d.fps)):
        d.step()
    d.fade_card("title", False, 0.7)

    # 1 import
    d.mark("import")
    d.show_caption(1, "Drop a trajectory. SOZLab finds the solvent.")
    d.hold(0.7)
    d.load(topology, trajectory)
    d.zoom_scene(1.25)
    bar = w.left_scroll.verticalScrollBar()
    d.pump(150)
    bar.setValue(0)  # the app scrolls to the zone card by itself; show the import summary first
    d.hold(2.0, orbit=0.45)
    target = max(0, w.zone_card.y() - 14)
    n = int(0.8 * d.fps)
    for i in range(1, n + 1):
        bar.setValue(int(lerp(0, target, smooth(i / n))))
        d.orbit(0.45)
        d.step()
    d.hold(0.8, orbit=0.45)
    d.hide_caption()

    # 2 define
    d.mark("define")
    d.show_caption(2, "Pick a part of your molecule, then drag one line.")
    chip = next((c for c in w.zone_card._chips if c.text().startswith("Ion NA")), w.zone_card._chips[0])
    d.click(chip, travel=1.0)
    d.settle()
    d.split(440, 0.9)
    d.hold(0.5, orbit=0.4)
    d.hide_caption(0.2)
    d.show_caption(2, "The zone follows live: in 3D and over time.")
    d.cursor_to(d.line_point(w.session.zone.cutoff), 0.9)
    d.mark("drag")
    start = w.session.zone.cutoff
    d.drag_cutoff(start, 2.0, 1.5)
    d.hold(0.35)
    d.drag_cutoff(2.0, 5.6, 3.2)
    d.hold(0.45)
    d.drag_cutoff(5.6, 3.25, 2.0)
    d.hold(1.0, orbit=0.4)
    d.mark("drag_end")
    d.hide_caption()

    # 3 run
    d.mark("run")
    w.run_card.f_workers.set_value(1, emit=True)  # one process: filming while worker processes start would only slow the run
    d.split(300, 0.8)
    d.show_caption(3, "Run. Every number comes with its uncertainty.")
    d.cursor_on = True
    d.cursor_to(d.centre(w.runbar.run_btn), 1.0)
    button = w.runbar.run_btn
    button.setDown(True)
    for _ in range(4):
        d.step()
    button.setDown(False)
    s.start_run()
    for i in range(12):
        d.ripple = i / 12
        d.step()
    d.ripple = None
    t0 = last = time.time()
    hold = max(1, round(d.fps / 8))
    while s.running and time.time() - t0 < 240:
        if time.time() - last > 0.4:  # film the progress bar now and then; in between leave the CPU to the engine
            d.pump()
            d.frame(hold)
            last = time.time()
        else:
            d.app.processEvents()
            time.sleep(0.02)
    d.wait(lambda: s.result is not None and not s.running, 30, "run")
    d.settle()
    d.pump(300)
    d.cursor_to(d.centre(w.results.t_tau, 0, 40), 0.9)
    d.count_up(0.7)
    d.hold(2.6)
    d.hide_caption()

    # 4 watch the exchange
    d.mark("watch")
    d.split(430, 0.8)
    d.show_caption(4, "Watch solvent enter and leave, frame by frame.")
    d.click(w.stage.play, action=lambda: None, travel=0.9)
    first, last = 60, 60 + 150
    for i in range(first, last):
        w.stage.set_index(i)
        d.settle()
        d.orbit(0.35)
        d.pump(2)
        d.frame()
    d.mark("watch_end")
    d.hold(0.5, orbit=0.3)
    d.hide_caption()

    # 5 trust
    d.mark("trust")
    d.split(330, 0.8)
    d.show_caption(5, "Know what to trust: caveats in plain language.")
    tab = w.results.tabbar
    d.click(tab, action=lambda: tab.setCurrentIndex(3), travel=0.9, dx=tab.tabRect(3).center().x() - tab.rect().center().x())
    d.hold(2.4)
    w.right.show_page("trust")
    d.cursor_to(d.centre(w.trust.list, 0, -40), 0.9)
    d.hold(2.2)
    d.hide_caption()

    # 6 export
    d.mark("export")
    d.show_caption(6, "Export with a manifest. Verify it any time.")
    out = str(d.tmp / "export")
    seg_btn = None
    for b in w.right.seg.findChildren(QtWidgets.QPushButton):
        if b.text().strip().lower() == "export":
            seg_btn = b
    d.click(seg_btn or w.right.seg, action=lambda: w.right.show_page("export"), travel=0.9)
    w.export.folder.setText(out)
    d.hold(0.5)
    d.click(w.export.save_report, travel=0.8)
    d.wait(lambda: w.export.links_host.isVisible() and not s.tasks.running, 120, "export")
    d.pump(300)
    d.hold(0.8)
    d.click(w.export.verify_btn, travel=0.7)
    d.hold(2.2)
    d.hide_caption()

    # 7 end
    d.mark("end")
    d.cursor_on = False
    d.fade_card("end", True, 0.8)
    for _ in range(int(3.4 * d.fps)):
        d.step()
    d.mark("done")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--system", choices=("tip3p", "demo"), default="tip3p")
    ap.add_argument("--data", default=str(ROOT / "bench" / "out" / "tip3p"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--preview", action="store_true", help="low-resolution 10 fps dry run (writes preview.mp4 into the scratch folder)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(x) for x in args.size.lower().split("x"))
    fps = 10 if args.preview else args.fps
    if args.preview:
        w, h = 1366, 768
    target = (Path(tempfile.gettempdir()) / "sozlab_preview.mp4") if args.preview else out / "sozlab-demo.mp4"
    if args.system == "tip3p":
        top, traj = Path(args.data) / "system.pdb", Path(args.data) / "rep0.xtc"
        if not (top.exists() and traj.exists()):
            raise SystemExit(f"{top} not found. Generate it with:\n  python bench/data/make_tip3p_box.py --out {args.data} --replicas 1 --ns 2.0 --dt-out 0.2\n"
                             "or run with --system demo")
        topology, trajectory = str(top), str(traj)
    else:
        from engine.synthetic import make_demo_system

        d0 = Path(tempfile.mkdtemp(prefix="sozlab_video_data_"))
        make_demo_system().write(str(d0), "demo", "xtc")
        topology, trajectory = str(d0 / "demo.gro"), str(d0 / "demo.xtc")
    t0 = time.time()
    d = Director(target, (w, h), fps, args.system, args.preview)
    try:
        film(d, topology, trajectory)
    finally:
        d.finish()
    seconds = d.frame_no / fps
    print(f"{target}: {d.frame_no} frames, {seconds:.1f} s at {fps} fps, rendered in {time.time() - t0:.0f} s")
    print("marks", d.marks)
    if not args.preview:
        json.dump({"fps": fps, "size": [w, h], "seconds": seconds, "marks": d.marks, "system": args.system}, open(out / "timeline.json", "w"), indent=2)
    d.win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
