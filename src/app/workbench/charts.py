"""Themed 2D charts (pyqtgraph) for the Workbench: units always in axis labels, direct and minimal styling."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench.theme import Theme

Qt = QtCore.Qt
pg.setConfigOptions(antialias=True, useOpenGL=False, foreground="#A8B9D1")


def _t() -> Theme:
    return Theme.instance()


def qc(token: str, alpha: float = 1.0) -> QtGui.QColor:
    c = QtGui.QColor(_t().t[token])
    c.setAlphaF(alpha)
    return c


def pen(token: str, width: float = 1.6, alpha: float = 1.0, style=Qt.PenStyle.SolidLine) -> QtGui.QPen:
    p = pg.mkPen(qc(token, alpha), width=width, style=style)
    p.setCosmetic(True)
    return p


def brush(token: str, alpha: float = 0.25) -> QtGui.QBrush:
    return pg.mkBrush(qc(token, alpha))


def style_plot(pw: pg.PlotWidget, x_label: str, y_label: str, grid: bool = True) -> None:
    t = _t()
    pw.setBackground(t.t["bg1"])
    pi = pw.getPlotItem()
    pi.showGrid(x=grid, y=grid, alpha=0.12)
    f = t.font("ui", 8.5)
    for name in ("left", "bottom", "right", "top"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(t.t["line2"]))
        ax.setTextPen(pg.mkPen(t.t["ink2"]))
        ax.setStyle(tickFont=f, tickTextOffset=6)
    pi.setLabel("bottom", x_label, color=t.t["ink2"], **{"font-size": "9pt"})
    pi.setLabel("left", y_label, color=t.t["ink2"], **{"font-size": "9pt"})
    pi.getViewBox().setBorder(None)
    pi.setMenuEnabled(False)
    pi.hideButtons()


class _Plot(pg.PlotWidget):
    """PlotWidget with a modest size hint (the stock one asks for 640x480 and balloons its container)."""

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(300, 170)

    def minimumSizeHint(self):  # noqa: N802
        return QtCore.QSize(120, 100)


class _Chart(QtWidgets.QWidget):
    """Common base: a PlotWidget with a caption row and an overlay message for the empty state."""

    def __init__(self, title: str, x_label: str, y_label: str, parent=None):
        super().__init__(parent)
        self._x_label, self._y_label = x_label, y_label
        self.plot = _Plot()
        self.plot.setMinimumHeight(_t().px(110))
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        from app.workbench.widgets import make_label

        self.caption = make_label(title, "muted", word_wrap=True)
        lay.addWidget(self.caption)
        lay.addWidget(self.plot, 1)
        self.empty = make_label("", "muted", word_wrap=True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setParent(self.plot)
        self.empty.hide()
        self.refresh_theme()

    def show_empty(self, text: str) -> None:
        self.empty.setText(text)
        self.empty.show()
        self.empty.raise_()
        self.empty.setGeometry(self.plot.rect().adjusted(20, 20, -20, -20))

    def hide_empty(self) -> None:
        self.empty.hide()

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if self.empty.isVisible():
            self.empty.setGeometry(self.plot.rect().adjusted(20, 20, -20, -20))

    def refresh_theme(self) -> None:
        style_plot(self.plot, self._x_label, self._y_label)


# ------------------------------------------------------------------------------------------------ timeline
class TimelineChart(_Chart):
    """n_solvent over time with the mean ± SE band, optional comparison series and a draggable frame cursor."""

    cursorMoved = QtCore.pyqtSignal(int)  # sample index

    def __init__(self, parent=None):
        super().__init__("Solvent molecules in the zone", "Time (ns)", "molecules in zone", parent)
        self._t = np.empty(0)
        self._n = np.empty(0)
        self._cmp: Optional[pg.PlotDataItem] = None
        self._items: List = []
        self.cursor = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pen("solute", 1.6),
                                      hoverPen=pen("solute", 2.4))
        self.cursor.setZValue(20)
        self.cursor.sigPositionChanged.connect(self._on_cursor)
        self._block = False
        self.show_empty("Run the analysis to see the occupancy over time.")

    def set_data(self, times_ps: np.ndarray, n: np.ndarray, mean: Optional[float] = None, se: Optional[float] = None) -> None:
        self.plot.clear()
        self._items = []
        self.hide_empty()
        self._t = np.asarray(times_ps, dtype=float) * 1e-3  # ps -> ns
        self._n = np.asarray(n, dtype=float)
        if self._t.size == 0:
            self.show_empty("No frames.")
            return
        curve = pg.PlotDataItem(self._t, self._n, pen=pen("solvent", 1.6), fillLevel=0, fillBrush=brush("solvent", 0.14))
        self.plot.addItem(curve)  # downsampling/clipping only after the item has a view (pyqtgraph 0.13 quirk)
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        if mean is not None and math.isfinite(mean):
            if se is not None and math.isfinite(se) and se > 0:
                band = pg.LinearRegionItem(values=(mean - 1.96 * se, mean + 1.96 * se), orientation="horizontal", movable=False,
                                           brush=brush("accent", 0.22), pen=pg.mkPen(None))
                band.setZValue(5)
                self.plot.addItem(band)
            line = pg.InfiniteLine(pos=mean, angle=0, pen=pen("accent", 1.4, style=Qt.PenStyle.DashLine),
                                   label=f"mean {mean:.3g}", labelOpts={"color": _t().t["accent"], "position": 0.98, "anchors": [(1, 1), (1, 1)]})
            line.setZValue(6)
            self.plot.addItem(line)
        self._cmp = None
        self.plot.addItem(self.cursor)
        self.plot.setLimits(xMin=float(self._t.min()), xMax=float(self._t.max()), yMin=0)
        self.plot.autoRange()
        self.cursor.setBounds((float(self._t.min()), float(self._t.max())))

    def set_compare(self, times_ps: Optional[np.ndarray], n: Optional[np.ndarray]) -> None:
        if self._cmp is not None:
            self.plot.removeItem(self._cmp)
            self._cmp = None
        if times_ps is None or n is None or len(times_ps) == 0:
            return
        self._cmp = pg.PlotDataItem(np.asarray(times_ps) * 1e-3, np.asarray(n), pen=pen("warn", 1.3, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self._cmp)
        self._cmp.setDownsampling(auto=True, method="peak")

    def set_cursor(self, sample: int) -> None:
        if self._t.size == 0:
            return
        sample = int(max(0, min(sample, self._t.size - 1)))
        self._block = True
        self.cursor.setValue(float(self._t[sample]))
        self._block = False

    def _on_cursor(self) -> None:
        if self._block or self._t.size == 0:
            return
        idx = int(np.abs(self._t - self.cursor.value()).argmin())
        self.cursorMoved.emit(idx)


# ------------------------------------------------------------------------------------------------ events
class EventsChart(_Chart):
    def __init__(self, parent=None):
        super().__init__("Entries (up) and exits (down) between consecutive analysed frames", "Time (ns)", "events per bin", parent)
        self.show_empty("Run the analysis to see solvent entering and leaving the zone.")

    def set_data(self, times_ps: np.ndarray, entries: np.ndarray, exits: np.ndarray, max_bars: int = 400) -> None:
        self.plot.clear()
        self.hide_empty()
        t = np.asarray(times_ps, dtype=float) * 1e-3
        e, x = np.asarray(entries, dtype=float), np.asarray(exits, dtype=float)
        if t.size == 0:
            self.show_empty("No frames.")
            return
        if t.size > max_bars:  # bin so thousands of frames stay readable
            edges = np.linspace(0, t.size, max_bars + 1).astype(int)
            e = np.array([e[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
            x = np.array([x[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
            t = np.array([t[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        width = (t[-1] - t[0]) / max(len(t) - 1, 1) * 0.8 if len(t) > 1 else 0.1
        self.plot.addItem(pg.BarGraphItem(x=t, height=e, width=width, brush=brush("solvent", 0.85), pen=pg.mkPen(None)))
        self.plot.addItem(pg.BarGraphItem(x=t, height=-x, width=width, brush=brush("solute", 0.85), pen=pg.mkPen(None)))
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pen("line2", 1.0)))
        self.plot.autoRange()


# ------------------------------------------------------------------------------------------------ histogram
class HistogramChart(_Chart):
    def __init__(self, parent=None):
        super().__init__("How often the zone holds k molecules", "n_solvent (molecules)", "Fraction of frames", parent)
        self.show_empty("Run the analysis to see the occupancy distribution.")

    def set_data(self, n: np.ndarray, mean: Optional[float] = None) -> None:
        self.plot.clear()
        self.hide_empty()
        n = np.asarray(n)
        if n.size == 0:
            self.show_empty("No frames.")
            return
        vals, counts = np.unique(n.astype(int), return_counts=True)
        self.plot.addItem(pg.BarGraphItem(x=vals, height=counts / counts.sum(), width=0.8, brush=brush("solvent", 0.85), pen=pg.mkPen(None)))
        if mean is not None and math.isfinite(mean):
            self.plot.addItem(pg.InfiniteLine(pos=mean, angle=90, pen=pen("accent", 1.5, style=Qt.PenStyle.DashLine),
                                              label=f"mean {mean:.3g}", labelOpts={"color": _t().t["accent"], "position": 0.92}))
        self.plot.autoRange()


# ------------------------------------------------------------------------------------------------ survival
class SurvivalChart(_Chart):
    def __init__(self, parent=None):
        super().__init__("Survival correlation function S(τ): probability a molecule inside the zone stays inside for τ",
                         "Lag τ (ps)", "S(τ)", parent)
        self.show_empty("Run the analysis to see how long solvent stays in the zone.")

    def set_data(self, survival: pd.DataFrame, tau_fit: Optional[float] = None, tau_integral: Optional[float] = None,
                 min_origins: int = 20) -> None:
        self.plot.clear()
        self.hide_empty()
        if survival is None or len(survival) < 2 or not np.isfinite(survival["tau"]).any():
            self.show_empty("The survival function needs at least two frames with trustworthy time stamps.")
            return
        ok = (survival["origins_continuous"] >= min_origins) & np.isfinite(survival["S_continuous"]) & (survival["S_continuous"] > 0)
        tau = survival["tau"].to_numpy()[ok.to_numpy()]
        s = survival["S_continuous"].to_numpy()[ok.to_numpy()]
        if tau.size == 0:
            self.show_empty("Too few (molecule, origin) pairs to estimate S(τ).")
            return
        self.plot.setLogMode(x=False, y=True)
        self.plot.addItem(pg.PlotDataItem(tau, s, pen=pen("solvent", 2.0), name="continuous"))
        ok2 = (survival["origins_intermittent"] >= min_origins) & np.isfinite(survival["S_intermittent"]) & (survival["S_intermittent"] > 0)
        if ok2.any() and not np.allclose(survival["S_intermittent"][ok2], survival["S_continuous"][ok2], equal_nan=True):
            self.plot.addItem(pg.PlotDataItem(survival["tau"].to_numpy()[ok2.to_numpy()], survival["S_intermittent"].to_numpy()[ok2.to_numpy()],
                                              pen=pen("solute", 1.6, style=Qt.PenStyle.DashLine), name="intermittent"))
        if tau_fit and math.isfinite(tau_fit) and tau_fit > 0:
            tt = np.linspace(0, float(tau.max()), 200)
            self.plot.addItem(pg.PlotDataItem(tt, np.exp(-tt / tau_fit), pen=pen("accent", 1.4, style=Qt.PenStyle.DotLine), name="exp fit"))
        if tau_integral and math.isfinite(tau_integral) and tau_integral > 0:
            self.plot.addItem(pg.InfiniteLine(pos=tau_integral, angle=90, pen=pen("warn", 1.2, style=Qt.PenStyle.DashLine),
                                              label=f"∫S = {tau_integral:.3g} ps", labelOpts={"color": _t().t["warn"], "position": 0.15}))
        self.plot.setYRange(math.log10(max(float(s.min()) * 0.8, 1e-3)), math.log10(1.05))
        self.plot.setXRange(0, float(tau.max()))


class StrideChart(_Chart):
    """Residence time versus sampling interval, with the experimental dt → 0 extrapolation."""

    def __init__(self, parent=None):
        super().__init__("Does the answer depend on the frame interval?", "Sampling interval dt (ps)", "τ from ∫S (ps)", parent)
        self.show_empty("Stride sensitivity appears after a run.")

    def set_data(self, rows: Sequence[Dict[str, object]], extrapolation: Optional[Dict[str, object]] = None) -> None:
        self.plot.clear()
        self.hide_empty()
        pts = [(float(r["dt"]), float(r["tau_survival_integral"])) for r in rows
               if r.get("tau_survival_integral") is not None and math.isfinite(float(r.get("tau_survival_integral", float("nan"))))
               and math.isfinite(float(r.get("dt", float("nan"))))]
        if len(pts) < 2:
            self.show_empty("Needs at least 2 sampling intervals with a converged residence time.")
            return
        x, y = np.array(pts).T
        self.plot.addItem(pg.PlotDataItem(x, y, pen=pen("accent", 1.4), symbol="o", symbolSize=9,
                                          symbolBrush=brush("accent", 1.0), symbolPen=pg.mkPen(None)))
        if extrapolation and extrapolation.get("valid"):
            slope = extrapolation["slope_per_sqrt_time"]
            t0 = extrapolation["tau_dt0"]
            xx = np.linspace(0, x.max(), 60)
            self.plot.addItem(pg.PlotDataItem(xx, t0 + slope * np.sqrt(xx), pen=pen("warn", 1.3, style=Qt.PenStyle.DashLine)))
            self.plot.addItem(pg.ScatterPlotItem([0.0], [t0], symbol="d", size=12, brush=brush("warn", 1.0), pen=pg.mkPen(None)))
            lab = pg.TextItem(f"dt→0: {t0:.3g} ps (experimental)", color=_t().t["warn"], anchor=(0, 1))
            lab.setPos(0, t0)
            self.plot.addItem(lab)
        self.plot.setXRange(-0.02 * x.max(), x.max() * 1.1)
        self.plot.autoRange()


# ------------------------------------------------------------------------------------------------ density
class DensityChart(QtWidgets.QWidget):
    """Three orthogonal slices of the density map with a shared colour scale, reliability masking and hover read-out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from app.workbench.widgets import Segmented, make_label

        self._data: Optional[Dict[str, object]] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(make_label("Show", "muted"))
        self.mode = Segmented([("ρ  (Å⁻³)", "rho"), ("g(r)", "g"), ("−ln g  (kT)", "pmf"), ("±SE", "se")], "rho")
        self.mode.changed.connect(lambda *_: self._render())
        bar.addWidget(self.mode)
        self.mask_check = QtWidgets.QCheckBox("Hide unreliable voxels")
        self.mask_check.setChecked(True)
        self.mask_check.setToolTip("Voxels outside the fully covered radius, or with relative standard error above 35%, are hidden")
        self.mask_check.toggled.connect(lambda *_: self._render())
        bar.addWidget(self.mask_check)
        bar.addStretch(1)
        self.readout = make_label("", "mono")
        bar.addWidget(self.readout)
        lay.addLayout(bar)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        self.views = {}
        self.images = {}
        self.sliders = {}
        for col, (key, title, xl, yl) in enumerate((("xy", "XY slice", "x (Å)", "y (Å)"), ("xz", "XZ slice", "x (Å)", "z (Å)"),
                                                    ("yz", "YZ slice", "y (Å)", "z (Å)"))):
            pw = _Plot()
            pw.setAspectLocked(True)
            pw.setMinimumHeight(_t().px(150))
            style_plot(pw, xl, yl, grid=False)
            img = pg.ImageItem()
            pw.addItem(img)
            self.views[key], self.images[key] = pw, img
            grid.addWidget(make_label(title, "caption"), 0, col)
            grid.addWidget(pw, 1, col)
            sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            sl.valueChanged.connect(lambda *_: self._render())
            self.sliders[key] = sl
            grid.addWidget(sl, 2, col)
            pw.scene().sigMouseMoved.connect(lambda pos, k=key: self._hover(k, pos))
        lay.addLayout(grid, 1)
        self.colorbar = pg.GradientLegend((14, 120), (-20, 20)) if hasattr(pg, "GradientLegend") else None
        self.cb_label = make_label("", "caption")
        lay.addWidget(self.cb_label)
        self.empty = make_label("Add a density map in Run settings → Density, then run.", "muted", word_wrap=True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.empty)
        self._set_visible(False)

    def _set_visible(self, has: bool) -> None:
        for pw in self.views.values():
            pw.setVisible(has)
        for s in self.sliders.values():
            s.setVisible(has)
        self.mode.setVisible(has)
        self.mask_check.setVisible(has)
        self.empty.setVisible(not has)
        self.cb_label.setVisible(has)

    def refresh_theme(self) -> None:
        for key, pw in self.views.items():
            style_plot(pw, pw.getPlotItem().getAxis("bottom").labelText, pw.getPlotItem().getAxis("left").labelText, grid=False)
        if self._data is not None:
            self._render()

    def set_result(self, density) -> None:
        if density is None or density.grid is None:
            self._data = None
            self._set_visible(False)
            return
        self._data = {"d": density}
        shape = density.grid.shape
        for key, sl in self.sliders.items():
            n = {"xy": shape[2], "xz": shape[1], "yz": shape[0]}[key]
            sl.blockSignals(True)
            sl.setRange(0, max(n - 1, 0))
            sl.setValue(n // 2)
            sl.blockSignals(False)
        self._set_visible(True)
        self._render()

    def _volume(self):
        d = self._data["d"]
        mode = self.mode.value()
        mask_on = self.mask_check.isChecked() and d.reliable is not None
        grid = np.asarray(d.grid, dtype=float)
        if mode == "g" and d.g is not None:
            vol, unit, cmap, lo, hi = np.asarray(d.g, dtype=float), "g(r) = ρ/ρ_bulk", "RdBu_r", 0.0, 2.0
        elif mode == "pmf" and d.g is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                vol = -np.log(np.where(d.g > 1e-3, d.g, np.nan))
            unit, cmap, lo, hi = "−ln g  (kT, positive = depleted)", "RdBu", -1.5, 1.5
        elif mode == "se" and d.se is not None:
            vol, unit, cmap, lo, hi = np.asarray(d.se, dtype=float), "standard error of ρ (Å⁻³)", "magma", 0.0, None
        else:
            vol, unit, cmap, lo, hi = grid, "ρ (atoms/Å³)", "viridis", 0.0, None
        if mask_on:
            vol = np.where(d.reliable, vol, np.nan)
        finite = vol[np.isfinite(vol)]
        if hi is None:
            hi = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        return vol, unit, cmap, lo, max(hi, lo + 1e-12)

    def _render(self) -> None:
        if self._data is None:
            return
        d = self._data["d"]
        vol, unit, cmap_name, lo, hi = self._volume()
        try:
            cmap = pg.colormap.get(cmap_name, source="matplotlib")
        except Exception:
            cmap = pg.colormap.get("viridis")
        spacing = float(d.metadata.get("grid_spacing", 1.0))
        ax = d.axes
        ix, iy, iz = (self.sliders["yz"].value(), self.sliders["xz"].value(), self.sliders["xy"].value())
        slices = {"xy": vol[:, :, iz], "xz": vol[:, iy, :], "yz": vol[ix, :, :]}
        origins = {"xy": (ax["x"][0], ax["y"][0]), "xz": (ax["x"][0], ax["z"][0]), "yz": (ax["y"][0], ax["z"][0])}
        for key, arr in slices.items():
            img = self.images[key]
            img.setImage(arr, autoLevels=False)  # NaN (hidden/unreliable) voxels render transparent
            img.setLookupTable(cmap.getLookupTable(nPts=256))
            img.setLevels((lo, hi))
            ox, oy = origins[key]
            img.setRect(QtCore.QRectF(ox - spacing / 2, oy - spacing / 2, arr.shape[0] * spacing, arr.shape[1] * spacing))
        self.cb_label.setText(f"{unit}   scale {lo:.3g} → {hi:.3g}   ·   grid {tuple(d.grid.shape)} at {spacing:g} Å")

    def _hover(self, key: str, pos) -> None:
        if self._data is None:
            return
        d = self._data["d"]
        pw, img = self.views[key], self.images[key]
        p = pw.getPlotItem().vb.mapSceneToView(pos)
        spacing = float(d.metadata.get("grid_spacing", 1.0))
        ax = d.axes
        ix, iy, iz = (self.sliders["yz"].value(), self.sliders["xz"].value(), self.sliders["xy"].value())
        try:
            if key == "xy":
                i, j, k = int(round((p.x() - ax["x"][0]) / spacing)), int(round((p.y() - ax["y"][0]) / spacing)), iz
            elif key == "xz":
                i, j, k = int(round((p.x() - ax["x"][0]) / spacing)), iy, int(round((p.y() - ax["z"][0]) / spacing))
            else:
                i, j, k = ix, int(round((p.x() - ax["y"][0]) / spacing)), int(round((p.y() - ax["z"][0]) / spacing))
            if 0 <= i < d.grid.shape[0] and 0 <= j < d.grid.shape[1] and 0 <= k < d.grid.shape[2]:
                rho = float(d.grid[i, j, k])
                se = float(d.se[i, j, k]) if d.se is not None and np.isfinite(d.se[i, j, k]) else float("nan")
                g = float(d.g[i, j, k]) if d.g is not None else float("nan")
                self.readout.setText(f"ρ = {rho:.4f} ± {se:.4f} Å⁻³   g = {g:.2f}")
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ cutoff guide
class CutoffGuide(_Chart):
    """Define step: distribution of the solvent's distance to the anchor, with a draggable cutoff line."""

    cutoffChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__("Where is the solvent? Drag the line to set the cutoff", "Distance from the selection (Å)",
                         "molecules in this frame", parent)
        self._r = np.empty(0)
        self._line = pg.InfiniteLine(pos=3.5, angle=90, movable=True, bounds=(0.5, 30), pen=pen("solute", 2.0),
                                     hoverPen=pen("solute", 3.0))
        self._line.setZValue(30)
        self._line.sigPositionChanged.connect(lambda: self._emit())
        self._blocking = False
        self._suggest: Optional[float] = None
        self.show_empty("Pick a selection to see where the solvent sits around it.")

    def _emit(self) -> None:
        if not self._blocking:
            self.cutoffChanged.emit(float(self._line.value()))

    def set_cutoff(self, r: float) -> None:
        self._blocking = True
        self._line.setValue(float(r))
        self._blocking = False

    def set_distances(self, nearest_distance: np.ndarray, cutoff: float, r_max: float = 12.0) -> Optional[float]:
        self.plot.clear()
        d = np.asarray(nearest_distance, dtype=float)
        d = d[np.isfinite(d) & (d <= r_max)]
        self._r = d
        if d.size < 5:
            self.show_empty("No solvent found near this selection in this frame.")
            return None
        self.hide_empty()
        bins = np.arange(0.0, r_max + 0.25, 0.25)
        counts, edges = np.histogram(d, bins=bins)
        self.plot.addItem(pg.BarGraphItem(x=edges[:-1] + 0.125, height=counts, width=0.22, brush=brush("solvent", 0.8), pen=pg.mkPen(None)))
        cum = np.cumsum(counts)
        right = pg.ViewBox()
        self.plot.showAxis("right")
        self.plot.scene().addItem(right)
        self.plot.getAxis("right").linkToView(right)
        right.setXLink(self.plot)
        self.plot.getAxis("right").setLabel("cumulative molecules", color=_t().t["ink2"])
        self.plot.getAxis("right").setPen(pg.mkPen(_t().t["line2"]))
        self.plot.getAxis("right").setTextPen(pg.mkPen(_t().t["ink2"]))
        right.addItem(pg.PlotDataItem(edges[1:], cum, pen=pen("accent", 1.6)))
        right.setYRange(0, max(cum.max(), 1) * 1.05)

        def upd():
            right.setGeometry(self.plot.getViewBox().sceneBoundingRect())
            right.linkedViewChanged(self.plot.getViewBox(), right.XAxis)

        upd()
        self.plot.getViewBox().sigResized.connect(upd)
        self.plot.addItem(self._line)
        self.set_cutoff(cutoff)
        self.plot.setXRange(0, r_max)
        # suggest: first local minimum after the first peak of the smoothed histogram
        sm = np.convolve(counts.astype(float), np.ones(3) / 3, mode="same")
        peak = int(np.argmax(sm[: len(sm) // 2 + 1]))
        self._suggest = None
        for i in range(peak + 1, len(sm) - 1):
            if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1] and sm[i] < 0.85 * sm[peak]:
                self._suggest = float(edges[i] + 0.125)
                break
        return self._suggest

    def suggestion(self) -> Optional[float]:
        return self._suggest


# ------------------------------------------------------------------------------------------------ tables
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(v, (float, np.floating)):
                return "n/a" if not math.isfinite(v) else f"{v:.4g}"
            return str(v)
        if role == Qt.ItemDataRole.UserRole:
            return v
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) if isinstance(v, (int, float, np.number)) else None
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return None


class _SortProxy(QtCore.QSortFilterProxyModel):
    def lessThan(self, left, right):  # noqa: N802
        a, b = left.data(Qt.ItemDataRole.UserRole), right.data(Qt.ItemDataRole.UserRole)
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class TableChart(QtWidgets.QWidget):
    """Sortable, filterable table with a search box."""

    rowActivated = QtCore.pyqtSignal(dict)

    def __init__(self, placeholder: str = "Filter…", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)
        self.model = PandasModel()
        self.proxy = _SortProxy()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(_t().px(26))
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.view, 1)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.view.clicked.connect(self._clicked)

    def set_frame(self, df: pd.DataFrame) -> None:
        self.model.set_frame(df)
        self.view.resizeColumnsToContents()

    def _clicked(self, index) -> None:
        src = self.proxy.mapToSource(index)
        row = self.model._df.iloc[src.row()].to_dict()
        self.rowActivated.emit(row)
