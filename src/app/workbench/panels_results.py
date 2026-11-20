"""Centre-bottom: headline numbers (value ± error, unit, trust badge) and the charts behind them.

Before a run the panel shows a live preview of the zone being defined (occupancy over ~80 sampled frames), so the
first screen already answers "does this zone hold what I think it holds?"."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench.charts import (DensityChart, EventsChart, HistogramChart, StrideChart, SurvivalChart, TableChart,
                                  TimelineChart)
from app.workbench.session import Session
from app.workbench.theme import Theme, set_prop
from app.workbench.widgets import Chip, FlowLayout, Segmented, StatTile, make_label

Qt = QtCore.Qt


def _t() -> Theme:
    return Theme.instance()


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def trust_level(items: List[dict], prefixes: List[str]) -> Optional[str]:
    """Worst level among the trust items whose id starts with one of ``prefixes`` (run-wide timing items always count)."""
    level = None
    for it in items:
        if any(it["id"].startswith(p) for p in prefixes + ["timing."]):
            if it["level"] == "warning":
                return "warning"
            if it["level"] == "caution":
                level = "caution"
    return level


class TileGrid(QtWidgets.QWidget):
    """Tiles that wrap to the available width (2-6 per row)."""

    def __init__(self, tiles: List[StatTile], parent=None):
        super().__init__(parent)
        self.tiles = tiles
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        self._cols = 0
        self._place(3)

    def _place(self, cols: int) -> None:
        if cols == self._cols:
            return
        self._cols = cols
        for t in self.tiles:
            self._grid.removeWidget(t)
        for i, t in enumerate(self.tiles):
            self._grid.addWidget(t, i // cols, i % cols)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        for c in range(cols, 8):
            self._grid.setColumnStretch(c, 0)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        w = max(self.width(), 1)
        self._place(int(max(2, min(len(self.tiles), w // _t().px(116)))))


class ResultsPanel(QtWidgets.QWidget):
    cursorMoved = QtCore.pyqtSignal(int)        # sample index chosen on the timeline
    moleculeActivated = QtCore.pyqtSignal(dict)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._times = np.empty(0)
        self._result = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        # settings strip: what produced these numbers
        self.strip_host = QtWidgets.QWidget()
        self.strip = FlowLayout(self.strip_host, 0, 6, 6)
        lay.addWidget(self.strip_host)
        # tiles
        self.t_mean = StatTile("In zone", "molecules", "n_solvent", decimals=4)
        self.t_occ = StatTile("Occupied", "% of frames", "occupancy_fraction", decimals=4)
        self.t_tau = StatTile("Residence τ", "ps", "tau_survival_integral", decimals=3)
        self.t_rate = StatTile("Entry rate", "per ps", "entry_rate_per_time", decimals=3)
        self.t_visits = StatTile("Visits", "events", "visit", decimals=5)
        self.t_frames = StatTile("Frames analysed", "", None, decimals=6)
        self.t_frames.hide()  # shown in the settings strip instead
        self.tiles = TileGrid([self.t_mean, self.t_occ, self.t_tau, self.t_rate, self.t_visits])
        lay.addWidget(self.tiles)
        # preview banner
        self.banner = make_label("", "caption", word_wrap=True)
        lay.addWidget(self.banner)
        # tabs
        self.tabbar = QtWidgets.QTabBar()
        self.tabbar.setDrawBase(False)
        self.tabbar.setExpanding(False)
        self.tabbar.setUsesScrollButtons(True)
        self.tabbar.setElideMode(Qt.TextElideMode.ElideNone)
        self.stack = QtWidgets.QStackedWidget()
        lay.addWidget(self.tabbar)
        lay.addWidget(self.stack, 1)
        # occupancy
        self.timeline = TimelineChart()
        self.hist = HistogramChart()
        occ = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        occ.addWidget(self.timeline)
        occ.addWidget(self.hist)
        occ.setStretchFactor(0, 3)
        occ.setStretchFactor(1, 2)
        occ.setChildrenCollapsible(False)
        self._add_tab("Occupancy", occ)
        # residence
        self.survival = SurvivalChart()
        res = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        res.addWidget(self.survival)
        self.res_info = self._info_panel()
        res.addWidget(self.res_info[0])
        res.setStretchFactor(0, 3)
        res.setStretchFactor(1, 2)
        res.setChildrenCollapsible(False)
        self._add_tab("Residence", res)
        # events
        self.events = EventsChart()
        self._add_tab("Entries / exits", self.events)
        # sampling
        self.stride = StrideChart()
        samp = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(samp)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.stride, 1)
        self.samp_text = make_label("", "muted", word_wrap=True)
        sl.addWidget(self.samp_text)
        self._add_tab("Sampling", samp)
        # density
        self.density = DensityChart()
        self._add_tab("Density", self.density)
        # tables
        tab = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tab)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        self.table_mode = Segmented([("Per molecule", "solvent"), ("Visits", "visits"), ("Per frame", "frames")], "solvent")
        self.table_mode.changed.connect(lambda *_: self._fill_table())
        tl.addWidget(self.table_mode, 0, Qt.AlignmentFlag.AlignLeft)
        self.table = TableChart("Filter rows…")
        self.table.rowActivated.connect(self.moleculeActivated)
        tl.addWidget(self.table, 1)
        self._add_tab("Tables", tab)
        self.tabbar.currentChanged.connect(self.stack.setCurrentIndex)
        self.timeline.cursorMoved.connect(self.cursorMoved)
        self.clear()

    def _add_tab(self, title: str, widget: QtWidgets.QWidget) -> None:
        self.tabbar.addTab(title)
        self.stack.addWidget(widget)

    def _info_panel(self):
        host = QtWidgets.QFrame()
        host.setObjectName("Plate")
        form = QtWidgets.QFormLayout(host)
        form.setContentsMargins(14, 12, 14, 12)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        labels: Dict[str, QtWidgets.QLabel] = {}
        for key, title in (("visits", "Visits (complete / cut by window)"), ("censored", "Cut by window edges"),
                           ("mean", "Mean visit duration (complete)"), ("km", "Kaplan–Meier restricted mean"),
                           ("median", "Median visit (Kaplan–Meier)"), ("integral", "τ from ∫S(τ)dτ"),
                           ("fit", "τ from exp(−τ/τ₀) fit"), ("gap", "Brief-exit tolerance"),
                           ("dt0", "τ at dt → 0 (experimental)")):
            lab = make_label("–", "mono")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            labels[key] = lab
            form.addRow(make_label(title, "caption"), lab)
        return host, labels

    # ------------------------------------------------------------------------------------ states
    def clear(self) -> None:
        self._result = None
        for t in self.tiles.tiles:
            t.clear()
        self.timeline.set_compare(None, None)
        self.banner.setText("Numbers appear here after you run the analysis. Until then the chart shows a live preview of the zone you are defining.")
        self.banner.show()
        self._fill_strip(None)

    def set_preview(self, out: dict) -> None:
        """Occupancy of the zone being defined over a sample of frames (before any run)."""
        if self._result is not None or not out.get("ok"):
            return
        t, n = out["times"], out["n"]
        self._times = np.asarray(t)
        mean = float(np.mean(n)) if len(n) else float("nan")
        self.timeline.caption.setText(f"PREVIEW · solvent molecules in the zone at {len(n)} sampled frames (Run for exact, uncertainty-aware numbers)")
        self.timeline.set_data(t, n, None, None)
        self.hist.set_data(n, mean)
        self.t_mean.set_value(mean, None, note="preview of sampled frames", trust="preview")

    def set_result(self, result) -> None:
        from engine.analysis import AnalysisResult

        self._result = result
        name = next(iter(result.soz_results), None)
        if name is None:
            return
        z = result.soz_results[name]
        s = z.summary
        unc = s.get("uncertainty", {})
        res = s.get("residence", {}).get("continuous", {})
        trust = result.qc_summary.get("trust", [])
        unit = s.get("time_unit", "ps")
        timing_ok = bool(result.qc_summary.get("timing", {}).get("trusted", True))
        self.banner.hide()
        mean_u = unc.get("mean_n_solvent") or {}
        self.t_mean.set_value(s["mean_n_solvent"], mean_u.get("se"), "molecules", trust_level(trust, ["occupancy.few", "occupancy.stride"]),
                              note=f"median {s['median_n_solvent']:g} · max {s['max_n_solvent']}")
        occ_u = unc.get("occupancy_fraction") or {}
        self.t_occ.set_value(100.0 * s["occupancy_fraction"], 100.0 * occ_u["se"] if _fin(occ_u.get("se")) else None, "% of frames",
                             trust_level(trust, ["occupancy.zero"]))
        tau = res.get("tau_survival_integral", float("nan"))
        tau_u = unc.get("tau_survival_integral") or {}
        note = []
        if _fin(res.get("tau_survival_fit")):
            note.append(f"fit {res['tau_survival_fit']:.3g}")
        if _fin(res.get("km_restricted_mean")):
            note.append(f"KM {res['km_restricted_mean']:.3g}")
        self.t_tau.set_value(tau if timing_ok else float("nan"), tau_u.get("se"), unit, trust_level(trust, ["residence."]),
                             note=" · ".join(note) if timing_ok else "needs trusted time stamps")
        rate = s.get("entry_rate_per_time", float("nan"))
        rate_u = unc.get("entry_rate_per_time") or {}
        self.t_rate.set_value(rate if timing_ok else float("nan"), rate_u.get("se"), f"per {unit}", trust_level(trust, ["entries."]),
                              note="depends on Δt")
        nv, nc = int(res.get("n_visits", 0)), int(res.get("n_visits_complete", 0))
        self.t_visits.set_value(float(nv), None, "events", trust_level(trust, ["residence.few", "residence.censored"]),
                                note=f"{nc} complete · {100 * res.get('fraction_censored', 0):.0f}% cut")
        dt = s.get("dt", float("nan"))
        self.t_frames.set_value(float(s["n_frames"]), None, "", None,
                                note=f"every {dt:g} {unit}" if _fin(dt) and timing_ok else "")
        # timeline / hist
        pf = z.per_frame
        times = pf["time"].to_numpy(float)
        self._times = times
        self.timeline.caption.setText("Solvent molecules in the zone (line) · mean (dashed) · 95% interval of the mean (band)")
        self.timeline.set_data(times, pf["n_solvent"].to_numpy(float), s["mean_n_solvent"], mean_u.get("se"))
        self.hist.set_data(pf["n_solvent"].to_numpy(), s["mean_n_solvent"])
        self.events.set_data(times, pf["entries"].to_numpy(float), pf["exits"].to_numpy(float))
        self.survival.set_data(z.survival, res.get("tau_survival_fit"), tau if _fin(tau) else None)
        self._fill_residence(res, s)
        sens = z.stride_sensitivity or s.get("stride_sensitivity")
        self.stride.set_data(sens or [], s.get("tau_extrapolation"))
        self.samp_text.setText(self._sampling_text(sens, s))
        dens = next(iter(result.density_results.values()), None)
        self.density.set_result(dens)
        self._fill_table()
        self._fill_strip(result)

    # ------------------------------------------------------------------------------------ pieces
    def _fill_residence(self, res: dict, s: dict) -> None:
        lab = self.res_info[1]
        unit = s.get("time_unit", "ps")

        def f(v):
            return f"{v:.4g} {unit}" if _fin(v) else "n/a"

        lab["visits"].setText(f"{res.get('n_visits', 0)}  /  {res.get('n_visits', 0) - res.get('n_visits_complete', 0)}")
        lab["censored"].setText(f"{100 * res.get('fraction_censored', 0):.1f}%  (left {res.get('n_left_censored', 0)}, right {res.get('n_right_censored', 0)})")
        lab["mean"].setText(f(res.get("mean_visit_duration_complete")))
        lab["km"].setText(f(res.get("km_restricted_mean")))
        lab["median"].setText(f(res.get("km_median")))
        conv = "" if res.get("tau_survival_integral_converged", True) else "   (not decayed: lower bound)"
        lab["integral"].setText(f(res.get("tau_survival_integral")) + conv)
        lab["fit"].setText(f(res.get("tau_survival_fit")) + (f"   R² = {res['tau_survival_fit_r2']:.3f}" if _fin(res.get("tau_survival_fit_r2")) else ""))
        gap = s.get("gap_tolerance_time", 0.0)
        lab["gap"].setText(f"{gap:g} {unit}" if _fin(gap) and gap > 0 else "none (strict)")
        ex = s.get("tau_extrapolation") or {}
        lab["dt0"].setText(f(ex.get("tau_dt0")) if ex.get("valid") else "not available")

    @staticmethod
    def _sampling_text(sens, s: dict) -> str:
        if not sens or len(sens) < 2:
            return "Stride sensitivity was not computed (enable it in Run settings → Advanced)."
        bits = []
        for row in sens[1:]:
            ch = row.get("tau_survival_integral_rel_change")
            ch2 = row.get("entry_rate_per_time_rel_change")
            bits.append(f"{row['stride_multiplier']}× coarser: τ {ch:+.0%}" if _fin(ch) else f"{row['stride_multiplier']}× coarser: τ n/a")
            if _fin(ch2):
                bits[-1] += f", entry rate {ch2:+.0%}"
        verdict = "Changes above ~10% mean the number depends on the frame interval; quote it together with that interval."
        return "  ·  ".join(bits) + "\n" + verdict

    def _fill_strip(self, result) -> None:
        while self.strip.count():
            it = self.strip.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if result is None:
            self.strip_host.hide()
            return
        proj = self.session.run_project
        a = proj.analysis if proj else None
        s = next(iter(result.soz_results.values())).summary
        unit = s.get("time_unit", "ps")
        chips = [("zone", self.session.zone.describe())]
        bits = [f"{s['n_frames']} frames"]
        if _fin(s.get("dt")):
            bits.append(f"Δt {s['dt']:g} {unit}")
        if a is not None:
            if a.stride > 1:
                bits.append(f"stride {a.stride}")
            bits.append(f"{a.uncertainty_blocks} blocks")
            if a.gap_tolerance_time:
                bits.append(f"brief-exit tolerance {a.gap_tolerance_time:g} {unit}")
        chips.append(("frames", " · ".join(bits)))
        prov = result.provenance or {}
        chips.append(("ver", f"config {str(prov.get('config_sha256', ''))[:8]}"))
        timing = result.qc_summary.get("timing", {})
        if timing.get("source") != "trajectory":
            chips.append(("time", "time stamps assumed (1 ps/frame)" if timing.get("source") == "assumed" else f"time: {timing.get('source')}"))
        for key, text in chips:
            c = Chip(text, "warn" if key == "time" else "neutral")
            c.setToolTip({"zone": "The zone definition behind every number on this page",
                          "ver": "SOZLab version and the hash of the exact configuration (full values in the report and manifest)",
                          "frames": "Frames analysed, sampling interval, stride and number of blocks used for the standard errors"}.get(key, ""))
            self.strip.addWidget(c)
        self.strip_host.show()

    def _fill_table(self) -> None:
        if self._result is None:
            return
        z = next(iter(self._result.soz_results.values()))
        mode = self.table_mode.value()
        if mode == "visits" and z.visits is not None:
            df = z.visits.copy()
        elif mode == "frames":
            df = z.per_frame.drop(columns=[c for c in ("solvent_ids", "frame_fp") if c in z.per_frame.columns]).copy()
        else:
            cols = ["solvent_id", "resname", "resid", "occupancy_pct", "frames_present", "n_visits", "n_visits_complete",
                    "mean_res_time_cont_complete", "max_res_time_cont", "time_in_zone", "entries"]
            df = z.per_solvent[[c for c in cols if c in z.per_solvent.columns]].copy()
        self.table.set_frame(df)

    def set_cursor(self, sample: int) -> None:
        self.timeline.set_cursor(sample)

    def refresh_theme(self) -> None:
        for w in (self.timeline, self.hist, self.survival, self.events, self.stride, self.density):
            w.refresh_theme()
        if self._result is not None:
            self.set_result(self._result)
