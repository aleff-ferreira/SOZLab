"""Session: the single source of truth for the Workbench (inputs, solvent, zone, run settings, results, history).

The window only renders this state and forwards user intent; tests drive the very same methods, so GUI behaviour
is checkable without pixels."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from PyQt6 import QtCore

from app.workbench import model
from app.workbench.model import RunSpec, SolventSpec, ZoneSpec
from app.workbench.services import PreviewService, TaskRunner
from engine.analysis import AnalysisResult, SOZAnalysisEngine
from engine.models import InputConfig, ProjectConfig

STEPS = [("import", "Import"), ("define", "Define"), ("run", "Run"), ("inspect", "Inspect"), ("export", "Export")]


@dataclass
class Issue:
    level: str  # "error" | "warn" | "info"
    code: str
    message: str
    fix_label: Optional[str] = None
    fix: Optional[Callable[[], None]] = None


@dataclass
class Snapshot:
    id: int
    label: str
    created: str
    settings: Dict[str, object]
    key: Dict[str, float]
    times_ps: np.ndarray
    n_solvent: np.ndarray
    survival: Optional[pd.DataFrame] = None
    run_id: str = ""
    pinned: bool = False


def _key_results(result: AnalysisResult) -> Dict[str, float]:
    name = next(iter(result.soz_results))
    s = result.soz_results[name].summary
    u = s.get("uncertainty", {})
    res = s.get("residence", {}).get("continuous", {})
    return {
        "mean_n": float(s["mean_n_solvent"]), "mean_n_se": float((u.get("mean_n_solvent") or {}).get("se", float("nan"))),
        "occupancy": float(s["occupancy_fraction"]), "tau": float(res.get("tau_survival_integral", float("nan"))),
        "tau_se": float((u.get("tau_survival_integral") or {}).get("se", float("nan"))),
        "entry_rate": float(s.get("entry_rate_per_time", float("nan"))), "n_frames": float(s["n_frames"]),
        "visits": float(res.get("n_visits", 0)), "censored": float(res.get("fraction_censored", float("nan"))),
    }


class Session(QtCore.QObject):
    inputsChanged = QtCore.pyqtSignal()          # topology / trajectory path changed (loading starts)
    summaryReady = QtCore.pyqtSignal(object)     # input summary dict
    summaryFailed = QtCore.pyqtSignal(str)
    zoneChanged = QtCore.pyqtSignal()
    settingsChanged = QtCore.pyqtSignal()
    issuesChanged = QtCore.pyqtSignal(object)
    sceneReady = QtCore.pyqtSignal(object)
    curveReady = QtCore.pyqtSignal(object)
    selectionInfo = QtCore.pyqtSignal(object)
    seriesReady = QtCore.pyqtSignal(object)
    runStarted = QtCore.pyqtSignal()
    runProgress = QtCore.pyqtSignal(int, int, str)
    resultReady = QtCore.pyqtSignal(object)
    runFailed = QtCore.pyqtSignal(str)
    runCancelled = QtCore.pyqtSignal()
    taskProgress = QtCore.pyqtSignal(str, int, int, str)
    taskDone = QtCore.pyqtSignal(str, object)
    taskFailed = QtCore.pyqtSignal(str, str)
    snapshotsChanged = QtCore.pyqtSignal()
    stepsChanged = QtCore.pyqtSignal()
    message = QtCore.pyqtSignal(str, str)        # text, kind

    def __init__(self, parent=None):
        super().__init__(parent)
        self.topology: Optional[str] = None
        self.trajectory: Optional[str] = None
        self.summary: Optional[Dict[str, object]] = None
        self.solvent = SolventSpec()
        self.zone = ZoneSpec()
        self.run = RunSpec()
        self.result: Optional[AnalysisResult] = None
        self.run_project: Optional[ProjectConfig] = None
        self.snapshots: List[Snapshot] = []
        self.exported = False
        self.running = False
        self.loading = False
        self.selection_ok: Optional[bool] = None
        self.selection_message = ""
        self._snap_counter = 0
        self._run_signature = ""
        self.preview_frame = 0
        self.preview = PreviewService(self)
        self.preview.loaded.connect(self._on_loaded)
        self.preview.loadFailed.connect(self._on_load_failed)
        self.preview.sceneReady.connect(self.sceneReady)
        self.preview.curveReady.connect(self.curveReady)
        self.preview.selectionInfo.connect(self._on_selection_info)
        self.preview.seriesReady.connect(self.seriesReady)
        self.runner = TaskRunner(self)
        self.runner.progress.connect(self.runProgress)
        self.runner.finished.connect(self._on_run_finished)
        self.runner.failed.connect(self._on_run_failed)
        self.tasks = TaskRunner(self)
        self._task_name = ""
        self.tasks.progress.connect(lambda c, t, m: self.taskProgress.emit(self._task_name, c, t, m))
        self.tasks.finished.connect(lambda r: self.taskDone.emit(self._task_name, r))
        self.tasks.failed.connect(lambda m, tr: self.taskFailed.emit(self._task_name, m))

    # ------------------------------------------------------------------------------------ steps
    def step_state(self) -> Dict[str, str]:
        has_in = self.summary is not None
        zone_ok = has_in and self.zone.is_defined and self.selection_ok is not False and not any(
            i.level == "error" for i in self.issues())
        ran = self.result is not None
        st = {k: "todo" for k, _ in STEPS}
        st["import"] = "done" if has_in else "active"
        st["define"] = "done" if (zone_ok and (ran or self.running)) else ("active" if has_in and not zone_ok else ("done" if zone_ok else "todo"))
        st["run"] = "done" if ran else ("active" if zone_ok else "blocked")
        st["inspect"] = "done" if ran else "blocked"
        st["export"] = "done" if self.exported else ("active" if ran else "blocked")
        if ran and not self.exported:
            st["inspect"] = "done"
        return st

    # ------------------------------------------------------------------------------------ inputs
    def open_paths(self, paths: List[str]) -> bool:
        topo, traj, notes = model.classify_paths(paths)
        if topo is None:
            self.message.emit("I could not tell which file is the topology. Drop a .gro/.pdb/.psf/.tpr file (and optionally a trajectory).", "warn")
            return False
        for n in notes:
            self.message.emit(n, "info")
        self.open_inputs(topo, traj)
        return True

    def open_inputs(self, topology: str, trajectory: Optional[str]) -> None:
        self.topology, self.trajectory = topology, trajectory
        self.summary, self.result, self.exported = None, None, False
        self.loading = True
        self.run.output_dir = model.default_output_dir(topology)
        self.inputsChanged.emit()
        self.stepsChanged.emit()
        self.preview.load(topology, trajectory)

    def use_demo(self, directory: str) -> None:
        """Generate the synthetic demo system (in a worker thread) and open it."""
        def job(progress, cancelled):
            from engine.synthetic import make_demo_system

            os.makedirs(directory, exist_ok=True)
            gro = os.path.join(directory, "demo.gro")
            xtc = os.path.join(directory, "demo.xtc")
            if not (os.path.isfile(gro) and os.path.isfile(xtc)):
                progress(0, 2, "Generating a synthetic hydrated solute…")
                make_demo_system().write(directory, "demo", "xtc")
            return gro, xtc

        self._task_name = "demo"
        self.tasks.finished.disconnect()
        self.tasks.finished.connect(self._on_demo_ready)
        self.tasks.start(job)

    def _on_demo_ready(self, paths) -> None:
        self.tasks.finished.disconnect()
        self.tasks.finished.connect(lambda r: self.taskDone.emit(self._task_name, r))
        self.open_inputs(paths[0], paths[1])
        self.message.emit("Demo loaded: a synthetic solute in water (not a real simulation).", "info")

    def _on_loaded(self, summary: dict) -> None:
        self.loading = False
        self.summary = summary
        sv = self.solvent
        if sv.auto:
            if summary["water_resnames"]:
                sv.resnames = list(summary["water_resnames"])
            sv.probe_selection = summary["probe_selection"]
        # a sensible default anchor, so the first screen after import already shows a zone
        if not self.zone.anchor.strip() and summary["components"]:
            ligands = [c for c in summary["components"] if c["kind"] == "ligand"]
            pick = ligands[0] if ligands else summary["components"][0]
            self.zone.anchor = pick["selection"]
            self.zone.name = "Zone"
        n = int(summary["n_frames"])
        self.run.frame_stop = None
        self.run.stride = 1 if n <= 3000 else max(1, n // 2000)
        self.summaryReady.emit(summary)
        self.zoneChanged.emit()
        self.stepsChanged.emit()
        self._emit_issues()
        self.refresh_selection()

    def _on_load_failed(self, msg: str) -> None:
        self.loading = False
        self.summary = None
        self.summaryFailed.emit(msg)
        self.stepsChanged.emit()

    # ------------------------------------------------------------------------------------ zone
    def set_zone(self, **kw) -> None:
        changed = False
        for k, v in kw.items():
            if getattr(self.zone, k) != v:
                setattr(self.zone, k, v)
                changed = True
        if changed:
            self.result_stale()
            self.zoneChanged.emit()
            if "anchor" in kw:
                self.refresh_selection()
            self._emit_issues()
            self.stepsChanged.emit()

    def set_solvent(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.solvent, k, v)
        self.solvent.auto = False if kw else self.solvent.auto
        self.result_stale()
        self.settingsChanged.emit()
        self._emit_issues()

    def set_run(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.run, k, v)
        self.settingsChanged.emit()
        self._emit_issues()

    def result_stale(self) -> None:
        pass  # results are kept (and shown) until the next run; the Run card marks settings as changed

    def refresh_selection(self) -> None:
        if self.summary is None:
            return
        self.preview.ask_selection(self.zone.anchor)

    def _on_selection_info(self, info: dict) -> None:
        if info["selection"].strip() != self.zone.anchor.strip():
            return
        self.selection_ok = bool(info["ok"])
        self.selection_message = info["message"]
        self.selectionInfo.emit(info)
        self._emit_issues()
        self.stepsChanged.emit()

    # ------------------------------------------------------------------------------------ validation
    def effective_frames(self) -> int:
        if self.summary is None:
            return 0
        n = int(self.summary["n_frames"])
        stop = n if self.run.frame_stop is None else min(self.run.frame_stop, n)
        return max(0, len(range(max(self.run.frame_start, 0), stop, max(1, self.run.stride))))

    def effective_dt(self) -> Optional[float]:
        if self.summary is None:
            return None
        dt = self.summary["timing"].get("dt_median")
        if dt is None:
            return None
        return float(dt) * max(1, self.run.stride)

    def issues(self) -> List[Issue]:
        out: List[Issue] = []
        s = self.summary
        if s is None:
            return out
        for note in s.get("notes", []):
            out.append(Issue("warn", "input.note", note))
        if not self.zone.anchor.strip():
            out.append(Issue("error", "zone.anchor", "Choose what the zone surrounds: pick a part above or type a selection."))
        elif self.selection_ok is False:
            out.append(Issue("error", "zone.selection", self.selection_message or "The selection matches no atoms."))
        if self.zone.cutoff <= 0:
            out.append(Issue("error", "zone.cutoff", "The cutoff must be positive."))
        lim = s.get("min_image_limit")
        if lim and self.zone.cutoff > lim:
            out.append(Issue("warn", "zone.halfbox", f"The cutoff {self.zone.cutoff:g} Å exceeds half the box ({lim:.1f} Å): the zone would overlap its own periodic image."))
        if s["water_resnames"] and not set(self.solvent.resnames) & set(s["water_resnames"]) and not self.solvent.include_ions:
            found = ", ".join(s["water_resnames"])
            out.append(Issue("error", "solvent.names", f"None of the solvent names match this system (found: {found}).", f"Use {found}",
                             lambda: self.set_solvent(resnames=list(s["water_resnames"]), probe_selection=s["probe_selection"])))
        if not s["water_resnames"] and not self.solvent.resnames:
            out.append(Issue("error", "solvent.none", "No solvent residues are defined."))
        n = self.effective_frames()
        if n == 0:
            out.append(Issue("error", "run.window", "The frame window selects no frames."))
        elif n == 1:
            out.append(Issue("warn", "run.oneframe", "One frame only: residence times and rates need at least two frames."))
        if s.get("time_assumed"):
            out.append(Issue("warn", "time.assumed", "This format has no time stamps: durations assume 1 ps per frame until you set the frame interval (Run settings → Advanced)."))
        if self.run.density and not (self.run.density_align or self.zone.anchor.strip()):
            out.append(Issue("error", "density.align", "A density map needs a selection to superpose on."))
        return out

    def _emit_issues(self) -> None:
        self.issuesChanged.emit(self.issues())

    def can_run(self) -> bool:
        return (self.summary is not None and not self.running and not self.loading and self.zone.is_defined
                and not any(i.level == "error" for i in self.issues()))

    # ------------------------------------------------------------------------------------ project
    def signature(self) -> str:
        """Everything that changes the numbers (not where they are written)."""
        return json.dumps({"z": self.zone.__dict__, "s": self.solvent.__dict__,
                           "r": {k: v for k, v in self.run.__dict__.items() if k != "output_dir"}}, sort_keys=True, default=str)

    def is_stale(self) -> bool:
        """True when results are on screen but the settings were changed after they were produced."""
        return self.result is not None and self.signature() != self._run_signature

    def window_frames(self) -> List[int]:
        if self.summary is None:
            return []
        n = int(self.summary["n_frames"])
        stop = n if self.run.frame_stop is None else min(self.run.frame_stop, n)
        return list(range(max(self.run.frame_start, 0), stop, max(1, self.run.stride)))

    def set_preview_frame(self, frame: int, prev: Optional[int] = None) -> None:
        self.preview_frame = int(frame)
        self.ask_scene(self.preview_frame, prev)

    def build_project(self) -> ProjectConfig:
        assert self.topology is not None
        inputs = InputConfig(topology=self.topology, trajectory=self.trajectory)
        return model.build_project(inputs, self.solvent, self.zone, self.run)

    def scene_request(self, frame: int, prev_frame: Optional[int] = None) -> dict:
        sels, soz = self.zone.compile() if self.zone.is_defined else ({}, None)
        zone = None
        if soz is not None:
            zone = {"selections": {k: v.to_dict() for k, v in sels.items()}, "soz": soz.to_dict()}
        return {"frame": frame, "prev_frame": prev_frame, "solvent": self.solvent.to_config().to_dict(), "zone": zone}

    def ask_scene(self, frame: int, prev_frame: Optional[int] = None) -> None:
        if self.summary is not None:
            self.preview.ask_scene(self.scene_request(frame, prev_frame))

    def ask_series(self, max_frames: int = 80) -> None:
        """Occupancy of the current zone over ~max_frames evenly spaced frames of the analysis window (live preview)."""
        if self.summary is None or not self.zone.is_defined or self.selection_ok is False:
            return
        n = int(self.summary["n_frames"])
        stop = n if self.run.frame_stop is None else min(self.run.frame_stop, n)
        window = list(range(max(self.run.frame_start, 0), stop, max(1, self.run.stride)))
        if not window:
            return
        if len(window) > max_frames:
            window = [window[int(round(i))] for i in np.linspace(0, len(window) - 1, max_frames)]
        sels, soz = self.zone.compile()
        zone = {"selections": {k: v.to_dict() for k, v in sels.items()}, "soz": soz.to_dict()}
        self.preview.ask_series({"frames": window, "solvent": self.solvent.to_config().to_dict(), "zone": zone})

    def ask_curve(self, frame: int) -> None:
        if self.summary is not None and self.zone.anchor.strip() and self.selection_ok is not False:
            self.preview.ask_curve({"frame": frame, "anchor": self.zone.anchor, "solvent": self.solvent.to_config().to_dict()})

    # ------------------------------------------------------------------------------------ run
    def start_run(self) -> None:
        if not self.can_run():
            return
        project = self.build_project()
        self.run_project = project
        self._run_signature = self.signature()
        self.running = True
        self.exported = False
        self.runStarted.emit()
        self.stepsChanged.emit()
        cmd = "SOZLab Workbench"

        def job(progress, cancelled):
            from engine.logging_utils import setup_run_logger

            from engine.analysis import RunCancelledError

            logger, _ = setup_run_logger(project.outputs.output_dir)
            try:
                return SOZAnalysisEngine(project).run(progress=progress, cancel_flag=cancelled, logger=logger, command_line=cmd)
            except RunCancelledError:
                return None  # cancelled before the first frame: nothing to show, and not an error

        self.runner.start(job)

    def cancel_run(self) -> None:
        self.runner.cancel()

    def _on_run_finished(self, result: Optional[AnalysisResult]) -> None:
        self.running = False
        if result is None:
            self.message.emit("Run cancelled before any frame was analysed; nothing was kept.", "info")
            self.runCancelled.emit()
            self.stepsChanged.emit()
            return
        self.result = result
        if result.complete and result.soz_results:
            self._add_snapshot(result)
        self.resultReady.emit(result)
        self.stepsChanged.emit()

    def _on_run_failed(self, msg: str, trace: str) -> None:
        self.running = False
        self.runFailed.emit(msg)
        self.stepsChanged.emit()

    # ------------------------------------------------------------------------------------ history / compare
    def _add_snapshot(self, result: AnalysisResult) -> None:
        name = next(iter(result.soz_results))
        z = result.soz_results[name]
        self._snap_counter += 1
        a = self.run_project.analysis if self.run_project else None
        settings = {"zone": self.zone.describe(), "cutoff_A": self.zone.cutoff, "stride": a.stride if a else None,
                    "frames": int(result.qc_summary.get("frames_analysed", 0)),
                    "gap_time_ps": a.gap_tolerance_time if a else None, "blocks": a.uncertainty_blocks if a else None,
                    "density": bool(result.density_results)}
        snap = Snapshot(
            id=self._snap_counter, label=f"Run {self._snap_counter}", created=time.strftime("%H:%M:%S"), settings=settings,
            key=_key_results(result), times_ps=z.per_frame["time"].to_numpy(float) if not z.per_frame.empty else np.empty(0),
            n_solvent=z.per_frame["n_solvent"].to_numpy(float) if not z.per_frame.empty else np.empty(0),
            survival=z.survival, run_id=str(result.provenance.get("run_id", "")),
        )
        self.snapshots.append(snap)
        if len(self.snapshots) > 12:
            self.snapshots = [s for s in self.snapshots if s.pinned][-6:] + self.snapshots[-6:]
        self.snapshotsChanged.emit()

    def pin(self, snap_id: int, pinned: bool = True) -> None:
        for s in self.snapshots:
            if s.id == snap_id:
                s.pinned = pinned
        self.snapshotsChanged.emit()

    # ------------------------------------------------------------------------------------ export
    def export_data(self, directory: str) -> None:
        assert self.result is not None and self.run_project is not None
        project = self.run_project
        result = self.result
        out_dir = directory

        def job(progress, cancelled):
            import copy

            from engine.export import export_results

            p = copy.deepcopy(project)
            p.outputs.output_dir = out_dir
            progress(0, 1, "Writing data…")
            return export_results(result, p, allow_partial=not result.complete)

        self._start_task("export_data", job)

    def export_report(self, directory: str) -> None:
        assert self.result is not None and self.run_project is not None
        project, result, out_dir = self.run_project, self.result, directory

        def job(progress, cancelled):
            import copy

            from engine.export import export_results
            from engine.report import generate_report

            p = copy.deepcopy(project)
            p.outputs.output_dir = out_dir
            progress(0, 2, "Writing data…")
            export_results(result, p, allow_partial=not result.complete)
            progress(1, 2, "Rendering report…")
            return generate_report(result, p, command_line="SOZLab Workbench")

        self._start_task("export_report", job)

    def extract_frames(self, rule: str, directory: str, fmt: str = "xtc", min_run: int = 1, gap: int = 0) -> None:
        assert self.result is not None and self.run_project is not None
        project, result = self.run_project, self.result
        name = next(iter(result.soz_results))

        def job(progress, cancelled):
            import MDAnalysis as mda

            from engine.extract import select_from_result, write_extracted_trajectory

            sel = select_from_result(result, name, rule, min_run_length=min_run, gap_tolerance=gap)
            if not sel.frame_indices:
                raise ValueError("The rule selects no frames. Lower the threshold.")
            u = mda.Universe(project.inputs.topology, project.inputs.trajectory)
            return write_extracted_trajectory(u, sel, directory, fmt=fmt, progress_cb=progress, cancel=cancelled,
                                              topology_path=project.inputs.topology, trajectory_path=project.inputs.trajectory)

        self._start_task("extract", job)

    def preview_extraction(self, rule: str, min_run: int = 1, gap: int = 0):
        from engine.extract import select_from_result

        name = next(iter(self.result.soz_results))
        return select_from_result(self.result, name, rule, min_run_length=min_run, gap_tolerance=gap)

    def _start_task(self, name: str, job) -> None:
        if self.tasks.running:
            self.message.emit("Another task is still running.", "warn")
            return
        self._task_name = name
        self.tasks.start(job)

    # ------------------------------------------------------------------------------------ project files
    def save_project(self, path: str) -> None:
        """One file format for both front ends: a plain SOZLab project (what ``sozlab-cli run --project`` reads) plus a
        ``workbench`` block holding the Workbench's own state (zone chips, advanced settings). The CLI ignores the block."""
        data = self.build_project().to_dict()
        data["workbench"] = {"version": 1, "zone": dict(self.zone.__dict__), "solvent": dict(self.solvent.__dict__),
                             "run": dict(self.run.__dict__)}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)

    def load_project(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        wb = data.get("workbench")
        if wb:
            for obj, key in ((self.zone, "zone"), (self.solvent, "solvent"), (self.run, "run")):
                for k, v in (wb.get(key) or {}).items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        else:  # a plain SOZLab project (CLI / v2.0)
            from engine.models import ProjectConfig as PC

            proj = PC.from_dict(data)
            if proj.sozs:
                root = proj.sozs[0].root
                sel_label = root.params.get("selection_label")
                if sel_label in proj.selections:
                    self.zone.anchor = proj.selections[sel_label].selection or ""
                cut = root.params.get("cutoff", (root.params.get("cutoffs") or [3.5])[0])
                self.zone.cutoff = float(cut)
                self.zone.name = proj.sozs[0].name
            self.solvent.resnames = list(proj.solvent.water_resnames)
            self.solvent.probe_selection = proj.solvent.probe.selection
            self.run.stride, self.run.frame_start, self.run.frame_stop = proj.analysis.stride, proj.analysis.frame_start, proj.analysis.frame_stop
            self.run.output_dir = proj.outputs.output_dir
        inp = data["inputs"]
        self.open_inputs(inp["topology"], inp.get("trajectory"))

    def shutdown(self) -> None:
        self.preview.shutdown()
