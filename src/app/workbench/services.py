"""Background services: input scanning, live zone preview, analysis runs, exports. Everything that touches an
MDAnalysis Universe happens off the GUI thread, with its own Universe (they are not thread safe)."""
from __future__ import annotations

import os
import threading
import traceback
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt6 import QtCore

from engine.models import ProjectConfig, SolventConfig


# ------------------------------------------------------------------------------------------------ scanning
def _guess_probe(species: List[dict]) -> str:
    for sp in species:
        if sp["kind"] == "water":
            names = [n.upper() for n in sp["atom_names"]]
            for cand in ("OW", "OH2", "O", "OW1", "OH"):
                if cand in names:
                    return f"name {cand}"
            for n in names:
                if n.startswith("O"):
                    return f"name {n}"
    return "name O OW OH2"


def describe_open_error(exc: Exception, topology: str, trajectory: Optional[str]) -> str:
    """A message a researcher can act on (MDAnalysis errors are often empty or cryptic)."""
    msg = str(exc).strip()
    low = msg.lower()
    names = os.path.basename(topology) + (f" + {os.path.basename(trajectory)}" if trajectory else "")
    if "number of atoms" in low or "atoms in" in low and "!=" in low:
        return f"The trajectory and the topology do not describe the same system ({msg}). Use the topology the trajectory was written from."
    if isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        return msg or f"File not found: {names}"
    ext = os.path.splitext(topology)[1] or "file"
    vague = not msg or isinstance(exc, (StopIteration, EOFError, IndexError, UnicodeDecodeError, KeyError))
    why = (f"the file is empty, truncated or not a valid {ext} file" if vague else msg)
    return (f"Could not read {names}: {why}. Supported topologies: .gro .pdb .psf .prmtop/.parm7 .tpr .mol2 .xyz; "
            "trajectories: .xtc .trr .dcd .nc .crd .trj .lammpstrj.")


def open_universe(topology: str, trajectory: Optional[str]):
    import MDAnalysis as mda

    try:
        return mda.Universe(topology, trajectory) if trajectory else mda.Universe(topology)
    except Exception as exc:
        raise ValueError(describe_open_error(exc, topology, trajectory)) from exc


def scan_inputs(topology: str, trajectory: Optional[str]) -> Dict[str, object]:
    """Open the inputs and summarise everything the first screen needs. Raises with a readable message."""
    import MDAnalysis as mda

    from engine.solvent import detect_solvent_species
    from engine.timing import SOURCE_ASSUMED, assess_timing, detect_time_source
    from engine.periodic import box_from_dimensions

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not os.path.isfile(topology):
            raise FileNotFoundError(f"Topology not found: {topology}")
        if trajectory and not os.path.isfile(trajectory):
            raise FileNotFoundError(f"Trajectory not found: {trajectory}")
        u = open_universe(topology, trajectory)
        n_atoms, n_frames = len(u.atoms), len(u.trajectory)
        ts0 = u.trajectory[0]
        box = box_from_dimensions(ts0.dimensions)
        source = detect_time_source(u)
        idx = sorted(set(list(range(min(n_frames, 60))) + [n_frames - 1])) if n_frames > 1 else [0]
        times = [float(u.trajectory[i].time) for i in idx]
        timing = assess_timing(idx, times, source)
        u.trajectory[0]
        species = detect_solvent_species(u)
        water_names = [s["resname"] for s in species if s["kind"] == "water"]
        ion_names = [s["resname"] for s in species if s["kind"] == "ion"]
        comps: List[Dict[str, object]] = []
        n_prot = len(u.select_atoms("protein"))
        if n_prot:
            comps.append({"label": "Protein", "selection": "protein", "n_atoms": n_prot, "kind": "protein"})
        try:
            n_nuc = len(u.select_atoms("nucleic"))
        except Exception:
            n_nuc = 0
        if n_nuc:
            comps.append({"label": "Nucleic acid", "selection": "nucleic", "n_atoms": n_nuc, "kind": "nucleic"})
        for sp in species:
            if sp["kind"] == "water":
                continue
            name = sp["resname"]
            if sp["kind"] == "ion":
                comps.append({"label": f"Ion {name}", "selection": f"resname {name}", "n_atoms": sp["n_residues"], "kind": "ion"})
            elif sp["n_residues"] <= 40 and sp["atoms_per_residue"] >= 3 and name not in {"ACE", "NME", "NMA"}:
                prot_names = set(u.select_atoms("protein").residues.resnames) if n_prot else set()
                if name not in prot_names:
                    comps.append({"label": f"{name} ({sp['n_residues']}×{sp['atoms_per_residue']} atoms)", "selection": f"resname {name}",
                                  "n_atoms": sp["n_residues"] * sp["atoms_per_residue"], "kind": "ligand"})
        notes: List[str] = []
        if box is None:
            notes.append("No periodic box in the inputs: distances will not be periodic.")
        if not water_names:
            notes.append("No water-like residues were found: set the solvent manually (Advanced).")
        return {
            "topology": topology, "trajectory": trajectory, "n_atoms": n_atoms, "n_frames": n_frames,
            "box": None if box is None else [float(x) for x in box.dims],
            "min_image_limit": None if box is None else float(box.min_image_limit),
            "timing": timing.to_dict(), "time_source": source, "time_assumed": source == SOURCE_ASSUMED,
            "species": species, "water_resnames": water_names, "ion_resnames": ion_names,
            "probe_selection": _guess_probe(species), "components": comps, "notes": notes,
        }


# ------------------------------------------------------------------------------------------------ preview
class PreviewWorker(QtCore.QObject):
    loaded = QtCore.pyqtSignal(object)
    loadFailed = QtCore.pyqtSignal(str)
    sceneReady = QtCore.pyqtSignal(object)
    curveReady = QtCore.pyqtSignal(object)
    selectionInfo = QtCore.pyqtSignal(object)

    def __init__(self, latest: Dict[str, int]):
        super().__init__()
        self._u = None
        self._latest = latest
        self._cache_key = None
        self._solvent = None
        self._sel_cache: Dict[str, object] = {}

    # --------------------------------------------------------------------------------------- load
    @QtCore.pyqtSlot(str, object)
    def load(self, topology: str, trajectory) -> None:
        import MDAnalysis as mda

        try:
            summary = scan_inputs(topology, trajectory)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._u = open_universe(topology, trajectory)
            self._cache_key, self._solvent, self._sel_cache = None, None, {}
            self.loaded.emit(summary)
        except Exception as exc:
            self.loadFailed.emit(str(exc) or f"{type(exc).__name__} while reading the files")

    # --------------------------------------------------------------------------------------- helpers
    def _get_solvent(self, solvent_cfg: dict):
        from engine.solvent import build_solvent

        key = repr(sorted(solvent_cfg.items(), key=lambda kv: kv[0]))
        if key != self._cache_key:
            self._solvent = build_solvent(self._u, SolventConfig.from_dict(solvent_cfg))
            self._cache_key = key
        return self._solvent

    def _selection(self, text: str, updating: bool = False):
        from engine.models import SelectionSpec
        from engine.resolver import resolve_selection

        key = (text, updating)
        if key not in self._sel_cache:
            if len(self._sel_cache) > 64:
                self._sel_cache.clear()
            self._sel_cache[key] = resolve_selection(self._u, SelectionSpec(label="x", selection=text, updating=updating))
        return self._sel_cache[key]

    # --------------------------------------------------------------------------------------- selection info
    @QtCore.pyqtSlot(object)
    def selection_info(self, req: dict) -> None:
        if self._u is None:
            return
        if self._latest.get("selection") != req["token"]:
            return
        text = (req.get("selection") or "").strip()
        out = {"token": req["token"], "selection": text, "ok": False, "n_atoms": 0, "n_residues": 0, "resnames": [], "message": ""}
        if not text:
            out["message"] = "Type a selection, or pick one of the detected parts above."
            self.selectionInfo.emit(out)
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from engine.resolver import sanitize_selection_string
                g = self._u.select_atoms(sanitize_selection_string(text))
            out.update(ok=len(g) > 0, n_atoms=len(g), n_residues=len(g.residues),
                       resnames=sorted(set(map(str, g.residues.resnames)))[:6])
            out["message"] = "" if len(g) else "This selection matches no atoms."
        except Exception as exc:
            out["message"] = f"Selection error: {exc}"
        self.selectionInfo.emit(out)

    # --------------------------------------------------------------------------------------- scene
    @QtCore.pyqtSlot(object)
    def request_scene(self, req: dict) -> None:
        if self._u is None or self._latest.get("scene") != req["token"]:
            return
        out: Dict[str, object] = {"token": req["token"], "ok": False}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = self._scene(req)
        except Exception as exc:
            out = {"token": req["token"], "ok": False, "error": str(exc), "trace": traceback.format_exc(limit=3)}
        self.sceneReady.emit(out)

    def _scene(self, req: dict) -> Dict[str, object]:
        from engine.models import SOZDefinition
        from engine.periodic import box_from_dimensions, minimum_image
        from engine.soz_eval import EvaluationContext, evaluate_mask_at, _Items

        u = self._u
        frame = int(max(0, min(req["frame"], len(u.trajectory) - 1)))
        solvent = self._get_solvent(req["solvent"])
        lay = solvent.layout
        zone = req.get("zone")
        selections = {}
        soz = None
        if zone:
            sels = {k: v for k, v in zone["selections"].items()}
            from engine.models import SelectionSpec
            from engine.resolver import resolve_selection

            for label, spec in sels.items():
                selections[label] = self._selection(spec["selection"], bool(spec.get("updating")))
            soz = SOZDefinition.from_dict(zone["soz"])
        ctx = EvaluationContext(u, solvent, selections)

        def membership(f: int) -> np.ndarray:
            ts = u.trajectory[f]
            pos = ts.positions
            box = box_from_dimensions(ts.dimensions)
            return evaluate_mask_at(soz.root, ctx, pos, box, _Items(solvent, pos, box))

        mask = membership(frame) if soz is not None and selections else np.zeros(lay.n_solvent, bool)
        prev = req.get("prev_frame")
        prev_mask = membership(int(prev)) if (prev is not None and soz is not None and selections) else None
        ts = u.trajectory[frame]
        pos = ts.positions
        box = box_from_dimensions(ts.dimensions)

        anchor_idx = np.asarray(selections["anchor"].group.indices, dtype=np.int64) if "anchor" in selections else np.empty(0, np.int64)
        solvent_atoms = set(map(int, lay.all_idx.tolist()))
        # solute: every non-solvent atom (downsampled), anchor flagged
        n_atoms = len(u.atoms)
        is_solvent = np.zeros(n_atoms, bool)
        is_solvent[lay.all_idx] = True
        solute_idx = np.flatnonzero(~is_solvent)
        if len(solute_idx) > 20000:
            keep = np.unique(np.concatenate([solute_idx[:: max(1, len(solute_idx) // 20000)], anchor_idx]))
            solute_idx = keep[~is_solvent[keep]]
        center = pos[anchor_idx].mean(axis=0) if len(anchor_idx) else (pos[solute_idx].mean(axis=0) if len(solute_idx) else np.zeros(3))
        sx = np.asarray(pos[solute_idx], dtype=np.float64)
        # make the solute whole & centred in the box for display
        if box is not None and len(sx) > 1 and len(anchor_idx):
            sx = center + minimum_image(sx - center, box)
        roles = np.zeros(len(solute_idx), np.uint8)
        if len(anchor_idx):
            roles[np.isin(solute_idx, anchor_idx)] = 1
        # solvent (probe atoms): wrap to the image nearest the anchor centre, keep those in the view radius
        pxyz, owner = (pos[lay.probe_idx], lay.probe_owner)
        pxyz = np.asarray(pxyz, dtype=np.float64)
        wrapped = center + minimum_image(pxyz - center, box) if box is not None else pxyz
        radius = float(req.get("view_radius", 16.0))
        dist = np.linalg.norm(wrapped - center, axis=1)
        sel = np.flatnonzero(dist <= radius)
        if len(sel) > 4000:
            sel = sel[np.argsort(dist[sel])[:4000]]
        # one probe per molecule for display
        first_per_mol = {}
        for k in sel:
            first_per_mol.setdefault(int(owner[k]), int(k))
        ks = np.array(sorted(first_per_mol.values()), dtype=np.int64)
        ord_sel = owner[ks] if len(ks) else np.empty(0, np.int64)
        state = np.zeros(len(ks), np.uint8)
        if len(ks):
            inside = mask[ord_sel]
            state[inside] = 1
            if prev_mask is not None:
                state[inside & ~prev_mask[ord_sel]] = 2
                state[(~inside) & prev_mask[ord_sel]] = 3
        labels = [solvent.record_by_resindex[int(lay.resindex_by_ord[o])].display_id for o in ord_sel]
        # hydrogens of the displayed waters (3-atom molecules only), kept attached to their wrapped oxygen
        hyd = None
        if len(ks):
            try:
                n_atoms_mol = np.bincount(lay.all_owner, minlength=lay.n_solvent)
                order = np.argsort(lay.all_owner, kind="stable")
                starts = np.concatenate([[0], np.cumsum(n_atoms_mol)[:-1]])
                hyd = np.full((len(ks), 2, 3), np.nan, dtype=np.float32)
                for row, (k, o) in enumerate(zip(ks, ord_sel)):
                    if n_atoms_mol[o] != 3:
                        continue
                    members = lay.all_idx[order[starts[o]: starts[o] + 3]]
                    others = [a for a in members if a != lay.probe_idx[k]][:2]
                    if len(others) == 2:
                        o_true = pxyz[k]
                        for j, a in enumerate(others):
                            d = np.asarray(pos[a], dtype=np.float64) - o_true
                            if box is not None:
                                d = minimum_image(d[None, :], box)[0]
                            hyd[row, j] = (wrapped[k] + d).astype(np.float32)
            except Exception:
                hyd = None
        return {
            "token": req["token"], "ok": True, "frame": frame, "time_ps": float(ts.time),
            "solute_xyz": sx.astype(np.float32), "solute_roles": roles,
            "solvent_xyz": wrapped[ks].astype(np.float32), "solvent_h": hyd, "solvent_state": state, "solvent_labels": labels,
            "solvent_resindex": lay.resindex_by_ord[ord_sel].astype(np.int64),
            "box": None if box is None else np.asarray(box.dims, dtype=np.float64), "center": center.astype(np.float32),
            "n_in_zone": int(mask.sum()), "n_solvent_total": int(lay.n_solvent),
            "n_entered": int((mask & ~prev_mask).sum()) if prev_mask is not None else 0,
            "n_left": int((~mask & prev_mask).sum()) if prev_mask is not None else 0,
            "n_anchor": int(len(anchor_idx)),
        }

    # --------------------------------------------------------------------------------------- cutoff curve
    @QtCore.pyqtSlot(object)
    def request_curve(self, req: dict) -> None:
        if self._u is None or self._latest.get("curve") != req["token"]:
            return
        out: Dict[str, object] = {"token": req["token"], "ok": False}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from MDAnalysis.lib import distances

                from engine.periodic import box_from_dimensions

                u = self._u
                frame = int(max(0, min(req["frame"], len(u.trajectory) - 1)))
                solvent = self._get_solvent(req["solvent"])
                lay = solvent.layout
                sel = self._selection(req["anchor"])
                ts = u.trajectory[frame]
                box = box_from_dimensions(ts.dimensions)
                dims = None if box is None else np.asarray(box.dims, dtype=np.float32)
                seed = np.asarray(ts.positions[np.asarray(sel.group.indices, dtype=np.int64)], dtype=np.float32)
                items = np.asarray(ts.positions[lay.probe_idx], dtype=np.float32)
                r_max = float(req.get("r_max", 12.0))
                pairs, dist = distances.capped_distance(seed, items, max_cutoff=r_max, box=dims, return_distances=True)
                nearest = np.full(lay.n_solvent, np.inf)
                if len(pairs):
                    np.minimum.at(nearest, lay.probe_owner[pairs[:, 1]], dist)
                out = {"token": req["token"], "ok": True, "nearest": nearest[np.isfinite(nearest)], "frame": frame}
        except Exception as exc:
            out = {"token": req["token"], "ok": False, "error": str(exc)}
        self.curveReady.emit(out)


class SeriesWorker(PreviewWorker):
    """Occupancy of the current zone over a sample of frames, on its own thread and Universe so that dragging the
    cutoff never waits for it (latest request wins; an obsolete request stops after the frame it is on)."""

    seriesReady = QtCore.pyqtSignal(object)

    @QtCore.pyqtSlot(str, object)
    def open(self, topology: str, trajectory) -> None:
        import MDAnalysis as mda

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._u = mda.Universe(topology, trajectory) if trajectory else mda.Universe(topology)
            self._cache_key, self._solvent, self._sel_cache = None, None, {}
        except Exception:
            self._u = None

    @QtCore.pyqtSlot(object)
    def request_series(self, req: dict) -> None:
        if self._u is None or self._latest.get("series") != req["token"]:
            return
        out: Dict[str, object] = {"token": req["token"], "ok": False}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from engine.models import SelectionSpec, SOZDefinition
                from engine.periodic import box_from_dimensions
                from engine.soz_eval import EvaluationContext, _Items, evaluate_mask_at

                u = self._u
                solvent = self._get_solvent(req["solvent"])
                zone = req["zone"]
                selections = {label: self._selection(spec["selection"], bool(spec.get("updating")))
                              for label, spec in zone["selections"].items()}
                soz = SOZDefinition.from_dict(zone["soz"])
                ctx = EvaluationContext(u, solvent, selections)
                frames, times, counts = [], [], []
                for f in req["frames"]:
                    if self._latest.get("series") != req["token"]:
                        return  # superseded: stop without emitting
                    ts = u.trajectory[int(f)]
                    pos = ts.positions
                    box = box_from_dimensions(ts.dimensions)
                    mask = evaluate_mask_at(soz.root, ctx, pos, box, _Items(solvent, pos, box))
                    frames.append(int(f))
                    times.append(float(ts.time))
                    counts.append(int(mask.sum()))
                out = {"token": req["token"], "ok": True, "frames": np.asarray(frames), "times": np.asarray(times),
                       "n": np.asarray(counts, dtype=float)}
        except Exception as exc:
            out = {"token": req["token"], "ok": False, "error": str(exc)}
        self.seriesReady.emit(out)


class PreviewService(QtCore.QObject):
    """Owns the preview thread. Call the ``ask_*`` methods from the GUI thread; results arrive as signals."""

    loaded = QtCore.pyqtSignal(object)
    loadFailed = QtCore.pyqtSignal(str)
    sceneReady = QtCore.pyqtSignal(object)
    curveReady = QtCore.pyqtSignal(object)
    selectionInfo = QtCore.pyqtSignal(object)
    seriesReady = QtCore.pyqtSignal(object)
    _load = QtCore.pyqtSignal(str, object)
    _open_series = QtCore.pyqtSignal(str, object)
    _series = QtCore.pyqtSignal(object)
    _scene = QtCore.pyqtSignal(object)
    _curve = QtCore.pyqtSignal(object)
    _selinfo = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.latest: Dict[str, int] = {}
        self._token = 0
        self.thread = QtCore.QThread()
        self.worker = PreviewWorker(self.latest)
        self.worker.moveToThread(self.thread)
        self._load.connect(self.worker.load)
        self._scene.connect(self.worker.request_scene)
        self._curve.connect(self.worker.request_curve)
        self._selinfo.connect(self.worker.selection_info)
        self.worker.loaded.connect(self.loaded)
        self.worker.loadFailed.connect(self.loadFailed)
        self.worker.sceneReady.connect(self._scene_ready)
        self.worker.curveReady.connect(self._curve_ready)
        self.worker.selectionInfo.connect(self._sel_ready)
        self.thread.start()
        self.series_thread = QtCore.QThread()
        self.series_worker = SeriesWorker(self.latest)
        self.series_worker.moveToThread(self.series_thread)
        self._open_series.connect(self.series_worker.open)
        self._series.connect(self.series_worker.request_series)
        self.series_worker.seriesReady.connect(self._series_ready)
        self.series_thread.start()

    def _next(self, kind: str) -> int:
        self._token += 1
        self.latest[kind] = self._token
        return self._token

    def load(self, topology: str, trajectory) -> None:
        self._load.emit(topology, trajectory)
        self._open_series.emit(topology, trajectory)

    def ask_series(self, req: dict) -> None:
        req = dict(req, token=self._next("series"))
        self._series.emit(req)

    def ask_scene(self, req: dict) -> None:
        req = dict(req, token=self._next("scene"))
        self._scene.emit(req)

    def ask_curve(self, req: dict) -> None:
        req = dict(req, token=self._next("curve"))
        self._curve.emit(req)

    def ask_selection(self, selection: str) -> None:
        self._selinfo.emit({"selection": selection, "token": self._next("selection")})

    # drop stale results
    def _scene_ready(self, out):
        if out.get("token") == self.latest.get("scene"):
            self.sceneReady.emit(out)

    def _curve_ready(self, out):
        if out.get("token") == self.latest.get("curve"):
            self.curveReady.emit(out)

    def _series_ready(self, out):
        if out.get("token") == self.latest.get("series"):
            self.seriesReady.emit(out)

    def _sel_ready(self, out):
        if out.get("token") == self.latest.get("selection"):
            self.selectionInfo.emit(out)

    def shutdown(self) -> None:
        for th in (self.thread, self.series_thread):
            th.quit()
            th.wait(3000)


# ------------------------------------------------------------------------------------------------ generic tasks
class TaskWorker(QtCore.QObject):
    """Run ``fn(progress, cancelled)`` in a thread; emits progress/finished/failed."""

    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.fn(lambda c, t, m: self.progress.emit(int(c), int(t), str(m)), self.cancel_event.is_set)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc(limit=6))


class TaskRunner(QtCore.QObject):
    """Owns one background task at a time."""

    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[TaskWorker] = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start(self, fn: Callable) -> None:
        if self.running:
            raise RuntimeError("A task is already running")
        self.thread = QtCore.QThread()
        self.worker = TaskWorker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _cleanup(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)

    def _done(self, result) -> None:
        self._cleanup()
        self.finished.emit(result)

    def _failed(self, msg: str, trace: str) -> None:
        self._cleanup()
        self.failed.emit(msg, trace)
