"""Per-frame evaluation and (optionally parallel) execution over a frame window.

Why this is split from the statistics
-------------------------------------
Evaluating a frame - "which solvent molecules are inside each zone?" - depends only on that
frame's coordinates, so frames are independent and can be evaluated by many processes.
Everything that depends on *order* (entries, exits, residence, survival, uncertainty) is
computed afterwards, in one process, from the records merged in frame order. Consequently
results are identical for any number of workers, which is verified bit-for-bit in the test
suite, and there is no cross-process state to get out of sync.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import MDAnalysis as mda

from engine.models import ProjectConfig
from engine.periodic import Box, box_from_dimensions, min_distance
from engine.provenance import frame_fingerprint, sample_atom_indices
from engine.resolver import ResolvedSelection, resolve_selection
from engine.solvent import SolventUniverse, build_solvent, resolve_probe_mode
from engine.soz_eval import EvaluationContext, _Items, evaluate_mask_at
from engine.units import to_internal_length

log = logging.getLogger("sozlab")

#: Frames per task handed to a worker process. Small enough for responsive cancel/progress.
DEFAULT_CHUNK = 24


@dataclass
class FrameRecord:
    sample: int
    frame: int
    time: float
    fp: str
    box_limit: float  # minimum-image limit of this frame's box, NaN when there is no valid box
    zones: Dict[str, np.ndarray]  # SOZ name -> resindices (int64) inside the zone
    min_dist: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _collect_probe_modes(node, default_position: str) -> set:
    modes = set()
    if node.type in ("distance", "shell"):
        raw = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        modes.add(resolve_probe_mode(raw, default_position))
    for child in node.children:
        modes |= _collect_probe_modes(child, default_position)
    return modes


def min_distance_mode(node, default_position: str) -> str:
    """Probe mode used for the minimum-distance traces of one SOZ."""
    modes = _collect_probe_modes(node, default_position)
    if "all" in modes:
        return "all"
    if "atom" in modes:
        return "atom"
    return resolve_probe_mode("probe", default_position)


def load_universe(project: ProjectConfig) -> mda.Universe:
    inputs = project.inputs
    if inputs.trajectory:
        return mda.Universe(inputs.topology, inputs.trajectory)
    return mda.Universe(inputs.topology)


class FrameEvaluator:
    """Everything needed to evaluate frames of one project in one process."""

    def __init__(self, project: ProjectConfig, universe: Optional[mda.Universe] = None,
                 solvent: Optional[SolventUniverse] = None,
                 selections: Optional[Dict[str, ResolvedSelection]] = None):
        self.project = project
        self.universe = universe if universe is not None else load_universe(project)
        self.solvent = solvent if solvent is not None else build_solvent(self.universe, project.solvent)
        self.selections = selections if selections is not None else {
            label: resolve_selection(self.universe, spec) for label, spec in project.selections.items()
        }
        self.context = EvaluationContext(self.universe, self.solvent, self.selections)
        self.sozs = list(project.sozs)
        self.modes = {s.name: min_distance_mode(s.root, self.solvent.probe.position) for s in self.sozs}
        self.store_min = bool(project.analysis.store_min_distances)
        self._fp_idx = sample_atom_indices(len(self.universe.atoms))
        self._layout = self.solvent.layout

    def evaluate(self, sample: int, frame: int) -> FrameRecord:
        ts = self.universe.trajectory[frame]
        pos = ts.positions
        dims = ts.dimensions
        box = box_from_dimensions(dims)
        items = _Items(self.solvent, pos, box)
        zones: Dict[str, np.ndarray] = {}
        min_d: Dict[str, Dict[str, float]] = {}
        resindex_by_ord = self._layout.resindex_by_ord
        for soz in self.sozs:
            mask = evaluate_mask_at(soz.root, self.context, pos, box, items)
            zones[soz.name] = resindex_by_ord[mask]
            if self.store_min:
                per_label: Dict[str, float] = {}
                if mask.any():
                    item_pos, owner = items.get(self.modes[soz.name])
                    in_zone = item_pos[mask[owner]]
                for label, sel in self.selections.items():
                    if not mask.any():
                        per_label[label] = float("nan")
                    else:
                        seed_idx = np.asarray(sel.group.indices, dtype=np.int64)
                        per_label[label] = min_distance(pos[seed_idx], in_zone, box)
                min_d[soz.name] = per_label
        return FrameRecord(
            sample=sample,
            frame=int(frame),
            time=float(ts.time),
            fp=frame_fingerprint(pos, dims, self._fp_idx),
            box_limit=float(box.min_image_limit) if box is not None else float("nan"),
            zones=zones,
            min_dist=min_d,
        )


# ------------------------------------------------------------------ worker-process plumbing
_WORKER: Dict[str, FrameEvaluator] = {}


def _init_worker(project_payload: dict) -> None:
    project = ProjectConfig.from_dict(project_payload)
    _WORKER["ev"] = FrameEvaluator(project)


def _eval_chunk(chunk: Sequence[Tuple[int, int]]) -> List[FrameRecord]:
    ev = _WORKER["ev"]
    return [ev.evaluate(s, f) for s, f in chunk]


def resolve_worker_count(workers: Optional[int]) -> int:
    cpu_total = int(os.cpu_count() or 1)
    if workers is None or int(workers) <= 0:
        return max(1, cpu_total - 1)
    return max(1, min(int(workers), cpu_total))


@dataclass
class RunInfo:
    workers_requested: Optional[int]
    workers_used: int
    mode: str  # "serial" | "parallel"
    reason: str
    seconds: float = 0.0
    est_serial_seconds: float = float("nan")
    fell_back: bool = False
    frames_done: int = 0
    cancelled: bool = False


def run_frames(
    project: ProjectConfig,
    evaluator: FrameEvaluator,
    frames: Sequence[int],
    workers: Optional[int],
    consume: Callable[[FrameRecord], None],
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    chunk: int = DEFAULT_CHUNK,
    parallel_min_seconds: float = 10.0,
) -> RunInfo:
    """Evaluate ``frames``; call ``consume(record)`` for each record **in frame order**.

    The first few frames always run in this process and are timed; processes are started only
    when the estimated serial time of the rest exceeds ``parallel_min_seconds`` (process
    start-up costs seconds). If cancelled, the records consumed form a contiguous prefix.
    """
    t0 = time.perf_counter()
    n = len(frames)
    w_eff = resolve_worker_count(workers)
    state = {"done": 0}

    def _emit(rec: FrameRecord) -> None:
        consume(rec)
        state["done"] += 1

    def _serial(start: int) -> bool:
        for sample in range(start, n):
            if cancel and cancel():
                return False
            _emit(evaluator.evaluate(sample, int(frames[sample])))
            if progress:
                progress(state["done"], n)
        return True

    probe_n = min(3, n)
    if not _serial_range(0, probe_n, evaluator, frames, _emit, cancel, progress, n, state):
        return RunInfo(workers, 1, "serial", "cancelled", time.perf_counter() - t0, frames_done=state["done"], cancelled=True)
    per_frame = (time.perf_counter() - t0) / max(probe_n, 1)
    remaining = n - probe_n
    est_serial = per_frame * remaining
    use_parallel = w_eff > 1 and remaining >= 2 * chunk and est_serial >= parallel_min_seconds
    if not use_parallel:
        if w_eff > 1 and remaining > 0:
            reason = (f"short run (estimated {est_serial:.1f} s serial): starting processes would cost more than it saves"
                      if est_serial < parallel_min_seconds else "too few frames to split")
        else:
            reason = "single worker" if w_eff == 1 else "no frames left"
        completed = _serial(probe_n)
        return RunInfo(workers, 1, "serial", "cancelled" if not completed else reason, time.perf_counter() - t0,
                       est_serial, frames_done=state["done"], cancelled=not completed)

    payload = project.to_dict()
    todo = [(s, int(frames[s])) for s in range(probe_n, n)]
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    pending: Dict[int, List[FrameRecord]] = {}
    next_chunk = 0
    cancelled = False
    ctx = multiprocessing.get_context("spawn")
    try:
        with cf.ProcessPoolExecutor(max_workers=min(w_eff, len(chunks)), mp_context=ctx,
                                    initializer=_init_worker, initargs=(payload,)) as pool:
            futures = {pool.submit(_eval_chunk, c): i for i, c in enumerate(chunks)}
            for fut in cf.as_completed(futures):
                pending[futures[fut]] = fut.result()
                while next_chunk in pending:  # consume strictly in order
                    for rec in pending.pop(next_chunk):
                        _emit(rec)
                    next_chunk += 1
                if progress:
                    progress(state["done"], n)
                if cancel and cancel():
                    cancelled = True
                    for f in futures:
                        f.cancel()
                    break
    except Exception as exc:  # infrastructure failure (spawn, pickling, broken pool)
        if state["done"] > probe_n:
            raise  # records were already consumed; cannot restart cleanly
        log.warning("Parallel frame evaluation failed (%s); running serially.", exc)
        completed = _serial(probe_n)
        return RunInfo(workers, 1, "serial", f"parallel failed ({type(exc).__name__}: {exc})", time.perf_counter() - t0,
                       est_serial, fell_back=True, frames_done=state["done"], cancelled=not completed)
    return RunInfo(workers, min(w_eff, len(chunks)), "parallel",
                   "cancelled" if cancelled else f"estimated {est_serial:.1f} s serial",
                   time.perf_counter() - t0, est_serial, frames_done=state["done"], cancelled=cancelled)


def _serial_range(start, stop, evaluator, frames, emit, cancel, progress, n, state) -> bool:
    for sample in range(start, stop):
        if cancel and cancel():
            return False
        emit(evaluator.evaluate(sample, int(frames[sample])))
        if progress:
            progress(state["done"], n)
    return True
