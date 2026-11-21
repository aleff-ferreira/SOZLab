"""Main analysis engine for SOZLab."""
from __future__ import annotations

import json
import logging
import os
import warnings as pywarnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances

from engine._version import ALGORITHM_VERSIONS, __version__
from engine.density import (
    DensityMapResult,
    arrays_sha256,
    run_solute_density,
    selection_is_dynamic,
    write_density_outputs,
    _default_slices,
)
from engine.frames import (
    FrameEvaluator,
    FrameRecord,
    RunInfo,
    load_universe,
    resolve_worker_count,
    run_frames,
)
from engine.models import (
    AnalysisOptions,
    DensityMapConfig,
    InputConfig,
    ProjectConfig,
    SOZDefinition,
    SOZNode,
)
from engine.preflight import density_conditioning_issue, run_preflight
from engine.provenance import InputsChangedError, build_provenance, file_fingerprint, verify_inputs
from engine.resolver import sanitize_selection_string
from engine.solvent import SolventUniverse, resolve_probe_mode, solvent_positions
from engine.soz_eval import EvaluationContext, evaluate_node, max_cutoff_angstrom
from engine.stats import StatsAccumulator
from engine.timing import (
    SOURCE_USER,
    FrameTiming,
    assess_timing,
    detect_time_source,
    gap_tolerance_in_frames,
)
from engine.trust import build_trust
from engine.units import to_internal_length

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in practice
    yaml = None


ProgressCallback = Callable[[int, int, str], None]
PROGRESS_RESERVE_UNITS = 2
STRIDE_MULTIPLIERS = (2, 4)


def analysis_frame_indices(n_frames: int, options: AnalysisOptions) -> range:
    """Return the trajectory[start:stop:step] window used by SOZLab analyses."""
    if options.stride <= 0:
        raise ValueError("Frame stride must be positive")
    # Same slicing semantics as MDAnalysis AnalysisBase.run(start, stop, step) so that
    # validate() audits exactly the frames run() analyses.
    start = max(int(options.frame_start), 0)
    stop = n_frames if options.frame_stop is None else min(int(options.frame_stop), n_frames)
    return range(start, stop, int(options.stride))


# ------------------------------------------------------------------------------------ results
class RunCancelledError(ValueError):
    """The run was cancelled before any frame was analysed: there is no (partial) result to return."""


@dataclass
class HydrationResult:  # kept for import compatibility; the analysis itself was removed
    name: str
    table: pd.DataFrame
    frame_times: List[float]
    frame_labels: List[int]
    contact_frames_total: Dict[int, List[int]]
    contact_frames_given_soz: Dict[int, List[int]]
    mode: str


@dataclass
class SOZResult:
    name: str
    summary: Dict[str, object]
    per_frame: pd.DataFrame
    per_solvent: pd.DataFrame
    min_distance_traces: Optional[pd.DataFrame]
    # Frame counts per continuous/intermittent residence segment (samples, not time);
    # multiply by summary["dt"] only for uniformly sampled runs - use `visits` for durations.
    residence_cont: Dict[int, List[int]]
    residence_inter: Dict[int, List[int]]
    # ---- added in 2.1 ----
    survival: Optional[pd.DataFrame] = None
    visits: Optional[pd.DataFrame] = None
    stride_sensitivity: Optional[List[Dict[str, object]]] = None


@dataclass
class WaterDynamicsResult:  # kept for import compatibility; the analysis itself was removed
    name: str
    sp_tau: pd.DataFrame
    mean_residence_time: float
    residence_mode: str
    hbl: Optional[pd.DataFrame]
    hbl_summary: Optional[pd.DataFrame]
    wor: Optional[pd.DataFrame]
    notes: List[str]


@dataclass
class AnalysisResult:
    soz_results: Dict[str, SOZResult]
    hbond_hydration_results: Dict[str, HydrationResult]
    density_results: Dict[str, DensityMapResult]
    water_dynamics_results: Dict[str, WaterDynamicsResult]
    solvent_records: Dict[int, object]
    warnings: List[str]
    qc_summary: Dict[str, object]
    # ---- added in 2.1 ----
    timing: Optional[FrameTiming] = None
    provenance: Dict[str, object] = field(default_factory=dict)
    complete: bool = True
    cancelled: bool = False
    run_info: Dict[str, object] = field(default_factory=dict)


def _dedupe(messages: List[str]) -> List[str]:
    seen, out = set(), []
    for m in messages:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ------------------------------------------------------------------------------------ engine
class SOZAnalysisEngine:
    def __init__(self, project: ProjectConfig):
        self.project = project

    def _load_universe(self) -> mda.Universe:
        return load_universe(self.project)

    def _frame_indices(self, n_frames: int, options: AnalysisOptions) -> range:
        return analysis_frame_indices(n_frames, options)

    def run(
        self,
        progress: Optional[ProgressCallback] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        logger: Optional[logging.Logger] = None,
        command_line: Optional[str] = None,
        use_cache: bool = True,
    ) -> AnalysisResult:
        project = self.project
        options = project.analysis
        warnings: List[str] = []

        # ---- provenance of the inputs, taken before any frame is read --------------------------
        provenance = build_provenance(project, command_line=command_line)
        universe = self._load_universe()

        preflight = run_preflight(project, universe)
        if not preflight.ok:
            raise ValueError("Preflight failed: " + "; ".join(preflight.errors))
        warnings.extend(preflight.warnings)
        if project.hbond_hydration or project.water_dynamics:
            warnings.append(
                "hbond_hydration / water_dynamics configurations are ignored: these analyses were disabled in "
                "the SOZLab 2.0 GUI and CLI and have been removed from the engine."
            )

        n_frames_total = len(universe.trajectory)
        provenance["trajectory"] = {"n_frames": n_frames_total, "n_atoms": len(universe.atoms)}
        if logger:
            logger.info("Topology: %s", project.inputs.topology)
            logger.info("Trajectory: %s", project.inputs.trajectory or "None")
            logger.info("Total atoms: %d | frames: %d", len(universe.atoms), n_frames_total)

        time_source = SOURCE_USER if options.dt_override else detect_time_source(universe)
        time_unit = "ps"

        evaluator = FrameEvaluator(project, universe=universe)
        solvent = evaluator.solvent
        for note in solvent.notes:
            warnings.append(note)
        for label, sel in evaluator.selections.items():
            warnings.extend(sel.notes)
        if logger:
            logger.info(
                "Solvent residues: %d | probe atoms: %d | probe position: %s",
                len(solvent.residues), len(solvent.probe.atoms), solvent.probe.position,
            )
            for label, sel in evaluator.selections.items():
                logger.info("Selection %s: %d atoms | %s", label, len(sel.group), sel.selection_string)

        frame_indices = self._frame_indices(n_frames_total, options)
        total_frames = len(frame_indices)
        if total_frames == 0:
            raise ValueError("No frames selected for analysis")
        workers_req = options.workers
        workers = resolve_worker_count(workers_req)

        # ---- progress plan --------------------------------------------------------------------------
        density_cfgs = list(project.density_maps)
        density_units = 0
        for cfg in density_cfgs:
            if (cfg.engine or "solute").lower() == "solute":
                density_units += len(_density_window(cfg, n_frames_total, options))
            else:
                density_units += 1
        progress_total = 3 + total_frames + 1 + density_units + 1 + PROGRESS_RESERVE_UNITS
        state = {"units": 0}

        def emit(message: str, units: int | None = None) -> None:
            if units is not None:
                state["units"] = units
            if progress:
                progress(state["units"], progress_total, message)

        emit("Loaded inputs", 1)
        emit("Validated inputs", 2)
        emit("Compiled selections", 3)

        # ---- accumulators ----------------------------------------------------------------------------
        id_map = {idx: rec.display_id for idx, rec in solvent.record_by_resindex.items()}
        accs: Dict[str, StatsAccumulator] = {}
        light: Dict[Tuple[str, int], StatsAccumulator] = {}
        mind_rows: Dict[str, List[Dict[str, float]]] = {}
        for soz in project.sozs:
            accs[soz.name] = StatsAccumulator(
                solvent_records=solvent.record_by_resindex,
                gap_tolerance=options.gap_tolerance,
                frame_stride=options.stride,
                store_ids=options.store_ids,
                store_frame_table=project.outputs.write_per_frame,
                id_map=id_map,
            )
            if options.stride_sensitivity:
                for m in STRIDE_MULTIPLIERS:
                    if total_frames // m >= 20:
                        light[(soz.name, m)] = StatsAccumulator(
                            solvent.record_by_resindex, options.gap_tolerance, options.stride * m,
                            store_ids=False, store_frame_table=False, id_map=id_map,
                        )
            mind_rows[soz.name] = []
        frames_seen: List[int] = []
        times_seen: List[float] = []
        fps: List[str] = []
        box_limits: List[float] = []
        pbc_missing = {"n": 0}
        dt_override = options.dt_override

        def consume(rec: FrameRecord) -> None:
            t = float(rec.frame * dt_override) if dt_override else rec.time
            frames_seen.append(rec.frame)
            times_seen.append(t)
            fps.append(rec.fp)
            box_limits.append(rec.box_limit)
            if not np.isfinite(rec.box_limit):
                pbc_missing["n"] += 1
            for soz in project.sozs:
                present = set(int(i) for i in rec.zones[soz.name])
                accs[soz.name].update(rec.sample, t, present, frame_label=rec.frame, frame_fp=rec.fp)
                for m in STRIDE_MULTIPLIERS:
                    acc = light.get((soz.name, m))
                    if acc is not None and rec.sample % m == 0:
                        acc.update(rec.sample // m, t, present, frame_label=rec.frame)
                if options.store_min_distances:
                    mind_rows[soz.name].append(rec.min_dist.get(soz.name, {}))

        def frame_progress(done: int, total: int) -> None:
            emit(f"Per-frame SOZ evaluation ({done}/{total} frames)", 3 + done)

        run_info = run_frames(
            project, evaluator, list(frame_indices), workers_req, consume,
            progress=frame_progress, cancel=cancel_flag, parallel_min_seconds=options.parallel_min_seconds,
        )
        cancelled = run_info.cancelled
        n_done = len(frames_seen)
        if logger:
            logger.info(
                "Frame evaluation: %s (workers requested=%s used=%d) %d/%d frames in %.2fs - %s",
                run_info.mode, workers_req if workers_req is not None else "auto", run_info.workers_used,
                n_done, total_frames, run_info.seconds, run_info.reason,
            )
        if n_done == 0:
            raise RunCancelledError("The run was cancelled before any frame was analysed")
        if cancelled:
            warnings.append(
                f"The run was cancelled after {n_done} of {total_frames} frames; results cover only that window "
                "and are marked partial."
            )
        emit("Per-frame evaluation complete", 3 + total_frames)

        # ---- timing ---------------------------------------------------------------------------------------
        timing = assess_timing(frames_seen, times_seen, time_source, time_unit)
        for note in timing.notes:
            warnings.append(f"Timing: {note}")
        gap = options.gap_tolerance
        if options.gap_tolerance_time:
            gap = gap_tolerance_in_frames(options.gap_tolerance_time, timing)
            if not timing.usable:
                warnings.append("gap_tolerance_time needs a usable sampling interval; the frame-based tolerance is used.")
                gap = options.gap_tolerance
        n_blocks = int(max(2, options.uncertainty_blocks))

        # ---- per-SOZ results --------------------------------------------------------------------------
        soz_results: Dict[str, SOZResult] = {}
        for soz in project.sozs:
            stats = accs[soz.name].finalize(time_unit=time_unit, timing=timing, n_blocks=n_blocks, gap_tolerance=gap)
            summary = stats["summary"]
            per_frame = stats["per_frame"]
            min_df = None
            if options.store_min_distances:
                min_df = _min_distance_dataframe(mind_rows[soz.name], times_seen, frames_seen)
                if not per_frame.empty:
                    per_frame = pd.concat(
                        [per_frame.reset_index(drop=True), min_df.drop(columns=["frame", "time"]).reset_index(drop=True)],
                        axis=1,
                    )
            sens = _stride_sensitivity(soz.name, summary, light, timing, gap, n_blocks)
            summary["stride_sensitivity"] = sens
            summary["tau_extrapolation"] = _extrapolate_tau(sens)
            summary["gap_tolerance_samples"] = gap
            summary["gap_tolerance_time"] = float(gap * timing.dt) if timing.usable else float("nan")
            soz_results[soz.name] = SOZResult(
                name=soz.name,
                summary=summary,
                per_frame=per_frame,
                per_solvent=stats["per_solvent"],
                min_distance_traces=min_df,
                residence_cont=stats["residence_cont"],
                residence_inter=stats["residence_inter"],
                survival=stats["survival"],
                visits=stats["visits"],
                stride_sensitivity=sens,
            )
            if logger:
                logger.info(
                    "SOZ %s: occupancy_fraction=%.4f mean_n_solvent=%.3f",
                    soz.name, float(summary.get("occupancy_fraction", 0.0)), float(summary.get("mean_n_solvent", 0.0)),
                )
        emit("Derived metrics", 3 + total_frames + 1)

        # ---- box / cutoff checks over the frames actually analysed ------------------------------------
        finite_limits = [b for b in box_limits if np.isfinite(b)]
        min_limit = float(min(finite_limits)) if finite_limits else float("nan")
        if pbc_missing["n"] == n_done:
            warnings.append("No valid box vectors found; distances are NOT periodic.")
        elif pbc_missing["n"]:
            warnings.append(f"{pbc_missing['n']} of {n_done} frames have no valid box; those frames are not periodic.")
        for soz in project.sozs:
            cut = max_cutoff_angstrom(soz.root)
            if np.isfinite(min_limit) and cut > min_limit:
                warnings.append(
                    f"SOZ '{soz.name}': largest cut-off {cut:.2f} A exceeds half the shortest box width "
                    f"({min_limit:.2f} A at its smallest); the zone overlaps its own periodic images."
                )

        # ---- density maps --------------------------------------------------------------------------------
        density_results: Dict[str, DensityMapResult] = {}
        if density_cfgs and not cancelled:
            def density_progress(base: int, window: int):
                def _cb(done: int, total: int) -> None:
                    emit(f"Density map ({done}/{total} frames)", base + int(done * window / max(total, 1)))
                return _cb

            base_units = 3 + total_frames + 1
            for cfg in density_cfgs:
                win = len(_density_window(cfg, n_frames_total, options)) if (cfg.engine or "solute").lower() == "solute" else 1
                try:
                    res = _run_one_density(
                        universe, cfg, options, project.outputs.output_dir, logger, project.inputs,
                        provenance["inputs"], workers, density_progress(base_units, win), cancel_flag,
                        use_cache=use_cache, parallel_min_seconds=options.parallel_min_seconds,
                    )
                except InterruptedError:
                    cancelled = True
                    warnings.append(f"Density map '{cfg.name}' was cancelled.")
                    break
                if res is not None:
                    density_results[cfg.name] = res
                    for w in res.metadata.get("analysis_warnings", []):
                        warnings.append(f"Density map '{cfg.name}': {w}")
                base_units += win
        emit("Density maps", 3 + total_frames + 1 + density_units)

        # ---- diagnostics ------------------------------------------------------------------------------------
        zero_occupancy = [n for n, s in soz_results.items() if float(s.summary.get("occupancy_fraction", 0.0)) < 1e-9]
        zero_diagnostics: Dict[str, List[str]] = {}
        if zero_occupancy:
            warnings.append("SOZs with zero occupancy: " + ", ".join(zero_occupancy))
            lookup = {s.name: s for s in project.sozs}
            context = evaluator.context
            sample_frames = list(frames_seen[: min(50, n_done)])
            for name in zero_occupancy:
                soz_def = lookup.get(name)
                if soz_def:
                    diag = _diagnose_zero_occupancy(soz_def, context, sample_frames, sample_limit=min(50, len(sample_frames)))
                    if diag:
                        zero_diagnostics[name] = diag
                        warnings.extend(diag)

        # ---- the inputs must still be what we fingerprinted at the start -----------------------------
        diffs = verify_inputs(provenance["inputs"])
        if diffs:
            raise InputsChangedError(diffs)

        probe_counts = [len(v) for v in solvent.probe.resindex_to_atom_indices.values()]
        qc_summary = {
            "preflight": preflight.to_dict(),
            "versions": {"MDAnalysis": mda.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                         "sozlab": __version__},
            "analysis_warnings": _dedupe(warnings),
            "workers": run_info.workers_used,
            "workers_requested": workers_req,
            "parallel": {"mode": run_info.mode, "reason": run_info.reason, "seconds": round(run_info.seconds, 3),
                         "fell_back": run_info.fell_back},
            "pbc_used": pbc_missing["n"] < n_done,
            "pbc_missing_frames": pbc_missing["n"],
            "min_image_limit": min_limit,
            "total_frames": total_frames,
            "frames_analysed": n_done,
            "complete": not cancelled,
            "zero_occupancy_sozs": zero_occupancy,
            "zero_occupancy_diagnostics": zero_diagnostics,
            "occupancy_definition": "n_solvent = solvent molecules meeting the SOZ logic tree per frame "
                                    "(probe within cutoff of the selection, minimum image)",
            "time_unit": time_unit,
            "timing": timing.to_dict(),
            "probe_definition": {
                "selection": solvent.probe.selection,
                "position": solvent.probe.position,
                "probe_atom_count": len(solvent.probe.atom_indices),
                "solvent_residue_count": len(solvent.residues),
                "residues_with_probe": sum(1 for c in probe_counts if c > 0),
                "residues_multi_probe": sum(1 for c in probe_counts if c > 1),
                "probe_atoms_per_residue_min": min(probe_counts) if probe_counts else 0,
                "probe_atoms_per_residue_max": max(probe_counts) if probe_counts else 0,
                "probe_source": getattr(project.solvent, "probe_source", None),
            },
            "resolved_selections": {
                label: {"selection": sel.selection_string, "count": len(sel.group), "dynamic": sel.dynamic}
                for label, sel in evaluator.selections.items()
            },
            "soz_definitions": {s.name: _node_qc_entries(s.root, solvent.probe.position) for s in project.sozs},
            "density_definitions": {c.name: c.to_dict() for c in project.density_maps},
            "algorithm_versions": dict(ALGORITHM_VERSIONS),
        }
        qc_summary["trust"] = build_trust(soz_results, density_results, timing, qc_summary)
        emit("Finalizing analysis", 3 + total_frames + 1 + density_units + 1)

        return AnalysisResult(
            soz_results=soz_results,
            hbond_hydration_results={},
            density_results=density_results,
            water_dynamics_results={},
            solvent_records=solvent.record_by_resindex,
            warnings=_dedupe(warnings),
            qc_summary=qc_summary,
            timing=timing,
            provenance={**provenance, "frame_fingerprints": fps},
            complete=not cancelled,
            cancelled=cancelled,
            run_info={
                "mode": run_info.mode, "workers_used": run_info.workers_used, "workers_requested": workers_req,
                "seconds": run_info.seconds, "reason": run_info.reason, "frames_done": n_done,
            },
        )


# --------------------------------------------------------------------------------------- helpers
def _density_window(cfg: DensityMapConfig, n_frames: int, options: AnalysisOptions) -> range:
    start = cfg.frame_start if cfg.frame_start is not None else options.frame_start
    stop = cfg.frame_stop if cfg.frame_stop is not None else options.frame_stop
    step = cfg.stride if cfg.stride else options.stride
    stop_eff = n_frames if stop is None else min(int(stop), n_frames)
    return range(max(int(start), 0), stop_eff, int(step))


def _min_distance_to_soz(seed_group, solvent: SolventUniverse, frame_set: set, box, mode: str) -> float:
    """Smallest distance from a selection to the probes of ``frame_set`` (compatibility helper)."""
    if not frame_set or len(seed_group) == 0:
        return float("nan")
    solvent_pos, _ = solvent_positions(solvent, mode, resindices=frame_set)
    if solvent_pos.size == 0:
        return float("nan")
    dist = distances.distance_array(seed_group.positions, solvent_pos, box=box)
    return float(np.min(dist)) if dist.size else float("nan")


def _min_distance_dataframe(min_distances: List[Dict[str, float]], frame_times: List[float],
                            frame_labels: List[int]) -> pd.DataFrame:
    rows = []
    for idx, data in enumerate(min_distances):
        label = frame_labels[idx] if idx < len(frame_labels) else idx
        row = {"frame": label, "time": frame_times[idx]}
        row.update(data)
        rows.append(row)
    return pd.DataFrame(rows)


def _stride_sensitivity(name: str, summary: Dict[str, object], light: Dict[Tuple[str, int], StatsAccumulator],
                        timing: FrameTiming, gap: int, n_blocks: int) -> List[Dict[str, object]]:
    """How much do the headline numbers change if sampling were 2x/4x coarser?"""

    def pick(sm: Dict[str, object], mult: int) -> Dict[str, object]:
        res = sm["residence"]["continuous"]  # type: ignore[index]
        return {
            "stride_multiplier": mult,
            "dt": sm["dt"],
            "n_samples": sm["n_frames"],
            "mean_n_solvent": sm["mean_n_solvent"],
            "occupancy_fraction": sm["occupancy_fraction"],
            "entry_rate_per_time": sm["entry_rate_per_time"],
            "mean_visit_duration_complete": res["mean_visit_duration_complete"],
            "tau_survival_integral": res["tau_survival_integral"],
            "tau_survival_integral_converged": res["tau_survival_integral_converged"],
            "tau_survival_fit": res["tau_survival_fit"],
        }

    rows = [pick(summary, 1)]
    for m in STRIDE_MULTIPLIERS:
        acc = light.get((name, m))
        if acc is None:
            continue
        sub = assess_timing(acc.frame_labels, acc.frame_times, timing.source, timing.time_unit)
        sm = acc.finalize(time_unit=timing.time_unit, timing=sub, n_blocks=n_blocks, gap_tolerance=max(gap // m, 0))["summary"]
        rows.append(pick(sm, m))
    base = rows[0]
    for row in rows[1:]:
        for key in ("mean_n_solvent", "entry_rate_per_time", "mean_visit_duration_complete", "tau_survival_integral",
                    "tau_survival_fit"):
            a, b = base.get(key), row.get(key)
            try:
                row[f"{key}_rel_change"] = (float(b) / float(a) - 1.0) if (a and np.isfinite(a) and np.isfinite(b)) else float("nan")
            except (TypeError, ZeroDivisionError):
                row[f"{key}_rel_change"] = float("nan")
    return rows


def _extrapolate_tau(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Experimental continuous-time estimate: apparent residence time is ~linear in sqrt(dt).

    Discrete sampling moves the effective zone boundary outward by 0.5826 sqrt(2 D dt) (Broadie, Glasserman &
    Kou 1997), so for diffusive boundary crossing tau(dt) ~ tau0 (1 + c sqrt(dt)). A straight-line fit of tau against
    sqrt(dt) through the 1x/2x/4x points extrapolated to dt -> 0 recovered the exact continuous-time value of a
    Brownian ground truth to -2..-4% (raw values were +12..+50% off). Not valid for ballistic/inertial crossings.
    """
    pts = [(float(np.sqrt(r["dt"])), float(r["tau_survival_integral"])) for r in rows
           if r.get("tau_survival_integral_converged") and np.isfinite(r.get("dt", np.nan))
           and np.isfinite(r.get("tau_survival_integral", np.nan))]
    out: Dict[str, object] = {
        "method": "linear fit of tau_survival_integral against sqrt(dt) over the 1x/2x/4x strides, extrapolated to dt = 0",
        "status": "experimental",
        "caveat": "assumes diffusive zone-boundary crossing on the sampled time scale. Verified on a Brownian ground truth "
                  "(error -2..-4%), but on TIP3P Na+ first-shell water measured against 0.02 ps sampling it was within -24..+4% "
                  "for dt <= 0.32 ps and over-estimated by +28..+64% at dt = 0.64..1.28 ps: indicative only, do not quote it",
        "tau_dt0": float("nan"), "slope_per_sqrt_time": float("nan"), "n_points": len(pts), "valid": False,
    }
    if len(pts) >= 3:
        x, y = np.array(pts).T
        slope, icpt = np.polyfit(x, y, 1)
        out.update({"tau_dt0": float(icpt) if icpt > 0 else float("nan"), "slope_per_sqrt_time": float(slope),
                    "valid": bool(icpt > 0 and slope > 0)})
    return out


def _collect_leaves(node: SOZNode) -> List[SOZNode]:
    out: List[SOZNode] = []
    if node.type in ("distance", "shell"):
        out.append(node)
    for child in node.children:
        out.extend(_collect_leaves(child))
    return out


def _node_label(node: SOZNode) -> str:
    label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
    if node.type == "distance":
        return f"distance({label}, cutoff={node.params.get('cutoff', '?')}{node.params.get('unit', 'A')})"
    if node.type == "shell":
        return f"shell({label}, cutoffs={node.params.get('cutoffs', [])}{node.params.get('unit', 'A')})"
    return node.type


def _node_qc_entries(node: SOZNode, default_position: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if node.type in ("distance", "shell"):
        raw_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        unit = node.params.get("unit", "A")
        entry: Dict[str, object] = {
            "type": node.type,
            "selection_label": node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed"),
            "probe_mode": resolve_probe_mode(raw_mode, default_position),
            "unit": unit,
        }
        if node.type == "distance":
            cutoff = float(node.params.get("cutoff", 3.5))
            entry["cutoff"] = cutoff
            entry["cutoff_angstrom"] = to_internal_length(cutoff, unit)
        else:
            cutoffs = [float(v) for v in node.params.get("cutoffs", [3.5])]
            entry["cutoffs"] = cutoffs
            entry["cutoffs_angstrom"] = [to_internal_length(v, unit) for v in cutoffs]
            entry["shell_mode"] = str(node.params.get("mode", "chained")).lower()
        entries.append(entry)
    for child in node.children:
        entries.extend(_node_qc_entries(child, default_position))
    return entries


def _diagnose_zero_occupancy(soz: SOZDefinition, context: EvaluationContext, frame_indices: List[int],
                             sample_limit: int = 50) -> List[str]:
    leaves = _collect_leaves(soz.root)
    if not leaves:
        return [f"SOZ '{soz.name}' has no distance/shell nodes to evaluate."]
    sample_frames = frame_indices[: max(1, min(sample_limit, len(frame_indices)))]
    hits = {i: 0 for i in range(len(leaves))}
    for frame in sample_frames:
        context.universe.trajectory[frame]
        for i, node in enumerate(leaves):
            if evaluate_node(node, context):
                hits[i] += 1
    if not any(hits.values()):
        return [
            f"SOZ '{soz.name}' has zero occupancy; leaf nodes are empty in sampled frames. "
            "Check selection strings and cutoffs."
        ]
    parts = [f"{_node_label(n)}: {hits[i]}/{len(sample_frames)} frames" for i, n in enumerate(leaves)]
    return [
        f"SOZ '{soz.name}' has zero occupancy but leaf nodes are populated in sampled frames "
        f"({'; '.join(parts[:4])}). Check combine mode (AND vs OR) or cutoffs."
    ]


# ----------------------------------------------------------------------------------------- density
def _density_cache_key(cfg: DensityMapConfig, window: range, inputs_fp: Dict[str, object]) -> Dict[str, object]:
    d = cfg.to_dict()
    for k in ("name", "output_format", "view_mode"):
        d.pop(k, None)

    def ident(fp):
        return None if not fp else {"size": fp["size"], "sha256": fp["sha256"], "hash_mode": fp["hash_mode"]}

    extra = {e["path"]: ident(e) for e in (inputs_fp.get("extra") or [])}
    return {
        "algorithm": ALGORITHM_VERSIONS["density"],
        "sozlab": __version__,
        "mdanalysis": mda.__version__,
        "config": d,
        "window": [window.start, window.stop, window.step, len(window)],
        "topology": ident(inputs_fp.get("topology")),
        "trajectory": ident(inputs_fp.get("trajectory")),
        "extra_inputs": extra,
    }


def _load_density_cache(density_dir: str, key: Dict[str, object]) -> Tuple[Optional[DensityMapResult], str]:
    cache_path = os.path.join(density_dir, "cache.json")
    npz_path = os.path.join(density_dir, "grid.npz")
    if not (os.path.exists(cache_path) and os.path.exists(npz_path)):
        return None, "no cache"
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
    except Exception as exc:
        return None, f"cache unreadable ({exc})"
    if json.loads(json.dumps(key, sort_keys=True, default=str)) != json.loads(json.dumps(cached.get("key"), sort_keys=True, default=str)):
        return None, "inputs, configuration or software changed"
    try:
        data = np.load(npz_path)
        arrays = {k: data[k] for k in data.files}
    except Exception as exc:
        return None, f"cached grid unreadable ({exc})"
    if cached.get("arrays_sha256") != arrays_sha256(arrays):
        return None, "cached grid content does not match its checksum"
    meta = cached.get("metadata", {})
    result = DensityMapResult(
        name=meta.get("name", ""), grid_path=meta.get("grid_path", ""), grid=arrays["grid"],
        axes={"x": arrays["x"], "y": arrays["y"], "z": arrays["z"]}, slices=_default_slices(arrays["grid"]), metadata=meta,
        counts=arrays.get("counts"), se=arrays.get("se"), g=arrays.get("g"), reliable=arrays.get("reliable"),
    )
    return result, "cache hit"


def _store_density_cache(density_dir: str, key: Dict[str, object], result: DensityMapResult) -> None:
    from engine.provenance import write_json_atomic

    write_json_atomic(
        os.path.join(density_dir, "cache.json"),
        {"key": key, "arrays_sha256": result.metadata.get("arrays_sha256"), "metadata": result.metadata},
    )


def _legacy_density(universe: mda.Universe, cfg: DensityMapConfig, window: range, w_start, w_stop, w_step,
                    warnings_list: List[str], logger: Optional[logging.Logger]) -> DensityMapResult:
    """The v2.0.0 MDAnalysis-based map, made deterministic (grid from the window's first frame)."""
    from MDAnalysis.analysis import align as align_module
    from MDAnalysis.analysis import density as density_module

    selection = sanitize_selection_string(cfg.species_selection)
    dynamic = selection_is_dynamic(selection)
    density_universe = universe
    if cfg.align and cfg.align_selection:
        density_universe = universe.copy()
        ref = density_universe
        if cfg.align_reference == "structure" and cfg.align_reference_path:
            ref = mda.Universe(cfg.align_reference_path)
        aligner = align_module.AlignTraj(density_universe, ref, select=cfg.align_selection, in_memory=True)
        aligner.run(start=w_start, stop=w_stop, step=w_step)
    atom_group = density_universe.select_atoms(selection, updating=dynamic) if selection else density_universe.atoms
    density_universe.trajectory[window[0]]  # explicit: the grid is defined by the window's first frame
    analysis = density_module.DensityAnalysis(atom_group, delta=cfg.grid_spacing, padding=cfg.padding)
    analysis.run(start=w_start, stop=w_stop, step=w_step)
    dens = analysis.results.density
    grid = np.asarray(dens.grid)
    origin = np.asarray(dens.origin, dtype=float)
    delta = cfg.grid_spacing
    axes = {k: origin[i] + np.arange(grid.shape[i]) * delta for i, k in enumerate("xyz")}
    ts = universe.trajectory[window[0]]
    vol = float(ts.volume) if ts.dimensions is not None and ts.volume > 0 else float("nan")
    n_atoms = len(atom_group) if not dynamic else float("nan")
    rho_box = n_atoms / vol if np.isfinite(vol) and np.isfinite(n_atoms) else float("nan")
    meta = {
        "engine": "mdanalysis", "algorithm_version": "legacy", "grid_spacing": delta, "padding": cfg.padding,
        "shape": list(grid.shape), "origin": origin.tolist(), "frame_start": w_start, "frame_stop": w_stop, "stride": w_step,
        "n_frames": len(window), "align": cfg.align, "align_selection": cfg.align_selection,
        "align_reference": cfg.align_reference, "align_reference_path": cfg.align_reference_path,
        "density_unit": "Angstrom^-3", "selection": selection, "selection_is_dynamic": dynamic,
        "rho_bulk": rho_box, "rho_bulk_approx": rho_box, "bulk": {"method": "box", "value": rho_box},
        "analysis_warnings": warnings_list, "is_scientifically_safe": cfg.align and not warnings_list,
        "notes": ["MDAnalysis DensityAnalysis grid: bounding box of the selection at the first window frame, "
                  "no periodic images, no uncertainty (legacy engine)."],
    }
    return DensityMapResult(name=cfg.name, grid_path="", grid=grid, axes=axes, slices=_default_slices(grid), metadata=meta)


def _run_one_density(universe, cfg, options, output_dir, logger, inputs, inputs_fp, workers, progress, cancel,
                     use_cache=True, parallel_min_seconds=10.0) -> Optional[DensityMapResult]:
    policy = getattr(cfg, "conditioning_policy", "strict")
    warnings_list: List[str] = []
    msg = density_conditioning_issue(cfg)
    if msg is not None:
        if policy == "strict":
            if logger:
                logger.error(msg)
            return None  # preflight reports this as an error; defensive backstop
        warnings_list.append(msg)
        if logger:
            logger.warning(msg)
    window = _density_window(cfg, len(universe.trajectory), options)
    if len(window) == 0:
        raise ValueError(f"Density map '{cfg.name}': no frames in the window")
    density_dir = os.path.join(output_dir, f"density_map_{cfg.name}")
    os.makedirs(density_dir, exist_ok=True)
    key = _density_cache_key(cfg, window, inputs_fp)
    cache_info = {"hit": False, "reason": "caching disabled"}
    result = None
    if use_cache:
        result, reason = _load_density_cache(density_dir, key)
        cache_info = {"hit": result is not None, "reason": reason}
        if result is not None and logger:
            logger.info("Density map %s: reused cached result (inputs and configuration unchanged).", cfg.name)
        elif logger:
            logger.info("Density map %s: computing (%s).", cfg.name, reason)
    if result is None:
        if (cfg.engine or "solute").lower() == "mdanalysis":
            w_start = cfg.frame_start if cfg.frame_start is not None else options.frame_start
            w_stop = cfg.frame_stop if cfg.frame_stop is not None else options.frame_stop
            w_step = cfg.stride if cfg.stride else options.stride
            result = _legacy_density(universe, cfg, window, w_start, w_stop, w_step, warnings_list, logger)
            if progress:
                progress(1, 1)
        else:
            result = run_solute_density(universe, cfg, options, output_dir, inputs, workers=workers, progress=progress,
                                        cancel=cancel, parallel_min_seconds=parallel_min_seconds)
            result.metadata["analysis_warnings"] = _dedupe(warnings_list + list(result.metadata.get("analysis_warnings", [])))
        result.metadata["name"] = cfg.name
    result.metadata["cache"] = cache_info
    write_density_outputs(result, cfg, output_dir)
    if use_cache and not cache_info["hit"]:
        _store_density_cache(density_dir, key, result)
    return result


def _run_density_maps(
    universe: mda.Universe,
    configs: List[DensityMapConfig],
    options: AnalysisOptions,
    output_dir: str,
    progress: Optional[Callable[[int, int], None]],
    logger: Optional[logging.Logger],
    inputs: Optional[InputConfig] = None,
    workers: int = 1,
    cancel: Optional[Callable[[], bool]] = None,
    use_cache: bool = True,
) -> Dict[str, DensityMapResult]:
    """Compute density maps for ``universe`` (also the entry point used by tests and tools)."""
    results: Dict[str, DensityMapResult] = {}
    if not configs:
        return results
    topology = getattr(universe, "filename", None)
    trajectory = getattr(getattr(universe, "trajectory", None), "filename", None)
    if inputs is None:
        inputs = InputConfig(topology=str(topology) if topology else "", trajectory=str(trajectory) if trajectory else None)
    fps: Dict[str, object] = {"topology": None, "trajectory": None, "extra": []}
    for key_name, path in (("topology", topology), ("trajectory", trajectory)):
        if path and os.path.isfile(str(path)):
            fps[key_name] = file_fingerprint(str(path))
        elif key_name == "trajectory" and path is None:
            fps[key_name] = None
    # An in-memory universe has no file to fingerprint: its results must not be cached across runs.
    cacheable = use_cache and fps["topology"] is not None
    for cfg in configs:
        res = _run_one_density(universe, cfg, options, output_dir, logger, inputs, fps, workers, progress, cancel,
                               use_cache=cacheable)
        if res is not None:
            results[cfg.name] = res
    return results


# ----------------------------------------------------------------------------------- project I/O
def write_project_json(project: ProjectConfig, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    data = project.to_dict()
    if path.endswith((".yaml", ".yml")) and yaml is not None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


def load_project_json(path: str) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    return ProjectConfig.from_dict(data or {})
