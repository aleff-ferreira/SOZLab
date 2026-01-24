"""Main analysis engine for SOZLab."""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances

from engine.models import (
    AnalysisOptions,
    BridgeConfig,
    ProjectConfig,
    ResidueHydrationConfig,
    SOZNode,
)
from engine.resolver import resolve_selection
from engine.solvent import (
    SolventUniverse,
    build_solvent,
    distance_resindices,
    resolve_probe_mode,
    solvent_positions,
)
from engine.soz_eval import EvaluationContext, evaluate_node
from engine.stats import StatsAccumulator
from engine.units import to_internal_length
from engine.preflight import run_preflight

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in practice
    yaml = None


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SOZFrameData:
    accumulator: StatsAccumulator
    frame_times: List[float]
    frame_labels: List[int]
    min_distances: List[Dict[str, float]]


@dataclass
class BridgeFrameData:
    accumulator: StatsAccumulator


@dataclass
class HydrationResult:
    name: str
    table: pd.DataFrame


@dataclass
class SOZResult:
    name: str
    summary: Dict[str, object]
    per_frame: pd.DataFrame
    per_solvent: pd.DataFrame
    min_distance_traces: Optional[pd.DataFrame]
    residence_cont: Dict[int, List[float]]
    residence_inter: Dict[int, List[float]]


@dataclass
class BridgeResult:
    name: str
    per_frame: pd.DataFrame
    per_solvent: pd.DataFrame


@dataclass
class AnalysisResult:
    soz_results: Dict[str, SOZResult]
    bridge_results: Dict[str, BridgeResult]
    hydration_results: Dict[str, HydrationResult]
    solvent_records: Dict[int, object]
    warnings: List[str]
    qc_summary: Dict[str, object]


class SOZAnalysisEngine:
    def __init__(self, project: ProjectConfig):
        self.project = project

    def _load_universe(self) -> mda.Universe:
        if self.project.inputs.trajectory:
            return mda.Universe(self.project.inputs.topology, self.project.inputs.trajectory)
        return mda.Universe(self.project.inputs.topology)

    def _resolve_selections(self, universe: mda.Universe):
        resolved = {}
        for label, spec in self.project.selections.items():
            resolved[label] = resolve_selection(universe, spec)
        return resolved

    def _frame_indices(self, n_frames: int, options: AnalysisOptions) -> range:
        if options.stride <= 0:
            raise ValueError("Frame stride must be positive")
        start = max(options.frame_start, 0)
        stop = options.frame_stop if options.frame_stop is not None else n_frames
        stop = min(stop, n_frames)
        return range(start, stop, options.stride)

    def run(
        self,
        progress: Optional[ProgressCallback] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> AnalysisResult:
        universe = self._load_universe()
        warnings: List[str] = []

        preflight = run_preflight(self.project, universe)
        if not preflight.ok:
            raise ValueError("Preflight failed: " + "; ".join(preflight.errors))
        if preflight.warnings:
            warnings.extend(preflight.warnings)

        if logger:
            logger.info("Topology: %s", self.project.inputs.topology)
            logger.info(
                "Trajectory: %s",
                self.project.inputs.trajectory if self.project.inputs.trajectory else "None",
            )
            logger.info("Total atoms: %d", len(universe.atoms))
            logger.info("Total frames: %d", len(universe.trajectory))

        dims = universe.trajectory.ts.dimensions
        if dims is None or not np.all(np.array(dims[:3]) > 0):
            warnings.append("No valid box vectors found; PBC distances may be unreliable.")
            if logger:
                logger.warning("No valid box vectors found; PBC distances may be unreliable.")

        solvent = build_solvent(universe, self.project.solvent)
        if logger:
            logger.info(
                "Solvent residues: %d | atoms_all: %d | probe_atoms: %d | probe_position: %s | index_shift: %d",
                len(solvent.residues),
                len(solvent.atoms_all),
                len(solvent.probe.atoms),
                solvent.probe.position,
                solvent.atom_index_shift,
            )
        resolved_selections = self._resolve_selections(universe)
        if logger:
            for label, selection in resolved_selections.items():
                logger.info(
                    "Selection %s: %d atoms | selection: %s",
                    label,
                    len(selection.group),
                    selection.selection_string,
                )
        context = EvaluationContext(universe=universe, solvent=solvent, selections=resolved_selections)

        min_distance_modes = {
            soz.name: _min_distance_mode(soz.root, solvent.probe.position)
            for soz in self.project.sozs
        }

        options = self.project.analysis
        frame_indices = self._frame_indices(len(universe.trajectory), options)
        total_frames = len(frame_indices)
        if total_frames == 0:
            raise ValueError("No frames selected for analysis")
        if logger:
            logger.info(
                "Frame selection: start=%s stop=%s stride=%s -> %d frames",
                options.frame_start,
                options.frame_stop if options.frame_stop is not None else "end",
                options.stride,
                total_frames,
            )
        log_every = max(1, int(total_frames / 100))
        pbc_missing_frames = 0

        soz_frame_data: Dict[str, SOZFrameData] = {}
        for soz in self.project.sozs:
            soz_frame_data[soz.name] = SOZFrameData(
                accumulator=StatsAccumulator(
                    solvent_records=solvent.record_by_resindex,
                    gap_tolerance=options.gap_tolerance,
                    frame_stride=options.stride,
                    store_ids=options.store_ids,
                    store_frame_table=self.project.outputs.write_per_frame,
                ),
                frame_times=[],
                frame_labels=[],
                min_distances=[],
            )

        bridge_frame_data: Dict[str, BridgeFrameData] = {}
        for bridge in self.project.bridges:
            bridge_frame_data[bridge.name] = BridgeFrameData(
                accumulator=StatsAccumulator(
                    solvent_records=solvent.record_by_resindex,
                    gap_tolerance=options.gap_tolerance,
                    frame_stride=options.stride,
                    store_ids=options.store_ids,
                    store_frame_table=self.project.outputs.write_per_frame,
                )
            )

        hydration_configs = self.project.residue_hydration
        hydration_counts: Dict[str, Dict[int, int]] = {cfg.name: {} for cfg in hydration_configs}
        hydration_residue_groups = {
            cfg.name: universe.select_atoms(cfg.residue_selection).residues for cfg in hydration_configs
        }
        hydration_modes: Dict[str, str] = {}
        for cfg in hydration_configs:
            if cfg.probe_mode is None:
                warnings.append(
                    f"Hydration '{cfg.name}' probe_mode not set; using all solvent atoms (legacy behavior)."
                )
                hydration_modes[cfg.name] = "all"
            else:
                hydration_modes[cfg.name] = resolve_probe_mode(
                    cfg.probe_mode, solvent.probe.position
                )

        for sample_index, frame_index in enumerate(frame_indices):
            if cancel_flag and cancel_flag():
                if logger:
                    logger.info(
                        "Cancellation requested at sample %d (frame %d).",
                        sample_index,
                        frame_index,
                    )
                break

            ts = universe.trajectory[frame_index]
            if context.pbc_box is None:
                pbc_missing_frames += 1
            frame_sets_current: Dict[str, set[int]] = {}
            for soz in self.project.sozs:
                frame_set = evaluate_node(soz.root, context)
                frame_sets_current[soz.name] = frame_set
                soz_frame_data[soz.name].frame_times.append(float(ts.time))
                soz_frame_data[soz.name].frame_labels.append(frame_index)
                soz_frame_data[soz.name].accumulator.update(
                    sample_index,
                    float(ts.time),
                    frame_set,
                    frame_label=frame_index,
                )

                if options.store_min_distances:
                    min_distances = {}
                    for selection_label, selection in resolved_selections.items():
                        min_distances[selection_label] = _min_distance_to_soz(
                            selection.group,
                            solvent,
                            frame_set,
                            context.pbc_box,
                            mode=min_distance_modes.get(soz.name, solvent.probe.position),
                        )
                    soz_frame_data[soz.name].min_distances.append(min_distances)

            for bridge in self.project.bridges:
                bridge_set = _bridge_frame_set(bridge, context, solvent)
                bridge_frame_data[bridge.name].accumulator.update(
                    sample_index,
                    float(ts.time),
                    bridge_set,
                    frame_label=frame_index,
                )

            for cfg in hydration_configs:
                soz_name = cfg.soz_name or (self.project.sozs[0].name if self.project.sozs else None)
                if not soz_name or soz_name not in soz_frame_data:
                    continue
                _update_hydration_counts(
                    cfg,
                    hydration_counts[cfg.name],
                    hydration_residue_groups[cfg.name],
                    solvent,
                    frame_sets_current.get(soz_name, set()),
                    context.pbc_box,
                    hydration_modes.get(cfg.name, solvent.probe.position),
                )

            if progress:
                progress(sample_index + 1, total_frames, f"Frame {frame_index}")
            if logger and (sample_index % log_every == 0 or sample_index + 1 == total_frames):
                if frame_sets_current:
                    counts = ", ".join(
                        f"{name}={len(frame_sets_current.get(name, set()))}"
                        for name in frame_sets_current
                    )
                else:
                    counts = "no SOZs"
                logger.info(
                    "Frame %d/%d (index=%d time=%.3f): %s",
                    sample_index + 1,
                    total_frames,
                    frame_index,
                    float(ts.time),
                    counts,
                )

        soz_results: Dict[str, SOZResult] = {}
        for soz in self.project.sozs:
            data = soz_frame_data[soz.name]
            stats = data.accumulator.finalize()
            min_distance_df = None
            if options.store_min_distances:
                min_distance_df = _min_distance_dataframe(
                    data.min_distances,
                    data.frame_times,
                    data.frame_labels,
                )
                if not stats["per_frame"].empty:
                    stats["per_frame"] = stats["per_frame"].merge(
                        min_distance_df, on=["frame", "time"], how="left"
                    )
            soz_results[soz.name] = SOZResult(
                name=soz.name,
                summary=stats["summary"],
                per_frame=stats["per_frame"],
                per_solvent=stats["per_solvent"],
                min_distance_traces=min_distance_df,
                residence_cont=stats["residence_cont"],
                residence_inter=stats["residence_inter"],
            )
            if logger:
                summary = stats["summary"]
                logger.info(
                    "SOZ %s summary: occupancy_fraction=%.4f mean_n_solvent=%.3f",
                    soz.name,
                    float(summary.get("occupancy_fraction", 0.0)),
                    float(summary.get("mean_n_solvent", 0.0)),
                )

        bridge_results: Dict[str, BridgeResult] = {}
        for bridge in self.project.bridges:
            stats = bridge_frame_data[bridge.name].accumulator.finalize()
            bridge_results[bridge.name] = BridgeResult(
                name=bridge.name,
                per_frame=stats["per_frame"],
                per_solvent=stats["per_solvent"],
            )
            if logger:
                logger.info("Bridge %s: %d frames", bridge.name, len(stats["per_frame"]))

        hydration_results: Dict[str, HydrationResult] = {}
        for cfg in hydration_configs:
            table = _finalize_hydration_table(
                hydration_counts[cfg.name],
                hydration_residue_groups[cfg.name],
                total_frames,
            )
            hydration_results[cfg.name] = HydrationResult(name=cfg.name, table=table)
            if logger:
                logger.info("Hydration %s: %d residues", cfg.name, len(table))

        zero_occupancy = [
            name
            for name, soz in soz_results.items()
            if float(soz.summary.get("occupancy_fraction", 0.0)) == 0.0
        ]
        if zero_occupancy:
            warnings.append("SOZs with zero occupancy: " + ", ".join(zero_occupancy))

        versions = {}
        try:
            versions = {
                "MDAnalysis": getattr(mda, "__version__", "unknown"),
                "numpy": getattr(np, "__version__", "unknown"),
                "pandas": getattr(pd, "__version__", "unknown"),
            }
        except Exception:
            versions = {}

        time_unit = None
        try:
            time_unit = universe.trajectory.units.get("time")
        except Exception:
            time_unit = None

        probe_counts = [len(v) for v in solvent.probe.resindex_to_atom_indices.values()]
        probe_summary = {
            "selection": solvent.probe.selection,
            "position": solvent.probe.position,
            "probe_atom_count": len(solvent.probe.atom_indices),
            "solvent_residue_count": len(solvent.residues),
            "residues_with_probe": sum(1 for count in probe_counts if count > 0),
            "residues_multi_probe": sum(1 for count in probe_counts if count > 1),
            "probe_atoms_per_residue_min": min(probe_counts) if probe_counts else 0,
            "probe_atoms_per_residue_max": max(probe_counts) if probe_counts else 0,
            "probe_source": getattr(self.project.solvent, "probe_source", None),
        }

        resolved_selection_info = {
            label: {
                "selection": sel.selection_string,
                "count": len(sel.group),
            }
            for label, sel in resolved_selections.items()
        }

        soz_definitions = {
            soz.name: _node_qc_entries(soz.root, solvent.probe.position)
            for soz in self.project.sozs
        }

        bridge_definitions = {
            bridge.name: {
                "selection_a": bridge.selection_a,
                "selection_b": bridge.selection_b,
                "probe_mode": resolve_probe_mode(bridge.atom_mode, solvent.probe.position),
                "cutoff_a": bridge.cutoff_a,
                "cutoff_b": bridge.cutoff_b,
                "unit": bridge.unit,
                "cutoff_a_nm": to_internal_length(bridge.cutoff_a, bridge.unit),
                "cutoff_b_nm": to_internal_length(bridge.cutoff_b, bridge.unit),
            }
            for bridge in self.project.bridges
        }

        hydration_definitions = {
            cfg.name: {
                "residue_selection": cfg.residue_selection,
                "probe_mode": hydration_modes.get(cfg.name, solvent.probe.position),
                "cutoff": cfg.cutoff,
                "unit": cfg.unit,
                "cutoff_nm": to_internal_length(cfg.cutoff, cfg.unit),
                "soz_name": cfg.soz_name,
            }
            for cfg in hydration_configs
        }

        qc_summary = {
            "preflight": preflight.to_dict(),
            "versions": versions,
            "pbc_used": pbc_missing_frames < total_frames,
            "pbc_missing_frames": pbc_missing_frames,
            "total_frames": total_frames,
            "zero_occupancy_sozs": zero_occupancy,
            "occupancy_definition": "n_solvent = solvent residues meeting the SOZ logic tree per frame",
            "time_unit": time_unit or "ps",
            "probe_definition": probe_summary,
            "resolved_selections": resolved_selection_info,
            "soz_definitions": soz_definitions,
            "bridge_definitions": bridge_definitions,
            "hydration_definitions": hydration_definitions,
        }

        return AnalysisResult(
            soz_results=soz_results,
            bridge_results=bridge_results,
            hydration_results=hydration_results,
            solvent_records=solvent.record_by_resindex,
            warnings=warnings,
            qc_summary=qc_summary,
        )


def _min_distance_to_soz(
    seed_group: mda.core.groups.AtomGroup,
    solvent: SolventUniverse,
    frame_set: set[int],
    box,
    mode: str,
) -> float:
    if not frame_set or len(seed_group) == 0:
        return float("nan")
    seed_pos = seed_group.positions
    solvent_pos, _ = solvent_positions(solvent, mode, resindices=frame_set)
    if solvent_pos.size == 0:
        return float("nan")
    dist = distances.distance_array(seed_pos, solvent_pos, box=box)
    return float(np.min(dist)) if dist.size else float("nan")


def _min_distance_dataframe(
    min_distances: List[Dict[str, float]],
    frame_times: List[float],
    frame_labels: List[int],
) -> pd.DataFrame:
    rows = []
    for idx, data in enumerate(min_distances):
        label = frame_labels[idx] if idx < len(frame_labels) else idx
        row = {"frame": label, "time": frame_times[idx]}
        row.update(data)
        rows.append(row)
    return pd.DataFrame(rows)


def _bridge_frame_set(
    bridge: BridgeConfig,
    context: EvaluationContext,
    solvent: SolventUniverse,
) -> set[int]:
    selection_a = context.selections[bridge.selection_a].group
    selection_b = context.selections[bridge.selection_b].group
    cutoff_a_nm = to_internal_length(bridge.cutoff_a, bridge.unit)
    cutoff_b_nm = to_internal_length(bridge.cutoff_b, bridge.unit)

    mode = resolve_probe_mode(bridge.atom_mode, solvent.probe.position)
    solvent_pos, atom_map = solvent_positions(solvent, mode)
    set_a = distance_resindices(
        selection_a.positions, solvent_pos, atom_map, cutoff_a_nm, context.pbc_box
    )
    set_b = distance_resindices(
        selection_b.positions, solvent_pos, atom_map, cutoff_b_nm, context.pbc_box
    )
    return set_a & set_b


def _collect_probe_modes(node: SOZNode, default_position: str) -> set[str]:
    modes: set[str] = set()
    if node.type in ("distance", "shell"):
        raw_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        modes.add(resolve_probe_mode(raw_mode, default_position))
    for child in node.children:
        modes |= _collect_probe_modes(child, default_position)
    return modes


def _min_distance_mode(node: SOZNode, default_position: str) -> str:
    modes = _collect_probe_modes(node, default_position)
    if "all" in modes:
        return "all"
    if "atom" in modes:
        return "atom"
    return resolve_probe_mode("probe", default_position)


def _node_qc_entries(node: SOZNode, default_position: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if node.type in ("distance", "shell"):
        raw_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        unit = node.params.get("unit", "A")
        entry: Dict[str, object] = {
            "type": node.type,
            "selection_label": node.params.get("selection_label")
            or node.params.get("seed_label")
            or node.params.get("seed"),
            "probe_mode": resolve_probe_mode(raw_mode, default_position),
            "unit": unit,
        }
        if node.type == "distance":
            cutoff = float(node.params.get("cutoff", 3.5))
            entry["cutoff"] = cutoff
            entry["cutoff_nm"] = to_internal_length(cutoff, unit)
        else:
            cutoffs = [float(val) for val in node.params.get("cutoffs", [3.5])]
            entry["cutoffs"] = cutoffs
            entry["cutoffs_nm"] = [to_internal_length(val, unit) for val in cutoffs]
        entries.append(entry)
    for child in node.children:
        entries.extend(_node_qc_entries(child, default_position))
    return entries


def _update_hydration_counts(
    cfg: ResidueHydrationConfig,
    counts: Dict[int, int],
    residues: mda.core.groups.ResidueGroup,
    solvent: SolventUniverse,
    frame_set: set[int],
    box,
    mode: str,
) -> None:
    if len(residues) == 0:
        return
    if not frame_set:
        return
    solvent_pos, _ = solvent_positions(solvent, mode, resindices=frame_set)
    if solvent_pos.size == 0:
        return
    cutoff_nm = to_internal_length(cfg.cutoff, cfg.unit)
    pairs = distances.capped_distance(
        residues.atoms.positions,
        solvent_pos,
        max_cutoff=cutoff_nm,
        box=box,
        return_distances=False,
    )
    if pairs is None:
        return
    if isinstance(pairs, tuple):
        pair_indices = pairs[0]
    else:
        pair_indices = pairs
    if len(pair_indices) == 0:
        return
    residue_atom_indices = pair_indices[:, 0]
    if residue_atom_indices.size:
        min_idx = int(residue_atom_indices.min())
        max_idx = int(residue_atom_indices.max())
        n_residue_atoms = len(residues.atoms)
        if min_idx >= 1 and max_idx == n_residue_atoms:
            residue_atom_indices = residue_atom_indices - 1
        if min_idx < 0 or max_idx >= n_residue_atoms:
            residue_atom_indices = residue_atom_indices[
                (residue_atom_indices >= 0) & (residue_atom_indices < n_residue_atoms)
            ]
    if residue_atom_indices.size == 0:
        return
    residue_indices: set[int] = set()
    n_residue_atoms = len(residues.atoms)
    for idx in residue_atom_indices:
        idx_i = int(idx)
        if 0 <= idx_i < n_residue_atoms:
            residue_indices.add(int(residues.atoms[idx_i].resindex))
    for resindex in residue_indices:
        counts[resindex] = counts.get(resindex, 0) + 1


def _finalize_hydration_table(
    counts: Dict[int, int],
    residues: mda.core.groups.ResidueGroup,
    total_frames: int,
) -> pd.DataFrame:
    rows = []
    for residue in residues:
        resindex = int(residue.resindex)
        count = counts.get(resindex, 0)
        rows.append(
            {
                "resindex": resindex,
                "resid": int(residue.resid),
                "resname": str(residue.resname),
                "segid": str(residue.segid) if residue.segid else "",
                "hydration_freq": count / max(total_frames, 1),
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(by=["hydration_freq", "resid"], ascending=False, inplace=True)
    return df


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
