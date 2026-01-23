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
from engine.solvent import SolventUniverse, build_solvent
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
        if len(solvent.atoms_oxygen) == len(solvent.atoms_all):
            warnings.append(
                "Solvent oxygen selection empty; using all solvent atoms for O-mode distances."
            )
            if logger:
                logger.warning("Solvent oxygen selection empty; using all solvent atoms for O-mode distances.")
        if logger:
            logger.info(
                "Solvent residues: %d | atoms_all: %d | atoms_oxygen: %d | index_shift: %d",
                len(solvent.residues),
                len(solvent.atoms_all),
                len(solvent.atoms_oxygen),
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

        soz_use_all_atoms = {
            soz.name: _soz_uses_all_atoms(soz.root) for soz in self.project.sozs
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
                            use_oxygen=not soz_use_all_atoms.get(soz.name, False),
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

        qc_summary = {
            "preflight": preflight.to_dict(),
            "versions": versions,
            "pbc_used": pbc_missing_frames < total_frames,
            "pbc_missing_frames": pbc_missing_frames,
            "total_frames": total_frames,
            "zero_occupancy_sozs": zero_occupancy,
            "occupancy_definition": "n_solvent = solvent residues meeting the SOZ logic tree per frame",
            "time_unit": time_unit or "ps",
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
    use_oxygen: bool = True,
) -> float:
    if not frame_set or len(seed_group) == 0:
        return float("nan")
    if use_oxygen:
        if len(solvent.atoms_oxygen) == 0:
            return float("nan")
        resindices = np.asarray(solvent.atom_to_resindex_oxygen, dtype=int)
        if resindices.size == 0:
            return float("nan")
        mask = np.isin(resindices, list(frame_set))
        if not np.any(mask):
            return float("nan")
        solvent_atoms = solvent.atoms_oxygen[mask]
    else:
        atom_indices = []
        for resindex in sorted(frame_set):
            atom_indices.extend(solvent.record_by_resindex[resindex].atom_indices)
        if atom_indices:
            atom_indices = [idx for idx in atom_indices if 0 <= idx < solvent.n_atoms]
        solvent_atoms = solvent.atoms_all.universe.atoms[atom_indices]
    dist = distances.distance_array(seed_group.positions, solvent_atoms.positions, box=box)
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

    if bridge.atom_mode.lower() == "all":
        solvent_atoms = solvent.atoms_all
        atom_map = solvent.atom_to_resindex_all
    else:
        solvent_atoms = solvent.atoms_oxygen
        atom_map = solvent.atom_to_resindex_oxygen

    set_a = _distance_resindices(
        selection_a, solvent_atoms, atom_map, cutoff_a_nm, context.pbc_box
    )
    set_b = _distance_resindices(
        selection_b, solvent_atoms, atom_map, cutoff_b_nm, context.pbc_box
    )
    return set_a & set_b


def _distance_resindices(
    seed_group: mda.core.groups.AtomGroup,
    solvent_atoms: mda.core.groups.AtomGroup,
    atom_map: list[int],
    cutoff_nm: float,
    box,
) -> set[int]:
    if len(seed_group) == 0 or len(solvent_atoms) == 0:
        return set()
    try:
        pairs = distances.capped_distance(
            seed_group.positions,
            solvent_atoms.positions,
            max_cutoff=cutoff_nm,
            box=box,
            return_distances=False,
        )
        if pairs is None:
            return set()
        if isinstance(pairs, tuple):
            pair_indices = pairs[0]
        else:
            pair_indices = pairs
        if len(pair_indices) == 0:
            return set()
        solvent_atom_indices = pair_indices[:, 1]
        if solvent_atom_indices.size:
            min_idx = int(solvent_atom_indices.min())
            max_idx = int(solvent_atom_indices.max())
            if min_idx >= 1 and max_idx == len(atom_map):
                solvent_atom_indices = solvent_atom_indices - 1
            if min_idx < 0 or max_idx >= len(atom_map):
                solvent_atom_indices = solvent_atom_indices[
                    (solvent_atom_indices >= 0) & (solvent_atom_indices < len(atom_map))
                ]
        if solvent_atom_indices.size == 0:
            return set()
        resindices: set[int] = set()
        atom_map_len = len(atom_map)
        for idx in solvent_atom_indices:
            idx_i = int(idx)
            if 0 <= idx_i < atom_map_len:
                resindices.add(atom_map[idx_i])
        return resindices
    except Exception:
        dist = distances.distance_array(
            seed_group.positions,
            solvent_atoms.positions,
            box=box,
        )
        if dist.size == 0:
            return set()
        min_dist = dist.min(axis=0)
        solvent_atom_indices = np.where(min_dist <= cutoff_nm)[0]
        if solvent_atom_indices.size == 0:
            return set()
        resindices: set[int] = set()
        atom_map_len = len(atom_map)
        for idx in solvent_atom_indices:
            idx_i = int(idx)
            if 0 <= idx_i < atom_map_len:
                resindices.add(atom_map[idx_i])
        return resindices


def _soz_uses_all_atoms(node: SOZNode) -> bool:
    if node.type in ("distance", "shell"):
        atom_mode = str(node.params.get("atom_mode", "O")).lower()
        return atom_mode == "all"
    return any(_soz_uses_all_atoms(child) for child in node.children)


def _update_hydration_counts(
    cfg: ResidueHydrationConfig,
    counts: Dict[int, int],
    residues: mda.core.groups.ResidueGroup,
    solvent: SolventUniverse,
    frame_set: set[int],
    box,
) -> None:
    if len(residues) == 0:
        return
    if not frame_set:
        return
    atom_indices = []
    for resindex in sorted(frame_set):
        atom_indices.extend(solvent.record_by_resindex[resindex].atom_indices)
    if atom_indices:
        atom_indices = [idx for idx in atom_indices if 0 <= idx < solvent.n_atoms]
    solvent_atoms = solvent.atoms_all.universe.atoms[atom_indices]
    if len(solvent_atoms) == 0:
        return
    cutoff_nm = to_internal_length(cfg.cutoff, cfg.unit)
    pairs = distances.capped_distance(
        residues.atoms.positions,
        solvent_atoms.positions,
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
