"""Main analysis engine for SOZLab."""
from __future__ import annotations

import json
import os
import logging
import inspect
import warnings
import concurrent.futures
import multiprocessing
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances

from engine.models import (
    AnalysisOptions,

    DistanceBridgeConfig,
    HbondWaterBridgeConfig,
    HbondHydrationConfig,
    DensityMapConfig,
    WaterDynamicsConfig,
    InputConfig,
    SolventConfig,
    ProjectConfig,
    SOZNode,
)
from engine.resolver import resolve_selection, sanitize_selection_string
from engine.solvent import (
    SolventUniverse,
    build_solvent,
    distance_resindices,
    resolve_probe_mode,
    solvent_positions,
)
from engine.soz_eval import EvaluationContext, evaluate_node
from engine.stats import StatsAccumulator, compute_residence_lengths
from engine.units import to_internal_length
from engine.preflight import run_preflight

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in practice
    yaml = None


ProgressCallback = Callable[[int, int, str], None]
PROGRESS_RESERVE_UNITS = 2


def _merge_hydration_results(results: List[HydrationResult]) -> HydrationResult:
    if not results:
        raise ValueError("Cannot merge empty result list")
    
    base = results[0]
    if len(results) == 1:
        return base
    
    frame_times = []
    frame_labels = []
    contact_frames_total = {}
    contact_frames_given_soz = {}
    
    merged_data = {} 
    
    soz_populated_frames = 0
    total_frames = 0
    
    for res in results:
        total_frames += len(res.frame_labels)
        frame_times.extend(res.frame_times)
        frame_labels.extend(res.frame_labels)
        
        for r_idx, frames in res.contact_frames_total.items():
            contact_frames_total.setdefault(r_idx, []).extend(frames)
        for r_idx, frames in res.contact_frames_given_soz.items():
            contact_frames_given_soz.setdefault(r_idx, []).extend(frames)

        if not res.table.empty:
            if soz_populated_frames == 0:
                 col = "frames_where_soz_populated"
                 if col in res.table.columns:
                     soz_populated_frames = int(res.table.iloc[0][col])

            for _, row in res.table.iterrows():
                ridx = int(row["resindex"])
                if ridx not in merged_data:
                    merged_data[ridx] = {
                        "resindex": ridx,
                        "resid": int(row["resid"]),
                        "resname": str(row["resname"]),
                        "segid": str(row.get("segid", "")),
                        "frames_with_contact_total": 0.0,
                        "frames_with_contact_given_soz": 0.0,
                    }
                merged_data[ridx]["frames_with_contact_total"] += float(row["frames_with_contact_total"])
                merged_data[ridx]["frames_with_contact_given_soz"] += float(row["frames_with_contact_given_soz"])

    for frames in contact_frames_total.values():
        frames.sort()
    for frames in contact_frames_given_soz.values():
        frames.sort()
        
    total_max = max(total_frames, 1)
    soz_norm = max(soz_populated_frames, 1)
    
    new_rows = []
    for ridx, data in merged_data.items():
        row = data.copy()
        c_total = data["frames_with_contact_total"]
        c_soz = data["frames_with_contact_given_soz"]
        
        row["frames_with_contact_total"] = int(c_total)
        row["frames_with_contact_given_soz"] = int(c_soz)
        row["frames_where_soz_populated"] = int(soz_populated_frames)
        row["freq_total"] = c_total / total_max
        row["freq_given_soz"] = c_soz / soz_norm
        row["soz_populated_freq"] = soz_populated_frames / total_max
        new_rows.append(row)
            
    new_table = pd.DataFrame(new_rows)
    if not new_table.empty:
        new_table.sort_values(by=["freq_total", "resid"], ascending=False, inplace=True)
        
    return HydrationResult(
        name=base.name,
        table=new_table,
        frame_times=frame_times,
        frame_labels=frame_labels,
        contact_frames_total=contact_frames_total,
        contact_frames_given_soz=contact_frames_given_soz,
        mode=base.mode,
    )


def resolve_worker_count(workers: int | None) -> int:
    try:
        cpu_total = int(os.cpu_count() or 1)
    except Exception:
        cpu_total = 1
    if workers is None:
        return max(1, cpu_total - 1)
    try:
        workers_i = int(workers)
    except (TypeError, ValueError):
        return max(1, cpu_total - 1)
    if workers_i <= 0:
        return max(1, cpu_total - 1)
    return max(1, min(workers_i, cpu_total))


def _load_universe_from_inputs(inputs: InputConfig) -> mda.Universe:
    if inputs.trajectory:
        return mda.Universe(inputs.topology, inputs.trajectory)
    return mda.Universe(inputs.topology)


def _prepare_worker_context(
    inputs_data: InputConfig | Dict[str, object],
    solvent_data: SolventConfig | Dict[str, object],
) -> tuple[mda.Universe, SolventUniverse]:
    if isinstance(inputs_data, InputConfig):
        inputs = inputs_data
    else:
        inputs = InputConfig.from_dict(inputs_data)
    if isinstance(solvent_data, SolventConfig):
        solvent_cfg = solvent_data
    else:
        solvent_cfg = SolventConfig.from_dict(solvent_data)
    universe = _load_universe_from_inputs(inputs)
    solvent = build_solvent(universe, solvent_cfg)
    return universe, solvent


def _chunk_configs(configs: List[object], n_chunks: int) -> List[List[object]]:
    if n_chunks <= 1 or len(configs) <= 1:
        return [configs]
    n_chunks = max(1, min(n_chunks, len(configs)))
    chunk_size = max(1, int(np.ceil(len(configs) / n_chunks)))
    return [configs[i : i + chunk_size] for i in range(0, len(configs), chunk_size)]


def _run_hbond_bridge_chunk(
    inputs_data: Dict[str, object],
    solvent_data: Dict[str, object],
    bridge_configs_data: List[Dict[str, object]],
    frame_indices: List[int],
    frame_times: List[float],
    frame_index_map: Dict[int, int],
    options_data: Dict[str, object],
    water_resnames: List[str],
    store_frame_table: bool,
    selection_lookup: Dict[str, str] | None,
) -> tuple[Dict[str, BridgeResult], List[str]]:
    universe, solvent = _prepare_worker_context(inputs_data, solvent_data)
    configs = [HbondWaterBridgeConfig.from_dict(cfg) for cfg in bridge_configs_data]
    options = AnalysisOptions.from_dict(options_data)
    return _run_hbond_water_bridges(
        universe=universe,
        solvent=solvent,
        bridge_configs=configs,
        frame_indices=frame_indices,
        frame_times=frame_times,
        frame_index_map=frame_index_map,
        options=options,
        water_resnames=water_resnames,
        store_frame_table=store_frame_table,
        selection_lookup=selection_lookup,
        logger=None,
    )


def _run_hbond_hydration_chunk(
    inputs_data: Dict[str, object],
    solvent_data: Dict[str, object],
    hydration_configs_data: List[Dict[str, object]],
    frame_indices: List[int],
    frame_times: List[float],
    frame_index_map: Dict[int, int],
    soz_frames_for_hbond: Dict[str, List[set[int]]],
    total_frames: int,
    options_data: Dict[str, object],
    water_resnames: List[str],
) -> tuple[Dict[str, HydrationResult], List[str]]:
    universe, solvent = _prepare_worker_context(inputs_data, solvent_data)
    configs = [HbondHydrationConfig.from_dict(cfg) for cfg in hydration_configs_data]
    options = AnalysisOptions.from_dict(options_data)
    return _run_hbond_hydration(
        universe=universe,
        solvent=solvent,
        configs=configs,
        frame_indices=frame_indices,
        frame_times=frame_times,
        frame_index_map=frame_index_map,
        soz_frames_for_hbond=soz_frames_for_hbond,
        total_frames=total_frames,
        options=options,
        water_resnames=water_resnames,
        logger=None,
    )


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
    residence_cont: Dict[int, List[float]]
    residence_inter: Dict[int, List[float]]


@dataclass
class BridgeResult:
    name: str
    per_frame: pd.DataFrame
    per_solvent: pd.DataFrame
    bridge_type: str
    summary: Dict[str, object]
    residence_cont: Dict[int, List[int]]
    residence_inter: Dict[int, List[int]]
    edge_list: Optional[pd.DataFrame] = None


@dataclass
class DensityMapResult:
    name: str
    grid_path: str
    grid: Optional[np.ndarray]
    axes: Dict[str, np.ndarray]
    slices: Dict[str, np.ndarray]
    metadata: Dict[str, object]


@dataclass
class WaterDynamicsResult:
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
    distance_bridge_results: Dict[str, BridgeResult]
    hbond_bridge_results: Dict[str, BridgeResult]
    hbond_hydration_results: Dict[str, HydrationResult]
    density_results: Dict[str, DensityMapResult]
    water_dynamics_results: Dict[str, WaterDynamicsResult]
    solvent_records: Dict[int, object]
    warnings: List[str]
    qc_summary: Dict[str, object]


class SOZAnalysisEngine:
    def __init__(self, project: ProjectConfig):
        self.project = project

    def _load_universe(self) -> mda.Universe:
        if self.project.inputs.trajectory:
            u = mda.Universe(self.project.inputs.topology, self.project.inputs.trajectory)
        else:
            u = mda.Universe(self.project.inputs.topology)
        
        if not hasattr(u.atoms, "charges"):
            try:
                from MDAnalysis.core.topologyattrs import Charges
                u.add_TopologyAttr(Charges(np.zeros(len(u.atoms))))
                logging.getLogger("sozlab").info("Added dummy charges for H-bond analysis compatibility.")
            except Exception as exc:
                logging.getLogger("sozlab").warning("Failed to add dummy charges: %s", exc)
        
        # Bond guessing is slow for 140k atoms. We'll skip global guessing 
        # and handle it in analysis modules for relevant subsets if needed.
        # However, H-bond analysis needs them. 
        # We'll guess bonds only for non-water atoms first, and handle water separately.
        if not hasattr(u.atoms, "bonds") or len(u.atoms.bonds) == 0:
             try:
                 # Guessing for everything is slow. Guessing for protein + ligands is fast.
                 # Water bonds are usually standard (OH1, OH2).
                 non_solvent = u.select_atoms("not (resname TIP3 or resname HOH or resname SOL)")
                 if len(non_solvent) > 0:
                     non_solvent.guess_bonds()
                     # Also try to guess elements to help HBA
                     try:
                         non_solvent.guess_elements()
                     except Exception: pass
                     logging.getLogger("sozlab").info("Guessed bonds/elements for non-solvent atoms (%d atoms).", len(non_solvent))
             except Exception as exc:
                 logging.getLogger("sozlab").warning("Failed to guess bonds for non-solvent: %s", exc)
        return u

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
        resolved_selection_strings = {
            label: selection.selection_string for label, selection in resolved_selections.items()
        }
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
        workers = resolve_worker_count(options.workers)
        if logger:
            logger.info(
                "Workers: %d (requested=%s)", workers, options.workers if options.workers is not None else "auto"
            )
        frame_indices = self._frame_indices(len(universe.trajectory), options)
        total_frames = len(frame_indices)
        if total_frames == 0:
            raise ValueError("No frames selected for analysis")
        progress_total = None
        progress_units = 0
        if progress:
            optional_units = 0
            if self.project.hbond_water_bridges:
                optional_units += 1
            if self.project.hbond_hydration:
                optional_units += 1
            if self.project.density_maps:
                optional_units += 1
            if self.project.water_dynamics:
                optional_units += 1
            analysis_units = 1 + 1 + 1 + total_frames + 1 + optional_units + 1
            progress_total = analysis_units + PROGRESS_RESERVE_UNITS

            def _emit_progress(message: str, current_override: int | None = None) -> None:
                if progress_total is None:
                    return
                current_val = progress_units if current_override is None else current_override
                progress(current_val, progress_total, message)

            progress_units += 1
            _emit_progress("Loaded inputs")
            progress_units += 1
            _emit_progress("Validated inputs")
            progress_units += 1
            _emit_progress("Compiled selections")
        else:
            _emit_progress = None
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

        distance_bridge_frame_data: Dict[str, BridgeFrameData] = {}
        for bridge in self.project.distance_bridges:
            distance_bridge_frame_data[bridge.name] = BridgeFrameData(
                accumulator=StatsAccumulator(
                    solvent_records=solvent.record_by_resindex,
                    gap_tolerance=options.gap_tolerance,
                    frame_stride=options.stride,
                    store_ids=options.store_ids,
                    store_frame_table=self.project.outputs.write_per_frame,
                )
            )


        hbond_hydration_configs = self.project.hbond_hydration
        hbond_hydration_frames_total: Dict[str, Dict[int, List[int]]] = {
            cfg.name: {} for cfg in hbond_hydration_configs
        }
        hbond_hydration_frames_soz: Dict[str, Dict[int, List[int]]] = {
            cfg.name: {} for cfg in hbond_hydration_configs
        }
        hbond_hydration_counts_total: Dict[str, Dict[int, int]] = {
            cfg.name: {} for cfg in hbond_hydration_configs
        }
        hbond_hydration_counts_soz: Dict[str, Dict[int, int]] = {
            cfg.name: {} for cfg in hbond_hydration_configs
        }
        hbond_hydration_soz_populated: Dict[str, int] = {
            cfg.name: 0 for cfg in hbond_hydration_configs
        }
        hbond_hydration_residue_groups = {
            cfg.name: universe.select_atoms(cfg.residue_selection).residues
            for cfg in hbond_hydration_configs
        }

        frame_times: List[float] = []
        frame_labels: List[int] = []
        frame_index_map: Dict[int, int] = {}

        soz_frames_for_hbond: Dict[str, List[set[int]]] = {}
        needed_soz_for_hbond = {
            (cfg.soz_name or (self.project.sozs[0].name if self.project.sozs else None))
            for cfg in hbond_hydration_configs
        }
        for name in needed_soz_for_hbond:
            if name:
                soz_frames_for_hbond[name] = []

        water_dynamics_selection_configs = [
            cfg for cfg in self.project.water_dynamics if cfg.region_mode == "selection"
        ]
        water_dynamics_soz_configs = [
            cfg for cfg in self.project.water_dynamics if cfg.region_mode == "soz"
        ]
        
        all_wd_configs = water_dynamics_selection_configs + water_dynamics_soz_configs
        water_dynamics_frame_sets: Dict[str, List[set[int]]] = {
            cfg.name: [] for cfg in all_wd_configs
        }
        water_dynamics_region_groups: Dict[str, mda.core.groups.AtomGroup] = {
            cfg.name: universe.select_atoms(cfg.region_selection or "") for cfg in water_dynamics_selection_configs
        }
        # Pre-select solute groups for SOZ cutoff filtering
        water_dynamics_solute_groups: Dict[str, mda.core.groups.AtomGroup] = {}
        for cfg in water_dynamics_soz_configs:
            if cfg.region_cutoff > 0 and cfg.solute_selection:
                try:
                    water_dynamics_solute_groups[cfg.name] = universe.select_atoms(cfg.solute_selection)
                except Exception:
                    pass

        frame_base = progress_units
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
            frame_times.append(float(ts.time))
            frame_labels.append(frame_index)
            frame_index_map[frame_index] = sample_index
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
                if soz.name in soz_frames_for_hbond:
                    soz_frames_for_hbond[soz.name].append(set(frame_set))

            for bridge in self.project.distance_bridges:
                bridge_set = _bridge_frame_set(bridge, context, solvent)
                distance_bridge_frame_data[bridge.name].accumulator.update(
                    sample_index,
                    float(ts.time),
                    bridge_set,
                    frame_label=frame_index,
                )


            for cfg in water_dynamics_selection_configs:
                region_group = water_dynamics_region_groups.get(cfg.name)
                if region_group is None or len(region_group) == 0:
                    water_dynamics_frame_sets[cfg.name].append(set())
                    continue
                mode = resolve_probe_mode(cfg.region_probe_mode, solvent.probe.position)
                cutoff_nm = to_internal_length(cfg.region_cutoff, cfg.region_unit)
                solvent_pos, atom_map = solvent_positions(solvent, mode)
                region_set = distance_resindices(
                    region_group.positions,
                    solvent_pos,
                    atom_map,
                    cutoff_nm,
                    context.pbc_box,
                )
                water_dynamics_frame_sets[cfg.name].append(region_set)

            for cfg in water_dynamics_soz_configs:
                soz_set = frame_sets_current.get(cfg.soz_name or "", set())
                final_set = soz_set
                
                # Apply optional distance cutoff (intersect SOZ with shell around solute)
                if cfg.region_cutoff > 0 and cfg.name in water_dynamics_solute_groups:
                    solute_group = water_dynamics_solute_groups[cfg.name]
                    if len(solute_group) > 0:
                        mode = resolve_probe_mode(cfg.region_probe_mode, solvent.probe.position)
                        cutoff_nm = to_internal_length(cfg.region_cutoff, cfg.region_unit)
                        solvent_pos, atom_map = solvent_positions(solvent, mode)
                        shell_set = distance_resindices(
                            solute_group.positions,
                            solvent_pos,
                            atom_map,
                            cutoff_nm,
                            context.pbc_box,
                        )
                        final_set = soz_set & shell_set
                
                water_dynamics_frame_sets[cfg.name].append(final_set)

            if progress_total is not None:
                progress(frame_base + sample_index + 1, progress_total, f"Per-frame SOZ evaluation (frame {frame_index})")
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

        progress_units = frame_base + total_frames
        if progress_total is not None and _emit_progress is not None:
            _emit_progress("Per-frame evaluation complete", progress_units)

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

        distance_bridge_results: Dict[str, BridgeResult] = {}
        for bridge in self.project.distance_bridges:
            stats = distance_bridge_frame_data[bridge.name].accumulator.finalize()
            distance_bridge_results[bridge.name] = BridgeResult(
                name=bridge.name,
                per_frame=stats["per_frame"],
                per_solvent=stats["per_solvent"],
                bridge_type="distance",
                summary=stats["summary"],
                residence_cont=stats["residence_cont"],
                residence_inter=stats["residence_inter"],
            )
            if logger:
                logger.info(
                    "Distance bridge %s: %d frames", bridge.name, len(stats["per_frame"])
                )

        if progress_total is not None and _emit_progress is not None:
            progress_units += 1
            _emit_progress("Derived metrics")

        inputs_payload = self.project.inputs.to_dict()
        solvent_payload = self.project.solvent.to_dict()
        options_payload = options.to_dict()
        water_resnames = list(self.project.solvent.water_resnames)

        hbond_bridge_configs = self.project.hbond_water_bridges
        if hbond_bridge_configs and logger:
            logger.info("Starting H-bond water bridge analysis...")
        if hbond_bridge_configs and workers > 1 and len(hbond_bridge_configs) > 1:
            hbond_bridge_results = {}
            hbond_bridge_warnings = []
            chunks = _chunk_configs([cfg.to_dict() for cfg in hbond_bridge_configs], workers)
            try:
                max_workers = min(workers, len(chunks))
                ctx = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    futures = [
                        executor.submit(
                            _run_hbond_bridge_chunk,
                            inputs_payload,
                            solvent_payload,
                            chunk,
                            frame_labels,
                            frame_times,
                            frame_index_map,
                            options_payload,
                            water_resnames,
                            self.project.outputs.write_per_frame,
                            resolved_selection_strings,
                        )
                        for chunk in chunks
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        results_chunk, warnings_chunk = future.result()
                        hbond_bridge_results.update(results_chunk)
                        if warnings_chunk:
                            hbond_bridge_warnings.extend(warnings_chunk)
                hbond_bridge_results = {
                    cfg.name: hbond_bridge_results[cfg.name]
                    for cfg in hbond_bridge_configs
                    if cfg.name in hbond_bridge_results
                }
            except Exception as exc:
                msg = f"H-bond water bridge parallel execution failed; running sequentially. ({exc})"
                warnings.append(msg)
                if logger:
                    logger.warning(msg)
                hbond_bridge_results, hbond_bridge_warnings = _run_hbond_water_bridges(
                    universe=universe,
                    solvent=solvent,
                    bridge_configs=hbond_bridge_configs,
                    frame_indices=frame_labels,
                    frame_times=frame_times,
                    frame_index_map=frame_index_map,
                    options=options,
                    water_resnames=water_resnames,
                    store_frame_table=self.project.outputs.write_per_frame,
                    selection_lookup=resolved_selection_strings,
                    logger=logger,
                )
        else:
            hbond_bridge_results, hbond_bridge_warnings = _run_hbond_water_bridges(
                universe=universe,
                solvent=solvent,
                bridge_configs=hbond_bridge_configs,
                frame_indices=frame_labels,
                frame_times=frame_times,
                frame_index_map=frame_index_map,
                options=options,
                water_resnames=water_resnames,
                store_frame_table=self.project.outputs.write_per_frame,
                selection_lookup=resolved_selection_strings,
                logger=logger,
            )
        if hbond_bridge_warnings:
            warnings.extend(hbond_bridge_warnings)
        if progress_total is not None and _emit_progress is not None and self.project.hbond_water_bridges:
            progress_units += 1
            _emit_progress("H-bond water bridges")


        if hbond_hydration_configs and logger:
            logger.info("Starting H-bond hydration analysis...")
        if hbond_hydration_configs and workers > 1 and len(hbond_hydration_configs) > 1:
            hbond_hydration_results = {}
            hbond_hydration_warnings = []
            chunks = _chunk_configs([cfg.to_dict() for cfg in hbond_hydration_configs], workers)
            try:
                max_workers = min(workers, len(chunks))
                ctx = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    futures = [
                        executor.submit(
                            _run_hbond_hydration_chunk,
                            inputs_payload,
                            solvent_payload,
                            chunk,
                            frame_labels,
                            frame_times,
                            frame_index_map,
                            soz_frames_for_hbond,
                            total_frames, # Pass global total frames, might affect freq in partials but we fix in merge
                            options_payload,
                            water_resnames,
                        )
                        for chunk in chunks
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        results_chunk, warnings_chunk = future.result()
                        hbond_hydration_results.update(results_chunk)
                        if warnings_chunk:
                            hbond_hydration_warnings.extend(warnings_chunk)
                hbond_hydration_results = {
                    cfg.name: hbond_hydration_results[cfg.name]
                    for cfg in hbond_hydration_configs
                    if cfg.name in hbond_hydration_results
                }
            except Exception as exc:
                msg = f"H-bond hydration parallel execution failed; running sequentially. ({exc})"
                warnings.append(msg)
                if logger:
                    logger.warning(msg)
                hbond_hydration_results, hbond_hydration_warnings = _run_hbond_hydration(
                    universe=universe,
                    solvent=solvent,
                    configs=hbond_hydration_configs,
                    frame_indices=frame_labels,
                    frame_times=frame_times,
                    frame_index_map=frame_index_map,
                    soz_frames_for_hbond=soz_frames_for_hbond,
                    total_frames=total_frames,
                    options=options,
                    water_resnames=water_resnames,
                    logger=logger,
                )
        elif hbond_hydration_configs and workers > 1 and len(hbond_hydration_configs) == 1:
            # Frame-based parallelization for single config
            try:
                frame_chunks = _chunk_configs(list(range(len(frame_labels))), workers)
                chunk_defs = []
                for i, idx_chunk in enumerate(frame_chunks):
                    sub_indices = [frame_labels[i] for i in idx_chunk]
                    sub_times = [frame_times[i] for i in idx_chunk]
                    # Note: we pass the full config list (len=1) to each
                    chunk_defs.append((sub_indices, sub_times))
                
                partial_results_list = []
                hbond_hydration_warnings = []
                
                ctx = multiprocessing.get_context("spawn")
                max_workers = min(workers, len(chunk_defs))
                
                if logger:
                    logger.info("Parallelizing single hydration config across %d workers", max_workers)

                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    futures = [
                        executor.submit(
                            _run_hbond_hydration_chunk,
                            inputs_payload,
                            solvent_payload,
                            [hbond_hydration_configs[0].to_dict()],
                            c_indices,
                            c_times,
                            frame_index_map,
                            soz_frames_for_hbond,
                            len(c_indices), # Pass chunk size as total frames for correct partial freq? No, but merge will fix
                            options_payload,
                            water_resnames,
                        )
                        for c_indices, c_times in chunk_defs
                    ]
                    
                    for future in futures:
                        p_res, p_warn = future.result()
                        if p_res:
                            partial_results_list.append(next(iter(p_res.values())))
                        if p_warn:
                            hbond_hydration_warnings.extend(p_warn)
                            
                merged = _merge_hydration_results(partial_results_list)
                hbond_hydration_results = {merged.name: merged}
            except Exception as exc:
                msg = f"H-bond hydration frame parallelization failed; running sequentially. ({exc})"
                warnings.append(msg)
                if logger:
                    logger.warning(msg)
                hbond_hydration_results, hbond_hydration_warnings = _run_hbond_hydration(
                    universe=universe,
                    solvent=solvent,
                    configs=hbond_hydration_configs,
                    frame_indices=frame_labels,
                    frame_times=frame_times,
                    frame_index_map=frame_index_map,
                    soz_frames_for_hbond=soz_frames_for_hbond,
                    total_frames=total_frames,
                    options=options,
                    water_resnames=water_resnames,
                    logger=logger,
                )
        else:
            hbond_hydration_results, hbond_hydration_warnings = _run_hbond_hydration(
                universe=universe,
                solvent=solvent,
                configs=hbond_hydration_configs,
                frame_indices=frame_labels,
                frame_times=frame_times,
                frame_index_map=frame_index_map,
                soz_frames_for_hbond=soz_frames_for_hbond,
                total_frames=total_frames,
                options=options,
                water_resnames=water_resnames,
                logger=logger,
            )
        if hbond_hydration_warnings:
            warnings.extend(hbond_hydration_warnings)
        if progress_total is not None and _emit_progress is not None and self.project.hbond_hydration:
            progress_units += 1
            _emit_progress("H-bond hydration")

        density_results = _run_density_maps(
            universe=universe,
            configs=self.project.density_maps,
            options=options,
            output_dir=self.project.outputs.output_dir,
            progress=None,
            logger=logger,
        )
        if progress_total is not None and _emit_progress is not None and self.project.density_maps:
            progress_units += 1
            _emit_progress("Density maps")

        water_dynamics_results = _run_water_dynamics(
            universe=universe,
            solvent=solvent,
            configs=self.project.water_dynamics,
            soz_results=soz_results,
            frame_times=frame_times,
            frame_labels=frame_labels,
            frame_index_map=frame_index_map,
            selection_frame_sets=water_dynamics_frame_sets,
            options=options,
            water_resnames=water_resnames,
            logger=logger,
        )
        if progress_total is not None and _emit_progress is not None and self.project.water_dynamics:
            progress_units += 1
            _emit_progress("Water dynamics")

        zero_occupancy = [
            name
            for name, soz in soz_results.items()
            if float(soz.summary.get("occupancy_fraction", 0.0)) == 0.0
        ]
        zero_diagnostics: Dict[str, List[str]] = {}
        if zero_occupancy:
            warnings.append("SOZs with zero occupancy: " + ", ".join(zero_occupancy))
            soz_lookup = {soz.name: soz for soz in self.project.sozs}
            sample_frames = list(frame_indices[: min(50, total_frames)])
            for name in zero_occupancy:
                soz_def = soz_lookup.get(name)
                if not soz_def:
                    continue
                diagnostics = _diagnose_zero_occupancy(
                    soz_def,
                    context,
                    sample_frames,
                    sample_limit=min(50, len(sample_frames)),
                )
                if diagnostics:
                    zero_diagnostics[name] = diagnostics
                    warnings.extend(diagnostics)

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

        distance_bridge_definitions = {
            bridge.name: {
                "selection_a": bridge.selection_a,
                "selection_b": bridge.selection_b,
                "probe_mode": resolve_probe_mode(bridge.atom_mode, solvent.probe.position),
                "cutoff_a": bridge.cutoff_a,
                "cutoff_b": bridge.cutoff_b,
                "unit": bridge.unit,
                "cutoff_a_nm": to_internal_length(bridge.cutoff_a, bridge.unit),
                "cutoff_b_nm": to_internal_length(bridge.cutoff_b, bridge.unit),
                "type": "distance_bridge",
            }
            for bridge in self.project.distance_bridges
        }

        hbond_bridge_definitions = {
            bridge.name: {
                "selection_a": bridge.selection_a,
                "selection_b": bridge.selection_b,
                "distance": bridge.distance,
                "angle": bridge.angle,
                "water_selection": bridge.water_selection,
                "donors_selection": bridge.donors_selection,
                "hydrogens_selection": bridge.hydrogens_selection,
                "acceptors_selection": bridge.acceptors_selection,
                "update_selections": bridge.update_selections,
                "type": "hbond_water_bridge",
            }
            for bridge in self.project.hbond_water_bridges
        }



        hbond_hydration_definitions = {
            cfg.name: {
                "residue_selection": cfg.residue_selection,
                "distance": cfg.distance,
                "angle": cfg.angle,
                "conditioning": cfg.conditioning,
                "soz_name": cfg.soz_name,
                "water_selection": cfg.water_selection,
                "donors_selection": cfg.donors_selection,
                "hydrogens_selection": cfg.hydrogens_selection,
                "acceptors_selection": cfg.acceptors_selection,
                "update_selections": cfg.update_selections,
                "type": "hbond_hydration",
            }
            for cfg in hbond_hydration_configs
        }

        density_definitions = {
            cfg.name: cfg.to_dict() for cfg in self.project.density_maps
        }

        water_dynamics_definitions = {
            cfg.name: cfg.to_dict() for cfg in self.project.water_dynamics
        }

        qc_summary = {
            "preflight": preflight.to_dict(),
            "versions": versions,
            "analysis_warnings": warnings,
            "workers": workers,
            "pbc_used": pbc_missing_frames < total_frames,
            "pbc_missing_frames": pbc_missing_frames,
            "total_frames": total_frames,
            "zero_occupancy_sozs": zero_occupancy,
            "zero_occupancy_diagnostics": zero_diagnostics,
            "occupancy_definition": "n_solvent = solvent residues meeting the SOZ logic tree per frame",
            "time_unit": time_unit or "ps",
            "probe_definition": probe_summary,
            "resolved_selections": resolved_selection_info,
            "soz_definitions": soz_definitions,
            "distance_bridge_definitions": distance_bridge_definitions,
            "hbond_bridge_definitions": hbond_bridge_definitions,

            "hbond_hydration_definitions": hbond_hydration_definitions,
            "density_definitions": density_definitions,
            "water_dynamics_definitions": water_dynamics_definitions,
        }

        if progress_total is not None and _emit_progress is not None:
            progress_units += 1
            _emit_progress("Finalizing analysis")

        return AnalysisResult(
            soz_results=soz_results,
            distance_bridge_results=distance_bridge_results,
            hbond_bridge_results=hbond_bridge_results,
            hbond_hydration_results=hbond_hydration_results,
            density_results=density_results,
            water_dynamics_results=water_dynamics_results,
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
    bridge: DistanceBridgeConfig,
    context: EvaluationContext,
    solvent: SolventUniverse,
) -> set[int]:
    sel_a = context.selections.get(bridge.selection_a)
    sel_b = context.selections.get(bridge.selection_b)
    if sel_a is None or sel_b is None:
        return set()
    selection_a = sel_a.group
    selection_b = sel_b.group
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


def _leaf_nodes(node: SOZNode) -> List[SOZNode]:
    leaves: List[SOZNode] = []
    if node.type in ("distance", "shell"):
        leaves.append(node)
    for child in node.children:
        leaves.extend(_leaf_nodes(child))
    return leaves


def _node_label(node: SOZNode) -> str:
    label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
    if node.type == "distance":
        cutoff = node.params.get("cutoff", "?")
        unit = node.params.get("unit", "A")
        return f"distance({label}, cutoff={cutoff}{unit})"
    if node.type == "shell":
        cutoffs = node.params.get("cutoffs", [])
        unit = node.params.get("unit", "A")
        return f"shell({label}, cutoffs={cutoffs}{unit})"
    return node.type


def _diagnose_zero_occupancy(
    soz: SOZDefinition,
    context: EvaluationContext,
    frame_indices: List[int],
    sample_limit: int = 50,
) -> List[str]:
    leaves = _leaf_nodes(soz.root)
    if not leaves:
        return [f"SOZ '{soz.name}' has no distance/shell nodes to evaluate."]
    sample_frames = frame_indices[: max(1, min(sample_limit, len(frame_indices)))]
    leaf_hits = {idx: 0 for idx in range(len(leaves))}
    for frame in sample_frames:
        context.universe.trajectory[frame]
        for idx, node in enumerate(leaves):
            if evaluate_node(node, context):
                leaf_hits[idx] += 1
    any_leaf = any(count > 0 for count in leaf_hits.values())
    if not any_leaf:
        return [
            f"SOZ '{soz.name}' has zero occupancy; leaf nodes are empty in sampled frames. "
            "Check selection strings and cutoffs."
        ]
    parts = []
    for idx, node in enumerate(leaves):
        parts.append(f"{_node_label(node)}: {leaf_hits[idx]}/{len(sample_frames)} frames")
    return [
        f"SOZ '{soz.name}' has zero occupancy but leaf nodes are populated in sampled frames "
        f"({'; '.join(parts[:4])}). Check combine mode (AND vs OR) or cutoffs."
    ]


def _residue_contact_indices(
    residues: mda.core.groups.ResidueGroup,
    solvent: SolventUniverse,
    box,
    mode: str,
    cutoff: float,
    unit: str,
    resindices: set[int] | None = None,
) -> set[int]:
    if len(residues) == 0:
        return set()
    solvent_pos, _ = solvent_positions(solvent, mode, resindices=resindices)
    if solvent_pos.size == 0:
        return set()
    cutoff_nm = to_internal_length(cutoff, unit)
    pairs = distances.capped_distance(
        residues.atoms.positions,
        solvent_pos,
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
        return set()
    residue_indices: set[int] = set()
    n_residue_atoms = len(residues.atoms)
    for idx in residue_atom_indices:
        idx_i = int(idx)
        if 0 <= idx_i < n_residue_atoms:
            residue_indices.add(int(residues.atoms[idx_i].resindex))
    return residue_indices


def _update_contact_counts(
    counts: Dict[int, int],
    frame_map: Dict[int, List[int]],
    residue_indices: set[int],
    sample_index: int,
) -> None:
    for resindex in residue_indices:
        counts[resindex] = counts.get(resindex, 0) + 1
        frame_map.setdefault(resindex, []).append(sample_index)


def _finalize_contact_table(
    counts_total: Dict[int, int],
    counts_soz: Dict[int, int],
    residues: mda.core.groups.ResidueGroup,
    total_frames: int,
    soz_populated_frames: int,
) -> pd.DataFrame:
    rows = []
    total_frames = max(total_frames, 1)
    soz_populated_frames = max(soz_populated_frames, 0)
    soz_norm = max(soz_populated_frames, 1)
    soz_populated_freq = soz_populated_frames / total_frames
    for residue in residues:
        resindex = int(residue.resindex)
        count_total = counts_total.get(resindex, 0)
        count_soz = counts_soz.get(resindex, 0)
        rows.append(
            {
                "resindex": resindex,
                "resid": int(residue.resid),
                "resname": str(residue.resname),
                "segid": str(residue.segid) if residue.segid else "",
                "frames_with_contact_total": int(count_total),
                "frames_with_contact_given_soz": int(count_soz),
                "frames_where_soz_populated": int(soz_populated_frames),
                "freq_total": count_total / total_frames,
                "freq_given_soz": count_soz / soz_norm,
                "soz_populated_freq": soz_populated_freq,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(by=["freq_total", "resid"], ascending=False, inplace=True)
    return df


def _filter_kwargs(callable_obj, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _run_analysis_with_frames(analysis, start: int, stop: int | None, step: int) -> List[str]:
    try:
        sig = inspect.signature(analysis.run)
    except (TypeError, ValueError):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis.run()
        return [str(w.message) for w in caught if getattr(w, "message", None)]
    kwargs: Dict[str, object] = {}
    if "start" in sig.parameters:
        kwargs["start"] = start
    if stop is not None and "stop" in sig.parameters:
        kwargs["stop"] = stop
    if "step" in sig.parameters:
        kwargs["step"] = step
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis.run(**kwargs)
    return [str(w.message) for w in caught if getattr(w, "message", None)]


def _default_water_selection(water_resnames: List[str]) -> str:
    if not water_resnames:
        return "resname SOL or resname HOH or resname TIP3 or resname TIP4"
    return "resname " + " ".join(water_resnames)


def _normalize_atom_indices(indices: np.ndarray, n_atoms: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=int)
    if arr.size == 0:
        return arr
    if arr.min() >= 1 and arr.max() == n_atoms:
        arr = arr - 1
    mask = (arr >= 0) & (arr < n_atoms)
    if not np.any(mask):
        return np.asarray([], dtype=int)
    return arr[mask]


def _map_frame_index(frame: int, frame_index_map: Dict[int, int], n_frames: int) -> Optional[int]:
    sample_idx = frame_index_map.get(int(frame))
    if sample_idx is None and 0 <= int(frame) < n_frames:
        sample_idx = int(frame)
    return sample_idx


def _extract_hbond_array(analysis) -> np.ndarray:
    if hasattr(analysis, "results") and hasattr(analysis.results, "hbonds"):
        hbonds = analysis.results.hbonds
    else:
        hbonds = getattr(analysis, "hbonds", None)
    if hbonds is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(hbonds)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    return arr


def _init_hbond_analysis(
    universe: mda.Universe,
    selection_a: str,
    selection_b: str,
    distance: float,
    angle: float,
    donors_selection: Optional[str],
    hydrogens_selection: Optional[str],
    acceptors_selection: Optional[str],
    update_selections: bool,
    pbc: bool,
):
    from MDAnalysis.analysis import hydrogenbonds

    cls = hydrogenbonds.HydrogenBondAnalysis
    init_params = inspect.signature(cls.__init__).parameters
    kwargs: Dict[str, object] = {}
    if "selection1" in init_params:
        kwargs["selection1"] = selection_a
        kwargs["selection2"] = selection_b
    elif "between" in init_params:
        kwargs["between"] = (selection_a, selection_b)
    elif "selection" in init_params:
        kwargs["selection"] = selection_a
        if "selection2" in init_params:
            kwargs["selection2"] = selection_b

    if "distance" in init_params:
        kwargs["distance"] = distance
    elif "d_a_cutoff" in init_params:
        kwargs["d_a_cutoff"] = distance

    if "angle" in init_params:
        kwargs["angle"] = angle
    elif "d_h_a_angle" in init_params:
        kwargs["d_h_a_angle"] = angle
    elif "d_h_a_angle_cutoff" in init_params:
        kwargs["d_h_a_angle_cutoff"] = angle

    for key, value in [
        ("donors_sel", donors_selection),
        ("donors_selection", donors_selection),
        ("hydrogens_sel", hydrogens_selection),
        ("hydrogens_selection", hydrogens_selection),
        ("acceptors_sel", acceptors_selection),
        ("acceptors_selection", acceptors_selection),
    ]:
        if value and key in init_params:
            kwargs[key] = value

    if "update_selections" in init_params:
        kwargs["update_selections"] = update_selections
    elif "update_selection" in init_params:
        kwargs["update_selection"] = update_selections

    if "pbc" in init_params:
        kwargs["pbc"] = pbc
    elif "use_pbc" in init_params:
        kwargs["use_pbc"] = pbc

    kwargs = _filter_kwargs(cls.__init__, kwargs)
    return cls(universe, **kwargs)


def _hbonds_to_water_sets(
    hbonds: np.ndarray,
    universe: mda.Universe,
    water_resindices: set[int],
    frame_index_map: Dict[int, int],
) -> Dict[int, set[int]]:
    sets_by_frame: Dict[int, set[int]] = {}
    if hbonds.size == 0:
        return sets_by_frame
    n_atoms = len(universe.atoms)
    n_frames = len(frame_index_map)
    frames = hbonds[:, 0].astype(int)
    donor_idx = _normalize_atom_indices(hbonds[:, 1], n_atoms)
    acceptor_idx = _normalize_atom_indices(hbonds[:, 3], n_atoms)
    if donor_idx.size == 0 or acceptor_idx.size == 0:
        return sets_by_frame
    valid_len = min(len(frames), len(donor_idx), len(acceptor_idx))
    frames = frames[:valid_len]
    donor_idx = donor_idx[:valid_len]
    acceptor_idx = acceptor_idx[:valid_len]
    atoms = universe.atoms
    for frame, donor, acceptor in zip(frames, donor_idx, acceptor_idx):
        sample_idx = _map_frame_index(int(frame), frame_index_map, n_frames)
        if sample_idx is None:
            continue
        donor_res = int(atoms[int(donor)].resindex)
        acceptor_res = int(atoms[int(acceptor)].resindex)
        water_res = None
        if donor_res in water_resindices:
            water_res = donor_res
        elif acceptor_res in water_resindices:
            water_res = acceptor_res
        if water_res is None:
            continue
        sets_by_frame.setdefault(sample_idx, set()).add(water_res)
    return sets_by_frame



def _resolve_hbond_bridge_backend(
    cfg: HbondWaterBridgeConfig,
    logger: Optional[logging.Logger] = None,
) -> tuple[Any, str, Optional[str]]:
    """
    Resolve the WaterBridgeAnalysis backend class based on config.
    Returns: (backend_cls, backend_type, error_message)
    backend_type is "waterbridge" or "hbond".
    """
    
    # helper to try import
    def try_import():
        try:
            from MDAnalysis.analysis.hydrogenbonds import WaterBridgeAnalysis
            return WaterBridgeAnalysis, None
        except ImportError:
            return None, "WaterBridgeAnalysis not found in MDAnalysis.analysis.hydrogenbonds"
        except Exception as e:
            return None, str(e)

    if cfg.backend == "waterbridge":
        cls, err = try_import()
        if cls:
            return cls, "waterbridge", None
        else:
            return None, "waterbridge", f"Explicit WaterBridgeAnalysis backend failed: {err}"
    
    elif cfg.backend == "hbond_analysis":
        return None, "hbond", None
    
    else: # auto
        cls, err = try_import()
        if cls:
             return cls, "waterbridge", None
        else:
             if logger:
                 logger.info(f"WaterBridgeAnalysis unavailable (auto mode): {err}. Using fallback.")
             return None, "hbond", None


def _run_hbond_water_bridges(
    universe: mda.Universe,
    solvent: SolventUniverse,
    bridge_configs: List[HbondWaterBridgeConfig],
    frame_indices: List[int],
    frame_times: List[float],
    frame_index_map: Dict[int, int],
    options: AnalysisOptions,
    water_resnames: List[str],
    store_frame_table: bool,
    selection_lookup: Dict[str, str] | None,
    logger: Optional[logging.Logger],
) -> tuple[Dict[str, BridgeResult], List[str]]:
    results: Dict[str, BridgeResult] = {}
    warnings_list: List[str] = []
    if not bridge_configs:
        return results, warnings_list

    # Removed top-level import check to avoid premature warnings.
    # Backend resolution is now handled per-config in the loop.

    default_water_selection = _default_water_selection(water_resnames)
    start = frame_indices[0] if frame_indices else 0
    stop = (frame_indices[-1] + 1) if frame_indices else None
    step = options.stride

    for cfg in bridge_configs:
        if logger:
            logger.info("Processing H-bond water bridge: %s...", cfg.name)
        water_sel = cfg.water_selection or default_water_selection
        sel_a = selection_lookup.get(cfg.selection_a, cfg.selection_a) if selection_lookup else cfg.selection_a
        sel_b = selection_lookup.get(cfg.selection_b, cfg.selection_b) if selection_lookup else cfg.selection_b
        hbonds = None
        notes: List[str] = []
        


        use_waterbridge = False
        waterbridge_class = None
        
        # Centralized resolution
        wb_cls, backend_type, wb_err = _resolve_hbond_bridge_backend(cfg, logger)

        if backend_type == "waterbridge":
            if wb_cls:
                use_waterbridge = True
                waterbridge_class = wb_cls
            elif wb_err:
                 # Explicit failure
                 warnings_list.append(wb_err)
                 if logger: logger.error(wb_err)
                 # Do not backup, abort this bridge
                 continue
        
        # If use_waterbridge is False, we fall through to hbond_analysis fallback logic below (if hbonds is None).

        if use_waterbridge and waterbridge_class:
            try:
                wba = _init_water_bridge_analysis(
                    universe,
                    sel_a,
                    sel_b,
                    cfg,
                    water_sel,
                    cls=waterbridge_class
                )
                wba_warnings = _run_analysis_with_frames(wba, start=start, stop=stop, step=step)
                hbonds = _extract_hbond_array(wba)
                if hbonds.size == 0:
                    hbonds = None
                if wba_warnings:
                    notes.extend(wba_warnings)
            except Exception as exc:
                if cfg.backend == "waterbridge":
                     msg = f"Explicit WaterBridgeAnalysis execution failed for '{cfg.name}': {exc}"
                     warnings_list.append(msg)
                     if logger: logger.error(msg)
                     continue 
                else: 
                     # Auto mode fallback
                     msg = (
                        f"H-bond water bridge '{cfg.name}' WaterBridgeAnalysis execution failed; "
                        f"falling back to HydrogenBondAnalysis. ({exc})"
                     )
                     warnings_list.append(msg)
                     if logger:
                        logger.warning(msg)
                     hbonds = None

        if hbonds is None:
            hbonds_a, warn_a = _run_hbond_contacts(
                universe,
                sel_a,
                water_sel,
                cfg,
                start,
                stop,
                step,
            )
            hbonds_b, warn_b = _run_hbond_contacts(
                universe,
                sel_b,
                water_sel,
                cfg,
                start,
                stop,
                step,
            )
            notes.extend(warn_a)
            notes.extend(warn_b)
            water_resindices = set(
                int(res.resindex)
                for res in universe.select_atoms(water_sel).residues
            )
            sets_a = _hbonds_to_water_sets(hbonds_a, universe, water_resindices, frame_index_map)
            sets_b = _hbonds_to_water_sets(hbonds_b, universe, water_resindices, frame_index_map)
        else:
            water_resindices = set(
                int(res.resindex)
                for res in universe.select_atoms(water_sel).residues
            )
            sets_a = _hbonds_to_water_sets(hbonds, universe, water_resindices, frame_index_map)
            sets_b = sets_a

        no_hbonds, other_notes = _split_hbond_warnings(notes)
        if no_hbonds:
            note_msg = (
                f"H-bond water bridge '{cfg.name}': no hydrogen bonds matched the criteria "
                f"(angle {cfg.angle} deg, distance {cfg.distance}A)."
            )
            warnings_list.append(note_msg)
            if logger:
                logger.info(note_msg)
            other_notes.append("No hydrogen bonds found for selected criteria.")
        notes = other_notes

        bridge_sets = []
        for idx in range(len(frame_times)):
            set_a = sets_a.get(idx, set())
            set_b = sets_b.get(idx, set())
            bridge_sets.append(set_a & set_b)

        acc = StatsAccumulator(
            solvent_records=solvent.record_by_resindex,
            gap_tolerance=options.gap_tolerance,
            frame_stride=options.stride,
            store_ids=options.store_ids,
            store_frame_table=store_frame_table,
        )
        for idx, bridge_set in enumerate(bridge_sets):
            frame_label = frame_indices[idx] if idx < len(frame_indices) else idx
            time_val = float(frame_times[idx]) if idx < len(frame_times) else float(idx)
            acc.update(idx, time_val, bridge_set, frame_label=frame_label)
        stats = acc.finalize()
        stats["summary"]["analysis_method"] = "HBA fallback" if hbonds is None else "WaterBridgeAnalysis"
        if notes:
            stats["summary"]["notes"] = notes

        edge_list = _bridge_edge_list(
            cfg.selection_a,
            cfg.selection_b,
            bridge_sets,
            solvent,
        )

        results[cfg.name] = BridgeResult(
            name=cfg.name,
            per_frame=stats["per_frame"],
            per_solvent=stats["per_solvent"],
            bridge_type="hbond",
            summary=stats["summary"],
            residence_cont=stats["residence_cont"],
            residence_inter=stats["residence_inter"],
            edge_list=edge_list,
        )
        if logger:
            logger.info(
                "H-bond water bridge %s: %d frames", cfg.name, len(stats["per_frame"])
            )
    return results, warnings_list


def _init_water_bridge_analysis(
    universe: mda.Universe,
    selection_a: str,
    selection_b: str,
    cfg: HbondWaterBridgeConfig,
    water_selection: str,
    cls: Any = None,
):
    if cls is None:
        # Fallback if not injected (though should be now) - attempt correct path if possible
        try:
             from MDAnalysis.analysis.hydrogenbonds import WaterBridgeAnalysis
             cls = WaterBridgeAnalysis
        except ImportError:
             # Legacy path just in case, or fail
             from MDAnalysis.analysis import waterbridge
             cls = waterbridge.WaterBridgeAnalysis

    init_params = inspect.signature(cls.__init__).parameters
    init_params = inspect.signature(cls.__init__).parameters
    kwargs: Dict[str, object] = {}
    if "selection1" in init_params:
        kwargs["selection1"] = selection_a
        kwargs["selection2"] = selection_b
    elif "between" in init_params:
        kwargs["between"] = (selection_a, selection_b)
    elif "selection" in init_params:
        kwargs["selection"] = selection_a
        if "selection2" in init_params:
            kwargs["selection2"] = selection_b
    for key in ("water_selection", "water_sel"):
        if key in init_params:
            kwargs[key] = water_selection
            break
    if "distance" in init_params:
        kwargs["distance"] = cfg.distance
    elif "d_a_cutoff" in init_params:
        kwargs["d_a_cutoff"] = cfg.distance
    if "angle" in init_params:
        kwargs["angle"] = cfg.angle
    elif "d_h_a_angle" in init_params:
        kwargs["d_h_a_angle"] = cfg.angle
    if "update_selections" in init_params:
        kwargs["update_selections"] = cfg.update_selections
    elif "update_selection" in init_params:
        kwargs["update_selection"] = cfg.update_selections

    for key, value in [
        ("donors_sel", cfg.donors_selection),
        ("donors_selection", cfg.donors_selection),
        ("hydrogens_sel", cfg.hydrogens_selection),
        ("hydrogens_selection", cfg.hydrogens_selection),
        ("acceptors_sel", cfg.acceptors_selection),
        ("acceptors_selection", cfg.acceptors_selection),
    ]:
        if value and key in init_params:
            kwargs[key] = value

    kwargs = _filter_kwargs(cls.__init__, kwargs)
    return cls(universe, **kwargs)


def _is_static_selection(selection: str) -> bool:
    """Check if a selection is likely static (not distance/geometric)."""
    dynamic_keywords = {
        "around",
        "sphlayer",
        "sphzone",
        "cylayer",
        "cylzone",
        "point",
        "prop",
    }
    # Simple check: if any dynamic keyword is present, assume dynamic
    tokens = set(re.split(r"\s+|\(|\)", selection.lower()))
    return not bool(tokens & dynamic_keywords)


def _run_hbond_contacts_batch(
    universe_args: Tuple[str, str],
    selection_a: str,
    selection_b: str,
    cfg_params: Dict[str, Any],
    start: int,
    stop: int,
    step: int,
) -> Tuple[np.ndarray, List[str]]:
    import MDAnalysis as mda
    import warnings
    topo, traj = universe_args
    u = mda.Universe(topo, traj)
    # Reconstruct config-like object or pass params directly
    # Here we simulate the config object for _init_hbond_analysis
    # We need to adapt _init_hbond_analysis to take direct args or a dummy object
    # For now, let's assume _init_hbond_analysis is available
    
    # We need to recreate the analysis object.
    # To avoid pickling issues with the cfg object, we passed cfg_params (dict).
    
    distance = cfg_params.get("distance", 3.0)
    angle = cfg_params.get("angle", 150.0)
    donors = cfg_params.get("donors_selection")
    hydrogens = cfg_params.get("hydrogens_selection")
    acceptors = cfg_params.get("acceptors_selection")
    update = cfg_params.get("update_selections", True)
    
    hba = _init_hbond_analysis(
        u,
        selection_a,
        selection_b,
        distance,
        angle,
        donors,
        hydrogens,
        acceptors,
        update,
        pbc=True,
    )
    
    # Suppress warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hba.run(start=start, stop=stop, step=step)
    
    warns = [str(w.message) for w in caught if getattr(w, "message", None)]
    hbonds = _extract_hbond_array(hba)
    return hbonds, warns



def _run_hbond_contacts(
    universe: mda.Universe,
    selection_a: str,
    selection_b: str,
    cfg: Any,
    start: int,
    stop: int | None,
    step: int,
) -> Tuple[np.ndarray, List[str]]:
    # Attempt to parallelize if range is large and joblib available
    n_frames_total = (stop if stop is not None else universe.trajectory.n_frames) - start
    # Determine effectively analyzed frames
    n_analyzed = n_frames_total // step
    
    use_parallel = False
    try:
        import joblib
        use_parallel = n_analyzed > 50  # Threshold
    except ImportError:
        pass

    if use_parallel:
        n_jobs = int(os.environ.get("OMP_NUM_THREADS", 4))
        try:
             # Split range
            import math
            chunk_size = math.ceil(n_analyzed / n_jobs)
            ranges = []
            curr = start
            start_offset = 0
            
            # We need to be careful with steps.
            # Easiest: split the indices [start, start+step, ...] into chunks
            all_indices = np.arange(start, stop if stop is not None else universe.trajectory.n_frames, step)
            if len(all_indices) == 0:
                 use_parallel = False
            else:
                chunks = np.array_split(all_indices, n_jobs)
                formatted_ranges = []
                for chunk in chunks:
                    if len(chunk) == 0: continue
                    chunk_start = int(chunk[0])
                    chunk_stop = int(chunk[-1]) + 1 # exclusive
                    formatted_ranges.append((chunk_start, chunk_stop, step))
                
                cfg_params = {
                    "distance": getattr(cfg, "distance", getattr(cfg, "hbond_distance", 3.0)),
                    "angle": getattr(cfg, "angle", getattr(cfg, "hbond_angle", 150.0)),
                    "donors_selection": getattr(cfg, "donors_selection", None),
                    "hydrogens_selection": getattr(cfg, "hydrogens_selection", None),
                    "acceptors_selection": getattr(cfg, "acceptors_selection", None),
                    "update_selections": getattr(cfg, "update_selections", True),
                }
                
                # Need files for new universe
                topo = str(universe.filename)
                traj = str(universe.trajectory.filename) if hasattr(universe.trajectory, 'filename') else None
                if not traj or not os.path.exists(topo) or not os.path.exists(traj):
                    use_parallel = False
                else:
                    results = joblib.Parallel(n_jobs=n_jobs)(
                        joblib.delayed(_run_hbond_contacts_batch)(
                            (topo, traj),
                            selection_a,
                            selection_b,
                            cfg_params,
                            r[0], r[1], r[2]
                        ) for r in formatted_ranges
                    )
                    
                    # Merge results
                    all_hbonds = []
                    all_notes = []
                    for hb, notes in results:
                        all_notes.extend(notes)
                        if hb.size > 0:
                            all_hbonds.append(hb)
                    
                    if not all_hbonds:
                        final_hbonds = np.empty((0, 6))
                    else:
                        final_hbonds = np.concatenate(all_hbonds, axis=0)
                    
                    return final_hbonds, all_notes

        except Exception as exc:
             # Fallback to serial on error
             print(f"Parallel hbond failed: {exc}")
             use_parallel = False

    try:
        distance = getattr(cfg, "distance", getattr(cfg, "hbond_distance", 3.0))
        angle = getattr(cfg, "angle", getattr(cfg, "hbond_angle", 150.0))
        hba = _init_hbond_analysis(
            universe,
            selection_a,
            selection_b,
            distance,
            angle,
            getattr(cfg, "donors_selection", None),
            getattr(cfg, "hydrogens_selection", None),
            getattr(cfg, "acceptors_selection", None),
            getattr(cfg, "update_selections", True),
            True,
        )
        if getattr(cfg, "update_selections", True):
            # Optimization: Disable update_selections if both are static
            if _is_static_selection(selection_a) and _is_static_selection(selection_b):
                # Check optional selections if present
                extras_static = True
                for attr in ["donors_selection", "hydrogens_selection", "acceptors_selection"]:
                    val = getattr(cfg, attr, None)
                    if val and not _is_static_selection(val):
                        extras_static = False
                        break
                
                if extras_static:
                    logging.getLogger(__name__).info(
                        "Optimizing HBA: Disabling 'update_selections' for static criteria (%s, %s)",
                        selection_a,
                        selection_b,
                    )
                    hba.update_selection = False

        warnings_list = _run_analysis_with_frames(hba, start=start, stop=stop, step=step)
        return _extract_hbond_array(hba), warnings_list
    except Exception as exc:
        msg = f"HydrogenBondAnalysis failed for selections '{selection_a}'/'{selection_b}': {exc}"
        return np.empty((0, 0), dtype=float), [msg]


def _split_hbond_warnings(messages: List[str]) -> tuple[List[str], List[str]]:
    no_hbonds = []
    other = []
    for msg in messages:
        if "No hydrogen bonds were found" in msg:
            no_hbonds.append(msg)
        else:
            other.append(msg)
    return no_hbonds, other


def _run_hbond_hydration(
    universe: mda.Universe,
    solvent: SolventUniverse,
    configs: List[HbondHydrationConfig],
    frame_indices: List[int],
    frame_times: List[float],
    frame_index_map: Dict[int, int],
    soz_frames_for_hbond: Dict[str, List[set[int]]],
    total_frames: int,
    options: AnalysisOptions,
    water_resnames: List[str],
    logger: Optional[logging.Logger],
) -> tuple[Dict[str, HydrationResult], List[str]]:
    results: Dict[str, HydrationResult] = {}
    warnings_list: List[str] = []
    if not configs:
        return results, warnings_list

    default_water_selection = _default_water_selection(water_resnames)
    start = frame_indices[0] if frame_indices else 0
    stop = (frame_indices[-1] + 1) if frame_indices else None
    step = options.stride

    for cfg in configs:
        if logger:
            logger.info("Processing H-bond hydration: %s...", cfg.name)
        water_sel = cfg.water_selection or default_water_selection
        residue_group = universe.select_atoms(cfg.residue_selection).residues
        solute_resindices = {int(res.resindex) for res in residue_group}
        water_resindices = {
            int(res.resindex) for res in universe.select_atoms(water_sel).residues
        }
        hbonds, hb_warn = _run_hbond_contacts(
            universe,
            cfg.residue_selection,
            water_sel,
            cfg,
            start,
            stop,
            step,
        )
        no_hbonds, other_warn = _split_hbond_warnings(hb_warn)
        if no_hbonds:
            msg = (
                f"H-bond hydration '{cfg.name}': no hydrogen bonds matched the criteria "
                f"(angle {cfg.angle} deg, distance {cfg.distance}{cfg.unit})."
            )
            warnings_list.append(msg)
            if logger:
                logger.info(msg)
        if other_warn:
            msg = f"H-bond hydration '{cfg.name}': warnings during HBA run. {other_warn[0]}"
            warnings_list.append(msg)
            if logger:
                logger.warning(msg)

        contacts_all: Dict[int, set[int]] = {res: set() for res in solute_resindices}
        contacts_soz: Dict[int, set[int]] = {res: set() for res in solute_resindices}
        soz_name = cfg.soz_name
        if not soz_name and soz_frames_for_hbond:
            soz_name = next(iter(soz_frames_for_hbond.keys()))
        soz_frames = soz_frames_for_hbond.get(soz_name, []) if soz_name else []

        if hbonds.size:
            n_atoms = len(universe.atoms)
            frames = hbonds[:, 0].astype(int)
            donor_idx = _normalize_atom_indices(hbonds[:, 1], n_atoms)
            acceptor_idx = _normalize_atom_indices(hbonds[:, 3], n_atoms)
            valid_len = min(len(frames), len(donor_idx), len(acceptor_idx))
            frames = frames[:valid_len]
            donor_idx = donor_idx[:valid_len]
            acceptor_idx = acceptor_idx[:valid_len]
            atoms = universe.atoms
            n_frames = len(frame_times)
            for frame, donor, acceptor in zip(frames, donor_idx, acceptor_idx):
                sample_idx = _map_frame_index(int(frame), frame_index_map, n_frames)
                if sample_idx is None:
                    continue
                donor_res = int(atoms[int(donor)].resindex)
                acceptor_res = int(atoms[int(acceptor)].resindex)
                solute_res = None
                water_res = None
                if donor_res in solute_resindices and acceptor_res in water_resindices:
                    solute_res = donor_res
                    water_res = acceptor_res
                elif acceptor_res in solute_resindices and donor_res in water_resindices:
                    solute_res = acceptor_res
                    water_res = donor_res
                if solute_res is None or water_res is None:
                    continue
                contacts_all.setdefault(solute_res, set()).add(sample_idx)
                if soz_frames and sample_idx < len(soz_frames) and water_res in soz_frames[sample_idx]:
                    contacts_soz.setdefault(solute_res, set()).add(sample_idx)

        contacts_total = contacts_soz if cfg.conditioning == "soz" else contacts_all
        counts_total = {res: len(frames) for res, frames in contacts_total.items()}
        counts_soz = {res: len(frames) for res, frames in contacts_soz.items()}
        soz_populated = sum(1 for frame_set in soz_frames if frame_set) if soz_frames else 0
        table = _finalize_contact_table(
            counts_total,
            counts_soz,
            residue_group,
            total_frames,
            soz_populated,
        )

        contact_frames_total = {
            res: sorted(list(frames)) for res, frames in contacts_total.items()
        }
        contact_frames_soz = {res: sorted(list(frames)) for res, frames in contacts_soz.items()}

        results[cfg.name] = HydrationResult(
            name=cfg.name,
            table=table,
            frame_times=frame_times,
            frame_labels=frame_indices,
            contact_frames_total=contact_frames_total,
            contact_frames_given_soz=contact_frames_soz,
            mode="hbond",
        )
        if logger:
            logger.info("H-bond hydration %s: %d residues", cfg.name, len(table))
    return results, warnings_list


def _bridge_edge_list(
    selection_a: str,
    selection_b: str,
    bridge_sets: List[set[int]],
    solvent: SolventUniverse,
) -> pd.DataFrame:
    counts: Dict[tuple[str, str], int] = {}
    for frame_set in bridge_sets:
        for resindex in frame_set:
            record = solvent.record_by_resindex.get(resindex)
            if record is None:
                continue
            node = record.stable_id
            counts[(selection_a, node)] = counts.get((selection_a, node), 0) + 1
            counts[(selection_b, node)] = counts.get((selection_b, node), 0) + 1
    rows = [
        {"source": src, "target": tgt, "frames_present": count}
        for (src, tgt), count in counts.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["frames_present", "source"], ascending=False, inplace=True)
    return df


def _clone_universe(universe: mda.Universe) -> mda.Universe:
    try:
        if hasattr(universe, "copy"):
            return universe.copy()
    except Exception:
        pass
    try:
        def _jsonable_path(value: object) -> object:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value]
            return str(value)

        topology = _jsonable_path(getattr(universe, "filename", None))
        trajectory = _jsonable_path(getattr(getattr(universe, "trajectory", None), "filename", None))
        if topology and trajectory:
            return mda.Universe(topology, trajectory)
        if topology:
            return mda.Universe(topology)
    except Exception:
        pass
    return universe


def _run_density_maps(
    universe: mda.Universe,
    configs: List[DensityMapConfig],
    options: AnalysisOptions,
    output_dir: str,
    progress: Optional[ProgressCallback],
    logger: Optional[logging.Logger],
) -> Dict[str, DensityMapResult]:
    results: Dict[str, DensityMapResult] = {}
    if not configs:
        return results
    try:
        from MDAnalysis.analysis import density as density_module
        from MDAnalysis.analysis import align as align_module
    except Exception as exc:
        if logger:
            logger.warning("Density analysis unavailable: %s", exc)
        return results

    for cfg in configs:
        # --- 1. Conditioning Gating ---
        policy = getattr(cfg, "conditioning_policy", "strict")
        is_safe = True
        warnings_list = []
        
        # Check alignment
        if not cfg.align and policy != "unsafe":
            msg = f"Density map '{cfg.name}' requested without alignment. " \
                  "This is scientfically unsafe for solvent density."
            if policy == "strict":
                if logger: logger.error(msg)
                continue # Skip this map
            else:
                warnings_list.append(msg)
                if logger: logger.warning(msg)
                is_safe = False

        # TODO: Check PBC correction if metadata available (hard to do generically on Universe without extra info)
        
        density_dir = os.path.join(output_dir, f"density_map_{cfg.name}")
        os.makedirs(density_dir, exist_ok=True)
        cache_path = os.path.join(density_dir, "cache.json")
        npz_path = os.path.join(density_dir, "grid.npz")

        start = cfg.frame_start if cfg.frame_start is not None else options.frame_start
        stop = cfg.frame_stop if cfg.frame_stop is not None else options.frame_stop
        step = cfg.stride if cfg.stride else options.stride
        output_format = (cfg.output_format or "dx").lower()
        if output_format not in ("dx", "npy", "npz"):
            output_format = "dx"
        grid_path = os.path.join(density_dir, f"{cfg.name}.{output_format}")

        topology = getattr(universe, "filename", None)
        trajectory = getattr(getattr(universe, "trajectory", None), "filename", None)
        
        raw_selection = getattr(cfg, "species_selection", getattr(cfg, "selection", ""))
        selection = sanitize_selection_string(raw_selection)
        if selection != (raw_selection or "") and logger:
            logger.info(
                "Density map %s selection normalized from %r to %r",
                cfg.name,
                raw_selection,
                selection,
            )

        cache_key = {
            "selection": selection,
            "grid_spacing": cfg.grid_spacing,
            "padding": cfg.padding,
            "stride": step,
            "frame_start": start,
            "frame_stop": stop,
            "align": cfg.align,
            "align_selection": cfg.align_selection,
            "align_reference": cfg.align_reference,
            "align_reference_path": cfg.align_reference_path,
            "output_format": output_format,
            "topology": topology,
            "trajectory": trajectory,
            "policy": policy,
        }
        reuse_cache = False
        if os.path.exists(cache_path) and os.path.exists(npz_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if cached.get("config") == cache_key:
                    reuse_cache = True
            except Exception:
                reuse_cache = False

        if reuse_cache:
            data = np.load(npz_path)
            grid = data["grid"]
            axes = {
                "x": data["x"],
                "y": data["y"],
                "z": data["z"],
            }
            metadata = cached.get("metadata", {})
            # Ensure runtime warnings are preserved/merged? For now, cached is assumed 'what happened'
            grid_path = metadata.get("grid_path", grid_path)
        else:
            density_universe = universe
            # Apply Alignment (Conditioning)
            if cfg.align and cfg.align_selection:
                density_universe = _clone_universe(universe)
                ref = density_universe
                if cfg.align_reference == "structure" and cfg.align_reference_path:
                    ref = mda.Universe(cfg.align_reference_path)
                align_kwargs = {
                    "select": cfg.align_selection,
                    "in_memory": True,
                }
                align_kwargs = _filter_kwargs(
                    align_module.AlignTraj.__init__,
                    align_kwargs,
                )
                # [Fix] Robust alignment: mismatched masses (e.g. PDB vs TPR) cause SelectionError.
                # Fallback to geometric alignment (weights=None) if mass-weighted fails.
                try:
                    aligner = align_module.AlignTraj(density_universe, ref, **align_kwargs)
                except Exception as e:
                    # Check for mass mismatch message or generic selection error
                    if "mass" in str(e).lower() or "selection" in str(e).lower():
                        msg = f"Mass-weighted alignment failed ({e}); falling back to geometric alignment."
                        warnings_list.append(msg)
                        if logger: logger.warning(f"Density map {cfg.name}: {msg}")
                        # [Fix] MDAnalysis enforces mass checks even with weights=None in some versions.
                        # We explicitly sync masses to bypass this check.
                        try:
                            m_sel = density_universe.select_atoms(cfg.align_selection)
                            r_sel = ref.select_atoms(cfg.align_selection)
                            if m_sel.n_atoms == r_sel.n_atoms:
                                r_sel.masses = m_sel.masses
                        except Exception:
                            # Ignore errors here, let AlignTraj fail if it must
                            pass
                            
                        # Retry without masses
                        aligner = align_module.AlignTraj(density_universe, ref, weights=None, **align_kwargs)
                    else:
                        raise e

                align_warn = _run_analysis_with_frames(
                    aligner,
                    start=start,
                    stop=stop,
                    step=step,
                )
                if align_warn:
                     formatted = f"Align warning: {align_warn[0]}"
                     warnings_list.append(formatted)
                     if logger: logger.info(f"Density map {cfg.name}: {formatted}")

            density_kwargs = {
                "delta": cfg.grid_spacing,
                "padding": cfg.padding,
            }
            density_kwargs = _filter_kwargs(
                density_module.DensityAnalysis.__init__,
                density_kwargs,
            )
            
            # [Fix] Explicitly create AtomGroup for density calculation
            # Previously 'selection' kwarg was passed but ignored by DensityAnalysis,
            # resulting in density of the entire universe (all atoms).
            selection_is_dynamic = bool(selection and not _is_static_selection(selection))
            if selection:
                atom_group = density_universe.select_atoms(selection, updating=selection_is_dynamic)
            else:
                atom_group = density_universe.atoms
            if selection_is_dynamic and logger:
                logger.info(
                    "Density map %s: using UpdatingAtomGroup for dynamic selection %r",
                    cfg.name,
                    selection,
                )
                
            analysis = density_module.DensityAnalysis(atom_group, **density_kwargs)
            if progress:
                progress(0, 1, f"Density map {cfg.name}")
            dens_warn = _run_analysis_with_frames(analysis, start=start, stop=stop, step=step)
            if dens_warn:
                warnings_list.append(f"Density warning: {dens_warn[0]}")
                if logger: logger.info("Density map %s warnings: %s", cfg.name, dens_warn[0])

            density_obj = getattr(analysis, "results", None)
            grid_obj = None
            if density_obj is not None and hasattr(density_obj, "density"):
                grid_obj = density_obj.density
            elif hasattr(analysis, "density"):
                grid_obj = analysis.density
            if grid_obj is None:
                raise ValueError(f"Density map '{cfg.name}' produced no grid.")
            if hasattr(grid_obj, "grid"):
                grid = np.asarray(grid_obj.grid)
            else:
                grid = np.asarray(grid_obj)
                
            # --- Views Computation ---
            # 1. Physical (raw) is `grid`
            
            # 2. Bulk (Scalar)
            # Estimate bulk density simply as mean matching the selection count? 
            # Or use explicit bulk_density param? 
            # Ideally mda.density provides density, not counts. 
            # If mda returns probability density (A^-3), we can compare to water bulk (0.0334 A^-3).
            # If mda returns counts, we convert to density.
            # MDAnalysis DensityAnalysis default is 'probability' (A^-3) if parameters unchecked?
            # Actually default is usually A^-3 if not massive changes.
            # Let's assume physical units (A^-3).
            
            # Crude bulk estimate: Total N / Box Volume (average)
            # But we filtered selection.
            # So we use a standard constant for water (0.0334) or just 1.0 if unknown.
            # For now, let's store 0.0334 (standard water density) as rho_bulk reference 
            # but allow override later.
            rho_bulk = 0.0334 
            
            # Save raw grid
            if output_format == "dx" and hasattr(grid_obj, "export"):
                try:
                    grid_obj.export(grid_path)
                except Exception:
                    fallback_path = os.path.join(density_dir, f"{cfg.name}.npy")
                    np.save(fallback_path, grid)
                    grid_path = fallback_path
                    output_format = "npy"
            elif output_format == "npy":
                np.save(grid_path, grid)
            elif output_format == "npz":
                np.savez(grid_path, grid=grid)
            else:
                fallback_path = os.path.join(density_dir, f"{cfg.name}.npy")
                np.save(fallback_path, grid)
                grid_path = fallback_path
                output_format = "npy"

            delta = cfg.grid_spacing
            shape = grid.shape
            origin = None
            if hasattr(grid_obj, "origin"):
                try:
                    origin = np.asarray(grid_obj.origin, dtype=float)
                except Exception:
                    origin = None
            if origin is None or origin.size < 3:
                origin = np.zeros(3, dtype=float)
            axes = {
                "x": origin[0] + np.arange(shape[0]) * delta,
                "y": origin[1] + np.arange(shape[1]) * delta,
                "z": origin[2] + np.arange(shape[2]) * delta,
            }
            metadata = {
                "grid_spacing": delta,
                "padding": cfg.padding,
                "shape": shape,
                "origin": origin.tolist(),
                "frame_start": start,
                "frame_stop": stop,
                "stride": step,
                "align": cfg.align,
                "align_selection": cfg.align_selection,
                "align_reference": cfg.align_reference,
                "align_reference_path": cfg.align_reference_path,
                "grid_path": grid_path,
                "output_format": output_format,
                "analysis_warnings": warnings_list,
                "is_scientifically_safe": is_safe,
                "rho_bulk_approx": rho_bulk,
                "density_unit": "Angstrom^-3",
                "selection_is_dynamic": selection_is_dynamic,
            }
            np.savez(npz_path, grid=grid, x=axes["x"], y=axes["y"], z=axes["z"])
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump({"config": cache_key, "metadata": metadata}, handle, indent=2)

        slices = _density_default_slices(grid)
        _export_density_slices(density_dir, axes, slices)
        results[cfg.name] = DensityMapResult(
            name=cfg.name,
            grid_path=grid_path,
            grid=grid,
            axes=axes,
            slices=slices,
            metadata=metadata,
        )
    return results


def _density_default_slices(grid: np.ndarray) -> Dict[str, np.ndarray]:
    slices: Dict[str, np.ndarray] = {}
    if grid.size == 0:
        return slices
    cx = grid.shape[0] // 2
    cy = grid.shape[1] // 2
    cz = grid.shape[2] // 2
    slices["xy"] = grid[:, :, cz]
    slices["xz"] = grid[:, cy, :]
    slices["yz"] = grid[cx, :, :]
    slices["max_projection"] = grid.max(axis=2)
    return slices


def _export_density_slices(
    density_dir: str,
    axes: Dict[str, np.ndarray],
    slices: Dict[str, np.ndarray],
) -> None:
    for key, data in slices.items():
        if key == "xy":
            x = axes["x"]
            y = axes["y"]
        elif key == "xz":
            x = axes["x"]
            y = axes["z"]
        elif key == "yz":
            x = axes["y"]
            y = axes["z"]
        else:
            x = axes["x"]
            y = axes["y"]
        xx, yy = np.meshgrid(x, y, indexing="ij")
        flat = pd.DataFrame(
            {
                "x": xx.ravel(),
                "y": yy.ravel(),
                "density": data.ravel(),
                "slice": key,
            }
        )
        flat.to_csv(os.path.join(density_dir, f"{key}.csv"), index=False)


def _run_water_dynamics(
    universe: mda.Universe,
    solvent: SolventUniverse,
    configs: List[WaterDynamicsConfig],
    soz_results: Dict[str, SOZResult],
    frame_times: List[float],
    frame_labels: List[int],
    frame_index_map: Dict[int, int],
    selection_frame_sets: Dict[str, List[set[int]]],
    options: AnalysisOptions,
    water_resnames: List[str],
    logger: Optional[logging.Logger],
) -> Dict[str, WaterDynamicsResult]:
    results: Dict[str, WaterDynamicsResult] = {}
    if not configs:
        return results

    dt = 1.0
    if len(frame_times) > 1:
        dt = float(np.median(np.diff(frame_times)))

    waterdynamics_module = None
    try:
        from MDAnalysis.analysis import waterdynamics as waterdynamics_module  # type: ignore
    except Exception:
        try:
            import waterdynamics as waterdynamics_module  # type: ignore
        except Exception:
            waterdynamics_module = None

    for cfg in configs:
        notes: List[str] = []
        residence_lengths: Dict[int, List[int]] = {}
        if cfg.region_mode == "soz":
            soz_name = cfg.soz_name or (next(iter(soz_results.keys())) if soz_results else None)
            if soz_name and soz_name in soz_results:
                residence_lengths = (
                    soz_results[soz_name].residence_cont
                    if cfg.residence_mode == "continuous"
                    else soz_results[soz_name].residence_inter
                )
                dt = float(soz_results[soz_name].summary.get("dt", dt))
            else:
                notes.append("SOZ not found; SP(tau) unavailable.")
        else:
            frame_sets = selection_frame_sets.get(cfg.name, [])
            stats = compute_residence_lengths(frame_sets, options.gap_tolerance)
            residence_lengths = (
                stats.continuous if cfg.residence_mode == "continuous" else stats.intermittent
            )

        notes.append(
            f"Tau uses median frame spacing dt={dt:.3g} (trajectory time units); stride={options.stride}."
        )

        durations = []
        for lengths in residence_lengths.values():
            durations.extend([length * dt for length in lengths])
        durations = np.array(durations, dtype=float)
        durations = durations[durations > 0]
        durations.sort()
        if durations.size:
            survival = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
            sp_df = pd.DataFrame({"tau": durations, "survival": survival})
            mean_res = float(np.mean(durations))
        else:
            sp_df = pd.DataFrame({"tau": [], "survival": []})
            mean_res = 0.0

        hbl_df = None
        hbl_summary = None
        wor_df = None
        if waterdynamics_module is None:
            notes.append("waterdynamics package not available; HBL/WOR disabled.")
        else:
            try:
                hbl_cls = getattr(waterdynamics_module, "HydrogenBondLifetimes", None)
                wor_cls = getattr(waterdynamics_module, "WaterOrientationalRelaxation", None)
                if hbl_cls and cfg.solute_selection:
                    hbl_kwargs = _filter_kwargs(
                        hbl_cls.__init__,
                        {
                            "selection1": cfg.solute_selection,
                            "selection2": cfg.water_selection
                            or _default_water_selection(water_resnames),
                            "distance": cfg.hbond_distance,
                            "angle": cfg.hbond_angle,
                            "update_selections": cfg.update_selections,
                        },
                    )
                    hbl = hbl_cls(universe, **hbl_kwargs)
                    hbl_warn = _run_analysis_with_frames(
                        hbl, start=options.frame_start, stop=options.frame_stop, step=options.stride
                    )
                    if hbl_warn:
                        notes.append(f"HBL warnings: {hbl_warn[0]}")
                    data = getattr(hbl, "results", hbl)
                    series = getattr(data, "tau", None)
                    corr = getattr(data, "correlation", None)
                    if series is not None and corr is not None:
                        hbl_df = pd.DataFrame({"tau": np.asarray(series), "correlation": np.asarray(corr)})
                
                # Pure-numpy fallback for HBL curve if module fails or curve is None
                if hbl_df is None and cfg.solute_selection:
                    try:
                        # We'll need the hbonds which are calculated later, 
                        # but we can move this check or do it after _run_hbond_contacts.
                        pass
                    except Exception:
                        pass
                if wor_cls:
                    wor_kwargs = _filter_kwargs(
                        wor_cls.__init__,
                        {
                            "selection": cfg.water_selection
                            or _default_water_selection(water_resnames)
                        },
                    )
                    wor = wor_cls(universe, **wor_kwargs)
                    wor_warn = _run_analysis_with_frames(
                        wor, start=options.frame_start, stop=options.frame_stop, step=options.stride
                    )
                    if wor_warn:
                        notes.append(f"WOR warnings: {wor_warn[0]}")
                    data = getattr(wor, "results", wor)
                    series = getattr(data, "tau", None)
                    corr = getattr(data, "correlation", None)
                    if series is not None and corr is not None:
                        wor_df = pd.DataFrame({"tau": np.asarray(series), "correlation": np.asarray(corr)})
            except Exception as exc:
                notes.append(f"waterdynamics analysis failed: {exc}")
                
            # Pure-numpy fallbacks for WOR if module fails
            if wor_df is None and cfg.water_selection:
                try:
                    from .water_fallbacks import compute_wor_fallback
                    wor_df_fb = compute_wor_fallback(universe, cfg.water_selection, options)
                    if wor_df_fb is not None:
                        wor_df = wor_df_fb
                        notes.append("Using pure-numpy fallback for Water Orientational Relaxation.")
                except Exception as fb_exc:
                     notes.append(f"WOR fallback failed: {fb_exc}")

        if cfg.solute_selection:
            try:
                water_sel = cfg.water_selection or _default_water_selection(water_resnames)
                start = options.frame_start
                stop = options.frame_stop
                step = options.stride
                hbonds, hb_warn = _run_hbond_contacts(
                    universe,
                    cfg.solute_selection,
                    water_sel,
                    cfg,
                    start,
                    stop,
                    step,
                )
                no_hbonds, other_warn = _split_hbond_warnings(hb_warn)
                if no_hbonds:
                    notes.append(
                        "No hydrogen bonds found for water dynamics criteria; HBL summary may be empty."
                    )
                if other_warn:
                    notes.append(f"HBL warnings: {other_warn[0]}")
                frame_sets: List[set[int]] = [set() for _ in range(len(frame_labels))]
                if hbonds.size:
                    n_atoms = len(universe.atoms)
                    frames = hbonds[:, 0].astype(int)
                    donor_idx = _normalize_atom_indices(hbonds[:, 1], n_atoms)
                    acceptor_idx = _normalize_atom_indices(hbonds[:, 3], n_atoms)
                    valid_len = min(len(frames), len(donor_idx), len(acceptor_idx))
                    frames = frames[:valid_len]
                    donor_idx = donor_idx[:valid_len]
                    acceptor_idx = acceptor_idx[:valid_len]
                    atoms = universe.atoms
                    solute_resindices = {
                        int(res.resindex)
                        for res in universe.select_atoms(cfg.solute_selection).residues
                    }
                    water_resindices = {
                        int(res.resindex)
                        for res in universe.select_atoms(water_sel).residues
                    }
                    n_frames = len(frame_labels)
                    for frame, donor, acceptor in zip(frames, donor_idx, acceptor_idx):
                        sample_idx = _map_frame_index(int(frame), frame_index_map, n_frames)
                        if sample_idx is None or sample_idx >= len(frame_sets):
                            continue
                        donor_res = int(atoms[int(donor)].resindex)
                        acceptor_res = int(atoms[int(acceptor)].resindex)
                        solute_res = None
                        if donor_res in solute_resindices and acceptor_res in water_resindices:
                            solute_res = donor_res
                        elif acceptor_res in solute_resindices and donor_res in water_resindices:
                            solute_res = acceptor_res
                        if solute_res is not None:
                            frame_sets[sample_idx].add(solute_res)
                stats = compute_residence_lengths(frame_sets, options.gap_tolerance)
                lengths = (
                    stats.continuous if cfg.residence_mode == "continuous" else stats.intermittent
                )
                rows = []
                for resindex, segments in lengths.items():
                    mean_len = float(np.mean(segments)) * dt if segments else 0.0
                    residue = universe.residues[resindex]
                    rows.append(
                        {
                            "resindex": int(resindex),
                            "resid": int(residue.resid),
                            "resname": str(residue.resname),
                            "segid": str(residue.segid) if residue.segid else "",
                            "mean_lifetime": mean_len,
                            "n_segments": len(segments),
                        }
                    )
                hbl_summary = pd.DataFrame(rows)
                if not hbl_summary.empty:
                    hbl_summary.sort_values(
                        by=["mean_lifetime", "resid"], ascending=False, inplace=True
                    )
                
                # Fill hbl_df curve if still missing using the already computed hbonds
                if hbl_df is None and hbonds.size:
                    try:
                         from .water_fallbacks import compute_hbl_fallback_curve
                         hbl_df_fb = compute_hbl_fallback_curve(universe, hbonds, options, dt)
                         if hbl_df_fb is not None:
                             hbl_df = hbl_df_fb
                             notes.append("Using pure-numpy fallback for H-bond lifetime correlation.")
                    except Exception as fb_exc:
                         notes.append(f"HBL curve fallback failed: {fb_exc}")
            except Exception as exc:
                notes.append(f"HBL summary fallback failed: {exc}")

        results[cfg.name] = WaterDynamicsResult(
            name=cfg.name,
            sp_tau=sp_df,
            mean_residence_time=mean_res,
            residence_mode=cfg.residence_mode,
            hbl=hbl_df,
            hbl_summary=hbl_summary,
            wor=wor_df,
            notes=notes,
        )
        if logger:
            logger.info("Water dynamics %s: %d SP points", cfg.name, len(sp_df))
    return results


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
