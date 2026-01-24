"""Validation helpers for SOZLab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib import distances

from engine.models import AnalysisOptions, ProjectConfig, SOZNode
from engine.resolver import resolve_selection
from engine.solvent import build_solvent, resolve_probe_mode, solvent_positions
from engine.soz_eval import EvaluationContext
from engine.units import to_internal_length
from engine.preflight import run_preflight, PreflightReport


@dataclass
class ValidationResult:
    soz_name: str
    mismatched_frames: int
    total_frames: int
    first_mismatch: Dict[str, object] | None


def validate_project(
    project: ProjectConfig,
    max_frames: int = 200,
) -> List[ValidationResult]:
    if project.inputs.trajectory:
        universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
    else:
        universe = mda.Universe(project.inputs.topology)

    preflight = run_preflight(project, universe)
    if not preflight.ok:
        raise ValueError("Preflight failed: " + "; ".join(preflight.errors))

    solvent = build_solvent(universe, project.solvent)
    resolved_selections = {
        label: resolve_selection(universe, spec) for label, spec in project.selections.items()
    }
    context = EvaluationContext(universe=universe, solvent=solvent, selections=resolved_selections)

    options: AnalysisOptions = project.analysis
    frame_indices = list(range(0, len(universe.trajectory), options.stride))[:max_frames]

    results = []
    for soz in project.sozs:
        mismatches = 0
        first_mismatch = None
        for frame_index in frame_indices:
            universe.trajectory[frame_index]
            fast_set = evaluate_node_fast(soz.root, context)
            slow_set = evaluate_node_slow(soz.root, context)
            if fast_set != slow_set:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "frame": frame_index,
                        "fast_count": len(fast_set),
                        "slow_count": len(slow_set),
                        "fast_only": sorted(fast_set - slow_set),
                        "slow_only": sorted(slow_set - fast_set),
                    }
        results.append(
            ValidationResult(
                soz_name=soz.name,
                mismatched_frames=mismatches,
                total_frames=len(frame_indices),
                first_mismatch=first_mismatch,
            )
        )
    return results


def evaluate_node_fast(node: SOZNode, context: EvaluationContext) -> set[int]:
    from engine.soz_eval import evaluate_node

    return evaluate_node(node, context)


def _distance_resindices_slow(
    seed_positions: np.ndarray,
    solvent_positions_arr: np.ndarray,
    atom_to_resindex: list[int],
    cutoff_nm: float,
    box,
) -> set[int]:
    if seed_positions.size == 0 or solvent_positions_arr.size == 0:
        return set()
    dist = distances.distance_array(seed_positions, solvent_positions_arr, box=box)
    if dist.size == 0:
        return set()
    min_dist = np.min(dist, axis=0)
    solvent_atom_indices = np.where(min_dist <= cutoff_nm)[0]
    if solvent_atom_indices.size == 0:
        return set()
    resindices: set[int] = set()
    atom_map_len = len(atom_to_resindex)
    for idx in solvent_atom_indices:
        idx_i = int(idx)
        if 0 <= idx_i < atom_map_len:
            resindices.add(atom_to_resindex[idx_i])
    return resindices


def evaluate_node_slow(node: SOZNode, context: EvaluationContext) -> set[int]:
    if node.type in ("and", "or"):
        if not node.children:
            return set()
        child_sets = [evaluate_node_slow(child, context) for child in node.children]
        if node.type == "and":
            result = child_sets[0].copy()
            for child_set in child_sets[1:]:
                result &= child_set
            return result
        result = set()
        for child_set in child_sets:
            result |= child_set
        return result

    if node.type == "not":
        if not node.children:
            return set()
        child_set = evaluate_node_slow(node.children[0], context)
        return context.solvent.all_resindex_set() - child_set

    if node.type == "distance":
        seed_label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
        cutoff = float(node.params.get("cutoff", 3.5))
        unit = node.params.get("unit", "A")
        atom_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        cutoff_nm = to_internal_length(cutoff, unit)
        seed = context.selections[seed_label].group
        mode = resolve_probe_mode(atom_mode, context.solvent.probe.position)
        solvent_pos, atom_map = solvent_positions(context.solvent, mode)
        return _distance_resindices_slow(seed.positions, solvent_pos, atom_map, cutoff_nm, context.pbc_box)

    if node.type == "shell":
        seed_label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
        cutoffs = node.params.get("cutoffs", [3.5])
        unit = node.params.get("unit", "A")
        atom_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        cutoffs_nm = [to_internal_length(float(value), unit) for value in cutoffs]
        mode = resolve_probe_mode(atom_mode, context.solvent.probe.position)
        seed = context.selections[seed_label].group
        shell_sets: list[set[int]] = []
        current_seed_positions = seed.positions
        for cutoff_nm in cutoffs_nm:
            solvent_pos, atom_map = solvent_positions(context.solvent, mode)
            resindices = _distance_resindices_slow(
                current_seed_positions,
                solvent_pos,
                atom_map,
                cutoff_nm,
                context.pbc_box,
            )
            resindices = resindices - set().union(*shell_sets) if shell_sets else resindices
            shell_sets.append(resindices)
            if not resindices:
                current_seed_positions = np.empty((0, 3))
            else:
                current_seed_positions, _ = solvent_positions(
                    context.solvent,
                    mode,
                    resindices=resindices,
                )
        result = set()
        for shell in shell_sets:
            result |= shell
        return result

    raise ValueError(f"Unsupported node type: {node.type}")
