"""SOZ evaluation logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib import distances

from engine.models import SOZNode
from engine.units import to_internal_length
from engine.resolver import ResolvedSelection
from engine.solvent import SolventUniverse


@dataclass
class EvaluationContext:
    universe: mda.Universe
    solvent: SolventUniverse
    selections: Dict[str, ResolvedSelection] | None = None
    seeds: Dict[str, ResolvedSelection] | None = None

    def __post_init__(self) -> None:
        if self.selections is None and self.seeds is not None:
            self.selections = self.seeds
        if self.selections is None:
            self.selections = {}
        if self.seeds is None:
            self.seeds = self.selections

    @property
    def pbc_box(self):
        dims = self.universe.trajectory.ts.dimensions
        if dims is None:
            return None
        if np.all(np.array(dims[:3]) > 0):
            return dims
        return None


def _distance_resindices(
    seed_group: mda.core.groups.AtomGroup,
    solvent_atoms: mda.core.groups.AtomGroup,
    atom_to_resindex: list[int],
    cutoff_nm: float,
    box,
) -> Set[int]:
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
            if min_idx >= 1 and max_idx == len(atom_to_resindex):
                solvent_atom_indices = solvent_atom_indices - 1
            if min_idx < 0 or max_idx >= len(atom_to_resindex):
                solvent_atom_indices = solvent_atom_indices[
                    (solvent_atom_indices >= 0) & (solvent_atom_indices < len(atom_to_resindex))
                ]
        if solvent_atom_indices.size == 0:
            return set()

        resindices: Set[int] = set()
        atom_map_len = len(atom_to_resindex)
        for idx in solvent_atom_indices:
            idx_i = int(idx)
            if 0 <= idx_i < atom_map_len:
                resindices.add(atom_to_resindex[idx_i])
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
        resindices: Set[int] = set()
        atom_map_len = len(atom_to_resindex)
        for idx in solvent_atom_indices:
            idx_i = int(idx)
            if 0 <= idx_i < atom_map_len:
                resindices.add(atom_to_resindex[idx_i])
        return resindices


def _resolve_selection_label(node: SOZNode) -> str:
    params = node.params
    label = params.get("selection_label") or params.get("seed_label") or params.get("seed")
    if not label:
        raise ValueError(f"Node of type '{node.type}' missing selection_label")
    return label


def evaluate_node(node: SOZNode, context: EvaluationContext) -> Set[int]:
    if node.type in ("and", "or"):
        if not node.children:
            return set()
        child_sets = [evaluate_node(child, context) for child in node.children]
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
        child_set = evaluate_node(node.children[0], context)
        return context.solvent.all_resindex_set() - child_set

    if node.type == "distance":
        selection_label = _resolve_selection_label(node)
        cutoff = float(node.params.get("cutoff", 3.5))
        unit = node.params.get("unit", "A")
        atom_mode = node.params.get("atom_mode", "O")
        cutoff_internal = to_internal_length(cutoff, unit)
        seed = context.selections[selection_label].group
        if atom_mode.lower() == "all":
            solvent_atoms = context.solvent.atoms_all
            atom_map = context.solvent.atom_to_resindex_all
        else:
            solvent_atoms = context.solvent.atoms_oxygen
            atom_map = context.solvent.atom_to_resindex_oxygen
        return _distance_resindices(seed, solvent_atoms, atom_map, cutoff_internal, context.pbc_box)

    if node.type == "shell":
        selection_label = _resolve_selection_label(node)
        cutoffs = node.params.get("cutoffs", [3.5])
        unit = node.params.get("unit", "A")
        atom_mode = node.params.get("atom_mode", "O")
        cutoffs_internal = [to_internal_length(float(value), unit) for value in cutoffs]
        if atom_mode.lower() == "all":
            solvent_atoms = context.solvent.atoms_all
            atom_map = context.solvent.atom_to_resindex_all
        else:
            solvent_atoms = context.solvent.atoms_oxygen
            atom_map = context.solvent.atom_to_resindex_oxygen

        seed = context.selections[selection_label].group
        shell_sets: list[set[int]] = []
        current_seed_atoms = seed
        for cutoff_nm in cutoffs_internal:
            resindices = _distance_resindices(
                current_seed_atoms,
                solvent_atoms,
                atom_map,
                cutoff_nm,
                context.pbc_box,
            )
            resindices = resindices - set().union(*shell_sets) if shell_sets else resindices
            shell_sets.append(resindices)
            if not resindices:
                current_seed_atoms = context.solvent.atoms_all[[]]
            else:
                atom_indices = []
                if atom_mode.lower() == "all":
                    for resindex in sorted(resindices):
                        atom_indices.extend(context.solvent.record_by_resindex[resindex].atom_indices)
                else:
                    for atom_index, resindex in zip(
                        context.solvent.oxygen_atom_indices,
                        context.solvent.atom_to_resindex_oxygen,
                    ):
                        if resindex in resindices:
                            if 0 <= atom_index < context.solvent.n_atoms:
                                atom_indices.append(int(atom_index))
                if atom_indices:
                    atom_indices = [idx for idx in atom_indices if 0 <= idx < context.solvent.n_atoms]
                current_seed_atoms = context.universe.atoms[atom_indices]
        result = set()
        for shell in shell_sets:
            result |= shell
        return result

    raise ValueError(f"Unsupported node type: {node.type}")
