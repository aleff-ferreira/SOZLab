"""SOZ evaluation logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np
import MDAnalysis as mda

from engine.models import SOZNode
from engine.units import to_internal_length
from engine.resolver import ResolvedSelection
from engine.solvent import SolventUniverse, distance_resindices, resolve_probe_mode, solvent_positions


@dataclass
class EvaluationContext:
    universe: mda.Universe
    solvent: SolventUniverse
    selections: Dict[str, ResolvedSelection] | None = None

    def __post_init__(self) -> None:
        if self.selections is None:
            self.selections = {}

    @property
    def pbc_box(self):
        dims = self.universe.trajectory.ts.dimensions
        if dims is None:
            return None
        if np.all(np.array(dims[:3]) > 0):
            return dims
        return None


def _resolve_selection_label(node: SOZNode) -> str:
    params = node.params
    label = params.get("selection_label") or params.get("seed_label") or params.get("seed")
    if not label:
        raise ValueError(f"Node of type '{node.type}' missing selection_label")
    return label


def evaluate_node(node: SOZNode, context: EvaluationContext) -> Set[int]:
    """Evaluate one SOZ logic-tree node and return matching solvent resindices.

    Leaf nodes implement the solvent occupancy zone distance criterion:
    all solvent residues whose probe atom lies within *cutoff* of any
    atom in the selection (Lazaridis, J. Phys. Chem. B 102, 1998).
    Boolean nodes (and/or/not) compose child results via set algebra.
    """
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
        atom_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        cutoff_internal = to_internal_length(cutoff, unit)
        seed = context.selections[selection_label].group
        mode = resolve_probe_mode(atom_mode, context.solvent.probe.position)
        solvent_pos, atom_map = solvent_positions(context.solvent, mode)
        seed_pos = seed.positions
        return distance_resindices(seed_pos, solvent_pos, atom_map, cutoff_internal, context.pbc_box)

    if node.type == "shell":
        selection_label = _resolve_selection_label(node)
        cutoffs = node.params.get("cutoffs", [3.5])
        unit = node.params.get("unit", "A")
        atom_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
        cutoffs_internal = [to_internal_length(float(value), unit) for value in cutoffs]
        mode = resolve_probe_mode(atom_mode, context.solvent.probe.position)
        seed = context.selections[selection_label].group
        shell_sets: list[set[int]] = []
        current_seed_positions = seed.positions
        for cutoff_ang in cutoffs_internal:
            solvent_pos, atom_map = solvent_positions(context.solvent, mode)
            resindices = distance_resindices(
                current_seed_positions,
                solvent_pos,
                atom_map,
                cutoff_ang,
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
