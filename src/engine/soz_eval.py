"""SOZ evaluation logic.

A SOZ definition is a logic tree whose leaves are distance criteria and whose inner nodes
combine solvent sets with AND / OR / NOT. Evaluation is vectorised: a node yields a boolean
mask over the solvent molecules, so set algebra is a numpy operation and per-frame cost is
dominated by one neighbour search per leaf.

Leaf semantics
--------------
``distance``  a solvent molecule is in the zone when its probe lies within ``cutoff`` of
              *any* atom of the selection (minimum image under periodic boundaries;
              Lazaridis, J. Phys. Chem. B 102, 3531, 1998 for the zone concept).
``shell``     ``cutoffs = [r1, r2, ...]``. With ``mode="chained"`` (the v2.0 behaviour, kept
              as the default so existing projects reproduce) shell *k* contains solvent within
              ``r_k`` of the molecules of shell *k-1*, so the zone reaches up to ``sum(r_k)``
              from the selection. With ``mode="radial"`` every ``r_k`` is a distance from the
              selection itself and the union is simply the zone of radius ``max(r_k)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import MDAnalysis as mda

from engine.models import SOZNode
from engine.periodic import Box, box_from_dimensions
from engine.resolver import ResolvedSelection
from engine.solvent import SolventUniverse, resolve_probe_mode, solvent_items, zone_mask
from engine.units import to_internal_length

_SEED_KEYS = ("selection_label", "seed_label", "seed")
_PARAMS = {
    "distance": {"selection_label", "seed_label", "seed", "cutoff", "unit", "probe_mode", "atom_mode"},
    "shell": {"selection_label", "seed_label", "seed", "cutoffs", "unit", "probe_mode", "atom_mode", "mode"},
    "and": set(),
    "or": set(),
    "not": set(),
}
SHELL_MODES = ("chained", "radial")


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
        """Raw ``[a, b, c, alpha, beta, gamma]`` of the current frame, or None (legacy accessor)."""
        dims = self.universe.trajectory.ts.dimensions
        if dims is None:
            return None
        if np.all(np.array(dims[:3]) > 0):
            return dims
        return None

    @property
    def box(self) -> Optional[Box]:
        return box_from_dimensions(self.universe.trajectory.ts.dimensions)


def _resolve_selection_label(node: SOZNode) -> str:
    params = node.params
    label = params.get("selection_label") or params.get("seed_label") or params.get("seed")
    if not label:
        raise ValueError(f"Node of type '{node.type}' missing selection_label")
    return label


class _Items:
    """Per-frame cache of solvent probe items by probe mode."""

    def __init__(self, solvent: SolventUniverse, positions: np.ndarray, box: Optional[Box]):
        self.solvent = solvent
        self.positions = positions
        self.box = box
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        if mode not in self._cache:
            self._cache[mode] = solvent_items(self.solvent, mode, self.positions, self.box)
        return self._cache[mode]


def _leaf_distance(node: SOZNode, ctx: EvaluationContext, positions: np.ndarray, box: Optional[Box],
                   items: _Items) -> np.ndarray:
    label = _resolve_selection_label(node)
    cutoff = to_internal_length(float(node.params.get("cutoff", 3.5)), node.params.get("unit", "A"))
    mode = resolve_probe_mode(node.params.get("probe_mode", node.params.get("atom_mode", "probe")),
                              ctx.solvent.probe.position)
    seed_idx = np.asarray(ctx.selections[label].group.indices, dtype=np.int64)
    pos, owner = items.get(mode)
    return zone_mask(positions[seed_idx], pos, owner, cutoff, box, ctx.solvent.layout.n_solvent)


def _leaf_shell(node: SOZNode, ctx: EvaluationContext, positions: np.ndarray, box: Optional[Box],
                items: _Items) -> np.ndarray:
    label = _resolve_selection_label(node)
    unit = node.params.get("unit", "A")
    cutoffs = [to_internal_length(float(v), unit) for v in node.params.get("cutoffs", [3.5])]
    shell_mode = str(node.params.get("mode", "chained")).lower()
    mode = resolve_probe_mode(node.params.get("probe_mode", node.params.get("atom_mode", "probe")),
                              ctx.solvent.probe.position)
    n_solv = ctx.solvent.layout.n_solvent
    seed_idx = np.asarray(ctx.selections[label].group.indices, dtype=np.int64)
    seed_pos = positions[seed_idx]
    pos, owner = items.get(mode)
    if shell_mode == "radial":
        return zone_mask(seed_pos, pos, owner, max(cutoffs), box, n_solv)
    union = np.zeros(n_solv, dtype=bool)
    current = seed_pos
    for cutoff in cutoffs:
        members = zone_mask(current, pos, owner, cutoff, box, n_solv) & ~union
        union |= members
        if not members.any():
            current = np.empty((0, 3))
        else:
            current = pos[members[owner]]
    return union


def evaluate_mask_at(node: SOZNode, ctx: EvaluationContext, positions: np.ndarray, box: Optional[Box],
                     items: Optional[_Items] = None) -> np.ndarray:
    """Boolean mask over solvent ordinals for ``node`` given one frame's coordinates."""
    items = items or _Items(ctx.solvent, positions, box)
    n_solv = ctx.solvent.layout.n_solvent
    if node.type in ("and", "or"):
        if not node.children:
            return np.zeros(n_solv, dtype=bool)
        masks = [evaluate_mask_at(c, ctx, positions, box, items) for c in node.children]
        out = masks[0].copy()
        for m in masks[1:]:
            out = (out & m) if node.type == "and" else (out | m)
        return out
    if node.type == "not":
        if not node.children:
            return np.zeros(n_solv, dtype=bool)
        return ~evaluate_mask_at(node.children[0], ctx, positions, box, items)
    if node.type == "distance":
        return _leaf_distance(node, ctx, positions, box, items)
    if node.type == "shell":
        return _leaf_shell(node, ctx, positions, box, items)
    raise ValueError(f"Unsupported node type: {node.type}")


def evaluate_mask(node: SOZNode, context: EvaluationContext) -> np.ndarray:
    """Mask for the universe's current frame."""
    ts = context.universe.trajectory.ts
    return evaluate_mask_at(node, context, ts.positions, context.box)


def evaluate_node(node: SOZNode, context: EvaluationContext) -> Set[int]:
    """Evaluate one SOZ logic-tree node and return matching solvent resindices (current frame)."""
    mask = evaluate_mask(node, context)
    return {int(r) for r in context.solvent.layout.resindex_by_ord[mask]}


# --------------------------------------------------------------------------- validation
def validate_node(node: SOZNode, selection_labels: Optional[Set[str]] = None,
                  path: str = "root") -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for one SOZ tree.

    v2.0.0 silently ignored unknown parameters and silently substituted defaults (a shell node
    with ``distance: 5.0`` instead of ``cutoffs`` ran with 3.5 A). Everything that would change the
    meaning of a definition is reported here.
    """
    errors: List[str] = []
    warnings: List[str] = []
    allowed = _PARAMS.get(node.type)
    if allowed is None:
        errors.append(f"{path}: unsupported node type '{node.type}'")
        return errors, warnings
    unknown = sorted(set(node.params) - allowed)
    if unknown:
        hint = ""
        if node.type == "shell" and {"distance", "cutoff"} & set(unknown) and "cutoffs" not in node.params:
            hint = " (a shell node takes 'cutoffs': [r1, r2, ...]; the 3.5 A default is being used)"
        warnings.append(f"{path} ({node.type}): parameter(s) {', '.join(unknown)} are ignored{hint}")
    if node.type == "distance":
        if "cutoff" not in node.params:
            warnings.append(f"{path} (distance): no 'cutoff' given; default 3.5 A is used")
        else:
            try:
                if float(node.params["cutoff"]) <= 0:
                    errors.append(f"{path} (distance): cutoff must be positive")
            except (TypeError, ValueError):
                errors.append(f"{path} (distance): cutoff is not a number")
    if node.type == "shell":
        cutoffs = node.params.get("cutoffs")
        if cutoffs is None:
            warnings.append(f"{path} (shell): no 'cutoffs' given; default [3.5] A is used")
        else:
            try:
                vals = [float(v) for v in cutoffs]
                if not vals or any(v <= 0 for v in vals):
                    errors.append(f"{path} (shell): cutoffs must be positive numbers")
                elif len(vals) > 1 and str(node.params.get("mode", "chained")).lower() == "chained":
                    warnings.append(
                        f"{path} (shell): chained shells - shell k is measured from the molecules of shell k-1, so the "
                        f"zone reaches up to {sum(vals):g} {node.params.get('unit', 'A')} from the selection. "
                        "Set mode='radial' for distances measured from the selection itself."
                    )
            except (TypeError, ValueError):
                errors.append(f"{path} (shell): cutoffs must be a list of numbers")
        mode = str(node.params.get("mode", "chained")).lower()
        if mode not in SHELL_MODES:
            errors.append(f"{path} (shell): mode must be one of {', '.join(SHELL_MODES)}")
    if node.type in ("distance", "shell"):
        label = next((node.params[k] for k in _SEED_KEYS if node.params.get(k)), None)
        if not label:
            errors.append(f"{path} ({node.type}): missing selection_label")
        elif selection_labels is not None and label not in selection_labels:
            errors.append(f"{path} ({node.type}): selection '{label}' is not defined")
    if node.type in ("and", "or") and not node.children:
        warnings.append(f"{path} ({node.type}): no children; the zone is always empty")
    if node.type == "not" and len(node.children) != 1:
        warnings.append(f"{path} (not): expects exactly one child (found {len(node.children)})")
    for i, child in enumerate(node.children):
        e, w = validate_node(child, selection_labels, f"{path}/{node.type}[{i}]")
        errors.extend(e)
        warnings.extend(w)
    return errors, warnings


def max_cutoff_angstrom(node: SOZNode) -> float:
    """Largest single distance criterion in the tree (A); used for box-size checks."""
    best = 0.0
    if node.type == "distance":
        best = to_internal_length(float(node.params.get("cutoff", 3.5)), node.params.get("unit", "A"))
    elif node.type == "shell":
        unit = node.params.get("unit", "A")
        vals = [to_internal_length(float(v), unit) for v in node.params.get("cutoffs", [3.5])]
        best = max(vals) if vals else 0.0
    for child in node.children:
        best = max(best, max_cutoff_angstrom(child))
    return best
