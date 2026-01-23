"""Selection resolution helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
import re

import MDAnalysis as mda

from engine.models import SelectionSpec

logger = logging.getLogger("sozlab")


@dataclass
class ResolvedSelection:
    label: str
    group: mda.core.groups.AtomGroup
    selection_string: str


def _build_selection(spec: SelectionSpec) -> str:
    parts = []
    if spec.resid is not None:
        parts.append(f"resid {spec.resid}")
    if spec.resname:
        parts.append(f"resname {spec.resname}")
    if spec.atomname:
        parts.append(f"name {spec.atomname}")
    if spec.segid:
        parts.append(f"segid {spec.segid}")
    if spec.chain:
        parts.append(f"chainID {spec.chain}")
    return " and ".join(parts)


_INDEX_STOP_TOKENS = {"and", "or", "not", "(", ")"}
_INDEX_PATTERN = re.compile(r"\bindex\b", re.IGNORECASE)


def _shift_index_token(token: str, shift: int) -> str:
    if "," in token:
        return ",".join(_shift_index_token(part, shift) for part in token.split(","))
    if token.isdigit():
        return str(int(token) + shift)
    if ":" in token:
        left, right = token.split(":", 1)
        if left.isdigit() and right.isdigit():
            return f"{int(left) + shift}:{int(right) + shift}"
    if "-" in token:
        left, right = token.split("-", 1)
        if left.isdigit() and right.isdigit():
            return f"{int(left) + shift}-{int(right) + shift}"
    return token


def _shift_index_selection(selection: str, shift: int) -> str:
    spaced = selection.replace("(", " ( ").replace(")", " ) ")
    tokens = spaced.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.lower() == "index":
            out.append(tok)
            i += 1
            while i < len(tokens):
                nxt = tokens[i]
                if nxt.lower() in _INDEX_STOP_TOKENS:
                    break
                out.append(_shift_index_token(nxt, shift))
                i += 1
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def _looks_one_based(indices, n_atoms: int) -> bool:
    if indices.size == 0:
        return False
    min_idx = int(indices.min())
    max_idx = int(indices.max())
    return min_idx >= 1 and max_idx == n_atoms


def resolve_selection(universe: mda.Universe, spec: SelectionSpec) -> ResolvedSelection:
    n_atoms = len(universe.atoms)
    if spec.selection:
        selection = spec.selection
        if selection.strip().lower().startswith(("and ", "or ", "not ")):
            raise ValueError(
                f"Selection '{spec.label}' starts with an operator: {selection!r}"
            )
        if selection:
            logger.info("Selection %s (raw): %s", spec.label, selection)
        try:
            group = universe.select_atoms(selection)
        except IndexError:
            if _INDEX_PATTERN.search(selection):
                shifted = _shift_index_selection(selection, -1)
                if shifted != selection:
                    logger.info(
                        "Selection %s shifted for index fix: %s", spec.label, shifted
                    )
                    group = universe.select_atoms(shifted)
                    selection = shifted
                else:
                    raise
            else:
                raise
        if _INDEX_PATTERN.search(selection) and _looks_one_based(group.indices, n_atoms):
            shifted = _shift_index_selection(selection, -1)
            if shifted != selection:
                logger.info(
                    "Selection %s shifted for one-based indices: %s", spec.label, shifted
                )
                group = universe.select_atoms(shifted)
                selection = shifted
    elif spec.atom_indices:
        raw_indices = [int(idx) for idx in spec.atom_indices]
        best_shift = 0
        best_valid = -1
        for shift in (0, -1):
            valid = sum(1 for idx in raw_indices if 0 <= idx + shift < n_atoms)
            if valid > best_valid:
                best_valid = valid
                best_shift = shift
        indices = [idx + best_shift for idx in raw_indices if 0 <= idx + best_shift < n_atoms]
        if not indices:
            raise ValueError(
                f"Selection '{spec.label}' atom_indices are out of bounds for {n_atoms} atoms"
            )
        selection = f"index {' '.join(str(idx) for idx in indices)}"
        group = universe.atoms[indices]
    elif spec.pdb_serials:
        selection = "bynum " + " ".join(str(num) for num in spec.pdb_serials)
        group = universe.select_atoms(selection)
    else:
        selection = _build_selection(spec)
        if not selection:
            raise ValueError(f"Selection '{spec.label}' has no selection data")
        group = universe.select_atoms(selection)

    if len(group) == 0:
        raise ValueError(f"Selection '{spec.label}' resolved to 0 atoms")

    # Guard against out-of-bounds indices from upstream parsers.
    indices = group.indices
    if _looks_one_based(indices, n_atoms):
        indices = indices - 1
        indices = indices[(indices >= 0) & (indices < n_atoms)]
        if indices.size == 0:
            raise ValueError(
                f"Selection '{spec.label}' resolved to 0 atoms after index fix"
            )
        group = universe.atoms[indices]
    else:
        invalid = (indices < 0) | (indices >= n_atoms)
        if invalid.any():
            indices = indices[~invalid]
            if indices.size == 0:
                raise ValueError(
                    f"Selection '{spec.label}' resolved to 0 atoms after index fix"
                )
            group = universe.atoms[indices]

    if spec.require_unique and len(group) != 1:
        raise ValueError(
            f"Selection '{spec.label}' expected unique atom but resolved {len(group)} atoms"
        )

    if spec.expect_count is not None and len(group) != spec.expect_count:
        raise ValueError(
            f"Selection '{spec.label}' expected {spec.expect_count} atoms but resolved {len(group)}"
        )

    return ResolvedSelection(label=spec.label, group=group, selection_string=selection)


def resolve_selection_from_label(
    universe: mda.Universe,
    selection_label: str,
    selections: dict[str, SelectionSpec],
) -> ResolvedSelection:
    if selection_label not in selections:
        raise KeyError(f"Selection label '{selection_label}' not found")
    return resolve_selection(universe, selections[selection_label])


# Backwards-compatible aliases (kept for legacy calls/tests).
ResolvedSeed = ResolvedSelection


def resolve_seed(universe: mda.Universe, spec: SelectionSpec) -> ResolvedSelection:
    return resolve_selection(universe, spec)


def resolve_seed_from_label(
    universe: mda.Universe,
    seed_label: str,
    seeds: dict[str, SelectionSpec],
) -> ResolvedSelection:
    return resolve_selection_from_label(universe, seed_label, seeds)
