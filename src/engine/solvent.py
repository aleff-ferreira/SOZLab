"""Solvent identification utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib import distances

from engine.models import SolventConfig


@dataclass
class SolventRecord:
    resindex: int
    resid: int
    resname: str
    segid: str
    atom_indices: List[int]

    @property
    def stable_id(self) -> str:
        segid = self.segid if self.segid else "-"
        return f"{self.resname}:{self.resid}:{segid}"


@dataclass
class SolventProbe:
    selection: str
    position: str
    atoms: mda.core.groups.AtomGroup
    atom_indices: List[int]
    atom_to_resindex: List[int]
    resindex_to_atom_indices: Dict[int, List[int]]


@dataclass
class SolventUniverse:
    residues: mda.core.groups.ResidueGroup
    atoms_all: mda.core.groups.AtomGroup
    probe: SolventProbe
    record_by_resindex: Dict[int, SolventRecord]
    solvent_resindices: List[int]
    atom_to_resindex_all: List[int]
    n_atoms: int
    atom_index_shift: int = 0

    def all_resindex_set(self) -> set[int]:
        return set(self.solvent_resindices)


def _resname_selection(resnames: List[str]) -> str:
    return " or ".join(f"resname {name}" for name in resnames)


def _best_index_shift(indices: Iterable[int], n_items: int) -> int:
    arr = np.asarray(list(indices), dtype=int)
    if arr.size == 0:
        return 0
    valid_no_shift = np.logical_and(arr >= 0, arr < n_items).sum()
    valid_minus_one = np.logical_and(arr - 1 >= 0, arr - 1 < n_items).sum()
    return -1 if valid_minus_one > valid_no_shift else 0


def _normalize_indices(indices: Iterable[int], n_items: int, shift: int) -> List[int]:
    arr = np.asarray(list(indices), dtype=int)
    if arr.size == 0:
        return []
    shifted = arr + shift
    valid = (shifted >= 0) & (shifted < n_items)
    if not np.any(valid):
        return []
    return shifted[valid].astype(int).tolist()


def _normalize_probe_position(position: str) -> str:
    value = (position or "atom").strip().lower()
    if value in ("atom", "atoms"):
        return "atom"
    if value in ("com", "center_of_mass"):
        return "com"
    if value in ("cog", "center_of_geometry"):
        return "cog"
    raise ValueError(f"Unsupported probe position: {position}")


def resolve_probe_mode(mode: str | None, default_position: str) -> str:
    raw = str(mode or "probe").strip().lower()
    if raw in ("o", "oxygen"):
        return "atom"
    if raw in ("probe", "default"):
        return _normalize_probe_position(default_position)
    if raw in ("atom", "atoms"):
        return "atom"
    if raw in ("all", "all_atoms"):
        return "all"
    if raw in ("com", "center_of_mass"):
        return "com"
    if raw in ("cog", "center_of_geometry"):
        return "cog"
    raise ValueError(f"Unsupported probe mode: {mode}")


def solvent_positions(
    solvent: SolventUniverse,
    mode: str,
    resindices: Iterable[int] | None = None,
) -> Tuple[np.ndarray, List[int]]:
    mode_norm = mode.strip().lower()
    if mode_norm == "all":
        positions = solvent.atoms_all.positions
        atom_map = np.asarray(solvent.atom_to_resindex_all, dtype=int)
    elif mode_norm == "atom":
        positions = solvent.probe.atoms.positions
        atom_map = np.asarray(solvent.probe.atom_to_resindex, dtype=int)
    elif mode_norm in ("com", "cog"):
        res_list = list(resindices) if resindices is not None else list(solvent.solvent_resindices)
        if not res_list:
            return np.empty((0, 3)), []
        if mode_norm == "com":
            try:
                _ = solvent.atoms_all.masses
            except NoDataError as exc:
                raise ValueError(
                    "Probe position 'com' requires atom masses; add masses to topology or use COG."
                ) from exc
        positions_list = []
        mapping: List[int] = []
        for resindex in res_list:
            atom_indices = solvent.probe.resindex_to_atom_indices.get(resindex, [])
            if not atom_indices:
                raise ValueError(
                    f"Probe selection missing atoms for solvent residue resindex {resindex}."
                )
            group = solvent.atoms_all.universe.atoms[atom_indices]
            try:
                if mode_norm == "com":
                    pos = group.center_of_mass(unwrap=True)
                else:
                    pos = group.center_of_geometry(unwrap=True)
            except NoDataError:
                # Fallback if bonds are missing (cannot unwrap)
                if mode_norm == "com":
                    pos = group.center_of_mass()
                else:
                    pos = group.center_of_geometry()
            positions_list.append(pos)
            mapping.append(resindex)
        return np.asarray(positions_list), mapping
    else:
        raise ValueError(f"Unsupported probe mode: {mode}")

    if resindices is None:
        return np.asarray(positions), atom_map.astype(int).tolist()

    res_set = set(int(idx) for idx in resindices)
    if not res_set:
        return np.empty((0, 3)), []
    mask = np.isin(atom_map, list(res_set))
    if not np.any(mask):
        return np.empty((0, 3)), []
    filtered_positions = np.asarray(positions)[mask]
    filtered_map = atom_map[mask].astype(int).tolist()
    return filtered_positions, filtered_map


def distance_resindices(
    seed_positions: np.ndarray,
    solvent_positions: np.ndarray,
    atom_map: List[int],
    cutoff_nm: float,
    box,
) -> set[int]:
    if seed_positions.size == 0 or solvent_positions.size == 0:
        return set()
    try:
        pairs = distances.capped_distance(
            seed_positions,
            solvent_positions,
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
            seed_positions,
            solvent_positions,
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


def build_solvent(universe: mda.Universe, config: SolventConfig) -> SolventUniverse:
    resnames = list(config.water_resnames)
    if config.include_ions:
        resnames += list(config.ion_resnames)

    if not resnames:
        raise ValueError("No solvent resnames configured")

    selection = _resname_selection(resnames)
    residues = universe.select_atoms(selection).residues

    if len(residues) == 0:
        raise ValueError(
            f"Solvent selection resolved to 0 residues (resnames: {', '.join(resnames)})"
        )

    atoms_all = residues.atoms
    probe_selection = (config.probe.selection or "").strip()
    if not probe_selection:
        raise ValueError("Probe selection is empty. Provide a solvent probe selection.")
    probe_position = _normalize_probe_position(config.probe.position)
    try:
        probe_atoms = atoms_all.select_atoms(probe_selection)
    except Exception as exc:
        raise ValueError(f"Probe selection failed: {exc}") from exc

    n_atoms = len(universe.atoms)
    index_shift = 0
    if len(atoms_all) > 0:
        all_indices = atoms_all.indices
        if all_indices.size and int(all_indices.min()) >= 1 and int(all_indices.max()) == n_atoms:
            index_shift = -1

    n_atoms = len(universe.atoms)
    index_shift = _best_index_shift(atoms_all.indices, n_atoms)

    record_by_resindex: Dict[int, SolventRecord] = {}
    solvent_resindices: List[int] = []
    for residue in residues:
        resindex = residue.resindex
        solvent_resindices.append(resindex)
        atom_indices = _normalize_indices(residue.atoms.indices, n_atoms, index_shift)
        record_by_resindex[resindex] = SolventRecord(
            resindex=resindex,
            resid=int(residue.resid),
            resname=str(residue.resname),
            segid=str(residue.segid) if residue.segid else "",
            atom_indices=atom_indices,
        )

    solvent_resindices.sort(key=lambda idx: (
        record_by_resindex[idx].resname,
        record_by_resindex[idx].resid,
        record_by_resindex[idx].segid,
        record_by_resindex[idx].resindex,
    ))

    atom_to_resindex_all = [int(atom.resindex) for atom in atoms_all]
    atom_to_resindex_probe = [int(atom.resindex) for atom in probe_atoms]
    probe_atom_indices = _normalize_indices(probe_atoms.indices, n_atoms, index_shift)

    resindex_to_probe_atom_indices: Dict[int, List[int]] = {idx: [] for idx in solvent_resindices}
    for atom_index, resindex in zip(probe_atom_indices, atom_to_resindex_probe):
        if resindex in resindex_to_probe_atom_indices:
            resindex_to_probe_atom_indices[resindex].append(int(atom_index))
    if len(probe_atom_indices) == 0:
        raise ValueError(
            f"Probe selection '{probe_selection}' resolved to 0 atoms within solvent residues."
        )
    missing_probe = [idx for idx, atoms in resindex_to_probe_atom_indices.items() if not atoms]
    if missing_probe:
        raise ValueError(
            "Probe selection did not match atoms for all solvent residues. "
            f"Missing residues: {missing_probe[:10]}"
        )

    return SolventUniverse(
        residues=residues,
        atoms_all=atoms_all,
        probe=SolventProbe(
            selection=probe_selection,
            position=probe_position,
            atoms=probe_atoms,
            atom_indices=probe_atom_indices,
            atom_to_resindex=atom_to_resindex_probe,
            resindex_to_atom_indices=resindex_to_probe_atom_indices,
        ),
        record_by_resindex=record_by_resindex,
        solvent_resindices=solvent_resindices,
        atom_to_resindex_all=atom_to_resindex_all,
        n_atoms=n_atoms,
        atom_index_shift=index_shift,
    )
