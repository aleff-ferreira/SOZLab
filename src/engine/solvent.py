"""Solvent identification utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import MDAnalysis as mda

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
class SolventUniverse:
    residues: mda.core.groups.ResidueGroup
    atoms_all: mda.core.groups.AtomGroup
    atoms_oxygen: mda.core.groups.AtomGroup
    oxygen_atom_indices: List[int]
    record_by_resindex: Dict[int, SolventRecord]
    solvent_resindices: List[int]
    atom_to_resindex_all: List[int]
    atom_to_resindex_oxygen: List[int]
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
    oxygen_names = set(config.water_oxygen_names)
    atoms_oxygen = atoms_all.select_atoms("name " + " ".join(sorted(oxygen_names)))
    if config.include_ions and config.ion_resnames:
        ion_sel = _resname_selection(list(config.ion_resnames))
        ion_atoms = atoms_all.select_atoms(ion_sel)
        atoms_oxygen = atoms_oxygen.union(ion_atoms)
    if len(atoms_oxygen) == 0:
        atoms_oxygen = atoms_all

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
    atom_to_resindex_oxygen = [int(atom.resindex) for atom in atoms_oxygen]
    oxygen_atom_indices = [int(idx) + index_shift for idx in atoms_oxygen.indices]

    return SolventUniverse(
        residues=residues,
        atoms_all=atoms_all,
        atoms_oxygen=atoms_oxygen,
        oxygen_atom_indices=oxygen_atom_indices,
        record_by_resindex=record_by_resindex,
        solvent_resindices=solvent_resindices,
        atom_to_resindex_all=atom_to_resindex_all,
        atom_to_resindex_oxygen=atom_to_resindex_oxygen,
        n_atoms=n_atoms,
        atom_index_shift=index_shift,
    )
