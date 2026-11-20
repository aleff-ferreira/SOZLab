"""Solvent identification and per-frame solvent geometry.

Identity
--------
A solvent molecule is identified by its **residue index** in the topology (unique by
construction). Human-readable IDs follow ``resname:resid:segid`` as before, but when two
molecules share that triple (residue numbers wrap at 9,999 in PDB and 99,999 in GRO files)
every member of the colliding group gets a ``#<resindex>`` suffix, so exported IDs are
always unique. IDs that were already unique are unchanged.

Geometry
--------
Probe positions for every solvent molecule are computed in a single vectorised pass:
the oxygen/probe atom itself (``atom``), every atom (``all``), or the centre of geometry or
mass of the molecule's probe atoms (``cog`` / ``com``). Centres use the minimum-image
convention relative to a reference atom, so no bond information is needed and molecules split
across the periodic boundary are handled correctly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib import distances

from engine.models import SolventConfig
from engine.periodic import Box, box_from_dimensions, group_layout, minimum_image, molecule_centers


@dataclass
class SolventRecord:
    resindex: int
    resid: int
    resname: str
    segid: str
    atom_indices: List[int]
    unique_id: str = ""

    @property
    def stable_id(self) -> str:
        """Legacy ``resname:resid:segid`` identifier (not guaranteed unique)."""
        segid = self.segid if self.segid else "-"
        return f"{self.resname}:{self.resid}:{segid}"

    @property
    def display_id(self) -> str:
        """Identifier used in exports: unique even when residue numbers repeat."""
        return self.unique_id or self.stable_id


@dataclass
class SolventProbe:
    selection: str
    position: str
    atoms: mda.core.groups.AtomGroup
    atom_indices: List[int]
    atom_to_resindex: List[int]
    resindex_to_atom_indices: Dict[int, List[int]]


@dataclass
class _Layout:
    """Array view of the solvent used by the vectorised geometry (built lazily)."""

    n_solvent: int
    resindex_by_ord: np.ndarray  # (M,)
    ord_by_resindex: Dict[int, int]
    probe_idx: np.ndarray  # universe atom indices, grouped by owner
    probe_owner: np.ndarray  # owner ordinal per probe atom
    probe_starts: np.ndarray
    probe_counts: np.ndarray
    probe_masses: Optional[np.ndarray]
    all_idx: np.ndarray
    all_owner: np.ndarray


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
    notes: List[str] = field(default_factory=list)
    _layout_cache: Optional[_Layout] = field(default=None, repr=False, compare=False)

    def all_resindex_set(self) -> set[int]:
        return set(self.solvent_resindices)

    @property
    def layout(self) -> _Layout:
        if self._layout_cache is None:
            self._layout_cache = _build_layout(self)
        return self._layout_cache


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


# --------------------------------------------------------------------------- layout & positions
def _build_layout(solvent: SolventUniverse) -> _Layout:
    resindex_by_ord = np.asarray(solvent.solvent_resindices, dtype=np.int64)
    ord_by_resindex = {int(r): i for i, r in enumerate(resindex_by_ord)}

    probe_atoms = solvent.probe.atoms
    p_idx = np.asarray(probe_atoms.indices, dtype=np.int64)
    p_res = np.asarray(solvent.probe.atom_to_resindex, dtype=np.int64)
    p_owner = np.fromiter((ord_by_resindex[int(r)] for r in p_res), dtype=np.int64, count=len(p_res))
    order, starts, counts = group_layout(p_owner)
    p_idx, p_owner = p_idx[order], p_owner[order]

    masses = None
    try:
        masses = np.asarray(solvent.atoms_all.universe.atoms[p_idx].masses, dtype=np.float64)
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            masses = None
    except (NoDataError, AttributeError):
        masses = None

    a_idx = np.asarray(solvent.atoms_all.indices, dtype=np.int64)
    a_res = np.asarray(solvent.atom_to_resindex_all, dtype=np.int64)
    a_owner = np.fromiter((ord_by_resindex[int(r)] for r in a_res), dtype=np.int64, count=len(a_res))
    return _Layout(
        n_solvent=len(resindex_by_ord),
        resindex_by_ord=resindex_by_ord,
        ord_by_resindex=ord_by_resindex,
        probe_idx=p_idx,
        probe_owner=p_owner,
        probe_starts=starts,
        probe_counts=counts,
        probe_masses=masses,
        all_idx=a_idx,
        all_owner=a_owner,
    )


def _as_box(box) -> Optional[Box]:
    if box is None or isinstance(box, Box):
        return box
    return box_from_dimensions(box)


def solvent_items(
    solvent: SolventUniverse,
    mode: str,
    positions: np.ndarray,
    box: Optional[Box],
) -> Tuple[np.ndarray, np.ndarray]:
    """Probe items of every solvent molecule for the current frame.

    Returns ``(positions (n, 3), owner ordinal (n,))``. ``positions`` is the full-system
    coordinate array of the frame (``ts.positions``).
    """
    lay = solvent.layout
    mode_norm = mode.strip().lower()
    if mode_norm == "all":
        return positions[lay.all_idx], lay.all_owner
    if mode_norm == "atom":
        return positions[lay.probe_idx], lay.probe_owner
    if mode_norm in ("com", "cog"):
        weights = None
        if mode_norm == "com":
            if lay.probe_masses is None:
                raise ValueError("Probe position 'com' requires atom masses; add masses to topology or use COG.")
            weights = lay.probe_masses
        centers = molecule_centers(positions[lay.probe_idx], lay.probe_starts, lay.probe_counts, weights, box)
        return centers, lay.probe_owner[lay.probe_starts]
    raise ValueError(f"Unsupported probe mode: {mode}")


def solvent_positions(
    solvent: SolventUniverse,
    mode: str,
    resindices: Iterable[int] | None = None,
    box=None,
) -> Tuple[np.ndarray, List[int]]:
    """Legacy-compatible probe positions for the universe's *current* frame.

    Returns ``(positions, owner resindex per item)``. ``box`` defaults to the current frame's
    dimensions.
    """
    universe = solvent.atoms_all.universe
    ts = universe.trajectory.ts
    box_obj = _as_box(box if box is not None else ts.dimensions)
    pos, owner = solvent_items(solvent, mode, ts.positions, box_obj)
    resindex_owner = solvent.layout.resindex_by_ord[owner]
    if resindices is None:
        return np.asarray(pos), resindex_owner.astype(int).tolist()
    res_set = np.fromiter((int(i) for i in resindices), dtype=np.int64)
    if res_set.size == 0:
        return np.empty((0, 3)), []
    keep = np.isin(resindex_owner, res_set)
    if not keep.any():
        return np.empty((0, 3)), []
    return np.asarray(pos)[keep], resindex_owner[keep].astype(int).tolist()


def zone_mask(
    seed_positions: np.ndarray,
    item_positions: np.ndarray,
    owner: np.ndarray,
    cutoff: float,
    box: Optional[Box],
    n_solvent: int,
) -> np.ndarray:
    """Boolean mask over solvent ordinals: any item within ``cutoff`` of any seed atom."""
    mask = np.zeros(n_solvent, dtype=bool)
    if len(seed_positions) == 0 or len(item_positions) == 0:
        return mask
    dims = None if box is None else np.asarray(box.dims, dtype=np.float32)
    pairs = distances.capped_distance(
        np.asarray(seed_positions, dtype=np.float32),
        np.asarray(item_positions, dtype=np.float32),
        max_cutoff=float(cutoff),
        box=dims,
        return_distances=False,
    )
    if len(pairs):
        mask[owner[pairs[:, 1]]] = True
    return mask


def distance_resindices(
    seed_positions: np.ndarray,
    solvent_positions: np.ndarray,
    atom_map: List[int],
    cutoff_ang: float,
    box,
) -> set[int]:
    """Return solvent resindices within *cutoff_ang* of any selection atom.

    Uses MDAnalysis ``capped_distance`` (cell-list neighbour search; Michaud-Agrawal et al.,
    J. Comput. Chem. 32, 2011) with the minimum-image convention. Kept for API compatibility;
    the engine itself uses :func:`zone_mask`.
    """
    if len(seed_positions) == 0 or len(solvent_positions) == 0:
        return set()
    owner_map = np.asarray(atom_map, dtype=np.int64)
    dims = None if box is None else np.asarray(getattr(box, "dims", box), dtype=np.float32)
    pairs = distances.capped_distance(
        np.asarray(seed_positions, dtype=np.float32),
        np.asarray(solvent_positions, dtype=np.float32),
        max_cutoff=float(cutoff_ang),
        box=dims,
        return_distances=False,
    )
    if len(pairs) == 0:
        return set()
    return {int(r) for r in np.unique(owner_map[pairs[:, 1]])}


# --------------------------------------------------------------------------- species detection
_WATER_NAME_PREFIXES = ("O", "H", "D")
_VIRTUAL_PREFIXES = ("M", "EP", "LP", "DUM")


def _classify_residue(atom_names: Sequence[str], elements: Optional[Sequence[str]]) -> str:
    n = len(atom_names)
    if n == 1:
        return "ion"
    names = [str(a).strip().upper() for a in atom_names]
    if elements is not None and all(str(e).strip() for e in elements):
        elems = [str(e).strip().upper() for e in elements]
    else:
        elems = [("EP" if nm.startswith(_VIRTUAL_PREFIXES[1:]) else nm[:1]) for nm in names]
    counts = Counter(e for e in elems if e not in ("M", "EP", "LP", "DUM", "X"))
    heavy_h = counts.get("H", 0) + counts.get("D", 0)
    if 3 <= n <= 5 and counts.get("O", 0) == 1 and heavy_h == 2 and set(counts) <= {"O", "H", "D"}:
        return "water"
    return "other"


def detect_solvent_species(universe: mda.Universe, max_species: int = 40) -> List[Dict[str, object]]:
    """Group residues by name and classify by composition: water-like, ion or other.

    Residue names are unreliable (``TIP3``, ``TP3``, ``T3P``, ``SPC``, ``OPC``, ``HOH`` ...);
    composition is not. A residue is water-like when it has exactly one O and two H(D) atoms
    (plus optional virtual sites), as in 3-, 4- and 5-site rigid models.
    """
    out: List[Dict[str, object]] = []
    resnames = np.asarray(universe.residues.resnames).astype(str)
    uniq, first_idx, counts = np.unique(resnames, return_index=True, return_counts=True)
    has_elements = hasattr(universe.atoms, "elements")
    for name, first, count in sorted(zip(uniq, first_idx, counts), key=lambda t: -int(t[2])):
        residue = universe.residues[int(first)]
        atoms = residue.atoms
        elems = list(atoms.elements) if has_elements else None
        kind = _classify_residue(list(atoms.names), elems)
        if kind == "other" and len(atoms) > 40:
            continue  # polymer residues are not solvent candidates
        out.append(
            {
                "resname": str(name),
                "n_residues": int(count),
                "atoms_per_residue": int(len(atoms)),
                "atom_names": [str(a) for a in atoms.names],
                "kind": kind,
            }
        )
        if len(out) >= max_species:
            break
    return out


def suggest_water_resnames(universe: mda.Universe) -> List[str]:
    return [sp["resname"] for sp in detect_solvent_species(universe) if sp["kind"] == "water"]


# --------------------------------------------------------------------------- integrity
def _atom_count_outliers(solvent: SolventUniverse, dominance: float = 0.8) -> Dict[str, Dict[str, object]]:
    """Residues whose atom count differs from the dominant count of residues with the same name.

    File parsers merge consecutive molecules that share residue number and name into one residue,
    so a ``SOL`` residue with 6 atoms among thousands with 3 is almost certainly two waters.
    """
    lay = solvent.layout
    n_atoms = np.bincount(lay.all_owner, minlength=lay.n_solvent)
    names = np.asarray([solvent.record_by_resindex[int(r)].resname for r in lay.resindex_by_ord])
    out: Dict[str, Dict[str, object]] = {}
    for name in np.unique(names):
        sel = np.flatnonzero(names == name)
        vals, cnt = np.unique(n_atoms[sel], return_counts=True)
        mode = int(vals[np.argmax(cnt)])
        if cnt.max() / len(sel) < dominance:
            continue
        odd = sel[n_atoms[sel] != mode]
        if len(odd):
            ex = [f"{solvent.record_by_resindex[int(lay.resindex_by_ord[o])].display_id} ({int(n_atoms[o])} atoms)"
                  for o in odd[:3]]
            out[str(name)] = {"expected_atoms": mode, "n_outliers": int(len(odd)), "examples": ex}
    return out


def integrity_report(solvent: SolventUniverse, positions: np.ndarray, box: Optional[Box],
                     sample: int = 20000) -> Dict[str, object]:
    """Check that each solvent molecule is geometrically intact in one frame.

    The largest minimum-image distance of an atom from its molecule's first atom must not exceed
    what a chain of ``n`` atoms can span (1.7 A per bond, +1 A tolerance). Larger values mean the
    topology and trajectory do not describe the same atoms in the same order, that consecutive molecules
    were merged into one residue by the file parser (identical residue number and name), or that the
    molecule is larger than half the box. Without a valid box molecules split by wrapping cannot be
    recognised, so the check is skipped rather than guessed.
    """
    lay = solvent.layout
    idx = lay.all_idx
    owner = lay.all_owner
    outliers = _atom_count_outliers(solvent)
    if box is None:
        return {"n_checked": 0, "n_broken": 0, "max_extent": 0.0, "examples": [], "skipped": "no valid box",
                "atom_count_outliers": outliers}
    if len(idx) == 0:
        return {"n_checked": 0, "n_broken": 0, "max_extent": 0.0, "examples": [], "atom_count_outliers": outliers}
    order, starts, counts = group_layout(owner)
    pos = np.asarray(positions[idx][order], dtype=np.float64)
    ref = np.repeat(pos[starts], counts, axis=0)
    disp = minimum_image(pos - ref, box)
    extent = np.maximum.reduceat(np.linalg.norm(disp, axis=1), starts)
    bound = 1.7 * (counts - 1) + 1.0
    bad = np.flatnonzero(extent > bound)
    bad_ord = owner[order][starts[bad]]
    examples = []
    for o, b in zip(bad_ord[:5], bad[:5]):
        rec = solvent.record_by_resindex[int(lay.resindex_by_ord[int(o)])]
        examples.append(f"{rec.display_id} ({int(counts[b])} atoms)")
    return {"n_checked": int(len(extent)), "n_broken": int(len(bad)), "max_extent": float(extent.max()),
            "examples": examples, "atom_count_outliers": outliers}


# --------------------------------------------------------------------------- construction
def build_solvent(universe: mda.Universe, config: SolventConfig) -> SolventUniverse:
    """Build the solvent universe from topology and config.

    The *probe* is the representative atom (or centre) used for all distance calculations -
    typically the water oxygen, following the convention of hydration-shell analysis (Impey
    et al., J. Phys. Chem. 87, 1983).
    """
    resnames = list(config.water_resnames)
    if config.include_ions:
        resnames += list(config.ion_resnames)

    if not resnames:
        raise ValueError("No solvent resnames configured")

    selection = _resname_selection(resnames)
    residues = universe.select_atoms(selection).residues

    if len(residues) == 0:
        found = suggest_water_resnames(universe)
        hint = f" Water-like residues found in the topology: {', '.join(found)}." if found else ""
        raise ValueError(
            f"Solvent selection resolved to 0 residues (resnames: {', '.join(resnames)}).{hint}"
        )

    notes: List[str] = []
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
    resindices_all = np.asarray(residues.resindices, dtype=np.int64)

    # Monatomic residues (ions) have no "O"; their only atom is the natural probe.
    covered = set(int(r) for r in np.unique(probe_atoms.resindices)) if len(probe_atoms) else set()
    sizes = np.asarray([len(r.atoms) for r in residues]) if len(residues) < 5000 else \
        np.bincount(np.asarray(atoms_all.resindices, dtype=np.int64))[resindices_all]
    mono = [int(r) for r, s in zip(resindices_all, sizes) if s == 1 and int(r) not in covered]
    if mono:
        extra = universe.residues[mono].atoms
        merged = np.unique(np.concatenate([np.asarray(probe_atoms.indices, dtype=np.int64),
                                           np.asarray(extra.indices, dtype=np.int64)]))
        probe_atoms = universe.atoms[merged]
        notes.append(
            f"{len(mono)} single-atom solvent residues (ions) use their only atom as probe "
            f"(probe selection '{probe_selection}' matches no atom in them)."
        )

    record_by_resindex: Dict[int, SolventRecord] = {}
    solvent_resindices: List[int] = []
    for residue in residues:
        resindex = int(residue.resindex)
        solvent_resindices.append(resindex)
        record_by_resindex[resindex] = SolventRecord(
            resindex=resindex,
            resid=int(residue.resid),
            resname=str(residue.resname),
            segid=str(residue.segid) if residue.segid else "",
            atom_indices=[int(i) for i in residue.atoms.indices],
        )

    solvent_resindices.sort(key=lambda idx: (
        record_by_resindex[idx].resname,
        record_by_resindex[idx].resid,
        record_by_resindex[idx].segid,
        record_by_resindex[idx].resindex,
    ))

    # unique display identifiers
    counts = Counter(rec.stable_id for rec in record_by_resindex.values())
    n_dup = 0
    for rec in record_by_resindex.values():
        if counts[rec.stable_id] > 1:
            rec.unique_id = f"{rec.stable_id}#{rec.resindex}"
            n_dup += 1
        else:
            rec.unique_id = rec.stable_id
    if n_dup:
        notes.append(
            f"{n_dup} solvent molecules share a resname:resid:segid identifier (residue numbers wrap); "
            "exports use '<id>#<resindex>' for them."
        )

    atom_to_resindex_all = [int(r) for r in atoms_all.resindices]
    atom_to_resindex_probe = [int(r) for r in probe_atoms.resindices]
    probe_atom_indices = [int(i) for i in probe_atoms.indices]

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
        names = sorted({record_by_resindex[i].resname for i in missing_probe})
        raise ValueError(
            "Probe selection did not match atoms for all solvent residues. "
            f"Missing {len(missing_probe)} residues (resnames: {', '.join(names)}; "
            f"first resindices: {missing_probe[:10]})."
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
        atom_index_shift=0,
        notes=notes,
    )
