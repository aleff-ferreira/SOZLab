"""Synthetic molecular systems with analytically known answers.

These generators exist so that every scientific quantity SOZLab reports can be
checked against a ground truth that does not depend on another analysis tool:

* :func:`make_scripted_system` - solvent "visitors" that are inside the zone on
  exactly the frames you specify -> exact occupancy, entries/exits, residence.
* :func:`make_ideal_gas_system` - uniformly random solvent every frame ->
  binomial occupancy (mean ``N * V_zone / V_box``) and a flat density map.
* :func:`make_brownian_system` - free Brownian solvent -> exact continuous-time
  survival function for a spherical zone (Carslaw & Jaeger, *Conduction of Heat
  in Solids*, sec. 9.3) and the known discrete-monitoring (stride) bias.
* :func:`make_straddling_water_system` - rigid waters whose atoms are wrapped
  individually across the periodic boundary (raw-GROMACS style) -> tests that
  periodic geometry does not depend on bonds or molecule wholeness.

Units are Angstrom and picoseconds (MDAnalysis base units).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

import MDAnalysis as mda

# TIP3P geometry (Jorgensen et al., J. Chem. Phys. 79, 926, 1983)
_OH = 0.9572
_HOH = np.deg2rad(104.52)
WATER_TEMPLATE = np.array(
    [
        [0.0, 0.0, 0.0],
        [_OH, 0.0, 0.0],
        [_OH * np.cos(_HOH), _OH * np.sin(_HOH), 0.0],
    ],
    dtype=np.float64,
)


def random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly distributed rotation matrices, shape ``(n, 3, 3)``."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rot = np.empty((n, 3, 3))
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def waters_from_oxygens(oxygens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rigid TIP3P waters (O, H1, H2) at the given oxygen positions.

    ``oxygens`` has shape ``(..., 3)``; the result has shape ``(..., 3, 3)``
    (molecule, atom, xyz) with a fresh random orientation per molecule.
    """
    lead = oxygens.shape[:-1]
    flat = oxygens.reshape(-1, 3)
    rot = random_rotations(rng, flat.shape[0])
    atoms = np.einsum("nij,aj->nai", rot, WATER_TEMPLATE)  # (n, 3 atoms, 3)
    atoms += flat[:, None, :]
    return atoms.reshape(*lead, 3, 3)


def _wrap(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    return positions - np.floor(positions / box) * box


def _min_image_dist(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    d = a - b
    d -= np.round(d / box) * box
    return np.linalg.norm(d, axis=-1)


@dataclass
class SyntheticSystem:
    """In-memory trajectory plus the metadata needed to write it to disk."""

    universe: "mda.Universe"
    positions: np.ndarray  # (T, N, 3) float32
    box: np.ndarray  # (3,) orthorhombic lengths, Angstrom
    times: np.ndarray  # (T,) picoseconds
    truth: Dict[str, object] = field(default_factory=dict)
    angles: Tuple[float, float, float] = (90.0, 90.0, 90.0)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    def write(self, directory: str, stem: str = "synthetic", traj_format: str = "xtc") -> Tuple[str, str]:
        """Write ``<stem>.gro`` (frame 0) and ``<stem>.<traj_format>``; return both paths."""
        os.makedirs(directory, exist_ok=True)
        gro = os.path.join(directory, f"{stem}.gro")
        traj = os.path.join(directory, f"{stem}.{traj_format}")
        u = self.universe
        u.trajectory[0]
        u.atoms.write(gro)
        with mda.Writer(traj, n_atoms=len(u.atoms)) as writer:
            for i in range(self.n_frames):
                ts = u.trajectory[i]
                ts.dimensions = [*self.box, *self.angles]
                ts.time = float(self.times[i])
                writer.write(u.atoms)
        u.trajectory[0]
        return gro, traj


def _build_universe(
    n_solute_atoms: int,
    n_waters: int,
    atoms_per_water: int,
    positions: np.ndarray,
    box: np.ndarray,
    times: np.ndarray,
    water_names: Tuple[str, ...] = ("OW", "HW1", "HW2"),
    water_resname: str = "SOL",
    solute_resname: str = "LIG",
    solute_names: Optional[Tuple[str, ...]] = None,
    first_water_resid: int = 2,
    angles: Tuple[float, float, float] = (90.0, 90.0, 90.0),
) -> "mda.Universe":
    n_res = 1 + n_waters
    n_atoms = n_solute_atoms + n_waters * atoms_per_water
    atom_resindex = np.concatenate(
        [np.zeros(n_solute_atoms, dtype=int), np.repeat(np.arange(1, 1 + n_waters), atoms_per_water)]
    )
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_res,
        atom_resindex=atom_resindex,
        residue_segindex=np.zeros(n_res, dtype=int),
        trajectory=True,
    )
    solute_names = solute_names or tuple(f"C{i + 1}" for i in range(n_solute_atoms))
    names = list(solute_names) + list(water_names[:atoms_per_water]) * n_waters
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", [solute_resname] + [water_resname] * n_waters)
    u.add_TopologyAttr("resid", [1] + list(range(first_water_resid, first_water_resid + n_waters)))
    u.add_TopologyAttr("segid", ["A"])
    elements = ["C"] * n_solute_atoms + (["O", "H", "H"][:atoms_per_water]) * n_waters
    u.add_TopologyAttr("element", elements)
    masses = [12.011] * n_solute_atoms + ([15.999, 1.008, 1.008][:atoms_per_water]) * n_waters
    u.add_TopologyAttr("mass", masses)
    u.load_new(
        positions.astype(np.float32),
        order="fac",
        dimensions=np.tile(np.array([*box, *angles], dtype=np.float32), (positions.shape[0], 1)),
        dt=float(times[1] - times[0]) if len(times) > 1 else 1.0,
    )
    return u


def _far_points(rng: np.random.Generator, n: int, center: np.ndarray, box: np.ndarray, min_dist: float) -> np.ndarray:
    pts = np.empty((0, 3))
    while len(pts) < n:
        cand = rng.uniform(0.0, 1.0, size=(max(4 * n, 16), 3)) * box
        ok = _min_image_dist(cand, center, box) >= min_dist
        pts = np.vstack([pts, cand[ok]])
    return pts[:n]


def make_scripted_system(
    presence: np.ndarray,
    n_bulk: int = 60,
    box: float = 40.0,
    inside_r: float = 2.5,
    outside_min: float = 8.0,
    times: Optional[np.ndarray] = None,
    dt: float = 1.0,
    seed: int = 0,
) -> SyntheticSystem:
    """Solute at the box centre plus waters that are inside/outside on cue.

    ``presence[t, v]`` is True when visitor ``v`` has its oxygen ``inside_r``
    Angstrom from the solute atom at frame ``t`` and False when it is at least
    ``outside_min`` Angstrom away. ``n_bulk`` further waters are always far
    away, so for any zone cutoff in ``(inside_r, outside_min)`` the exact
    ``n_solvent`` per frame is ``presence.sum(axis=1)``.
    """
    presence = np.asarray(presence, dtype=bool)
    n_frames, n_vis = presence.shape
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center = box_v / 2.0
    if times is None:
        times = np.arange(n_frames, dtype=np.float64) * dt
    times = np.asarray(times, dtype=np.float64)
    n_waters = n_vis + n_bulk
    oxy = np.empty((n_frames, n_waters, 3))
    for t in range(n_frames):
        far = _far_points(rng, n_waters, center, box_v, outside_min)
        oxy[t] = far
        for v in range(n_vis):
            if presence[t, v]:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                oxy[t, v] = center + inside_r * direction
    atoms = waters_from_oxygens(oxy, rng)  # (T, W, 3, 3)
    solute = np.tile(center, (n_frames, 1, 1))
    positions = np.concatenate([solute, atoms.reshape(n_frames, n_waters * 3, 3)], axis=1)
    positions = _wrap(positions, box_v)
    u = _build_universe(1, n_waters, 3, positions, box_v, times)
    truth = {
        "n_solvent": presence.sum(axis=1).astype(int),
        "presence": presence,
        "inside_r": inside_r,
        "outside_min": outside_min,
        "visitor_resids": list(range(2, 2 + n_vis)),
    }
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)


def make_ideal_gas_system(
    n_waters: int = 800,
    n_frames: int = 400,
    box: float = 30.0,
    seed: int = 0,
    dt: float = 1.0,
) -> SyntheticSystem:
    """Independent uniform waters each frame; solute atom at box centre.

    Truth for a spherical zone of radius ``r`` around the solute atom:
    ``n_solvent ~ Binomial(N, p)`` with ``p = (4/3)*pi*r^3 / L^3`` (valid while
    ``r < L/2``), hence ``mean = N p``, ``var = N p (1-p)`` and
    ``P(n>=1) = 1 - (1-p)^N``. The time-averaged oxygen number density is
    uniform and equal to ``N / L^3``.
    """
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center = box_v / 2.0
    oxy = rng.uniform(0.0, 1.0, size=(n_frames, n_waters, 3)) * box_v
    atoms = waters_from_oxygens(oxy, rng)
    solute = np.tile(center, (n_frames, 1, 1))
    positions = np.concatenate([solute, atoms.reshape(n_frames, n_waters * 3, 3)], axis=1)
    positions = _wrap(positions, box_v)
    times = np.arange(n_frames, dtype=np.float64) * dt
    u = _build_universe(1, n_waters, 3, positions, box_v, times)
    truth = {"n_waters": n_waters, "box": float(box), "number_density": n_waters / float(box) ** 3}
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)


def sphere_survival(t: np.ndarray, radius: float, diffusion: float, n_terms: int = 400) -> np.ndarray:
    """Exact survival probability of a Brownian particle in a sphere.

    The particle starts uniformly distributed inside a sphere of radius ``R`` with an
    absorbing boundary (equivalently: the Impey/Rapaport continuous-presence
    correlation function for a freely diffusing solvent in equilibrium)::

        S(t) = (6 / pi^2) * sum_{n>=1} n^-2 * exp(-n^2 pi^2 D t / R^2)

    Its integral is ``R^2 / (15 D)``, the mean exit time from a uniform start.
    """
    t = np.asarray(t, dtype=np.float64)
    n = np.arange(1, n_terms + 1)[:, None]
    return (6.0 / np.pi**2) * np.sum(
        np.exp(-(n**2) * np.pi**2 * diffusion * t[None, :] / radius**2) / n**2, axis=0
    )


#: Discrete-monitoring (continuity) correction constant, ``-zeta(1/2)/sqrt(2 pi)``
#: (Broadie, Glasserman & Kou, Math. Finance 7, 325, 1997): sampling a diffusion every
#: ``dt`` is, to leading order, equivalent to continuous monitoring with the absorbing
#: boundary moved *outward* by ``BGK_BETA * sqrt(2 D dt)``.
BGK_BETA = 0.5825971579390106


def make_brownian_system(
    n_waters: int = 1500,
    n_frames: int = 6000,
    box: float = 36.0,
    diffusion: float = 0.25,
    dt: float = 0.5,
    seed: int = 0,
) -> SyntheticSystem:
    """Freely diffusing single-atom solvent (resname SOL, atom OW) around a fixed solute.

    ``diffusion`` is in A^2/ps (0.25 ~ 2.5e-5 cm^2/s, liquid water at 300 K). Positions at the
    output times are exact samples of Brownian motion (Gaussian increments), wrapped
    into the periodic box; there is no integration error. The mean number of
    solvent within a sphere of radius R is ``N * (4/3) pi R^3 / L^3``.
    """
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center = box_v / 2.0
    sigma = np.sqrt(2.0 * diffusion * dt)
    pos = rng.uniform(0.0, 1.0, size=(n_waters, 3)) * box_v
    traj = np.empty((n_frames, n_waters, 3), dtype=np.float64)
    for t in range(n_frames):
        traj[t] = pos
        pos = pos + rng.normal(scale=sigma, size=pos.shape)
        pos = _wrap(pos, box_v)
    solute = np.tile(center, (n_frames, 1, 1))
    positions = np.concatenate([solute, traj], axis=1)
    times = np.arange(n_frames, dtype=np.float64) * dt
    u = _build_universe(1, n_waters, 1, positions, box_v, times)
    truth = {"diffusion": diffusion, "dt": dt, "center": center.tolist(), "box": float(box)}
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)


def make_straddling_water_system(
    box: float = 20.0,
    n_frames: int = 3,
    seed: int = 0,
    solute_x: float = 2.0,
) -> SyntheticSystem:
    """A solute atom near the x=0 face and ONE water split across that face.

    The oxygen sits 0.1 A inside the +x face (x = box - 0.1) and the molecule's
    bisector points along +x, so both hydrogens wrap to the -x side (x ~ 0.49). Written the
    way an unprocessed GROMACS trajectory is (every atom wrapped individually, no bonds),
    the molecule is split in the raw coordinates: its raw centre of geometry is
    ~box/3 from the molecule's true position.

    Truth (minimum image): oxygen-solute = 2.1 A, true COG-solute = 1.71 A,
    true COM-solute = 2.03 A -> ``n_solvent == 1`` on every frame for a 3.5 A zone
    with an oxygen, COG or COM probe. A raw-coordinate COG lies ~5 A away -> 0.
    """
    del seed  # geometry is deterministic
    box_v = np.full(3, float(box))
    c = box / 2.0
    half = _HOH / 2.0
    o = np.array([box - 0.1, c, c])
    h1 = o + np.array([_OH * np.cos(half), +_OH * np.sin(half), 0.0])
    h2 = o + np.array([_OH * np.cos(half), -_OH * np.sin(half), 0.0])
    solute = np.array([solute_x, c, c])
    frame = _wrap(np.vstack([solute, o, h1, h2]), box_v)
    positions = np.tile(frame, (n_frames, 1, 1))
    times = np.arange(n_frames, dtype=np.float64)
    u = _build_universe(1, 1, 3, positions, box_v, times)
    truth = {
        "oxygen_to_solute": 2.1,
        "cog_to_solute": float(np.linalg.norm(((o + h1 + h2) / 3.0) - np.array([solute_x - box, c, c]))),
        "box": float(box),
    }
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)
