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


#: Vertices of a regular tetrahedron (unit circumradius): a rigid, non-planar, non-collinear solute.
TETRAHEDRON = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64) / np.sqrt(3.0)


def solute_template(n_atoms: int, radius: float = 1.5) -> np.ndarray:
    """Rigid solute geometry around the origin: one atom, or the first ``n_atoms`` (<=4) tetrahedron vertices."""
    if n_atoms == 1:
        return np.zeros((1, 3))
    if n_atoms > 4:
        raise ValueError("solute_template supports up to 4 atoms")
    return TETRAHEDRON[:n_atoms] * radius


def make_ideal_gas_system(
    n_waters: int = 800,
    n_frames: int = 400,
    box: float = 30.0,
    seed: int = 0,
    dt: float = 1.0,
    n_solute_atoms: int = 1,
    solute_position: Optional[Tuple[float, float, float]] = None,
) -> SyntheticSystem:
    """Independent uniform waters each frame around a fixed rigid solute (default: one atom at the box centre).

    Truth for a spherical zone of radius ``r`` around a single solute atom:
    ``n_solvent ~ Binomial(N, p)`` with ``p = (4/3)*pi*r^3 / L^3`` (valid while
    ``r < L/2``), hence ``mean = N p``, ``var = N p (1-p)`` and
    ``P(n>=1) = 1 - (1-p)^N``. The time-averaged oxygen number density is
    uniform and equal to ``N / L^3`` at every point of the periodic box - also next to
    the box faces when ``solute_position`` puts the solute there.
    """
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center = box_v / 2.0 if solute_position is None else np.asarray(solute_position, dtype=np.float64)
    oxy = rng.uniform(0.0, 1.0, size=(n_frames, n_waters, 3)) * box_v
    atoms = waters_from_oxygens(oxy, rng)
    solute = np.tile(center + solute_template(n_solute_atoms), (n_frames, 1, 1))
    positions = np.concatenate([solute, atoms.reshape(n_frames, n_waters * 3, 3)], axis=1)
    positions = _wrap(positions, box_v)
    times = np.arange(n_frames, dtype=np.float64) * dt
    u = _build_universe(n_solute_atoms, n_waters, 3, positions, box_v, times)
    truth = {"n_waters": n_waters, "box": float(box), "number_density": n_waters / float(box) ** 3,
             "solute_center": center.tolist()}
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)


def make_rotating_solute_system(
    n_frames: int = 120,
    n_waters: int = 300,
    box: float = 40.0,
    bound_distance: float = 3.0,
    seed: int = 0,
    drift: float = 0.0,
) -> SyntheticSystem:
    """A tetrahedral solute that tumbles randomly (and optionally drifts) with ONE water bound to it.

    In the solute frame of the first frame the bound water's oxygen sits at
    ``centre + bound_distance * x_hat`` on every frame, so a correctly superposed density map has a
    delta-like peak of ``n_frames`` counts at that point; without superposition it smears over a
    sphere of radius ``bound_distance``. Other waters are uniform and independent.
    """
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center0 = box_v / 2.0
    rots = random_rotations(rng, n_frames)
    rots[0] = np.eye(3)
    tmpl = solute_template(4)
    centers = center0 + np.cumsum(rng.normal(scale=drift, size=(n_frames, 3)), axis=0) if drift else np.tile(center0, (n_frames, 1))
    solute = np.einsum("tij,aj->tai", rots, tmpl) + centers[:, None, :]
    bound_o = centers + np.einsum("tij,j->ti", rots, np.array([bound_distance, 0.0, 0.0]))
    others = rng.uniform(0.0, 1.0, size=(n_frames, n_waters, 3)) * box_v
    for t in range(n_frames):  # keep the uniform waters out of the bound site so the peak is unambiguous
        d = others[t] - bound_o[t]
        d -= np.round(d / box_v) * box_v
        close = np.linalg.norm(d, axis=1) < 2.0
        others[t, close] += np.array([0.0, 0.0, 6.0])
    oxy = np.concatenate([bound_o[:, None, :], others], axis=1)
    atoms = waters_from_oxygens(oxy, rng)
    positions = np.concatenate([solute, atoms.reshape(n_frames, (n_waters + 1) * 3, 3)], axis=1)
    positions = _wrap(positions, box_v)
    times = np.arange(n_frames, dtype=np.float64)
    u = _build_universe(4, n_waters + 1, 3, positions, box_v, times)
    truth = {"bound_site_in_first_frame": (center0 + np.array([bound_distance, 0, 0])).tolist(), "n_frames": n_frames}
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


def sphere_survival_discrete(t: np.ndarray, radius: float, diffusion: float, dt: float, n_terms: int = 600) -> np.ndarray:
    """Survival of a Brownian particle that starts uniformly inside ``radius`` and is *observed every* ``dt``.

    Discrete observation is, to leading order, equivalent to continuous monitoring with the absorbing wall moved
    outward to ``R_abs = R + BGK_BETA * sqrt(2 D dt)``, while the start remains uniform in the original sphere ``R``
    (the Impey estimator conditions on being inside ``R`` at the origin sample)::

        S(t) = (3/R^3) sum_n b_n (-1)^(n+1) R_abs^2/(n pi) exp(-D (n pi / R_abs)^2 t),
        b_n  = (2/R_abs) int_0^R r sin(n pi r / R_abs) dr

    It reduces to :func:`sphere_survival` for ``dt -> 0``. Against a fine-step brute-force simulation the leading-order
    correction is accurate to ~0.02 in S for ``sqrt(2 D dt) / R ~ 0.12``.
    """
    t = np.asarray(t, dtype=np.float64)
    r_abs = radius + BGK_BETA * np.sqrt(2.0 * diffusion * dt)
    n = np.arange(1, n_terms + 1)[:, None]
    k = n * np.pi / r_abs
    int_r_sin = (np.sin(k * radius) - k * radius * np.cos(k * radius)) / k**2
    b = (2.0 / r_abs) * int_r_sin
    w = b * ((-1.0) ** (n + 1)) * r_abs**2 / (n * np.pi)
    return (3.0 / radius**3) * np.sum(w * np.exp(-diffusion * k**2 * t[None, :]), axis=0)


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


def make_demo_system(n_waters: int = 900, n_frames: int = 200, box: float = 30.0, dt: float = 0.5, seed: int = 7,
                     n_solute_atoms: int = 14, diffusion: float = 0.25, well_depth: float = 1.3) -> SyntheticSystem:
    """A small *synthetic* hydrated solute for demonstrations and GUI tests (NOT a molecular dynamics simulation).

    Waters (rigid TIP3P geometry, random orientation) perform overdamped Langevin dynamics (diffusion ``diffusion``
    A^2/ps) around a rigid cluster of ``n_solute_atoms`` atoms: a soft repulsive core (sigma 2.3 A) plus a Gaussian
    attractive well at 3.0 A per solute atom, so a hydration shell with a density peak near 3 A forms and exchanges.
    There is no water-water interaction. Everything is seeded and reproducible.
    """
    rng = np.random.default_rng(seed)
    box_v = np.full(3, float(box))
    center = box_v / 2.0
    # compact random solute cluster, minimum atom spacing 1.5 A
    atoms = []
    while len(atoms) < n_solute_atoms:
        cand = rng.normal(scale=2.0, size=3)
        if np.linalg.norm(cand) < 4.2 and all(np.linalg.norm(cand - a) > 1.5 for a in atoms):
            atoms.append(cand)
    solute_local = np.array(atoms)
    solute_local -= solute_local.mean(axis=0)
    solute = solute_local + center
    sigma, well_r, well_w = 2.3, 3.0, 0.45
    kT_force = diffusion  # D/kT with U in kT units -> drift = D * (-grad U)
    sub = max(1, int(round(dt / 0.02)))
    dts = dt / sub
    noise = np.sqrt(2.0 * diffusion * dts)

    def force(pos: np.ndarray) -> np.ndarray:
        d = pos[:, None, :] - solute[None, :, :]
        d -= np.round(d / box_v) * box_v
        r = np.linalg.norm(d, axis=2) + 1e-9
        rep = 24.0 * sigma**6 / r**7  # -d/dr of 4 (sigma/r)^6
        wellf = -well_depth * np.exp(-(((r - well_r) / well_w) ** 2)) * (2.0 * (r - well_r) / well_w**2)  # F_r = -dU/dr, U = -e exp(-x^2)
        mag = np.where(r < 8.0, rep + wellf, 0.0)  # force along +r (away from the solute atom)
        return np.sum((mag / r)[:, :, None] * d, axis=1)

    pos = rng.uniform(0, 1, size=(n_waters, 3)) * box_v
    for _ in range(50):  # remove initial overlaps
        d = pos[:, None, :] - solute[None, :, :]
        d -= np.round(d / box_v) * box_v
        bad = np.linalg.norm(d, axis=2).min(axis=1) < 2.4
        if not bad.any():
            break
        pos[bad] = rng.uniform(0, 1, size=(int(bad.sum()), 3)) * box_v
    oxy = np.empty((n_frames, n_waters, 3))
    for t in range(n_frames):
        oxy[t] = pos
        for _ in range(sub):
            pos = pos + kT_force * force(pos) * dts + rng.normal(scale=noise, size=pos.shape)
            pos = pos % box_v
    waters = waters_from_oxygens(oxy, rng)
    positions = np.concatenate([np.tile(solute, (n_frames, 1, 1)), waters.reshape(n_frames, n_waters * 3, 3)], axis=1)
    positions = _wrap(positions, box_v)
    times = np.arange(n_frames, dtype=np.float64) * dt
    u = _build_universe(n_solute_atoms, n_waters, 3, positions, box_v, times)
    truth = {"synthetic": True, "n_waters": n_waters, "box": float(box), "shell_radius": well_r, "diffusion": diffusion}
    return SyntheticSystem(u, positions.astype(np.float32), box_v, times, truth)
