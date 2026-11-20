"""Periodic-boundary geometry used by zone evaluation and density mapping.

Design rules (each one fixes a v2.0.0 behaviour):

* **No bonds required.** Molecule centres (COG/COM probes) are computed relative to a
  reference atom with the minimum-image convention, so molecules split across the box by
  ``gmx trjconv``-less raw output work for GRO/PDB topologies. v2.0.0 called
  ``center_of_geometry(unwrap=True)``, which needs bonds and crashes without them.
* **Vectorised.** One pass over all probe atoms instead of a Python loop per solvent molecule.
* **Honest limits.** :attr:`Box.min_image_limit` is the radius up to which the minimum-image
  convention and a periodic neighbour search are exact (half the smallest perpendicular box
  width); callers compare cut-offs and grid extents with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from MDAnalysis.lib import distances as mda_distances
from MDAnalysis.lib.mdamath import triclinic_vectors

_ORTHO_TOL = 1e-3  # degrees


@dataclass(frozen=True)
class Box:
    """A periodic cell given as MDAnalysis ``[a, b, c, alpha, beta, gamma]`` (A, degrees)."""

    dims: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.dims[:3], dtype=np.float64)

    @property
    def angles(self) -> np.ndarray:
        return np.asarray(self.dims[3:], dtype=np.float64)

    @property
    def orthorhombic(self) -> bool:
        return bool(np.all(np.abs(self.angles - 90.0) < _ORTHO_TOL))

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(triclinic_vectors(np.asarray(self.dims, dtype=np.float32)), dtype=np.float64)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.matrix)))

    @property
    def heights(self) -> np.ndarray:
        """Perpendicular distances between opposite faces (equal to lengths for orthorhombic cells)."""
        a, b, c = self.matrix
        v = self.volume
        return np.array([v / np.linalg.norm(np.cross(b, c)), v / np.linalg.norm(np.cross(a, c)),
                         v / np.linalg.norm(np.cross(a, b))])

    @property
    def min_image_limit(self) -> float:
        """Largest distance for which minimum image and cell-list searches are exact."""
        return float(0.5 * self.heights.min())


def box_from_dimensions(dims) -> Optional[Box]:
    """Return a :class:`Box`, or ``None`` if the dimensions do not describe a periodic cell."""
    if dims is None:
        return None
    arr = np.asarray(dims, dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)) or np.any(arr[:3] <= 0):
        return None
    if np.any(arr[3:] <= 0) or np.any(arr[3:] >= 180):
        return None
    return Box(arr)


def cutoff_warning(cutoff: float, box: Optional[Box], what: str = "cut-off") -> Optional[str]:
    """Message when ``cutoff`` exceeds the minimum-image limit of ``box``."""
    if box is None:
        return None
    limit = box.min_image_limit
    if cutoff > limit:
        return (
            f"{what} {cutoff:.2f} A exceeds half the shortest box width ({limit:.2f} A). "
            "The zone overlaps its own periodic images, so its volume is not the nominal sphere "
            "volume and occupancy saturates; reduce the cut-off or use a larger box."
        )
    return None


def minimum_image(vectors: np.ndarray, box: Optional[Box]) -> np.ndarray:
    """Apply the minimum-image convention to displacement vectors (shape ``(n, 3)``)."""
    if box is None or len(vectors) == 0:
        return vectors
    if box.orthorhombic:
        lengths = box.lengths
        return vectors - np.round(vectors / lengths) * lengths
    return mda_distances.minimize_vectors(np.ascontiguousarray(vectors, dtype=np.float64),
                                          np.asarray(box.dims, dtype=np.float32)).astype(np.float64)


def group_layout(owner: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort order, group starts and counts for atoms labelled by an owner (molecule) ordinal."""
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    if len(sorted_owner) == 0:
        return order, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_owner)) + 1])
    counts = np.diff(np.concatenate([starts, [len(sorted_owner)]]))
    return order, starts, counts


def molecule_centers(
    positions: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    weights: Optional[np.ndarray],
    box: Optional[Box],
) -> np.ndarray:
    """Weighted centre of each group of atoms, robust to groups split across the cell.

    ``positions`` must be grouped (contiguous per molecule). Every atom is moved to its
    minimum image relative to the first atom of its molecule, the weighted mean offset is
    added to that reference atom. Valid whenever a molecule is smaller than half the box.
    """
    pos = np.asarray(positions, dtype=np.float64)
    ref = np.repeat(pos[starts], counts, axis=0)
    disp = minimum_image(pos - ref, box)
    if weights is None:
        w = np.ones(len(pos))
    else:
        w = np.asarray(weights, dtype=np.float64)
    wsum = np.add.reduceat(w, starts)
    offset = np.add.reduceat(disp * w[:, None], starts, axis=0) / wsum[:, None]
    return pos[starts] + offset


def wrap_about(points: np.ndarray, center: np.ndarray, box: Optional[Box]) -> np.ndarray:
    """Move ``points`` to the periodic image closest to ``center``."""
    if box is None or len(points) == 0:
        return np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    return c + minimum_image(np.asarray(points, dtype=np.float64) - c, box)


def kabsch_fit(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Optimal rigid superposition of ``mobile`` onto ``reference`` (Kabsch 1976).

    Returns ``(R, mobile_center, reference_center, degenerate)`` such that
    ``reference ~= (mobile - mobile_center) @ R + reference_center``. For fewer than three
    atoms, or (near-)collinear atoms, the rotation is undefined; the identity rotation is
    returned and ``degenerate`` is True (translation-only superposition).
    """
    mob = np.asarray(mobile, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    w = np.ones(len(mob)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mc = (mob * w[:, None]).sum(axis=0)
    rc = (ref * w[:, None]).sum(axis=0)
    if len(mob) < 3:
        return np.eye(3), mc, rc, True
    p = mob - mc
    q = ref - rc
    cov = p.T @ (q * w[:, None])
    u, s, vt = np.linalg.svd(cov)
    if s[1] <= 1e-9 * max(s[0], 1e-30):
        return np.eye(3), mc, rc, True
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, mc, rc, False


def rotation_angle_deg(rot: np.ndarray) -> float:
    """Rotation angle of a proper rotation matrix, in degrees."""
    cos = (np.trace(rot) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def min_distance(a: np.ndarray, b: np.ndarray, box: Optional[Box]) -> float:
    """Smallest (minimum-image) distance between any point of ``a`` and any point of ``b``."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if box is None or box.orthorhombic:
        from scipy.spatial import cKDTree

        if box is None:
            return float(cKDTree(b64).query(a64, k=1)[0].min())
        lengths = box.lengths

        def _wrap(x: np.ndarray) -> np.ndarray:
            w = x - np.floor(x / lengths) * lengths
            w = np.where(w >= lengths, w - lengths, w)
            return np.where(w < 0, 0.0, w)

        tree = cKDTree(_wrap(b64), boxsize=lengths)
        return float(tree.query(_wrap(a64), k=1)[0].min())
    best = np.inf
    dims = np.asarray(box.dims, dtype=np.float32)
    step = max(1, int(4_000_000 // max(len(b64), 1)))
    for i in range(0, len(a64), step):
        d = mda_distances.distance_array(a64[i:i + step].astype(np.float32), b64.astype(np.float32), box=dims)
        best = min(best, float(d.min()))
    return float(best)
