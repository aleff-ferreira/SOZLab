"""Time-averaged solvent number-density maps in a solute-fixed frame (algorithm 2).

Quantity
--------
``rho(r) = < sum_i delta(r - r_i) >``: the mean number of probe atoms per A^3 at position
``r`` of the *solute frame* (units A^-3). Dividing by the bulk number density ``rho_bulk`` of the
same species gives ``g(r) = rho(r) / rho_bulk`` (dimensionless; 1 in bulk), and ``-ln g`` is the
potential of mean force of a probe at ``r`` in units of kT, valid only when ``g`` is
well converged.

What v2.0.0 got wrong, and what this module does instead
---------------------------------------------------------
* The grid was built by MDAnalysis from the bounding box of the selection *at whatever frame
  the shared Universe happened to be on* and atoms outside it were silently dropped. Here the
  grid is defined by the solute (centre + extent + padding) or given explicitly, and the fraction
  of probe atoms that fall outside is reported.
* Solvent was binned in raw coordinates, so molecules on the far side of the periodic boundary
  were missing and rotation by superposition left the box misaligned. Here every frame is
  processed as: make the solute whole -> wrap solvent to the periodic image nearest the solute
  -> superpose (Kabsch) onto the reference -> bin. This is the order recommended by GROMACS
  (``trjconv -pbc mol -center`` then ``-fit rot+trans``).
* Bulk density was the constant 0.0334 for every species. Here it is estimated from the
  simulation (far-field voxels, or ``<N>/<V>``) or supplied.
* No uncertainty. Here frames are split into contiguous blocks; the standard error of every
  voxel is reported together with a reliability mask.
* The whole trajectory was copied into memory for alignment. Here frames stream through and
  only integer count grids are kept, so parallel and serial runs are *exactly* identical.

The reliable region is the ball of radius ``r_cov`` (half the shortest perpendicular box width)
around the solute: beyond it the periodic image nearest the solute no longer covers the grid.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import MDAnalysis as mda

from engine.models import DensityMapConfig, InputConfig
from engine.periodic import Box, box_from_dimensions, kabsch_fit, minimum_image, rotation_angle_deg
from engine.uncertainty import block_ranges

log = logging.getLogger("sozlab")

_DYNAMIC_KEYWORDS = {"around", "sphlayer", "sphzone", "cylayer", "cylzone", "point", "prop", "same", "byres"}


def selection_is_dynamic(selection: str) -> bool:
    """True when a selection depends on coordinates (re-evaluated every frame to be correct)."""
    import re

    tokens = set(re.split(r"\s+|\(|\)", (selection or "").lower()))
    return bool(tokens & {"around", "sphlayer", "sphzone", "cylayer", "cylzone", "point", "prop"})


@dataclass
class DensityMapResult:
    name: str
    grid_path: str
    grid: Optional[np.ndarray]  # density, A^-3
    axes: Dict[str, np.ndarray]
    slices: Dict[str, np.ndarray]
    metadata: Dict[str, object]
    # ---- added in 2.1 (None for the legacy engine) ---------------------------------------
    counts: Optional[np.ndarray] = None
    se: Optional[np.ndarray] = None  # standard error of the density from blocks
    g: Optional[np.ndarray] = None  # density / rho_bulk
    reliable: Optional[np.ndarray] = None  # bool mask


@dataclass
class GridSpec:
    center: np.ndarray
    origin_edge: np.ndarray  # lower edge of the first voxel
    spacing: float
    shape: Tuple[int, int, int]

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def origin(self) -> np.ndarray:
        """Centre of the first voxel (DX convention)."""
        return self.origin_edge + 0.5 * self.spacing

    def axes(self) -> Dict[str, np.ndarray]:
        o = self.origin
        return {k: o[i] + np.arange(self.shape[i]) * self.spacing for i, k in enumerate("xyz")}


@dataclass
class DensityJob:
    """Everything a worker needs; plain data so it pickles cheaply."""

    inputs: dict
    species_selection: str
    species_dynamic: bool
    align: bool
    align_idx: np.ndarray
    align_weights: Optional[np.ndarray]
    ref_coords: Optional[np.ndarray]  # unwrapped reference of the align selection (m, 3)
    ref_center: np.ndarray
    grid_center: np.ndarray
    origin_edge: np.ndarray
    spacing: float
    shape: Tuple[int, int, int]
    n_blocks: int


def unwrap_chain(coords: np.ndarray, box: Optional[Box]) -> np.ndarray:
    """Make an ordered atom set whole by chaining minimum-image steps between consecutive atoms.

    Correct whenever consecutive selected atoms are closer than half the box (true for
    chains, residues and compact selections), which is much weaker than requiring the whole
    solute to fit within half a box of its first atom.
    """
    if box is None or len(coords) < 2:
        return np.asarray(coords, dtype=np.float64)
    pos = np.asarray(coords, dtype=np.float64)
    steps = minimum_image(np.diff(pos, axis=0), box)
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1:] = pos[0] + np.cumsum(steps, axis=0)
    return out


def _weighted_center(coords: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return coords.mean(axis=0)
    return (coords * weights[:, None]).sum(axis=0) / weights.sum()


def solute_transform(
    pos_all: np.ndarray,
    box: Optional[Box],
    align_idx: np.ndarray,
    weights: Optional[np.ndarray],
    ref_coords: Optional[np.ndarray],
    ref_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Per-frame map from the lab frame to the solute frame.

    Returns ``(solute_center_lab, R, ref_center, rotation_deg, degenerate)`` such that
    ``x_solute_frame = (wrap_about(x, solute_center) - solute_center) @ R + ref_center``.
    """
    sel = unwrap_chain(pos_all[align_idx], box)
    center = _weighted_center(sel, weights)
    if ref_coords is None:
        return center, np.eye(3), ref_center, 0.0, False
    rot, mc, rc, degenerate = kabsch_fit(sel, ref_coords, weights)
    return mc, rot, rc, rotation_angle_deg(rot), degenerate


def map_species(
    pos_species: np.ndarray,
    box: Optional[Box],
    solute_center: np.ndarray,
    rot: np.ndarray,
    ref_center: np.ndarray,
) -> np.ndarray:
    d = minimum_image(np.asarray(pos_species, dtype=np.float64) - solute_center, box)
    return d @ rot + ref_center


def bin_counts(points: np.ndarray, origin_edge: np.ndarray, spacing: float, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, int]:
    """Histogram ``points`` onto the grid; returns ``(flat int64 counts, number inside)``."""
    n_vox = int(np.prod(shape))
    if len(points) == 0:
        return np.zeros(n_vox, dtype=np.int64), 0
    idx = np.floor((points - origin_edge) / spacing).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    idx = idx[inside]
    flat = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
    return np.bincount(flat, minlength=n_vox).astype(np.int64), int(inside.sum())


def savez_deterministic(path: str, **arrays: np.ndarray) -> None:
    """Like ``np.savez`` but with fixed zip timestamps, so identical arrays give identical bytes."""
    import io
    import zipfile

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def arrays_sha256(arrays: Dict[str, np.ndarray]) -> str:
    """Content hash of named arrays (independent of how they are stored on disk)."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------- worker
_UNIVERSES: Dict[str, mda.Universe] = {}


def _universe_for(inputs: dict) -> mda.Universe:
    key = repr((inputs.get("topology"), inputs.get("trajectory")))
    u = _UNIVERSES.get(key)
    if u is None:
        _UNIVERSES.clear()
        cfg = InputConfig.from_dict(inputs)
        u = mda.Universe(cfg.topology, cfg.trajectory) if cfg.trajectory else mda.Universe(cfg.topology)
        _UNIVERSES[key] = u
    return u


def accumulate_frames(
    job: DensityJob,
    items: Sequence[Tuple[int, int, int]],
    universe: Optional[mda.Universe] = None,
) -> Dict[str, object]:
    """Bin ``items = [(sample, frame, block), ...]``; returns count grids and per-frame QC."""
    u = universe if universe is not None else _universe_for(job.inputs)
    species = u.select_atoms(job.species_selection, updating=job.species_dynamic)
    static_species_idx = None if job.species_dynamic else np.asarray(species.indices, dtype=np.int64)
    n_vox = int(np.prod(job.shape))
    counts = np.zeros((job.n_blocks, n_vox), dtype=np.int64)
    rows = []
    for sample, frame, block in items:
        ts = u.trajectory[frame]
        pos = ts.positions
        box = box_from_dimensions(ts.dimensions)
        if job.align:
            center, rot, rc, rot_deg, degenerate = solute_transform(
                pos, box, job.align_idx, job.align_weights, job.ref_coords, job.ref_center
            )
        else:
            center, rot, rc, rot_deg, degenerate = job.grid_center, np.eye(3), job.grid_center, 0.0, False
        idx = static_species_idx if static_species_idx is not None else np.asarray(species.indices, dtype=np.int64)
        pts = map_species(pos[idx], box, center, rot, rc)
        flat, n_in = bin_counts(pts, job.origin_edge, job.spacing, job.shape)
        counts[block] += flat
        rows.append((sample, frame, block, len(idx), n_in,
                     float(box.volume) if box is not None else float("nan"),
                     float(box.min_image_limit) if box is not None else float("nan"),
                     float(center[0]), float(center[1]), float(center[2]),
                     rot_deg, float(degenerate)))
    return {"counts": counts, "rows": np.asarray(rows, dtype=np.float64)}


def _density_chunk(job: DensityJob, items: Sequence[Tuple[int, int, int]]) -> Dict[str, object]:
    return accumulate_frames(job, items)


# ------------------------------------------------------------------------------------- orchestration
def _resolve_window(cfg: DensityMapConfig, n_frames: int, options) -> Tuple[range, int, Optional[int], int]:
    start = cfg.frame_start if cfg.frame_start is not None else options.frame_start
    stop = cfg.frame_stop if cfg.frame_stop is not None else options.frame_stop
    step = cfg.stride if cfg.stride else options.stride
    stop_eff = n_frames if stop is None else min(int(stop), n_frames)
    return range(max(int(start), 0), stop_eff, int(step)), int(start), stop, int(step)


def _reference(u: mda.Universe, cfg: DensityMapConfig, align_idx: np.ndarray, frames: range) -> Tuple[np.ndarray, List[str]]:
    """Whole reference coordinates of the align selection (first analysed frame or structure)."""
    notes: List[str] = []
    if cfg.align_reference == "structure" and cfg.align_reference_path:
        ref_u = mda.Universe(cfg.align_reference_path)
        ref_sel = ref_u.select_atoms(cfg.align_selection)
        if len(ref_sel) != len(align_idx):
            raise ValueError(
                f"Alignment reference has {len(ref_sel)} atoms for '{cfg.align_selection}' but the trajectory "
                f"selection has {len(align_idx)}; the selections must match atom for atom."
            )
        box = box_from_dimensions(ref_u.dimensions)
        return unwrap_chain(ref_sel.positions, box), notes
    ts = u.trajectory[frames[0]]
    box = box_from_dimensions(ts.dimensions)
    return unwrap_chain(ts.positions[align_idx], box), notes


def _grid_from_reference(cfg: DensityMapConfig, ref_center: np.ndarray, region_coords: np.ndarray,
                         r_limit: float) -> Tuple[GridSpec, List[str]]:
    notes: List[str] = []
    delta = float(cfg.grid_spacing)
    center = np.asarray(cfg.grid_center, dtype=np.float64) if cfg.grid_center is not None else ref_center
    if cfg.grid_size is not None:
        size = np.asarray(cfg.grid_size, dtype=np.float64)
        if size.size == 1:
            size = np.repeat(size, 3)
    else:
        half = np.abs(region_coords - center).max(axis=0) + float(cfg.region_padding)
        size = 2.0 * half
    if np.isfinite(r_limit) and np.any(size / 2.0 > r_limit):
        notes.append(
            f"Grid half-width {float(np.max(size) / 2):.1f} A exceeds the fully covered radius {r_limit:.1f} A "
            "(half the shortest box width): voxels beyond it are flagged unreliable."
        )
    shape = tuple(int(max(1, np.ceil(s / delta - 1e-9))) for s in size)
    full = np.asarray(shape) * delta
    origin_edge = center - full / 2.0
    return GridSpec(center=center, origin_edge=origin_edge, spacing=delta, shape=shape), notes


def _bulk_density(cfg: DensityMapConfig, grid: GridSpec, density: np.ndarray, se: np.ndarray, rows: np.ndarray,
                  region_coords: np.ndarray, r_cov: float) -> Dict[str, object]:
    """Reference number density of the species: far-field voxels, ``<N/V>``, or user value."""
    n_species = rows[:, 3]
    vol = rows[:, 5]
    box_est = float(np.nanmean(n_species / vol)) if np.isfinite(vol).any() else float("nan")

    ax = grid.axes()
    gx, gy, gz = np.meshgrid(ax["x"], ax["y"], ax["z"], indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    from scipy.spatial import cKDTree

    dist_region = cKDTree(region_coords).query(centers, k=1)[0].reshape(grid.shape)
    dist_center = np.linalg.norm(centers - grid.center, axis=1).reshape(grid.shape)
    far = (dist_region >= cfg.far_field_distance) & (dist_center <= max(r_cov - grid.spacing, 0.0))
    n_far = int(far.sum())
    far_est = float(density[far].mean()) if n_far >= 200 else float("nan")

    method = cfg.bulk_method
    if cfg.bulk_density is not None or method == "user":
        value, used = float(cfg.bulk_density), "user"
    elif method == "far_field":
        value, used = far_est, "far_field"
    elif method == "box":
        value, used = box_est, "box"
    else:  # auto
        if np.isfinite(far_est):
            value, used = far_est, "far_field"
        else:
            value, used = box_est, "box"
    if not np.isfinite(value) or value <= 0:
        value, used = float("nan"), "unavailable"
    return {"value": value, "method": used, "box_mean": box_est, "far_field": far_est, "far_field_voxels": n_far,
            "far_vs_box_ratio": (far_est / box_est) if np.isfinite(far_est) and box_est else float("nan"),
            "far_field_distance": cfg.far_field_distance}


def run_solute_density(
    universe: mda.Universe,
    cfg: DensityMapConfig,
    options,
    output_dir: str,
    inputs: InputConfig,
    workers: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    parallel_min_seconds: float = 10.0,
) -> DensityMapResult:
    """Compute one density map with the solute-frame engine and return it with full metadata."""
    t0 = time.perf_counter()
    frames, w_start, w_stop, w_step = _resolve_window(cfg, len(universe.trajectory), options)
    if len(frames) == 0:
        raise ValueError(f"Density map '{cfg.name}': no frames in the window")
    notes: List[str] = []
    warnings_list: List[str] = []

    selection = (cfg.species_selection or "").strip()
    dyn = selection_is_dynamic(selection)
    sel_group = universe.select_atoms(selection, updating=dyn)
    if not dyn and len(sel_group) == 0:
        raise ValueError(f"Density map '{cfg.name}': species selection '{selection}' matched no atoms")

    align_sel = cfg.align_selection if cfg.align else None
    region_sel = cfg.region_selection or cfg.align_selection
    if cfg.align and not cfg.align_selection:
        raise ValueError(f"Density map '{cfg.name}': alignment requires align_selection")

    ts0 = universe.trajectory[frames[0]]
    box0 = box_from_dimensions(ts0.dimensions)
    r_cov = float(box0.min_image_limit) if box0 is not None else float("inf")

    if cfg.align:
        align_group = universe.select_atoms(align_sel)
        if len(align_group) == 0:
            raise ValueError(f"Density map '{cfg.name}': align_selection '{align_sel}' matched no atoms")
        align_idx = np.asarray(align_group.indices, dtype=np.int64)
        try:
            masses = np.asarray(align_group.masses, dtype=np.float64)
            weights = masses if np.all(np.isfinite(masses)) and np.all(masses > 0) else None
        except Exception:
            weights = None
        if weights is None:
            notes.append("Alignment is geometric (atom masses unavailable).")
        ref_coords, ref_notes = _reference(universe, cfg, align_idx, frames)
        notes += ref_notes
        ref_center = _weighted_center(ref_coords, weights)
    else:
        align_idx = np.empty(0, dtype=np.int64)
        weights = None
        ref_coords = None
        ref_center = None

    # region used for extent and for the far-field estimate
    if region_sel:
        region_group = universe.select_atoms(region_sel)
        region_idx = np.asarray(region_group.indices, dtype=np.int64)
        if cfg.align and region_sel == cfg.align_selection:
            region_coords = ref_coords
        else:
            region_coords = unwrap_chain(ts0.positions[region_idx], box0)
            if cfg.align and ref_coords is not None:
                rot0, mc0, rc0, _ = kabsch_fit(unwrap_chain(ts0.positions[align_idx], box0), ref_coords, weights)
                region_coords = (region_coords - mc0) @ rot0 + rc0
    else:
        region_coords = np.asarray(ts0.positions[np.asarray(sel_group.indices, dtype=np.int64)] if len(sel_group) else
                                   np.zeros((1, 3)), dtype=np.float64)
    if ref_center is None:
        ref_center = _weighted_center(region_coords, None)
    center_for_grid = ref_center
    grid, grid_notes = _grid_from_reference(cfg, center_for_grid, region_coords, r_cov)
    notes += grid_notes
    n_vox = grid.n_voxels
    if n_vox > 60_000_000:
        raise ValueError(f"Density map '{cfg.name}': grid has {n_vox:,} voxels; increase grid_spacing or reduce the region.")

    n_blocks = int(max(1, min(cfg.n_blocks, len(frames))))
    block_of = np.zeros(len(frames), dtype=int)
    for b, (s0, s1) in enumerate(block_ranges(len(frames), n_blocks)):
        block_of[s0:s1] = b
    items = [(i, int(f), int(block_of[i])) for i, f in enumerate(frames)]

    job = DensityJob(
        inputs=inputs.to_dict(), species_selection=selection, species_dynamic=dyn, align=bool(cfg.align),
        align_idx=align_idx, align_weights=weights, ref_coords=ref_coords, ref_center=np.asarray(ref_center),
        grid_center=np.asarray(grid.center), origin_edge=grid.origin_edge, spacing=grid.spacing, shape=grid.shape,
        n_blocks=n_blocks,
    )

    counts = np.zeros((n_blocks, n_vox), dtype=np.int64)
    qc_rows: List[np.ndarray] = []
    # time one chunk serially to decide whether processes are worth starting
    probe = items[: min(len(items), 8)]
    t_probe = time.perf_counter()
    first = accumulate_frames(job, probe, universe)
    per_frame = (time.perf_counter() - t_probe) / max(len(probe), 1)
    counts += first["counts"]
    qc_rows.append(first["rows"])
    rest = items[len(probe):]
    est_serial = per_frame * len(rest)
    done = len(probe)
    cancelled = False
    mode = "serial"
    chunk = 48
    if workers > 1 and est_serial >= parallel_min_seconds and len(rest) >= 2 * chunk:
        mode = "parallel"
        chunks = [rest[i:i + chunk] for i in range(0, len(rest), chunk)]
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(chunks)), mp_context=ctx) as pool:
            futs = [pool.submit(_density_chunk, job, c) for c in chunks]
            results: Dict[int, Dict[str, object]] = {}
            index = {f: i for i, f in enumerate(futs)}
            for fut in cf.as_completed(futs):
                results[index[fut]] = fut.result()
                done += len(chunks[index[fut]])
                if progress:
                    progress(done, len(items))
                if cancel and cancel():
                    cancelled = True
                    for f in futs:
                        f.cancel()
                    break
        for i in range(len(chunks)):
            if i not in results:
                break
            counts += results[i]["counts"]
            qc_rows.append(results[i]["rows"])
    else:
        for i in range(0, len(rest), chunk):
            if cancel and cancel():
                cancelled = True
                break
            part = accumulate_frames(job, rest[i:i + chunk], universe)
            counts += part["counts"]
            qc_rows.append(part["rows"])
            done += len(rest[i:i + chunk])
            if progress:
                progress(done, len(items))
    if cancelled:
        raise InterruptedError(f"Density map '{cfg.name}' was cancelled")

    rows = np.concatenate(qc_rows, axis=0)
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    n_used = len(rows)

    # ---- per-block and total densities -----------------------------------------------------
    vox_vol = grid.spacing ** 3
    block_frames = np.bincount(rows[:, 2].astype(int), minlength=n_blocks).astype(float)
    total_counts = counts.sum(axis=0)
    density = (total_counts / (n_used * vox_vol)).reshape(grid.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        block_dens = counts / (block_frames[:, None] * vox_vol)
    good_blocks = block_frames > 0
    if good_blocks.sum() >= 2:
        se = (block_dens[good_blocks].std(axis=0, ddof=1) / np.sqrt(good_blocks.sum())).reshape(grid.shape)
    else:
        se = np.full(grid.shape, np.nan)

    # ---- coverage / drift QC --------------------------------------------------------------------
    n_species = rows[:, 3]
    n_in = rows[:, 4]
    r_cov_min = float(np.nanmin(rows[:, 6])) if np.isfinite(rows[:, 6]).any() else float("inf")
    ax = grid.axes()
    gx, gy, gz = np.meshgrid(ax["x"] - grid.center[0], ax["y"] - grid.center[1], ax["z"] - grid.center[2], indexing="ij")
    covered = (np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 0.5 * grid.spacing * np.sqrt(3)) <= r_cov_min if np.isfinite(r_cov_min) \
        else np.ones(grid.shape, dtype=bool)
    if np.isfinite(r_cov_min) and not covered.all():
        warnings_list.append(
            f"{100 * (1 - covered.mean()):.0f}% of the grid lies beyond the fully covered radius "
            f"({r_cov_min:.1f} A = half the shortest box width) and is marked unreliable."
        )

    bulk = _bulk_density(cfg, grid, density, se, rows, region_coords, r_cov_min)
    rho_bulk = bulk["value"]
    g = density / rho_bulk if np.isfinite(rho_bulk) else np.full(grid.shape, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = se / density
    expected = total_counts.reshape(grid.shape)
    reliable = covered & (expected >= 10) & np.isfinite(rel) & (rel <= 0.35)
    if not np.isfinite(se).any():
        reliable = covered & (expected >= 10)

    if cfg.align:
        # drift of the solute centre between frames (minimum image: the centre may cross the box edge)
        d0 = rows[:, 7:10] - rows[0, 7:10]
        drift_max = float(np.linalg.norm(minimum_image(d0, box0), axis=1).max())
        rot_max = float(np.nanmax(rows[:, 10]))
        degenerate_frac = float(np.mean(rows[:, 11]))
    else:
        drift_max = rot_max = degenerate_frac = 0.0
    if cfg.align and degenerate_frac > 0:
        warnings_list.append(
            "The alignment selection has fewer than 3 non-collinear atoms: only translation was removed, not rotation."
        )
    inside_fraction = float(n_in.sum() / n_species.sum()) if n_species.sum() else float("nan")

    unsafe = not cfg.align  # never "safe": the engine cannot verify that the solute is static
    if unsafe and getattr(cfg, "conditioning_policy", "strict") != "unsafe":
        warnings_list.append(
            f"Density map '{cfg.name}' requested without alignment. This is scientifically unsafe for solvent density "
            "unless the trajectory was already centred and superposed on the solute."
        )

    slices = _default_slices(density)
    out_dir = os.path.join(output_dir, f"density_map_{cfg.name}")
    os.makedirs(out_dir, exist_ok=True)

    metadata: Dict[str, object] = {
        "engine": "solute",
        "algorithm_version": "2",
        "grid_spacing": grid.spacing,
        "shape": list(grid.shape),
        "origin": grid.origin.tolist(),
        "origin_convention": "centre of first voxel",
        "center": grid.center.tolist(),
        "edge_lengths": (np.asarray(grid.shape) * grid.spacing).tolist(),
        "density_unit": "Angstrom^-3",
        "frame_start": w_start,
        "frame_stop": w_stop,
        "stride": w_step,
        "n_frames": n_used,
        "n_blocks": n_blocks,
        "align": bool(cfg.align),
        "align_selection": cfg.align_selection,
        "align_reference": cfg.align_reference,
        "align_reference_path": cfg.align_reference_path,
        "align_weighting": "mass" if (cfg.align and weights is not None) else ("geometric" if cfg.align else None),
        "region_selection": region_sel,
        "region_padding": cfg.region_padding,
        "selection": selection,
        "selection_is_dynamic": bool(dyn),
        "bulk": bulk,
        "rho_bulk": rho_bulk,
        "rho_bulk_approx": rho_bulk,  # legacy key (GUI/tests)
        "species_atoms_per_frame_mean": float(np.mean(n_species)),
        "fraction_of_species_inside_grid": inside_fraction,
        "fully_covered_radius": r_cov_min,
        "solute_com_drift_max": drift_max,
        "solute_rotation_max_deg": rot_max,
        "reliable_fraction": float(reliable.mean()),
        "analysis_warnings": warnings_list,
        "notes": notes,
        "is_scientifically_safe": not unsafe and not any("fewer than 3" in w for w in warnings_list),
        "seconds": round(time.perf_counter() - t0, 3),
        "mode": mode,
        "workers": workers if mode == "parallel" else 1,
        "view_mode": cfg.view_mode,
    }

    result = DensityMapResult(
        name=cfg.name, grid_path="", grid=density, axes=ax, slices=slices, metadata=metadata,
        counts=expected, se=se, g=g, reliable=reliable,
    )
    return result


def _default_slices(grid: np.ndarray) -> Dict[str, np.ndarray]:
    slices: Dict[str, np.ndarray] = {}
    if grid.size == 0:
        return slices
    cx, cy, cz = grid.shape[0] // 2, grid.shape[1] // 2, grid.shape[2] // 2
    slices["xy"] = grid[:, :, cz]
    slices["xz"] = grid[:, cy, :]
    slices["yz"] = grid[cx, :, :]
    slices["max_projection"] = grid.max(axis=2)
    return slices


def write_density_outputs(result: DensityMapResult, cfg: DensityMapConfig, output_dir: str) -> str:
    """Write DX/NPY/NPZ grids plus ``grid.npz`` (all fields) and slice CSVs; returns the main grid path."""
    import pandas as pd

    out_dir = os.path.join(output_dir, f"density_map_{cfg.name}")
    os.makedirs(out_dir, exist_ok=True)
    fmt = (cfg.output_format or "dx").lower()
    if fmt not in ("dx", "npy", "npz"):
        fmt = "dx"
    spacing = float(result.metadata["grid_spacing"])
    origin = np.asarray(result.metadata["origin"], dtype=float)
    path = os.path.join(out_dir, f"{cfg.name}.{fmt}")
    grid = np.asarray(result.grid)
    if fmt == "dx":
        try:
            from gridData import Grid

            Grid(grid, origin=origin, delta=spacing).export(path, file_format="dx")
        except Exception as exc:  # fall back to npy so the run does not fail
            log.warning("DX export failed (%s); writing npy instead.", exc)
            path = os.path.join(out_dir, f"{cfg.name}.npy")
            np.save(path, grid)
            fmt = "npy"
    elif fmt == "npy":
        np.save(path, grid)
    else:
        np.savez(path, grid=grid)
    result.metadata["output_format"] = fmt
    result.metadata["grid_path"] = path
    result.grid_path = path
    arrays = {"grid": grid, "x": result.axes["x"], "y": result.axes["y"], "z": result.axes["z"]}
    for key in ("counts", "se", "g", "reliable"):
        val = getattr(result, key)
        if val is not None:
            arrays[key] = val
    savez_deterministic(os.path.join(out_dir, "grid.npz"), **arrays)
    result.metadata["arrays_sha256"] = arrays_sha256(arrays)
    for key, data in result.slices.items():
        if key == "xz":
            xs, ys = result.axes["x"], result.axes["z"]
        elif key == "yz":
            xs, ys = result.axes["y"], result.axes["z"]
        else:
            xs, ys = result.axes["x"], result.axes["y"]
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "density": data.ravel(), "slice": key}).to_csv(
            os.path.join(out_dir, f"{key}.csv"), index=False, lineterminator="\n")
    return path
