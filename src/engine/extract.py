"""Frame extraction for SOZLab based on occupancy rules.

Extraction used to trust frame *numbers*: the analysis results said "frames 10-14 are occupied"
and the extractor opened whatever trajectory the project pointed at *now* and copied frames
10-14. If that file had been replaced, extended or re-processed since the analysis, the output
silently contained other frames. Here every written frame is verified against what the analysis
read:

* the **time stamp** of the frame must match the recorded one, and
* its **frame fingerprint** (checksum of ~128 spread atoms + box, recorded in ``per_frame.csv``
  as ``frame_fp``) must match, which proves the coordinates are the analysed ones.

Nothing is finalised until every frame verified: files are staged in a temporary directory and
moved into place at the end, and earlier extractions are never overwritten unless asked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda

from engine.preflight import collect_metadata_warnings
from engine.provenance import (
    InputsChangedError,
    frame_fingerprint,
    sample_atom_indices,
    verify_inputs,
    write_json_atomic,
)
from engine.serialization import to_jsonable

SUPPORTED_FORMATS = ("xtc", "trr", "dcd")


class ExtractionMismatchError(RuntimeError):
    """The trajectory being read does not contain the frames that were analysed."""

    def __init__(self, message: str, mismatches: Optional[List[str]] = None):
        self.mismatches = list(mismatches or [])
        super().__init__(message)


@dataclass
class SelectionResult:
    frame_indices: List[int]
    frame_times: List[float]
    params: Dict[str, object]
    manifest: pd.DataFrame
    frame_fps: Optional[List[str]] = None  # fingerprints recorded at analysis time
    inputs: Optional[Dict[str, object]] = None  # input fingerprints recorded at analysis time
    n_atoms: Optional[int] = None


_RULE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|>=|==|>|<)\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")


def parse_rule(rule: str) -> Tuple[str, str, float]:
    match = _RULE_RE.match(rule)
    if not match:
        raise ValueError(f"Invalid rule format: {rule!r}")
    metric, op, value = match.groups()
    return metric, op, float(value)


def _apply_rule(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if op == ">=":
            out = values >= threshold
        elif op == ">":
            out = values > threshold
        elif op == "<=":
            out = values <= threshold
        elif op == "<":
            out = values < threshold
        elif op == "==":
            out = values == threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return out & np.isfinite(values)  # a missing value never satisfies a rule


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _close_gaps(mask: np.ndarray, gap_tolerance: int) -> np.ndarray:
    if gap_tolerance <= 0 or mask.size == 0:
        return mask
    mask = mask.copy()
    runs = _runs(mask)
    for (_, a_end), (b_start, _) in zip(runs, runs[1:]):
        if b_start - a_end - 1 <= gap_tolerance:
            mask[a_end + 1 : b_start] = True
    return mask


def _apply_min_run(mask: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1 or mask.size == 0:
        return mask
    mask = mask.copy()
    for start, end in _runs(mask):
        if end - start + 1 < min_run_length:
            mask[start : end + 1] = False
    return mask


def select_frames(
    per_frame: pd.DataFrame,
    rule: str,
    min_run_length: int = 1,
    gap_tolerance: int = 0,
    time_unit: str = "ps",
    min_run_time: Optional[float] = None,
    gap_tolerance_time: Optional[float] = None,
    inputs: Optional[Dict[str, object]] = None,
) -> SelectionResult:
    """Choose analysed frames by a rule such as ``n_solvent>=2``.

    ``min_run_length`` / ``gap_tolerance`` count *analysed* frames (rows of ``per_frame``), so with a
    stride of 10 a run of 5 means 50 trajectory frames. ``min_run_time`` / ``gap_tolerance_time`` (ps) state
    the same thing in time and are converted with the median sampling interval of ``per_frame``.
    """
    per_frame = per_frame.reset_index(drop=True)  # positional logic below must not depend on a filtered index
    metric, op, threshold = parse_rule(rule)
    if metric not in per_frame.columns:
        if metric == "occupancy_fraction":
            values = (pd.to_numeric(per_frame["n_solvent"], errors="coerce").to_numpy(float) > 0).astype(float)
        else:
            raise ValueError(f"Metric '{metric}' not found in per_frame data.")
    else:
        values = pd.to_numeric(per_frame[metric], errors="coerce").to_numpy(float)

    times_all = per_frame["time"].to_numpy(float) if "time" in per_frame.columns else None
    dt = float("nan")
    if times_all is not None and len(times_all) > 1:
        diffs = np.diff(times_all)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        dt = float(np.median(diffs)) if diffs.size else float("nan")
    if gap_tolerance_time is not None:
        if np.isfinite(dt):
            gap_tolerance = int(np.floor(gap_tolerance_time / dt + 1e-9))
        else:
            raise ValueError("gap_tolerance_time needs at least two frames with increasing time stamps")
    if min_run_time is not None:
        if np.isfinite(dt):
            min_run_length = int(np.ceil(min_run_time / dt - 1e-9))
        else:
            raise ValueError("min_run_time needs at least two frames with increasing time stamps")

    mask = _apply_rule(values, op, threshold)
    mask = _close_gaps(mask, int(gap_tolerance))
    mask = _apply_min_run(mask, int(min_run_length))

    frame_indices = per_frame["frame"].to_numpy().astype(int) if "frame" in per_frame.columns else np.arange(len(per_frame))
    times = times_all if times_all is not None else np.zeros(len(per_frame))

    selected = np.flatnonzero(mask)
    cols = [c for c in ("frame", "time", "n_solvent") if c in per_frame.columns]
    manifest = per_frame.iloc[selected][cols].copy()
    if "solvent_ids" in per_frame.columns:
        manifest["solvent_ids_hash"] = per_frame.iloc[selected]["solvent_ids"].fillna("").map(
            lambda v: hashlib.sha1(str(v).encode("utf-8")).hexdigest() if v else ""
        )
    fps = None
    if "frame_fp" in per_frame.columns:
        fps = per_frame.iloc[selected]["frame_fp"].astype(str).tolist()
        manifest["frame_fp"] = fps

    params = {
        "rule": rule,
        "metric": metric,
        "operator": op,
        "threshold": threshold,
        "min_run_length": int(min_run_length),
        "gap_tolerance": int(gap_tolerance),
        "min_run_time": min_run_time,
        "gap_tolerance_time": gap_tolerance_time,
        "sampling_interval": None if not np.isfinite(dt) else dt,
        "time_unit": time_unit,
        "selected_count": int(len(selected)),
        "total_frames": int(len(per_frame)),
        "content_verification": "frame_fp" if fps is not None else "time stamps only (per_frame has no frame_fp)",
    }
    return SelectionResult(
        frame_indices=frame_indices[selected].tolist(),
        frame_times=times[selected].tolist(),
        params=params,
        manifest=manifest,
        frame_fps=fps,
        inputs=inputs,
    )


def _unique_prefix(output_dir: str, prefix: str, fmt: str) -> str:
    def taken(p: str) -> bool:
        return any(os.path.exists(os.path.join(output_dir, f"{p}{suffix}"))
                   for suffix in (f".{fmt}", "_ref.pdb", "_frames.csv", "_params.json"))

    if not taken(prefix):
        return prefix
    i = 2
    while taken(f"{prefix}_{i}"):
        i += 1
    return f"{prefix}_{i}"


def verify_frames(universe: mda.Universe, selection: SelectionResult,
                  cancel: Optional[Callable[[], bool]] = None) -> Dict[str, object]:
    """Check that the frames in ``universe`` are the ones the analysis read (no output is written)."""
    problems: List[str] = []
    n_atoms = len(universe.atoms)
    if selection.n_atoms is not None and selection.n_atoms != n_atoms:
        raise ExtractionMismatchError(
            f"The trajectory has {n_atoms} atoms but {selection.n_atoms} were analysed.")
    n_frames = len(universe.trajectory)
    idx = sample_atom_indices(n_atoms)
    use_fp = selection.frame_fps is not None
    for k, frame in enumerate(selection.frame_indices):
        if cancel and cancel():
            raise InterruptedError("extraction cancelled")
        if frame >= n_frames:
            problems.append(f"frame {frame} does not exist (trajectory has {n_frames} frames)")
            continue
        ts = universe.trajectory[frame]
        want_t = selection.frame_times[k]
        if not np.isclose(ts.time, want_t, rtol=1e-6, atol=1e-3):
            problems.append(f"frame {frame}: time {ts.time:.6g} ps, analysed at {want_t:.6g} ps")
            continue
        if use_fp:
            got = frame_fingerprint(ts.positions, ts.dimensions, idx)
            if got != selection.frame_fps[k]:
                problems.append(f"frame {frame}: coordinates differ from the analysed frame (fingerprint {got} != {selection.frame_fps[k]})")
        if len(problems) >= 20:
            break
    if problems:
        raise ExtractionMismatchError(
            "The trajectory does not contain the frames that were analysed; nothing was written. "
            + "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""),
            problems,
        )
    return {"mode": "frame_fp+time" if use_fp else "time", "n_frames_verified": len(selection.frame_indices)}


def write_extracted_trajectory(
    universe: mda.Universe,
    selection: SelectionResult,
    output_dir: str,
    prefix: str = "extracted",
    fmt: str = "xtc",
    progress_cb=None,
    verify: bool = True,
    overwrite: bool = False,
    cancel: Optional[Callable[[], bool]] = None,
    topology_path: Optional[str] = None,
    trajectory_path: Optional[str] = None,
) -> Dict[str, str]:
    """Write the selected frames after verifying them; returns the written paths."""
    fmt = (fmt or "xtc").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported extraction format '{fmt}' (supported: {', '.join(SUPPORTED_FORMATS)})")
    os.makedirs(output_dir, exist_ok=True)
    frame_indices = sorted(selection.frame_indices)
    if frame_indices != list(selection.frame_indices):
        order = np.argsort(selection.frame_indices, kind="stable")
        selection = SelectionResult(
            [selection.frame_indices[i] for i in order], [selection.frame_times[i] for i in order],
            selection.params, selection.manifest, None if selection.frame_fps is None else [selection.frame_fps[i] for i in order],
            selection.inputs, selection.n_atoms)

    verification: Dict[str, object] = {"mode": "skipped", "n_frames_verified": 0}
    if verify and selection.inputs and (topology_path or trajectory_path):
        diffs = verify_inputs(selection.inputs, topology_path, trajectory_path)
        if diffs:
            raise InputsChangedError(diffs)
        verification["inputs"] = "files identical to those analysed (content hash)"

    if not frame_indices:
        pfx = prefix if overwrite else _unique_prefix(output_dir, prefix, fmt)
        params_path = os.path.join(output_dir, f"{pfx}_params.json")
        write_json_atomic(params_path, selection.params)
        return {"params": params_path}

    if verify:
        verification.update(verify_frames(universe, selection, cancel))

    pfx = prefix if overwrite else _unique_prefix(output_dir, prefix, fmt)
    stage = tempfile.mkdtemp(prefix=".sozlab_extract_", dir=output_dir)
    try:
        traj_tmp = os.path.join(stage, f"{pfx}.{fmt}")
        total = len(frame_indices)
        with mda.Writer(traj_tmp, n_atoms=universe.atoms.n_atoms) as writer:
            for i, idx in enumerate(frame_indices, start=1):
                if cancel and cancel():
                    raise InterruptedError("extraction cancelled")
                universe.trajectory[idx]
                writer.write(universe.atoms)
                if progress_cb:
                    progress_cb(i, total, "Writing extracted trajectory")

        metadata_warnings = collect_metadata_warnings(universe)
        seen = set(metadata_warnings)
        universe.trajectory[frame_indices[0]]
        ref_tmp = os.path.join(stage, f"{pfx}_ref.pdb")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            universe.atoms.write(ref_tmp)
        for warning in caught:
            message = str(warning.message)
            if message and message not in seen:
                metadata_warnings.append(message)
                seen.add(message)

        manifest_tmp = os.path.join(stage, f"{pfx}_frames.csv")
        selection.manifest.to_csv(manifest_tmp, index=False, lineterminator="\n")
        payload = dict(selection.params)
        payload.update(
            {
                "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "format": fmt,
                "coordinate_precision": "0.001 nm (XTC lossy compression)" if fmt == "xtc" else "float32 (no compression; unit conversion rounds ~2e-6 A)",
                "n_frames_written": total,
                "n_atoms": int(universe.atoms.n_atoms),
                "verification": verification,
                "analysed_inputs": selection.inputs,
            }
        )
        if metadata_warnings:
            payload["warnings"] = metadata_warnings
        params_tmp = os.path.join(stage, f"{pfx}_params.json")
        with open(params_tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, allow_nan=False)

        # everything verified and written: move into place
        final = {}
        for key, tmp in (("trajectory", traj_tmp), ("reference", ref_tmp), ("manifest", manifest_tmp), ("params", params_tmp)):
            dest = os.path.join(output_dir, os.path.basename(tmp))
            os.replace(tmp, dest)
            final[key] = dest
        if metadata_warnings:
            final["warnings"] = metadata_warnings
        return final
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def extract_from_results(
    results_dir: str,
    soz: Optional[str],
    rule: str,
    output_dir: str,
    topology: Optional[str] = None,
    trajectory: Optional[str] = None,
    min_run_length: int = 1,
    gap_tolerance: int = 0,
    min_run_time: Optional[float] = None,
    gap_tolerance_time: Optional[float] = None,
    prefix: str = "extracted",
    fmt: str = "xtc",
    overwrite: bool = False,
    progress_cb=None,
) -> Tuple[Dict[str, str], SelectionResult]:
    """Extract frames from a saved run without re-running the analysis; inputs are verified by content."""
    with open(os.path.join(results_dir, "provenance.json"), "r", encoding="utf-8") as handle:
        prov = json.load(handle)
    soz_dirs = sorted(d[4:] for d in os.listdir(results_dir) if d.startswith("soz_") and os.path.isdir(os.path.join(results_dir, d)))
    if not soz_dirs:
        raise ValueError(f"No soz_* results found in {results_dir}")
    name = soz or soz_dirs[0]
    if name not in soz_dirs:
        raise ValueError(f"SOZ '{name}' not found in {results_dir} (available: {', '.join(soz_dirs)})")
    per_frame = pd.read_csv(os.path.join(results_dir, f"soz_{name}", "per_frame.csv"))
    inputs = prov.get("inputs") or {}
    topo_rec = (inputs.get("topology") or {}).get("path")
    traj_rec = (inputs.get("trajectory") or {}).get("path")
    topology = topology or topo_rec
    trajectory = trajectory or traj_rec
    if not trajectory:
        raise ValueError("The analysed run had no trajectory; nothing to extract.")
    selection = select_frames(per_frame, rule, min_run_length, gap_tolerance, inputs=inputs,
                              min_run_time=min_run_time, gap_tolerance_time=gap_tolerance_time)
    selection.n_atoms = (prov.get("trajectory") or {}).get("n_atoms")
    universe = mda.Universe(topology, trajectory)
    outputs = write_extracted_trajectory(universe, selection, output_dir, prefix=prefix, fmt=fmt,
                                         progress_cb=progress_cb, overwrite=overwrite,
                                         topology_path=topology, trajectory_path=trajectory)
    return outputs, selection


def select_from_result(result, soz_name: str, rule: str, **kwargs) -> SelectionResult:
    """Select frames from an in-memory :class:`AnalysisResult`, carrying its input identity.

    The returned selection knows the fingerprints of the files that were analysed and the atom
    count, so :func:`write_extracted_trajectory` can refuse to read from anything else.
    """
    per_frame = result.soz_results[soz_name].per_frame
    prov = getattr(result, "provenance", None) or {}
    sel = select_frames(per_frame, rule, inputs=prov.get("inputs"), **kwargs)
    sel.n_atoms = (prov.get("trajectory") or {}).get("n_atoms")
    return sel
