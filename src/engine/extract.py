"""Frame extraction for SOZLab based on occupancy rules."""
from __future__ import annotations

import json
import os
import re
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda

from engine.serialization import to_jsonable


@dataclass
class SelectionResult:
    frame_indices: List[int]
    frame_times: List[float]
    params: Dict[str, object]
    manifest: pd.DataFrame


_RULE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|>=|==|>|<)\s*([0-9]*\.?[0-9]+)\s*$")


def parse_rule(rule: str) -> Tuple[str, str, float]:
    match = _RULE_RE.match(rule)
    if not match:
        raise ValueError(f"Invalid rule format: {rule!r}")
    metric, op, value = match.groups()
    return metric, op, float(value)


def _apply_rule(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op == ">=":
        return values >= threshold
    if op == ">":
        return values > threshold
    if op == "<=":
        return values <= threshold
    if op == "<":
        return values < threshold
    if op == "==":
        return values == threshold
    raise ValueError(f"Unsupported operator: {op}")


def _close_gaps(mask: np.ndarray, gap_tolerance: int) -> np.ndarray:
    if gap_tolerance <= 0 or mask.size == 0:
        return mask
    mask = mask.copy()
    indices = np.where(mask)[0]
    if indices.size == 0:
        return mask
    runs = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    for (a_start, a_end), (b_start, b_end) in zip(runs, runs[1:]):
        gap = b_start - a_end - 1
        if gap <= gap_tolerance:
            mask[a_end + 1 : b_start] = True
    return mask


def _apply_min_run(mask: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1 or mask.size == 0:
        return mask
    mask = mask.copy()
    indices = np.where(mask)[0]
    if indices.size == 0:
        return mask
    runs = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    for start, end in runs:
        if end - start + 1 < min_run_length:
            mask[start : end + 1] = False
    return mask


def select_frames(
    per_frame: pd.DataFrame,
    rule: str,
    min_run_length: int = 1,
    gap_tolerance: int = 0,
    time_unit: str = "ps",
) -> SelectionResult:
    metric, op, threshold = parse_rule(rule)
    if metric not in per_frame.columns:
        if metric == "occupancy_fraction":
            values = (per_frame["n_solvent"].to_numpy() > 0).astype(float)
        else:
            raise ValueError(f"Metric '{metric}' not found in per_frame data.")
    else:
        values = pd.to_numeric(per_frame[metric], errors="coerce").fillna(0).to_numpy()

    mask = _apply_rule(values, op, threshold)
    mask = _close_gaps(mask, gap_tolerance)
    mask = _apply_min_run(mask, min_run_length)

    if "frame" in per_frame.columns:
        frame_indices = per_frame["frame"].to_numpy().astype(int)
    else:
        frame_indices = np.arange(len(per_frame))

    times = per_frame["time"].to_numpy() if "time" in per_frame.columns else np.zeros(len(per_frame))

    selected = np.where(mask)[0]
    selected_frames = frame_indices[selected].tolist()
    selected_times = times[selected].tolist()

    manifest = per_frame.loc[selected, ["frame", "time", "n_solvent"]].copy()
    if "solvent_ids" in per_frame.columns:
        def _hash_ids(val: str) -> str:
            return hashlib.sha1(val.encode("utf-8")).hexdigest() if val else ""
        manifest["solvent_ids_hash"] = per_frame.loc[selected, "solvent_ids"].fillna("").map(_hash_ids)

    params = {
        "rule": rule,
        "metric": metric,
        "operator": op,
        "threshold": threshold,
        "min_run_length": min_run_length,
        "gap_tolerance": gap_tolerance,
        "time_unit": time_unit,
        "selected_count": len(selected_frames),
        "total_frames": len(per_frame),
    }

    return SelectionResult(
        frame_indices=selected_frames,
        frame_times=selected_times,
        params=params,
        manifest=manifest,
    )


def write_extracted_trajectory(
    universe: mda.Universe,
    selection: SelectionResult,
    output_dir: str,
    prefix: str = "extracted",
    fmt: str = "xtc",
    progress_cb=None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    frame_indices = sorted(selection.frame_indices)
    if not frame_indices:
        params_path = os.path.join(output_dir, f"{prefix}_params.json")
        with open(params_path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(selection.params), handle, indent=2)
        return {"params": params_path}

    traj_path = os.path.join(output_dir, f"{prefix}.{fmt}")
    ref_path = os.path.join(output_dir, f"{prefix}_ref.pdb")
    manifest_path = os.path.join(output_dir, f"{prefix}_frames.csv")
    params_path = os.path.join(output_dir, f"{prefix}_params.json")

    total = len(frame_indices)
    with mda.Writer(traj_path, n_atoms=universe.atoms.n_atoms) as writer:
        for i, idx in enumerate(frame_indices, start=1):
            universe.trajectory[idx]
            writer.write(universe.atoms)
            if progress_cb:
                progress_cb(i, total, "Writing extracted trajectory")

    universe.trajectory[frame_indices[0]]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Found no information for attr.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Found chainIDs with invalid length.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Atom with index >=100000 cannot write bonds.*",
            category=UserWarning,
        )
        universe.atoms.write(ref_path)
    selection.manifest.to_csv(manifest_path, index=False)
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(selection.params), handle, indent=2)

    return {
        "trajectory": traj_path,
        "reference": ref_path,
        "manifest": manifest_path,
        "params": params_path,
    }
