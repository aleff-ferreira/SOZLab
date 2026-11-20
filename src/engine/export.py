"""Export utilities for SOZLab results.

Guarantees
----------
* Every file is written through a temp file and renamed, so readers never see half a file.
* JSON is strict (NaN/inf become ``null``); CSV uses ``\\n`` line endings on every platform, so
  identical results produce byte-identical files on Windows, Linux and macOS.
* A *partial* result (cancelled run) is never written over a complete export: it needs
  ``allow_partial=True`` and goes to ``<output_dir>/partial``.
* ``provenance.json`` records what produced the files; ``manifest.json`` (written last) lists
  every file with its SHA-256 so an export can be verified later with :func:`verify_export`.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from engine._version import SCHEMA_VERSION, __version__
from engine.analysis import AnalysisResult
from engine.density import write_density_outputs
from engine.dictionary import data_dictionary_markdown
from engine.models import DensityMapConfig, ProjectConfig
from engine.provenance import output_manifest, write_json_atomic


class PartialResultError(RuntimeError):
    """Raised when a cancelled (partial) result would be exported as if it were complete."""


def _atomic_replace(temp_path: str, final_path: str) -> None:
    os.replace(temp_path, final_path)


def _write_dataframe(df: pd.DataFrame, path_csv: str, write_parquet: bool) -> None:
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    temp_csv = path_csv + ".tmp"
    df.to_csv(temp_csv, index=False, lineterminator="\n")
    _atomic_replace(temp_csv, path_csv)
    if write_parquet:
        parquet_path = os.path.splitext(path_csv)[0] + ".parquet"
        temp_parquet = parquet_path + ".tmp"
        df.to_parquet(temp_parquet, index=False)
        _atomic_replace(temp_parquet, parquet_path)


def export_results(
    result: AnalysisResult,
    project: ProjectConfig,
    allow_partial: bool = False,
) -> Dict[str, str]:
    """Write all outputs; returns ``{"output_dir": ..., "manifest": ...}``."""
    complete = getattr(result, "complete", True)
    if not complete and not allow_partial:
        raise PartialResultError(
            "The run was cancelled, so its results are partial. They were not exported, to protect any complete "
            "results already in the output directory. Re-run, or export explicitly with allow_partial=True "
            "(written to the 'partial' subfolder)."
        )
    output_dir = project.outputs.output_dir
    if not complete:
        output_dir = os.path.join(output_dir, "partial")
    os.makedirs(output_dir, exist_ok=True)
    write_parquet = project.outputs.write_parquet
    provenance = {k: v for k, v in (result.provenance or {}).items() if k != "frame_fingerprints"}

    run_status = {
        "status": "complete" if complete else "partial",
        "run_id": provenance.get("run_id"),
        "frames_analysed": result.qc_summary.get("frames_analysed"),
        "frames_requested": result.qc_summary.get("total_frames"),
        "sozlab_version": __version__,
        "schema_version": SCHEMA_VERSION,
    }
    write_json_atomic(os.path.join(output_dir, "run_status.json"), run_status)

    project_dict = project.to_dict()
    write_json_atomic(
        os.path.join(output_dir, "metadata.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "project": project_dict,
            "warnings": result.warnings,
            "qc_summary": result.qc_summary,
            "provenance": provenance,
        },
    )
    write_json_atomic(os.path.join(output_dir, "provenance.json"), provenance)

    for name, soz_result in result.soz_results.items():
        soz_dir = os.path.join(output_dir, f"soz_{name}")
        os.makedirs(soz_dir, exist_ok=True)
        _write_dataframe(soz_result.per_frame, os.path.join(soz_dir, "per_frame.csv"), write_parquet)
        _write_dataframe(soz_result.per_solvent, os.path.join(soz_dir, "per_solvent.csv"), write_parquet)
        write_json_atomic(os.path.join(soz_dir, "summary.json"), soz_result.summary)
        if soz_result.min_distance_traces is not None:
            _write_dataframe(soz_result.min_distance_traces, os.path.join(soz_dir, "min_distance_traces.csv"), write_parquet)
        if soz_result.survival is not None:
            _write_dataframe(soz_result.survival, os.path.join(soz_dir, "survival.csv"), False)
        if soz_result.visits is not None:
            _write_dataframe(soz_result.visits, os.path.join(soz_dir, "visits.csv"), write_parquet)

    cfgs = {c.name: c for c in project.density_maps}
    for name, density_result in result.density_results.items():
        cfg: Optional[DensityMapConfig] = cfgs.get(name)
        density_dir = os.path.join(output_dir, f"density_map_{name}")
        os.makedirs(density_dir, exist_ok=True)
        if cfg is not None:
            write_density_outputs(density_result, cfg, output_dir)  # self-contained even when exporting elsewhere
        write_json_atomic(os.path.join(density_dir, "metadata.json"), density_result.metadata)

    with open(os.path.join(output_dir, "DATA_DICTIONARY.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data_dictionary_markdown(__version__, SCHEMA_VERSION))

    manifest = output_manifest(output_dir)
    manifest["run_id"] = provenance.get("run_id")
    stale = _stale_entries(output_dir, set(result.soz_results), set(result.density_results))
    if stale:
        manifest["not_part_of_this_run"] = stale
    write_json_atomic(os.path.join(output_dir, "manifest.json"), manifest)
    return {"output_dir": output_dir, "manifest": os.path.join(output_dir, "manifest.json")}


def _stale_entries(output_dir: str, soz_names: set, density_names: set) -> List[str]:
    """Directories left over from earlier runs (never deleted, but reported)."""
    stale: List[str] = []
    for entry in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, entry)
        if not os.path.isdir(path):
            continue
        if entry.startswith("soz_") and entry[4:] not in soz_names:
            stale.append(entry)
        if entry.startswith("density_map_") and entry[len("density_map_"):] not in density_names:
            stale.append(entry)
    return stale


def verify_export(output_dir: str) -> List[str]:
    """Return differences between ``manifest.json`` and the files now on disk (empty = intact)."""
    with open(os.path.join(output_dir, "manifest.json"), "r", encoding="utf-8") as handle:
        recorded = json.load(handle)["files"]
    current = output_manifest(output_dir)["files"]
    problems: List[str] = []
    for rel, info in recorded.items():
        now = current.get(rel)
        if now is None:
            problems.append(f"missing: {rel}")
        elif now["sha256"] != info["sha256"]:
            problems.append(f"modified: {rel}")
    return problems
