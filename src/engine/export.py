"""Export utilities for SOZLab results."""
from __future__ import annotations

import json
import os

import pandas as pd

from engine.analysis import AnalysisResult
from engine.serialization import to_jsonable
from engine.models import ProjectConfig


def _atomic_replace(temp_path: str, final_path: str) -> None:
    os.replace(temp_path, final_path)


def _write_dataframe(df: pd.DataFrame, path_csv: str, write_parquet: bool) -> None:
    temp_csv = path_csv + ".tmp"
    df.to_csv(temp_csv, index=False)
    _atomic_replace(temp_csv, path_csv)
    if write_parquet:
        parquet_path = os.path.splitext(path_csv)[0] + ".parquet"
        temp_parquet = parquet_path + ".tmp"
        df.to_parquet(temp_parquet, index=False)
        _atomic_replace(temp_parquet, parquet_path)


def export_results(result: AnalysisResult, project: ProjectConfig) -> None:
    output_dir = project.outputs.output_dir
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "project": project.to_dict(),
        "warnings": result.warnings,
        "qc_summary": result.qc_summary,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    temp_metadata = metadata_path + ".tmp"
    with open(temp_metadata, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metadata), handle, indent=2)
    _atomic_replace(temp_metadata, metadata_path)

    for name, soz_result in result.soz_results.items():
        soz_dir = os.path.join(output_dir, f"soz_{name}")
        os.makedirs(soz_dir, exist_ok=True)

        _write_dataframe(
            soz_result.per_frame,
            os.path.join(soz_dir, "per_frame.csv"),
            project.outputs.write_parquet,
        )
        _write_dataframe(
            soz_result.per_solvent,
            os.path.join(soz_dir, "per_solvent.csv"),
            project.outputs.write_parquet,
        )
        summary_path = os.path.join(soz_dir, "summary.json")
        temp_summary = summary_path + ".tmp"
        with open(temp_summary, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(soz_result.summary), handle, indent=2)
        _atomic_replace(temp_summary, summary_path)

        if soz_result.min_distance_traces is not None:
            _write_dataframe(
                soz_result.min_distance_traces,
                os.path.join(soz_dir, "min_distance_traces.csv"),
                project.outputs.write_parquet,
            )

    for name, bridge_result in result.bridge_results.items():
        bridge_dir = os.path.join(output_dir, f"bridge_{name}")
        os.makedirs(bridge_dir, exist_ok=True)
        _write_dataframe(
            bridge_result.per_frame,
            os.path.join(bridge_dir, "per_frame.csv"),
            project.outputs.write_parquet,
        )
        _write_dataframe(
            bridge_result.per_solvent,
            os.path.join(bridge_dir, "per_solvent.csv"),
            project.outputs.write_parquet,
        )

    for name, hydration_result in result.hydration_results.items():
        hydration_dir = os.path.join(output_dir, f"hydration_{name}")
        os.makedirs(hydration_dir, exist_ok=True)
        _write_dataframe(
            hydration_result.table,
            os.path.join(hydration_dir, "hydration_table.csv"),
            project.outputs.write_parquet,
        )
