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

    for name, bridge_result in result.distance_bridge_results.items():
        bridge_dir = os.path.join(output_dir, f"distance_bridge_{name}")
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

    for name, bridge_result in result.hbond_bridge_results.items():
        bridge_dir = os.path.join(output_dir, f"hbond_water_bridge_{name}")
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
        if bridge_result.edge_list is not None and not bridge_result.edge_list.empty:
            _write_dataframe(
                bridge_result.edge_list,
                os.path.join(bridge_dir, "edge_list.csv"),
                project.outputs.write_parquet,
            )


    for name, hydration_result in result.hbond_hydration_results.items():
        hydration_dir = os.path.join(output_dir, f"hbond_hydration_{name}")
        os.makedirs(hydration_dir, exist_ok=True)
        _write_dataframe(
            hydration_result.table,
            os.path.join(hydration_dir, "contact_table.csv"),
            project.outputs.write_parquet,
        )
        definition = result.qc_summary.get("hbond_hydration_definitions", {}).get(name, {})
        metadata_path = os.path.join(hydration_dir, "metadata.json")
        temp_metadata = metadata_path + ".tmp"
        with open(temp_metadata, "w", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(
                    {
                        "definition": definition,
                        "solvent": project.solvent.to_dict(),
                        "analysis_frames": {
                            "frame_start": project.analysis.frame_start,
                            "frame_stop": project.analysis.frame_stop,
                            "stride": project.analysis.stride,
                        },
                        "time_unit": result.qc_summary.get("time_unit", "ps"),
                    }
                ),
                handle,
                indent=2,
            )
        _atomic_replace(temp_metadata, metadata_path)

    for name, density_result in result.density_results.items():
        density_dir = os.path.join(output_dir, f"density_map_{name}")
        os.makedirs(density_dir, exist_ok=True)
        metadata_path = os.path.join(density_dir, "metadata.json")
        temp_metadata = metadata_path + ".tmp"
        with open(temp_metadata, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(density_result.metadata), handle, indent=2)
        _atomic_replace(temp_metadata, metadata_path)

    for name, dynamics_result in result.water_dynamics_results.items():
        dynamics_dir = os.path.join(output_dir, f"water_dynamics_{name}")
        os.makedirs(dynamics_dir, exist_ok=True)
        _write_dataframe(
            dynamics_result.sp_tau,
            os.path.join(dynamics_dir, "sp_tau.csv"),
            project.outputs.write_parquet,
        )
        if dynamics_result.hbl is not None:
            _write_dataframe(
                dynamics_result.hbl,
                os.path.join(dynamics_dir, "hbl.csv"),
                project.outputs.write_parquet,
            )
        if dynamics_result.hbl_summary is not None:
            _write_dataframe(
                dynamics_result.hbl_summary,
                os.path.join(dynamics_dir, "hbl_summary.csv"),
                project.outputs.write_parquet,
            )
        if dynamics_result.wor is not None:
            _write_dataframe(
                dynamics_result.wor,
                os.path.join(dynamics_dir, "wor.csv"),
                project.outputs.write_parquet,
            )
        metadata_path = os.path.join(dynamics_dir, "metadata.json")
        temp_metadata = metadata_path + ".tmp"
        with open(temp_metadata, "w", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(
                    {
                        "mean_residence_time": dynamics_result.mean_residence_time,
                        "residence_mode": dynamics_result.residence_mode,
                        "notes": dynamics_result.notes,
                    }
                ),
                handle,
                indent=2,
            )
        _atomic_replace(temp_metadata, metadata_path)
