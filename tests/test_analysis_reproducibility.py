import pytest


pytest.importorskip("MDAnalysis")

from pathlib import Path

from engine.analysis import SOZAnalysisEngine
from engine.models import (
    AnalysisOptions,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SelectionSpec,
    SolventConfig,
    SOZDefinition,
    SOZNode,
)


def _atom_line(
    idx: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {idx:5d} {name:<4} {resname:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}"
    )


def _write_multimodel_pdb(path: Path) -> None:
    frames = [
        # frame 1
        [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), (6.0, 0.0, 0.0)],
        # frame 2
        [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (6.0, 0.0, 0.0)],
        # frame 3
        [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    ]
    lines = []
    for model_idx, coords in enumerate(frames, start=1):
        lines.append(f"MODEL        {model_idx}")
        lines.append(_atom_line(1, "CA", "ALA", "A", 1, *coords[0], "C"))
        lines.append(_atom_line(2, "NZ", "LYS", "A", 2, *coords[1], "N"))
        lines.append(_atom_line(3, "O", "HOH", "W", 3, *coords[2], "O"))
        lines.append(_atom_line(4, "O", "HOH", "W", 4, *coords[3], "O"))
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_analysis_reproducibility(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_multimodel_pdb(pdb_path)

    inputs = InputConfig(topology=str(pdb_path), trajectory=None)
    solvent = SolventConfig(
        solvent_label="Water",
        water_resnames=["HOH"],
        water_oxygen_names=["O"],
        water_hydrogen_names=[],
        include_ions=False,
        probe=ProbeConfig(selection="name O", position="atom"),
    )
    selections = {
        "sel_a": SelectionSpec(
            label="sel_a",
            selection="resid 1 and name CA",
            require_unique=True,
        ),
        "sel_b": SelectionSpec(
            label="sel_b",
            selection="resid 2 and name NZ",
            require_unique=True,
        ),
    }
    root = SOZNode(
        type="or",
        children=[
            SOZNode(
                type="distance",
                params={
                    "selection_label": "sel_a",
                    "cutoff": 2.5,
                    "unit": "A",
                    "probe_mode": "probe",
                },
            ),
            SOZNode(
                type="distance",
                params={
                    "selection_label": "sel_b",
                    "cutoff": 2.5,
                    "unit": "A",
                    "probe_mode": "probe",
                },
            ),
        ],
    )
    soz = SOZDefinition(name="SOZ", description="test", root=root)
    analysis = AnalysisOptions(frame_start=0, frame_stop=None, stride=1, store_min_distances=False)
    outputs = OutputConfig(output_dir=str(tmp_path / "out"), write_per_frame=True, write_parquet=False)
    project = ProjectConfig(
        inputs=inputs,
        solvent=solvent,
        selections=selections,
        sozs=[soz],
        analysis=analysis,
        outputs=outputs,
    )

    engine = SOZAnalysisEngine(project)
    result_a = engine.run()
    result_b = engine.run()

    per_frame = result_a.soz_results["SOZ"].per_frame
    assert per_frame["n_solvent"].tolist() == [2, 1, 0]
    summary = result_a.soz_results["SOZ"].summary
    assert summary["occupancy_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_n_solvent"] == pytest.approx(1.0)

    per_frame_b = result_b.soz_results["SOZ"].per_frame
    assert per_frame_b["n_solvent"].tolist() == per_frame["n_solvent"].tolist()
    summary_b = result_b.soz_results["SOZ"].summary
    assert summary_b["occupancy_fraction"] == summary["occupancy_fraction"]
