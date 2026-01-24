import os
import subprocess

import pytest

pytest.importorskip("MDAnalysis")
import numpy as np
import MDAnalysis as mda

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
from engine.resolver import resolve_selection
from engine.solvent import build_solvent
from engine.soz_eval import EvaluationContext, evaluate_node


def _water_universe():
    u = mda.Universe.empty(
        n_atoms=2,
        n_residues=2,
        atom_resindex=[0, 1],
        residue_segindex=[0, 0],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA", "O"])
    u.add_TopologyAttr("resname", ["ALA", "SOL"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["A"])
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ]
    )
    u.load_new(coords, order="fac")
    return u


def _counts_for_project(project: ProjectConfig, universe: mda.Universe) -> list[int]:
    solvent = build_solvent(universe, project.solvent)
    selections = {
        label: resolve_selection(universe, spec) for label, spec in project.selections.items()
    }
    context = EvaluationContext(universe, solvent, selections)
    counts = []
    for _ in universe.trajectory:
        counts.append(len(evaluate_node(project.sozs[0].root, context)))
    return counts


def test_probe_water_regression():
    u = _water_universe()
    sel = SelectionSpec(label="selection_a", selection="resid 1 and name CA", require_unique=True)
    root = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 3.5, "unit": "A"}
    )
    legacy = ProjectConfig.from_dict(
        {
            "version": "1.0",
            "inputs": {"topology": "dummy"},
            "solvent": {"water_resnames": ["SOL"], "water_oxygen_names": ["O"]},
            "selections": {
                "selection_a": {
                    "label": "selection_a",
                    "selection": "resid 1 and name CA",
                    "require_unique": True,
                }
            },
            "sozs": [
                {
                    "name": "SOZ_1",
                    "description": "",
                    "root": {
                        "type": "distance",
                        "params": {
                            "selection_label": "selection_a",
                            "cutoff": 3.5,
                            "unit": "A",
                        },
                        "children": [],
                    },
                }
            ],
            "analysis": {"stride": 1},
            "outputs": {"output_dir": "results"},
        }
    )
    probe = ProjectConfig(
        inputs=InputConfig(topology="dummy"),
        solvent=SolventConfig(
            water_resnames=["SOL"],
            probe=ProbeConfig(selection="name O", position="atom"),
        ),
        selections={"selection_a": sel},
        sozs=[SOZDefinition(name="SOZ_1", description="", root=root)],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="results"),
    )

    assert _counts_for_project(legacy, u) == _counts_for_project(probe, u)


def _seed_solvent_universe(solvent_resname: str, solvent_atoms: list[str], solvent_coords: list[list[float]]):
    n_atoms = 1 + len(solvent_atoms)
    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=2,
        atom_resindex=[0] + [1] * len(solvent_atoms),
        residue_segindex=[0, 0],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA"] + solvent_atoms)
    u.add_TopologyAttr("resname", ["ALA", solvent_resname])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["A"])
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    coords = np.array([[ [0.0, 0.0, 0.0] ] + solvent_coords])
    u.load_new(coords, order="fac")
    return u


def test_probe_nonwater_methanol():
    u = _seed_solvent_universe("MET", ["C", "O", "H1"], [[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    solvent = build_solvent(
        u,
        SolventConfig(
            water_resnames=["MET"],
            probe=ProbeConfig(selection="name O", position="atom"),
        ),
    )
    sel = resolve_selection(u, SelectionSpec(label="selection_a", selection="resid 1 and name CA"))
    context = EvaluationContext(u, solvent, {"selection_a": sel})
    node = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 3.0, "unit": "A"}
    )
    u.trajectory[0]
    assert solvent.residues[0].resindex in evaluate_node(node, context)


def test_probe_nonwater_dmso():
    u = _seed_solvent_universe("DMS", ["S", "O", "C1"], [[2.0, 0.0, 0.0], [2.6, 0.0, 0.0], [3.2, 0.0, 0.0]])
    solvent = build_solvent(
        u,
        SolventConfig(
            water_resnames=["DMS"],
            probe=ProbeConfig(selection="name S", position="atom"),
        ),
    )
    sel = resolve_selection(u, SelectionSpec(label="selection_a", selection="resid 1 and name CA"))
    context = EvaluationContext(u, solvent, {"selection_a": sel})
    node = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 2.5, "unit": "A"}
    )
    u.trajectory[0]
    assert solvent.residues[0].resindex in evaluate_node(node, context)


def test_probe_com_position():
    u = _seed_solvent_universe("ETH", ["C1", "C2"], [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    solvent_com = build_solvent(
        u,
        SolventConfig(
            water_resnames=["ETH"],
            probe=ProbeConfig(selection="name C1 C2", position="com"),
        ),
    )
    sel = resolve_selection(u, SelectionSpec(label="selection_a", selection="resid 1 and name CA"))
    context = EvaluationContext(u, solvent_com, {"selection_a": sel})
    node = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 5.0, "unit": "A"}
    )
    u.trajectory[0]
    assert solvent_com.residues[0].resindex not in evaluate_node(node, context)


def test_gmx_mpi_crosscheck_distance(tmp_path):
    gmx_bin = "/usr/local/gromacs20253/bin/gmx_mpi"
    if not os.path.exists(gmx_bin):
        pytest.skip("gmx_mpi not available")
    try:
        version = subprocess.run(
            [gmx_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        pytest.skip("gmx_mpi not runnable in this environment")
    if version.returncode != 0:
        pytest.skip("gmx_mpi not runnable in this environment")

    gro_path = tmp_path / "mini.gro"
    gro_path.write_text(
        "\n".join(
            [
                "Mini",
                "2",
                "    1RES  A1    1   0.000   0.000   0.000",
                "    1RES  A2    2   0.100   0.000   0.000",
                "   1.00000   1.00000   1.00000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "dist.xvg"
    cmd = [
        gmx_bin,
        "distance",
        "-s",
        str(gro_path),
        "-f",
        str(gro_path),
        "-select",
        "atomnr 1 plus atomnr 2",
        "-oall",
        str(out_path),
        "-quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
        pytest.skip("gmx_mpi distance failed in this environment")

    data_lines = [
        line for line in out_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "@"))
    ]
    assert data_lines, "No distance data produced by gmx_mpi"
    last = data_lines[-1].split()
    assert len(last) >= 2
    distance_nm = float(last[1])
    assert np.isclose(distance_nm, 0.1, atol=1e-3)
