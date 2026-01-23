import pytest

pytest.importorskip("MDAnalysis")
import MDAnalysis as mda

from engine.models import (
    ProjectConfig,
    InputConfig,
    SolventConfig,
    SelectionSpec,
    SOZDefinition,
    SOZNode,
    AnalysisOptions,
    OutputConfig,
)
from engine.preflight import run_preflight


def _make_universe():
    u = mda.Universe.empty(
        n_atoms=4,
        n_residues=2,
        atom_resindex=[0, 0, 1, 1],
        residue_segindex=[0, 0],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["O", "H1", "NZ", "CA"])
    u.add_TopologyAttr("resname", ["WAT", "LYS"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["A"])
    return u


def _make_project(selection: str, water_resnames=None):
    if water_resnames is None:
        water_resnames = ["WAT"]
    sel = SelectionSpec(label="selection_a", selection=selection, require_unique=True)
    root = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 3.5, "unit": "A"}
    )
    return ProjectConfig(
        inputs=InputConfig(topology="dummy"),
        solvent=SolventConfig(water_resnames=water_resnames, water_oxygen_names=["O"]),
        selections={"selection_a": sel},
        sozs=[SOZDefinition(name="SOZ_1", description="", root=root)],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="results"),
    )


def test_preflight_operator_leading_selection():
    u = _make_universe()
    project = _make_project("and name O")
    report = run_preflight(project, u)
    assert not report.ok
    assert any("starts with an operator" in err for err in report.errors)


def test_preflight_zero_atom_seed():
    u = _make_universe()
    project = _make_project("resid 999 and name O")
    report = run_preflight(project, u)
    assert not report.ok
    assert any("resolved to 0 atoms" in err for err in report.errors)


def test_preflight_solvent_mismatch():
    u = _make_universe()
    project = _make_project("name O", water_resnames=["TIP3"])
    report = run_preflight(project, u)
    assert not report.ok
    assert any("Water resnames did not match" in err for err in report.errors)


def test_preflight_non_unique_seed():
    u = _make_universe()
    project = _make_project("resname WAT", water_resnames=["WAT"])
    report = run_preflight(project, u)
    assert not report.ok
    assert any("expected unique atom" in err for err in report.errors)


def test_preflight_wildcard_resname_selection():
    u = mda.Universe.empty(
        n_atoms=2,
        n_residues=1,
        atom_resindex=[0, 0],
        residue_segindex=[0],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["NE2", "CA"])
    u.add_TopologyAttr("resname", ["HSD"])
    u.add_TopologyAttr("resid", [223])
    u.add_TopologyAttr("segid", ["A"])

    sel = SelectionSpec(
        label="selection_a", selection="resname HS* and name NE2", require_unique=True
    )
    root = SOZNode(
        type="distance", params={"selection_label": "selection_a", "cutoff": 3.5, "unit": "A"}
    )
    project = ProjectConfig(
        inputs=InputConfig(topology="dummy"),
        solvent=SolventConfig(water_resnames=["HSD"], water_oxygen_names=["NE2"]),
        selections={"selection_a": sel},
        sozs=[SOZDefinition(name="SOZ_1", description="", root=root)],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="results"),
    )
    report = run_preflight(project, u)
    assert report.ok
    assert report.selection_checks["selection_a"].count == 1
