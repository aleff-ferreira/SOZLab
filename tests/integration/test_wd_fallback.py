
import pytest
import numpy as np
import logging
from engine.analysis import SOZAnalysisEngine
from engine.models import (
    ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, 
    OutputConfig, SOZDefinition, SOZNode, WaterDynamicsConfig, SelectionSpec
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

def test_water_dynamics_fallback_with_selection(tmp_path):
    """
    Regression test: Ensure HBL summary is computed (via fallback) when waterdynamics module is missing/mocked,
    provided solute_selection is valid.
    """
    # Use laao_water data
    data_dir = "/home/aleff/laao_water"
    topo = f"{data_dir}/step3_input.gro"
    traj = f"{data_dir}/step7_production_mol_rect_3_system_10ns.xtc"

    inputs = InputConfig(topology=topo, trajectory=traj)
    solvent = SolventConfig(solvent_label="Water")
    
    # We define SOZ just to satisfy dependencies, but use selection mode for WD to be direct
    soz_root = SOZNode(type="shell", params={"selection_label": "active_site", "distance": 5.0})
    sozs = [SOZDefinition(name="SOZ_1", description="wd fallback smoke", root=soz_root)]
    selections = {"active_site": SelectionSpec(label="active_site", selection="resname FAD")}

    # Config with explicit solute selection to trigger HBL
    wd = WaterDynamicsConfig(
        name="Test_Dynamics",
        region_mode="soz",
        soz_name="SOZ_1",
        residence_mode="continuous",
        solute_selection="resname FAD", # Required for HBL
        donors_selection="name N* O*",    # Required for GRO file without elements
        acceptors_selection="name N* O*"
    )

    project = ProjectConfig(
        inputs=inputs,
        solvent=solvent,
        sozs=sozs,
        selections=selections,
        water_dynamics=[wd],
        analysis=AnalysisOptions(frame_start=0, frame_stop=20, stride=5),
        outputs=OutputConfig(output_dir=str(tmp_path))
    )

    engine = SOZAnalysisEngine(project)
    res = engine.run()
    
    wd_res = res.water_dynamics_results.get("Test_Dynamics")
    assert wd_res is not None
    # We expect HBL summary to be populated (fallback logic checks solute_selection)
    # Even if waterdynamics module is missing.
    # Note: If MDAnalysis HBA fails to find bonds, it returns empty table, but it SHOULD exist (not None).
    assert wd_res.hbl_summary is not None
    # We hope for rows, but even empty table confirms the logic ran to completion (didn't crash).
    if not wd_res.hbl_summary.empty:
        assert "mean_lifetime" in wd_res.hbl_summary.columns
