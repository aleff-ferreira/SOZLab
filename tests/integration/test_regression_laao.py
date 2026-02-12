
import pytest
import numpy as np
import logging
from pathlib import Path
from engine.analysis import SOZAnalysisEngine
from engine.models import (
    ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, 
    OutputConfig, SOZDefinition, SOZNode, HbondHydrationConfig,
    SelectionSpec
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Baseline metrics (from generate_baseline.py)
BASELINE = {
    "mean_n_solvent": 21.3,
    "hydration_count": 2,
    "top_hydration_residue_freq": 0.0
}

@pytest.fixture
def laao_project(tmp_path):
    data_dir = Path("/home/aleff/laao_water")
    if not data_dir.exists():
        pytest.skip("LAAO dataset not found")
        
    topo = data_dir / "step3_input.gro"
    traj = data_dir / "step7_production_mol_rect_3_system_10ns.xtc"
    
    inputs = InputConfig(topology=str(topo), trajectory=str(traj))
    solvent = SolventConfig(
        solvent_label="Water",
        water_resnames=["TIP3", "HOH", "SOL"],
        include_ions=True
    )
    selections = {
        "protein": SelectionSpec(label="protein", selection="protein"),
        "active_site": SelectionSpec(label="active_site", selection="resname FAD"),
    }
    soz_root = SOZNode(
        type="shell", 
        params={"selection_label": "active_site", "distance": 5.0, "unit": "A"}
    )
    sozs = [SOZDefinition(name="ActiveSiteShell", description="5A shell around FAD", root=soz_root)]
    
    hydration = HbondHydrationConfig(
        name="FAD_hydration",
        residue_selection="resname FAD",
        distance=3.5,
        angle=150.0,
        conditioning="soz",
        soz_name="ActiveSiteShell"
    )
    
    analysis_opt = AnalysisOptions(
        frame_start=0, frame_stop=10, stride=1, workers=1
    )
    
    return ProjectConfig(
        inputs=inputs,
        solvent=solvent,
        selections=selections,
        sozs=sozs,
        hbond_hydration=[hydration],
        analysis=analysis_opt,
        outputs=OutputConfig(output_dir=str(tmp_path / "out"))
    )

def test_laao_regression(laao_project):
    """Regression test using LAAO dataset."""
    engine = SOZAnalysisEngine(laao_project)
    result = engine.run()
    
    soz_summary = result.soz_results["ActiveSiteShell"].summary
    mean_val = float(soz_summary.get("mean_n_solvent", 0.0))
    
    # Check mean solvent count (allow small floating point diff)
    assert np.isclose(mean_val, BASELINE["mean_n_solvent"], atol=0.1), \
        f"Mean solvent count changed: {mean_val} != {BASELINE['mean_n_solvent']}"
        
    # Check hydration count
    hyd_table = result.hbond_hydration_results["FAD_hydration"].table
    assert len(hyd_table) == BASELINE["hydration_count"], \
        f"Hydration table length changed: {len(hyd_table)} != {BASELINE['hydration_count']}"
        
    # Check freq (if table not empty)
    if not hyd_table.empty:
        top_freq = float(hyd_table.iloc[0]["freq_given_soz"])
        assert np.isclose(top_freq, BASELINE["top_hydration_residue_freq"], atol=0.01), \
            f"Top freq changed: {top_freq} != {BASELINE['top_hydration_residue_freq']}"
