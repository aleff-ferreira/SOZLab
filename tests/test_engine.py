import pytest
import MDAnalysis as mda
from engine.analysis import SOZAnalysisEngine
from engine.models import (
    ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, OutputConfig,
    SOZDefinition, SOZNode
)
from engine.solvent import build_solvent

@pytest.mark.skipif(not pytest.importorskip("MDAnalysis"), reason="MDAnalysis not installed")
def test_engine_initialization(mock_project_config):
    """Test that engine initializes with a valid project config."""
    engine = SOZAnalysisEngine(mock_project_config)
    assert engine.project == mock_project_config

def test_engine_run_laao_smoke(laao_data_dir, tmp_path):
    """Smoke test running the engine on a few frames of LAAO data if available."""
    if laao_data_dir is None:
        pytest.skip("LAAO data not found")
    
    pdb_path = str(laao_data_dir / "extracted_ref.pdb")
    xtc_path = str(laao_data_dir / "step7_production_mol_rect_3_system_10ns.xtc")
    
    # Minimal config
    soz_def = SOZDefinition(
        name="TestSOZ",
        description="Test SOZ",
        root=SOZNode(type="distance", params={"selection_label": "mysel", "cutoff": 4.0})
    )
    
    inputs = InputConfig(topology=pdb_path, trajectory=xtc_path)
    analysis = AnalysisOptions(stride=1000, frame_stop=1001) # Very short run
    output = OutputConfig(output_dir=str(tmp_path))
    
    from engine.models import SelectionSpec
    project = ProjectConfig(
        inputs=inputs,
        solvent=SolventConfig(water_resnames=["SOL", "HOH", "TIP3"]),
        selections={"mysel": SelectionSpec(label="mysel", selection="resid 50")},
        sozs=[soz_def],
        analysis=analysis,
        outputs=output,
        hbond_hydration=[]
    )
    
    engine = SOZAnalysisEngine(project)
    
    # We expect this to run without error
    result = engine.run()
    
    assert "TestSOZ" in result.soz_results
    assert len(result.soz_results["TestSOZ"].per_frame) > 0
