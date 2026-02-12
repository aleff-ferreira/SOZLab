
import pytest
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import density
from engine.analysis import _run_density_maps
from engine.models import DensityMapConfig, AnalysisOptions, ProjectConfig, InputConfig, OutputConfig

def test_density_selection_compliance(tmp_path):
    """
    Verify that SOZLab's density map calculation respects the 'species_selection'
    and matches raw MDAnalysis output.
    """
    # 1. Setup minimal universe with two "species" (Protein vs Water)
    # We create a universe with 2 atoms: 1 'CA' (protein) at origin, 1 'OW' (water) moving
    u = mda.Universe.empty(n_atoms=2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
    u.add_TopologyAttr("resnames", ["PROT", "SOL"])
    u.add_TopologyAttr("names", ["CA", "OW"])
    
    # Frame 0: Water at (2,0,0)
    u.atoms.positions = [[0,0,0], [2,0,0]]
    
    # 2. Run raw MDAnalysis DensityAnalysis on JUST WATER
    water_grp = u.select_atoms("resname SOL")
    ref_analysis = density.DensityAnalysis(water_grp, delta=1.0, padding=2.0)
    ref_analysis.run()
    ref_grid = ref_analysis.results.density.grid
    
    # 3. Reference: Density on ALL atoms (what the bugs produces)
    bug_analysis = density.DensityAnalysis(u.atoms, delta=1.0, padding=2.0)
    bug_analysis.run()
    bug_grid = bug_analysis.results.density.grid
    
    # 4. Run SOZLab logic
    # We construct minimal configs
    cfg = DensityMapConfig(
        name="test_water",
        species_selection="resname SOL",
        grid_spacing=1.0,
        padding=2.0,
        align=False, # Skip alignment for this simple test
        conditioning_policy="unsafe" # Allow unaligned
    )
    
    options = AnalysisOptions(stride=1, workers=1)
    
    # Mocking output dir
    out_dir = str(tmp_path)
    
    results = _run_density_maps(
        universe=u,
        configs=[cfg],
        options=options,
        output_dir=out_dir,
        progress=None,
        logger=None
    )
    
    assert "test_water" in results
    soz_grid = results["test_water"].grid
    
    # 5. Assertions
    # Shape check
    assert soz_grid.shape == ref_grid.shape, f"Grid shapes mismatch: SOZ {soz_grid.shape} != Ref {ref_grid.shape}"
    
    # Value check - compare sum or max
    # Standard: SOZLab should match Reference (only water)
    diff = np.abs(soz_grid - ref_grid).max()
    assert diff < 1e-6, f"Mismatch with Reference! Max diff: {diff}"
    
    # Negative Control: Should NOT match Bug (all atoms)
    # The protein is at (0,0,0), water at (2,0,0).
    # If correct, density at (0,0,0) should be 0.
    # If bug, density at (0,0,0) > 0.
    
    # Note: grids might have different origins/shapes if padding is relative to selection?
    # MDAnalysis padding is relative to the bounds of the atomgroup coordinates over trajectory.
    # If we select 2 atoms vs 1 atom, the bounds differ.
    # So direct grid comparison requires identical grid definition.
    # But checking if they are 'close' numerically on value is valid if grids align?
    # Actually, DensityAnalysis grid depends on input atoms min/max.
    # If we select ALL, box is bigger or different.
    # So comparing grids directly implies shapes match.
    # If shapes differ, that ITSELF proves correctness (selection was respected).
    
    if soz_grid.shape != bug_grid.shape:
        # Shapes differ implies selection worked (probably)
        pass 
    else:
        # Shapes match (e.g. if padding dominates), check values
        diff_bug = np.abs(soz_grid - bug_grid).max()
        assert diff_bug > 1e-3, "SOZLab output matches 'All Atoms' bug behavior!"

if __name__ == "__main__":
    # Allow running directly
    import sys
    sys.exit(pytest.main(["-v", __file__]))
