
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import hydrogenbonds
from engine.analysis import _init_hbond_analysis

def test_hbond_angle_parameter_compliance():
    """
    Verify that _init_hbond_analysis correctly maps the 'angle' parameter
    to whatever the current MDAnalysis version requires (e.g. d_h_a_angle_cutoff).
    """
    # 1. create minimal universe
    u = mda.Universe.empty(n_atoms=10, n_residues=10, atom_resindex=range(10), trajectory=True)
    u.add_TopologyAttr("resnames", ["SOL"]*10)
    u.add_TopologyAttr("names", ["O", "H", "O", "H", "O", "H", "O", "H", "O", "H"])
    u.add_TopologyAttr("charges", [0.0]*10)
    u.add_TopologyAttr("masses", [16.0, 1.0, 16.0, 1.0, 16.0, 1.0, 16.0, 1.0, 16.0, 1.0])
    
    # 2. Call _init_hbond_analysis with a custom angle
    target_angle = 120.0
    hba = _init_hbond_analysis(
        universe=u,
        selection_a="name O",
        selection_b="name H",
        distance=3.0,
        angle=target_angle,
        donors_selection="name O",
        hydrogens_selection="name H",
        acceptors_selection="name O",
        update_selections=True,
        pbc=True
    )
    
    # 3. Inspect the resulting object to see if the angle was actually set
    # MDAnalysis stores this in different places depending on version/class structure
    # Common ones: 'angle', 'd_h_a_angle', 'd_h_a_angle_cutoff'
    
    found_angle = None
    if hasattr(hba, "angle"):
        found_angle = hba.angle
    elif hasattr(hba, "d_h_a_angle"):
        found_angle = hba.d_h_a_angle
    elif hasattr(hba, "d_h_a_angle_cutoff"):
        found_angle = hba.d_h_a_angle_cutoff
    elif hasattr(hba, "angle_cutoff"):
         found_angle = hba.angle_cutoff
         
    assert found_angle is not None, "Could not locate angle attribute on HBA object"
    assert found_angle == target_angle, f"Angle mismatch! Expected {target_angle}, got {found_angle}. (Default is often 150)"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
