
import pytest
import numpy as np
import MDAnalysis as mda
from engine.solvent import solvent_positions, SolventUniverse, SolventProbe, _normalize_indices

def test_solvent_pbc_unwrapping():
    """
    Verify that solvent positions (COG/COM) are correctly unwrapped 
    when residues cross PBC boundaries.
    """
    # 1. Setup Universe with 10x10x10 box
    u = mda.Universe.empty(n_atoms=2, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("resnames", ["TEST"])
    u.add_TopologyAttr("names", ["A", "B"])
    u.add_TopologyAttr("bonds", [[0, 1]])
    u.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    
    # 2. Place atoms across boundary
    # A at x=0.1, B at x=9.9 (wrapped). 
    # Real distance is 0.2. Center should be at x=0.0 (or 10.0).
    # If not unwrapped, Center is at (0.1 + 9.9)/2 = 5.0 (Wrong!)
    u.atoms.positions = [[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]]
    
    # 3. Construct Minimal SolventUniverse manually (mocking build_solvent complexity)
    # We cheat and just build the objects needed for solvent_positions
    residues = u.residues
    atoms_all = u.atoms
    
    # Mock Probe: usually created by build_solvent
    # We use ALL atoms as probe for COG
    probe = SolventProbe(
        selection="all",
        position="cog",
        atoms=u.atoms,
        atom_indices=[0, 1],
        atom_to_resindex=[0, 0],
        resindex_to_atom_indices={0: [0, 1]}
    )
    
    solvent = SolventUniverse(
        residues=residues,
        atoms_all=atoms_all,
        probe=probe,
        record_by_resindex={}, # Not needed for this test
        solvent_resindices=[0],
        atom_to_resindex_all=[0, 0],
        n_atoms=2,
        atom_index_shift=0
    )
    
    # 4. Calculate positions (COG)
    # This calls solvent_positions -> center_of_geometry
    positions, _ = solvent_positions(solvent, mode="cog")
    
    assert len(positions) == 1
    com = positions[0]
    
    # 5. Assertions
    # If unwrapping works, x should be close to 0 or 10.
    # If fails, x will be 5.
    x = com[0]
    distance_from_edge = min(abs(x - 0.0), abs(x - 10.0))
    distance_from_center = abs(x - 5.0)
    
    print(f"Calculated COG x: {x}")
    
    if distance_from_center < 1.0:
        pytest.fail(f"COG calculated at center ({x}), implying PBC unwrapping failed! Should be at edge.")
    
    assert distance_from_edge < 0.2, f"COG {x} is not near edge (0/10)!"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
