import pytest

pytest.importorskip("MDAnalysis")
import numpy as np
import MDAnalysis as mda

from engine.analysis import _min_distance_to_soz
from engine.models import SolventConfig
from engine.solvent import build_solvent


def _make_universe():
    # Residue 0: seed (ALA), Residue 1: water (WAT with O + H)
    u = mda.Universe.empty(
        n_atoms=3,
        n_residues=2,
        atom_resindex=[0, 1, 1],
        residue_segindex=[0, 0],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA", "O", "H1"])
    u.add_TopologyAttr("resname", ["ALA", "WAT"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["A"])
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],  # seed CA
                [10.0, 0.0, 0.0],  # water O far
                [1.0, 0.0, 0.0],  # water H near
            ]
        ]
    )
    u.load_new(coords, order="fac")
    return u


def test_min_distance_oxygen_only():
    u = _make_universe()
    solvent = build_solvent(u, SolventConfig(water_resnames=["WAT"], water_oxygen_names=["O"]))
    seed = u.select_atoms("resid 1 and name CA")
    frame_set = {solvent.residues[0].resindex}

    # All-atoms distance should see H at 1.0 A.
    dist_all = _min_distance_to_soz(seed, solvent, frame_set, box=None, use_oxygen=False)
    # Oxygen-only distance should see O at 10.0 A.
    dist_oxy = _min_distance_to_soz(seed, solvent, frame_set, box=None, use_oxygen=True)

    assert np.isclose(dist_all, 1.0)
    assert np.isclose(dist_oxy, 10.0)
