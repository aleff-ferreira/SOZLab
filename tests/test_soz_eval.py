import numpy as np
import pytest

pytest.importorskip("MDAnalysis")
import MDAnalysis as mda

from engine.models import SelectionSpec, SolventConfig, SOZNode
from engine.resolver import resolve_selection
from engine.solvent import build_solvent
from engine.soz_eval import EvaluationContext, evaluate_node


def _make_universe():
    n_atoms = 8
    n_residues = 3
    atom_resindex = [0, 0, 1, 1, 1, 2, 2, 2]
    residue_segindex = [0, 0, 0]
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA", "CB", "O", "H1", "H2", "O", "H1", "H2"])
    u.add_TopologyAttr("resname", ["ALA", "SOL", "SOL"])
    u.add_TopologyAttr("resid", [1, 2, 3])
    u.add_TopologyAttr("segid", ["A"])

    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [3.2, 0.8, 0.0],
                [3.2, -0.8, 0.0],
                [7.0, 0.0, 0.0],
                [7.4, 0.8, 0.0],
                [7.4, -0.8, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.2, 0.0, 0.0],
                [3.6, 0.8, 0.0],
                [3.6, -0.8, 0.0],
                [3.4, 0.0, 0.0],
                [3.8, 0.8, 0.0],
                [3.8, -0.8, 0.0],
            ],
        ]
    )
    u.load_new(coords, order="fac")
    return u


def test_distance_node():
    u = _make_universe()
    solvent = build_solvent(u, SolventConfig())
    sel = resolve_selection(u, SelectionSpec(label="selection_a", selection="resid 1 and name CA"))
    context = EvaluationContext(u, solvent, {"selection_a": sel})

    node = SOZNode(
        type="distance",
        params={"selection_label": "selection_a", "cutoff": 3.5, "unit": "A", "atom_mode": "O"},
    )

    u.trajectory[0]
    result0 = evaluate_node(node, context)
    assert 1 in result0
    assert 2 not in result0

    u.trajectory[1]
    result1 = evaluate_node(node, context)
    assert 1 in result1
    assert 2 in result1
