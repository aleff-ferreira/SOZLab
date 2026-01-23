import pytest

pytest.importorskip("MDAnalysis")
import numpy as np
import MDAnalysis as mda
import pandas as pd

from engine.models import ProjectConfig
from engine.solvent import build_solvent
from engine.resolver import resolve_selection
from engine.soz_eval import EvaluationContext, evaluate_node
from engine.stats import StatsAccumulator
from engine.extract import select_frames


def _make_universe():
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


def _project_from_dict(data: dict) -> ProjectConfig:
    return ProjectConfig.from_dict(data)


def _compute_stats(project: ProjectConfig, universe: mda.Universe):
    solvent = build_solvent(universe, project.solvent)
    selections = {
        label: resolve_selection(universe, spec) for label, spec in project.selections.items()
    }
    context = EvaluationContext(universe=universe, solvent=solvent, selections=selections)
    acc = StatsAccumulator(
        solvent_records=solvent.record_by_resindex,
        gap_tolerance=project.analysis.gap_tolerance,
        frame_stride=project.analysis.stride,
        store_ids=True,
        store_frame_table=True,
    )
    for idx, ts in enumerate(universe.trajectory):
        frame_set = evaluate_node(project.sozs[0].root, context)
        acc.update(idx, float(ts.time), frame_set, frame_label=idx)
    stats = acc.finalize(time_unit="ps")
    return stats["per_frame"], stats["summary"]


def test_seed_vs_selection_schema_regression():
    u = _make_universe()

    old_schema = {
        "version": "1.0",
        "inputs": {"topology": "dummy"},
        "solvent": {"water_resnames": ["SOL"], "water_oxygen_names": ["O"]},
        "seeds": {
            "seed_a": {
                "label": "seed_a",
                "selection": "resid 1 and name CA",
                "require_unique": True,
            }
        },
        "sozs": [
            {
                "name": "shell_only",
                "description": "",
                "root": {
                    "type": "distance",
                    "params": {"seed_label": "seed_a", "cutoff": 3.5, "unit": "A"},
                    "children": [],
                },
            }
        ],
        "analysis": {"stride": 1, "gap_tolerance": 0},
        "outputs": {"output_dir": "results", "write_per_frame": True},
    }

    new_schema = {
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
                "name": "shell_only",
                "description": "",
                "root": {
                    "type": "distance",
                    "params": {"selection_label": "selection_a", "cutoff": 3.5, "unit": "A"},
                    "children": [],
                },
            }
        ],
        "analysis": {"stride": 1, "gap_tolerance": 0},
        "outputs": {"output_dir": "results", "write_per_frame": True},
    }

    project_old = _project_from_dict(old_schema)
    project_new = _project_from_dict(new_schema)

    per_frame_old, summary_old = _compute_stats(project_old, u)
    per_frame_new, summary_new = _compute_stats(project_new, u)

    pd.testing.assert_frame_equal(per_frame_old, per_frame_new)

    for key in (
        "mean_n_solvent",
        "median_n_solvent",
        "max_n_solvent",
        "occupancy_fraction",
        "entry_rate",
        "exit_rate",
    ):
        assert summary_old[key] == summary_new[key]

    selection_old = select_frames(per_frame_old, rule="n_solvent>=1", min_run_length=1)
    selection_new = select_frames(per_frame_new, rule="n_solvent>=1", min_run_length=1)
    assert selection_old.frame_indices == selection_new.frame_indices
