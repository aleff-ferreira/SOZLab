import pytest

pytest.importorskip("MDAnalysis")
import numpy as np
import pandas as pd
import MDAnalysis as mda

from engine.extract import select_frames, write_extracted_trajectory


def test_select_frames_min_run_and_gap():
    per_frame = pd.DataFrame(
        {
            "frame": list(range(8)),
            "time": [i * 10.0 for i in range(8)],
            "n_solvent": [0, 1, 1, 0, 1, 1, 1, 0],
        }
    )
    selection = select_frames(per_frame, rule="n_solvent>=1", min_run_length=2, gap_tolerance=0)
    assert selection.frame_indices == [1, 2, 4, 5, 6]

    selection_gap = select_frames(per_frame, rule="n_solvent>=1", min_run_length=3, gap_tolerance=1)
    assert selection_gap.frame_indices == [1, 2, 3, 4, 5, 6]


def test_write_extracted_trajectory(tmp_path):
    n_atoms = 3
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [2.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [1.3, 0.0, 0.0], [2.3, 0.0, 0.0]],
        ]
    )
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("name", ["A", "B", "C"])
    u.load_new(coords, order="fac")
    selection = select_frames(
        pd.DataFrame(
            {"frame": [0, 1, 2, 3], "time": [0.0, 1.0, 2.0, 3.0], "n_solvent": [0, 1, 1, 0]}
        ),
        rule="n_solvent>=1",
    )
    outputs = write_extracted_trajectory(u, selection, output_dir=str(tmp_path), prefix="test", fmt="xtc")
    assert "trajectory" in outputs
    xtc_path = outputs["trajectory"]

    # Reload and verify frame count
    ref_path = outputs["reference"]
    u2 = mda.Universe(ref_path, xtc_path)
    assert len(u2.trajectory) == len(selection.frame_indices)
