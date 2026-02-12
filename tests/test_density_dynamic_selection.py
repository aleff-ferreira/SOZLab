import numpy as np
import pytest

pytest.importorskip("MDAnalysis")
import MDAnalysis as mda
from MDAnalysis.analysis import density as density_module

from engine.analysis import _run_density_maps
from engine.models import AnalysisOptions, DensityMapConfig


def _dynamic_selection_universe() -> mda.Universe:
    u = mda.Universe.empty(
        n_atoms=2,
        n_residues=2,
        atom_resindex=[0, 1],
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA", "OW"])
    u.add_TopologyAttr("resname", ["PRO", "TIP3"])

    # Frame 0: OW is close to CA and selected by "around".
    # Frame 1: OW moves outside the "around" shell and should be excluded.
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    u.load_new(coords, order="fac")
    return u


def test_density_dynamic_selection_matches_updating_atomgroup(tmp_path):
    u = _dynamic_selection_universe()
    selection = "name OW and around 1.0 name CA"

    cfg = DensityMapConfig(
        name="dynamic_density",
        species_selection=selection,
        grid_spacing=0.5,
        padding=0.5,
        align=False,
        conditioning_policy="unsafe",
    )
    opts = AnalysisOptions(frame_start=0, frame_stop=2, stride=1, workers=1)

    results = _run_density_maps(
        universe=u,
        configs=[cfg],
        options=opts,
        output_dir=str(tmp_path),
        progress=None,
        logger=None,
    )
    assert "dynamic_density" in results
    got = results["dynamic_density"]

    ref_group = u.select_atoms(selection, updating=True)
    ref_analysis = density_module.DensityAnalysis(ref_group, delta=0.5, padding=0.5)
    ref_analysis.run(start=0, stop=2, step=1)
    ref_grid = np.asarray(ref_analysis.results.density.grid, dtype=float)

    got_grid = np.asarray(got.grid, dtype=float)
    assert got.metadata.get("selection_is_dynamic") is True
    assert got_grid.shape == ref_grid.shape
    assert np.max(np.abs(got_grid - ref_grid)) < 1e-6

