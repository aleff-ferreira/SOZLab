"""Dynamic (geometric) species selections are re-evaluated every frame."""
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import density as density_module

from engine.analysis import _run_density_maps
from engine.models import AnalysisOptions, DensityMapConfig


def _dynamic_selection_universe() -> mda.Universe:
    u = mda.Universe.empty(n_atoms=2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "OW"])
    u.add_TopologyAttr("resname", ["PRO", "TIP3"])
    u.add_TopologyAttr("mass", [12.0, 16.0])
    # Frame 0: OW is close to CA and selected by "around". Frame 1: OW moved outside the shell.
    coords = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=float)
    u.load_new(coords, order="fac")
    return u


def test_legacy_engine_dynamic_selection_matches_updating_atomgroup(tmp_path):
    u = _dynamic_selection_universe()
    selection = "name OW and around 1.0 name CA"
    cfg = DensityMapConfig(name="dynamic_density", species_selection=selection, grid_spacing=0.5, padding=0.5,
                           align=False, conditioning_policy="unsafe", engine="mdanalysis")
    got = _run_density_maps(u, [cfg], AnalysisOptions(frame_start=0, frame_stop=2, stride=1, workers=1),
                            str(tmp_path), None, None)["dynamic_density"]
    u.trajectory[0]
    ref_group = u.select_atoms(selection, updating=True)
    ref_analysis = density_module.DensityAnalysis(ref_group, delta=0.5, padding=0.5)
    ref_analysis.run(start=0, stop=2, step=1)
    ref_grid = np.asarray(ref_analysis.results.density.grid, dtype=float)
    assert got.metadata.get("selection_is_dynamic") is True
    assert got.grid.shape == ref_grid.shape
    assert np.max(np.abs(np.asarray(got.grid) - ref_grid)) < 1e-6


def test_solute_engine_counts_only_frames_where_the_selection_holds(tmp_path):
    """OW is in the 'around' shell on frame 0 only: it must be binned once, i.e. density = 1/(2 frames * V)."""
    u = _dynamic_selection_universe()
    delta = 0.5
    cfg = DensityMapConfig(name="d", species_selection="name OW and around 1.0 name CA", grid_spacing=delta,
                           align=False, conditioning_policy="unsafe", grid_center=[0.5, 0.0, 0.0], grid_size=[3, 3, 3])
    got = _run_density_maps(u, [cfg], AnalysisOptions(frame_start=0, frame_stop=2, stride=1, workers=1),
                            str(tmp_path), None, None)["d"]
    assert got.metadata["selection_is_dynamic"] is True
    total = got.grid.sum() * delta ** 3  # integral of density = mean number of selected atoms per frame
    assert total == pytest.approx(0.5)  # 1 atom on 1 of 2 frames
    assert got.counts.sum() == 1

    # cross-check against MDAnalysis with the same user grid and an updating group
    u.trajectory[0]
    grp = u.select_atoms("name OW and around 1.0 name CA", updating=True)
    ref = density_module.DensityAnalysis(grp, delta=delta, gridcenter=np.array([0.5, 0.0, 0.0]), xdim=3, ydim=3, zdim=3)
    ref.run(start=0, stop=2, step=1)
    assert np.asarray(ref.results.density.grid).sum() * delta ** 3 == pytest.approx(total)
