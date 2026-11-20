"""Density maps agree with MDAnalysis where both are well defined, and respect the selection."""
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import density

from engine.analysis import _run_density_maps
from engine.models import AnalysisOptions, DensityMapConfig


def _two_species_universe():
    u = mda.Universe.empty(n_atoms=2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
    u.add_TopologyAttr("resnames", ["PROT", "SOL"])
    u.add_TopologyAttr("names", ["CA", "OW"])
    u.atoms.positions = [[0, 0, 0], [2, 0, 0]]
    return u


def test_legacy_engine_respects_species_selection_and_matches_mdanalysis(tmp_path):
    """engine='mdanalysis' reproduces raw DensityAnalysis on the selected species only."""
    u = _two_species_universe()
    water = u.select_atoms("resname SOL")
    ref = density.DensityAnalysis(water, delta=1.0, padding=2.0)
    ref.run()
    ref_grid = ref.results.density.grid

    cfg = DensityMapConfig(name="test_water", species_selection="resname SOL", grid_spacing=1.0, padding=2.0,
                           align=False, conditioning_policy="unsafe", engine="mdanalysis")
    res = _run_density_maps(u, [cfg], AnalysisOptions(stride=1, workers=1), str(tmp_path), None, None)["test_water"]
    assert res.grid.shape == ref_grid.shape
    assert np.abs(res.grid - ref_grid).max() < 1e-6

    bug = density.DensityAnalysis(u.atoms, delta=1.0, padding=2.0)
    bug.run()
    if res.grid.shape == bug.results.density.grid.shape:
        assert np.abs(res.grid - bug.results.density.grid).max() > 1e-3, "selection ignored (all atoms binned)"


def test_solute_engine_matches_mdanalysis_on_an_identical_grid(tmp_path):
    """Established-tool cross-check: same grid, same (already wrapped) coordinates -> identical map."""
    rng = np.random.default_rng(3)
    n_w, n_f, box = 60, 25, 12.0
    u = mda.Universe.empty(n_w + 1, n_residues=n_w + 1, atom_resindex=np.arange(n_w + 1), trajectory=True)
    u.add_TopologyAttr("name", ["CA"] + ["OW"] * n_w)
    u.add_TopologyAttr("resname", ["PRO"] + ["SOL"] * n_w)
    u.add_TopologyAttr("mass", [12.0] + [16.0] * n_w)
    coords = rng.uniform(0, box, size=(n_f, n_w + 1, 3)).astype(np.float32)
    coords[:, 0, :] = box / 2  # fixed solute at the box centre: no alignment needed
    u.load_new(coords, order="fac", dimensions=np.tile([box, box, box, 90, 90, 90], (n_f, 1)))

    delta = 1.0
    mda_density = density.DensityAnalysis(u.select_atoms("name OW"), delta=delta, gridcenter=np.array([box / 2] * 3),
                                          xdim=box, ydim=box, zdim=box)
    mda_density.run()
    ref = np.asarray(mda_density.results.density.grid)

    cfg = DensityMapConfig(name="x", species_selection="name OW", grid_spacing=delta, align=False,
                           conditioning_policy="unsafe", grid_center=[box / 2] * 3, grid_size=[box] * 3, n_blocks=5)
    res = _run_density_maps(u, [cfg], AnalysisOptions(stride=1, workers=1), str(tmp_path), None, None)["x"]
    assert res.grid.shape == ref.shape == (12, 12, 12)
    np.testing.assert_allclose(res.grid, ref, rtol=0, atol=1e-12)
