import logging

import numpy as np
import pytest
import MDAnalysis as mda

from engine.analysis import _run_density_maps
from engine.models import AnalysisOptions, DensityMapConfig


@pytest.fixture
def mock_universe(tmp_path):
    # Small universe with water oxygens only
    rng = np.random.default_rng(0)
    u = mda.Universe.empty(n_atoms=10, n_residues=10, atom_resindex=np.arange(10), trajectory=True)
    u.add_TopologyAttr("resname", ["SOL"] * 10)
    u.add_TopologyAttr("name", ["OW"] * 10)
    u.add_TopologyAttr("mass", [15.999] * 10)
    u.atoms.positions = rng.random((10, 3)) * 10
    u.dimensions = [10, 10, 10, 90, 90, 90]
    return u


@pytest.fixture
def analysis_options():
    return AnalysisOptions(stride=1)


def test_density_strict_no_align(mock_universe, analysis_options, caplog, tmp_path):
    """Strict policy blocks unaligned density."""
    cfg = DensityMapConfig(name="test_strict", species_selection="name OW", align=False, conditioning_policy="strict")
    with caplog.at_level(logging.ERROR):
        results = _run_density_maps(mock_universe, [cfg], analysis_options, str(tmp_path), None, logging.getLogger())
    assert "test_strict" not in results
    assert "scientifically unsafe" in caplog.text


def test_density_warn_no_align(mock_universe, analysis_options, caplog, tmp_path):
    """Warn policy runs but warns."""
    cfg = DensityMapConfig(name="test_warn", species_selection="name OW", align=False, conditioning_policy="warn",
                           grid_center=[5, 5, 5], grid_size=[10, 10, 10])
    with caplog.at_level(logging.WARNING):
        results = _run_density_maps(mock_universe, [cfg], analysis_options, str(tmp_path), None, logging.getLogger())
    assert "test_warn" in results
    result = results["test_warn"]
    assert not result.metadata["is_scientifically_safe"]
    assert any("scientifically unsafe" in w for w in result.metadata["analysis_warnings"])


def test_density_unsafe_no_align_is_silent_but_never_reported_safe(mock_universe, analysis_options, tmp_path):
    """'unsafe' policy: the user accepted the risk (no warning text), but the flag stays honest."""
    cfg = DensityMapConfig(name="test_unsafe", species_selection="name OW", align=False, conditioning_policy="unsafe",
                           grid_center=[5, 5, 5], grid_size=[10, 10, 10])
    results = _run_density_maps(mock_universe, [cfg], analysis_options, str(tmp_path), None, logging.getLogger())
    meta = results["test_unsafe"].metadata
    assert meta["is_scientifically_safe"] is False
    assert not any("scientifically unsafe" in w for w in meta["analysis_warnings"])


def test_density_align_ok(mock_universe, analysis_options, tmp_path):
    """Alignment satisfies the safety check; the bulk reference is estimated, not a constant."""
    cfg = DensityMapConfig(name="test_align", species_selection="name OW", align=True, align_selection="name OW",
                           conditioning_policy="strict", region_padding=2.0)
    results = _run_density_maps(mock_universe, [cfg], analysis_options, str(tmp_path), None, logging.getLogger())
    meta = results["test_align"].metadata
    assert meta["is_scientifically_safe"]
    # 10 oxygens in a 10 A box -> <N>/<V> = 0.01 A^-3 (single frame: far-field voxels are unavailable)
    assert meta["bulk"]["method"] == "box"
    assert meta["rho_bulk"] == pytest.approx(0.01)
    assert meta["rho_bulk_approx"] == meta["rho_bulk"]  # legacy key kept for older consumers


def test_metadata_structure(mock_universe, analysis_options, tmp_path):
    cfg = DensityMapConfig(name="test_meta", species_selection="name OW", align=True, align_selection="name OW",
                           conditioning_policy="strict", region_padding=2.0)
    results = _run_density_maps(mock_universe, [cfg], analysis_options, str(tmp_path), None, None)
    meta = results["test_meta"].metadata
    for key in ("grid_spacing", "rho_bulk_approx", "is_scientifically_safe", "bulk", "n_frames", "fraction_of_species_inside_grid",
                "fully_covered_radius", "reliable_fraction", "cache"):
        assert key in meta
