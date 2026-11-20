"""Solute-frame density engine: known-answer physics, periodic images, rotation removal, caching, parallel identity."""
from __future__ import annotations

import json

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from _helpers import make_project
from engine import synthetic as syn
from engine.analysis import SOZAnalysisEngine
from engine.density import bin_counts, map_species, unwrap_chain
from engine.models import AnalysisOptions, DensityMapConfig
from engine.periodic import box_from_dimensions


def _cfg(**kw):
    base = dict(name="w", species_selection="name OW", grid_spacing=2.0, align=True, align_selection="resname LIG",
                conditioning_policy="strict", n_blocks=5)
    base.update(kw)
    return DensityMapConfig(**base)


def _run(system, tmp_path, cfg, analysis=None, stem="t", out="out", **kw):
    gro, xtc = system.write(tmp_path, stem)
    project = make_project(gro, xtc, tmp_path / out, density=[cfg], analysis=analysis or AnalysisOptions(
        store_min_distances=False, workers=1))
    return SOZAnalysisEngine(project).run(**kw), project, (gro, xtc)


class TestPrimitives:
    def test_bin_counts_matches_numpy_histogram_and_drops_outside(self):
        rng = np.random.default_rng(0)
        shape, spacing, origin = (5, 6, 7), 0.8, np.array([-1.0, 0.5, 2.0])
        pts = rng.uniform([-1.5, 0.0, 1.5], [3.5, 6.0, 8.5], size=(500, 3))
        flat, n_in = bin_counts(pts, origin, spacing, shape)
        edges = [origin[i] + spacing * np.arange(shape[i] + 1) for i in range(3)]
        ref, _ = np.histogramdd(pts, bins=edges)
        np.testing.assert_array_equal(flat.reshape(shape), ref.astype(np.int64))
        assert n_in == int(ref.sum()) and n_in < len(pts)

    def test_unwrap_chain_makes_a_split_chain_whole(self):
        box = box_from_dimensions([20, 20, 20, 90, 90, 90])
        whole = np.array([[18.0, 5, 5], [19.0, 5, 5], [20.5, 5, 5], [21.5, 5, 5]])
        wrapped = whole.copy(); wrapped[:, 0] %= 20.0
        un = unwrap_chain(wrapped, box)
        np.testing.assert_allclose(np.diff(un, axis=0), np.diff(whole, axis=0), atol=1e-9)

    def test_map_species_wraps_to_the_image_nearest_the_solute(self):
        box = box_from_dimensions([20, 20, 20, 90, 90, 90])
        out = map_species(np.array([[19.0, 10, 10]]), box, np.array([1.0, 10, 10]), np.eye(3), np.array([1.0, 10, 10]))
        assert out[0, 0] == pytest.approx(-1.0)  # 19.0 is 2 A "below" x=1 through the boundary


class TestIdealGas:
    def test_uniform_density_poisson_counts_and_bulk_normalisation(self, tmp_path):
        N, L = 1000, 30.0
        system = syn.make_ideal_gas_system(n_waters=N, n_frames=300, box=L, seed=21, n_solute_atoms=4)
        res, _, _ = _run(system, tmp_path, _cfg(grid_size=[24, 24, 24]))
        d = res.density_results["w"]
        rho = N / L**3
        assert d.grid.shape == (12, 12, 12)
        assert d.grid.mean() == pytest.approx(rho, rel=0.015)
        counts = d.counts.astype(float)
        assert counts.var(ddof=1) / counts.mean() == pytest.approx(1.0, abs=0.12)  # Poisson dispersion
        b = d.metadata["bulk"]
        assert b["method"] == "far_field" and b["box_mean"] == pytest.approx(rho, rel=1e-6)
        assert b["far_vs_box_ratio"] == pytest.approx(1.0, abs=0.03)
        assert d.g.mean() == pytest.approx(1.0, abs=0.03)
        # standard error from 5 blocks vs the Poisson expectation sqrt(mean count)/(n * V_voxel)
        poisson_se = np.sqrt(counts.mean()) / (300 * 2.0**3)
        assert d.se.mean() / poisson_se == pytest.approx(1.0, abs=0.3)
        assert d.metadata["is_scientifically_safe"] and d.metadata["n_frames"] == 300
        assert d.reliable[6, 6, 6] and not d.reliable[0, 0, 0]  # corner is beyond the fully covered ball

    def test_density_next_to_the_box_face_is_not_depleted(self, tmp_path):
        """Solute 2.5 A from the box corner: raw binning loses the half of the grid that lies outside the cell."""
        N, L = 1000, 30.0
        pos = (2.5, 2.5, 2.5)
        system = syn.make_ideal_gas_system(n_waters=N, n_frames=250, box=L, seed=22, n_solute_atoms=4, solute_position=pos)
        res, _, (gro, xtc) = _run(system, tmp_path, _cfg(grid_size=[16, 16, 16]))
        d = res.density_results["w"]
        rho = N / L**3
        x = d.axes["x"]
        outside_half = d.grid[x < 1.5].mean()  # voxels whose raw coordinates are negative (other side of the cell)
        assert outside_half == pytest.approx(rho, rel=0.04)
        assert d.grid.mean() == pytest.approx(rho, rel=0.03)
        # what naive raw-coordinate binning on the same grid would report
        u = mda.Universe(str(gro), str(xtc))
        edges = [np.asarray(pos[i] - 8 + 2.0 * np.arange(9)) for i in range(3)]
        raw = np.zeros((8, 8, 8))
        for ts in u.trajectory:
            raw += np.histogramdd(u.select_atoms("name OW").positions, bins=edges)[0]
        raw = raw / (250 * 8.0)
        assert raw[:3].mean() < 0.55 * rho  # the half beyond the face is almost empty

    def test_user_bulk_density_overrides_estimates(self, tmp_path):
        system = syn.make_ideal_gas_system(n_waters=400, n_frames=60, box=24.0, seed=5, n_solute_atoms=4)
        res, _, _ = _run(system, tmp_path, _cfg(grid_size=[16, 16, 16], bulk_density=0.0334, bulk_method="user"))
        d = res.density_results["w"]
        assert d.metadata["bulk"]["method"] == "user" and d.metadata["rho_bulk"] == 0.0334
        assert d.g.mean() == pytest.approx(d.grid.mean() / 0.0334)


class TestRotationRemoval:
    def test_bound_water_is_a_delta_peak_only_after_superposition(self, tmp_path):
        system = syn.make_rotating_solute_system(n_frames=120, n_waters=300, box=40.0, bound_distance=3.4, seed=1)
        center = np.array(system.box) / 2
        cfg = _cfg(grid_spacing=1.0, grid_center=(center + 0.5).tolist(), grid_size=[12, 12, 12], bulk_method="box")
        res, _, _ = _run(system, tmp_path, cfg)
        d = res.density_results["w"]
        # voxel index of centre+(3.4,0,0): origin edge = centre+0.5-6
        idx = tuple(np.floor((np.array([3.4, 0.0, 0.0]) - 0.5 + 6.0)).astype(int))
        assert d.counts[idx] >= 0.95 * 120
        assert d.metadata["solute_rotation_max_deg"] > 90  # the solute really tumbles
        assert d.metadata["is_scientifically_safe"]
        # without superposition the same water smears over a sphere of radius 3.4 A
        cfg2 = _cfg(name="w2", grid_spacing=1.0, grid_center=(center + 0.5).tolist(), grid_size=[12, 12, 12],
                    align=False, conditioning_policy="unsafe", bulk_method="box")
        res2, _, _ = _run(system, tmp_path, cfg2, out="out2")
        d2 = res2.density_results["w2"]
        assert d2.counts[idx] <= 0.10 * 120
        assert not d2.metadata["is_scientifically_safe"]

    def test_drifting_tumbling_solute_still_gives_a_sharp_peak(self, tmp_path):
        system = syn.make_rotating_solute_system(n_frames=100, n_waters=200, box=40.0, bound_distance=3.4, seed=2, drift=0.4)
        c_ref = system.positions[0, :4].astype(np.float64).mean(axis=0)  # first-frame solute centre (equal masses)
        cfg = _cfg(grid_spacing=1.0, grid_center=(c_ref + 0.5).tolist(), grid_size=[12, 12, 12], bulk_method="box")
        res, _, _ = _run(system, tmp_path, cfg)
        d = res.density_results["w"]
        idx = tuple(np.floor((np.array([3.4, 0.0, 0.0]) - 0.5 + 6.0)).astype(int))
        assert d.counts[idx] >= 0.95 * 100
        assert d.metadata["solute_com_drift_max"] > 1.0  # drift was real and removed


class TestWindowCacheAndParallel:
    def test_density_inherits_the_analysis_window(self, tmp_path):
        system = syn.make_ideal_gas_system(n_waters=200, n_frames=40, box=24.0, seed=3, n_solute_atoms=4)
        res, _, _ = _run(system, tmp_path, _cfg(grid_size=[12, 12, 12]), analysis=AnalysisOptions(
            stride=5, frame_start=2, store_min_distances=False, workers=1))
        m = res.density_results["w"].metadata
        assert (m["stride"], m["frame_start"], m["n_frames"]) == (5, 2, len(range(2, 40, 5)))
        assert res.soz_results["Z"].summary["n_frames"] == m["n_frames"]  # same frames as the SOZ analysis

    def test_cache_hits_only_when_content_and_configuration_are_identical(self, tmp_path):
        a = syn.make_ideal_gas_system(n_waters=300, n_frames=30, box=24.0, seed=1, n_solute_atoms=4)
        b = syn.make_ideal_gas_system(n_waters=300, n_frames=30, box=24.0, seed=2, n_solute_atoms=4)
        cfg = _cfg(grid_size=[12, 12, 12])
        r1, project, (gro, xtc) = _run(a, tmp_path, cfg)
        assert r1.density_results["w"].metadata["cache"] == {"hit": False, "reason": "no cache"}
        g1 = r1.density_results["w"].grid.copy()

        r2 = SOZAnalysisEngine(project).run()  # nothing changed
        assert r2.density_results["w"].metadata["cache"]["hit"] is True
        np.testing.assert_array_equal(r2.density_results["w"].grid, g1)

        b.write(tmp_path, "t")  # SAME PATH, different content (the v2.0 stale-cache scenario)
        r3 = SOZAnalysisEngine(project).run()
        m3 = r3.density_results["w"].metadata["cache"]
        assert m3["hit"] is False and "changed" in m3["reason"]
        assert np.abs(r3.density_results["w"].grid - g1).max() > 0
        fresh, _, _ = _run(b, tmp_path, cfg, stem="t", out="fresh")
        np.testing.assert_array_equal(r3.density_results["w"].grid, fresh.density_results["w"].grid)  # not stale

        project.density_maps[0].grid_spacing = 3.0  # configuration change invalidates too
        r4 = SOZAnalysisEngine(project).run()
        assert r4.density_results["w"].metadata["cache"]["hit"] is False
        project.density_maps[0].grid_spacing = 2.0
        assert SOZAnalysisEngine(project).run(use_cache=False).density_results["w"].metadata["cache"]["hit"] is False

    def test_tampered_cache_is_detected(self, tmp_path):
        system = syn.make_ideal_gas_system(n_waters=200, n_frames=20, box=24.0, seed=1, n_solute_atoms=4)
        r1, project, _ = _run(system, tmp_path, _cfg(grid_size=[12, 12, 12]))
        npz = tmp_path / "out" / "density_map_w" / "grid.npz"
        data = dict(np.load(npz))
        data["grid"] = data["grid"] * 2.0
        np.savez(npz, **data)
        r2 = SOZAnalysisEngine(project).run()
        assert r2.density_results["w"].metadata["cache"]["hit"] is False
        assert "checksum" in r2.density_results["w"].metadata["cache"]["reason"]
        np.testing.assert_array_equal(r2.density_results["w"].grid, r1.density_results["w"].grid)

    def test_parallel_density_and_zones_are_bitwise_identical_to_serial(self, tmp_path):
        system = syn.make_ideal_gas_system(n_waters=300, n_frames=260, box=26.0, seed=4, n_solute_atoms=4)
        cfg = _cfg(grid_size=[16, 16, 16])
        gro, xtc = system.write(tmp_path, "t")
        out = {}
        for workers in (1, 3):
            opts = AnalysisOptions(workers=workers, parallel_min_seconds=0.0, store_min_distances=True)
            project = make_project(gro, xtc, tmp_path / f"o{workers}", density=[cfg], analysis=opts)
            out[workers] = SOZAnalysisEngine(project).run(use_cache=False)
        assert out[1].qc_summary["parallel"]["mode"] == "serial"
        assert out[3].qc_summary["parallel"]["mode"] == "parallel"
        d1, d3 = out[1].density_results["w"], out[3].density_results["w"]
        np.testing.assert_array_equal(d1.counts, d3.counts)
        np.testing.assert_array_equal(d1.grid, d3.grid)
        np.testing.assert_array_equal(d1.se, d3.se)
        assert d3.metadata["mode"] == "parallel"
        z1, z3 = out[1].soz_results["Z"], out[3].soz_results["Z"]
        import pandas as pd
        pd.testing.assert_frame_equal(z1.per_frame, z3.per_frame, check_exact=True)
        pd.testing.assert_frame_equal(z1.per_solvent, z3.per_solvent, check_exact=True)
        pd.testing.assert_frame_equal(z1.visits, z3.visits, check_exact=True)
        pd.testing.assert_frame_equal(z1.survival, z3.survival, check_exact=True)
        assert json.dumps(z1.summary, default=str, sort_keys=True) == json.dumps(z3.summary, default=str, sort_keys=True)
