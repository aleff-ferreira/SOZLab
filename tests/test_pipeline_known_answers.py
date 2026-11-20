"""End-to-end engine runs on synthetic systems whose correct answers are known analytically."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

mda = pytest.importorskip("MDAnalysis")

from _helpers import make_project, oracle_n_solvent
from engine import synthetic as syn
from engine.analysis import SOZAnalysisEngine
from engine.export import PartialResultError, export_results, verify_export
from engine.models import AnalysisOptions, SOZNode, SolventConfig


def _run(system, tmp_path, **kw):
    gro, xtc = system.write(tmp_path, "t")
    project = make_project(gro, xtc, tmp_path / "out", **kw)
    return SOZAnalysisEngine(project).run(), project


class TestOccupancyAndEvents:
    def test_scripted_occupancy_events_and_censoring(self, tmp_path):
        T = 16
        pres = np.zeros((T, 2), bool)
        pres[0:4, 0] = True   # visitor 0: present at start (left-censored), then again 8..11
        pres[8:12, 0] = True
        pres[5:7, 1] = True   # visitor 1: one complete visit
        system = syn.make_scripted_system(pres, n_bulk=10, box=30.0, dt=2.0, seed=1)
        result, _ = _run(system, tmp_path)
        z = result.soz_results["Z"]
        pf, ps, sm = z.per_frame, z.per_solvent, z.summary

        np.testing.assert_array_equal(pf["n_solvent"], pres.sum(axis=1))
        # frame 0 has no predecessor -> no events; visitor 1 enters at 5, visitor 0 re-enters at 8
        assert pf["entries"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        assert sm["n_entry_events"] == 2 and sm["initial_population"] == 1
        assert sm["occupancy_fraction"] == pytest.approx(pres.any(axis=1).mean())
        assert sm["mean_n_solvent"] == pytest.approx(pres.sum(axis=1).mean())

        visits = z.visits
        assert len(visits) == 3
        durations = dict(zip(zip(visits["resindex"], visits["start_frame"]), visits["duration"]))
        assert sorted(visits["duration"].tolist()) == [4.0, 8.0, 8.0]  # 2 frames, 4 frames x dt=2 ps
        assert visits["censored_left"].sum() == 1 and visits["censored_right"].sum() == 0
        res = sm["residence"]["continuous"]
        assert (res["n_visits"], res["n_visits_complete"], res["n_left_censored"]) == (3, 2, 1)
        assert res["mean_visit_duration_complete"] == pytest.approx(6.0)
        row0 = ps[ps["present_at_start"]].iloc[0]
        assert row0["n_visits_censored"] == 1 and row0["entries"] == 1

    def test_irregular_timestamps_use_midpoint_rule(self, tmp_path):
        times = np.array([0, 1, 2, 3, 4, 104, 204, 304, 404, 405, 406, 407], float)
        pres = np.zeros((len(times), 1), bool)
        pres[5:9, 0] = True
        result, _ = _run(syn.make_scripted_system(pres, n_bulk=15, box=30.0, times=times, seed=4), tmp_path)
        z = result.soz_results["Z"]
        truth = (times[8] + times[9]) / 2 - (times[4] + times[5]) / 2  # 350.5 ps
        assert z.visits["duration"].iloc[0] == pytest.approx(truth)
        assert z.per_solvent["mean_res_time_cont"].max() == pytest.approx(truth)  # v2.0 reported 4.0
        assert z.summary["timing"]["uniform"] is False
        assert any(t["id"] == "timing.irregular" for t in result.qc_summary["trust"])

    def test_single_frame_has_no_fabricated_dt(self, tmp_path):
        pres = np.zeros((6, 1), bool); pres[:, 0] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=2)
        gro, xtc = system.write(tmp_path, "t")
        project = make_project(gro, xtc, tmp_path / "o", analysis=AnalysisOptions(
            frame_start=2, frame_stop=3, store_min_distances=False, workers=1))
        res = SOZAnalysisEngine(project).run()
        z = res.soz_results["Z"]
        assert np.isnan(z.summary["dt"])
        occupied = z.per_solvent[z.per_solvent["frames_present"] > 0]
        assert len(occupied) == 1 and np.isnan(occupied["mean_res_time_cont"].iloc[0])  # v2.0 reported 1.0 "ps"
        assert not res.timing.trusted
        assert any(t["id"] == "timing.single_frame" and t["level"] == "warning" for t in res.qc_summary["trust"])
        # strict JSON export of NaN: null, never the NaN token
        export_results(res, project)
        text = (tmp_path / "o" / "soz_Z" / "summary.json").read_text()
        assert "NaN" not in text and json.loads(text)["dt"] is None

    def test_format_without_timestamps_is_flagged_and_dt_override_fixes_it(self, tmp_path):
        pres = np.zeros((10, 1), bool); pres[3:7, 0] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=3)
        pdb = tmp_path / "notime.pdb"
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mda.Writer(str(pdb), n_atoms=len(system.universe.atoms), multiframe=True) as w:
                for i in range(system.n_frames):
                    system.universe.trajectory[i]
                    w.write(system.universe.atoms)
            r = SOZAnalysisEngine(make_project(pdb, None, tmp_path / "o1")).run()
        assert r.timing.source == "assumed" and not r.timing.trusted
        assert any(t["id"] == "timing.assumed" for t in r.qc_summary["trust"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = SOZAnalysisEngine(make_project(pdb, None, tmp_path / "o2", analysis=AnalysisOptions(
                store_min_distances=False, workers=1, dt_override=0.5))).run()
        assert r2.timing.source == "user" and r2.timing.trusted
        assert r2.soz_results["Z"].visits["duration"].iloc[0] == pytest.approx(4 * 0.5)

    def test_gap_tolerance_in_time_is_stride_independent(self, tmp_path):
        # dt = 1 ps. Visitor inside 0-9 and 12-21 (gap of 2 frames = 2 ps), and 40-49.
        pres = np.zeros((60, 1), bool)
        pres[0:10] = True; pres[12:22] = True; pres[40:50] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=5)
        gro, xtc = system.write(tmp_path, "t")
        out = {}
        for stride in (1, 2):
            project = make_project(gro, xtc, tmp_path / f"o{stride}", analysis=AnalysisOptions(
                stride=stride, store_min_distances=False, workers=1, gap_tolerance_time=2.0))
            res = SOZAnalysisEngine(project).run()
            inter = res.soz_results["Z"].summary["residence"]["intermittent"]
            out[stride] = (inter["n_visits"], res.soz_results["Z"].summary["gap_tolerance_samples"])
        assert out[1] == (2, 2)   # the 2 ps gap is bridged: 2 intermittent visits (v2.0: depends on stride)
        assert out[2] == (2, 1)   # same physical definition at stride 2 -> tolerance of 1 sample

    def test_ideal_gas_occupancy_is_binomial(self, tmp_path):
        n, box, r = 800, 30.0, 4.0
        system = syn.make_ideal_gas_system(n_waters=n, n_frames=600, box=box, seed=11)
        result, _ = _run(system, tmp_path, cutoff=r)
        s = result.soz_results["Z"].summary
        p = (4 / 3) * np.pi * r**3 / box**3
        unc = s["uncertainty"]["mean_n_solvent"]
        assert abs(s["mean_n_solvent"] - n * p) < 4 * unc["se"] + 1e-9
        pf = result.soz_results["Z"].per_frame["n_solvent"].to_numpy()
        assert pf.var(ddof=1) == pytest.approx(n * p * (1 - p), rel=0.2)
        assert s["occupancy_fraction"] == pytest.approx(1 - (1 - p) ** n, abs=0.02)
        # independent oracle agrees frame by frame - on the data actually read back from the XTC
        # (the file quantises coordinates to 0.01 A; waters within 0.005 A of the cutoff would flip otherwise)
        gro, xtc = tmp_path / "t.gro", tmp_path / "t.xtc"
        ref = mda.Universe(str(gro), str(xtc))
        stored = np.array([ts.positions.copy() for ts in ref.trajectory])
        np.testing.assert_array_equal(pf, oracle_n_solvent(stored, system.box, r))
        assert unc["tau_int"] < 2.0  # independent frames: no autocorrelation

    def test_min_distance_traces_match_geometry(self, tmp_path):
        pres = np.zeros((6, 1), bool); pres[2:5, 0] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, inside_r=2.5, seed=7)
        gro, xtc = system.write(tmp_path, "t")
        project = make_project(gro, xtc, tmp_path / "o", analysis=AnalysisOptions(store_min_distances=True, workers=1))
        pf = SOZAnalysisEngine(project).run().soz_results["Z"].per_frame
        assert np.isnan(pf["sel"].iloc[0]) and np.isnan(pf["sel"].iloc[5])
        assert pf["sel"].iloc[2:5].to_numpy() == pytest.approx(2.5, abs=0.02)


class TestPeriodicGeometry:
    @pytest.mark.parametrize("mode,sel", [("atom", "name OW"), ("cog", "name OW HW1 HW2"), ("com", "name OW HW1 HW2")])
    def test_split_molecule_without_bonds(self, tmp_path, mode, sel):
        """v2.0 crashed (COG/COM) on bond-less topologies; a split molecule must be found in every probe mode."""
        system = syn.make_straddling_water_system(box=20.0, n_frames=3)
        result, _ = _run(system, tmp_path, cutoff=3.5, probe=(sel, mode))
        assert result.soz_results["Z"].per_frame["n_solvent"].tolist() == [1, 1, 1]

    def test_zone_across_triclinic_boundary_matches_bruteforce(self, tmp_path):
        rng = np.random.default_rng(0)
        n_w, n_f = 200, 5
        dims = np.array([24.0, 26.0, 28.0, 80.0, 85.0, 75.0])
        u = mda.Universe.empty(1 + n_w, n_residues=1 + n_w, atom_resindex=np.arange(1 + n_w), trajectory=True)
        u.add_TopologyAttr("name", ["C1"] + ["OW"] * n_w)
        u.add_TopologyAttr("resname", ["LIG"] + ["SOL"] * n_w)
        u.add_TopologyAttr("resid", np.arange(1, n_w + 2))  # without ids GRO parsers merge consecutive residues
        u.add_TopologyAttr("mass", [12.0] + [16.0] * n_w)
        frac = rng.uniform(0, 1, (n_f, 1 + n_w, 3))
        from MDAnalysis.lib.mdamath import triclinic_vectors
        mat = triclinic_vectors(dims)
        coords = (frac @ mat).astype(np.float32)
        coords[:, 0, :] = (np.array([0.02, 0.03, 0.97]) @ mat)  # solute at a corner: images on all sides
        u.load_new(coords, order="fac", dimensions=np.tile(dims, (n_f, 1)))
        gro = tmp_path / "tri.gro"; u.atoms.write(str(gro))
        xtc = tmp_path / "tri.xtc"
        with mda.Writer(str(xtc), n_atoms=1 + n_w) as w:
            for ts in u.trajectory:
                w.write(u.atoms)
        res = SOZAnalysisEngine(make_project(gro, xtc, tmp_path / "o", cutoff=6.0)).run()
        got = res.soz_results["Z"].per_frame["n_solvent"].to_numpy()
        # brute force over 27 images from the stored (quantised) trajectory
        ref = mda.Universe(str(gro), str(xtc))
        expected = []
        import itertools
        for ts in ref.trajectory:
            pos = ref.atoms.positions.astype(np.float64)
            m = triclinic_vectors(ts.dimensions).astype(np.float64)
            shifts = np.array([i * m[0] + j * m[1] + k * m[2] for i, j, k in itertools.product((-1, 0, 1), repeat=3)])
            d = (pos[1:, None, :] + shifts[None, :, :]) - pos[0]
            expected.append(int((np.linalg.norm(d, axis=2).min(axis=1) <= 6.0).sum()))
        np.testing.assert_array_equal(got, expected)
        assert got.sum() > 0

    def test_merged_residues_are_detected_instead_of_silently_analysed(self, tmp_path):
        """Consecutive molecules with identical resid+resname are merged by GRO/PDB parsers: must not pass silently."""
        pres = np.ones((3, 1), bool)
        system = syn.make_scripted_system(pres, n_bulk=20, box=30.0, seed=1)
        system.universe.residues[5].resid = system.universe.residues[4].resid  # two adjacent bulk waters share a resid
        gro, xtc = system.write(tmp_path, "t")
        result = SOZAnalysisEngine(make_project(gro, xtc, tmp_path / "o")).run()  # geometry may or may not flag it ...
        assert any("different number of atoms" in w for w in result.warnings)  # ... the atom-count check always does


class TestIdentityAndValidation:
    def test_duplicate_resids_get_unique_ids_that_round_trip(self, tmp_path):
        pres = np.ones((4, 2), bool)
        system = syn.make_scripted_system(pres, n_bulk=2, box=30.0, seed=7)
        system.universe.residues[4].resid = system.universe.residues[1].resid  # resid wrap, non-adjacent
        result, _ = _run(system, tmp_path)
        ids = result.soz_results["Z"].per_solvent["solvent_id"].tolist()
        assert len(ids) == len(set(ids)) == 4
        assert sum(1 for i in ids if "#" in i) == 2  # only the colliding pair is disambiguated
        assert any("duplicate" in w.lower() or "share a resname" in w for w in result.warnings)

    def test_unknown_shell_parameter_is_reported(self, tmp_path):
        pres = np.ones((3, 1), bool)
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, inside_r=4.5, outside_min=9.0, seed=12)
        node = SOZNode(type="shell", params={"selection_label": "sel", "distance": 5.0, "unit": "A"})
        result, _ = _run(system, tmp_path, node=node)
        assert any("'distance'" in w or "distance are ignored" in w or "ignored" in w for w in result.warnings)
        assert any("[3.5]" in w for w in result.warnings)

    def test_cutoff_larger_than_half_box_is_warned(self, tmp_path):
        pres = np.ones((3, 1), bool)
        system = syn.make_scripted_system(pres, n_bulk=5, box=16.0, outside_min=7.0, seed=13)
        result, _ = _run(system, tmp_path, cutoff=12.0)
        assert any("half the shortest box width" in w for w in result.warnings)

    def test_shell_chained_vs_radial(self, tmp_path):
        # solute at centre; water A 3.0 A away; water B 4 A beyond A (7 A from the solute)
        box = np.full(3, 40.0); c = box / 2
        oxy = np.array([c + [3.0, 0, 0], c + [7.0, 0, 0], c + [-15.0, 0, 0]])
        n_f = 3
        rng = np.random.default_rng(0)
        atoms = syn.waters_from_oxygens(np.tile(oxy, (n_f, 1, 1)), rng)
        positions = np.concatenate([np.tile(c, (n_f, 1, 1)), atoms.reshape(n_f, 9, 3)], axis=1).astype(np.float32)
        u = syn._build_universe(1, 3, 3, positions, box, np.arange(n_f, dtype=float))
        system = syn.SyntheticSystem(u, positions, box, np.arange(n_f, dtype=float))
        gro, xtc = system.write(tmp_path, "t")
        counts = {}
        for mode in ("chained", "radial"):
            node = SOZNode(type="shell", params={"selection_label": "sel", "cutoffs": [3.5, 5.0], "unit": "A", "mode": mode})
            res = SOZAnalysisEngine(make_project(gro, xtc, tmp_path / mode, node=node)).run()
            counts[mode] = res.soz_results["Z"].per_frame["n_solvent"].tolist()
        assert counts["chained"] == [2, 2, 2]  # A (<=3.5 of solute) + B (<=5 of A): reach 8.5 A
        assert counts["radial"] == [1, 1, 1]   # only A is within 5.0 A of the solute

    def test_legacy_v20_project_file_still_loads_and_runs(self, tmp_path):
        from engine.analysis import load_project_json
        pres = np.zeros((8, 1), bool); pres[2:5, 0] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=1)
        gro, xtc = system.write(tmp_path, "t")
        legacy = {  # exactly the keys a 2.0.0 project file has: no 2.1 fields anywhere
            "version": "1.0",
            "inputs": {"topology": str(gro), "trajectory": str(xtc), "processed_trajectory": None, "preprocessing_notes": None},
            "solvent": {"solvent_label": "Water", "water_resnames": ["SOL"], "water_oxygen_names": ["OW"],
                        "water_hydrogen_names": ["HW1", "HW2"], "ion_resnames": ["NA", "CL"], "include_ions": False,
                        "probe": {"selection": "name OW", "position": "atom"}},
            "selections": {"sel": {"label": "sel", "selection": "resname LIG", "require_unique": True}},
            "sozs": [{"name": "Z", "description": "", "root": {"type": "distance", "params": {
                "selection_label": "sel", "cutoff": 3.5, "unit": "A"}, "children": []}}],
            "analysis": {"frame_start": 0, "frame_stop": None, "stride": 1, "gap_tolerance": 0, "workers": 1,
                         "store_ids": True, "store_min_distances": False, "preview_frames": 200},
            "outputs": {"output_dir": str(tmp_path / "o"), "write_per_frame": True, "write_parquet": False, "report_format": "md"},
            "hbond_hydration": [], "density_maps": [], "water_dynamics": [],
        }
        p = tmp_path / "legacy.json"; p.write_text(json.dumps(legacy))
        res = SOZAnalysisEngine(load_project_json(str(p))).run()
        assert res.soz_results["Z"].per_frame["n_solvent"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
        for col in ("frame", "time", "n_solvent", "solvent_ids", "entries", "exits"):  # v2.0 columns unchanged
            assert col in res.soz_results["Z"].per_frame.columns


class TestCancellationAndExport:
    def test_cancelled_run_is_marked_partial_and_never_overwrites_complete_results(self, tmp_path):
        pres = np.ones((20, 1), bool)
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=11)
        gro, xtc = system.write(tmp_path, "t")
        project = make_project(gro, xtc, tmp_path / "out")
        full = SOZAnalysisEngine(project).run()
        export_results(full, project)
        before = (tmp_path / "out" / "soz_Z" / "per_frame.csv").read_bytes()

        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 7

        part = SOZAnalysisEngine(project).run(cancel_flag=cancel)
        assert part.cancelled and not part.complete and len(part.soz_results["Z"].per_frame) == 7
        assert part.qc_summary["complete"] is False
        with pytest.raises(PartialResultError):
            export_results(part, project)
        assert (tmp_path / "out" / "soz_Z" / "per_frame.csv").read_bytes() == before  # untouched
        paths = export_results(part, project, allow_partial=True)
        assert paths["output_dir"].endswith("partial")
        assert json.loads((tmp_path / "out" / "partial" / "run_status.json").read_text())["status"] == "partial"

    def test_export_is_self_verifying_and_byte_reproducible(self, tmp_path):
        pres = np.zeros((10, 1), bool); pres[2:6, 0] = True
        system = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=2)
        gro, xtc = system.write(tmp_path, "t")
        hashes = []
        for k in (1, 2):
            project = make_project(gro, xtc, tmp_path / f"o{k}")
            res = SOZAnalysisEngine(project).run()
            paths = export_results(res, project)
            assert verify_export(paths["output_dir"]) == []
            m = json.loads((tmp_path / f"o{k}" / "manifest.json").read_text())["files"]
            hashes.append({f: v["sha256"] for f, v in m.items() if f.startswith("soz_Z/") and not f.endswith("per_frame.csv") or f == "soz_Z/per_frame.csv"})
        # identical inputs -> identical scientific outputs byte for byte (summary.json holds no wall-clock data)
        assert hashes[0] == hashes[1]
        (tmp_path / "o1" / "soz_Z" / "per_solvent.csv").write_text("tampered")
        assert any("modified" in p for p in verify_export(str(tmp_path / "o1")))
        assert (tmp_path / "o1" / "DATA_DICTIONARY.md").exists()
        assert (tmp_path / "o1" / "provenance.json").exists()
