"""Foundation tests: timing model, provenance fingerprints, periodic geometry."""
from __future__ import annotations

import itertools
import json
import math
import os
import warnings

import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")

from engine import provenance as prov
from engine import timing as tm
from engine.periodic import (
    Box,
    box_from_dimensions,
    cutoff_warning,
    group_layout,
    kabsch_fit,
    minimum_image,
    molecule_centers,
    rotation_angle_deg,
    wrap_about,
)
from engine.serialization import to_jsonable
from engine import synthetic as syn


# ----------------------------------------------------------------------------- timing
class TestTiming:
    def test_uniform_durations_equal_frames_times_dt(self):
        times = np.arange(10) * 2.5
        d = tm.segment_durations([0, 3, 7], [2, 3, 9], times)
        assert d == pytest.approx([3 * 2.5, 1 * 2.5, 3 * 2.5])

    def test_irregular_durations_use_midpoint_rule(self):
        times = np.array([0, 1, 2, 3, 4, 104, 204, 304, 404, 405, 406, 407], float)
        # samples 5..8 inside: from (t4+t5)/2 = 54 to (t8+t9)/2 = 404.5
        d = tm.segment_durations([5], [8], times)[0]
        assert d == pytest.approx(404.5 - 54.0)

    def test_single_sample_has_no_duration(self):
        t = tm.assess_timing([3], [30.0])
        assert not t.usable and not t.trusted
        assert math.isnan(t.dt)
        assert np.isnan(tm.segment_durations([0], [0], [30.0])).all()
        assert any("one frame" in n for n in t.notes)

    def test_irregular_flags_and_notes(self):
        t = tm.assess_timing(range(5), [0, 1, 2, 10, 11])
        assert t.usable and not t.uniform and t.monotonic
        assert any("irregular" in n for n in t.notes)

    def test_nonmonotonic_is_untrusted_duration_source(self):
        t = tm.assess_timing(range(5), [0, 1, 2, 1, 2])
        assert not t.monotonic and not t.usable

    def test_assumed_source_is_not_trusted(self):
        t = tm.assess_timing(range(4), [0, 1, 2, 3], source=tm.SOURCE_ASSUMED)
        assert t.usable and not t.trusted
        assert any("dt_override" in n for n in t.notes)

    def test_gap_tolerance_in_frames_is_stride_independent_in_time(self):
        fine = tm.assess_timing(range(100), np.arange(100) * 1.0)
        coarse = tm.assess_timing(range(0, 400, 4), np.arange(0, 400, 4) * 1.0)
        assert tm.gap_tolerance_in_frames(2.0, fine) == 2
        assert tm.gap_tolerance_in_frames(2.0, coarse) == 0  # 2 ps < one 4 ps interval

    def test_detect_time_source(self, tmp_path):
        s = syn.make_ideal_gas_system(n_waters=5, n_frames=3, box=10.0)
        gro, xtc = s.write(tmp_path, "t")
        assert tm.detect_time_source(mda.Universe(gro, xtc)) == tm.SOURCE_TRAJECTORY
        pdb = tmp_path / "m.pdb"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mda.Writer(str(pdb), n_atoms=len(s.universe.atoms), multiframe=True) as w:
                for i in range(3):
                    s.universe.trajectory[i]
                    w.write(s.universe.atoms)
            assert tm.detect_time_source(mda.Universe(str(pdb))) == tm.SOURCE_ASSUMED
        assert tm.detect_time_source(s.universe) == tm.SOURCE_MEMORY


# ----------------------------------------------------------------------------- provenance
class TestProvenance:
    def test_same_size_rewrite_with_restored_mtime_is_detected(self, tmp_path):
        p = tmp_path / "traj.bin"
        p.write_bytes(os.urandom(4096))
        fp = prov.file_fingerprint(p)
        st = p.stat()
        p.write_bytes(os.urandom(4096))  # same size, different content
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # restore mtime: size+mtime cannot see this
        assert prov.verify_fingerprint(fp), "content change must be detected by hash"

    def test_unchanged_content_with_new_mtime_is_not_a_difference(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"x" * 1000)
        fp = prov.file_fingerprint(p)
        os.utime(p, ns=(1, 1))
        assert prov.verify_fingerprint(fp) == []

    def test_size_change_and_missing_file(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc")
        fp = prov.file_fingerprint(p)
        p.write_bytes(b"abcd")
        assert "size changed" in prov.verify_fingerprint(fp)[0]
        p.unlink()
        assert "not found" in prov.verify_fingerprint(fp)[0]

    def test_sampled_mode_is_deterministic_and_detects_head_edit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prov, "_HEAD_TAIL", 64)
        monkeypatch.setattr(prov, "_SAMPLE_BLOCK", 32)
        monkeypatch.setattr(prov, "_N_SAMPLE_BLOCKS", 8)
        p = tmp_path / "big.bin"
        data = bytearray(os.urandom(20_000))
        p.write_bytes(bytes(data))
        fp = prov.file_fingerprint(p, mode="sampled")
        assert fp["hash_mode"] == "sampled"
        assert prov.verify_fingerprint(fp) == []
        data[10] ^= 0xFF
        p.write_bytes(bytes(data))
        assert prov.verify_fingerprint(fp)

    def test_assert_inputs_unchanged_raises_with_differences(self, tmp_path):
        top = tmp_path / "t.gro"; top.write_text("x")
        traj = tmp_path / "t.xtc"; traj.write_bytes(b"1234")
        rec = {"topology": prov.file_fingerprint(top), "trajectory": prov.file_fingerprint(traj), "extra": []}
        prov.assert_inputs_unchanged(rec)
        traj.write_bytes(b"1235")
        with pytest.raises(prov.InputsChangedError) as exc:
            prov.assert_inputs_unchanged(rec)
        assert any("t.xtc" in d for d in exc.value.differences)

    def test_config_hash_ignores_output_dir_and_workers(self):
        from engine.models import default_project

        a = default_project("a.gro", "a.xtc")
        b = default_project("a.gro", "a.xtc")
        b.outputs.output_dir = "elsewhere"
        b.analysis.workers = 7
        assert prov.build_provenance(a, fingerprint_mode="sampled")["config_sha256"] == \
            prov.build_provenance(b, fingerprint_mode="sampled")["config_sha256"]
        b.analysis.stride = 3
        assert prov.build_provenance(a)["config_sha256"] != prov.build_provenance(b)["config_sha256"]

    def test_frame_fingerprint_changes_with_content_not_with_box_order(self):
        pos = np.random.default_rng(0).uniform(0, 10, (50, 3)).astype(np.float32)
        idx = prov.sample_atom_indices(50, 16)
        a = prov.frame_fingerprint(pos, np.array([10, 10, 10, 90, 90, 90.0]), idx)
        assert a == prov.frame_fingerprint(pos.copy(), np.array([10, 10, 10, 90, 90, 90.0]), idx)
        pos2 = pos.copy(); pos2[idx[3], 1] += 0.01
        assert a != prov.frame_fingerprint(pos2, np.array([10, 10, 10, 90, 90, 90.0]), idx)

    def test_to_jsonable_maps_nan_and_inf_to_null_and_roundtrips_strictly(self):
        payload = {"a": float("nan"), "b": np.float64("inf"), "c": np.array([1.0, np.nan]), "d": {1, 2}, "e": np.int64(3)}
        text = json.dumps(to_jsonable(payload), allow_nan=False)
        assert json.loads(text) == {"a": None, "b": None, "c": [1.0, None], "d": [1, 2], "e": 3}

    def test_output_manifest_hashes_files(self, tmp_path):
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / "f.csv").write_bytes(b"a,b\n1,2\n")
        m = prov.output_manifest(tmp_path)
        assert m["files"]["x/f.csv"]["size"] == 8 and len(m["files"]["x/f.csv"]["sha256"]) == 64


# ----------------------------------------------------------------------------- periodic
def _brute_min_image(vec, matrix):
    best = vec.copy()
    best_n = np.linalg.norm(vec)
    for i, j, k in itertools.product((-1, 0, 1), repeat=3):
        cand = vec + i * matrix[0] + j * matrix[1] + k * matrix[2]
        n = np.linalg.norm(cand)
        if n < best_n:
            best, best_n = cand, n
    return best


class TestPeriodic:
    def test_box_limits_orthorhombic_and_triclinic(self):
        ortho = box_from_dimensions([30, 40, 50, 90, 90, 90])
        assert ortho.orthorhombic and ortho.min_image_limit == pytest.approx(15.0)
        assert ortho.volume == pytest.approx(30 * 40 * 50)
        tri = box_from_dimensions([40, 40, 40, 60, 60, 90])
        assert not tri.orthorhombic
        assert tri.min_image_limit < 20.0  # skew reduces the exact-search radius
        assert box_from_dimensions([0, 10, 10, 90, 90, 90]) is None
        assert box_from_dimensions(None) is None
        assert box_from_dimensions([10, 10, 10, 90, 90, 180]) is None

    def test_cutoff_warning(self):
        box = box_from_dimensions([20, 20, 20, 90, 90, 90])
        assert cutoff_warning(9.9, box) is None
        assert "half the shortest box width" in cutoff_warning(10.5, box)
        assert cutoff_warning(50.0, None) is None

    @pytest.mark.parametrize("dims", [[25, 25, 25, 90, 90, 90], [25, 28, 31, 70, 80, 75]])
    def test_minimum_image_matches_brute_force(self, dims):
        box = box_from_dimensions(dims)
        rng = np.random.default_rng(1)
        vecs = rng.uniform(-60, 60, (200, 3))
        got = minimum_image(vecs, box)
        # compare lengths (minimum image is defined up to the shortest vector)
        mat = box.matrix
        # shift into the primary neighbourhood first, then brute force the 27 images
        for v, g in zip(vecs, got):
            frac = np.linalg.solve(mat.T, v)
            shifted = v - np.round(frac) @ mat
            ref = _brute_min_image(shifted, mat)
            assert np.linalg.norm(g) == pytest.approx(np.linalg.norm(ref), abs=1e-4)

    def test_molecule_centers_split_across_boundary_without_bonds(self):
        box = box_from_dimensions([20, 20, 20, 90, 90, 90])
        # O at x=19.9; both H wrapped to x~0.49: raw COG would be ~6.96
        pos = np.array([[19.9, 10, 10], [0.4866, 10.756, 10], [0.4866, 9.244, 10]])
        order, starts, counts = group_layout(np.zeros(3, dtype=int))
        solute = np.array([2.0, 10, 10])
        cog = molecule_centers(pos[order], starts, counts, None, box)[0]
        # centre is returned in the image of the reference atom (O at 19.9): 19.9 + 2*0.5866/3
        assert cog[0] == pytest.approx(19.9 + 2 * 0.5866 / 3, abs=2e-3)
        assert np.linalg.norm(minimum_image((cog - solute)[None, :], box)) == pytest.approx(1.709, abs=0.01)
        com = molecule_centers(pos[order], starts, counts, np.array([15.999, 1.008, 1.008]), box)[0]
        assert com[0] == pytest.approx(19.9 + 2 * 1.008 * 0.5866 / 18.015, abs=2e-3)
        raw = pos.mean(axis=0)
        assert np.linalg.norm(raw - solute) > 4.0  # the naive raw-coordinate centre is far from the solute

    def test_molecule_centers_multiple_molecules_independent(self):
        box = box_from_dimensions([10, 10, 10, 90, 90, 90])
        pos = np.array([[9.8, 5, 5], [0.2, 5, 5], [5, 5, 5], [5.5, 5, 5]])
        owner = np.array([0, 0, 1, 1])
        order, starts, counts = group_layout(owner)
        c = molecule_centers(pos[order], starts, counts, None, box)
        assert c[0, 0] == pytest.approx(10.0, abs=1e-9)  # 9.8 and 10.2 -> 10.0
        assert c[1, 0] == pytest.approx(5.25)

    def test_wrap_about_places_points_in_nearest_image(self):
        box = box_from_dimensions([10, 10, 10, 90, 90, 90])
        out = wrap_about(np.array([[9.5, 1, 1], [0.5, 1, 1]]), np.array([0.0, 1, 1]), box)
        assert out[0, 0] == pytest.approx(-0.5) and out[1, 0] == pytest.approx(0.5)

    def test_kabsch_recovers_rotation_and_translation(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(size=(30, 3)) * 5
        rot = syn.random_rotations(rng, 1)[0]
        shift = np.array([3.0, -2.0, 7.5])
        mobile = ref @ np.linalg.inv(rot) + shift  # inverse of what fit must apply
        r, mc, rc, degenerate = kabsch_fit(mobile, ref)
        assert not degenerate
        fitted = (mobile - mc) @ r + rc
        assert np.abs(fitted - ref).max() < 1e-9
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_kabsch_mass_weighting_changes_nothing_for_exact_fit_but_is_accepted(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(size=(12, 3))
        mobile = ref + np.array([1.0, 2.0, 3.0])
        r, mc, rc, _ = kabsch_fit(mobile, ref, weights=rng.uniform(1, 16, 12))
        assert np.abs((mobile - mc) @ r + rc - ref).max() < 1e-9
        assert rotation_angle_deg(r) == pytest.approx(0.0, abs=1e-6)

    def test_kabsch_degenerate_inputs_fall_back_to_translation(self):
        _, mc, rc, degenerate = kabsch_fit(np.array([[1.0, 1, 1]]), np.array([[0.0, 0, 0]]))
        assert degenerate
        line_m = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
        r, _, _, degenerate = kabsch_fit(line_m + 5, line_m)
        assert degenerate and np.allclose(r, np.eye(3))
