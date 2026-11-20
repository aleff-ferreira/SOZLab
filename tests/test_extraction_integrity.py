"""Frame extraction must write exactly the frames that were analysed, or refuse."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

mda = pytest.importorskip("MDAnalysis")

from _helpers import make_project
from engine import synthetic as syn
from engine.analysis import SOZAnalysisEngine
from engine.export import export_results
from engine.extract import (
    ExtractionMismatchError,
    extract_from_results,
    select_frames,
    select_from_result,
    write_extracted_trajectory,
)
from engine.provenance import InputsChangedError


def _scripted(frames_on, T=30, seed=3):
    pres = np.zeros((T, 1), bool)
    pres[frames_on, 0] = True
    return syn.make_scripted_system(pres, n_bulk=20, box=30.0, seed=seed)


def _count_in_zone(universe, cutoff=3.5):
    box = np.array(universe.dimensions[:3])
    counts = []
    for ts in universe.trajectory:
        pos = universe.atoms.positions
        d = pos[1::3] - pos[0]
        d -= np.round(d / box) * box
        counts.append(int((np.linalg.norm(d, axis=1) <= cutoff).sum()))
    return counts


@pytest.fixture
def analysed(tmp_path):
    a = _scripted(slice(10, 15))
    gro, xtc = a.write(tmp_path, "traj")
    project = make_project(gro, xtc, tmp_path / "out")
    result = SOZAnalysisEngine(project).run()
    return tmp_path, gro, xtc, project, result


def test_extracted_frames_are_the_analysed_ones(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    assert sel.frame_indices == [10, 11, 12, 13, 14] and sel.frame_fps and sel.n_atoms
    out = write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(tmp_path / "ex"),
                                     topology_path=str(gro), trajectory_path=str(xtc))
    ue = mda.Universe(out["reference"], out["trajectory"])
    assert len(ue.trajectory) == 5 and _count_in_zone(ue) == [1] * 5
    params = json.loads(open(out["params"]).read())
    assert params["verification"]["mode"] == "frame_fp+time" and params["verification"]["n_frames_verified"] == 5
    assert params["analysed_inputs"]["trajectory"]["sha256"]
    assert "frame_fp" in pd.read_csv(out["manifest"]).columns


def test_replaced_trajectory_is_refused_and_nothing_is_left_behind(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    _scripted(slice(20, 25)).write(tmp_path, "traj")  # the v2.0 scenario: same path, other data
    outdir = tmp_path / "ex"
    # (a) content check alone, as the legacy call sequence uses it
    with pytest.raises(ExtractionMismatchError) as exc:
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(outdir))
    assert "does not contain the frames that were analysed" in str(exc.value) and exc.value.mismatches
    assert not outdir.exists() or [p for p in outdir.iterdir()] == []  # no partial/staged files
    # (b) with the file identity check: refused before any frame is read
    with pytest.raises(InputsChangedError):
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(outdir),
                                   topology_path=str(gro), trajectory_path=str(xtc))
    assert not outdir.exists() or [p for p in outdir.iterdir()] == []


def test_truncated_trajectory_is_refused(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    short = syn.make_scripted_system(np.zeros((12, 1), bool), n_bulk=20, box=30.0, seed=3)
    short.write(tmp_path, "traj")  # only 12 frames: frames 12-14 no longer exist
    with pytest.raises(ExtractionMismatchError, match="does not exist|differ|time"):
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(tmp_path / "ex"))


def test_earlier_extractions_are_never_overwritten_silently(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    u = mda.Universe(str(gro), str(xtc))
    first = write_extracted_trajectory(u, sel, str(tmp_path / "ex"))
    second = write_extracted_trajectory(u, sel, str(tmp_path / "ex"))
    assert os.path.basename(first["trajectory"]) == "extracted.xtc"
    assert os.path.basename(second["trajectory"]) == "extracted_2.xtc"
    third = write_extracted_trajectory(u, sel, str(tmp_path / "ex"), overwrite=True)
    assert os.path.basename(third["trajectory"]) == "extracted.xtc"
    assert not [p for p in os.listdir(tmp_path / "ex") if p.startswith(".sozlab_extract_")]


def test_cancelled_extraction_leaves_no_files(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 8

    with pytest.raises(InterruptedError):
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(tmp_path / "ex"), cancel=cancel)
    assert os.listdir(tmp_path / "ex") == []


def test_lossless_trr_roundtrip_and_xtc_precision(analysed):
    tmp_path, gro, xtc, project, result = analysed
    sel = select_from_result(result, "Z", "n_solvent>=1")
    u = mda.Universe(str(gro), str(xtc))
    ref = np.array([u.trajectory[i].positions.copy() for i in sel.frame_indices])
    # TRR keeps float32 (only the A<->nm unit conversion rounds, ~1 ulp = 2e-6 A at 28 A); XTC quantises to 0.001 nm
    for fmt, tol in (("trr", 1e-5), ("xtc", 0.0051)):
        out = write_extracted_trajectory(u, sel, str(tmp_path / f"ex_{fmt}"), fmt=fmt)
        back = mda.Universe(out["reference"], out["trajectory"])
        got = np.array([ts.positions.copy() for ts in back.trajectory])
        assert np.abs(got - ref).max() <= tol
        assert json.loads(open(out["params"]).read())["format"] == fmt
    with pytest.raises(ValueError, match="Unsupported extraction format"):
        write_extracted_trajectory(u, sel, str(tmp_path / "bad"), fmt="gro")


def test_filtered_per_frame_index_does_not_shift_the_manifest():
    pf = pd.DataFrame({"frame": [0, 1, 2, 3, 4], "time": [0.0, 1, 2, 3, 4], "n_solvent": [0, 2, 3, 0, 4]})
    shifted = pf.copy(); shifted.index = shifted.index + 100  # e.g. a filtered/sorted GUI table
    a, b = select_frames(pf, "n_solvent>=2"), select_frames(shifted, "n_solvent>=2")
    assert a.frame_indices == b.frame_indices == [1, 2, 4]
    assert b.manifest["n_solvent"].tolist() == [2, 3, 4]  # .loc with positional labels would raise / mis-select


def test_missing_values_never_satisfy_a_rule():
    pf = pd.DataFrame({"frame": range(4), "time": [0.0, 1, 2, 3], "n_solvent": [1, 1, 1, 1], "d": [np.nan, 2.0, np.nan, 0.5]})
    assert select_frames(pf, "d<=1.0").frame_indices == [3]  # v2.0 filled NaN with 0 and selected frames 0 and 2 too
    assert select_frames(pf, "d>=-1e0").frame_indices == [1, 3]  # signed / exponent literals parse


def test_run_and_gap_in_time_are_stride_independent():
    t = np.arange(0, 40, 2.0)  # stride-2 data: dt = 2 ps
    n = np.zeros(20, int); n[3:6] = 1; n[7:12] = 1  # runs of 3 and 5 samples (6 and 10 ps), gap of 1 sample (2 ps)
    pf = pd.DataFrame({"frame": np.arange(20) * 2, "time": t, "n_solvent": n})
    sel = select_frames(pf, "n_solvent>=1", min_run_time=8.0)  # >= 8 ps -> 4 samples: drops the 3-sample run
    assert sel.frame_indices == [14, 16, 18, 20, 22] and sel.params["min_run_length"] == 4
    merged = select_frames(pf, "n_solvent>=1", gap_tolerance_time=2.0)
    assert merged.params["gap_tolerance"] == 1 and len(merged.frame_indices) == 9  # 3 + gap + 5
    with pytest.raises(ValueError, match="time stamps"):
        select_frames(pf.iloc[:1], "n_solvent>=1", min_run_time=2.0)


def test_legacy_per_frame_without_fingerprints_falls_back_to_time_verification(analysed):
    tmp_path, gro, xtc, project, result = analysed
    per_frame = result.soz_results["Z"].per_frame.drop(columns=["frame_fp"])
    sel = select_frames(per_frame, "n_solvent>=1")
    assert sel.frame_fps is None and "time stamps only" in sel.params["content_verification"]
    out = write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(tmp_path / "ex"))
    assert json.loads(open(out["params"]).read())["verification"]["mode"] == "time"
    # shift the time axis of the file: even without fingerprints the extraction refuses
    pf2 = per_frame.copy(); pf2["time"] = pf2["time"] + 500.0
    with pytest.raises(ExtractionMismatchError, match="time"):
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), select_frames(pf2, "n_solvent>=1"), str(tmp_path / "ex2"))


def test_extract_from_saved_results_verifies_inputs_by_content(analysed):
    tmp_path, gro, xtc, project, result = analysed
    export_results(result, project)
    outputs, sel = extract_from_results(str(tmp_path / "out"), None, "n_solvent>=1", str(tmp_path / "ex"))
    assert sel.frame_indices == [10, 11, 12, 13, 14] and os.path.exists(outputs["trajectory"])
    _scripted(slice(20, 25)).write(tmp_path, "traj")
    with pytest.raises(InputsChangedError, match="content changed|size changed"):
        extract_from_results(str(tmp_path / "out"), None, "n_solvent>=1", str(tmp_path / "ex_again"))
