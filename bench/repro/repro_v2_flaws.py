"""Reproduce suspected SOZLab v2.0.0 flaws and record measured evidence.

The script only uses APIs that exist in both the released code (main @ 3dcf498) and
the fixed code, so the *same* file is run against both trees::

    PYTHONPATH=<tree>/src python bench/repro/repro_v2_flaws.py --label baseline --out bench/out/repro_baseline.json
    PYTHONPATH=src         python bench/repro/repro_v2_flaws.py --label fixed    --out bench/out/repro_fixed.json

Every check returns ``reproduced=True`` when the flaw is PRESENT. Ground truth comes from
``engine/synthetic.py`` (analytically known answers) and an independent numpy
minimum-image oracle, never from SOZLab itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]


def _load_synthetic():
    spec = importlib.util.spec_from_file_location("sozlab_synthetic", REPO / "src" / "engine" / "synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sozlab_synthetic"] = mod
    spec.loader.exec_module(mod)
    return mod


syn = _load_synthetic()

import MDAnalysis as mda  # noqa: E402
import engine  # noqa: E402
from engine.analysis import SOZAnalysisEngine, _run_density_maps  # noqa: E402,F401
from engine.extract import select_frames, write_extracted_trajectory  # noqa: E402
from engine.models import (  # noqa: E402
    AnalysisOptions,
    DensityMapConfig,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SelectionSpec,
    SolventConfig,
    SOZDefinition,
    SOZNode,
)


# ----------------------------------------------------------------------------- helpers
def make_project(topology, trajectory, out_dir, cutoff=3.5, analysis=None, density=None,
                 probe=("name OW", "atom"), node=None, resnames=("SOL",)):
    root = node or SOZNode(type="distance", params={"selection_label": "sel", "cutoff": cutoff, "unit": "A"})
    return ProjectConfig(
        inputs=InputConfig(topology=str(topology), trajectory=str(trajectory) if trajectory else None),
        solvent=SolventConfig(water_resnames=list(resnames), probe=ProbeConfig(selection=probe[0], position=probe[1])),
        selections={"sel": SelectionSpec(label="sel", selection="resname LIG")},
        sozs=[SOZDefinition(name="Z", description="repro", root=root)],
        analysis=analysis or AnalysisOptions(store_min_distances=False, workers=1),
        outputs=OutputConfig(output_dir=str(out_dir), write_per_frame=True),
        density_maps=list(density or []),
    )


def oracle_n_solvent(positions, box, n_visitors_total, solute_idx=0, water_o_stride=3):
    """Independent minimum-image count of waters whose O is within 3.5 A of the solute."""
    out = []
    for frame in positions:
        solute = frame[solute_idx]
        oxy = frame[1::water_o_stride]
        d = oxy - solute
        d -= np.round(d / box) * box
        out.append(int((np.linalg.norm(d, axis=1) <= 3.5).sum()))
    return np.array(out)


def _first(df, col, default=None):
    return df[col].iloc[0] if col in df.columns and len(df) else default


# ----------------------------------------------------------------------------- checks
def check_stale_density_cache(tmp):
    """Same trajectory PATH, different CONTENT -> cached density must not be reused."""
    sysA = syn.make_ideal_gas_system(n_waters=300, n_frames=12, box=24.0, seed=1)
    sysB = syn.make_ideal_gas_system(n_waters=300, n_frames=12, box=24.0, seed=2)
    d = Path(tmp) / "r1"
    gro, xtc = sysA.write(d, "traj")
    dens = DensityMapConfig(name="wat", species_selection="name OW", grid_spacing=1.5, padding=1.0,
                            align=True, align_selection="resname LIG", conditioning_policy="strict")
    out = d / "out"
    p1 = make_project(gro, xtc, out, density=[dens])
    res1 = SOZAnalysisEngine(p1).run()
    g1 = np.array(res1.density_results["wat"].grid)
    time.sleep(1.1)
    sysB.write(d, "traj")  # overwrite the SAME path with different data
    res2 = SOZAnalysisEngine(make_project(gro, xtc, out, density=[dens])).run()
    g2 = np.array(res2.density_results["wat"].grid)
    fresh = SOZAnalysisEngine(make_project(gro, xtc, d / "fresh", density=[dens])).run()
    gf = np.array(fresh.density_results["wat"].grid)
    stale_vs_run1 = float(np.max(np.abs(g2 - g1))) if g1.shape == g2.shape else float("inf")
    stale_vs_truth = float(np.max(np.abs(g2 - gf))) if g2.shape == gf.shape else float("inf")
    return dict(
        reproduced=bool(stale_vs_run1 == 0.0 and stale_vs_truth > 0.0),
        evidence=dict(max_abs_diff_run2_vs_run1=stale_vs_run1, max_abs_diff_run2_vs_fresh_recompute=stale_vs_truth,
                      cache_metadata_keys=sorted(res2.density_results["wat"].metadata.keys())[:40]),
        explanation="run2 used new trajectory content but returned run1's grid" if stale_vs_run1 == 0.0 else "cache invalidated",
    )


def check_extraction_mutated_inputs(tmp):
    """Analyse trajectory A, replace the file with B (same path/atoms/frames), then extract."""
    T = 30
    pres_a = np.zeros((T, 1), bool); pres_a[10:15, 0] = True
    pres_b = np.zeros((T, 1), bool); pres_b[20:25, 0] = True
    sysA = syn.make_scripted_system(pres_a, n_bulk=20, box=30.0, seed=3)
    sysB = syn.make_scripted_system(pres_b, n_bulk=20, box=30.0, seed=3)
    d = Path(tmp) / "r2"
    gro, xtc = sysA.write(d, "traj")
    project = make_project(gro, xtc, d / "out")
    res = SOZAnalysisEngine(project).run()
    per_frame = res.soz_results["Z"].per_frame
    sel = select_frames(per_frame, rule="n_solvent>=1")
    sysB.write(d, "traj")  # trajectory replaced after the analysis
    u = mda.Universe(str(gro), str(xtc))
    raised = None
    try:
        outputs = write_extracted_trajectory(u, sel, output_dir=str(d / "extracted"), prefix="x", fmt="xtc")
    except Exception as exc:  # fixed code is expected to refuse
        raised = f"{type(exc).__name__}: {str(exc)[:160]}"
        outputs = None
    extracted_n = None
    if outputs and "trajectory" in outputs:
        ue = mda.Universe(outputs["reference"], outputs["trajectory"])
        box = np.array(ue.dimensions[:3])
        counts = []
        for ts in ue.trajectory:
            pos = ue.atoms.positions
            solute = pos[0]
            oxy = pos[1::3]
            dd = oxy - solute
            dd -= np.round(dd / box) * box
            counts.append(int((np.linalg.norm(dd, axis=1) <= 3.5).sum()))
        extracted_n = counts
    wrong = extracted_n is not None and (np.mean(extracted_n) < 0.5)
    return dict(
        reproduced=bool(raised is None and wrong),
        evidence=dict(selected_frames=sel.frame_indices, refused_with=raised,
                      oracle_n_solvent_in_extracted_frames=extracted_n),
        explanation=("extraction silently wrote frames from the replaced file; the solvent the rule selected is not there"
                     if (raised is None and wrong) else "extraction refused or content consistent"),
    )


def check_timing_assumptions(tmp):
    """Irregular timestamps, single-frame runs and time-less formats."""
    out = {}
    # (a) irregular sampling: visitor present on frames 5..8 where spacing is 100 ps
    times = np.array([0, 1, 2, 3, 4, 104, 204, 304, 404, 405, 406, 407], float)
    pres = np.zeros((len(times), 1), bool); pres[5:9, 0] = True
    s = syn.make_scripted_system(pres, n_bulk=15, box=30.0, times=times, seed=4)
    d = Path(tmp) / "r3a"; gro, xtc = s.write(d, "traj")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out")).run()
    ps = r.soz_results["Z"].per_solvent
    row = ps[ps["frames_present"] > 0].iloc[0]
    reported = float(row["mean_res_time_cont"])
    # truth by midpoint rule: from (t4+t5)/2 to (t8+t9)/2
    truth = (times[8] + times[9]) / 2 - (times[4] + times[5]) / 2
    out["irregular_sampling"] = dict(reported_ps=reported, true_duration_ps=float(truth),
                                     rel_error=float(abs(reported - truth) / truth),
                                     summary_dt=r.soz_results["Z"].summary.get("dt"))
    # (b) single analysed frame
    r1 = SOZAnalysisEngine(make_project(gro, xtc, d / "out1", analysis=AnalysisOptions(
        frame_start=5, frame_stop=6, store_min_distances=False, workers=1))).run()
    s1 = r1.soz_results["Z"].summary
    ps1 = r1.soz_results["Z"].per_solvent
    row1 = ps1[ps1["frames_present"] > 0].iloc[0]
    out["single_frame"] = dict(summary_dt=s1.get("dt"), reported_res_time=float(row1["mean_res_time_cont"]),
                               timing_flagged=bool(s1.get("timing_trusted") is False or s1.get("dt") is None
                                                   or (isinstance(s1.get("dt"), float) and np.isnan(s1.get("dt")))))
    # (c) trajectory format without timestamps (multi-model PDB)
    pdb = d / "notime.pdb"
    with mda.Writer(str(pdb), n_atoms=len(s.universe.atoms), multiframe=True) as w:
        for i in range(s.n_frames):
            s.universe.trajectory[i]; w.write(s.universe.atoms)
    # a distance node cannot use a PDB written without resnames -> keep resnames: fine (names preserved)
    r2 = SOZAnalysisEngine(make_project(pdb, None, d / "out2")).run()
    qc = r2.qc_summary
    out["no_timestamps_format"] = dict(
        time_unit_reported=qc.get("time_unit"),
        mentions_assumed_timing=any(("dt" in w.lower() or "timing" in w.lower() or "time" in w.lower())
                                    for w in (qc.get("analysis_warnings", []) + qc.get("preflight", {}).get("warnings", []))),
        qc_timing_block=qc.get("timing"))
    flaw = (out["irregular_sampling"]["rel_error"] > 0.2
            and not out["single_frame"]["timing_flagged"]
            and not out["no_timestamps_format"]["mentions_assumed_timing"])
    return dict(reproduced=bool(flaw), evidence=out,
                explanation="duration = n_frames x median(dt); fabricated dt=1.0 for 1 frame; time-less formats reported as ps")


def check_initial_population_is_entries(tmp):
    pres = np.zeros((10, 2), bool); pres[:, 0] = True; pres[:, 1] = False
    s = syn.make_scripted_system(pres, n_bulk=10, box=30.0, seed=5)
    d = Path(tmp) / "r4"; gro, xtc = s.write(d, "traj")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out")).run()
    pf = r.soz_results["Z"].per_frame
    summary = r.soz_results["Z"].summary
    return dict(
        reproduced=bool(int(pf["entries"].iloc[0]) == 1),
        evidence=dict(entries_frame0=int(pf["entries"].iloc[0]), total_entries=int(pf["entries"].sum()),
                      entry_rate_per_frame=summary.get("entry_rate"), true_entry_events_in_window=0),
        explanation="a solvent already inside at frame 0 is counted as an entry event (no entry was observed)",
    )


def check_censoring_unflagged(tmp):
    pres = np.zeros((30, 1), bool); pres[0:5, 0] = True; pres[10:15, 0] = True; pres[25:30, 0] = True
    s = syn.make_scripted_system(pres, n_bulk=10, box=30.0, seed=6)
    d = Path(tmp) / "r5"; gro, xtc = s.write(d, "traj")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out")).run()
    ps = r.soz_results["Z"].per_solvent
    cols = list(ps.columns)
    has_flag = any("censor" in c.lower() for c in cols)
    return dict(reproduced=bool(not has_flag),
                evidence=dict(per_solvent_columns=cols, true_complete_segments=1, window_edge_segments=2),
                explanation="segments cut by the window edge are averaged with complete ones and not identified")


def check_duplicate_solvent_ids(tmp):
    pres = np.zeros((4, 2), bool); pres[:, :] = True
    s = syn.make_scripted_system(pres, n_bulk=2, box=30.0, seed=7)  # residues: LIG, 2 visitors, 2 bulk
    # resid wrap-around (PDB: 9999, GRO: 99999) makes NON-adjacent waters share (resname, resid, segid).
    # Adjacent duplicates would be merged into one residue by the GRO/PDB parsers, so use residue 4 vs 1.
    s.universe.residues[4].resid = s.universe.residues[1].resid
    d = Path(tmp) / "r6"; gro, xtc = s.write(d, "traj")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out")).run()
    ps = r.soz_results["Z"].per_solvent
    ids = ps["solvent_id"].tolist()
    return dict(reproduced=bool(len(ids) != len(set(ids))),
                evidence=dict(n_rows=len(ids), n_unique_ids=len(set(ids)), duplicated=sorted({i for i in ids if ids.count(i) > 1})[:4]),
                explanation="two different molecules share one solvent_id")


def check_broken_molecule_probe(tmp):
    """COM/COG probe on a bond-less topology with a molecule split across the periodic boundary."""
    s = syn.make_straddling_water_system(box=20.0, n_frames=3)
    d = Path(tmp) / "r7"; gro, xtc = s.write(d, "traj")
    res = {}
    for mode, sel in (("atom", "name OW"), ("cog", "name OW HW1 HW2"), ("com", "name OW HW1 HW2")):
        try:
            r = SOZAnalysisEngine(make_project(gro, xtc, d / f"out_{mode}", cutoff=3.5, probe=(sel, mode),
                                               analysis=AnalysisOptions(store_min_distances=False, workers=1))).run()
            res[mode] = r.soz_results["Z"].per_frame["n_solvent"].tolist()
        except Exception as exc:  # released code raises for bond-less topologies
            res[mode] = f"{type(exc).__name__}: {str(exc)[:90]}"
    truth = [1, 1, 1]
    bad = [m for m in ("cog", "com") if res[m] != truth]
    return dict(reproduced=bool(res["atom"] == truth and bad),
                evidence=dict(n_solvent_by_probe_mode=res, truth=truth, wrong_or_failing_modes=bad),
                explanation="COG/COM probes of a molecule split across the periodic boundary fail (crash or wrong zone membership) when the topology has no bonds")


def check_workers_is_noop(tmp):
    s = syn.make_ideal_gas_system(n_waters=900, n_frames=60, box=32.0, seed=9)
    d = Path(tmp) / "r8"; gro, xtc = s.write(d, "traj")
    import concurrent.futures as cf
    calls = {"n": 0}
    orig = cf.ProcessPoolExecutor.__init__

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    cf.ProcessPoolExecutor.__init__ = counting
    try:
        t0 = time.time()
        SOZAnalysisEngine(make_project(gro, xtc, d / "o1", analysis=AnalysisOptions(store_min_distances=False, workers=1))).run()
        t1 = time.time() - t0
        calls["n"] = 0
        t0 = time.time()
        SOZAnalysisEngine(make_project(gro, xtc, d / "o8", analysis=AnalysisOptions(store_min_distances=False, workers=8))).run()
        t8 = time.time() - t0
    finally:
        cf.ProcessPoolExecutor.__init__ = orig
    return dict(reproduced=bool(calls["n"] == 0),
                evidence=dict(process_pools_created_with_workers_8=calls["n"], wall_s_workers1=round(t1, 2), wall_s_workers8=round(t8, 2)),
                explanation="the per-frame SOZ loop is serial; 'workers' only affects the (stripped) H-bond path")


def check_density_ignores_global_window(tmp):
    s = syn.make_ideal_gas_system(n_waters=300, n_frames=40, box=24.0, seed=10)
    d = Path(tmp) / "r9"; gro, xtc = s.write(d, "traj")
    dens = DensityMapConfig(name="wat", species_selection="name OW", grid_spacing=1.5, padding=1.0,
                            align=True, align_selection="resname LIG", conditioning_policy="strict")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "o", density=[dens],
                                       analysis=AnalysisOptions(frame_start=0, frame_stop=None, stride=5, store_min_distances=False, workers=1))).run()
    meta = r.density_results["wat"].metadata
    n_soz = r.soz_results["Z"].summary["n_frames"]
    # origin depends on the frame the shared Universe was left on: vary only frame_stop of the SOZ loop
    origins = []
    dens2 = DensityMapConfig(name="wat", species_selection="name OW", grid_spacing=1.5, padding=1.0, frame_start=0, frame_stop=5,
                             stride=1, align=False, conditioning_policy="unsafe")
    for stop in (10, 11):
        rr = SOZAnalysisEngine(make_project(gro, xtc, d / f"o_{stop}", density=[dens2],
                                            analysis=AnalysisOptions(frame_start=0, frame_stop=stop, stride=1, store_min_distances=False, workers=1))).run()
        origins.append(np.round(rr.density_results["wat"].metadata["origin"], 3).tolist())
    return dict(reproduced=bool(meta.get("stride") != 5 or origins[0] != origins[1]),
                evidence=dict(soz_frames=n_soz, density_stride_used=meta.get("stride"), analysis_stride=5,
                              grid_origin_with_soz_stop_10=origins[0], grid_origin_with_soz_stop_11=origins[1]),
                explanation="density ignores the global stride and its grid extent depends on the last SOZ frame visited")


def check_cancel_returns_unmarked_partial(tmp):
    pres = np.zeros((20, 1), bool); pres[:, 0] = True
    s = syn.make_scripted_system(pres, n_bulk=5, box=30.0, seed=11)
    d = Path(tmp) / "r10"; gro, xtc = s.write(d, "traj")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 7

    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out")).run(cancel_flag=cancel)
    pf = r.soz_results["Z"].per_frame
    marked = bool(getattr(r, "cancelled", False)) or "cancel" in json.dumps(r.qc_summary, default=str).lower() \
        or "partial" in json.dumps(r.qc_summary, default=str).lower()
    return dict(reproduced=bool(len(pf) < 20 and not marked),
                evidence=dict(frames_returned=len(pf), frames_expected=20, result_marked_partial=marked),
                explanation="cancelled run returns normal-looking results; the GUI auto-exports them over earlier outputs")


def check_unknown_params_silent(tmp):
    pres = np.zeros((4, 1), bool); pres[:, 0] = True
    s = syn.make_scripted_system(pres, n_bulk=5, box=30.0, inside_r=4.5, outside_min=9.0, seed=12)
    d = Path(tmp) / "r11"; gro, xtc = s.write(d, "traj")
    node = SOZNode(type="shell", params={"selection_label": "sel", "distance": 5.0, "unit": "A"})  # key used in repo tests
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out", node=node)).run()
    qc = r.qc_summary
    msgs = qc.get("analysis_warnings", []) + qc.get("preflight", {}).get("warnings", [])
    warned = any(any(k in w.lower() for k in ("ignored", "unknown", "unrecognized", "unrecognised", "unused", "unsupported param", "defaulting", "default cutoff"))
                 for w in msgs)
    n = r.soz_results["Z"].per_frame["n_solvent"].tolist()
    return dict(reproduced=bool(not warned and n == [0, 0, 0, 0]),
                evidence=dict(n_solvent=n, intended_shell_radius_A=5.0, solvent_at_A=4.5, warned=warned, warning_messages=[m for m in msgs if "chain" not in m and "element" not in m][:4],
                              effective_cutoffs=[e.get("cutoffs") for e in qc.get("soz_definitions", {}).get("Z", [])]),
                explanation="'distance: 5.0' on a shell node is ignored and a 3.5 A default is used silently")


def check_cutoff_exceeds_half_box(tmp):
    pres = np.zeros((4, 1), bool); pres[:, 0] = True
    s = syn.make_scripted_system(pres, n_bulk=5, box=16.0, outside_min=7.0, seed=13)
    d = Path(tmp) / "r12"; gro, xtc = s.write(d, "traj")
    r = SOZAnalysisEngine(make_project(gro, xtc, d / "out", cutoff=12.0)).run()
    qc = r.qc_summary
    msgs = qc.get("analysis_warnings", []) + qc.get("preflight", {}).get("warnings", [])
    warned = any(("half" in m.lower() or "minimum image" in m.lower() or "minimum-image" in m.lower()) for m in msgs)
    return dict(reproduced=bool(not warned), evidence=dict(box_A=16.0, cutoff_A=12.0, warnings=msgs[:5]),
                explanation="cutoff larger than half the box silently breaks the minimum-image convention")


def check_gui_platform_forcing(tmp):
    code = ("import os\nos.environ.pop('QT_QPA_PLATFORM', None)\nimport app.main\nprint(os.environ.get('QT_QPA_PLATFORM'))\n")
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    env.setdefault("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    value = (p.stdout.strip().splitlines() or [""])[-1]
    return dict(reproduced=bool(value == "xcb" and sys.platform != "linux"),
                evidence=dict(QT_QPA_PLATFORM_after_import=value, platform=sys.platform, stderr_tail=p.stderr.strip()[-200:]),
                explanation="importing app.main forces the X11 'xcb' plugin, which does not exist on Windows/macOS")


CHECKS = [
    ("R01", "critical", "Stale density cache (path-only key)", check_stale_density_cache),
    ("R02", "critical", "Extraction against mutated inputs", check_extraction_mutated_inputs),
    ("R03", "critical", "Assumed single-frame / uniform timing", check_timing_assumptions),
    ("R04", "major", "Initial population counted as entries", check_initial_population_is_entries),
    ("R05", "major", "Window-edge (censored) residence segments unflagged", check_censoring_unflagged),
    ("R06", "major", "Non-unique solvent identifiers", check_duplicate_solvent_ids),
    ("R07", "critical", "COM/COG probe of PBC-split molecule", check_broken_molecule_probe),
    ("R08", "major", "'workers' has no effect on the SOZ loop", check_workers_is_noop),
    ("R09", "major", "Density ignores global window; grid extent depends on hidden state", check_density_ignores_global_window),
    ("R10", "major", "Cancelled run returns unmarked partial results", check_cancel_returns_unmarked_partial),
    ("R11", "major", "Unknown/missing SOZ node params silently defaulted", check_unknown_params_silent),
    ("R12", "major", "Cutoff > half box not detected", check_cutoff_exceeds_half_box),
    ("R13", "major", "GUI cannot start off-Linux (forced xcb)", check_gui_platform_forcing),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated check ids")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    tmp = tempfile.mkdtemp(prefix="sozlab_repro_")
    results = []
    print(f"engine under test: {Path(engine.__file__).parent}  label={args.label}")
    for cid, sev, title, fn in CHECKS:
        if only and cid not in only:
            continue
        t0 = time.time()
        try:
            r = fn(tmp)
            r.update(id=cid, severity=sev, title=title, seconds=round(time.time() - t0, 2))
        except Exception as exc:
            r = dict(id=cid, severity=sev, title=title, reproduced=None, error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc()[-1200:], seconds=round(time.time() - t0, 2))
        results.append(r)
        flag = {True: "FLAW PRESENT", False: "ok (not reproduced)", None: "ERROR"}[r["reproduced"]]
        print(f"[{cid}] {sev:8s} {flag:20s} {title}")
        if r["reproduced"] is None:
            print("      ", r["error"])
    shutil.rmtree(tmp, ignore_errors=True)
    payload = dict(label=args.label, engine=str(Path(engine.__file__).parent), mdanalysis=mda.__version__,
                   python=sys.version.split()[0], platform=sys.platform, results=results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("wrote", args.out)
    n_flaw = sum(1 for r in results if r["reproduced"])
    print(f"\n{n_flaw}/{len(results)} flaws present")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
