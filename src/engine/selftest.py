"""Self-test: known-answer checks users can run on their own installation (``sozlab-cli selftest``).

Every check builds a synthetic system whose correct answer is known analytically or by construction
(see :mod:`engine.synthetic`), runs the real engine on files written to a temporary directory, and compares.
A failing check means this installation (MDAnalysis/numpy build, platform, a local modification) does *not*
reproduce the validated behaviour and its results should not be trusted until that is understood.
"""
from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from engine import synthetic as syn
from engine._version import __version__
from engine.analysis import SOZAnalysisEngine
from engine.models import (
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


def _project(gro, xtc, out, cutoff=3.5, analysis=None, density=None, probe=("name OW", "atom")) -> ProjectConfig:
    return ProjectConfig(
        inputs=InputConfig(topology=str(gro), trajectory=str(xtc)),
        solvent=SolventConfig(water_resnames=["SOL"], probe=ProbeConfig(selection=probe[0], position=probe[1])),
        selections={"sel": SelectionSpec(label="sel", selection="resname LIG")},
        sozs=[SOZDefinition("Z", "selftest", SOZNode("distance", {"selection_label": "sel", "cutoff": cutoff, "unit": "A"}))],
        analysis=analysis or AnalysisOptions(store_min_distances=False, workers=1),
        outputs=OutputConfig(output_dir=str(out), write_per_frame=True),
        density_maps=list(density or []),
    )


def _check_occupancy_and_events(tmp: Path) -> str:
    pres = np.zeros((16, 2), bool)
    pres[0:4, 0] = True
    pres[8:12, 0] = True
    pres[5:7, 1] = True
    gro, xtc = syn.make_scripted_system(pres, n_bulk=10, box=30.0, dt=2.0, seed=1).write(tmp, "a")
    r = SOZAnalysisEngine(_project(gro, xtc, tmp / "oa")).run()
    z = r.soz_results["Z"]
    assert z.per_frame["n_solvent"].tolist() == pres.sum(axis=1).tolist(), "n_solvent differs from the scripted truth"
    assert z.summary["n_entry_events"] == 2 and z.summary["initial_population"] == 1, "entry accounting"
    assert sorted(z.visits["duration"]) == [4.0, 8.0, 8.0], "visit durations"
    assert int(z.visits["censored_left"].sum()) == 1, "left censoring flag"
    return "occupancy, entries (initial population excluded), visit durations, censoring"


def _check_irregular_time(tmp: Path) -> str:
    times = np.array([0, 1, 2, 3, 4, 104, 204, 304, 404, 405, 406, 407], float)
    pres = np.zeros((len(times), 1), bool)
    pres[5:9, 0] = True
    gro, xtc = syn.make_scripted_system(pres, n_bulk=15, box=30.0, times=times, seed=4).write(tmp, "b")
    r = SOZAnalysisEngine(_project(gro, xtc, tmp / "ob")).run()
    truth = (times[8] + times[9]) / 2 - (times[4] + times[5]) / 2
    got = float(r.soz_results["Z"].visits["duration"].iloc[0])
    assert abs(got - truth) < 1e-6, f"duration {got} ps, expected {truth} ps (midpoint rule)"
    return f"irregular time stamps: {got:.1f} ps (v2.0 reported 4.0 ps)"


def _check_split_molecule(tmp: Path) -> str:
    gro, xtc = syn.make_straddling_water_system().write(tmp, "c")
    for mode, sel in (("atom", "name OW"), ("cog", "name OW HW1 HW2"), ("com", "name OW HW1 HW2")):
        r = SOZAnalysisEngine(_project(gro, xtc, tmp / f"oc_{mode}", probe=(sel, mode))).run()
        assert r.soz_results["Z"].per_frame["n_solvent"].tolist() == [1, 1, 1], f"{mode} probe missed a split molecule"
    return "molecule split across the periodic boundary found with atom/COG/COM probes (no bonds)"


def _check_ideal_gas(tmp: Path) -> str:
    n, box, r_cut = 600, 28.0, 4.0
    system = syn.make_ideal_gas_system(n_waters=n, n_frames=250, box=box, seed=3, n_solute_atoms=4)
    gro, xtc = system.write(tmp, "d")
    cfg = DensityMapConfig(name="w", species_selection="name OW", grid_spacing=2.0, align=True,
                           align_selection="resname LIG", grid_size=[20, 20, 20], conditioning_policy="strict")
    r = SOZAnalysisEngine(_project(gro, xtc, tmp / "od", cutoff=r_cut, density=[cfg])).run()
    # exact check: independent float64 minimum-image oracle on the coordinates read back from the file
    import MDAnalysis as mda

    u = mda.Universe(str(gro), str(xtc))
    exact = []
    for ts in u.trajectory:
        pos = u.atoms.positions.astype(np.float64)
        dd = pos[4::3][:, None, :] - pos[:4][None, :, :]
        dd -= np.round(dd / box) * box
        exact.append(int((np.linalg.norm(dd, axis=2).min(axis=1) <= r_cut).sum()))
    assert r.soz_results["Z"].per_frame["n_solvent"].tolist() == exact, "per-frame counts differ from the numpy oracle"
    # statistical sanity: the zone is the union of spheres around the 4 solute atoms; volume fraction by Monte Carlo

    centre = np.full(3, box / 2) + syn.solute_template(4)
    pts = np.random.default_rng(0).uniform(0, box, size=(400_000, 3))
    d = pts[:, None, :] - centre[None, :, :]
    d -= np.round(d / box) * box
    p = float((np.linalg.norm(d, axis=2).min(axis=1) <= r_cut).mean())
    s = r.soz_results["Z"].summary
    se = s["uncertainty"]["mean_n_solvent"]["se"]
    assert abs(s["mean_n_solvent"] - n * p) < 4 * se + 0.02, "mean occupancy is not N * V_zone / V_box"
    d = r.density_results["w"]
    assert abs(d.grid.mean() / (n / box**3) - 1) < 0.03, "density map is not uniform N/V"
    return f"per-frame counts equal the numpy oracle; mean {s['mean_n_solvent']:.2f} vs {n * p:.2f} expected; uniform density N/V"


def _check_parallel_identity(tmp: Path) -> str:
    system = syn.make_ideal_gas_system(n_waters=250, n_frames=200, box=24.0, seed=5, n_solute_atoms=4)
    gro, xtc = system.write(tmp, "e")
    out = {}
    for w in (1, 2):
        opts = AnalysisOptions(workers=w, parallel_min_seconds=0.0, store_min_distances=True)
        out[w] = SOZAnalysisEngine(_project(gro, xtc, tmp / f"oe{w}", analysis=opts)).run(use_cache=False)
    assert out[2].qc_summary["parallel"]["mode"] == "parallel", "worker processes were not started"
    a, b = out[1].soz_results["Z"], out[2].soz_results["Z"]
    assert a.per_frame.equals(b.per_frame) and a.per_solvent.equals(b.per_solvent) and a.visits.equals(b.visits), \
        "serial and parallel results differ"
    return "1 worker and 2 worker processes give identical tables"


def _check_stale_inputs(tmp: Path) -> str:
    from engine.extract import ExtractionMismatchError, select_from_result, write_extracted_trajectory
    import MDAnalysis as mda

    cfg = DensityMapConfig(name="w", species_selection="name OW", grid_spacing=2.0, align=True,
                           align_selection="resname LIG", grid_size=[12, 12, 12], conditioning_policy="strict")
    a = syn.make_ideal_gas_system(n_waters=200, n_frames=24, box=22.0, seed=1, n_solute_atoms=4)
    b = syn.make_ideal_gas_system(n_waters=200, n_frames=24, box=22.0, seed=2, n_solute_atoms=4)
    gro, xtc = a.write(tmp, "f")
    project = _project(gro, xtc, tmp / "of", density=[cfg])
    first = SOZAnalysisEngine(project).run()
    assert SOZAnalysisEngine(project).run().density_results["w"].metadata["cache"]["hit"], "valid cache not reused"
    sel = select_from_result(first, "Z", "n_solvent>=1")
    b.write(tmp, "f")  # same path, different content
    again = SOZAnalysisEngine(project).run()
    assert not again.density_results["w"].metadata["cache"]["hit"], "STALE density cache was reused"
    try:
        write_extracted_trajectory(mda.Universe(str(gro), str(xtc)), sel, str(tmp / "ef"))
    except ExtractionMismatchError:
        return "density cache invalidated and extraction refused after the trajectory was replaced"
    raise AssertionError("extraction wrote frames from a replaced trajectory")


def _check_brownian_survival(tmp: Path) -> str:
    R, D, DT, L = 4.0, 0.25, 0.5, 36.0
    system = syn.make_brownian_system(n_waters=700, n_frames=2500, box=L, diffusion=D, dt=DT, seed=7)
    gro, xtc = system.write(tmp, "g")
    r = SOZAnalysisEngine(_project(gro, xtc, tmp / "og", cutoff=R, analysis=AnalysisOptions(
        store_min_distances=False, workers=1, store_ids=False))).run()
    surv = r.soz_results["Z"].survival
    lags = np.arange(1, 13)
    measured = surv["S_continuous"].to_numpy()[lags]
    theory = syn.sphere_survival_discrete(lags * DT, R, D, DT)
    dev = float(np.abs(measured - theory).max())
    assert dev < 0.06, f"survival curve deviates from analytic theory by {dev:.3f}"
    return f"survival correlation function within {dev:.3f} of the analytic Brownian-sphere solution"


CHECKS: List[Tuple[str, Callable[[Path], str]]] = [
    ("occupancy / events / censoring", _check_occupancy_and_events),
    ("irregular time stamps", _check_irregular_time),
    ("periodic geometry", _check_split_molecule),
    ("ideal-gas occupancy and density", _check_ideal_gas),
    ("parallel == serial", _check_parallel_identity),
    ("stale cache / extraction", _check_stale_inputs),
    ("Brownian survival function", _check_brownian_survival),
]


def run_selftest(verbose: bool = True) -> int:
    import MDAnalysis as mda

    if verbose:
        print(f"SOZLab {__version__} self-test (MDAnalysis {mda.__version__}, numpy {np.__version__})")
    root = Path(tempfile.mkdtemp(prefix="sozlab_selftest_"))
    failures = 0
    try:
        for name, fn in CHECKS:
            t0 = time.time()
            sub = root / name.replace(" ", "_").replace("/", "_")
            sub.mkdir(parents=True, exist_ok=True)
            try:
                detail = fn(sub)
                status = "PASS"
            except Exception as exc:
                failures += 1
                status = "FAIL"
                detail = f"{type(exc).__name__}: {exc}"
                if verbose:
                    traceback.print_exc()
            if verbose:
                print(f"  [{status}] {name:<36s} {time.time() - t0:5.1f}s  {detail}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if verbose:
        print("All checks passed." if not failures else f"{failures} check(s) FAILED: do not trust results from this installation.")
    return 1 if failures else 0
