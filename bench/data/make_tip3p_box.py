"""Generate a reproducible TIP3P water + NaCl reference trajectory with OpenMM.

Public software (OpenMM), public force field (Amber14 TIP3P, Joung-Cheatham ions), fixed seeds - nothing
sensitive. The *script and seeds* are the reproducible artefact; trajectories are written to ``--out`` and are
not committed (their SHA-256 is recorded in ``meta.json`` instead).

    python bench/data/make_tip3p_box.py --out bench/out/tip3p --box 3.2 --replicas 4 --ns 2.5

Protocol: build box with 0.25 M NaCl -> minimise -> 200 ps NPT equilibration (300 K, 1 bar) -> for every replica:
new velocity seed, 20 ps decorrelation, then NVT-ish production in the equilibrated box (barostat off, so the
box is constant and distances are unambiguous) writing XTC every ``--dt-out`` ps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit


def pick_platform():
    for name, props in (("OpenCL", {"Precision": "mixed"}), ("CUDA", {"Precision": "mixed"}), ("CPU", {})):
        try:
            return mm.Platform.getPlatformByName(name), props, name
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform available")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/out/tip3p")
    ap.add_argument("--box", type=float, default=3.2, help="cubic box edge, nm")
    ap.add_argument("--ionic", type=float, default=0.25, help="NaCl ionic strength, molar")
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--ns", type=float, default=2.5, help="production length per replica, ns")
    ap.add_argument("--dt-out", type=float, default=0.2, help="frame interval, ps")
    ap.add_argument("--seed", type=int, default=20241120)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    box = mm.Vec3(args.box, args.box, args.box) * unit.nanometer
    modeller.addSolvent(ff, model="tip3p", boxSize=box, ionicStrength=args.ionic * unit.molar,
                        positiveIon="Na+", negativeIon="Cl-")
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    n_atoms = system.getNumParticles()
    n_wat = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    n_na = sum(1 for r in modeller.topology.residues() if r.name == "NA")
    n_cl = sum(1 for r in modeller.topology.residues() if r.name == "CL")
    print(f"system: {n_atoms} atoms, {n_wat} waters, {n_na} Na+, {n_cl} Cl-", flush=True)

    platform, props, pname = pick_platform()
    temperature = 300 * unit.kelvin
    barostat = mm.MonteCarloBarostat(1 * unit.bar, temperature, 25)
    barostat.setRandomNumberSeed(args.seed)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(args.seed)
    sim = app.Simulation(modeller.topology, system, integrator, platform, props)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(temperature, args.seed)

    t0 = time.time()
    sim.step(100_000)  # 200 ps NPT
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    eq_box = st.getPeriodicBoxVectors()
    box_nm = float(eq_box[0][0].value_in_unit(unit.nanometer))
    density = n_wat * 18.01528 / 6.02214076e23 / ((box_nm * 1e-7) ** 3)  # g/cm^3 (waters only)
    print(f"equilibrated on {pname}: box {box_nm:.4f} nm, water mass density {density:.4f} g/cm3 "
          f"({time.time() - t0:.0f} s)", flush=True)

    # constant-volume production in the equilibrated box
    system.removeForce(system.getNumForces() - 1)  # barostat was added last
    positions = st.getPositions()
    pdb_path = out / "system.pdb"
    with open(pdb_path, "w") as f:
        app.PDBFile.writeFile(modeller.topology, positions, f, keepIds=True)
    meta = {"openmm": mm.version.full_version, "platform": pname, "seed": args.seed, "box_nm": box_nm,
            "n_atoms": n_atoms, "n_water": n_wat, "n_na": n_na, "n_cl": n_cl, "ionic_strength_M": args.ionic,
            "temperature_K": 300, "timestep_fs": 2.0, "frame_interval_ps": args.dt_out,
            "production_ns_per_replica": args.ns, "water_mass_density_g_cm3": density,
            "water_number_density_A3": n_wat / ((box_nm * 10.0) ** 3), "replicas": []}
    steps_out = int(round(args.dt_out / 0.002))
    n_steps = int(round(args.ns * 1000.0 / 0.002))
    for k in range(args.replicas):
        integrator2 = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        integrator2.setRandomNumberSeed(args.seed + 1 + k)
        sim2 = app.Simulation(modeller.topology, system, integrator2, platform, props)
        sim2.context.setPeriodicBoxVectors(*eq_box)
        sim2.context.setPositions(positions)
        sim2.context.setVelocitiesToTemperature(temperature, args.seed + 101 + k)
        sim2.step(10_000)  # 20 ps decorrelation (not saved)
        xtc = out / f"rep{k}.xtc"
        sim2.reporters.append(app.XTCReporter(str(xtc), steps_out))
        t1 = time.time()
        sim2.step(n_steps)
        for rep in list(sim2.reporters):
            if hasattr(rep, "_xtc"):
                pass
        del sim2
        speed = args.ns * 86400.0 / (time.time() - t1)
        meta["replicas"].append({"index": k, "file": xtc.name, "velocity_seed": args.seed + 101 + k,
                                 "integrator_seed": args.seed + 1 + k, "ns_per_day": round(speed, 1),
                                 "bytes": xtc.stat().st_size, "sha256": sha256(xtc)})
        print(f"replica {k}: {args.ns} ns in {time.time() - t1:.0f} s ({speed:.0f} ns/day)", flush=True)
    meta["system_pdb_sha256"] = sha256(pdb_path)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
