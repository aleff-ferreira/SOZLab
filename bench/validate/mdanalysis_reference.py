"""Zone membership on a public triclinic system against MDAnalysis' own periodic ``around`` selection.

Data: MDAnalysisTests ``adk_oplsaa.gro`` + ``adk_oplsaa.xtc`` (adenylate kinase in water and Na+, 47 681 atoms, 10 frames, a triclinic cell of
80.0 A with 60/60/90 degree angles).  For every frame the set of water molecules SOZLab puts in the zone "water oxygen within r of the
protein" is compared with ``select_atoms("name OW and around r protein")`` (MDAnalysis, periodic by default), molecule by molecule.

    python bench/validate/mdanalysis_reference.py --out docs/evidence/data/mdanalysis_reference.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MDAnalysis as mda  # noqa: E402
import numpy as np  # noqa: E402
from MDAnalysisTests import datafiles as df  # noqa: E402

from tip3p_validation import project  # noqa: E402
from engine.analysis import SOZAnalysisEngine  # noqa: E402
from engine.models import ProbeConfig, SolventConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/data/mdanalysis_reference.json")
    args = ap.parse_args()
    out = {"data": "MDAnalysisTests adk_oplsaa.gro + adk_oplsaa.xtc", "box_A_deg": [float(x) for x in mda.Universe(df.GRO, df.XTC).dimensions],
           "cutoffs": []}
    u = mda.Universe(df.GRO, df.XTC)
    out["atoms"], out["frames"], out["dt_ps"] = len(u.atoms), len(u.trajectory), float(u.trajectory.dt)
    for cutoff in (3.0, 4.0, 6.0):
        proj = project(df.GRO, df.XTC, tempfile.mkdtemp(prefix="sozlab_adk_"), cutoff, "protein")
        proj.solvent = SolventConfig(water_resnames=["SOL"], probe=ProbeConfig(selection="name OW", position="atom"))
        proj.analysis.store_ids = True
        res = SOZAnalysisEngine(proj).run()
        z = res.soz_results["shell"]
        rec_by_display = {rec.display_id: ri for ri, rec in res.solvent_records.items()}
        uu = mda.Universe(df.GRO, df.XTC)
        frames, mism_molecules, total = [], 0, 0
        for i, ts in enumerate(uu.trajectory):
            ref = set(int(r) for r in uu.select_atoms(f"name OW and around {cutoff} protein").resindices)
            row = z.per_frame.iloc[i]
            ids = [x for x in str(row["solvent_ids"]).split(";") if x]
            ours = set(rec_by_display[x] for x in ids if x in rec_by_display)
            diff = ref ^ ours
            mism_molecules += len(diff)
            total += len(ref)
            frames.append({"frame": i, "mdanalysis": len(ref), "sozlab": int(row["n_solvent"]), "symmetric_difference": len(diff)})
        out["cutoffs"].append({"cutoff_A": cutoff, "molecules_in_zone_total": total, "mismatching_molecules": mism_molecules,
                               "frames_with_difference": sum(f["symmetric_difference"] > 0 for f in frames), "per_frame": frames})
        print(f"cutoff {cutoff} A: {total} (frame, molecule) memberships, {mism_molecules} differ from MDAnalysis `around` "
              f"({sum(f['symmetric_difference'] > 0 for f in frames)} of {len(frames)} frames)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
