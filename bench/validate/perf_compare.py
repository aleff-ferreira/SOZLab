"""Wall time and peak memory of the same analysis on two source trees (released v2.0.0 vs this branch).

Only APIs present in both trees are used, so the identical file runs against each::

    PYTHONPATH=.baseline/src python bench/validate/perf_compare.py --label v2.0.0 --data bench/out/tip3p --what soz
    PYTHONPATH=src           python bench/validate/perf_compare.py --label v2.1   --data bench/out/tip3p --what soz

``--what soz``      zone occupancy / residence of "water within 3.2 A of any Na+" over every frame (serial)
``--what density``  time-averaged water-oxygen density map around one Na+ (each tree's default density engine)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import engine  # noqa: E402
from engine.analysis import SOZAnalysisEngine  # noqa: E402
from engine.models import (AnalysisOptions, DensityMapConfig, InputConfig, OutputConfig, ProbeConfig, ProjectConfig,  # noqa: E402
                           SelectionSpec, SolventConfig, SOZDefinition, SOZNode)


def peak_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 2 ** 20
    except Exception:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--data", default="bench/out/tip3p")
    ap.add_argument("--what", choices=("soz", "density"), default="soz")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--equal-work", action="store_true", help="disable the statistics v2.1 adds (stride sensitivity) so both trees do the same work")
    ap.add_argument("--dump", default=None, help="write the per-frame n_solvent series (.npy) to compare trees frame by frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = Path(args.data)
    top, traj = str(data / "system.pdb"), str(data / "rep0.xtc")
    out_dir = tempfile.mkdtemp(prefix="sozlab_perf_")
    density = []
    anchor = "resname NA"
    if args.what == "density":
        anchor = "index 3202"  # placeholder replaced below
        import MDAnalysis as mda

        ion = int(mda.Universe(top).select_atoms("resname NA").indices[0])
        anchor = f"index {ion}"
        kw = dict(name="na_water", species_selection="resname HOH and name O", grid_spacing=0.5, align=True, align_selection=anchor)
        have = {f.name for f in dataclasses.fields(DensityMapConfig)}
        for k, v in (("region_selection", anchor), ("region_padding", 9.0)):  # new-tree options; v2.0.0 uses its own extent rules
            if k in have:
                kw[k] = v
        density = [DensityMapConfig(**kw)]
    project = ProjectConfig(
        inputs=InputConfig(topology=top, trajectory=traj),
        solvent=SolventConfig(water_resnames=["HOH"], probe=ProbeConfig(selection="name O", position="atom")),
        selections={"anchor": SelectionSpec(label="anchor", selection=anchor)},
        sozs=[SOZDefinition(name="shell", description="perf", root=SOZNode(
            type="distance", params={"selection_label": "anchor", "cutoff": 3.2, "unit": "A"}))] if args.what == "soz" else [],
        analysis=AnalysisOptions(stride=args.stride, workers=args.workers, store_min_distances=False,
                                 **({"stride_sensitivity": False} if args.equal_work and "stride_sensitivity" in {f.name for f in dataclasses.fields(AnalysisOptions)} else {})),
        outputs=OutputConfig(output_dir=out_dir, write_per_frame=True),
        density_maps=density,
    )
    base_mb = peak_mb()
    t0 = time.time()
    res = SOZAnalysisEngine(project).run()
    seconds = time.time() - t0
    rec = {"label": args.label, "version": getattr(engine, "__version__", "?"), "what": args.what, "workers": args.workers,
           "stride": args.stride, "seconds": round(seconds, 2), "peak_memory_mb": round(peak_mb(), 1),
           "memory_before_run_mb": round(base_mb, 1), "frames": int(res.qc_summary.get("frames_analysed", 0))}
    rec["equal_work"] = bool(args.equal_work)
    if args.what == "soz":
        import numpy as np

        z = next(iter(res.soz_results.values()))
        rec["mean_n_solvent"] = float(z.summary["mean_n_solvent"])
        rec["frames"] = int(len(z.per_frame))
        if args.dump:
            np.save(args.dump, z.per_frame["n_solvent"].to_numpy())
    print(json.dumps(rec))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
