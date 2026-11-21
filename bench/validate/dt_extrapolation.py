"""Does the experimental dt -> 0 residence-time extrapolation work on real MD, not just on Brownian dynamics?

A 0.4 ns TIP3P + NaCl trajectory was written every 0.02 ps (20 000 frames), 10x finer than the sampling of ordinary
production runs.  The residence time of water in the Na+ first shell (r_c = 3.2 A) is measured

  * directly at every sampling interval dt = 0.02 ... 1.28 ps (the "truth" curve tau(dt), no extrapolation involved), and
  * by SOZLab's built-in extrapolation (linear in sqrt(dt) over 1x / 2x / 4x strides) started from coarser data.

    python bench/validate/dt_extrapolation.py --data bench/out/tip3p_fine --out docs/evidence/data/dt_extrapolation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tip3p_validation import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/out/tip3p_fine")
    ap.add_argument("--out", default="docs/evidence/data/dt_extrapolation.json")
    ap.add_argument("--r-c", type=float, default=3.2)
    args = ap.parse_args()
    data = Path(args.data)
    meta = json.load(open(data / "meta.json"))
    top, traj = str(data / "system.pdb"), str(data / "rep0.xtc")
    dt0 = float(meta["frame_interval_ps"])
    work = Path(tempfile.mkdtemp(prefix="sozlab_dt_"))
    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 64):
        sm = run(top, traj, work / f"s{stride}", args.r_c, "resname NA", stride=stride, sensitivity=True, blocks=5).soz_results["shell"].summary
        u = sm["uncertainty"].get("tau_survival_integral") or {}
        ext = sm.get("tau_extrapolation") or {}
        rows.append({"stride": stride, "dt_ps": dt0 * stride, "tau_ps": float(sm["residence"]["continuous"]["tau_survival_integral"]),
                     "tau_se_ps": float(u.get("se", np.nan)), "entry_rate_per_ps": float(sm["entry_rate_per_time"]),
                     "internal_extrapolation_ps": float(ext["tau_dt0"]) if ext.get("valid") else None})
        print(rows[-1], flush=True)
    dts = np.array([r["dt_ps"] for r in rows])
    taus = np.array([r["tau_ps"] for r in rows])
    # reference: linear fit in sqrt(dt) through the finest sampled points (dt <= 0.16 ps), no coarse data involved
    fine = dts <= 0.17
    slope, intercept = np.polyfit(np.sqrt(dts[fine]), taus[fine], 1)
    out = {"r_c_A": args.r_c, "frame_interval_ps": dt0, "frames": int(meta.get("production_ns_per_replica", 0) * 1000 / dt0),
           "direct": rows, "fine_fit_sqrt_dt": {"intercept_ps": float(intercept), "slope_ps_per_sqrt_ps": float(slope),
                                                "points": int(fine.sum())}, "tau_at_finest_dt_ps": float(taus[0])}
    # extrapolation started from coarser data: what SOZLab reports when only dt = b, 2b, 4b exist
    tests = []
    for r in rows:
        if r["internal_extrapolation_ps"] is not None:
            tests.append({"base_dt_ps": r["dt_ps"], "uses_dt_ps": [r["dt_ps"], 2 * r["dt_ps"], 4 * r["dt_ps"]],
                          "extrapolated_tau0_ps": r["internal_extrapolation_ps"],
                          "relative_to_finest_direct": r["internal_extrapolation_ps"] / taus[0] - 1.0,
                          "relative_to_fine_fit_intercept": r["internal_extrapolation_ps"] / intercept - 1.0,
                          "direct_tau_at_base_dt_relative_to_finest": r["tau_ps"] / taus[0] - 1.0})
    out["extrapolation_tests"] = tests
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out["fine_fit_sqrt_dt"]), "finest direct:", taus[0])
    for t in tests:
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
