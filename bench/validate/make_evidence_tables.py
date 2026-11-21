"""Turn the measured JSON of the flaw reproduction into the markdown table used by docs/evidence/README.md."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "docs" / "evidence"
b = json.load(open(root / "data" / "repro_baseline.json"))
f = json.load(open(root / "data" / "repro_fixed.json"))
fb = {r["id"]: r for r in f["results"]}
FIX = {
    "R01": "density cache keyed on a hash of the input content, the configuration and the algorithm version; array hash verified on load",
    "R02": "inputs verified by content and every frame by a fingerprint recorded at analysis time; changed or truncated files are refused; staged writes",
    "R03": "durations from stored time stamps (midpoint rule); a single frame has no duration; assumed time is flagged and can be overridden",
    "R04": "the first analysed frame has no events; the initial population is reported separately",
    "R05": "visits touching the window edge are flagged; complete-visit means, Kaplan-Meier and S(tau) are reported",
    "R06": "solvent identifiers are unique per molecule",
    "R07": "molecules are made whole before COM/COG; no bond information is needed",
    "R08": "frames are evaluated in worker processes and accumulated in order (bitwise equal to serial)",
    "R09": "density honours the global window; its grid is defined from the solute, not from hidden state",
    "R10": "partial results are flagged, labelled PARTIAL on screen, and exported only on request (to a separate folder)",
    "R11": "unknown or missing parameters are reported as validation warnings or errors",
    "R12": "cutoff is checked against half the shortest box width (CLI preflight and Workbench)",
    "R13": "no forced platform plugin; the Workbench runs and is tested on Windows",
}
lines = ["| ID | severity | flaw in v2.0.0 | v2.0.0, measured | v2.1.0, measured |", "|---|---|---|---|---|"]
for r in b["results"]:
    n = fb.get(r["id"], {})
    status = "**still present**" if n.get("reproduced") else "not reproduced"
    lines.append(f"| {r['id']} | {r['severity']} | {r['title']} | {'**reproduced**' if r['reproduced'] else 'not reproduced'}: {r['explanation']} | "
                 f"{status}; {FIX.get(r['id'], '')} |")
n_b = sum(r["reproduced"] for r in b["results"])
n_f = sum(r["reproduced"] for r in f["results"])
lines += ["", f"Reproduced on v2.0.0 (commit 3dcf498): {n_b}/{len(b['results'])}. Reproduced on v2.1.0: {n_f}/{len(f['results'])}. "
          "Each check compares against analytically known synthetic data or an independent numpy oracle (`bench/repro/repro_v2_flaws.py`)."]
out = "\n".join(lines)
(root / "repro_table.md").write_text(out + "\n", encoding="utf-8")
print(out[:300])
