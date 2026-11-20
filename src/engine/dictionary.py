"""Definitions of every quantity SOZLab reports: what it measures, units, when to trust it.

One source of truth for the exported ``DATA_DICTIONARY.md``, the report appendix, the
documentation, and the in-context help of the GUI.
"""
from __future__ import annotations

from typing import Dict, List

#: key -> definition. ``trust`` states the condition under which the number is meaningful.
DEFINITIONS: Dict[str, Dict[str, str]] = {
    "zone": {
        "title": "Solvent occupancy zone (SOZ)",
        "unit": "",
        "text": "The set of solvent molecules whose probe (usually the water oxygen) lies within a cut-off distance "
                "of any atom of a selection, combined by AND / OR / NOT. Distances use the minimum-image convention "
                "of the periodic box of each frame.",
        "trust": "Cut-offs must be smaller than half the shortest box width; selections must be what you intend at "
                 "every frame (geometric selections such as 'around' are evaluated once unless marked updating).",
    },
    "n_solvent": {
        "title": "n_solvent",
        "unit": "molecules",
        "text": "Number of solvent molecules inside the zone in one analysed frame.",
        "trust": "Exact for each frame; its time average is reliable once the run spans many correlation times "
                 "(see mean_n_solvent uncertainty).",
    },
    "occupancy_fraction": {
        "title": "Occupancy fraction",
        "unit": "fraction of frames",
        "text": "Fraction of analysed frames with n_solvent >= 1.",
        "trust": "Independent of the frame stride when frames are subsampled uniformly; the standard error accounts "
                 "for autocorrelation (integrated autocorrelation time).",
    },
    "occupancy_pct": {
        "title": "Per-molecule occupancy",
        "unit": "% of frames",
        "text": "Percentage of analysed frames in which that molecule was inside the zone.",
        "trust": "Reliable when the run is long compared with the molecule's exchange time; short runs overstate "
                 "the occupancy of molecules that happened to be present at the start.",
    },
    "entries_exits": {
        "title": "Entries / exits",
        "unit": "events",
        "text": "A molecule absent at the previous analysed frame and present now is an entry (the reverse is an exit). "
                "The first analysed frame has no predecessor, so it has no events (v2.0 counted the initial population "
                "as entries).",
        "trust": "Entries between frames are counted, entries that start and end between two frames are missed: "
                 "the count depends on the frame interval (boundary re-crossings). Compare rates only at equal "
                 "stride; see stride_sensitivity.",
    },
    "entry_rate_per_time": {
        "title": "Entry rate",
        "unit": "events per ps",
        "text": "Observed entry events divided by the time between the first and last analysed frame.",
        "trust": "Sampling-interval dependent (see above). Requires trustworthy time stamps.",
    },
    "visit": {
        "title": "Visit (continuous residence)",
        "unit": "frames / ps",
        "text": "One unbroken stay of a molecule inside the zone. Its duration uses the midpoint rule on the actual time "
                "stamps: each analysed frame represents the interval halfway to its neighbours, so a stay covering k "
                "evenly spaced frames lasts k x dt.",
        "trust": "Visits touching the first or last analysed frame are censored (true duration unknown): they are "
                 "flagged (censored_left / censored_right) and excluded from 'complete' averages.",
    },
    "visit_intermittent": {
        "title": "Intermittent visit",
        "unit": "frames / ps",
        "text": "Visits separated by at most the gap tolerance are merged ('sustained residence', Impey et al. 1983). "
                "The tolerance may be given in time, which keeps the definition independent of the stride.",
        "trust": "A tolerance comparable to the sampling interval removes most boundary flicker; large tolerances hide "
                 "real exchange.",
    },
    "survival": {
        "title": "Survival correlation function S(tau)",
        "unit": "dimensionless",
        "text": "Probability that a molecule inside the zone at a random time origin stays inside, without leaving, "
                "for the next tau (Impey et al. 1983; Rapaport 1983). Averages over all time origins and molecules.",
        "trust": "Needs a window much longer than the decay time; the tail is noisy where few origins remain "
                 "(see origins_* columns).",
    },
    "tau_survival_integral": {
        "title": "Residence time from S(tau) (integral)",
        "unit": "ps",
        "text": "Integral of S(tau) from 0 to the point where it has decayed (trapezoid rule). The mean forward "
                "residence time seen from a random origin.",
        "trust": "A lower bound if S has not decayed by the end of the window (flagged). Depends on the sampling "
                 "interval for diffusing solvent (stride_sensitivity).",
    },
    "tau_survival_fit": {
        "title": "Residence time from S(tau) (exponential fit)",
        "unit": "ps",
        "text": "Time constant of a single-exponential fit of S(tau) over its well sampled early part (weighted by "
                "origins * S^2).",
        "trust": "Meaningful when the fit r2 is high (mono-exponential decay); otherwise a characteristic time only.",
    },
    "km": {
        "title": "Kaplan-Meier visit duration",
        "unit": "ps",
        "text": "Restricted mean of visit durations estimated with right censoring (Kaplan & Meier 1958); visits that "
                "start at the first frame are excluded because their start is unknown.",
        "trust": "Removes most of the bias of the naive mean of complete visits; the restricted mean is capped at the "
                 "longest observed duration.",
    },
    "uncertainty": {
        "title": "Standard errors and 95% intervals",
        "unit": "",
        "text": "Means: integrated autocorrelation time with Sokal's window; se = std * sqrt(g / N). Residence times "
                "and rates: N contiguous blocks treated as independent replicas, Student-t interval.",
        "trust": "Valid when blocks are longer than the correlation time; the block length and converged fraction "
                 "are reported. Errors from force field and finite simulation length are NOT included.",
    },
    "stride_sensitivity": {
        "title": "Stride sensitivity",
        "unit": "relative change",
        "text": "The headline numbers recomputed from every 2nd and 4th analysed frame (no re-reading of the "
                "trajectory). Large changes mean the quantity depends on the sampling interval.",
        "trust": "Occupancy should not change; residence times and entry rates of diffusing solvent change with the "
                 "interval because discrete sampling misses boundary re-crossings (continuity correction "
                 "0.5826 x sqrt(2 D dt)).",
    },
    "density": {
        "title": "Solvent number density rho(r)",
        "unit": "atoms / A^3",
        "text": "Mean number of probe atoms per A^3 at position r of the solute-fixed frame, averaged over the frames of "
                "the window. Each frame: solute made whole -> solvent wrapped to the image nearest the solute -> "
                "optimal superposition on the reference -> binning.",
        "trust": "Reliable only inside the fully covered radius (half the shortest box width) and where the standard "
                 "error is small; both are exported as masks.",
    },
    "g": {
        "title": "Relative density g(r)",
        "unit": "dimensionless",
        "text": "rho(r) / rho_bulk, with rho_bulk estimated from far-field voxels (>= 8 A from the solute) or <N>/<V>, "
                "or supplied. g = 1 in bulk.",
        "trust": "Depends on the bulk estimate (reported with both candidates). -ln g is a potential of mean force "
                 "in kT only where g is converged.",
    },
    "frame_fp": {
        "title": "Frame fingerprint",
        "unit": "",
        "text": "64-bit checksum of the coordinates of ~128 evenly spread atoms and the box of the frame as read during "
                "analysis. Extraction re-computes it to prove it writes the frames that were analysed.",
        "trust": "Detects changed or reordered trajectories; it is not a cryptographic hash of the whole frame.",
    },
}

#: file -> list of (column, description)
FILES: Dict[str, List[tuple]] = {
    "soz_<name>/per_frame.csv": [
        ("frame", "trajectory frame index (0-based) of the analysed frame"),
        ("time", "time stamp of the frame in ps (see timing in summary.json for its reliability)"),
        ("n_solvent", "solvent molecules in the zone"),
        ("solvent_ids", "';'-joined unique solvent identifiers present (resname:resid:segid, '#resindex' appended when "
                        "resid wraps)"),
        ("entries", "molecules that entered since the previous analysed frame (0 for the first frame)"),
        ("exits", "molecules that left since the previous analysed frame (0 for the first frame)"),
        ("frame_fp", "frame fingerprint (see definitions)"),
        ("<selection label>", "minimum distance (A) from that selection to the in-zone probes, NaN when the zone is empty"),
    ],
    "soz_<name>/per_solvent.csv": [
        ("solvent_id, resindex, resname, resid, segid", "identity of the molecule (resindex is unique)"),
        ("frames_present, occupancy_pct", "analysed frames inside the zone and the percentage"),
        ("first_seen_frame, last_seen_frame", "trajectory frames of the first/last presence"),
        ("entries", "observed entry events (the initial presence is not an entry)"),
        ("mean_res_time_cont, median_res_time_cont", "mean/median duration of all visits in ps (0 if never inside); includes censored visits"),
        ("mean_res_time_inter, median_res_time_inter", "same for intermittent visits"),
        ("present_at_start, present_at_end", "inside the zone at the first / last analysed frame"),
        ("n_visits, n_visits_complete, n_visits_censored", "visits, those not cut by the window edges, and those that are (censored)"),
        ("mean_res_time_cont_complete", "mean duration of complete visits (NaN if none)"),
        ("max_res_time_cont, time_in_zone", "longest visit and total time inside, ps"),
    ],
    "soz_<name>/visits.csv": [
        ("solvent_id, resindex", "molecule"),
        ("start_frame, end_frame, start_time, end_time", "first and last analysed frame of the visit"),
        ("duration", "midpoint-rule duration, ps"),
        ("censored_left, censored_right", "visit touches the first / last analysed frame"),
    ],
    "soz_<name>/survival.csv": [
        ("lag_samples, tau", "lag in analysed frames and in ps"),
        ("S_continuous, S_intermittent", "survival correlation function for continuous / intermittent presence"),
        ("origins_continuous, origins_intermittent", "number of (molecule, origin) pairs contributing at that lag"),
    ],
    "soz_<name>/summary.json": [
        ("occupancy_fraction, mean_n_solvent, ...", "see definitions; 'uncertainty' holds standard errors and intervals"),
        ("residence", "visit statistics, censoring counts, survival-based residence times"),
        ("stride_sensitivity", "headline numbers at 1x, 2x and 4x coarser sampling"),
        ("timing", "source and regularity of the time axis; 'trusted' is false for assumed or single-frame timing"),
    ],
    "density_map_<name>/": [
        ("<name>.dx", "number density in atoms/A^3 on the grid (origin = centre of the first voxel)"),
        ("grid.npz", "arrays: grid (density), counts, se (standard error), g (relative density), reliable (mask), x, y, z"),
        ("metadata.json", "grid definition, normalisation (bulk estimates), coverage and drift QC, cache status"),
    ],
    "provenance.json": [
        ("inputs", "path, size and SHA-256 (or sampled SHA-256) of topology, trajectory and alignment references"),
        ("config_sha256, algorithm_versions", "hash of the scientific configuration; definition revisions"),
        ("environment", "SOZLab, Python and package versions, platform"),
    ],
    "manifest.json": [("files", "size and SHA-256 of every file written by this export")],
}


def data_dictionary_markdown(version: str, schema: str) -> str:
    lines: List[str] = [
        "# SOZLab data dictionary",
        "",
        f"SOZLab {version} - export schema {schema}. All distances are in angstrom, times in picoseconds unless stated.",
        "Columns added in schema 2.1 never replace earlier ones; every v2.0 column keeps its name.",
        "",
        "## Quantities",
        "",
    ]
    for key, d in DEFINITIONS.items():
        unit = f" ({d['unit']})" if d["unit"] else ""
        lines += [f"### {d['title']}{unit}", "", d["text"], "", f"*When to trust it:* {d['trust']}", ""]
    lines += ["## Files", ""]
    for fname, cols in FILES.items():
        lines += [f"### `{fname}`", ""]
        lines += [f"- **{c}** - {desc}" for c, desc in cols]
        lines.append("")
    lines += [
        "## References",
        "",
        "- Impey, Madden, McDonald. J. Phys. Chem. 87, 5071 (1983) - hydration shells, sustained residence.",
        "- Rapaport. Mol. Phys. 50, 1151 (1983) - continuous-presence correlation functions.",
        "- Flyvbjerg, Petersen. J. Chem. Phys. 91, 461 (1989); Grossfield et al. LiveCoMS 1, 5067 (2018) - uncertainty.",
        "- Kaplan, Meier. J. Am. Stat. Assoc. 53, 457 (1958) - censored data.",
        "- Broadie, Glasserman, Kou. Math. Finance 7, 325 (1997) - discrete-monitoring correction.",
        "",
    ]
    return "\n".join(lines)
