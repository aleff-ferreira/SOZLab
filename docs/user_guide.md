# SOZLab user guide

This guide follows the five steps shown at the top of the Workbench. Everything the Workbench does is also available
from the command line (last section); both use the same engine and give identical results.

Contents: [1 Import](#1-import) · [2 Define](#2-define-the-zone) · [3 Run](#3-run-settings) · [4 Inspect](#4-inspect) ·
[5 Export](#5-export) · [Recovering from problems](#recovering-from-problems) · [Command line](#command-line)

## 1. Import

Drop a **topology** (`.gro .pdb .psf .prmtop .tpr …`) and, optionally, a **trajectory** (`.xtc .trr .dcd .nc …`) on the
window, or use the folder menu (top right) → *Open topology / trajectory…*. The files are read where they are; nothing is copied.

The Import card tells you what SOZLab understood: number of atoms and frames, the time span and frame interval, the box,
and which residues it treats as solvent. Two warnings matter:

- **time stamps assumed** — the format stores no times (for example `.dcd` without a header time), so durations assume
  1 ps per frame. Set the real interval in *Run settings → Advanced → Frame interval*.
- **irregular time stamps** — the spacing between frames varies (restarts, concatenation). Durations still use the
  stored times; the survival function assumes the median interval.

No files at hand? The folder menu → **Load the demo system** creates a synthetic solute in water (not a real simulation).

## 2. Define the zone

A *solvent occupancy zone* is the set of solvent molecules whose probe atom (the water oxygen) lies within a **distance**
of any atom of a **selection**, using the minimum-image convention of each frame's periodic box.

- **Around** — one-click chips for the parts SOZLab detected (protein, ligand, ions…), or type any MDAnalysis selection,
  for example `protein and resid 10-20` or `resname LIG and not name H*`. The line under the box says how many atoms matched.
- **Distance** — the slider, the number box, and the red line on the histogram are the same control. The histogram
  shows where the solvent sits relative to the selection in the current frame; *Use the end of the first shell* jumps to
  the first minimum. The cumulative curve (right axis) gives the number of molecules inside any radius.
- The **3D view** shows the solute (coral), solvent inside the zone (bright teal), bulk solvent (dim), and with the frame
  slider which molecules just entered or left. Drag to rotate, scroll to zoom, double-click to refocus.
- The **occupancy preview** under the 3D view recomputes while you drag: it is computed on ~80 sampled frames, is marked
  *PREVIEW*, and is not a result.

*Advanced settings* in the zone card hold a second selection with its own distance (AND / OR), extra radii, re-evaluating the selection
every frame (needed for geometric selections such as `around 5 resname LIG`), the solvent residue names and probe atom,
and whether ions count as solvent.

**Check the distance against the box.** A cutoff larger than half the shortest box edge would overlap its own periodic
image; the Run card shows a warning, and the result would not mean what you intend.

## 3. Run settings

The left rail keeps your inputs and settings visible while you read results. The **Run** button is always at the bottom.

| Setting | Default | What it does |
|---|---|---|
| Use every n-th frame | 1 (larger for very long runs) | Sampling stride. Larger is faster but coarser in time; see *Sampling* in the results. |
| First / last frame | whole trajectory | Analysis window (*Advanced*). |
| Brief-exit tolerance (ps) | 0 (strict) | A molecule that leaves for less than this does not end its visit; defined in time, so it does not change with the stride. |
| Blocks for uncertainty | 5 | Contiguous blocks used for the standard errors of residence times and rates. |
| Report stride sensitivity | on | Re-computes the headline numbers at 2× and 4× coarser sampling. |
| CPU workers | auto | Processes start only when the run is long enough to benefit; results are bitwise identical to serial. |
| Compute a density map | off | Time-averaged solvent density around the selection, superposed on it every frame. |

Every *Advanced* setting shows a dot when it differs from its default, a ↺ button to reset it, and **Reset all**.
After a run, changing any setting shows *Settings changed since the last run*, so numbers on screen are never silently stale.

## 4. Inspect

**Tiles** show value ± standard error (SE), the unit, and a trust badge (green check, amber triangle). Click **?** for the
definition. The strip above the tiles repeats the settings that produced the numbers, plus the configuration hash.

| Tile | Meaning | Read it with |
|---|---|---|
| In zone | mean number of solvent molecules in the zone | SE accounts for autocorrelation |
| Occupied | fraction of frames with at least one molecule | |
| Residence τ | time a molecule stays: the integral of the survival function S(τ); hover for the number of visits behind it | depends on Δt; see *Sampling* |
| Entry rate | entry events per unit time | depends on Δt (re-crossings between frames are missed) |

**Tabs:** *Occupancy* (timeline with mean and 95 % interval, distribution), *Residence* (S(τ), visit counts and statistics, Kaplan–Meier),
*Entries / exits*, *Sampling* (how τ changes with the frame interval, with an experimental extrapolation to Δt → 0),
*Density* (three slices, ρ / g(r) / −ln g / SE, unreliable voxels hidden), *Tables* (per molecule, per visit, per frame).
Drag the cursor on the timeline to move the 3D view to that frame. The fold button at the right of the frame slider hides the 3D view to give the charts the full height.

**Trust** (right rail) lists every caveat the engine raised, in plain language, for example *"the residence time changes by
+32 % with 2× coarser sampling: quote it with its interval"*. If the list is empty, the numbers can be used as shown.

**Runs** keeps every run of the session. Change a setting, run again, and select an earlier run to overlay it on the
timeline and see what moved.

## 5. Export

*Save data only* writes CSV tables, a JSON summary, `provenance.json` (software versions, configuration hash, SHA-256 of the
input files, the environment), `DATA_DICTIONARY.md` (definitions of every column) and `manifest.json` (SHA-256 of every
file written). *Save data + report* (the main button) adds a readable report. *Verify files* re-hashes everything against the manifest.

*Extract frames* writes the frames where a rule holds (for example "the zone holds at least 2 molecules", at least 5
consecutive frames, bridging gaps of up to 2) to a new trajectory. Before writing, SOZLab checks that the input files
are the ones analysed and that every frame matches the fingerprint recorded at analysis time; if the trajectory was
modified, replaced or truncated since the run, extraction stops with an explanation instead of writing wrong frames.

## Recovering from problems

| What you see | Why | What to do |
|---|---|---|
| "Could not read … / do not describe the same system" | topology and trajectory have different atom counts | choose the topology the trajectory was written from |
| red *Run* blocked: "matches no atoms" | the selection selects nothing | click a detected chip, or fix the selection text |
| "None of the solvent names match" | the residue name of water differs (`HOH`, `TIP3`, `WAT`…) | press the *Use …* button shown with the message |
| cutoff exceeds half the box | the zone would overlap its periodic image | lower the distance, or use a larger system |
| "time stamps assumed" | no times in the file | set the frame interval under *Advanced* |
| tile with amber triangle | a caveat applies | open the *Trust* panel |
| run cancelled | you pressed *Cancel* | results are labelled PARTIAL and are never exported silently |
| extraction refused | inputs changed after the run | re-run the analysis on the current files |

## Command line

```bash
sozlab-cli run --project project.json --output results/ [--stride 5] [--workers 4] [--report]
sozlab-cli validate --project project.json              # check selections and zones on a few frames
sozlab-cli extract --from-results results/ --rule "n_solvent>=2" --min-run 5 --out frames/
sozlab-cli verify results/                               # manifest and input-identity check
sozlab-cli selftest                                      # synthetic known-answer checks
```

A project file is the JSON written by the folder menu → *Save project…* in the Workbench. It also holds the Workbench's own
settings, which the CLI ignores. You can write one by hand: `python examples/make_demo_data.py` produces a runnable example, and
`examples/gromacs_template_project.json` is a template. Projects from v2.0 open unchanged. Unknown or misspelled parameters
are reported as warnings, never silently replaced by defaults.
