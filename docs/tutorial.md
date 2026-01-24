# SOZLab Tutorial

## What SOZLab does
SOZLab is a Linux GUI + CLI for solvent occupancy zone (SOZ) analysis of MD trajectories. It evaluates, per frame, which solvent residues satisfy a user-defined logic tree around one or more seed selections and reports occupancy, entry/exit events, residence statistics, plots, and extractable frame sets.

### Operational definition of a SOZ
A SOZ is the set of solvent residues that satisfy a logic tree in a given frame. The evaluated set is residue-based (resindices/resids), not atoms. All occupancy counts and events refer to solvent residues, which avoids atom-count bias for multi-atom solvents.

### Per-frame metrics (definitions)
- `frame`: zero-based trajectory frame index after applying `analysis.frame_start`, `analysis.frame_stop`, and `analysis.stride`.
- `time`: `ts.time` from MDAnalysis for the frame. Values are written to CSV as-is.
- `n_solvent`: integer count of solvent residues in the SOZ for that frame.
- `entries`: number of solvent residues that appear relative to the previous sampled frame.
- `exits`: number of solvent residues that disappear relative to the previous sampled frame.

### Stride effects (critical)
Entries and exits are computed between sampled frames. If `analysis.stride` is large, short-lived events can be missed and entry/exit counts only reflect changes between sampled frames.

### Units and time bases
- Length cutoffs accept `unit: "A"` or `unit: "nm"`. Internally, SOZLab converts to angstroms (1 nm = 10 A).
- Time comes directly from MDAnalysis. The GUI divides `time` by 1000 and labels it in ns (assumes ps input). If your trajectory time units are not ps, confirm the unit before interpreting plots.
- GROMACS conventions use nm for length and ps for time. Always validate your expectations when switching between toolchains.

## Quick start

### GUI quick start (sample project)
1) Launch: `sozlab-gui`.
2) Click **Load** (header bar) and open `examples/sample_project.json`.
3) In the left drawer, review **Inputs**, **Analysis Window**, and **Output Settings**.
4) Click **Run**. The status bar shows progress and a log path.
5) Use the stepper tabs: **Project**, **Define**, **QC**, **Explore**, **Export**.
6) In **Explore**, inspect the Timeline and plots. Use **Export Data** or **Export Report** when ready.

### CLI quick start (sample project)
```bash
sozlab-cli run --project examples/sample_project.json --output out --progress --report
```

Validation (fast sanity check):
```bash
sozlab-cli validate --project examples/sample_project.json --max-frames 200
```

Extraction example:
```bash
sozlab-cli extract --project examples/sample_project.json --soz shell_only --rule "n_solvent>=1" --min-run 5 --gap 1 --out out/extracted
```

### Example projects
- `examples/sample_project.json`: minimal example.
- `examples/gromacs_template_project.json`: GROMACS-oriented template.
- `examples/soz_configs/`: individual SOZ definitions for Advanced JSON.

## CLI reference (current flags)

### Run
```text
sozlab-cli run --project PROJECT [--output OUTPUT] [--stride STRIDE] [--start START] [--stop STOP] [--no-ids] [--no-per-frame] [--progress] [--report]
```
- `--project`: project JSON file (required).
- `--output`: output directory for results and logs.
- `--stride`: frame stride override.
- `--start`: start frame index.
- `--stop`: stop frame index (exclusive).
- `--no-ids`: disable per-frame solvent IDs.
- `--no-per-frame`: disable per-frame table export.
- `--progress`: show progress bar in terminal.
- `--report`: generate report after analysis.

### Validate
```text
sozlab-cli validate --project PROJECT [--max-frames MAX_FRAMES]
```
- `--project`: project JSON file (required).
- `--max-frames`: maximum frames to validate (default 200).

### Extract
```text
sozlab-cli extract --project PROJECT [--soz SOZ] [--rule RULE] [--min-run MIN_RUN] [--gap GAP] [--out OUT] [--format FORMAT] [--prefix PREFIX] [--output OUTPUT]
```
- `--project`: project JSON file (required).
- `--soz`: SOZ name to filter.
- `--rule`: rule string (e.g., `n_solvent>=1`).
- `--min-run`: minimum consecutive frames.
- `--gap`: gap tolerance to merge runs.
- `--out`: extraction output directory.
- `--format`: trajectory format (default `xtc`).
- `--prefix`: output filename prefix (default `extracted`).
- `--output`: override output dir for logs only.

## Output directory behavior
- `outputs.output_dir` is the base directory for logs and exported results.
- GUI **Output Settings** edits `outputs.output_dir`. Export Data/Report prompts for a directory and updates this value.
- CLI `run --output OUT` overrides `outputs.output_dir` for that run.
- CLI `extract --output OUT` overrides only the log directory; `extract --out OUT_EXTRACT` controls extraction outputs.

## UI overview (layout and navigation)

### Header bar
- Stepper: **Project**, **Define**, **QC**, **Explore**, **Export** (switches main pages).
- Actions: **Load**, **Save**, **New SOZ**, **Quick**, **Run**, **Cancel**, **Export Data**, **Export Report**.
- Toggles: **Console** (diagnostics console), **Drawer** (left panel), **Inspector** (right panel).
- UI controls: **Theme** (Light/Dark), **Scale**, **Presentation** (larger UI, hides drawers).
- Shortcuts: see the Appendix.
- Command palette (`Ctrl+K`) opens a searchable list of common actions.

Action behavior:
- **Load** opens a project JSON/YAML file.
- **Save** writes the current project to JSON/YAML (prompts on first save).
- **New SOZ** applies the current Wizard definition to the project.
- **Quick** sets a coarse stride based on `analysis.preview_frames` and runs a fast preview.
- **Run** enforces Project Doctor checks before analysis.
- **Cancel** stops an in-progress run.

### Main layout
- Left drawer: project settings, output settings, Project Doctor, defined SOZ list.
- Center page: content for the selected stepper page.
- Right inspector (optional): context-sensitive status for the current page.
- Bottom console (optional): recent log lines and quick refresh.
- Status bar: run status messages and progress bar.

Inspector contents (when enabled):
- Project: Provenance Stamp with copy button.
- Define: live selection status tips.
- QC: QC status summary.
- Explore: active SOZ, metric, and time window.
- Export: extraction status hints.

## Project step (inputs and run configuration)
If no project is loaded, **Run** or **Quick** will prompt for topology and optional trajectory, then create a default project in memory.

### Inputs (left drawer)
- **Topology**: label shows current file; **Change** opens a file dialog.
- **Trajectory**: label shows current file; **Change** and **Clear** manage it.
- **Metadata**: summary line (SOZ count, selection count, stride, frames if available).

### Analysis Window (left drawer)
- **Frame start**: integer start frame index.
- **Frame stop**: integer stop frame index; use **End** to include full trajectory.
- **Stride**: sampling stride (integer >= 1).

### Output Settings (left drawer)
- **Output dir**: base directory for analysis outputs and logs.
- **Report format**: `html` or `md`.
- **Write per-frame CSV**: controls whether `per_frame.csv` is written.
- **Write parquet**: writes `.parquet` alongside CSV (requires `pyarrow`).

### Project Doctor (left drawer)
- **Run Project Doctor**: preflight checks on inputs, selections, solvent, probe, units, and PBC.
- **PBC Helper**: shows recommended `gmx trjconv` commands for preprocessing.
- Selection table: lists selection labels, counts, uniqueness requirements, and suggestions (columns: Selection label, Count, Require unique match, Expect count, Selection, Suggestions).

### Defined SOZs (left drawer)
- List of SOZ definitions currently in the project file.

### Project page (center)
- **Project Summary**: topology/trajectory details and PBC status.
- **Run Summary**: summarized results after analysis; **Show raw JSON** reveals raw summary.

## Define step (SOZ builder tabs)

### Wizard tab (main builder)
Fields and controls:
- **SOZ name**: name for the SOZ definition.
- **Solvent label**: display label only (does not change selection logic).
- **Solvent resnames**: comma-separated residue names used to identify solvent molecules.
- **Probe selection**: MDAnalysis selection applied within solvent residues.
- **Probe position**: `atom`, `com`, or `cog`.
- **Include ions** + **Ion resnames**: optional ion residues included as solvent.
- **Selection A** + **Selection A unique match**.
- **Shell cutoffs (A)**: comma-separated values in angstroms.
- **SOZ probe mode**: `probe`, `atom`, `com`, `cog`, or `all`.
- **Selection B (optional)** + **Selection B unique match**.
- **Selection B cutoff (A)**.
- **Selection B combine**: `AND` or `OR`.
- **Explain my SOZ**: a readable summary of the current wizard definition.
- **Preview**: evaluates the SOZ for frame 0 (requires inputs).

Notes:
- The Wizard uses angstrom inputs only. To use `unit: "nm"`, edit the SOZ JSON in **Advanced**.
- `Probe position = com` requires atom masses in the topology; otherwise, use `cog`.
- The Wizard stores the probe in `solvent.probe` and keeps legacy `water_oxygen_names` in sync.

### Selection match preview (Wizard)
- **Live**: update when inputs change.
- **Use trajectory**: include trajectory frames, not just topology.
- **Max rows**: limit output rows.
- **Preview**: Selection A or Selection B.
- Status lines show match counts and suggestions (columns: Atom index, Atom name, Residue name, Residue number, Segment ID, Chain ID, Molecule type).

### Selection Builder tab
Builds MDAnalysis selection strings:
- **Scope**: all atoms, protein, backbone, sidechain, nucleic.
- **Resname**, **Resid/resnum**, **Atom name**, **Segid**, **ChainID**.
- **Build** generates the selection string.
- **Use as Selection A/B** sends to Wizard.
- **Send to Tester** pushes to Selection Tester.

### Bridges tab
Defines solvent bridges between two selections.
- Table columns: `name`, `selection_a`, `selection_b`, `cutoff_a`, `cutoff_b`, `unit`, `probe_mode`.
- **Add Bridge**, **Remove Selected**.
- Each bridge produces `bridge_<name>/` outputs.

### Residue Hydration tab
Counts residue contacts with solvent in the SOZ.
- Table columns: `name`, `residue_selection`, `cutoff`, `unit`, `probe_mode`, `soz_name`.
- `hydration_freq` is the fraction of frames where the residue contacts the solvent set.
- If `soz_name` is empty, the first SOZ is used.

### Advanced tab
- Paste a SOZ JSON definition and click **Apply JSON to SOZ list**.
- Use this for multi-branch trees, non-A units, or advanced probe modes.

### Selection Tester tab
- **Test Selection**: evaluates a selection string.
- **Test Probe**: evaluates the current solvent probe selection.
- **Max rows** and **Load trajectory** control sampling.
- Results list atom/residue identifiers and suggestions (columns: index, resname, resid, resnum, segid, chainID, moltype, name).

## QC step

### QC Summary page
- **QC Summary**: a readable summary of preflight and run-time warnings.
- **Show raw QC JSON**: shows raw QC payload.
- **Diagnostics** tab: log viewer (same controls as the Logs tab).

### Diagnostics and logs
- **Log path**, **Refresh**, **Open Log**, **Copy Errors**.
- Filters: Level (All/INFO/WARNING/ERROR), search box, **Collapse tracebacks**.

## Explore step (interactive analysis)

### Timeline top area (tabbed)
The top plot region is tabbed: **Timeline** and **Entry/Exit** share the same plot area.

#### Timeline tab (Occupancy Timeline)
Controls:
- **SOZ** selector.
- **Metric** and **Secondary** (secondary plots on right axis).
- **Smooth** + **Window** (rolling mean in frames).
- Display toggles: **Overlay all SOZs**, **Step plot**, **Markers**, **Clamp y>=0**, **Mean line**, **Median line**, **Shade occupancy** (threshold), **Brush time window**.
- **Clear brush** resets the time window.
- Actions: **Save Plot**, **Copy** (clipboard), **Export** (file dialog).
- **Compute Timeline Statistics** calculates summary metrics (mean, median, entries/exits, residence stats).

Axes:
- X-axis: time in ns (GUI divides `time` by 1000).
- Y-axis: selected metric (count or fraction depending on metric).

Metric meanings:
- `n_solvent`, `entries`, `exits` are counts per frame.
- `occupancy_fraction` is 0/1 per frame (1 when `n_solvent>0`).

Brush behavior:
- When enabled, the brush time window filters Histogram, Heatmap, Event Raster, and tables.

#### Entry/Exit tab (Entry/Exit Rate)
Controls:
- **Entry/Exit mode**: Events per frame, Rate (events/ns), Cumulative events.
- **Normalize**: per ns, per 100 frames, or none (disabled for Rate/Cumulative).
- **Bin (ns)**: aggregates entries/exits into time bins (0 disables binning).
- **Split axes** and **Exits negative** adjust visualization.
- Actions: **Save Plot**, **Export Entry/Exit CSV**.

Axes:
- X-axis: time in ns.
- Y-axis: events per frame, events/ns, or cumulative events depending on mode.

Note: If `entries`/`exits` are missing, the GUI derives them from changes in `n_solvent` and warns in the status bar.

### Plots tab
Contains three plots (tabs): Histogram, Matrix / Heatmap, Event Raster.

#### Histogram
Controls: **Histogram metric**, **Bins**, **Normalize**, **Log Y**, **Split zeros**, **Plot Histogram**, **Save Plot**, **Copy**.

Axes:
- X-axis: selected metric (count or time in ns if `time`).
- Y-axis: frame count or fraction of frames if normalized.

Notes:
- Red line = mean; green line = median.
- **Split zeros** separates zero vs non-zero frames and computes mean/median on the non-zero subset.

When this plot is misleading:
- If most frames are zero occupancy, the histogram hides when occupancy occurs. Use Timeline or Event Raster.
- Large stride compresses transient events and can bias the distribution.

#### Matrix / Heatmap
Controls: **Matrix source**, **Top solvents**, **Min occ %**.

Modes:
- **Solvent occupancy (top N)**: binary presence matrix for the top N solvents by occupancy.
  - X-axis: time in ns.
  - Y-axis: solvent rank (row 0 = most occupied).
  - Color: 1 = present, 0 = absent.
- **Residue hydration**: heatmap of `hydration_freq` per residue (requires residue hydration config).

Note: Heatmap requires per-frame solvent IDs (`analysis.store_ids=true`).

#### Event Raster
Controls: **Event stride**, **Segments**, **Min duration**.

Axes:
- X-axis: time in ns.
- Y-axis: solvent rank (same ordering as the heatmap).

Interpretation:
- Points show occupancy events; **Segments** shows continuous occupancy spans.

When this plot is misleading:
- High stride can hide short events or merge nearby events.
- Small Top N can hide rare but important solvents.

### Tables tab
- **Filter**: text filter for row filtering.
- Tabs: **Per Frame**, **Per Solvent**.
- Tables are sortable; filtering applies to the active table.

## Export step (data, report, extraction)

### Export Data
- Writes `metadata.json`, `soz_<name>/`, `bridge_<name>/`, and `hydration_<name>/` outputs to disk.
- Uses the Output Settings directory unless overridden in the export dialog.

### Export Report
- Generates `report/report.md` and plots; `report/report.html` if `report_format=html`.
- The Report panel shows the output path and copies it to the clipboard after export.

### Extract tab (GUI)
Modes:
- **Threshold**: rule-based filtering with operator and threshold.
- **Percentile**: selects frames at/above a percentile of the metric distribution.
- **Top N frames**: selects the top N frames by metric (ties may add frames).

Controls:
- **SOZ**, **Metric**, **Operator**, **Threshold**, **Percentile**, **Top N frames**.
- **Min run length** and **Gap tolerance** (disabled for Top N mode).
- **Format**: currently `xtc`.
- **Output directory** and **Use Output Settings directory** toggle.
- **Preview** computes the rule and preview table; **Extract** writes outputs.

Outputs:
- `<prefix>.xtc`: extracted trajectory.
- `<prefix>_ref.pdb`: reference structure (first selected frame).
- `<prefix>_frames.csv`: manifest (frame, time, n_solvent, optional solvent_ids hash).
- `<prefix>_params.json`: extraction parameters.

## Plot export and PowerPoint guidance

### Export formats
**Save Plot** offers: PNG, SVG, EMF, PDF, CSV.
- **PNG**: raster image.
- **SVG**: vector (sanitized for PowerPoint edits).
- **EMF**: Office-friendly vector, best with Inkscape installed.
- **PDF**: vector, best with Inkscape installed.
- **CSV**: plot data (timeline, histogram, heatmap, event raster, entry/exit).

### PowerPoint tips
- SVG: Insert > Pictures, then Ungroup (or Convert to Shape). If Ungroup fails, try a cut/paste cycle.
- EMF: Insert > Pictures, then Group > Ungroup (accept conversion prompt).
- For best fidelity, install Inkscape so SVG->EMF/PDF conversions use `--export-area-drawing`.

### Plot CSV outputs (for reproducibility)
- Timeline CSV: `time_ns`, metric column, optional secondary metric; includes `soz` when overlay is enabled.
- Entry/Exit CSV: `time_ns`, `count`, `delta`, `entries`, `exits`, `entry_rate`, `exit_rate` (entries/exits follow current normalization/binning).
- Histogram CSV: `bin_left`, `bin_right`, `bin_center`, `count` (fraction if normalized).
- Heatmap CSV: `solvent_id` plus time columns (solvent occupancy) or hydration table columns for residue hydration.
- Event Raster CSV: points (`solvent_id`, `row`, `frame`, `time_ns`) or segments (`solvent_id`, `row`, `start_frame`, `end_frame`, `start_ns`, `end_ns`, `duration_frames`).

## Project JSON schema (current)

### Minimal skeleton
```json
{
  "version": "1.0",
  "inputs": {
    "topology": "path/to/topol.tpr",
    "trajectory": "path/to/traj.xtc"
  },
  "solvent": {
    "solvent_label": "Water",
    "water_resnames": ["SOL", "WAT", "TIP3", "HOH"],
    "water_oxygen_names": ["O", "OW", "OH2"],
    "water_hydrogen_names": ["H1", "H2", "HW1", "HW2"],
    "probe": {
      "selection": "name O OW OH2",
      "position": "atom"
    },
    "ion_resnames": ["NA", "CL"],
    "include_ions": false
  },
  "selections": {
    "selection_a": {
      "label": "selection_a",
      "selection": "protein and resid 10 and name CA",
      "require_unique": true
    }
  },
  "sozs": [],
  "analysis": {
    "frame_start": 0,
    "frame_stop": null,
    "stride": 1,
    "gap_tolerance": 1,
    "store_ids": true,
    "store_min_distances": true,
    "preview_frames": 200
  },
  "outputs": {
    "output_dir": "results",
    "write_per_frame": true,
    "write_parquet": false,
    "report_format": "md"
  },
  "extraction": {
    "rule": "n_solvent>=1",
    "min_run_length": 1,
    "gap_tolerance": 0,
    "output_format": "xtc",
    "output_dir": "results/extracted"
  },
  "bridges": [],
  "residue_hydration": []
}
```

### Selections
- Use `selections` (preferred) or legacy `seeds`.
- Each selection supports either a full `selection` string or structured fields (`resid`, `resname`, `atomname`, `segid`, `chain`, `atom_indices`, `pdb_serials`).
- `require_unique` enforces a single-atom match.
- Bridges also accept legacy `seed_a` and `seed_b` keys.

### SOZ logic tree
- Node types: `distance`, `shell`, `and`, `or`, `not`.
- Use `selection_label` in node params (legacy `seed_label` or `seed` are normalized).
- Cutoffs accept `unit: "A"` or `unit: "nm"`.
- Probe mode key is `probe_mode` (legacy `atom_mode` also accepted).

Allowed probe modes:
- `probe`: use `solvent.probe.position`.
- `atom`: use probe atoms directly.
- `com`: center of mass of probe atoms per solvent residue (requires masses).
- `cog`: center of geometry of probe atoms per residue.
- `all`: all atoms in solvent residues.
- Legacy `O` is treated as `atom`.

### Solvent configuration
- `water_resnames`: residue names that identify solvent molecules (UI shows "Solvent resnames").
- Legacy alias: `solvent_resnames` is accepted in project JSON.
- `probe.selection`: MDAnalysis selection for probe atoms within solvent residues.
- `probe.position`: `atom`, `com`, or `cog` (global default).
- `water_oxygen_names` and `water_hydrogen_names` are legacy fields kept for compatibility.
- `solvent_label` is a display label only; set it to the actual solvent name for clarity (e.g., Methanol, DMSO).

### Bridges
Each bridge uses two selections and two cutoffs.
- `selection_a`, `selection_b`, `cutoff_a`, `cutoff_b`, `unit`, `probe_mode`.

### Residue hydration
- `residue_selection`: residues to score for hydration.
- `cutoff`, `unit`, `probe_mode`.
- `soz_name`: optional SOZ to use; default is the first SOZ.

## Outputs and file map

### Output map
```text
Path                                   | Meaning
-------------------------------------- | ---------------------------------------------------------------
metadata.json                           | Project snapshot, warnings, QC summary.
soz_<name>/per_frame.csv               | Per-frame time series and events.
soz_<name>/per_solvent.csv             | Per-solvent residence and occupancy summary.
soz_<name>/summary.json                | Aggregated SOZ summary metrics.
soz_<name>/min_distance_traces.csv     | Min distance traces per selection (if enabled).
bridge_<name>/per_frame.csv            | Bridge-specific per-frame table.
bridge_<name>/per_solvent.csv          | Bridge-specific per-solvent table.
hydration_<name>/hydration_table.csv   | Residue hydration table.
report/report.md                       | Report markdown with figures.
report/report.html                     | Report HTML (if report_format=html).
```

### per_frame.csv columns
- `frame`, `time`, `n_solvent`, `entries`, `exits`.
- `solvent_ids` (semicolon-delimited `resname:resid:segid`) if `analysis.store_ids=true`.
- Optional min-distance columns (one per selection) if `analysis.store_min_distances=true`.

### per_solvent.csv columns
- `solvent_id`, `resindex`, `resname`, `resid`, `segid`.
- `frames_present`, `occupancy_pct`, `entries`.
- `mean_res_time_cont`, `median_res_time_cont`.
- `mean_res_time_inter`, `median_res_time_inter`.

### summary.json key fields
- `n_frames`, `frame_stride`, `dt`, `time_unit`.
- `occupancy_fraction`: fraction of sampled frames with `n_solvent > 0`.
- `mean_n_solvent`, `median_n_solvent`, `max_n_solvent`.
- `n_solvent_hist`, `entry_rate`, `exit_rate`.

## Extraction (GUI + CLI)

### Rule format
Rules are simple comparisons against per-frame metrics:
- `n_solvent>=1`
- `entries>=2`
- `occupancy_fraction==1`

`occupancy_fraction` is computed per frame as 1 if `n_solvent>0`, else 0.

### CLI extraction example
```bash
sozlab-cli extract \
  --project /path/to/project.json \
  --soz shell_only \
  --rule "n_solvent>=2" \
  --min-run 10 \
  --gap 2 \
  --out /path/to/extracted
```

### Validation of extraction
1) Confirm `<prefix>_frames.csv` matches the expected number of frames.
2) Cross-check against `per_frame.csv` for the same SOZ.
3) Load `<prefix>_ref.pdb` and `<prefix>.xtc` in a viewer for spot checks.

## Reproducibility and QC
- Keep the project JSON and `metadata.json` for each run.
- Record `analysis.frame_start`, `analysis.frame_stop`, `analysis.stride`, and `analysis.gap_tolerance`.
- Use `sozlab-cli validate --project ... --max-frames 200` to compare fast vs slow evaluation.
- Store preprocessing context in `inputs.processed_trajectory` and `inputs.preprocessing_notes`.

## Troubleshooting (Symptom -> Confirm -> Fix)

### Selection resolves to 0 atoms
Confirm: Selection Tester shows 0 hits.
Fix: Verify resname/resid, add segid/chainID, or adjust selection scope.

### Selection not unique but require_unique is checked
Confirm: Selection Tester shows multiple hits.
Fix: Add segid/chainID/resid or disable require_unique.

### Solvent resnames do not match any residues
Confirm: Project Doctor reports no solvent matches.
Fix: Update `solvent.water_resnames` to match your topology.

### Probe selection resolves to 0 atoms or misses residues
Confirm: Selection Tester -> Test Probe shows 0 hits or missing residues.
Fix: Update `solvent.probe.selection` and `solvent.probe.position`.

### Probe position com fails
Confirm: Error mentions missing masses.
Fix: Add atom masses in the topology or use `probe.position: "cog"`.

### Logic tree yields zero occupancy
Confirm: `per_frame.csv` has `n_solvent=0` for all frames.
Fix: Check cutoffs and selection labels; use the Wizard Preview to test frame 0.

### Heatmap/Event Raster says "No solvent IDs available"
Confirm: `per_frame.csv` lacks `solvent_ids`.
Fix: Set `analysis.store_ids=true` (CLI: omit `--no-ids`).

### Entry/Exit plot is flat
Confirm: `entries` and `exits` are zero or derived from `n_solvent`.
Fix: Reduce stride, re-run with per-frame output enabled.

### Per-frame tables are too large
Confirm: Disk usage is large or export is slow.
Fix: Set `outputs.write_per_frame=false` (CLI: `--no-per-frame`).

### Output directory not writable
Confirm: Logs show write errors.
Fix: Update Output Settings or pass `--output` for CLI runs.

### CLI command not found
Confirm: `sozlab-cli --help` fails.
Fix: Activate the `sozlab` environment or run the installed script from your env.

### SVG/EMF not editable in PowerPoint
Confirm: SVG opens as an icon or EMF is corrupted.
Fix: Export again with Inkscape installed; use the PowerPoint tips above.

## Appendix: keyboard shortcuts
- `Ctrl+O`: Load project
- `Ctrl+S`: Save project
- `Ctrl+R`: Run analysis
- `Ctrl+Shift+R`: Export report
- `Ctrl+E`: Export data
- `Ctrl+P`: Toggle presentation mode
- `Ctrl+K`: Command palette
- `Ctrl++` or `Ctrl+=`: Increase scale
- `Ctrl+-`: Decrease scale
- `Ctrl+0`: Reset scale
- `Ctrl+Q`: Quit
- `Ctrl+Mouse Wheel`: Adjust scale
