# SOZLab User Guide

## 1. Installation

Conda:
```bash
conda env create -f environment.yml
conda activate sozlab
pip install -e .
```

Virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Entry points:
- GUI: `sozlab-gui`
- CLI: `sozlab-cli`

Note: `python -m sozlab` is not a valid module entry in this repo.

## 2. CLI Reference (source: live `--help`)

### Main
```text
usage: sozlab-cli [-h] {run,validate,extract} ...
```

### `run`
```text
usage: sozlab-cli run [-h] --project PROJECT [--output OUTPUT]
                      [--stride STRIDE] [--start START] [--stop STOP]
                      [--no-ids] [--no-per-frame] [--progress] [--report]
                      [--workers WORKERS]
```

Flags:
- `--project` required project JSON
- `--output` output directory override
- `--stride` frame stride override
- `--start` start frame override
- `--stop` stop frame override (exclusive)
- `--no-ids` disable per-frame solvent IDs
- `--no-per-frame` disable per-frame table export
- `--progress` terminal progress bar
- `--report` generate report
- `--workers` CPU worker count (`0` or omitted = auto)

### `validate`
```text
usage: sozlab-cli validate [-h] --project PROJECT [--max-frames MAX_FRAMES]
                           [--workers WORKERS]
```

### `extract`
```text
usage: sozlab-cli extract [-h] --project PROJECT [--soz SOZ] [--rule RULE]
                          [--min-run MIN_RUN] [--gap GAP] [--out OUT]
                          [--format FORMAT] [--prefix PREFIX]
                          [--output OUTPUT] [--workers WORKERS]
```

## 3. GUI Navigation

Primary pages:
- `Project`
- `Define`
- `QC`
- `Explore`
- `Export`

Header actions:
- `Load`, `Save`, `New SOZ`, `Quick`, `Run`, `CPU workers`, `Cancel`, `Export Data`, `Export Report`
- toggles/appearance: `Console`, `Drawer`, `Theme`, `Density`, `Scale`, `Presentation`

## 4. Project Page

Left drawer includes:
- `Project Inputs`
  - topology/trajectory path fields with `Change`, `Clear`, `Copy`
- `Run Configuration`
  - `Analysis Window`: frame start/stop/stride
  - `Output Settings`: output directory, report format, CSV/parquet toggles
- `Project Doctor`
  - run checks, PBC helper, findings, diagnostics
- `Selection Table`
- `Defined SOZs`

## 5. Define Page

Current tabs:
- `Wizard`
- `Distance Bridges`
- `Density Maps`
- `Selection Builder`
- `Selection Tester`
- `Advanced`

### Wizard
Use this to define solvent behavior and SOZ logic from `Selection A` and optional `Selection B`.

### Distance Bridges
Configure bridge definitions with:
- `Name`
- `Selection A`, `Selection B`
- `Cutoff A`, `Cutoff B`
- `Unit`
- `Mode`

### Density Maps
Configure density analyses with:
- `Name`, `Density species`
- `Grid spacing (A)`, `Padding (A)`, `Stride`
- optional alignment fields
- advanced `Conditioning Policy` and `View Mode`

### Selection Builder / Tester
- builder composes MDAnalysis selections
- tester validates selections and probe selection on loaded inputs

### Advanced
Direct SOZ JSON paste/apply.

## 6. QC Page

Views:
- `QC Summary`
- `Diagnostics`

Includes health badges, findings text, optional raw QC JSON, and log filtering tools.

## 7. Explore Page

Mode toolbar:
- `SOZ Explorer`
- `Bridges`
- `Density`

### 7.1 SOZ Explorer
Sub-views:
- `Overview`
- `Events & Details`

#### Overview
Toolbar switch:
- `Timeline`
- `Entry/Exit`

`Timeline` (current visible controls):
- `SOZ`
- `Clamp y≥0`, `Brush time window`, `Clear brush`
- `Mean line`, `Median line`, `Shade occupancy`, threshold (`Shade ≥`)
- optional advanced display (`Overlay all SOZs`, `Markers`)
- `Save Plot`, copy, export, `Compute Timeline Statistics`

Current behavior:
- primary metric is fixed to `n_solvent`
- y-axis label is `Solvent molecules within cutoff`

`Entry/Exit` controls:
- `Entry/Exit mode`
- `Split axes`
- `Exits negative`
- `Save Plot`
- `Export Entry/Exit CSV`

#### Events & Details
`Histogram` view:
- actions: `Plot Histogram`, `Save Plot`, copy
- histogram is unified (no visible zero/non-zero split)
- no mean/median vertical lines

`Event Raster` view:
- `Event stride`
- `Segments`
- `Min duration`

Tables panel:
- `Per Frame` / `Per Solvent`
- text filter

### 7.2 Bridges
Current Bridges mode exposes Distance Bridge plotting only.
Visible controls:
- `Export Plot`

Displayed plots:
- `Distance Bridge`
- `Residence Time Distribution`
- `Top Bridging Residues`

Stats strip shows `n=`, `mean=`, `max=`, `top3=`.

### 7.3 Density
Top controls:
- `View` (`physical`, `relative`, `score`)
- `Colormap`
- slice controls (`X`, `Y`, `Z`, `Center Slices`)
- `Overlay reference`
- `Export Figure Pack`

Density view modes:
- `3D`
- `Plots`
- `Split`

3D viewer (`3D Molecular Viewer`):
- quick actions: `Shot`, `Reset`, `Opts`, `Layers`
- controls sections: `Density`, `Layers`, `Render`, `Pick`, `Insights`

## 8. Export Page

Top actions:
- `Export Data`
- `Export Report`

Extraction panel supports:
- `Threshold`, `Percentile`, `Top N frames`
- SOZ/metric/rule controls
- run-length and gap controls
- output format (`xtc`)
- `Preview` and `Extract`

## 9. Outputs

Typical outputs in `outputs.output_dir`:
- `metadata.json`
- `soz_<name>/per_frame.csv`
- `soz_<name>/per_solvent.csv`
- `soz_<name>/summary.json`
- `soz_<name>/min_distance_traces.csv` (when available)
- `distance_bridge_<name>/per_frame.csv` and `per_solvent.csv` (if configured)
- `density_map_<name>/metadata.json` (if configured)
- `report/report.md` (and `report/report.html` when format is `html`)
- extraction artifacts (`*.xtc`, `_ref.pdb`, `_frames.csv`, `_params.json`)

## 10. Plot context menu

Right-click plot menu is simplified to:
- `Export`
- `Plot Options` with `Grid` and `Alpha`
