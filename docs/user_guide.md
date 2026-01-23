# SOZLab User Guide

## Installation (Linux)
Conda (recommended):
- `conda env create -f environment.yml`
- `conda activate sozlab`
- Launch GUI: `sozlab-gui`
- CLI help: `sozlab-cli --help`
  - PyQt6 is installed via pip during env creation (from `environment.yml`/`pyproject.toml`).
  - If your conda uses strict channel priority with defaults first, retry with `conda env create -f environment.yml --override-channels`.

Virtualenv (alternative):
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -e .`
- Launch GUI: `sozlab-gui`
- CLI help: `sozlab-cli --help`

## Workflow
1) Load a project (`*.json`/`*.yaml`) or create one from the GUI by choosing topology/trajectory.
2) Define SOZs using the Wizard or Advanced JSON editor.
3) Run analysis; results populate the overview, timeline, plots, and tables.
4) Use the Plots tab for histograms, heatmaps, and event rasters.
5) Export results and report from the left panel.

Frame range controls:
- Use `Frame stop = End` to include the full trajectory.

## SOZ Builder
- **Wizard**: defines a multi-shell SOZ around Selection A, optionally combined with a Selection B distance constraint.
- **Advanced**: paste a SOZ JSON definition (see `examples/soz_configs/`).
  - Nodes reference selection labels defined in the project `selections` section.

## Quick Subset Test
Uses the `preview_frames` setting (default 200) to run a lightweight check before full analysis.

## CLI Example
```
sozlab-cli run --project examples/sample_project.json --output examples/output --progress --report
```

Validation (subset check):
```
sozlab-cli validate --project examples/sample_project.json --max-frames 50
```

## Outputs
- `metadata.json` (config + warnings)
- `soz_<name>/per_frame.csv`, `per_solvent.csv`, `summary.json`
- `bridge_<name>/...`
- `hydration_<name>/hydration_table.csv`
- `report/report.md` + figures
  - If `outputs.report_format` is `html`, `report/report.html` is also generated.

## Notes
- Distance cutoffs accept Angstrom or nm in project files. GUI Wizard uses Angstrom inputs.
- Ensure trajectories are preprocessed (nojump/center/compact) for best SOZ stability.
