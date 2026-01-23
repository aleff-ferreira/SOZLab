# SOZLab

SOZLab is a Linux-only GUI + CLI for solvent occupancy zone (SOZ) analysis of molecular dynamics trajectories. It lets you define solvent zones around one or more seed selections, compute per-frame occupancy and entry/exit events, and export tables, plots, and extracted trajectories.

## What it does
- Define SOZ logic trees (shells, distances, boolean logic) around seed selections.
- Compute per-frame occupancy (`n_solvent`), entry/exit events, and residence summaries.
- Visualize results in a GUI (timeline, histograms, heatmaps, event raster).
- Export CSV/JSON outputs, reports, and extracted frame subsets.

## Requirements
- OS: Linux (GUI is Linux-only).
- Python: 3.10+ (3.11 recommended).
- Optional: GROMACS for preprocessing (SOZLab does not require it).

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate sozlab
```
This installs dependencies and exposes `sozlab-gui` and `sozlab-cli`.

### Virtualenv (developer install)
```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Optional parquet support:
```bash
pip install -e .[parquet]
```

## Quick start

### CLI
```bash
sozlab-cli run --project examples/sample_project.json --output out --progress --report
```

### GUI
```bash
sozlab-gui
```
Then load `examples/sample_project.json` and click **Run**.

## Outputs (overview)
- Outputs go to `outputs.output_dir` in the project file.
- `sozlab-cli run --output OUT` overrides the output directory for that run.
- Extracted trajectories use `--out` (see `docs/tutorial.md`).

## Troubleshooting (top 5)
1) Selection resolves to 0 atoms: check resnames, segid/chainID, and `resid` vs `resnum`.
2) Water resnames mismatch: update `solvent.water_resnames` to match your topology.
3) PBC warning: `No valid box vectors found` means distances may be unreliable.
4) Matplotlib cache warning: set `MPLCONFIGDIR` to a writable directory.
5) Heatmap or raster empty: ensure `analysis.store_ids=true` (or avoid `--no-ids`).

## Documentation
- Tutorial source: `docs/tutorial.md`
- Tutorial PDF: `docs/tutorial.pdf`
- Additional guides: `docs/user_guide.md`, `docs/developer_guide.md`

## Citation
See `CITATION.cff`.

## License
MIT.

