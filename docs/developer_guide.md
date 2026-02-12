# SOZLab Developer Guide

## Repository layout
- `src/engine/`: analysis engine, models, export/report, extraction, validation
- `src/app/`: PyQt6 GUI (`main.py`) and 3D density viewer (`viz_3d.py`)
- `cli/`: CLI entry point (`cli/sozlab_cli.py`)
- `examples/`: template project JSON and SOZ JSON snippets
- `tests/`: unit/integration/gui tests
- `docs/`: user/developer/tutorial docs

## Runtime entry points
- GUI script: `sozlab-gui` -> `app.main:main`
- CLI script: `sozlab-cli` -> `cli.sozlab_cli:main`

CLI commands:
- `run`
- `validate`
- `extract`

## Current exposed feature set
UI and CLI currently expose:
- SOZ occupancy analysis
- distance bridges
- density maps + density explorer (2D/3D)
- export/report/extraction

These analysis families are currently stripped on project load/run:
- `hbond_water_bridges`
- `hbond_hydration`
- `water_dynamics`

Code references:
- GUI stripping: `src/app/main.py` (`_strip_removed_analysis_options`)
- CLI stripping: `cli/sozlab_cli.py` (`_strip_removed_analysis_options`)

## Development setup
```bash
conda env create -f environment.yml
conda activate sozlab
pip install -e .
```

## Testing
Quick default run:
```bash
python -m pytest -q
```

Markers:
- `unit`
- `integration`
- `gui`
- `slow`

Quick profile:
```bash
python -m pytest -q -m "not slow"
```

Full profile:
```bash
python -m pytest -q -m "slow or not slow"
```

## Docs maintenance workflow
When updating docs, treat these as source of truth:
- CLI: `cli/sozlab_cli.py` + live `--help` output
- GUI structure/labels: `src/app/main.py`
- 3D viewer controls: `src/app/viz_3d.py`
- output artifacts: `src/engine/export.py`, `src/engine/extract.py`, `src/engine/report.py`

Recommended docs verification commands:
```bash
sozlab-cli --help
sozlab-cli run --help
sozlab-cli validate --help
sozlab-cli extract --help
```
