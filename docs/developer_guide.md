# SOZLab Developer Guide

## Layout
- `src/engine/`: analysis backend (MDAnalysis, SOZ logic, stats)
- `src/app/`: PyQt6 GUI
- `cli/`: CLI entrypoints
- `examples/`: sample data and configs
- `docs/`: documentation

## Key Modules
- `engine/models.py`: project configuration dataclasses
- `engine/soz_eval.py`: logic-tree evaluation
- `engine/analysis.py`: main engine runner
- `engine/stats.py`: occupancy/residence/event statistics
- `engine/export.py`: CSV/JSON/Parquet output
- `engine/report.py`: report generation
- `engine/validation.py`: validation checks

## Development
- Create conda env: `conda env create -f environment.yml`
- Activate: `conda activate sozlab`
- PyQt6 is pulled via pip from `environment.yml`/`pyproject.toml` during env creation.
- Run tests: `pytest`
- GUI dev: `PYTHONPATH=src sozlab-gui`

## Testing Strategy
- Unit tests for unit conversion, SOZ logic, residence stats.
- Optional integration tests using `examples/data/sample.pdb`.

## Extending the SOZ Logic Tree
- Add a new node type in `engine/soz_eval.py`.
- Update JSON schema examples in `examples/soz_configs/`.
- Update GUI Builder as needed.
