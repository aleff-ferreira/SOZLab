# SOZLab

SOZLab is a Linux GUI + CLI for solvent occupancy zone (SOZ) analysis of molecular dynamics trajectories.

Current public workflow focuses on:
- SOZ occupancy analysis (`n_solvent`, entries/exits, residence summaries)
- Distance bridge analysis
- Density maps (2D slices + NGL-based 3D viewer)
- CSV/JSON/report export and frame extraction

## Entry Points
- GUI: `sozlab-gui`
- CLI: `sozlab-cli`

CLI surface:
```text
usage: sozlab-cli [-h] {run,validate,extract} ...
```

Note: `python -m sozlab` is not available in this repo. Use `sozlab-cli` or `python -m cli.sozlab_cli`.

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate sozlab
pip install -e .
```

### Virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional parquet support:
```bash
pip install -e .[parquet]
```

## Run

### GUI
```bash
sozlab-gui
```

### CLI
```bash
sozlab-cli --help
sozlab-cli run --help
sozlab-cli validate --help
sozlab-cli extract --help
```

## Tested CLI Quick Start (LAAO)

Create a minimal project JSON for `/home/aleff/laao_water`:
```bash
python - <<'PY'
from engine.models import (
    ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, OutputConfig,
    SOZDefinition, SOZNode, SelectionSpec, ExtractionConfig,
)
from engine.analysis import write_project_json

project = ProjectConfig(
    inputs=InputConfig(
        topology='/home/aleff/laao_water/extracted_ref.pdb',
        trajectory='/home/aleff/laao_water/step7_production_mol_rect_3_system_10ns.xtc',
    ),
    solvent=SolventConfig(water_resnames=['SOL', 'HOH', 'TIP3']),
    selections={'mysel': SelectionSpec(label='mysel', selection='resid 50')},
    sozs=[SOZDefinition(
        name='TestSOZ',
        description='Test SOZ',
        root=SOZNode(type='distance', params={'selection_label': 'mysel', 'cutoff': 4.0}),
    )],
    analysis=AnalysisOptions(frame_start=0, frame_stop=1001, stride=1000),
    outputs=OutputConfig(output_dir='/tmp/sozcodex_docs_out', report_format='md'),
    extraction=ExtractionConfig(output_dir='/tmp/sozcodex_docs_extract', rule='n_solvent>=1'),
)
write_project_json(project, '/tmp/sozcodex_docs_project.json')
print('/tmp/sozcodex_docs_project.json')
PY
```

Run validate, analysis, and extraction:
```bash
sozlab-cli validate --project /tmp/sozcodex_docs_project.json --max-frames 10
sozlab-cli run --project /tmp/sozcodex_docs_project.json --output /tmp/sozcodex_docs_out --progress --report --workers 1
sozlab-cli extract --project /tmp/sozcodex_docs_project.json --soz TestSOZ --rule "n_solvent>=1" --min-run 1 --gap 0 --out /tmp/sozcodex_docs_extract --prefix docs_demo --format xtc --workers 1
```

## Testing
Quick default suite:
```bash
python -m pytest -q
```

Explicit quick profile:
```bash
python -m pytest -q -m "not slow"
```

Full suite (includes slow tests):
```bash
python -m pytest -q -m "slow or not slow"
```

## Documentation
- `docs/tutorial.md`
- `docs/user_guide.md`
- `docs/developer_guide.md`
- `docs/design_notes.md`

PDF tutorial (generated): `docs/tutorial.pdf`.

## Citation
See `CITATION.cff`.

## License
MIT (`LICENSE`).
