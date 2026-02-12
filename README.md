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

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate sozlab
pip install -e .
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

## Documentation
- `docs/user_guide.md`
- `docs/developer_guide.md`
- `docs/design_notes.md`

## Citation

If you use SOZLab in academic work (papers, preprints, theses, posters, or talks), please cite:

**Ferreira Francisco, A.** *SOZLab: a Linux GUI and CLI for Solvent Occupancy Zone (SOZ) analysis of molecular dynamics trajectories.* Version 1.0.0, released 2026-02-12. GitHub repository: https://github.com/aleff-ferreira/SOZLab (accessed YYYY-MM-DD).

## License
MIT License

Copyright (c) 2025 SOZLab contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
