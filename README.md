# SOZLab

**Solvent occupancy zones, made trustworthy.** SOZLab measures the solvent around part of a molecule in molecular
dynamics trajectories: how many molecules are in a zone, how long they stay, how often they come and go, and where
the solvent density is enriched. Every number is reported with its definition, its unit, an uncertainty, and a note on
when it can and cannot be trusted. Every export records exactly which software, settings and input files produced it.

It comes as a **Workbench** (a GUI built around one screen: import, define, run, inspect, export) and a **CLI** that
share one engine, so the same project gives identical results in both (tested: identical per-frame solvent identities).

## What is new in 2.1

- **Workbench GUI**, redesigned from scratch: live 3D view of the zone, a live occupancy preview while you drag the
  cutoff, results with ± standard errors and trust badges, run history with overlay comparison, built-in definitions.
- **Correct time**: durations come from the trajectory's time stamps (midpoint rule, irregular sampling handled, a single
  frame has no duration instead of an invented one); units are visible everywhere.
- **Residence done properly**: visits are flagged when cut by the window edges; the survival function S(τ), the
  Kaplan–Meier estimate and the mean visit duration are reported side by side; stride sensitivity is measured, not assumed.
- **Uncertainty everywhere**: autocorrelation-aware standard errors for occupancy, block errors for residence and rates.
- **Density maps in the solute frame**: molecules wrapped to the solute, superposed every frame, normalised to a
  measured bulk density, with reliability masks and a content-keyed cache (a stale cache can no longer be served).
- **Verified frame extraction**: refuses inputs that changed since the analysis (per-frame content fingerprints).
- **Provenance**: SHA-256 of inputs, configuration hash, software versions and a manifest in every export; `verify` re-checks it.
- **Parallel frame evaluation** that is bitwise identical to serial.

The full list, including the deliberate changes of meaning, is in [CHANGELOG.md](CHANGELOG.md). The evidence behind the
claims (reproduced v2.0 flaws, known-answer and reference-system validation, comparison with other tools, remaining
limits) is in [docs/evidence/](docs/evidence/README.md).

## Install

```bash
pip install -e .            # the Workbench, the CLI and the engine
pip install -e ".[test]"    # + pytest, pytest-qt
pip install -e ".[parquet]" # optional Parquet tables
```

Python 3.10+. Developed and tested on Windows 11 with Python 3.12; the engine and CLI have no platform-specific code.
A conda environment is provided in `environment.yml`.

## Use it

### Workbench

```bash
sozlab-gui                      # start empty: drop a topology + trajectory on the window
sozlab-gui system.gro run.xtc   # open files directly
sozlab-gui --demo               # a built-in synthetic hydrated solute, no files needed
```

1. **Import** — drop the files. The window shows atoms, frames, box, time step and the solvent it found.
2. **Define** — pick what the zone surrounds (detected parts are offered as one-click chips), then drag the line on
   the distance histogram. The 3D view and an occupancy preview follow immediately.
3. **Run** — the settings stay visible beside the results; *Advanced* holds everything else, marks what you changed and resets it.
4. **Inspect** — tiles show value ± standard error with a trust badge; **?** opens the definition; the *Trust* panel lists caveats.
5. **Export** — data, a report, and frames, each with a manifest; *Verify files* re-hashes them.

[docs/user_guide.md](docs/user_guide.md) walks through every control.

### CLI

```bash
sozlab-cli run --project project.json --output results/
sozlab-cli validate --project project.json
sozlab-cli extract --from-results results/ --rule "n_solvent>=1" --out frames/
sozlab-cli verify results/
sozlab-cli selftest             # known-answer checks on this installation (about 10 seconds)
```

No data at hand? `python examples/make_demo_data.py` writes a small synthetic system and `examples/demo_project.json`; then
`sozlab-cli run --project examples/demo_project.json --output demo_results --report`.

Projects written by v2.0 still open (folder menu → *Open a project file…* in the Workbench, `--project` in the CLI); a project saved
from the Workbench runs unchanged with the CLI.

## Check your installation

```bash
sozlab-cli selftest             # synthetic systems with analytically known answers
python -m pytest -q             # the full suite (GUI tests run offscreen)
```

## Documentation

| | |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | the Workbench and CLI step by step |
| [docs/science/definitions.md](docs/science/definitions.md) | what every quantity measures, its unit, and when to trust it |
| [docs/developer_guide.md](docs/developer_guide.md) | architecture, tests, benchmarks, how to reproduce the evidence |
| [docs/evidence/README.md](docs/evidence/README.md) | validation, comparison with other tools, limits |
| [docs/design_notes.md](docs/design_notes.md) | the Workbench design system |

## Citation

If you use SOZLab in academic work, please cite (see also `CITATION.cff`):

**Ferreira Francisco, A.** *SOZLab: GUI and CLI for solvent occupancy zone analysis of molecular dynamics trajectories.*
Version 2.1.0. https://github.com/aleff-ferreira/SOZLab

## License

MIT License. Copyright (c) 2025 SOZLab contributors. See `LICENSE`.
