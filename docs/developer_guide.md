# SOZLab developer guide

## Layout

```
src/engine/    analysis engine (no GUI imports)
  analysis.py    SOZAnalysisEngine: preflight -> frame loop -> statistics -> density -> trust -> result
  frames.py      per-frame evaluation, serial or process-parallel, results accumulated in frame order (bitwise = serial)
  soz_eval.py    zone logic tree (distance / shell / and / or / not)
  solvent.py     solvent layout, probes, whole-molecule handling
  periodic.py    boxes, minimum image, triclinic
  timing.py      FrameTiming: time source, uniformity, midpoint-rule intervals
  stats.py       StatsAccumulator: per-frame / per-molecule / visits / survival / stride sensitivity
  residence.py   visits, censoring, survival, Kaplan-Meier        uncertainty.py  autocorrelation and block errors
  density.py     solute-frame density engine, content-keyed cache
  provenance.py  fingerprints, configuration hash, environment    export.py, report.py, extract.py, dictionary.py, trust.py
  synthetic.py   systems with analytically known answers          selftest.py     the checks behind `sozlab-cli selftest`
src/app/       PyQt6 front end
  main.py        entry point (`sozlab-gui`)
  workbench/     theme.py (tokens, QSS) · icons.py · widgets.py · charts.py · viewport3d.py (software 3D) ·
                 model.py (zone/run specs -> project) · services.py (background threads) · session.py (state, validation, history) ·
                 panels_left.py / panels_results.py / panels_right.py · window.py
cli/           sozlab_cli.py: run / validate / extract / verify / selftest
tests/         engine tests, tests/gui (pytest-qt, offscreen), tests/integration (CLI)
bench/         repro/ (v2.0 flaws), validate/ (reference systems), data/ (OpenMM generator), gui/ (usability harness)
docs/          user guide, science definitions, evidence
```

## Design rules

- **The window renders `Session`; it does not own state.** Tests drive the same `Session` methods the widgets call, so GUI behaviour is
  checkable without pixels. Anything that touches an MDAnalysis `Universe` runs off the GUI thread with its own `Universe`.
- **One engine, two front ends.** The Workbench builds a `ProjectConfig` and calls `SOZAnalysisEngine`; the CLI loads the same JSON.
  `tests/gui/test_gui_cli_agreement.py` fails if the per-frame solvent identities, summaries or configuration hash ever differ.
- **A number is never separated from its definition.** New quantities get an entry in `engine/dictionary.py` (shown in the GUI, the export and
  the docs) and, if they have a failure mode, a rule in `engine/trust.py`.
- **Bump `ALGORITHM_VERSIONS` when a definition changes**; it is part of every cache key and every provenance record.
- **Parallel results must equal serial results exactly**; add a case to `tests/test_pipeline_known_answers.py` for any new per-frame output.

## Tests

```bash
python -m pytest -q                       # everything not marked slow (GUI tests run offscreen)
python -m pytest -q tests/gui             # Workbench only
sozlab-cli selftest                       # the known-answer checks, on an installed copy
```

Known-answer tests use `engine/synthetic.py`: scripted occupancy, ideal gas (exact oracle), overdamped Brownian dynamics (analytic survival
with the discrete-monitoring correction), molecules split across the box, a rotating solute. They compare against brute-force oracles that
share no code with the engine.

## Reproducing the evidence

```bash
pip install -e ".[test,validation]"
python bench/repro/repro_v2_flaws.py --label fixed --out bench/out/repro_fixed.json        # 13 flaws of v2.0.0, measured
git worktree add .baseline 3dcf498                                                         # the released v2.0.0
PYTHONPATH=.baseline/src python bench/repro/repro_v2_flaws.py --label baseline --out bench/out/repro_baseline.json
python bench/data/make_tip3p_box.py --out bench/out/tip3p --replicas 4 --ns 2.0 --dt-out 0.2          # OpenMM, fixed seeds
python bench/data/make_tip3p_box.py --out bench/out/tip3p_fine --replicas 1 --ns 0.4 --dt-out 0.02
python bench/validate/tip3p_validation.py && python bench/validate/dt_extrapolation.py
python bench/validate/perf_compare.py --label v2.1 --what soz                              # and with PYTHONPATH=.baseline/src
```

Results land in `docs/evidence/data/`. Trajectories are not committed; `meta.json` holds their SHA-256 sums.

## Packaging

`pyproject.toml` lists `app`, `app.workbench`, `engine` and `cli`. The GUI needs only PyQt6 and pyqtgraph (no WebEngine, no OpenGL): the 3D view
is a numpy/QPainter renderer, so it behaves the same on a remote desktop or a machine without a GPU.
