# Tutorial Changelog

## 2026-01-24
- Rewrote `docs/tutorial.md` to match the current SOZLab GUI stepper (Project/Define/QC/Explore/Export) and all visible controls, with explicit labels and workflows for each tab and panel.
- Updated all solvent-facing terminology to use "Solvent" in narrative text while preserving legacy JSON keys (`water_resnames`, `water_oxygen_names`, `water_hydrogen_names`) and documenting the new probe system.
- Added a complete CLI reference section using the current `sozlab-cli --help` outputs, with exact flags for `run`, `validate`, and `extract`.
- Expanded plot interpretation sections to define axes, counts vs fractions, and known pitfalls (stride, sparsity) for Timeline, Entry/Exit, Histogram, Heatmap, and Event Raster.
- Documented plot export formats (PNG/SVG/EMF/PDF/CSV), Office editability tips, and the exact CSV columns produced by each plot export.
- Clarified output directory precedence across GUI and CLI, plus extraction outputs and rule semantics.
- Rebuilt `docs/tutorial.pdf` from the updated markdown using the ReportLab build script.

## Docs QA checklist
- CLI help verification: ran `/home/aleff/anaconda3/envs/sozlab/bin/sozlab-cli --help`, `run --help`, `validate --help`, `extract --help`.
- UI verification: cross-checked labels, controls, and tabs against `src/app/main.py` (Wizard, Selection Builder, Bridges, Residue Hydration, Advanced, Selection Tester, Timeline/Entry-Exit, Plots, Tables, Extract).
- Schema verification: cross-checked keys in `src/engine/models.py` and examples (`examples/sample_project.json`, `examples/gromacs_template_project.json`).
- Output verification: cross-checked output paths and columns with `src/engine/export.py`, `src/engine/stats.py`, `src/engine/extract.py`.
- PDF build: regenerated `docs/tutorial.pdf` using `/home/aleff/anaconda3/envs/sozlab/bin/python scripts/build_tutorial_pdf.py`.
- Copy/paste hygiene: ensured all snippets in `docs/tutorial.md` are ASCII and use literal operators (e.g., `>=`).
- Known gap: did not text-extract the previous PDF due to missing PDF text tools; relied on updated markdown and rebuild.
