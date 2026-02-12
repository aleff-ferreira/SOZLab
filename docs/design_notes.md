# Design Notes

## 1. Core analysis model
SOZ definitions are stored as `SOZNode` logic trees and evaluated per frame as solvent-residue sets.
Primary occupancy metrics are derived from these sets (`n_solvent`, entries, exits, residence summaries).

## 2. Current product scope
Current UI/CLI scope is intentionally narrower than historical internal modules.
Exposed workflow includes:
- SOZ occupancy analysis
- distance bridges
- density maps and density exploration
- extraction/report/export

H-bond bridge, H-bond hydration, and water-dynamics configs are currently stripped at runtime.

## 3. UI simplification choices
Recent UI decisions prioritize clarity and reduced control clutter:
- Timeline uses fixed `n_solvent` as primary metric.
- Histogram uses a single unified distribution (no visible zero/non-zero split panel).
- Entry/Exit view keeps only mode + axis/sign controls.
- Distance Bridge explorer keeps the three main plots with a single export action.
- Density explorer keeps view/colormap/slice controls; no density-map dropdown in the top toolbar.

## 4. Plot context menu policy
Plot right-click menus are simplified to:
- `Export`
- `Plot Options` with only `Grid` and `Alpha`

This avoids exposing lower-level pyqtgraph actions that are not part of the intended user workflow.

## 5. 3D density viewer
The 3D viewer (`viz_3d.py`) uses local NGL assets and a Qt WebEngine bridge.
Current controls are grouped into sections:
- Density
- Layers
- Render
- Pick
- Insights

Structure selection for the 3D overlay is constrained to avoid over-large PDB exports that can break rendering.

## 6. PBC and units
Distance calculations use trajectory box vectors (minimum image) when available; warnings are emitted when box metadata is missing.
Length cutoffs accept `A` or `nm` in configs (converted internally as needed).
