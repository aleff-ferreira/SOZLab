# Design Notes

## SOZ Logic-Tree Representation
SOZ definitions are stored as a tree of `SOZNode` objects. Each node returns a set of solvent residue indices:
- `distance`: solvent residues within a cutoff of a selection atom group.
- `shell`: iterative shell expansion around a selection (cutoffs list).
- Boolean nodes: `and`, `or`, `not` combine child sets.

Nodes are serialized in project JSON files to ensure reproducibility and session portability.

## Periodic Boundary Conditions (PBC)
Distance calculations use MDAnalysis distance routines with `box` vectors from the trajectory:
- Preferred: `capped_distance` for efficient minimum-image distances.
- Fallback: `distance_array` if needed.
When no box vectors are present, the engine emits a warning and still computes Euclidean distances.

## Indexing Strategy
Selection resolution supports:
- MDAnalysis selection strings
- GROMACS 1-based atom indices
- PDB serials and resid/resname/atomname constraints

Solvent identifiers are stable strings of `resname:resid:segid`, and per-frame lists are ordered deterministically by `(resname, resid, segid, resindex)`.

## Performance Strategy
- Trajectory streaming: frames are processed iteratively; no full-trajectory load.
- Vectorized distances: `capped_distance` and `distance_array` for batch computations.
- GUI responsiveness: analysis runs in a worker thread with progress updates.
- Plot downsampling in the GUI for large datasets.

## Reproducibility
Project JSON files capture inputs, SOZ definitions, analysis options, and outputs. The report includes metadata snapshots for auditability.
