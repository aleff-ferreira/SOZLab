"""Shared helpers for engine tests (synthetic systems with known answers)."""
from __future__ import annotations

import numpy as np

from engine.models import (
    AnalysisOptions,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SelectionSpec,
    SolventConfig,
    SOZDefinition,
    SOZNode,
)


def make_project(topology, trajectory, out_dir, cutoff=3.5, analysis=None, density=None,
                 probe=("name OW", "atom"), node=None, soz_name="Z", extra_sozs=()):
    root = node or SOZNode(type="distance", params={"selection_label": "sel", "cutoff": cutoff, "unit": "A"})
    sozs = [SOZDefinition(name=soz_name, description="test", root=root), *extra_sozs]
    return ProjectConfig(
        inputs=InputConfig(topology=str(topology), trajectory=str(trajectory) if trajectory else None),
        solvent=SolventConfig(water_resnames=["SOL"], probe=ProbeConfig(selection=probe[0], position=probe[1])),
        selections={"sel": SelectionSpec(label="sel", selection="resname LIG")},
        sozs=sozs,
        analysis=analysis or AnalysisOptions(store_min_distances=False, workers=1),
        outputs=OutputConfig(output_dir=str(out_dir), write_per_frame=True),
        density_maps=list(density or []),
    )


def oracle_n_solvent(positions: np.ndarray, box: np.ndarray, cutoff: float = 3.5, solute: int = 0, stride: int = 3):
    """Independent float64 minimum-image count of waters whose first atom is within ``cutoff`` of the solute atom."""
    out = []
    for frame in positions:
        d = frame[1::stride] - frame[solute]
        d -= np.round(d / box) * box
        out.append(int((np.linalg.norm(d, axis=1) <= cutoff).sum()))
    return np.array(out)
