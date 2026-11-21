"""Workbench domain model: what the user is defining (solvent, zone, run) and how it compiles to an engine project.

Kept free of widgets so the whole flow can be driven and tested without a display."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.models import (
    AnalysisOptions,
    DensityMapConfig,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SelectionSpec,
    SolventConfig,
    SOZDefinition,
    SOZNode,
)

TOPOLOGY_EXT = (".gro", ".pdb", ".psf", ".tpr", ".prmtop", ".parm7", ".top", ".mol2", ".pqr", ".data", ".xyz", ".cif", ".mmtf")
TRAJECTORY_EXT = (".xtc", ".trr", ".dcd", ".nc", ".ncdf", ".trj", ".mdcrd", ".crd", ".h5md", ".lammpstrj", ".dump", ".xyz", ".dms")


def classify_paths(paths: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Guess which dropped/selected file is the topology and which the trajectory."""
    notes: List[str] = []
    topo = traj = None
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".xtc", ".trr", ".dcd", ".nc", ".ncdf", ".mdcrd", ".h5md", ".lammpstrj", ".trj", ".dump", ".crd") and traj is None:
            traj = p
        elif ext in TOPOLOGY_EXT and topo is None:
            topo = p
        elif ext in TRAJECTORY_EXT and traj is None:
            traj = p
        else:
            notes.append(f"Ignored {os.path.basename(p)} (unrecognised or duplicate).")
    if topo is None and traj is not None and os.path.splitext(traj)[1].lower() in (".xyz", ".pdb"):
        topo, traj = traj, None  # a single multi-frame file serves as both
    return topo, traj, notes


@dataclass
class SolventSpec:
    resnames: List[str] = field(default_factory=lambda: ["SOL", "WAT", "TIP3", "HOH"])
    probe_selection: str = "name O OW OH2"
    probe_position: str = "atom"
    include_ions: bool = False
    ion_resnames: List[str] = field(default_factory=lambda: ["NA", "CL", "K", "CA", "MG"])
    auto: bool = True

    def to_config(self) -> SolventConfig:
        return SolventConfig(
            water_resnames=list(self.resnames), include_ions=self.include_ions, ion_resnames=list(self.ion_resnames),
            probe=ProbeConfig(selection=self.probe_selection, position=self.probe_position),
        )


@dataclass
class ZoneSpec:
    """The zone, as a user thinks of it: solvent within ``cutoff`` of a selection (optionally AND/OR a second one)."""

    name: str = "Zone"
    anchor: str = ""
    cutoff: float = 3.5  # Angstrom
    unit: str = "A"
    updating: bool = False
    shells: List[float] = field(default_factory=list)  # advanced: extra radii measured from the selection
    anchor_b: str = ""
    cutoff_b: float = 3.5
    combine: str = "AND"
    probe_mode: str = "probe"

    @property
    def is_defined(self) -> bool:
        return bool(self.anchor.strip()) and self.cutoff > 0

    def describe(self) -> str:
        if not self.is_defined:
            return "No zone defined yet."
        unit = "Å" if self.unit.lower().startswith("a") else self.unit
        txt = f"solvent within {self.cutoff:g} {unit} of [{self.anchor}]"
        if self.shells:
            txt = f"solvent within {max([self.cutoff] + list(self.shells)):g} {unit} of [{self.anchor}] (shells at {', '.join(f'{s:g}' for s in [self.cutoff] + list(self.shells))} {unit})"
        if self.anchor_b.strip():
            txt = f"({txt}) {self.combine} (solvent within {self.cutoff_b:g} {unit} of [{self.anchor_b}])"
        return txt

    def compile(self) -> Tuple[Dict[str, SelectionSpec], SOZDefinition]:
        sels = {"anchor": SelectionSpec(label="anchor", selection=self.anchor.strip(), updating=self.updating)}
        if self.shells:
            radii = sorted(set([self.cutoff] + [float(s) for s in self.shells]))
            node_a = SOZNode("shell", {"selection_label": "anchor", "cutoffs": radii, "unit": self.unit, "mode": "radial",
                                       "probe_mode": self.probe_mode})
        else:
            node_a = SOZNode("distance", {"selection_label": "anchor", "cutoff": self.cutoff, "unit": self.unit,
                                          "probe_mode": self.probe_mode})
        root = node_a
        if self.anchor_b.strip():
            sels["anchor_b"] = SelectionSpec(label="anchor_b", selection=self.anchor_b.strip(), updating=self.updating)
            node_b = SOZNode("distance", {"selection_label": "anchor_b", "cutoff": self.cutoff_b, "unit": self.unit,
                                          "probe_mode": self.probe_mode})
            root = SOZNode("and" if self.combine.upper() == "AND" else "or", {}, [node_a, node_b])
        return sels, SOZDefinition(name=self.name or "Zone", description=self.describe(), root=root)


@dataclass
class RunSpec:
    frame_start: int = 0
    frame_stop: Optional[int] = None
    stride: int = 1
    gap_time: float = 0.0  # ps; 0 = off
    blocks: int = 5
    workers: int = 0  # 0 = auto
    stride_sensitivity: bool = True
    density: bool = False
    density_spacing: float = 1.0
    density_padding: float = 12.0
    density_species: str = ""
    density_align: str = ""
    output_dir: str = ""
    report_format: str = "html"
    write_per_frame: bool = True
    dt_override: Optional[float] = None


def default_output_dir(topology: str) -> str:
    base = os.path.dirname(os.path.abspath(topology)) if topology else os.getcwd()
    return os.path.join(base, "sozlab_results")


def build_project(inputs: InputConfig, solvent: SolventSpec, zone: ZoneSpec, run: RunSpec) -> ProjectConfig:
    sels, soz = zone.compile()
    analysis = AnalysisOptions(
        frame_start=run.frame_start, frame_stop=run.frame_stop, stride=max(1, run.stride),
        gap_tolerance_time=run.gap_time or None, uncertainty_blocks=run.blocks,
        workers=None if run.workers <= 0 else run.workers, stride_sensitivity=run.stride_sensitivity,
        store_ids=True, store_min_distances=True, dt_override=run.dt_override,
    )
    density_maps: List[DensityMapConfig] = []
    if run.density and zone.is_defined:
        density_maps.append(DensityMapConfig(
            name="solvent_density", species_selection=run.density_species or _species_selection(solvent),
            grid_spacing=run.density_spacing, align=True, align_selection=run.density_align or zone.anchor,
            region_selection=run.density_align or zone.anchor, region_padding=run.density_padding,
            conditioning_policy="strict", output_format="dx",
        ))
    return ProjectConfig(
        inputs=inputs, solvent=solvent.to_config(), selections=sels, sozs=[soz] if zone.is_defined else [],
        analysis=analysis,
        outputs=OutputConfig(output_dir=run.output_dir or default_output_dir(inputs.topology), write_per_frame=True,
                             report_format=run.report_format),
        density_maps=density_maps,
    )


def _species_selection(solvent: SolventSpec) -> str:
    names = " ".join(solvent.resnames)
    probe = solvent.probe_selection
    return f"resname {names} and {probe}" if names else probe
