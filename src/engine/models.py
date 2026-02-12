"""Data models for SOZLab project and analysis configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SelectionSpec:
    label: str
    display_label: Optional[str] = None
    selection: Optional[str] = None
    atom_indices: Optional[List[int]] = None
    pdb_serials: Optional[List[int]] = None
    resid: Optional[int] = None
    resname: Optional[str] = None
    atomname: Optional[str] = None
    segid: Optional[str] = None
    chain: Optional[str] = None
    expect_count: Optional[int] = None
    require_unique: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "display_label": self.display_label,
            "selection": self.selection,
            "atom_indices": self.atom_indices,
            "pdb_serials": self.pdb_serials,
            "resid": self.resid,
            "resname": self.resname,
            "atomname": self.atomname,
            "segid": self.segid,
            "chain": self.chain,
            "expect_count": self.expect_count,
            "require_unique": self.require_unique,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionSpec":
        return cls(
            label=data.get("label", "selection"),
            display_label=data.get("display_label"),
            selection=data.get("selection"),
            atom_indices=data.get("atom_indices"),
            pdb_serials=data.get("pdb_serials"),
            resid=data.get("resid"),
            resname=data.get("resname"),
            atomname=data.get("atomname"),
            segid=data.get("segid"),
            chain=data.get("chain"),
            expect_count=data.get("expect_count"),
            require_unique=bool(data.get("require_unique", False)),
        )


# Backward-compatible alias
SeedSpec = SelectionSpec


@dataclass
class ProbeConfig:
    """Probe definition for solvent positioning."""
    selection: str = "name O OW OH2"
    position: str = "atom"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection": self.selection,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeConfig":
        return cls(
            selection=str(data.get("selection", "")),
            position=str(data.get("position", "atom")),
        )


@dataclass
class SolventConfig:
    solvent_label: str = "Water"
    water_resnames: List[str] = field(default_factory=lambda: ["SOL", "WAT", "TIP3", "HOH"])
    water_oxygen_names: List[str] = field(default_factory=lambda: ["O", "OW", "OH2"])
    water_hydrogen_names: List[str] = field(default_factory=lambda: ["H1", "H2", "HW1", "HW2"])
    ion_resnames: List[str] = field(default_factory=lambda: ["NA", "CL", "K", "CA", "MG"])
    include_ions: bool = False
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    probe_source: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solvent_label": self.solvent_label,
            "water_resnames": self.water_resnames,
            "water_oxygen_names": self.water_oxygen_names,
            "water_hydrogen_names": self.water_hydrogen_names,
            "ion_resnames": self.ion_resnames,
            "include_ions": self.include_ions,
            "probe": self.probe.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolventConfig":
        resnames = data.get("solvent_resnames", data.get("water_resnames", ["SOL", "WAT", "TIP3", "HOH"]))
        oxygen_names = list(data.get("water_oxygen_names", ["O", "OW", "OH2"]))

        probe = None
        probe_source = None
        probe_data = data.get("probe")
        if isinstance(probe_data, dict):
            probe = ProbeConfig.from_dict(probe_data)
            probe_source = "probe"
        else:
            selection = str(data.get("probe_selection", "") or "")
            position = str(data.get("probe_position", "") or "")
            if selection or position:
                probe = ProbeConfig(selection=selection, position=position or "atom")
                probe_source = "probe_fields"
            elif oxygen_names:
                selection = "name " + " ".join(oxygen_names)
                probe = ProbeConfig(selection=selection, position="atom")
                probe_source = "legacy_water_oxygen_names"
            else:
                probe = ProbeConfig()
                probe_source = "default_probe"

        cfg = cls(
            solvent_label=str(data.get("solvent_label", "Water")),
            water_resnames=list(resnames),
            water_oxygen_names=oxygen_names,
            water_hydrogen_names=list(data.get("water_hydrogen_names", ["H1", "H2", "HW1", "HW2"])),
            ion_resnames=list(data.get("ion_resnames", ["NA", "CL", "K", "CA", "MG"])),
            include_ions=bool(data.get("include_ions", False)),
            probe=probe,
        )
        cfg.probe_source = probe_source
        return cfg


@dataclass
class SOZNode:
    """Logic tree node for a SOZ definition."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["SOZNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "params": self.params,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOZNode":
        return cls(
            type=data.get("type", "and"),
            params=dict(data.get("params", {})),
            children=[cls.from_dict(child) for child in data.get("children", [])],
        )


@dataclass
class SOZDefinition:
    name: str
    description: str
    root: SOZNode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOZDefinition":
        return cls(
            name=data.get("name", "SOZ"),
            description=data.get("description", ""),
            root=SOZNode.from_dict(data.get("root", {"type": "and", "children": []})),
        )


@dataclass
class AnalysisOptions:
    frame_start: int = 0
    frame_stop: Optional[int] = None
    stride: int = 1
    gap_tolerance: int = 0
    workers: Optional[int] = None
    store_ids: bool = True
    store_min_distances: bool = True
    preview_frames: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "stride": self.stride,
            "gap_tolerance": self.gap_tolerance,
            "workers": self.workers,
            "store_ids": self.store_ids,
            "store_min_distances": self.store_min_distances,
            "preview_frames": self.preview_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisOptions":
        return cls(
            frame_start=int(data.get("frame_start", 0)),
            frame_stop=data.get("frame_stop"),
            stride=int(data.get("stride", 1)),
            gap_tolerance=int(data.get("gap_tolerance", 0)),
            workers=data.get("workers"),
            store_ids=bool(data.get("store_ids", True)),
            store_min_distances=bool(data.get("store_min_distances", True)),
            preview_frames=int(data.get("preview_frames", 200)),
        )


@dataclass
class DistanceBridgeConfig:
    name: str
    selection_a: str
    selection_b: str
    cutoff_a: float
    cutoff_b: float
    unit: str = "A"
    atom_mode: str = "probe"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "selection_a": self.selection_a,
            "selection_b": self.selection_b,
            "cutoff_a": self.cutoff_a,
            "cutoff_b": self.cutoff_b,
            "unit": self.unit,
            "probe_mode": self.atom_mode,
            "atom_mode": self.atom_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceBridgeConfig":
        mode = data.get("probe_mode", data.get("atom_mode", "probe"))
        return cls(
            name=data.get("name", "bridge"),
            selection_a=data.get("selection_a", data.get("seed_a", "selection_a")),
            selection_b=data.get("selection_b", data.get("seed_b", "selection_b")),
            cutoff_a=float(data.get("cutoff_a", 3.5)),
            cutoff_b=float(data.get("cutoff_b", 3.5)),
            unit=data.get("unit", "A"),
            atom_mode=mode,
        )

    @property
    def seed_a(self) -> str:
        return self.selection_a

    @property
    def seed_b(self) -> str:
        return self.selection_b

    @seed_a.setter
    def seed_a(self, value: str) -> None:
        self.selection_a = value

    @seed_b.setter
    def seed_b(self, value: str) -> None:
        self.selection_b = value




@dataclass
class HbondWaterBridgeConfig:
    name: str
    selection_a: str
    selection_b: str
    distance: float = 3.0
    angle: float = 150.0
    water_selection: Optional[str] = None
    donors_selection: Optional[str] = None
    hydrogens_selection: Optional[str] = None
    acceptors_selection: Optional[str] = None
    update_selections: bool = True
    backend: str = "auto"  # "auto", "waterbridge", "hbond_analysis"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "selection_a": self.selection_a,
            "selection_b": self.selection_b,
            "distance": self.distance,
            "angle": self.angle,
            "water_selection": self.water_selection,
            "donors_selection": self.donors_selection,
            "hydrogens_selection": self.hydrogens_selection,
            "acceptors_selection": self.acceptors_selection,
            "update_selections": self.update_selections,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HbondWaterBridgeConfig":
        return cls(
            name=data.get("name", "hbond_bridge"),
            selection_a=data.get("selection_a", "protein"),
            selection_b=data.get("selection_b", "protein"),
            distance=float(data.get("distance", data.get("d_a_cutoff", 3.0))),
            angle=float(data.get("angle", data.get("d_h_a_angle", 150.0))),
            water_selection=data.get("water_selection"),
            donors_selection=data.get("donors_selection", data.get("donors_sel")),
            hydrogens_selection=data.get("hydrogens_selection", data.get("hydrogens_sel")),
            acceptors_selection=data.get("acceptors_selection", data.get("acceptors_sel")),
            update_selections=bool(data.get("update_selections", True)),
            backend=data.get("backend", "auto"),
        )


@dataclass
class HbondHydrationConfig:
    name: str
    residue_selection: str
    distance: float = 3.0
    angle: float = 150.0
    unit: str = "A"
    conditioning: str = "soz"
    soz_name: Optional[str] = None
    water_selection: Optional[str] = None
    donors_selection: Optional[str] = None
    hydrogens_selection: Optional[str] = None
    acceptors_selection: Optional[str] = None
    update_selections: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "residue_selection": self.residue_selection,
            "distance": self.distance,
            "angle": self.angle,
            "unit": self.unit,
            "conditioning": self.conditioning,
            "soz_name": self.soz_name,
            "water_selection": self.water_selection,
            "donors_selection": self.donors_selection,
            "hydrogens_selection": self.hydrogens_selection,
            "acceptors_selection": self.acceptors_selection,
            "update_selections": self.update_selections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HbondHydrationConfig":
        return cls(
            name=data.get("name", "hbond_hydration"),
            residue_selection=data.get("residue_selection", "protein"),
            distance=float(data.get("distance", data.get("d_a_cutoff", 3.0))),
            angle=float(data.get("angle", data.get("d_h_a_angle", 150.0))),
            unit=data.get("unit", "A"),
            conditioning=data.get("conditioning", data.get("mode", "soz")),
            soz_name=data.get("soz_name"),
            donors_selection=data.get("donors_selection", data.get("donors_sel")),
            hydrogens_selection=data.get("hydrogens_selection", data.get("hydrogens_sel")),
            acceptors_selection=data.get("acceptors_selection", data.get("acceptors_sel")),
            update_selections=bool(data.get("update_selections", True)),
            water_selection=data.get("water_selection"),
        )


@dataclass
class DensityMapConfig:
    name: str
    species_selection: str
    grid_spacing: float = 1.0
    padding: float = 2.0
    stride: int = 1
    frame_start: Optional[int] = None
    frame_stop: Optional[int] = None
    align: bool = False
    align_selection: Optional[str] = None
    align_reference: str = "first_frame"
    align_reference_path: Optional[str] = None
    output_format: str = "dx"
    view_mode: str = "physical"
    conditioning_policy: str = "strict"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "species_selection": self.species_selection,
            "grid_spacing": self.grid_spacing,
            "padding": self.padding,
            "stride": self.stride,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "align": self.align,
            "align_selection": self.align_selection,
            "align_reference": self.align_reference,
            "align_reference_path": self.align_reference_path,
            "output_format": self.output_format,
            "view_mode": self.view_mode,
            "conditioning_policy": self.conditioning_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DensityMapConfig":
        return cls(
            name=data.get("name", "density_map"),
            species_selection=data.get("species_selection", data.get("selection", "")),
            grid_spacing=float(data.get("grid_spacing", data.get("delta", 1.0))),
            padding=float(data.get("padding", 2.0)),
            stride=int(data.get("stride", 1)),
            frame_start=data.get("frame_start"),
            frame_stop=data.get("frame_stop"),
            align=bool(data.get("align", False)),
            align_selection=data.get("align_selection"),
            align_reference=data.get("align_reference", "first_frame"),
            align_reference_path=data.get("align_reference_path"),
            output_format=data.get("output_format", "dx"),
            view_mode=data.get("view_mode", "physical"),
            conditioning_policy=data.get("conditioning_policy", "strict"),
        )

    @property
    def selection(self) -> str:
        return self.species_selection

    @selection.setter
    def selection(self, value: str) -> None:
        self.species_selection = value


@dataclass
class WaterDynamicsConfig:
    name: str
    region_mode: str = "soz"
    soz_name: Optional[str] = None
    region_selection: Optional[str] = None
    region_cutoff: float = 3.5
    region_unit: str = "A"
    region_probe_mode: str = "probe"
    residence_mode: str = "continuous"
    hbond_distance: float = 3.0
    hbond_angle: float = 150.0
    solute_selection: Optional[str] = None
    water_selection: Optional[str] = None
    donors_selection: Optional[str] = None
    acceptors_selection: Optional[str] = None
    update_selections: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "region_mode": self.region_mode,
            "soz_name": self.soz_name,
            "region_selection": self.region_selection,
            "region_cutoff": self.region_cutoff,
            "region_unit": self.region_unit,
            "region_probe_mode": self.region_probe_mode,
            "residence_mode": self.residence_mode,
            "hbond_distance": self.hbond_distance,
            "hbond_angle": self.hbond_angle,
            "solute_selection": self.solute_selection,
            "water_selection": self.water_selection,
            "donors_selection": self.donors_selection,
            "acceptors_selection": self.acceptors_selection,
            "update_selections": self.update_selections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaterDynamicsConfig":
        return cls(
            name=data.get("name", "water_dynamics"),
            region_mode=data.get("region_mode", data.get("mode", "soz")),
            soz_name=data.get("soz_name"),
            region_selection=data.get("region_selection"),
            region_cutoff=float(data.get("region_cutoff", 3.5)),
            region_unit=data.get("region_unit", "A"),
            region_probe_mode=data.get("region_probe_mode", data.get("probe_mode", "probe")),
            residence_mode=data.get("residence_mode", data.get("residence", "continuous")),
            hbond_distance=float(data.get("hbond_distance", data.get("distance", 3.0))),
            hbond_angle=float(data.get("hbond_angle", data.get("angle", 150.0))),
            solute_selection=data.get("solute_selection"),
            water_selection=data.get("water_selection"),
            donors_selection=data.get("donors_selection"),
            acceptors_selection=data.get("acceptors_selection"),
            update_selections=bool(data.get("update_selections", True)),
        )


# Backwards-compatible aliases
BridgeConfig = DistanceBridgeConfig



@dataclass
class InputConfig:
    topology: str
    trajectory: Optional[str] = None
    processed_trajectory: Optional[str] = None
    preprocessing_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "trajectory": self.trajectory,
            "processed_trajectory": self.processed_trajectory,
            "preprocessing_notes": self.preprocessing_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputConfig":
        return cls(
            topology=data.get("topology", ""),
            trajectory=data.get("trajectory"),
            processed_trajectory=data.get("processed_trajectory"),
            preprocessing_notes=data.get("preprocessing_notes"),
        )


@dataclass
class OutputConfig:
    output_dir: str
    write_per_frame: bool = True
    write_parquet: bool = False
    report_format: str = "html"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "write_per_frame": self.write_per_frame,
            "write_parquet": self.write_parquet,
            "report_format": self.report_format,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(
            output_dir=data.get("output_dir", "results"),
            write_per_frame=bool(data.get("write_per_frame", True)),
            write_parquet=bool(data.get("write_parquet", False)),
            report_format=data.get("report_format", "html"),
        )


@dataclass
class ExtractionConfig:
    soz_name: Optional[str] = None
    rule: str = "n_solvent>=1"
    min_run_length: int = 1
    gap_tolerance: int = 0
    output_format: str = "xtc"
    output_dir: str = "results/extracted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soz_name": self.soz_name,
            "rule": self.rule,
            "min_run_length": self.min_run_length,
            "gap_tolerance": self.gap_tolerance,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionConfig":
        return cls(
            soz_name=data.get("soz_name"),
            rule=data.get("rule", "n_solvent>=1"),
            min_run_length=int(data.get("min_run_length", 1)),
            gap_tolerance=int(data.get("gap_tolerance", 0)),
            output_format=data.get("output_format", "xtc"),
            output_dir=data.get("output_dir", "results/extracted"),
        )


@dataclass
class ProjectConfig:
    inputs: InputConfig
    solvent: SolventConfig
    selections: Dict[str, SelectionSpec]
    sozs: List[SOZDefinition]
    analysis: AnalysisOptions
    outputs: OutputConfig
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    distance_bridges: List[DistanceBridgeConfig] = field(default_factory=list)
    hbond_water_bridges: List[HbondWaterBridgeConfig] = field(default_factory=list)

    hbond_hydration: List[HbondHydrationConfig] = field(default_factory=list)
    density_maps: List[DensityMapConfig] = field(default_factory=list)
    water_dynamics: List[WaterDynamicsConfig] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "inputs": self.inputs.to_dict(),
            "solvent": self.solvent.to_dict(),
            "selections": {name: sel.to_dict() for name, sel in self.selections.items()},
            "sozs": [soz.to_dict() for soz in self.sozs],
            "analysis": self.analysis.to_dict(),
            "outputs": self.outputs.to_dict(),
            "extraction": self.extraction.to_dict(),
            "distance_bridges": [bridge.to_dict() for bridge in self.distance_bridges],
            "hbond_water_bridges": [bridge.to_dict() for bridge in self.hbond_water_bridges],

            "hbond_hydration": [cfg.to_dict() for cfg in self.hbond_hydration],
            "density_maps": [cfg.to_dict() for cfg in self.density_maps],
            "water_dynamics": [cfg.to_dict() for cfg in self.water_dynamics],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        inputs = InputConfig.from_dict(data.get("inputs", {}))
        solvent = SolventConfig.from_dict(data.get("solvent", {}))
        selections_raw = data.get("selections", data.get("seeds", {})) or {}
        selections = {name: SelectionSpec.from_dict(sel) for name, sel in selections_raw.items()}
        sozs = [SOZDefinition.from_dict(item) for item in data.get("sozs", [])]
        analysis = AnalysisOptions.from_dict(data.get("analysis", {}))
        outputs = OutputConfig.from_dict(data.get("outputs", {}))
        extraction = ExtractionConfig.from_dict(data.get("extraction", {}))
        distance_bridges = [
            DistanceBridgeConfig.from_dict(item)
            for item in data.get("distance_bridges", data.get("bridges", []))
        ]
        hbond_water_bridges = [
            HbondWaterBridgeConfig.from_dict(item)
            for item in data.get("hbond_water_bridges", [])
        ]

        hbond_hydration = [
            HbondHydrationConfig.from_dict(item) for item in data.get("hbond_hydration", [])
        ]
        density_maps = [DensityMapConfig.from_dict(item) for item in data.get("density_maps", [])]
        water_dynamics = [
            WaterDynamicsConfig.from_dict(item) for item in data.get("water_dynamics", [])
        ]
        return cls(
            inputs=inputs,
            solvent=solvent,
            selections=_normalize_selection_labels(
                selections,
                sozs,
                distance_bridges,
                hbond_water_bridges,
            ),
            sozs=sozs,
            analysis=analysis,
            outputs=outputs,
            extraction=extraction,
            distance_bridges=distance_bridges,
            hbond_water_bridges=hbond_water_bridges,
            hbond_hydration=hbond_hydration,
            density_maps=density_maps,
            water_dynamics=water_dynamics,
            version=data.get("version", "1.0"),
        )

    @property
    def seeds(self) -> Dict[str, SelectionSpec]:
        return self.selections

    @seeds.setter
    def seeds(self, value: Dict[str, SelectionSpec]) -> None:
        self.selections = value

    @property
    def bridges(self) -> List[DistanceBridgeConfig]:
        return self.distance_bridges

    @bridges.setter
    def bridges(self, value: List[DistanceBridgeConfig]) -> None:
        self.distance_bridges = value



def _normalize_selection_labels(
    selections: Dict[str, SelectionSpec],
    sozs: List[SOZDefinition],
    distance_bridges: List[DistanceBridgeConfig],
    hbond_bridges: List[HbondWaterBridgeConfig],
) -> Dict[str, SelectionSpec]:
    mapping = {}
    normalized = {}
    for label, spec in selections.items():
        new_label = label.replace("seed_", "selection_") if label.startswith("seed_") else label
        mapping[label] = new_label
        if new_label in normalized:
            normalized[new_label] = spec
        else:
            spec.label = spec.label.replace("seed_", "selection_") if spec.label else new_label
            normalized[new_label] = spec

    if mapping:
        for soz in sozs:
            _rewrite_soz_labels(soz.root, mapping)
        for bridge in distance_bridges:
            if bridge.selection_a in mapping:
                bridge.selection_a = mapping[bridge.selection_a]
            if bridge.selection_b in mapping:
                bridge.selection_b = mapping[bridge.selection_b]
        for bridge in hbond_bridges:
            if bridge.selection_a in mapping:
                bridge.selection_a = mapping[bridge.selection_a]
            if bridge.selection_b in mapping:
                bridge.selection_b = mapping[bridge.selection_b]
    return normalized


def _rewrite_soz_labels(node: SOZNode, mapping: Dict[str, str]) -> None:
    if node.params:
        seed_label = node.params.get("seed_label") or node.params.get("selection_label") or node.params.get("seed")
        if seed_label in mapping:
            node.params["selection_label"] = mapping[seed_label]
            node.params.pop("seed_label", None)
            node.params.pop("seed", None)
    for child in node.children:
        _rewrite_soz_labels(child, mapping)


def default_project(topology: str, trajectory: Optional[str]) -> ProjectConfig:
    return ProjectConfig(
        inputs=InputConfig(topology=topology, trajectory=trajectory),
        solvent=SolventConfig(),
        selections={},
        sozs=[],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="results"),
        extraction=ExtractionConfig(),
    )
