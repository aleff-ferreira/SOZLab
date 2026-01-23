"""Data models for SOZLab project and analysis configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SelectionSpec:
    label: str
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
class SolventConfig:
    water_resnames: List[str] = field(default_factory=lambda: ["SOL", "WAT", "TIP3", "HOH"])
    water_oxygen_names: List[str] = field(default_factory=lambda: ["O", "OW", "OH2"])
    water_hydrogen_names: List[str] = field(default_factory=lambda: ["H1", "H2", "HW1", "HW2"])
    ion_resnames: List[str] = field(default_factory=lambda: ["NA", "CL", "K", "CA", "MG"])
    include_ions: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "water_resnames": self.water_resnames,
            "water_oxygen_names": self.water_oxygen_names,
            "water_hydrogen_names": self.water_hydrogen_names,
            "ion_resnames": self.ion_resnames,
            "include_ions": self.include_ions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolventConfig":
        return cls(
            water_resnames=list(data.get("water_resnames", ["SOL", "WAT", "TIP3", "HOH"])),
            water_oxygen_names=list(data.get("water_oxygen_names", ["O", "OW", "OH2"])),
            water_hydrogen_names=list(data.get("water_hydrogen_names", ["H1", "H2", "HW1", "HW2"])),
            ion_resnames=list(data.get("ion_resnames", ["NA", "CL", "K", "CA", "MG"])),
            include_ions=bool(data.get("include_ions", False)),
        )


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
    store_ids: bool = True
    store_min_distances: bool = True
    preview_frames: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "stride": self.stride,
            "gap_tolerance": self.gap_tolerance,
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
            store_ids=bool(data.get("store_ids", True)),
            store_min_distances=bool(data.get("store_min_distances", True)),
            preview_frames=int(data.get("preview_frames", 200)),
        )


@dataclass
class BridgeConfig:
    name: str
    selection_a: str
    selection_b: str
    cutoff_a: float
    cutoff_b: float
    unit: str = "A"
    atom_mode: str = "O"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "selection_a": self.selection_a,
            "selection_b": self.selection_b,
            "cutoff_a": self.cutoff_a,
            "cutoff_b": self.cutoff_b,
            "unit": self.unit,
            "atom_mode": self.atom_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeConfig":
        return cls(
            name=data.get("name", "bridge"),
            selection_a=data.get("selection_a", data.get("seed_a", "selection_a")),
            selection_b=data.get("selection_b", data.get("seed_b", "selection_b")),
            cutoff_a=float(data.get("cutoff_a", 3.5)),
            cutoff_b=float(data.get("cutoff_b", 3.5)),
            unit=data.get("unit", "A"),
            atom_mode=data.get("atom_mode", "O"),
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
class ResidueHydrationConfig:
    name: str
    residue_selection: str
    cutoff: float
    unit: str = "A"
    soz_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "residue_selection": self.residue_selection,
            "cutoff": self.cutoff,
            "unit": self.unit,
            "soz_name": self.soz_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidueHydrationConfig":
        return cls(
            name=data.get("name", "hydration"),
            residue_selection=data.get("residue_selection", "protein"),
            cutoff=float(data.get("cutoff", 3.5)),
            unit=data.get("unit", "A"),
            soz_name=data.get("soz_name"),
        )


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
    bridges: List[BridgeConfig] = field(default_factory=list)
    residue_hydration: List[ResidueHydrationConfig] = field(default_factory=list)
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
            "bridges": [bridge.to_dict() for bridge in self.bridges],
            "residue_hydration": [cfg.to_dict() for cfg in self.residue_hydration],
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
        bridges = [BridgeConfig.from_dict(item) for item in data.get("bridges", [])]
        residue_hydration = [ResidueHydrationConfig.from_dict(item) for item in data.get("residue_hydration", [])]
        return cls(
            inputs=inputs,
            solvent=solvent,
            selections=_normalize_selection_labels(selections, sozs, bridges),
            sozs=sozs,
            analysis=analysis,
            outputs=outputs,
            extraction=extraction,
            bridges=bridges,
            residue_hydration=residue_hydration,
            version=data.get("version", "1.0"),
        )

    @property
    def seeds(self) -> Dict[str, SelectionSpec]:
        return self.selections

    @seeds.setter
    def seeds(self, value: Dict[str, SelectionSpec]) -> None:
        self.selections = value


def _normalize_selection_labels(
    selections: Dict[str, SelectionSpec],
    sozs: List[SOZDefinition],
    bridges: List[BridgeConfig],
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
        for bridge in bridges:
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
