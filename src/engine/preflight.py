"""Pre-flight checks for SOZLab analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List

import numpy as np
import MDAnalysis as mda

from engine.models import ProjectConfig, SelectionSpec, SOZNode, BridgeConfig, ResidueHydrationConfig
from engine.units import to_internal_length

logger = logging.getLogger("sozlab")


@dataclass
class SelectionCheck:
    label: str
    selection: str
    count: int
    require_unique: bool
    expect_count: int | None
    resids: List[int] = field(default_factory=list)
    resnums: List[int] = field(default_factory=list)
    segids: List[str] = field(default_factory=list)
    chain_ids: List[str] = field(default_factory=list)
    moltypes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    selection_checks: Dict[str, SelectionCheck]
    solvent_summary: Dict[str, object]
    trajectory_summary: Dict[str, object]
    pbc_summary: Dict[str, object]
    gmx_summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "selection_checks": {
                k: _selection_check_to_dict(v) for k, v in self.selection_checks.items()
            },
            "solvent_summary": self.solvent_summary,
            "trajectory_summary": self.trajectory_summary,
            "pbc_summary": self.pbc_summary,
            "gmx_summary": self.gmx_summary,
        }

    @property
    def seed_checks(self) -> Dict[str, "SelectionCheck"]:
        return self.selection_checks

    @seed_checks.setter
    def seed_checks(self, value: Dict[str, "SelectionCheck"]) -> None:
        self.selection_checks = value


def _selection_check_to_dict(check: SelectionCheck) -> Dict[str, object]:
    return {
        "label": check.label,
        "selection": check.selection,
        "count": check.count,
        "require_unique": check.require_unique,
        "expect_count": check.expect_count,
        "resids": check.resids,
        "resnums": check.resnums,
        "segids": check.segids,
        "chain_ids": check.chain_ids,
        "moltypes": check.moltypes,
        "suggestions": check.suggestions,
    }


_LEADING_OPERATOR = re.compile(r"^\s*(and|or|not)\b", re.IGNORECASE)
_RESID_RE = re.compile(r"\bresid\b", re.IGNORECASE)
_RESNUM_RE = re.compile(r"\bresnum\b", re.IGNORECASE)


def _unique_sorted(values: List[object]) -> List[object]:
    seen = []
    for value in values:
        if value is None or value == "":
            continue
        if value not in seen:
            seen.append(value)
    return seen


def _suggest_for_zero(
    universe: mda.Universe,
    selection: str,
) -> List[str]:
    suggestions: List[str] = []

    if _RESID_RE.search(selection) and not _RESNUM_RE.search(selection):
        alt = _RESID_RE.sub("resnum", selection)
        try:
            if len(universe.select_atoms(alt)) > 0:
                suggestions.append(f"Try resnum instead of resid: {alt}")
        except Exception:
            pass

    if _RESNUM_RE.search(selection) and not _RESID_RE.search(selection):
        alt = _RESNUM_RE.sub("resid", selection)
        try:
            if len(universe.select_atoms(alt)) > 0:
                suggestions.append(f"Try resid instead of resnum: {alt}")
        except Exception:
            pass

    if "resname" in selection.lower() and ("his" in selection.lower() or "hs" in selection.lower()):
        his_like = sorted(
            {
                str(res.resname)
                for res in universe.residues
                if str(res.resname).upper().startswith(("HS", "HI"))
            }
        )
        if his_like:
            suggestions.append("Histidine-like resnames present: " + ", ".join(his_like))
            suggestions.append("Example: resname HS* and name NE2")

    if "name ne2" in selection.lower():
        try:
            hits = universe.select_atoms("name NE2")
            if len(hits) > 0:
                resids = _unique_sorted([int(atom.residue.resid) for atom in hits])
                segids = _unique_sorted([str(atom.residue.segid) for atom in hits])
                suggestions.append(f"NE2 exists in resids: {', '.join(map(str, resids[:10]))}")
                if segids:
                    suggestions.append(f"NE2 segids: {', '.join(segids[:10])}")
        except Exception:
            pass

    if "name nz" in selection.lower():
        try:
            hits = universe.select_atoms("name NZ")
            if len(hits) > 0:
                resids = _unique_sorted([int(atom.residue.resid) for atom in hits])
                segids = _unique_sorted([str(atom.residue.segid) for atom in hits])
                suggestions.append(f"NZ exists in resids: {', '.join(map(str, resids[:10]))}")
                if segids:
                    suggestions.append(f"NZ segids: {', '.join(segids[:10])}")
        except Exception:
            pass

    return suggestions


def _suggest_for_multiple(group: mda.core.groups.AtomGroup) -> List[str]:
    suggestions: List[str] = []
    segids = _unique_sorted([str(atom.residue.segid) for atom in group])
    chain_ids = _unique_sorted([getattr(atom.residue, "chainID", "") for atom in group])
    moltypes = _unique_sorted([getattr(atom.residue, "moltype", "") for atom in group])
    resids = _unique_sorted([int(atom.residue.resid) for atom in group])
    resnums = _unique_sorted([getattr(atom.residue, "resnum", None) for atom in group])

    if segids:
        suggestions.append("Segids: " + ", ".join(segids[:10]))
    if chain_ids:
        suggestions.append("ChainIDs: " + ", ".join(map(str, chain_ids[:10])))
    if moltypes:
        suggestions.append("Moltypes: " + ", ".join(map(str, moltypes[:10])))
    if resids and len(resids) <= 20:
        suggestions.append("Resids: " + ", ".join(map(str, resids)))
    if resnums and len(resnums) <= 20:
        suggestions.append("Resnums: " + ", ".join(map(str, resnums)))
    if suggestions:
        suggestions.append("Suggestion: add segid/chainID/moltype and resid/resnum to narrow.")
    return suggestions


def _check_selection(universe: mda.Universe, spec: SelectionSpec) -> SelectionCheck:
    selection = spec.selection or ""
    if not selection:
        return SelectionCheck(
            label=spec.label,
            selection="",
            count=0,
            require_unique=spec.require_unique,
            expect_count=spec.expect_count,
            suggestions=["Selection has no selection string."],
        )
    try:
        group = universe.select_atoms(selection)
    except Exception as exc:
        return SelectionCheck(
            label=spec.label,
            selection=selection,
            count=0,
            require_unique=spec.require_unique,
            expect_count=spec.expect_count,
            suggestions=[f"Selection error: {exc}"],
        )

    resids = _unique_sorted([int(atom.residue.resid) for atom in group])
    resnums = _unique_sorted([getattr(atom.residue, "resnum", None) for atom in group])
    segids = _unique_sorted([str(atom.residue.segid) for atom in group])
    chain_ids = _unique_sorted([getattr(atom.residue, "chainID", "") for atom in group])
    moltypes = _unique_sorted([getattr(atom.residue, "moltype", "") for atom in group])

    suggestions: List[str] = []
    if len(group) == 0:
        suggestions.extend(_suggest_for_zero(universe, selection))
    elif len(group) > 1 and (spec.require_unique or spec.expect_count == 1):
        suggestions.extend(_suggest_for_multiple(group))

    return SelectionCheck(
        label=spec.label,
        selection=selection,
        count=len(group),
        require_unique=spec.require_unique,
        expect_count=spec.expect_count,
        resids=resids,
        resnums=resnums,
        segids=segids,
        chain_ids=chain_ids,
        moltypes=moltypes,
        suggestions=suggestions,
    )


def run_preflight(project: ProjectConfig, universe: mda.Universe) -> PreflightReport:
    errors: List[str] = []
    warnings: List[str] = []

    if not project.sozs:
        warnings.append("No SOZ definitions found in project.")

    # Trajectory sanity
    traj = universe.trajectory
    n_atoms = len(universe.atoms)
    traj_atoms = getattr(traj, "n_atoms", None)
    if traj_atoms is not None and traj_atoms != n_atoms:
        errors.append(
            f"Topology/trajectory atom count mismatch: topology {n_atoms}, trajectory {traj_atoms}."
        )
    if len(traj) == 0:
        errors.append("Trajectory contains 0 frames.")
    else:
        try:
            traj[0]
        except Exception as exc:
            errors.append(f"Trajectory not readable: {exc}")

    trajectory_summary = {
        "n_atoms": n_atoms,
        "n_frames": len(traj),
        "trajectory_atoms": traj_atoms,
        "input_topology": project.inputs.topology,
        "input_trajectory": project.inputs.trajectory,
        "processed_trajectory": getattr(project.inputs, "processed_trajectory", None),
        "preprocessing_notes": getattr(project.inputs, "preprocessing_notes", None),
    }

    # Solvent sanity
    solvent_cfg = project.solvent
    resnames_present = sorted({str(res.resname) for res in universe.residues})

    water_matches = []
    water_residue_counts = {}
    for name in solvent_cfg.water_resnames:
        try:
            count = len(universe.select_atoms(f"resname {name}").residues)
        except Exception:
            count = 0
        water_residue_counts[name] = count
        if count > 0:
            water_matches.append(name)

    if not water_matches:
        errors.append("Water resnames did not match any residues.")

    oxygen_atom_count = 0
    if water_matches:
        water_sel = "resname " + " ".join(water_matches)
        oxygen_sel = "name " + " ".join(solvent_cfg.water_oxygen_names)
        try:
            oxygen_atom_count = len(universe.select_atoms(f"{water_sel} and {oxygen_sel}"))
        except Exception:
            oxygen_atom_count = 0
    if oxygen_atom_count == 0:
        warnings.append("Water oxygen names did not match any atoms; O-mode will fall back to all atoms.")

    ion_matches = []
    ion_counts = {}
    if solvent_cfg.include_ions:
        for name in solvent_cfg.ion_resnames:
            try:
                count = len(universe.select_atoms(f"resname {name}").residues)
            except Exception:
                count = 0
            ion_counts[name] = count
            if count > 0:
                ion_matches.append(name)
        if not ion_matches:
            warnings.append("Ion resnames did not match any residues.")

    solvent_summary = {
        "water_resnames": solvent_cfg.water_resnames,
        "water_residue_counts": water_residue_counts,
        "water_matches": water_matches,
        "water_oxygen_names": solvent_cfg.water_oxygen_names,
        "oxygen_atom_count": oxygen_atom_count,
        "ion_resnames": solvent_cfg.ion_resnames,
        "ion_matches": ion_matches,
        "include_ions": solvent_cfg.include_ions,
        "resnames_present": resnames_present[:50],
    }

    # Unit sanity
    def _check_node_units(node: SOZNode, errors_out: List[str]) -> None:
        if node.type in ("distance", "shell"):
            unit = node.params.get("unit", "A")
            try:
                to_internal_length(1.0, unit)
            except Exception:
                errors_out.append(f"Unsupported unit '{unit}' in node '{node.type}'.")
        for child in node.children:
            _check_node_units(child, errors_out)

    for soz in project.sozs:
        _check_node_units(soz.root, errors)

    for bridge in project.bridges:
        try:
            to_internal_length(1.0, bridge.unit)
        except Exception:
            errors.append(f"Unsupported unit '{bridge.unit}' in bridge '{bridge.name}'.")

    for cfg in project.residue_hydration:
        try:
            to_internal_length(1.0, cfg.unit)
        except Exception:
            errors.append(f"Unsupported unit '{cfg.unit}' in hydration '{cfg.name}'.")

    # Selection sanity
    selection_checks: Dict[str, SelectionCheck] = {}
    for label, spec in project.selections.items():
        if spec.selection and _LEADING_OPERATOR.match(spec.selection):
            errors.append(f"Selection '{label}' starts with an operator: {spec.selection!r}")
        selection_check = _check_selection(universe, spec)
        selection_checks[label] = selection_check
        if selection_check.count == 0:
            msg = f"Selection '{label}' resolved to 0 atoms."
            if selection_check.suggestions:
                msg += " Suggestions: " + " | ".join(selection_check.suggestions[:3])
            errors.append(msg)
        if spec.require_unique and selection_check.count != 1:
            msg = (
                f"Selection '{label}' expected unique atom but resolved {selection_check.count} atoms."
            )
            if selection_check.suggestions:
                msg += " Suggestions: " + " | ".join(selection_check.suggestions[:3])
            errors.append(msg)
        if spec.expect_count is not None and selection_check.count != spec.expect_count:
            msg = (
                f"Selection '{label}' expected {spec.expect_count} atoms but resolved {selection_check.count}."
            )
            if selection_check.suggestions:
                msg += " Suggestions: " + " | ".join(selection_check.suggestions[:3])
            errors.append(msg)

    # PBC sanity
    dims = universe.trajectory.ts.dimensions if len(universe.trajectory) else None
    has_box = bool(dims is not None and np.all(np.array(dims[:3]) > 0))
    if not has_box:
        warnings.append("No valid box vectors found; PBC distances may be unreliable.")
    pbc_summary = {"has_box": has_box, "dimensions": dims.tolist() if dims is not None else None}

    # GROMACS availability
    try:
        import shutil
        import subprocess

        gmx_path = shutil.which("gmx_mpi") or shutil.which("gmx")
        gmx_version = None
        if gmx_path:
            gmx_cmd = "gmx_mpi" if gmx_path.endswith("gmx_mpi") else "gmx"
            result = subprocess.run(
                [gmx_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            gmx_version = result.stdout.strip().splitlines()[0] if result.stdout else None
    except Exception:
        gmx_path = None
        gmx_version = None

    gmx_summary = {"available": bool(gmx_path), "path": gmx_path, "version": gmx_version}

    ok = len(errors) == 0
    if errors:
        for error in errors:
            logger.error("Preflight error: %s", error)
    for warning in warnings:
        logger.warning("Preflight warning: %s", warning)

    return PreflightReport(
        ok=ok,
        errors=errors,
        warnings=warnings,
        selection_checks=selection_checks,
        solvent_summary=solvent_summary,
        trajectory_summary=trajectory_summary,
        pbc_summary=pbc_summary,
        gmx_summary=gmx_summary,
    )
