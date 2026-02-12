"""Pre-flight checks for SOZLab analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import logging
import re
from typing import Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError

from engine.models import (
    ProjectConfig,
    SelectionSpec,
    SOZNode,
    DistanceBridgeConfig,

    HbondWaterBridgeConfig,
    HbondHydrationConfig,
    DensityMapConfig,
    WaterDynamicsConfig,
)
from engine.solvent import build_solvent, resolve_probe_mode
from engine.units import to_internal_length
from engine.resolver import sanitize_selection_string

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


def _collect_soz_selection_labels(node: SOZNode, labels: set[str]) -> None:
    if node.type in ("distance", "shell"):
        label = node.params.get("selection_label") or node.params.get("seed_label") or node.params.get("seed")
        if label:
            labels.add(label)
    for child in node.children:
        _collect_soz_selection_labels(child, labels)


def _unique_sorted(values: List[object]) -> List[object]:
    try:
        # Fast path for sortable types
        return sorted(list(set(values)))
    except TypeError:
        # Fallback for mixed types that can't be compared
        seen = set()
        out = []
        for value in values:
            if value is None or value == "":
                continue
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out


def collect_metadata_warnings(universe: mda.Universe) -> List[str]:
    messages: List[str] = []
    atoms = universe.atoms
    n_atoms = len(atoms)

    chainids = getattr(atoms, "chainIDs", None)
    if chainids is None:
        messages.append(
            "Topology has no chain IDs; PDB exports will use 'X'. Add chain IDs if you rely on chain-based selections."
        )
    else:
        chainids_arr = np.asarray(chainids, dtype=object)
        missing_mask = np.array(
            [cid is None or str(cid).strip() == "" for cid in chainids_arr],
            dtype=bool,
        )
        missing_count = int(missing_mask.sum())
        if missing_count == n_atoms and n_atoms > 0:
            messages.append(
                "All atoms are missing chain IDs; PDB exports will use 'X'. Add chain IDs if you rely on chain-based selections."
            )
        elif missing_count > 0:
            messages.append(
                f"{missing_count}/{n_atoms} atoms are missing chain IDs; PDB exports will use 'X' for those atoms."
            )
        invalid_mask = np.array(
            [
                cid is not None
                and str(cid).strip() != ""
                and len(str(cid).strip()) != 1
                for cid in chainids_arr
            ],
            dtype=bool,
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            messages.append(
                f"{invalid_count}/{n_atoms} atoms have chain IDs longer than 1 character; PDB exports will truncate them."
            )

    elements = getattr(atoms, "elements", None)
    if elements is None:
        messages.append(
            "Topology has no element information; element- or mass-based analyses and PDB parsing may be unreliable."
        )
    else:
        elements_arr = np.asarray(elements, dtype=object)
        missing_mask = np.array(
            [elem is None or str(elem).strip() == "" for elem in elements_arr],
            dtype=bool,
        )
        missing_count = int(missing_mask.sum())
        if missing_count == n_atoms and n_atoms > 0:
            messages.append(
                "All atoms are missing element information; element- or mass-based analyses and PDB parsing may be unreliable."
            )
        elif missing_count > 0:
            messages.append(
                f"{missing_count}/{n_atoms} atoms are missing element information; element- or mass-based analyses and PDB parsing may be unreliable."
            )

    masses = getattr(atoms, "masses", None)
    if masses is None:
        messages.append("Topology has no mass information; mass-based analyses may be unreliable.")
    else:
        try:
            masses_arr = np.asarray(masses, dtype=float)
            invalid_mask = np.isnan(masses_arr) | (masses_arr <= 0)
        except (TypeError, ValueError):
            invalid_mask = np.array(
                [mass is None or mass == 0 for mass in masses],
                dtype=bool,
            )
        invalid_count = int(np.sum(invalid_mask))
        if invalid_count == n_atoms and n_atoms > 0:
            messages.append("All atom masses are missing or zero; mass-based analyses may be unreliable.")
        elif invalid_count > 0:
            messages.append(
                f"{invalid_count}/{n_atoms} atom masses are missing or zero; mass-based analyses may be unreliable."
            )

    return messages


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

    resname_match = re.search(r"\bresname\s+([^\s)]+)", selection, re.IGNORECASE)
    if resname_match:
        requested_resname = resname_match.group(1).strip()
        resnames_present = sorted({str(res.resname).strip() for res in universe.residues if str(res.resname).strip()})
        upper_to_original = {name.upper(): name for name in resnames_present}
        if requested_resname.upper() not in upper_to_original:
            close = difflib.get_close_matches(
                requested_resname.upper(),
                list(upper_to_original.keys()),
                n=3,
                cutoff=0.45,
            )
            if close:
                suggestions.append(
                    "Did you mean resname "
                    + ", ".join(upper_to_original[key] for key in close)
                    + "?"
                )
            solvent_like = [
                upper_to_original[key]
                for key in ("TIP3", "HOH", "WAT", "SOL", "TIP4", "TIP5")
                if key in upper_to_original
            ]
            if solvent_like:
                suggestions.append(
                    "Available solvent-like resnames: " + ", ".join(solvent_like)
                )
            if resnames_present:
                suggestions.append(
                    "Available resnames (sample): " + ", ".join(resnames_present[:12])
                )

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
    raw_selection = spec.selection or ""
    selection = sanitize_selection_string(raw_selection)
    prefix_suggestions: List[str] = []
    if selection != raw_selection:
        prefix_suggestions.append(f"Selection normalized to: {selection}")
    if not selection:
        return SelectionCheck(
            label=spec.label,
            selection="",
            count=0,
            require_unique=spec.require_unique,
            expect_count=spec.expect_count,
            suggestions=prefix_suggestions + ["Selection has no selection string."],
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
            suggestions=prefix_suggestions + [f"Selection error: {exc}"],
        )

    # Optimize using vectorized access where possible
    # Note: group.resids is already available.
    resids = sorted(np.unique(group.resids).tolist()) 
    
    # MDAnalysis might not cache residue attributes on the group, but we can access them.
    # Safe way: iterate residues (much fewer than atoms)
    residues = group.residues
    
    # Vectorized attribute access if available, else list comp over residues
    # resnums
    if hasattr(residues, "resnums"):
         resnums = sorted(np.unique(residues.resnums).tolist())
    else:
         resnums = _unique_sorted([getattr(r, "resnum", None) for r in residues])

    # segids
    segids = sorted(np.unique([str(s) for s in residues.segids]).tolist())
    
    # chainIDs - check availability
    if hasattr(residues, "chainIDs"):
         chain_ids = sorted(np.unique([str(c) for c in residues.chainIDs]).tolist())
    else:
         chain_ids = _unique_sorted([getattr(r, "chainID", "") for r in residues])

    # moltypes
    if hasattr(residues, "moltypes"):
         moltypes = sorted(np.unique([str(m) for m in residues.moltypes]).tolist())
    else:
         moltypes = _unique_sorted([getattr(r, "moltype", "") for r in residues])

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
        suggestions=prefix_suggestions + suggestions,
    )


def _check_selection_string(
    universe: mda.Universe,
    selection: str,
    label: str,
) -> SelectionCheck:
    spec = SelectionSpec(label=label, selection=selection)
    return _check_selection(universe, spec)


def run_preflight(project: ProjectConfig, universe: mda.Universe) -> PreflightReport:
    errors: List[str] = []
    warnings: List[str] = []

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

    warnings.extend(collect_metadata_warnings(universe))

    # Solvent sanity
    solvent_cfg = project.solvent
    resnames_present = sorted({str(res.resname) for res in universe.residues})

    solvent_matches = []
    solvent_residue_counts = {}
    for name in solvent_cfg.water_resnames:
        try:
            count = len(universe.select_atoms(f"resname {name}").residues)
        except Exception:
            count = 0
        solvent_residue_counts[name] = count
        if count > 0:
            solvent_matches.append(name)

    if not solvent_matches:
        errors.append("Solvent resnames did not match any residues.")

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

    probe_summary = {
        "selection": solvent_cfg.probe.selection,
        "position": solvent_cfg.probe.position,
        "probe_source": getattr(solvent_cfg, "probe_source", None),
        "probe_atom_count": 0,
        "residues_with_probe": 0,
        "residues_multi_probe": 0,
        "probe_atoms_per_residue_min": 0,
        "probe_atoms_per_residue_max": 0,
        "probe_residues_missing_count": 0,
        "probe_residues_multi_count": 0,
        "probe_residues_missing_sample": [],
        "probe_residues_multi_sample": [],
    }

    if solvent_cfg.probe_source == "legacy_water_oxygen_names":
        warnings.append(
            "Probe selection derived from legacy water_oxygen_names; review probe settings for non-water solvents."
        )

    try:
        resolve_probe_mode("probe", solvent_cfg.probe.position)
        solvent = build_solvent(universe, solvent_cfg)
        probe_counts = {
            resindex: len(atom_indices)
            for resindex, atom_indices in solvent.probe.resindex_to_atom_indices.items()
        }
        missing = [idx for idx, count in probe_counts.items() if count == 0]
        multi = [idx for idx, count in probe_counts.items() if count > 1]
        probe_summary.update(
            {
                "probe_atom_count": len(solvent.probe.atom_indices),
                "residues_with_probe": sum(1 for count in probe_counts.values() if count > 0),
                "residues_multi_probe": sum(1 for count in probe_counts.values() if count > 1),
                "probe_atoms_per_residue_min": min(probe_counts.values()) if probe_counts else 0,
                "probe_atoms_per_residue_max": max(probe_counts.values()) if probe_counts else 0,
                "probe_residues_missing_count": len(missing),
                "probe_residues_multi_count": len(multi),
                "probe_residues_missing_sample": missing[:10],
                "probe_residues_multi_sample": multi[:10],
            }
        )
        if solvent.probe.position == "atom" and multi:
            warnings.append(
                "Probe selection matched multiple atoms per solvent residue; atom-mode distances may be ambiguous."
            )
        stable_ids = [record.stable_id for record in solvent.record_by_resindex.values()]
        if len(stable_ids) != len(set(stable_ids)):
            warnings.append(
                "Solvent residues do not have unique IDs (resname:resid:segid duplicates detected)."
            )
    except Exception as exc:
        errors.append(f"Solvent/probe validation failed: {exc}")

    solvent_summary = {
        "solvent_resnames": solvent_cfg.water_resnames,
        "solvent_residue_counts": solvent_residue_counts,
        "solvent_matches": solvent_matches,
        "probe_summary": probe_summary,
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

    def _check_node_modes(node: SOZNode, errors_out: List[str]) -> None:
        if node.type in ("distance", "shell"):
            raw_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
            try:
                resolve_probe_mode(raw_mode, project.solvent.probe.position)
            except Exception as exc:
                errors_out.append(f"Unsupported probe mode '{raw_mode}' in node '{node.type}': {exc}")
        for child in node.children:
            _check_node_modes(child, errors_out)

    def _collect_node_modes(node: SOZNode, modes: set[str]) -> None:
        if node.type in ("distance", "shell"):
            raw_mode = node.params.get("probe_mode", node.params.get("atom_mode", "probe"))
            try:
                modes.add(resolve_probe_mode(raw_mode, project.solvent.probe.position))
            except Exception:
                return
        for child in node.children:
            _collect_node_modes(child, modes)

    for soz in project.sozs:
        _check_node_units(soz.root, errors)
        _check_node_modes(soz.root, errors)

    for bridge in project.distance_bridges:
        try:
            to_internal_length(1.0, bridge.unit)
        except Exception:
            errors.append(
                f"Unsupported unit '{bridge.unit}' in distance bridge '{bridge.name}'."
            )
        try:
            resolve_probe_mode(bridge.atom_mode, project.solvent.probe.position)
        except Exception as exc:
            errors.append(
                f"Unsupported probe mode '{bridge.atom_mode}' in distance bridge '{bridge.name}': {exc}"
            )
        if bridge.selection_a not in project.selections:
            errors.append(
                f"Distance bridge '{bridge.name}' selection_a '{bridge.selection_a}' not found in selections."
            )
        if bridge.selection_b not in project.selections:
            errors.append(
                f"Distance bridge '{bridge.name}' selection_b '{bridge.selection_b}' not found in selections."
            )

    for bridge in project.hbond_water_bridges:
        for label_name, selection_value in (
            ("selection_a", bridge.selection_a),
            ("selection_b", bridge.selection_b),
        ):
            if selection_value in project.selections:
                continue
            sel_check = _check_selection_string(
                universe,
                selection_value,
                f"hbond_water_bridge:{bridge.name}:{label_name}",
            )
            if sel_check.count == 0:
                errors.append(
                    f"H-bond water bridge '{bridge.name}' {label_name} '{selection_value}' resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append("  Suggestions: " + " | ".join(sel_check.suggestions[:3]))
            else:
                warnings.append(
                    f"H-bond water bridge '{bridge.name}' {label_name} uses a raw selection string "
                    "(not a saved selection label)."
                )
        if bridge.distance <= 0:
            errors.append(
                f"H-bond water bridge '{bridge.name}' distance must be positive."
            )
        if bridge.angle <= 0:
            errors.append(f"H-bond water bridge '{bridge.name}' angle must be positive.")
        if bridge.water_selection:
            sel_check = _check_selection_string(
                universe, bridge.water_selection, f"hbond_water_bridge:{bridge.name}:water"
            )
            if sel_check.count == 0:
                warnings.append(
                    f"H-bond water bridge '{bridge.name}' water_selection resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append("  Suggestions: " + " | ".join(sel_check.suggestions[:3]))
        for label, sel in [
            ("donors", bridge.donors_selection),
            ("hydrogens", bridge.hydrogens_selection),
            ("acceptors", bridge.acceptors_selection),
        ]:
            if sel:
                sel_check = _check_selection_string(
                    universe, sel, f"hbond_water_bridge:{bridge.name}:{label}"
                )
                if sel_check.count == 0:
                    warnings.append(
                        f"H-bond water bridge '{bridge.name}' {label}_selection resolved to 0 atoms."
                    )
                    if sel_check.suggestions:
                        warnings.append(
                            "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                        )



    for cfg in project.hbond_hydration:
        if cfg.conditioning not in ("soz", "all"):
            errors.append(
                f"H-bond hydration '{cfg.name}' conditioning must be 'soz' or 'all'."
            )
        sel_check = _check_selection_string(
            universe, cfg.residue_selection, f"hbond_hydration:{cfg.name}"
        )
        if sel_check.count == 0:
            warnings.append(
                f"H-bond hydration '{cfg.name}' residue_selection resolved to 0 atoms."
            )
            if sel_check.suggestions:
                warnings.append("  Suggestions: " + " | ".join(sel_check.suggestions[:3]))
        if cfg.water_selection:
            sel_check = _check_selection_string(
                universe, cfg.water_selection, f"hbond_hydration:{cfg.name}:water"
            )
            if sel_check.count == 0:
                warnings.append(
                    f"H-bond hydration '{cfg.name}' water_selection resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append("  Suggestions: " + " | ".join(sel_check.suggestions[:3]))
        for label, sel in [
            ("donors", cfg.donors_selection),
            ("hydrogens", cfg.hydrogens_selection),
            ("acceptors", cfg.acceptors_selection),
        ]:
            if sel:
                sel_check = _check_selection_string(
                    universe, sel, f"hbond_hydration:{cfg.name}:{label}"
                )
                if sel_check.count == 0:
                    warnings.append(
                        f"H-bond hydration '{cfg.name}' {label}_selection resolved to 0 atoms."
                    )
                    if sel_check.suggestions:
                        warnings.append(
                            "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                        )

    for cfg in project.density_maps:
        if cfg.grid_spacing <= 0:
            errors.append(
                f"Density map '{cfg.name}' grid_spacing must be positive."
            )
        if cfg.stride <= 0:
            errors.append(f"Density map '{cfg.name}' stride must be positive.")
        if not cfg.selection:
            errors.append(f"Density map '{cfg.name}' selection is empty.")
        else:
            sel_check = _check_selection_string(
                universe, cfg.selection, f"density_map:{cfg.name}"
            )
            if sel_check.count == 0:
                warnings.append(
                    f"Density map '{cfg.name}' selection resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append(
                        "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                    )
        if cfg.align and not cfg.align_selection:
            errors.append(
                f"Density map '{cfg.name}' align_selection required when alignment is enabled."
            )
        if cfg.align and cfg.align_selection:
            sel_check = _check_selection_string(
                universe, cfg.align_selection, f"density_map:{cfg.name}:align"
            )
            if sel_check.count == 0:
                warnings.append(
                    f"Density map '{cfg.name}' align_selection resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append(
                        "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                    )

    for cfg in project.water_dynamics:
        if cfg.region_mode not in ("soz", "selection"):
            errors.append(
                f"Water dynamics '{cfg.name}' region_mode must be 'soz' or 'selection'."
            )
        if cfg.region_mode == "soz" and cfg.soz_name:
            soz_names = {soz.name for soz in project.sozs}
            if cfg.soz_name not in soz_names:
                errors.append(
                    f"Water dynamics '{cfg.name}' soz_name '{cfg.soz_name}' not found."
                )
        if cfg.region_mode == "selection":
            if not cfg.region_selection:
                errors.append(
                    f"Water dynamics '{cfg.name}' region_selection required for selection mode."
                )
            else:
                sel_check = _check_selection_string(
                    universe, cfg.region_selection, f"water_dynamics:{cfg.name}:region"
                )
                if sel_check.count == 0:
                    warnings.append(
                        f"Water dynamics '{cfg.name}' region_selection resolved to 0 atoms."
                    )
                    if sel_check.suggestions:
                        warnings.append(
                            "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                        )
            try:
                resolve_probe_mode(cfg.region_probe_mode, project.solvent.probe.position)
            except Exception as exc:
                errors.append(
                    f"Unsupported probe mode '{cfg.region_probe_mode}' in water dynamics '{cfg.name}': {exc}"
                )
            try:
                to_internal_length(1.0, cfg.region_unit)
            except Exception:
                errors.append(
                    f"Unsupported unit '{cfg.region_unit}' in water dynamics '{cfg.name}'."
                )
        if cfg.residence_mode not in ("continuous", "intermittent"):
            errors.append(
                f"Water dynamics '{cfg.name}' residence_mode must be 'continuous' or 'intermittent'."
            )
        if cfg.solute_selection:
            sel_check = _check_selection_string(
                universe, cfg.solute_selection, f"water_dynamics:{cfg.name}:solute"
            )
            if sel_check.count == 0:
                warnings.append(
                    f"Water dynamics '{cfg.name}' solute_selection resolved to 0 atoms."
                )
                if sel_check.suggestions:
                    warnings.append(
                        "  Suggestions: " + " | ".join(sel_check.suggestions[:3])
                    )

    required_modes: set[str] = set()
    for soz in project.sozs:
        _collect_node_modes(soz.root, required_modes)
    for bridge in project.distance_bridges:
        try:
            required_modes.add(resolve_probe_mode(bridge.atom_mode, project.solvent.probe.position))
        except Exception:
            pass

    for cfg in project.water_dynamics:
        if cfg.region_mode == "selection":
            try:
                required_modes.add(
                    resolve_probe_mode(cfg.region_probe_mode, project.solvent.probe.position)
                )
            except Exception:
                pass
    if "com" in required_modes:
        try:
            _ = universe.atoms.masses
        except NoDataError:
            errors.append(
                "Probe position 'com' requires atom masses; add masses to topology or use COG."
            )

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

    required_labels: set[str] = set()
    for soz in project.sozs:
        _collect_soz_selection_labels(soz.root, required_labels)
    missing_labels = sorted(label for label in required_labels if label not in project.selections)
    if missing_labels:
        errors.append(
            "SOZ references missing selections: " + ", ".join(missing_labels)
        )

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
