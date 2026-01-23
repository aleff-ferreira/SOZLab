"""Unit handling utilities."""
from __future__ import annotations


def to_internal_length(value: float, unit: str) -> float:
    """Convert a length in user units into the internal (angstrom) units."""
    unit_norm = unit.strip().lower()
    if unit_norm in ("nm", "nanometer", "nanometers"):
        return float(value) * 10.0
    if unit_norm in ("a", "ang", "angstrom", "angstroms"):
        return float(value)
    raise ValueError(f"Unsupported unit: {unit}")


def from_internal_length(value_angstrom: float, unit: str) -> float:
    """Convert an internal (angstrom) length into the requested unit."""
    unit_norm = unit.strip().lower()
    if unit_norm in ("nm", "nanometer", "nanometers"):
        return float(value_angstrom) * 0.1
    if unit_norm in ("a", "ang", "angstrom", "angstroms"):
        return float(value_angstrom)
    raise ValueError(f"Unsupported unit: {unit}")


def to_nm(value: float, unit: str) -> float:
    """Convert a length to nanometers, regardless of input unit."""
    return from_internal_length(to_internal_length(value, unit), "nm")


def nm_to_unit(value_nm: float, unit: str) -> float:
    """Convert a nanometer value into the requested unit."""
    return from_internal_length(to_internal_length(value_nm, "nm"), unit)
