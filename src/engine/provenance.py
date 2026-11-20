"""Provenance: what exactly produced a result, and is it still true?

Three problems in SOZLab 2.0.0 had one root cause - results were tied to *file names*,
not to *file contents*:

* the density cache was keyed on the trajectory **path**, so a rewritten trajectory
  returned the previous run's grid;
* frame extraction re-opened whatever the project currently pointed at, so frames were
  written from a different file than the one analysed;
* exported metadata recorded neither input identity nor software versions.

This module fingerprints inputs by content, captures the environment, hashes the
configuration, and provides the single verification primitive
(:func:`verify_fingerprint`) used by the cache, the extraction engine and the CLI.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from engine._version import ALGORITHM_VERSIONS, SCHEMA_VERSION, __version__

#: Files up to this size are hashed completely; larger files use a sampled hash plus size/mtime.
FULL_HASH_LIMIT = 512 * 1024**2
_HEAD_TAIL = 4 * 1024**2
_SAMPLE_BLOCK = 1024**2
_N_SAMPLE_BLOCKS = 64
_CHUNK = 8 * 1024**2


class InputsChangedError(RuntimeError):
    """Raised when an input file no longer matches the fingerprint recorded for a result."""

    def __init__(self, differences: List[str]):
        self.differences = list(differences)
        super().__init__("Input files changed since the analysis: " + "; ".join(self.differences))


def _hash_file(path: Path, mode: str, size: int) -> str:
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with open(path, "rb") as handle:
        if mode == "full" or size <= 2 * _HEAD_TAIL + _N_SAMPLE_BLOCKS * _SAMPLE_BLOCK:
            while True:
                block = handle.read(_CHUNK)
                if not block:
                    break
                h.update(block)
            return h.hexdigest()
        handle.seek(0)
        h.update(handle.read(_HEAD_TAIL))
        handle.seek(max(size - _HEAD_TAIL, 0))
        h.update(handle.read(_HEAD_TAIL))
        span = max(size - _SAMPLE_BLOCK, 1)
        for i in range(_N_SAMPLE_BLOCKS):
            handle.seek((i * span) // (_N_SAMPLE_BLOCKS - 1))
            h.update(handle.read(_SAMPLE_BLOCK))
    return h.hexdigest()


def file_fingerprint(path: str | os.PathLike, mode: str = "auto") -> Dict[str, Any]:
    """Content fingerprint of one file.

    ``mode='auto'`` hashes the whole file up to :data:`FULL_HASH_LIMIT` and a deterministic
    sample (head, tail and 64 evenly spaced blocks, plus the size) above it. The mode used is
    recorded so verification repeats exactly the same computation.
    """
    p = Path(path)
    st = p.stat()
    size = int(st.st_size)
    if mode == "auto":
        mode = "full" if size <= FULL_HASH_LIMIT else "sampled"
    if mode not in {"full", "sampled"}:
        raise ValueError(f"Unknown fingerprint mode: {mode}")
    return {
        "path": str(p.resolve()),
        "name": p.name,
        "size": size,
        "mtime_ns": int(st.st_mtime_ns),
        "hash_mode": mode,
        "sha256": _hash_file(p, mode, size),
    }


def verify_fingerprint(recorded: Dict[str, Any], path: Optional[str] = None) -> List[str]:
    """Differences between a recorded fingerprint and the file now at ``path`` (default: the recorded path).

    Content decides: a changed modification time with identical content is *not* a difference
    (copying or restoring a file is harmless), but a changed size or hash always is.
    """
    target = Path(path or recorded["path"])
    label = recorded.get("name") or target.name
    if not target.exists():
        return [f"{label}: file not found at {target}"]
    diffs: List[str] = []
    st = target.stat()
    if int(st.st_size) != int(recorded["size"]):
        diffs.append(f"{label}: size changed ({recorded['size']} -> {st.st_size} bytes)")
        return diffs
    now = _hash_file(target, recorded.get("hash_mode", "full"), int(st.st_size))
    if now != recorded["sha256"]:
        diffs.append(f"{label}: content changed (sha256 {recorded['sha256'][:12]}... -> {now[:12]}...)")
    return diffs


def _exists(path: Optional[str]) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except OSError:
        return False


def fingerprint_inputs(inputs, extra_paths: Iterable[str] = (), mode: str = "auto") -> Dict[str, Any]:
    """Fingerprint topology, trajectory and any alignment reference structures."""
    out: Dict[str, Any] = {"topology": None, "trajectory": None, "extra": []}
    if _exists(getattr(inputs, "topology", None)):
        out["topology"] = file_fingerprint(inputs.topology, mode)
    traj = getattr(inputs, "trajectory", None)
    if _exists(traj):
        out["trajectory"] = file_fingerprint(traj, mode)
    for extra in extra_paths:
        if _exists(extra):
            out["extra"].append(file_fingerprint(extra, mode))
    return out


def verify_inputs(recorded_inputs: Dict[str, Any], topology: Optional[str] = None,
                  trajectory: Optional[str] = None) -> List[str]:
    """Compare recorded input fingerprints with the files at the given (or recorded) paths."""
    diffs: List[str] = []
    topo_fp = recorded_inputs.get("topology")
    traj_fp = recorded_inputs.get("trajectory")
    if topo_fp:
        diffs.extend(verify_fingerprint(topo_fp, topology))
    if traj_fp:
        diffs.extend(verify_fingerprint(traj_fp, trajectory))
    elif trajectory:
        diffs.append("trajectory was not fingerprinted when the analysis ran")
    for extra in recorded_inputs.get("extra", []):
        diffs.extend(verify_fingerprint(extra))
    return diffs


def assert_inputs_unchanged(recorded_inputs: Dict[str, Any], topology: Optional[str] = None,
                            trajectory: Optional[str] = None) -> None:
    diffs = verify_inputs(recorded_inputs, topology, trajectory)
    if diffs:
        raise InputsChangedError(diffs)


# ----------------------------------------------------------------------------- frames
def sample_atom_indices(n_atoms: int, k: int = 128) -> np.ndarray:
    """Deterministic, evenly spread atom indices used for per-frame content checksums."""
    if n_atoms <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, n_atoms - 1, num=min(k, n_atoms)).astype(np.int64))


def frame_fingerprint(positions: np.ndarray, dimensions: Optional[np.ndarray], atom_idx: np.ndarray) -> str:
    """64-bit content checksum of one frame (selected atoms + box), as 16 hex characters."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(positions[atom_idx], dtype="<f4").tobytes())
    if dimensions is not None:
        h.update(np.ascontiguousarray(dimensions, dtype="<f4").tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- config / environment
def canonical_json(obj: Any) -> str:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return str(o)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


def config_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _package_versions() -> Dict[str, str]:
    names = ["MDAnalysis", "numpy", "scipy", "pandas", "GridDataFormats", "matplotlib", "pyqtgraph", "PyQt6"]
    out: Dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except Exception:
            out[name] = "not installed"
    return out


def _git_commit() -> Optional[str]:
    try:
        here = Path(__file__).resolve().parent
        res = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        value = res.stdout.strip()
        return value or None
    except Exception:
        return None


def capture_environment() -> Dict[str, Any]:
    return {
        "sozlab_version": __version__,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
    }


def build_provenance(
    project,
    command_line: Optional[str] = None,
    fingerprint_mode: str = "auto",
    extra_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Provenance record for a run: inputs by content, configuration hash, environment."""
    cfg = project.to_dict()
    cfg_for_hash = json.loads(canonical_json(cfg))
    # Output location and worker count do not change the scientific result.
    cfg_for_hash.get("outputs", {}).pop("output_dir", None)
    cfg_for_hash.get("analysis", {}).pop("workers", None)
    extra = list(extra_paths)
    for dm in getattr(project, "density_maps", []) or []:
        if getattr(dm, "align_reference_path", None):
            extra.append(dm.align_reference_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sozlab_version": __version__,
        "algorithm_versions": dict(ALGORITHM_VERSIONS),
        "inputs": fingerprint_inputs(project.inputs, extra, fingerprint_mode),
        "config_sha256": config_hash(cfg_for_hash),
        "command_line": command_line,
        "working_directory": os.getcwd(),
        "environment": capture_environment(),
    }


def write_json_atomic(path: str | os.PathLike, payload: Any, **dump_kwargs) -> None:
    """Write JSON through a temp file + rename so readers never see a half-written file."""
    from engine.serialization import to_jsonable

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, allow_nan=False, **dump_kwargs)
    os.replace(tmp, p)


def output_manifest(directory: str | os.PathLike, exclude: Iterable[str] = ("manifest.json",)) -> Dict[str, Any]:
    """SHA-256 and size of every file under ``directory`` so an export can verify itself."""
    root = Path(directory)
    skip = set(exclude)
    files: Dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in skip and not p.name.endswith(".tmp"):
            h = hashlib.sha256()
            with open(p, "rb") as handle:
                for block in iter(lambda: handle.read(_CHUNK), b""):
                    h.update(block)
            files[p.relative_to(root).as_posix()] = {"size": p.stat().st_size, "sha256": h.hexdigest()}
    return {"schema_version": SCHEMA_VERSION, "files": files}
