"""Serialization helpers for JSON output."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency in practice
    pd = None


def to_jsonable(value):
    """Recursively convert objects into strictly valid JSON types.

    NaN and +/-infinity become ``None`` (``null``): the JSON standard has no representation
    for them, and Python's default ``NaN`` token breaks other parsers. Numpy scalars and
    arrays, pandas timestamps and tuples/sets are converted to plain Python values.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if pd is not None:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()
        if value is pd.NA or value is pd.NaT:
            return None

    if isinstance(value, Mapping):
        return {_json_key(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    return value


def _json_key(key):
    key = to_jsonable(key)
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return str(key)
