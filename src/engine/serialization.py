"""Serialization helpers for JSON output."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency in practice
    pd = None


def to_jsonable(value):
    """Recursively convert objects into JSON-serializable types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd is not None:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()

    if isinstance(value, Mapping):
        return {to_jsonable(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    return value
