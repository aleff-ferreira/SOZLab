"""Analysis engine for SOZLab."""

from __future__ import annotations

from engine._version import __version__  # noqa: F401

import warnings


# Silence noisy stdlib deprecation triggered by MDAnalysis importing xdrlib.
warnings.filterwarnings(
    "ignore",
    message=r".*xdrlib.*deprecated.*",
    category=DeprecationWarning,
)
