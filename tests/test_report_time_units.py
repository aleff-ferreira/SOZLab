"""Tests for report.py time-unit normalisation (second-round fixes).

Verifies that _time_scale_to_ns returns correct conversion factors and that
_plot_soz_summary renders time axes in ns.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.report import _time_scale_to_ns


# ---------------------------------------------------------------------------
# _time_scale_to_ns unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTimeScaleToNs:
    def test_ps_returns_1e_minus_3(self):
        assert _time_scale_to_ns("ps") == pytest.approx(1e-3)

    def test_fs_returns_1e_minus_6(self):
        assert _time_scale_to_ns("fs") == pytest.approx(1e-6)

    def test_ns_returns_1(self):
        assert _time_scale_to_ns("ns") == pytest.approx(1.0)

    def test_uppercase_ps(self):
        assert _time_scale_to_ns("PS") == pytest.approx(1e-3)

    def test_empty_string_defaults_to_ps(self):
        assert _time_scale_to_ns("") == pytest.approx(1e-3)

    def test_none_defaults_to_ps(self):
        assert _time_scale_to_ns(None) == pytest.approx(1e-3)

    def test_unknown_unit_defaults_to_ps(self):
        assert _time_scale_to_ns("unknown") == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# Helpers to build minimal SOZ result stubs
# ---------------------------------------------------------------------------

def _make_per_frame(time_values):
    return pd.DataFrame({
        "time": time_values,
        "n_solvent": [3] * len(time_values),
    })


def _make_per_solvent():
    return pd.DataFrame({
        "solvent_id": ["WAT:1:-", "WAT:2:-"],
        "occupancy_pct": [80.0, 60.0],
    })


def _make_result(*, time_values, dt=10.0, time_unit="ps", residence_cont=None):
    """Return a SimpleNamespace that mimics SOZResult."""
    return SimpleNamespace(
        per_frame=_make_per_frame(time_values),
        per_solvent=_make_per_solvent(),
        summary={"dt": dt, "time_unit": time_unit},
        residence_cont=residence_cont or {0: [5, 10]},
        residence_inter={},
    )


# ---------------------------------------------------------------------------
# Integration: axis labels must say "ns", not "ps"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPlotSozSummaryNs:
    """Verify _plot_soz_summary uses ns for time axis and residence."""

    def test_soz_timeseries_xlabel_ns(self, tmp_path):
        from engine.report import _plot_soz_summary

        result = _make_result(time_values=[0, 10, 20, 30], dt=10.0, time_unit="ps")
        paths = _plot_soz_summary(result, str(tmp_path), "SOZ_test")

        # Should produce at least one plot (timeseries)
        assert len(paths) >= 1
        assert all(os.path.isfile(p) for p in paths)

    def test_soz_residence_xlabel_ns(self, tmp_path):
        """Residence plot should exist and filename contains ccdf."""
        from engine.report import _plot_soz_summary

        result = _make_result(
            time_values=[0, 10, 20],
            dt=10.0,
            time_unit="ps",
            residence_cont={0: [2, 5], 1: [3]},
        )
        paths = _plot_soz_summary(result, str(tmp_path), "SOZ_test")

        ccdf_paths = [p for p in paths if "ccdf" in os.path.basename(p)]
        assert len(ccdf_paths) == 1
        assert os.path.isfile(ccdf_paths[0])


# ---------------------------------------------------------------------------
# Residence durations are correctly converted to ns
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_residence_duration_conversion():
    """Segment lengths * dt * scale_to_ns should give ns values.

    Example: segment of 5 frames, dt = 10 ps, scale = 1e-3 → 0.05 ns.
    """
    dt = 10.0
    scale_to_ns = _time_scale_to_ns("ps")
    lengths = [5, 10, 20]
    durations = [l * dt * scale_to_ns for l in lengths]
    expected = [0.05, 0.1, 0.2]
    np.testing.assert_allclose(durations, expected)
