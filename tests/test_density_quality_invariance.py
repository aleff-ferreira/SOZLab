import hashlib
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

os.environ.setdefault("SOZLAB_DISABLE_WEBENGINE", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.viz_3d import Density3DWidget


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["sozlab-test"])
    return app


def _sha1(arr: np.ndarray) -> str:
    data = np.ascontiguousarray(arr, dtype=np.float32)
    return hashlib.sha1(data.tobytes(order="C")).hexdigest()


def test_density_quality_toggle_keeps_scalar_field_invariant(qapp):
    grid = np.linspace(0.0, 1.0, num=64 * 48 * 36, dtype=np.float32).reshape(64, 48, 36)
    origin = np.array([1.25, -0.5, 2.0], dtype=float)

    widget = Density3DWidget()
    js_calls: list[tuple[str, tuple]] = []
    widget._js_invoke = lambda method, *args: js_calls.append((method, args))  # type: ignore[assignment]

    widget._update_data_slot(grid, 0.8, origin, "physical")
    qapp.processEvents()

    baseline_hash = _sha1(widget.grid_data)
    baseline_iso = float(widget.current_iso_level)
    baseline_file = widget._density_file
    baseline_file_version = int(widget._density_file_version)

    above_counts: list[int] = []
    for quality in ("Draft", "Balanced", "High", "Ultra"):
        js_calls.clear()
        widget.quality_combo.setCurrentText(quality)
        qapp.processEvents()

        update_payloads = [args[0] for method, args in js_calls if method == "updateDensity" and args]
        assert update_payloads, f"Expected updateDensity payload for quality={quality}"
        payload = update_payloads[-1]
        iso = float(payload["isolevel"])

        assert iso == pytest.approx(baseline_iso, abs=1e-12)
        assert _sha1(widget.grid_data) == baseline_hash
        assert widget._density_file == baseline_file
        assert int(widget._density_file_version) == baseline_file_version

        above_counts.append(int(np.count_nonzero(widget.grid_data >= iso)))

    assert len(set(above_counts)) == 1
    widget.close()

