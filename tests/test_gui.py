
import os

import pytest

# Force offscreen platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Avoid QtWebEngine startup in headless test runs.
os.environ.setdefault("SOZLAB_DISABLE_WEBENGINE", "1")

# Skip if PyQt6 not importable (although we checked installed list)
pytest.importorskip("PyQt6")
pytestmark = pytest.mark.gui

def test_imports():
    """Smoke test that GUI modules can be imported."""
    try:
        from app.main import MainWindow
    except ImportError as e:
        pytest.fail(f"Failed to import GUI modules: {e}")
