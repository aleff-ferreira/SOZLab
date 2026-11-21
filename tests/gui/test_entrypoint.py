"""The real application entry point (what `sozlab-gui` runs), started in a subprocess."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui
ROOT = Path(__file__).parents[2]


def _env():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    return env


def test_version_flag_does_not_need_a_display():
    r = subprocess.run([sys.executable, "-m", "app.main", "--version"], capture_output=True, text=True, env=_env(), timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("SOZLab 2.")


def test_app_starts_loads_the_demo_and_exits_cleanly(tmp_path):
    env = _env()
    env["APPDATA"] = str(tmp_path)  # keep the demo cache and the saved settings out of the user's profile
    env["XDG_DATA_HOME"] = str(tmp_path)
    env["HOME"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-m", "app.main", "--demo", "--theme", "light", "--reduce-motion", "--no-remember", "--quit-after", "25000"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Traceback" not in r.stderr and "Exceptions caught in Qt event loop" not in r.stderr
