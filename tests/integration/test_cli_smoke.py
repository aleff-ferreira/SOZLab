"""CLI smoke tests: entry points load and answer --help on every platform."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[2]
CLI = ROOT / "cli" / "sozlab_cli.py"


def _env() -> dict:
    """Inherit the environment (Windows needs SYSTEMROOT etc.) and extend PYTHONPATH portably."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return env


def _run(*args: str) -> subprocess.CompletedProcess:
    if not CLI.exists():
        pytest.skip(f"CLI entrypoint not found at {CLI}")
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, env=_env(), timeout=120)


def test_cli_help():
    result = _run("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    for command in ("run", "validate", "extract", "verify", "selftest"):
        assert command in result.stdout


def test_cli_subcommand_help():
    for command in ("run", "validate", "extract"):
        result = _run(command, "--help")
        assert result.returncode == 0 and "usage:" in result.stdout


def test_cli_reports_errors_without_traceback(tmp_path):
    result = _run("run", "--project", str(tmp_path / "missing.json"))
    assert result.returncode == 1
    assert "ERROR:" in result.stderr and "Traceback" not in result.stderr
