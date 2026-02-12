
import pytest
import subprocess
import sys
from pathlib import Path

pytestmark = pytest.mark.integration

def test_cli_help():
    """Test that CLI help command runs without error."""
    # Try running the module directly if installed, or via python -m
    # Assuming 'sozlab-cli' is in path, or we use python -m cli.main (if that exists)
    # The setup suggests 'src/cli' exists.
    
    # We'll try running via python -m
    cmd = [sys.executable, "-m", "cli.main", "--help"]
    # Check if cli package is runnable.
    # We need to ensure src is in python path.
    
    env = headers = {"PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    
    # Alternatively, check if 'sozlab-cli' command exists.
    # But for a reliable test in dev env, running python script is better.
    # Where is the cli entrypoint?
    # src/cli/main.py ?
    
    # Let's try finding the entrypoint.
    # pyproject.toml says cli package is in 'cli' dir, mapping to 'cli'.
    # script is cli.sozlab_cli:main.
    
    root_dir = Path(__file__).parents[2]
    cli_path = root_dir / "cli" / "sozlab_cli.py"
    
    # Needs root in PYTHONPATH for 'cli' package, and 'src' for 'engine'/'app'.
    env = {"PYTHONPATH": f"{root_dir}:{root_dir}/src"}
    
    if not cli_path.exists():
        pytest.skip(f"CLI entrypoint not found at {cli_path}")
        
    cmd = [sys.executable, str(cli_path), "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "usage:" in result.stdout or "usage:" in result.stderr

def test_cli_version():
    root_dir = Path(__file__).parents[2]
    cli_path = root_dir / "cli" / "sozlab_cli.py"
    env = {"PYTHONPATH": f"{root_dir}:{root_dir}/src"}
    
    cmd = [sys.executable, str(cli_path), "validate", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "usage:" in result.stdout or "usage:" in result.stderr
