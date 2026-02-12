import pytest
import os
import sys
from pathlib import Path

# Add src to path for all tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def test_data_dir():
    """Returns the path to the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def laao_data_dir():
    """Results path to LAAO dataset if available."""
    path = Path("/home/aleff/laao_water")
    if path.exists():
        return path
    return None

@pytest.fixture
def mock_project_config():
    """Returns a basic ProjectConfig for testing."""
    from engine.models import (
        ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, OutputConfig,
        ExtractionConfig
    )
    return ProjectConfig(
        inputs=InputConfig(topology="mock.pdb"),
        solvent=SolventConfig(),
        selections={},
        sozs=[],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="test_results"),
        extraction=ExtractionConfig()
    )
