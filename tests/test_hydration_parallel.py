

import multiprocessing
from pathlib import Path
from engine.analysis import SOZAnalysisEngine
from engine.models import (
    AnalysisOptions,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SelectionSpec,
    SolventConfig,
    HbondHydrationConfig,
)

def _atom_line(
    idx: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {idx:5d} {name:<4} {resname:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}"
    )

def _write_simple_pdb(path: Path):
    lines = []
    # Frame 1
    lines.append("MODEL        1")
    lines.append(_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"))
    lines.append(_atom_line(2, "O", "HOH", "W", 2, 3.0, 0.0, 0.0, "O"))
    lines.append(_atom_line(3, "H1", "HOH", "W", 2, 3.1, 0.0, 0.0, "H"))
    lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")

def test_hydration_parallel(tmp_path):
    pdb_path = tmp_path / "parallel.pdb"
    _write_simple_pdb(pdb_path)

    inputs = InputConfig(topology=str(pdb_path))
    solvent = SolventConfig(
        water_resnames=["HOH"],
        probe=ProbeConfig(selection="name O", position="atom")
    )
    selections = {
        "prot": SelectionSpec(label="prot", selection="resname ALA and name N", require_unique=True),
    }

    # Two configs to force parallel execution
    h1 = HbondHydrationConfig(
        name="h1", 
        residue_selection="resname ALA", 
        water_selection="resname HOH",
        conditioning="all"
    )
    h2 = HbondHydrationConfig(
        name="h2", 
        residue_selection="resname ALA", 
        water_selection="resname HOH",
        conditioning="all"
    )

    # force workers=2
    options = AnalysisOptions(stride=1, workers=2, store_min_distances=False)
    output = OutputConfig(output_dir=str(tmp_path / "out"))
    
    project = ProjectConfig(
        inputs=inputs, 
        solvent=solvent, 
        selections=selections, 
        sozs=[], 
        analysis=options, 
        outputs=output,
        hbond_hydration=[h1, h2]
    )
    
    engine = SOZAnalysisEngine(project)
    
    # Run should complete without hanging
    result = engine.run()
    
    assert "h1" in result.hbond_hydration_results
    assert "h2" in result.hbond_hydration_results
    assert len(result.hbond_hydration_results["h1"].table) == 1

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Manual run support
    class MockTmp:
        def __init__(self, p): self.path = Path(p)
        def __truediv__(self, other): return self.path / other
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        p = Path("test_run_tmp")
        p.mkdir(exist_ok=True)
        test_hydration_parallel(p)
        print("Test passed!")
