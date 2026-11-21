"""Write a small synthetic hydrated-solute system and a ready-to-run project file (no real simulation data needed).

    python examples/make_demo_data.py
    sozlab-cli run --project examples/demo_project.json --output demo_results --report
    sozlab-gui examples/data/demo.gro examples/data/demo.xtc        # or open examples/demo_project.json from the folder menu

The system is a 14-atom solute in 900 overdamped-Langevin waters (box 30 A, 200 frames of 0.5 ps): it is *not* a force-field simulation,
but its dynamics have a known character, which makes it a convenient first run and the basis of the self-test.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from engine.analysis import write_project_json  # noqa: E402
from engine.models import (AnalysisOptions, InputConfig, OutputConfig, ProbeConfig, ProjectConfig, SelectionSpec,  # noqa: E402
                           SolventConfig, SOZDefinition, SOZNode)
from engine.synthetic import make_demo_system  # noqa: E402


def main() -> None:
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    make_demo_system().write(str(data), "demo", "xtc")
    project = ProjectConfig(
        inputs=InputConfig(topology="examples/data/demo.gro", trajectory="examples/data/demo.xtc"),
        solvent=SolventConfig(water_resnames=["SOL"], probe=ProbeConfig(selection="name OW", position="atom")),
        selections={"solute": SelectionSpec(label="solute", selection="resname LIG")},
        sozs=[SOZDefinition(name="first_shell", description="water oxygens within 3.5 A of the solute",
                            root=SOZNode(type="distance", params={"selection_label": "solute", "cutoff": 3.5, "unit": "A"}))],
        analysis=AnalysisOptions(),
        outputs=OutputConfig(output_dir="demo_results"),
    )
    write_project_json(project, str(HERE / "demo_project.json"))
    print("wrote", data / "demo.gro", data / "demo.xtc", HERE / "demo_project.json")


if __name__ == "__main__":
    main()
