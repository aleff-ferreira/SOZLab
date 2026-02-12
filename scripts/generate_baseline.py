
import logging
import sys
import json
import numpy as np
from pathlib import Path
from engine.analysis import SOZAnalysisEngine
from engine.models import (
    ProjectConfig, InputConfig, SolventConfig, AnalysisOptions, 
    OutputConfig, SOZDefinition, SOZNode, HbondHydrationConfig,
    SelectionSpec
)
from engine.serialization import to_jsonable

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("baseline")

    # Dataset paths
    data_dir = Path("/home/aleff/laao_water")
    topo = data_dir / "step3_input.gro"
    traj = data_dir / "step7_production_mol_rect_3_system_10ns.xtc"

    if not topo.exists():
        logger.error(f"Topology not found: {topo}")
        return

    # Configuration
    inputs = InputConfig(topology=str(topo), trajectory=str(traj))
    solvent = SolventConfig(
        solvent_label="Water",
        water_resnames=["TIP3", "HOH", "SOL"],
        include_ions=True
    )
    
    # Selections
    selections = {
        "protein": SelectionSpec(label="protein", selection="protein"),
        "active_site": SelectionSpec(label="active_site", selection="resname FAD"),
    }

    # SOZ Definition (around active site)
    soz_root = SOZNode(
        type="shell", 
        params={
            "selection_label": "active_site",
            "distance": 5.0, # 5 Angstroms
            "unit": "A"
        }
    )
    sozs = [
         SOZDefinition(name="ActiveSiteShell", description="5A shell around FAD", root=soz_root)
    ]
    
    # Hydration Analysis
    hydration = HbondHydrationConfig(
        name="FAD_hydration",
        residue_selection="resname FAD",
        distance=3.5,
        angle=150.0,
        conditioning="soz",
        soz_name="ActiveSiteShell"
    )

    analysis_opt = AnalysisOptions(
        frame_start=0, 
        frame_stop=10, # Analyze first 10 frames for baseline to be fast
        stride=1,
        workers=2
    )

    project = ProjectConfig(
        inputs=inputs,
        solvent=solvent,
        selections=selections,
        sozs=sozs,
        hbond_hydration=[hydration],
        analysis=analysis_opt,
        outputs=OutputConfig(output_dir="baseline_results") 
    )

    engine = SOZAnalysisEngine(project)
    result = engine.run(logger=logger)
    
    # Extract key metrics for regression
    metrics = {
        "soz_occupancy": result.soz_results["ActiveSiteShell"].summary,
        "hydration_count": len(result.hbond_hydration_results["FAD_hydration"].table),
        "top_hydration_residue_freq": 0.0
    }
    
    if not result.hbond_hydration_results["FAD_hydration"].table.empty:
        df = result.hbond_hydration_results["FAD_hydration"].table
        metrics["top_hydration_residue_freq"] = float(df.iloc[0]["freq_given_soz"])

    print(json.dumps(to_jsonable(metrics), indent=2))

if __name__ == "__main__":
    main()
